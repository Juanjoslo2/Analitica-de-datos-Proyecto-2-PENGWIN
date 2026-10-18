"""Dashboard (05_app): volumen armado con cortes sueltos (cortes.py y servicio.procesar_cortes).

Con un fantasma sintético se prueba cada forma de decidir el orden (DICOM, metadatos del PNG, número
del nombre, parecido entre cortes), los repetidos, los huecos y los lotes o .zip malformados. Con
datos de PENGWIN (si están) se prueba el orden por contenido en un CT real, el sentido cabeza-pies
con el modelo y la subida completa por HTTP.
"""

import io
import json
import random
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from PIL import Image

APP = Path(__file__).resolve().parents[1] / "05_app"
sys.path.insert(0, str(APP))

import cortes  # noqa: E402
import exportar_cortes  # noqa: E402
from validacion import ArchivoInvalido, validar_volumen  # noqa: E402

CKPT = APP.parent / "checkpoints" / "v2_last" / "best.pth"


def _fantasma(nz=40, ny=96, nx=112, seed=0):
    """Volumen (Z, Y, X) en HU: aire, un cuerpo y tres «huesos» que se mueven en direcciones distintas
    a medida que sube z, así cada corte se parece más a sus vecinos que a cualquier otro."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:ny, :nx]
    vol = np.full((nz, ny, nx), -1000.0, np.float32)
    cuerpo = ((yy - ny / 2) / (0.42 * ny)) ** 2 + ((xx - nx / 2) / (0.45 * nx)) ** 2 <= 1
    tejido = -150.0 + 400.0 * xx / nx                     # grasa a músculo: un CT tiene cientos de tonos
    for z in range(nz):
        t = z / (nz - 1)
        s = np.where(cuerpo, tejido, -1000.0)
        for (y0, x0, vy, vx, r) in ((30, 30, 30, 10, 9), (60, 80, -25, -15, 7), (48, 56, 5, 30, 6 + 4 * t)):
            d = (yy - (y0 + vy * t)) ** 2 + (xx - (x0 + vx * t)) ** 2
            s = np.where(d <= r ** 2, 1300.0 - 600.0 * np.sqrt(d) / r, s)       # cortical densa, medular más blanda
        vol[z] = s + rng.normal(0, 15, (ny, nx))
    return vol


def _mha(tmp_path, vol, spacing=(0.7, 0.7, 1.5)):
    img = sitk.GetImageFromArray(vol.astype(np.int16))
    img.SetSpacing(spacing)
    p = tmp_path / "fantasma.mha"
    sitk.WriteImage(img, str(p))
    return p


def _lote(tmp_path, rutas_o_pares, barajar=True, seed=1):
    pares = [(r.name, r.read_bytes()) if isinstance(r, Path) else r for r in rutas_o_pares]
    if barajar:
        random.Random(seed).shuffle(pares)                 # el navegador puede mandarlos en cualquier orden
    p = tmp_path / "cortes.lote"
    cortes.escribir_lote(pares, p)
    return p


def _png(arr8, **texto):
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    for k, v in texto.items():
        info.add_text(k, str(v))
    buf = io.BytesIO()
    Image.fromarray(arr8).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def _gris(vol):
    from pengwin.data.data_loader import apply_bone_window
    return (np.clip(apply_bone_window(vol), 0, 1) * 255).round().astype(np.uint8)


def _hu_8bits(g):
    return np.where(g == 0, -1024.0, g.astype(np.float32) * 1800 / 255 - 500)


def _igual(a, b):
    """Mismo volumen salvo redondeo de float32 (v·1800/255 y v·(1800/255) no dan el mismo último bit)."""
    return a.shape == b.shape and np.allclose(a, b, atol=0.01)


def _hu_dicom(vol):
    return np.clip(vol.astype(np.int16), -1024, 3071).astype(np.float32)     # exportar_cortes recorta como el .mha


# ---------------------------------------------------------------- orden

def test_orden_por_numero_del_nombre_sin_ceros(tmp_path):
    vol = _fantasma()
    g = _gris(vol)
    # img_2 va antes que img_10 (orden numérico, no alfabético) y el número 075 del caso no cuenta
    pares = [(f"075_img_{k + 1}.png", _png(g[k])) for k in range(len(g))]
    rec = cortes.reconstruir(_lote(tmp_path, pares), tmp_path / "t")
    assert "número del nombre" in rec.informe["orden"] and "confirmado" in rec.informe["orden"]
    assert _igual(rec.hu, _hu_8bits(g))
    assert not rec.sentido_conocido and rec.informe["fuente_mm_px"] == "supuesto"


def test_orden_por_contenido_con_nombres_al_azar(tmp_path):
    vol = _fantasma(nz=50)
    g = _gris(vol)
    rng = random.Random(3)
    pares = [("".join(rng.choice("abcdefghij") for _ in range(8)) + ".png", _png(g[k])) for k in range(len(g))]
    rec = cortes.reconstruir(_lote(tmp_path, pares), tmp_path / "t")
    assert "parecido" in rec.informe["orden"]
    ref = _hu_8bits(g)
    assert _igual(rec.hu, ref) or _igual(rec.hu, ref[::-1])    # el sentido lo pone el modelo


def test_nombres_que_mienten_se_corrigen_con_el_contenido(tmp_path):
    vol = _fantasma(nz=40)
    g = _gris(vol)
    numeros = list(range(len(g)))
    random.Random(5).shuffle(numeros)                      # números que no tienen nada que ver con la altura
    pares = [(f"corte_{numeros[k]:03d}.png", _png(g[k])) for k in range(len(g))]
    rec = cortes.reconstruir(_lote(tmp_path, pares), tmp_path / "t")
    assert "parecido" in rec.informe["orden"]
    assert any("no sigue la anatomía" in a for a in rec.informe["avisos"])
    ref = _hu_8bits(g)
    assert _igual(rec.hu, ref) or _igual(rec.hu, ref[::-1])


def test_png_con_metadatos_da_orden_y_spacing(tmp_path):
    rutas = exportar_cortes.exportar(_mha(tmp_path, _fantasma()), tmp_path / "png", "png", desordenar=True)
    rec = cortes.reconstruir(_lote(tmp_path, rutas), tmp_path / "t")
    assert rec.informe["orden"] == "SliceLocation guardado en cada PNG"
    assert rec.spacing_xyz == pytest.approx((0.7, 0.7, 1.5), abs=1e-4)
    assert rec.informe["fuente_mm_px"] == "metadatos del PNG"


def test_dicom_desordenado_recupera_hu_geometria_y_sentido(tmp_path):
    vol = _fantasma()
    rutas = exportar_cortes.exportar(_mha(tmp_path, vol), tmp_path / "dcm", "dicom", desordenar=True, al_reves=True)
    rec = cortes.reconstruir(_lote(tmp_path, rutas), tmp_path / "t")
    assert rec.sentido_conocido and rec.informe["formato"] == "DICOM"
    assert rec.spacing_xyz == pytest.approx((0.7, 0.7, 1.5), abs=1e-4)
    assert _igual(rec.hu, _hu_dicom(vol))


def test_dicom_de_otra_serie_y_dicomdir_quedan_fuera(tmp_path):
    vol = _fantasma(nz=20)
    a = exportar_cortes.exportar(_mha(tmp_path, vol), tmp_path / "a", "dicom")
    b = exportar_cortes.exportar(_mha(tmp_path, vol[:9]), tmp_path / "b", "dicom")      # otra serie, más chica
    pares = [(f"a/{r.name}", r.read_bytes()) for r in a] + [(f"b/{r.name}", r.read_bytes()) for r in b]
    pares.append(("LEAME.txt", b"texto"))
    rec = cortes.reconstruir(_lote(tmp_path, pares), tmp_path / "t")
    assert rec.hu.shape[0] == 20
    assert any("otras 1 series" in i for i in rec.informe["ignorados"])
    assert any("LEAME.txt" in i for i in rec.informe["ignorados"])


def test_huecos_se_interpolan_y_repetidos_se_quitan(tmp_path):
    vol = _fantasma(nz=30)
    g = _gris(vol)
    pares = [(f"c{k:03d}.png", _png(g[k])) for k in range(30) if k not in (11, 12, 20)]
    pares.append(("c005 (copia).png", _png(g[5])))
    rec = cortes.reconstruir(_lote(tmp_path, pares), tmp_path / "t")
    assert rec.hu.shape[0] == 30                           # 0..29 con paso parejo
    assert rec.informe["interpolados"] == 3 and rec.informe["repetidos"] == 1
    assert rec.nombres[11] is None and rec.nombres[13] == "c013.png"
    ref = _hu_8bits(g)
    assert _igual(rec.hu[13], ref[13])
    assert np.allclose(rec.hu[11], ref[10] + (ref[13] - ref[10]) / 3, atol=1e-3)


def test_numeros_del_nombre_elige_el_grupo_que_cambia():
    nombres = [f"IM-0007-{k:04d}.dcm" for k in (3, 1, 2)]
    c = cortes.numeros_del_nombre(nombres)
    assert len(c) == 1 and c[0]["valores"] == [3.0, 1.0, 2.0] and not c[0]["decimal"]
    c = cortes.numeros_del_nombre(["z_-12.5mm.png", "z_-10.0mm.png"])
    assert c[0]["decimal"]
    assert cortes.numeros_del_nombre(["a.png", "b.png"]) == []


# ---------------------------------------------------------------- rechazos

def test_mascaras_se_rechazan(tmp_path):
    lab = np.zeros((12, 64, 64), np.uint8)
    lab[:, 20:40, 20:40] = 1
    lab[:, 45:60, 10:30] = 11
    pares = [(f"m{k}.png", _png(lab[k])) for k in range(12)]
    with pytest.raises(cortes.LoteInvalido, match="máscaras"):
        cortes.reconstruir(_lote(tmp_path, pares), tmp_path / "t")


def test_pocos_cortes_o_tamanos_mezclados(tmp_path):
    g = _gris(_fantasma(nz=10))
    with pytest.raises(cortes.LoteInvalido, match="al menos 8"):
        cortes.reconstruir(_lote(tmp_path, [(f"c{k}.png", _png(g[k])) for k in range(5)]), tmp_path / "t")
    pares = [(f"c{k}.png", _png(g[k])) for k in range(6)] + [(f"d{k}.png", _png(g[k][:80])) for k in range(6)]
    with pytest.raises(cortes.LoteInvalido, match="tamaños distintos"):
        cortes.reconstruir(_lote(tmp_path, pares), tmp_path / "t")


def test_lote_malformado(tmp_path):
    p = tmp_path / "x.lote"
    p.write_bytes(b"PWLOTE1\n" + struct.pack("<I", 10) + b"no es json")
    with pytest.raises(cortes.LoteInvalido):
        cortes.validar_lote(p)
    idx = json.dumps({"archivos": [{"nombre": "a.png", "tam": 999}]}).encode()
    p.write_bytes(b"PWLOTE1\n" + struct.pack("<I", len(idx)) + idx + b"corto")          # dice 999 y trae 5
    with pytest.raises(cortes.LoteInvalido, match="incompleto"):
        cortes.validar_lote(p)
    p.write_bytes(b"PWLOTE1\n" + struct.pack("<I", 2 ** 31))                            # índice gigante
    with pytest.raises(cortes.LoteInvalido):
        cortes.validar_lote(p)
    idx = json.dumps({"archivos": [{"nombre": "a", "tam": 0}] * (cortes.MAX_CORTES + 1)}).encode()
    p.write_bytes(b"PWLOTE1\n" + struct.pack("<I", len(idx)) + idx)
    with pytest.raises(cortes.LoteInvalido, match="entre 1 y"):
        cortes.validar_lote(p)
    p.write_bytes(b"PK\x03\x04 no es un zip")
    with pytest.raises(cortes.LoteInvalido, match="dañado"):
        cortes.validar_lote(p)


def test_zip_de_cortes_con_basura(tmp_path):
    g = _gris(_fantasma(nz=12))
    z = tmp_path / "cortes.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for k in range(12):
            zf.writestr(f"carpeta/corte_{k:02d}.png", _png(g[k]))
        zf.writestr("__MACOSX/carpeta/._corte_00.png", b"basura de mac")
        zf.writestr("carpeta/.DS_Store", b"basura")
        zf.writestr("carpeta/LEAME.txt", b"hola")
        zf.writestr("../../etc/passwd", b"el nombre nunca se usa como ruta")
    assert cortes.validar_lote(z)["formato"] == "zip"
    rec = cortes.reconstruir(z, tmp_path / "t")
    assert rec.hu.shape[0] == 12 and len(rec.informe["ignorados"]) == 2
    assert not (tmp_path / "etc").exists()


def test_zip_bomba_se_rechaza(tmp_path):
    z = tmp_path / "bomba.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.png", b"\x00" * (cortes.MAX_BYTES_CORTE + 1))                    # pesa ~65 KB comprimido
    assert z.stat().st_size < 200_000
    with pytest.raises(cortes.LoteInvalido, match="grande"):
        cortes.validar_lote(z)


def test_zip_con_un_volumen(tmp_path):
    mha = _mha(tmp_path, _fantasma(nz=12))
    z = tmp_path / "vol.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(mha, "caso/fantasma.mha")
    assert cortes.validar_lote(z)["formato"] == "zip con un volumen"
    (tmp_path / "trab").mkdir()
    out = cortes.extraer_volumen(z, tmp_path / "trab")
    assert out.name == "volumen.mha" and validar_volumen(out, out.name)["dims_xyz"] == [112, 96, 12]
    # un .mha que apunta a otro archivo del disco se rechaza también dentro del .zip
    b = mha.read_bytes()
    malo = b[:b.index(b"ElementDataFile")] + b"ElementDataFile = /etc/passwd\n"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("x.mha", malo)
    with pytest.raises(cortes.LoteInvalido, match="LOCAL"):
        cortes.extraer_volumen(z, tmp_path / "trab")
    with pytest.raises(ArchivoInvalido):
        validar_volumen(z, z.name)                         # un .zip nunca pasa como volumen


# ---------------------------------------------------------------- corte suelto

def test_corte_suelto_png_dicom_y_mm_px(tmp_path):
    vol = _fantasma(nz=12)
    mha = _mha(tmp_path, vol, spacing=(0.65, 0.65, 2.0))
    d = exportar_cortes.exportar(mha, tmp_path / "d", "dicom")[5].read_bytes()
    canales, mm, nota, _ = cortes.corte_suelto(d, tmp_path / "t")
    assert canales.shape == (1, 96, 112) and mm == pytest.approx(0.65) and "del archivo" in nota
    assert _igual(canales[0], _hu_dicom(vol)[5])
    p = exportar_cortes.exportar(mha, tmp_path / "p", "png")[5].read_bytes()
    assert cortes.corte_suelto(p, tmp_path / "t")[1] == pytest.approx(0.65)                  # sCAL / PixelSpacing
    sin = _png(_gris(vol)[5])
    _, mm, nota, avisos = cortes.corte_suelto(sin, tmp_path / "t")
    assert mm == cortes.MM_PX_DEFECTO_256 and "supuesto" in nota and avisos                  # 112 px: recorte del modelo
    assert cortes.corte_suelto(sin, tmp_path / "t", mm_px=0.9)[1] == 0.9
    rgb = np.stack([_gris(vol)[k] for k in (4, 5, 6)], -1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    assert cortes.corte_suelto(buf.getvalue(), tmp_path / "t")[0].shape[0] == 3              # entrada 2.5D
    with pytest.raises(cortes.LoteInvalido):
        cortes.corte_suelto(b"MZ\x90\x00" + b"\x00" * 200, tmp_path / "t")


def test_png_16_bits_hu_mas_1024(tmp_path):
    vol = _fantasma(nz=10)
    rutas = exportar_cortes.exportar(_mha(tmp_path, vol), tmp_path / "p16", "png16")
    rec = cortes.reconstruir(_lote(tmp_path, rutas), tmp_path / "t")
    assert "valor − 1024" in rec.informe["conversion"]
    assert np.array_equal(rec.hu, np.clip(vol.astype(np.int16), -1024, None).astype(np.float32))


# ---------------------------------------------------------------- con datos de PENGWIN

def _caso(data_dir, cid):
    return next(data_dir.glob(f"PENGWIN_CT_train_images_part*/{cid}.mha"))


def test_ct_real_ordenado_solo_por_contenido(data_dir, tmp_path):
    from pengwin.data.data_loader import load_and_standardize_mha
    hu, _, _ = load_and_standardize_mha(_caso(data_dir, "075"))
    rutas = exportar_cortes.exportar(_caso(data_dir, "075"), tmp_path / "c", "png", sin_metadatos=True, desordenar=True)
    rec = cortes.reconstruir(_lote(tmp_path, rutas), tmp_path / "t")
    assert "parecido" in rec.informe["orden"] and rec.hu.shape[0] == hu.shape[0]
    ref = _hu_8bits(_gris(hu))
    assert _igual(rec.hu, ref) or _igual(rec.hu, ref[::-1])


@pytest.fixture(scope="module")
def modelo():
    if not CKPT.exists():
        pytest.skip("sin checkpoint v2_last")
    import servicio
    return servicio.Modelo(CKPT, APP.parent / "configs" / "tuned.yaml")


def test_sentido_cabeza_pies_con_el_modelo(data_dir, modelo):
    import servicio
    from pengwin.data.data_loader import load_and_standardize_mha
    hu, sp, _ = load_and_standardize_mha(_caso(data_dir, "075"))
    bien = servicio.sentido_anatomico(modelo, hu, sp)
    alreves = servicio.sentido_anatomico(modelo, hu[::-1], sp)
    assert bien["seguro"] and not bien["invertir"] and bien["dz_mm"] > 15 and bien["dy_mm"] > 15
    assert alreves["seguro"] and alreves["invertir"]


def test_subida_de_cortes_por_http(data_dir, tmp_path, monkeypatch):
    """Lo que hace la página: lote → /api/subida → PUT de trozos → /fin → cola → resultado."""
    if not CKPT.exists():
        pytest.skip("sin checkpoint v2_last")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PENGWIN_USUARIOS", str(tmp_path / "u.db"))
    monkeypatch.setenv("PENGWIN_TRABAJOS", str(tmp_path / "trabajos"))
    monkeypatch.setenv("PENGWIN_EJEMPLOS", str(tmp_path / "ejemplos"))
    monkeypatch.setenv("PENGWIN_COOKIE_INSEGURA", "1")
    sys.modules.pop("app", None)
    import app as modulo
    modulo.usuarios.crear("profesor", "clave-larga-123")
    c = TestClient(modulo.app)
    assert c.post("/api/login", json={"usuario": "profesor", "clave": "clave-larga-123"}).status_code == 200
    rutas = exportar_cortes.exportar(_caso(data_dir, "097"), tmp_path / "c", "png", sin_metadatos=True, al_reves=True)
    lote = _lote(tmp_path, rutas).read_bytes()
    r = c.post("/api/subida", data={"nombre": "cortes.lote", "tamano": len(lote), "mm_px": "0,879", "mm_corte": "1"})
    assert r.status_code == 200, r.text
    tid, trozo = r.json()["id"], r.json()["trozo"]
    for i in range(0, -(-len(lote) // trozo)):
        parte = lote[i * trozo:(i + 1) * trozo]
        assert c.put(f"/api/subida/{tid}?i={i}&n={len(parte)}", content=parte).status_code == 200
    assert c.post(f"/api/subida/{tid}/fin").status_code == 200
    for _ in range(600):
        s = c.get(f"/api/trabajo/{tid}").json()
        if s["estado"] in ("listo", "error"):
            break
        time.sleep(1)
    assert s["estado"] == "listo", s
    res = c.get(f"/api/trabajo/{tid}/resumen").json()
    rc = res["reconstruccion"]
    assert rc["cortes"] == 300 and "número del nombre" in rc["orden"] and "se dieron vuelta" in rc["sentido"]
    assert rc["mm_px"] == [0.879, 0.879] and rc["fuente_mm_px"] == "escrito por el usuario"
    assert rc["corte_inferior"] == "corte_0300.png" and len(res["fragmentos"]) >= 3
    v = c.get(f"/api/trabajo/{tid}/volumen")
    assert v.status_code == 200 and v.content.startswith(b"ObjectType = Image")
    assert c.get(f"/api/trabajo/{tid}/corte/150").status_code == 200
    # opciones fuera de rango se rechazan antes de subir nada
    assert c.post("/api/subida", data={"nombre": "cortes.lote", "tamano": 10, "mm_px": "50"}).status_code == 400
