"""servicio.py.

Pipeline del dashboard: de un volumen CT (.mha / .nii / .nrrd) a los tres visualizadores.

    volumen → caché 256 px (mismo preprocesado que el entrenamiento, sin etiqueta)
            → modelo corte a corte con TTA (configs/tuned.yaml)
            → fragmentos 3D (edt_hmax) → grilla nativa → distancia de separación en mm
            → visualizador 1 (MIP en HU, sin modelo), 2 (corte a corte) y 3 (malla 3D)

Todo lo que el navegador pide sale de la carpeta del trabajo (``<trabajos>/<id>/``), así
recorrer el slider no vuelve a correr el modelo.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.data_loader import load_and_standardize_mha  # noqa: E402
from pengwin.data.slice_cache import build_case_cache, load_case_cache  # noqa: E402
from pengwin.detection.grid import decode  # noqa: E402
from pengwin.inference.volume import TTA_SETS, _tta_forward, predict_case, to_native  # noqa: E402
from pengwin.evaluation.fragment_metrics import distance_comparison, match_fragments, summarize  # noqa: E402
from pengwin.measurement.separation import separation_table  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402
from pengwin.postprocess.instances import instance_kwargs, resolve_postprocess, separate_instances  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402
from pengwin.visualization.mip_viewer import DISPLAY_MAX_HU, compute_bone_mip, rotating_mip  # noqa: E402

REGIONES = {1: "SA", 2: "LI", 3: "RI"}
NOMBRE = {"SA": "Sacro", "LI": "Coxal izq.", "RI": "Coxal der."}
# Paleta del proyecto (overlay.py): un color por macro-hueso
COLOR = {"SA": (42, 120, 214), "LI": (235, 104, 52), "RI": (27, 175, 122)}
AVISO = "Uso académico. No es un dispositivo médico ni apoya decisiones quirúrgicas reales."
ESCALA = 2                                  # el corte de 256 px se dibuja a 512 px


class NoEsCT(ValueError):
    """El archivo se leyó bien pero no es un CT en unidades Hounsfield; el mensaje va tal cual al usuario."""


class EsMascara(NoEsCT):
    """Subieron la máscara de los expertos (PENGWIN_CT_train_labels) en vez del CT."""


def revisar_ct(hu: np.ndarray) -> None:
    """Frena lo que no es un CT antes de gastar GPU. Una máscara de etiquetas (la carpeta
    PENGWIN_CT_train_labels) o un volumen sin aire ni hueso en HU daban MIP negras y 0 fragmentos
    sin ningún aviso."""
    lo, hi = float(hu.min()), float(hu.max())
    distintos = np.unique(hu[::4, ::4, ::4]).size
    if lo >= 0 and hi <= 255 and distintos <= 64:
        raise EsMascara("Este archivo es una máscara, no un CT: la carpeta PENGWIN_CT_train_labels trae lo que marcaron "
                        f"los expertos (0 = fondo y un número por fragmento; aquí {distintos} valores, de 0 a {hi:.0f}), "
                        "no la tomografía. El CT de cada caso tiene el mismo nombre y está en "
                        "PENGWIN_CT_train_images_part1 (casos 001-050) o _part2 (051-100).")
    if lo > -500 or hi < 250:
        raise NoEsCT(f"El volumen no parece un CT en unidades Hounsfield: va de {lo:.0f} a {hi:.0f}, y un CT tiene aire "
                     "cerca de −1000 HU y hueso por encima de 250 HU.")


def _fuente(tam: int):
    import matplotlib
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return ImageFont.truetype(str(ttf), tam) if ttf.exists() else ImageFont.load_default()


def tono(region: str, k: int) -> tuple:
    """Color del fragmento k de una región: el principal con el color pleno, los demás más claros."""
    r, g, b = COLOR[region]
    if k <= 1:
        return (r, g, b)
    f = min(0.25 + 0.15 * (k - 2), 0.7)
    return tuple(int(c + (255 - c) * f) for c in (r, g, b))


class Modelo:
    """Checkpoint + config final cargados una vez (GPU si hay, si no CPU)."""

    def __init__(self, ckpt: Path, config: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        self.net = PengwinNet(self.cfg["model"]).to(self.device).eval()
        self.net.load_state_dict(ck["model"])
        tuned = load_config(config)
        self.pp = resolve_postprocess(self.cfg.get("postprocess"), tuned.get("postprocess"))
        tta = (tuned.get("inference") or {}).get("tta")
        self.tta = TTA_SETS[str(tta)] if tta else None
        self.nombre = f"{ckpt.parent.name} · {config.name}"


# ---------------------------------------------------------------- máscara subida en vez del CT

def _particion() -> Dict[str, str]:
    ruta = REPO / "01_data" / "splits.json"
    if not ruta.exists():
        return {}
    return {c: p for p, casos in json.loads(ruta.read_text(encoding="utf-8"))["splits"].items() for c in casos}


PARTICION = _particion()
NOMBRE_PARTICION = {"train": "entrenamiento", "val": "validación", "test": "test"}
NOTA_PARTICION = {
    "test": "Caso de test: el modelo nunca lo vio, así que esta comparación es una medición honesta.",
    "val": "Caso de validación: no se entrenó con él, pero sirvió para elegir el modelo y el posproceso.",
    "train": "Caso de entrenamiento: el modelo aprendió con esta misma máscara, así que la comparación sale "
             "optimista. Para medir de verdad use un caso de test.",
}


def firma(ruta: Path | str) -> tuple:
    """Geometría del header (tamaño, spacing, origen, dirección). En PENGWIN cada caso tiene la suya y
    la máscara comparte la de su CT (comprobado en los 100): así se reconoce el caso sin fiarse del nombre."""
    f = sitk.ImageFileReader()
    f.SetFileName(str(ruta))
    f.ReadImageInformation()
    return (tuple(f.GetSize()), tuple(round(s, 3) for s in f.GetSpacing()),
            tuple(round(o, 1) for o in f.GetOrigin()), tuple(round(d, 3) for d in f.GetDirection()))


def indice_cts(carpeta: Path) -> Dict[tuple, Path]:
    """Firma -> CT, para los CT de PENGWIN guardados en el servidor (PENGWIN_CT_train_images_part*)."""
    return {firma(r): r for r in sorted(carpeta.glob("PENGWIN_CT_train_images_part*/*.mha"))}


def analizar_mascara(modelo: Modelo, ruta_mascara: Path, carpeta: Path, avance: Callable[[str, float], None],
                     indice: Dict[tuple, Path]) -> bool:
    """Si subieron la máscara de un caso cuyo CT está en el servidor, analiza ese CT y usa la máscara
    como referencia (lo mismo que los casos de ejemplo). Devuelve False si el caso no se reconoce."""
    ct = indice.get(firma(ruta_mascara))
    if ct is None:
        return False
    caso, parte = ct.stem, PARTICION.get(ct.stem)
    avance(f"Es la máscara del caso {caso}: se analiza su CT, guardado en el servidor, y la máscara queda "
           "como referencia", 0.05)
    procesar_volumen(modelo, ct, carpeta, avance, ruta_gt=ruta_mascara,
                     titulo=f"Caso {caso} · {NOMBRE_PARTICION.get(parte, 'PENGWIN')}",
                     nota_referencia=NOTA_PARTICION.get(parte))
    return True


# ---------------------------------------------------------------- volumen completo

def procesar_volumen(modelo: Modelo, ruta: Path, carpeta: Path, avance: Callable[[str, float], None],
                     ruta_gt: Path | None = None, titulo: str | None = None,
                     nota_referencia: str | None = None) -> Dict:
    """Corre todo el pipeline y deja en ``carpeta`` lo que piden los visualizadores."""
    t0 = time.time()
    carpeta.mkdir(parents=True, exist_ok=True)
    avance("Leyendo el volumen y orientándolo a LPS", 0.05)
    hu, spacing, orient = load_and_standardize_mha(ruta)
    if hu.ndim != 3 or min(hu.shape) < 8:
        raise ValueError(f"se esperaba un volumen 3D; llegó {hu.shape}")
    revisar_ct(hu)
    _mip(hu, spacing, carpeta, avance)
    del hu

    avance("Preprocesado: recorte óseo, ventana L400/W1800 y 256 px", 0.25)
    build_case_cache("caso", ruta, None, carpeta / "cache", window=tuple(_ventana(modelo.cfg)),
                     context_mm=float(modelo.cfg["data"].get("context_mm", 2.0)),
                     crop_margin_mm=float(modelo.cfg["data"].get("crop_margin_mm", 15)))
    avance("Modelo corte a corte (detección, región, borde) con TTA de 5 pasadas", 0.40)
    t_mod = time.time()
    pred = predict_case(modelo.net, carpeta / "cache", "caso", modelo.cfg, modelo.device, tta=modelo.tta)
    t_mod = time.time() - t_mod
    meta = pred["meta"]
    grid_sp = (meta["spacing_zyx"][0], meta["pixel_mm"], meta["pixel_mm"])

    avance("Fragmentos 3D: semillas por distancia + h-máximas y watershed", 0.65)
    lab = separate_instances(pred["semantic"], pred["edge"].astype(np.float32), grid_sp, **instance_kwargs(modelo.pp))
    avance("Distancia de separación en mm sobre la grilla nativa (EDT con el spacing del header)", 0.75)
    lab_nat = to_native(lab, meta, order=0)
    filas = separation_table(lab_nat, meta["spacing_zyx"])
    gt_filas = metricas_gt = None
    if ruta_gt is not None:              # solo los casos de ejemplo: comparación contra la referencia
        gt, sp_gt, _ = load_and_standardize_mha(ruta_gt, is_label=True)
        emp = match_fragments(gt, lab_nat, sp_gt)
        gt_filas = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
                    for r in distance_comparison(gt, lab_nat, sp_gt, emp)]
        metricas_gt = {k: (None if np.isnan(v) else round(v, 3)) for k, v in summarize(emp, gt_filas).items()}
        del gt
    del lab_nat

    avance("Malla 3D: marching cubes por fragmento", 0.85)
    fragmentos = _tabla_fragmentos(lab, grid_sp, filas)
    malla = _malla(lab, grid_sp, fragmentos)
    (carpeta / "malla.json").write_text(json.dumps(malla, separators=(",", ":")), encoding="utf-8")

    np.save(carpeta / "fragmentos.npy", lab)
    cajas = [{k: np.asarray(v).tolist() for k, v in b.items()} for b in pred["boxes"]]
    zs_hueso = [int(z) for z in np.nonzero(lab.reshape(lab.shape[0], -1).max(1) > 0)[0]]
    sec = np.isin(lab, [f["id"] for f in fragmentos if not f["principal"]])
    zs_sec = [int(z) for z in np.nonzero(sec.reshape(sec.shape[0], -1).any(1))[0]]
    del sec
    resumen = {
        "modelo": modelo.nombre, "dispositivo": str(modelo.device).upper(),
        "forma_nativa_zyx": meta["native_shape_zyx"], "spacing_zyx_mm": meta["spacing_zyx"],
        "orientacion_original": orient, "mm_por_px_modelo": round(meta["pixel_mm"], 3),
        "cortes": int(lab.shape[0]), "cortes_con_hueso": zs_hueso, "cortes_secundarios": zs_sec,
        "presencia": pred["cls"].round(3).tolist(), "cajas": cajas,
        "fragmentos": fragmentos, "separacion": filas, "separacion_gt": gt_filas, "metricas_gt": metricas_gt,
        "segundos_total": round(time.time() - t0, 1), "segundos_modelo": round(t_mod, 1),
        "ms_por_corte": round(1000 * t_mod / lab.shape[0], 1), "aviso": AVISO,
        "titulo": titulo, "nota_referencia": nota_referencia,
    }
    (carpeta / "resumen.json").write_text(json.dumps(resumen, default=float), encoding="utf-8")
    avance("Listo", 1.0)
    return resumen


def _ventana(cfg: Dict) -> List[float]:
    w = cfg["data"].get("window", {"level": 400, "width": 1800})
    return [float(w["level"]), float(w["width"])]


def _mip(hu: np.ndarray, spacing, carpeta: Path, avance) -> None:
    """Visualizador 1: MIP del volumen crudo con umbral óseo en HU (sin modelo)."""
    avance("Visualizador 1: MIP del hueso (umbral 250 HU, sin modelo)", 0.10)
    dz, dy, dx = spacing
    ax, cor, sag = compute_bone_mip(hu)
    for nombre, img, (sy, sx) in (("axial", ax, (dy, dx)), ("coronal", cor, (dz, dx)), ("sagital", sag, (dz, dy))):
        _png_hu(img, sy, sx).save(carpeta / f"mip_{nombre}.png")
    frames, (mz, mx) = rotating_mip(hu, spacing, angles_deg=list(range(0, 360, 15)))
    (carpeta / "mip_giro").mkdir(exist_ok=True)
    for k, f in enumerate(frames):
        _png_hu(f, mz, mx).save(carpeta / "mip_giro" / f"{k:02d}.png")


def _png_hu(img: np.ndarray, mm_y: float, mm_x: float) -> Image.Image:
    v = np.clip((img - 250.0) / (DISPLAY_MAX_HU - 250.0), 0, 1)
    im = Image.fromarray((v * 255).astype(np.uint8))
    h, w = img.shape
    ancho = 512
    alto = int(round(ancho * (h * mm_y) / (w * mm_x)))      # proporción física, no de píxeles
    return im.resize((ancho, max(alto, 1)), Image.BILINEAR)


def _tabla_fragmentos(lab: np.ndarray, sp, filas: List[Dict]) -> List[Dict]:
    dist = {f["label"]: f for f in filas}
    vox_cm3 = float(np.prod(sp)) / 1000.0
    cnt = np.bincount(lab.ravel(), minlength=31)
    out = []
    for i in np.nonzero(cnt)[0]:
        if i == 0:
            continue
        reg = REGIONES[(int(i) - 1) // 10 + 1]
        k = (int(i) - 1) % 10 + 1
        d = dist.get(int(i))
        out.append({"id": int(i), "region": reg, "k": k, "principal": k == 1,
                    "volumen_cm3": round(float(cnt[i]) * vox_cm3, 2),
                    "dist_mm": None if d is None else round(d["dist_mm"], 2),
                    "en_contacto": None if d is None else bool(d["in_contact"]),
                    "color": "#%02x%02x%02x" % tono(reg, k)})
    return out


def _malla(lab: np.ndarray, sp, fragmentos: List[Dict]) -> Dict:
    """Marching cubes (skimage) por fragmento sobre la máscara suavizada; coordenadas en mm."""
    from skimage.measure import marching_cubes
    trazas = []
    for f in fragmentos:
        m = lab == f["id"]
        if m.sum() < 30:
            continue
        sl = ndi.find_objects(m.astype(np.uint8))[0]
        sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in sl)
        # σ = 0,5 y no 0,8: con 0,8 una lámina de 1 vóxel queda por debajo de 0,5 tras suavizar
        # (erf(0,44) ≈ 0,47) y desaparece de la malla. En el ala ilíaca del 075 eso agrandaba un
        # hueco de 12 mm² de la máscara a 33 mm² en el dibujo; con 0,5 queda en 16 mm² y el JSON
        # pesa lo mismo. El hueco en sí viene de la grilla de 256 px (03_semana11, sección 1.1).
        sub = ndi.gaussian_filter(m[sl].astype(np.float32), 0.5)
        try:
            v, fc, _, _ = marching_cubes(sub, 0.5, spacing=sp, step_size=2 if f["principal"] else 1)
        except (ValueError, RuntimeError):
            continue
        v += np.array([s.start for s in sl]) * np.array(sp)
        # (z, y, x) en mm -> ejes de la figura: x (izq.-der.), y (ant.-post.), z (sup.)
        c = v.mean(0)
        etiqueta = f"{f['region']} principal" if f["principal"] else (
            f"{f['region']}-{f['k']}: {f['dist_mm']:.1f} mm" if f["dist_mm"] is not None else f"{f['region']}-{f['k']}")
        trazas.append({"id": f["id"], "color": f["color"], "etiqueta": etiqueta, "principal": f["principal"],
                       "x": np.round(v[:, 2], 1).tolist(), "y": np.round(v[:, 1], 1).tolist(),
                       "z": np.round(v[:, 0], 1).tolist(), "i": fc[:, 0].tolist(), "j": fc[:, 1].tolist(),
                       "k": fc[:, 2].tolist(), "centro": [round(float(c[2]), 1), round(float(c[1]), 1),
                                                         round(float(c[0]), 1)]})
    return {"trazas": trazas}


# ---------------------------------------------------------------- visualizador 2

def dibujar_corte(carpeta: Path, z: int, ver: Dict[str, bool]) -> Image.Image:
    """Corte z con máscaras por fragmento, cajas del modelo y la distancia sobre cada secundario."""
    image, _, _ = load_case_cache(carpeta / "cache" / "caso")
    lab = np.load(carpeta / "fragmentos.npy", mmap_mode="r")
    res = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    z = int(np.clip(z, 0, image.shape[0] - 1))
    return _render(np.asarray(image[z]) / 255.0, np.asarray(lab[z]), res["cajas"][z], res["fragmentos"], ver,
                   f"z = {z} / {image.shape[0] - 1}")


def _render(gris: np.ndarray, lab: np.ndarray, cajas: Dict, fragmentos: List[Dict], ver: Dict[str, bool],
            pie: str, mm_px: float | None = None) -> Image.Image:
    S = gris.shape[0] * ESCALA
    base = Image.fromarray((np.clip(gris, 0, 1) * 255).astype(np.uint8)).resize((S, S), Image.BILINEAR).convert("RGB")
    info = {f["id"]: f for f in fragmentos}
    lab_g = np.kron(lab, np.ones((ESCALA, ESCALA), np.uint8))
    if ver.get("mascaras", True) and lab_g.any():
        rgb = np.asarray(base).astype(np.float32)
        capa = np.zeros_like(rgb)
        for i in np.unique(lab_g):
            if i == 0:
                continue
            reg = REGIONES[(int(i) - 1) // 10 + 1]
            capa[lab_g == i] = tono(reg, (int(i) - 1) % 10 + 1)
        m = lab_g > 0
        rgb[m] = 0.55 * rgb[m] + 0.45 * capa[m]
        # contorno de cada fragmento: donde cambia la etiqueta
        borde = np.zeros_like(m)
        borde[1:] |= lab_g[1:] != lab_g[:-1]
        borde[:, 1:] |= lab_g[:, 1:] != lab_g[:, :-1]
        rgb[borde & m] = capa[borde & m]
        base = Image.fromarray(rgb.astype(np.uint8))
    d = ImageDraw.Draw(base)
    f_peq, f_med = _fuente(12), _fuente(14)
    if ver.get("cajas", True) and cajas:
        for b, s, l in zip(cajas["boxes"], cajas["scores"], cajas["labels"]):
            reg = ("SA", "LI", "RI")[int(l)]
            x0, y0, x1, y1 = (float(v) * ESCALA for v in b)
            d.rectangle([x0, y0, x1, y1], outline=COLOR[reg], width=2)
            _texto(d, (x0 + 2, max(y0 - 16, 0)), f"{reg} {s:.2f}", COLOR[reg], f_peq)
    if ver.get("distancias", True):
        ocupado = []
        for i in np.unique(lab):
            if i == 0 or info.get(int(i), {}).get("principal", True):
                continue
            f = info[int(i)]
            ys, xs = np.nonzero(lab == i)
            if ys.size < 3:
                continue
            txt = f"{f['region']}-{f['k']}  {f['dist_mm']:.1f} mm" if f["dist_mm"] is not None else f"{f['region']}-{f['k']}"
            if mm_px is not None and f["dist_mm"] is not None:
                txt += " (2D)"
            x, y = xs.mean() * ESCALA + 6, ys.mean() * ESCALA - 7
            w = d.textlength(txt, font=f_med) + 8
            while any(x < a + aw and a < x + w and abs(y - b) < 19 for a, b, aw in ocupado):
                y += 20                                  # no tapar otra etiqueta: se baja una línea
            ocupado.append((x, y, w))
            d.line([(xs.mean() * ESCALA, ys.mean() * ESCALA), (x, y + 7)], fill=f["color"], width=1)
            _texto(d, (x, y), txt, (255, 255, 255), f_med, borde=f["color"])
    _texto(d, (6, S - 20), pie, (200, 200, 200), f_peq)
    return base


def _texto(d: ImageDraw.ImageDraw, xy, txt: str, color, fuente, borde=None) -> None:
    x, y = xy
    caja = d.textbbox((x, y), txt, font=fuente)
    d.rectangle([caja[0] - 3, caja[1] - 2, caja[2] + 3, caja[3] + 2], fill=(0, 0, 0),
                outline=borde if borde else None)
    d.text((x, y), txt, fill=color, font=fuente)


# ---------------------------------------------------------------- imagen de un solo corte

@torch.no_grad()
def procesar_imagen(modelo: Modelo, img: Image.Image, mm_px: float):
    """Un corte suelto (PNG/JPG): si es RGB se toma como la entrada 2.5D (z−Δ, z, z+Δ) que ve el
    modelo; si es gris, el mismo corte en los tres canales. Sin volumen no hay 3D, así que los
    fragmentos y la distancia son 2D y aproximados (``mm_px`` lo da el usuario)."""
    arr = np.asarray(img.convert("RGB").resize((256, 256), Image.BILINEAR), np.float32) / 255.0
    if np.allclose(arr[..., 0], arr[..., 1]) and np.allclose(arr[..., 1], arr[..., 2]):
        arr = np.repeat(arr[..., 1:2], 3, axis=2)
    x = torch.from_numpy(arr.transpose(2, 0, 1)[None].copy()).to(modelo.device)
    amp = modelo.device.type == "cuda"
    if modelo.tta:
        out = _tta_forward(modelo.net, x, modelo.tta, amp, modelo.device)
        sem, edge = out["tta_sem"].argmax(1), out["tta_edge"][:, 0]
    else:
        with torch.amp.autocast(device_type=modelo.device.type, dtype=torch.float16, enabled=amp):
            out = modelo.net(x)
        sem, edge = out["seg_logits"].argmax(1), torch.sigmoid(out["edge_logits"].float())[:, 0]
    pp = modelo.pp
    det = decode(out["det_scores"], out["det_ltrb"], 8, 256, pp.get("det_score_threshold", 0.3),
                 pp.get("nms_iou", 0.5), pp.get("max_boxes_per_class", 1))[0]
    sem = sem.cpu().numpy().astype(np.uint8)
    edge = edge.float().cpu().numpy()
    # Un corte = volumen de 1 de espesor: se usa el mismo posproceso con dz grande para que no
    # mezcle nada en z
    sp = (10.0, mm_px, mm_px)
    lab = separate_instances(sem, edge, sp, **instance_kwargs(pp))
    filas = separation_table(lab, sp)
    fr = _tabla_fragmentos(lab, sp, filas)
    cajas = {k: v.cpu().numpy() for k, v in det.items()}
    return _render(arr[..., 1], lab[0], cajas, fr, {"mascaras": True, "cajas": True, "distancias": True},
                   f"corte suelto · {mm_px:.2f} mm/px · distancias 2D aproximadas", mm_px=mm_px), fr
