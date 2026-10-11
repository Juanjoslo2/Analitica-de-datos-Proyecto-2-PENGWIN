"""Seguridad del dashboard (05_app): usuarios, bloqueo, inyección SQL, sesiones y validación de archivos."""

import io
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from PIL import Image

APP = Path(__file__).resolve().parents[1] / "05_app"
sys.path.insert(0, str(APP))

from seguridad import MAX_INTENTOS, MSG_BLOQUEADO, MSG_MAL, Usuarios  # noqa: E402
from validacion import ArchivoInvalido, validar_imagen, validar_volumen  # noqa: E402

CLAVE = "clave-larga-123"


@pytest.fixture()
def db(tmp_path):
    u = Usuarios(tmp_path / "u.db")
    u.crear("profesor", CLAVE)
    return u


# ---------------------------------------------------------------- usuarios y bloqueo

def test_ingreso_correcto_da_sesion(db):
    token, msg = db.iniciar_sesion("profesor", CLAVE, "1.1.1.1")
    assert token and msg == "ok"
    assert db.usuario_de(token) == "profesor"
    db.cerrar_sesion(token)
    assert db.usuario_de(token) is None


def test_tres_fallos_bloquean_el_perfil(db):
    for _ in range(MAX_INTENTOS - 1):
        assert db.iniciar_sesion("profesor", "mala-clave-xx", "1.1.1.1") == (None, MSG_MAL)
    assert db.iniciar_sesion("profesor", "mala-clave-xx", "1.1.1.1") == (None, MSG_BLOQUEADO)
    # bloqueado: ni con la clave correcta
    assert db.iniciar_sesion("profesor", CLAVE, "1.1.1.1") == (None, MSG_BLOQUEADO)
    assert db.fijar_bloqueo("profesor", False)
    assert db.iniciar_sesion("profesor", CLAVE, "1.1.1.1")[0]


def test_un_acierto_reinicia_el_conteo(db):
    for _ in range(MAX_INTENTOS - 1):
        db.iniciar_sesion("profesor", "mala-clave-xx", "1.1.1.1")
    assert db.iniciar_sesion("profesor", CLAVE, "1.1.1.1")[0]
    for _ in range(MAX_INTENTOS - 1):
        db.iniciar_sesion("profesor", "mala-clave-xx", "1.1.1.1")
    assert db.iniciar_sesion("profesor", CLAVE, "1.1.1.1")[0]       # aún no se bloqueó


def test_usuario_inexistente_da_el_mismo_mensaje(db):
    assert db.iniciar_sesion("nadie", "lo-que-sea-123", "1.1.1.1") == (None, MSG_MAL)


def test_bloquear_cierra_las_sesiones_abiertas(db):
    token, _ = db.iniciar_sesion("profesor", CLAVE, "1.1.1.1")
    db.fijar_bloqueo("profesor", True)
    assert db.usuario_de(token) is None


@pytest.mark.parametrize("usuario", ["profesor' OR '1'='1", "' OR 1=1 --", "profesor'; DROP TABLE usuarios; --",
                                     "profesor\" OR \"\"=\"", "admin'/*"])
def test_inyeccion_sql_en_el_usuario(db, usuario):
    assert db.iniciar_sesion(usuario, "x' OR '1'='1", "1.1.1.1") == (None, MSG_MAL)
    assert [r[0] for r in db.listar()] == ["profesor"]                # la tabla sigue intacta


def test_inyeccion_sql_en_la_clave_no_entra(db):
    assert db.iniciar_sesion("profesor", "' OR '1'='1' --", "1.1.1.1")[0] is None


def test_la_clave_no_se_guarda_en_claro(db, tmp_path):
    with sqlite3.connect(tmp_path / "u.db") as c:
        sal, h = c.execute("SELECT sal, hash FROM usuarios").fetchone()
    assert CLAVE.encode() not in h and len(sal) == 16 and len(h) == 64


def test_politica_de_usuario_y_clave(tmp_path):
    u = Usuarios(tmp_path / "v.db")
    with pytest.raises(ValueError):
        u.crear("x'; DROP TABLE usuarios; --", CLAVE)
    with pytest.raises(ValueError):
        u.crear("corto", "123")


# ---------------------------------------------------------------- validación de archivos

def _volumen(dims=(16, 20, 24)):
    img = sitk.GetImageFromArray(np.random.default_rng(0).integers(-1000, 1500, dims[::-1]).astype(np.int16))
    img.SetSpacing((0.8, 0.8, 1.0))
    return img


@pytest.mark.parametrize("ext", [".mha", ".nii", ".nii.gz", ".nrrd"])
def test_volumenes_validos(tmp_path, ext):
    p = tmp_path / f"v{ext}"
    sitk.WriteImage(_volumen(), str(p), useCompression=ext != ".nii")
    info = validar_volumen(p, p.name)
    assert sorted(info["dims_xyz"]) == [16, 20, 24]


def test_mha_que_apunta_a_otro_archivo_se_rechaza(tmp_path):
    p = tmp_path / "v.mha"
    sitk.WriteImage(_volumen(), str(p), useCompression=False)
    secreto = tmp_path / "secreto.raw"
    secreto.write_bytes(b"\x00" * (16 * 20 * 24 * 2))
    b = p.read_bytes()
    i = b.index(b"ElementDataFile")
    p.write_bytes(b[:i] + f"ElementDataFile = {secreto}\n".encode())     # solo cabecera, datos afuera
    with pytest.raises(ArchivoInvalido, match="LOCAL"):
        validar_volumen(p, p.name)


def test_nrrd_separado_se_rechaza(tmp_path):
    p = tmp_path / "v.nrrd"
    p.write_bytes(b"NRRD0004\ntype: short\ndimension: 3\nsizes: 16 20 24\nencoding: raw\n"
                  b"data file: /etc/passwd\n\n")
    with pytest.raises(ArchivoInvalido, match="data file"):
        validar_volumen(p, p.name)


@pytest.mark.parametrize("contenido", [b"MZ\x90\x00 ejecutable de windows", b"#!/bin/sh\nrm -rf /\n", b"<?php system($_GET[c]); ?>",
                                       b"PK\x03\x04 un zip"])
def test_archivo_disfrazado_de_volumen(tmp_path, contenido):
    for ext in (".mha", ".nii", ".nrrd", ".nii.gz"):
        p = tmp_path / f"x{ext}"
        p.write_bytes(contenido)
        with pytest.raises(ArchivoInvalido):
            validar_volumen(p, p.name)


def test_extension_no_admitida(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"MZ")
    with pytest.raises(ArchivoInvalido, match="formato"):
        validar_volumen(p, p.name)


def test_volumen_2d_o_enorme_se_rechaza(tmp_path):
    p = tmp_path / "v.mha"
    sitk.WriteImage(sitk.Image(64, 64, sitk.sitkInt16), str(p))
    with pytest.raises(ArchivoInvalido):
        validar_volumen(p, p.name)
    b = tmp_path / "bomba.nrrd"
    b.write_bytes(b"NRRD0004\ntype: short\ndimension: 3\nsizes: 2048 2048 2048\nencoding: gzip\n\n")
    with pytest.raises(ArchivoInvalido, match="grande"):
        validar_volumen(b, b.name)


def _img(fmt, size=(256, 256)):
    buf = io.BytesIO()
    Image.new("L", size, 128).save(buf, fmt)
    return buf.getvalue()


def test_imagenes_validas_e_invalidas():
    assert validar_imagen(_img("PNG")).size == (256, 256)
    assert validar_imagen(_img("JPEG")).size == (256, 256)
    for malo in (_img("GIF"), _img("BMP"), b"MZ\x90\x00" + b"\x00" * 100, b"\x89PNG\r\n\x1a\n" + b"basura" * 10):
        with pytest.raises(ArchivoInvalido):
            validar_imagen(malo)
    with pytest.raises(ArchivoInvalido):
        validar_imagen(_img("PNG", (6000, 6000)))                       # 36 Mpx: posible bomba


# ---------------------------------------------------------------- rutas protegidas

@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    ckpt = APP.parent / "checkpoints" / "v2_last" / "best.pth"
    if not ckpt.exists():
        pytest.skip("sin checkpoint v2_last")
    monkeypatch.setenv("PENGWIN_USUARIOS", str(tmp_path / "u.db"))
    monkeypatch.setenv("PENGWIN_TRABAJOS", str(tmp_path / "trabajos"))
    monkeypatch.setenv("PENGWIN_EJEMPLOS", str(tmp_path / "ejemplos"))
    monkeypatch.setenv("PENGWIN_COOKIE_INSEGURA", "1")
    sys.modules.pop("app", None)
    import app as modulo
    from fastapi.testclient import TestClient
    modulo.usuarios.crear("profesor", CLAVE)
    return TestClient(modulo.app)


def test_sin_sesion_no_se_entra(cliente):
    assert cliente.get("/api/info").status_code == 401
    r = cliente.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert cliente.get("/login").status_code == 200
    assert cliente.post("/api/subida", data={"nombre": "a.mha", "tamano": 10}).status_code == 401


def test_con_sesion_y_cabeceras(cliente):
    assert cliente.post("/api/login", json={"usuario": "profesor", "clave": "mala-1234567"}).status_code == 401
    r = cliente.post("/api/login", json={"usuario": "profesor", "clave": CLAVE})
    assert r.status_code == 200 and "httponly" in r.headers["set-cookie"].lower()
    r = cliente.get("/api/info")
    assert r.status_code == 200 and r.json()["usuario"] == "profesor"
    assert r.headers["x-frame-options"] == "DENY" and "default-src 'self'" in r.headers["content-security-policy"]
    # otro origen no puede disparar acciones (CSRF)
    r = cliente.post("/api/subida", data={"nombre": "a.mha", "tamano": 10}, headers={"Origin": "https://malo.example"})
    assert r.status_code == 403
    # extensión no admitida y archivo disfrazado
    assert cliente.post("/api/subida", data={"nombre": "a.exe", "tamano": 10}).status_code == 400
    malo = b"#!/bin/sh\nrm -rf /\n"
    tid = cliente.post("/api/subida", data={"nombre": "a.mha", "tamano": len(malo)}).json()["id"]
    assert cliente.put(f"/api/subida/{tid}", content=malo).json()["recibido"] == len(malo)
    r = cliente.post(f"/api/subida/{tid}/fin")
    assert r.status_code == 400 and "rechazado" in r.json()["detail"]
    assert cliente.get("/api/trabajo/../../etc/passwd").status_code == 404
