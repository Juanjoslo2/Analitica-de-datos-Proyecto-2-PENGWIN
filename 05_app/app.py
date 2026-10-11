"""app.py.

Dashboard PENGWIN: FastAPI + una página (static/index.html) con los tres visualizadores.

    uvicorn app:app --host 127.0.0.1 --port 8780         (desde 05_app/)

Variables de entorno:
    PENGWIN_CKPT       checkpoint (por defecto checkpoints/v2_last/best.pth)
    PENGWIN_CONFIG     config de inferencia (por defecto configs/tuned.yaml)
    PENGWIN_TRABAJOS   carpeta de trabajos (por defecto 05_app/trabajos, no se versiona)
    PENGWIN_EJEMPLOS   carpeta con casos de ejemplo ya procesados (preparar_ejemplos.py)
    PENGWIN_USUARIOS   base SQLite de usuarios (por defecto 05_app/datos/usuarios.db, no se versiona)
    PENGWIN_COOKIE_INSEGURA=1   solo para probar en http://localhost (la cookie normal exige HTTPS)

Seguridad (seguridad.py, validacion.py):
    - Todo exige sesión salvo la página de ingreso. Usuarios en SQLite con scrypt; 3 claves
      equivocadas seguidas bloquean el perfil (se desbloquea con usuarios.py).
    - Cada trabajo subido pertenece a su usuario; los ejemplos los ven todos los usuarios.
    - Los archivos se validan por contenido antes de llegar al modelo (solo volúmenes de un archivo
      y cortes PNG/JPEG); el nombre del archivo nunca se usa como ruta.
    - Peticiones que cambian algo (POST/PUT) deben venir del mismo origen (protección CSRF además
      de la cookie SameSite=Strict).

Los volúmenes se suben por trozos de 8 MB (un CT de PENGWIN pesa 30-430 MB). Un trabajo a la vez en
la GPU; los trabajos subidos se borran a las 6 h.
"""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

import servicio
from seguridad import LimiteIntentos, Usuarios
from validacion import ArchivoInvalido, extension, validar_imagen, validar_volumen

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent
CKPT = Path(os.environ.get("PENGWIN_CKPT", REPO / "checkpoints" / "v2_last" / "best.pth"))
CONFIG = Path(os.environ.get("PENGWIN_CONFIG", REPO / "configs" / "tuned.yaml"))
TRABAJOS = Path(os.environ.get("PENGWIN_TRABAJOS", AQUI / "trabajos"))
EJEMPLOS = Path(os.environ.get("PENGWIN_EJEMPLOS", AQUI / "ejemplos"))
USUARIOS = Path(os.environ.get("PENGWIN_USUARIOS", AQUI / "datos" / "usuarios.db"))
COOKIE_SEGURA = os.environ.get("PENGWIN_COOKIE_INSEGURA") != "1"
COOKIE = "pengwin_sesion"
MAX_BYTES = 700 * 1024 ** 2
TROZO_MAX = 9 * 1024 ** 2
VIDA_S = 6 * 3600
ID_RE = re.compile(r"^[a-z0-9_-]{1,40}$")
PUBLICAS = {"/login", "/api/login", "/static/login.js"}

TRABAJOS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="PENGWIN · dashboard", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(GZipMiddleware, minimum_size=2048)
modelo = servicio.Modelo(CKPT, CONFIG)
usuarios = Usuarios(USUARIOS)
limite_login = LimiteIntentos(maximo=10, ventana_s=600)
estado: dict[str, dict] = {}
cola: "queue.Queue[str]" = queue.Queue()
candado = threading.Lock()

CABECERAS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
                                "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
}


def _ip(request: Request) -> str:
    # nginx pone X-Real-IP; si no hay proxy, la IP de la conexión
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "?")


@app.middleware("http")
async def proteger(request: Request, call_next):
    ruta = request.url.path
    if request.method in ("POST", "PUT", "DELETE"):
        origen = request.headers.get("origin")
        if origen and urlparse(origen).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "origen no permitido"}, status_code=403, headers=CABECERAS)
    if ruta not in PUBLICAS:
        usuario = usuarios.usuario_de(request.cookies.get(COOKIE))
        if usuario is None:
            if ruta.startswith("/api/"):
                return JSONResponse({"detail": "inicie sesión"}, status_code=401, headers=CABECERAS)
            return RedirectResponse("/login", status_code=303, headers=CABECERAS)
        request.state.usuario = usuario
    resp = await call_next(request)
    for k, v in CABECERAS.items():
        resp.headers.setdefault(k, v)
    return resp


# ---------------------------------------------------------------- sesión

@app.get("/login")
def pagina_login():
    return FileResponse(AQUI / "static" / "login.html", headers={"Cache-Control": "no-store"})


@app.post("/api/login")
async def login(request: Request):
    ip = _ip(request)
    if not limite_login.permitido(ip):
        raise HTTPException(429, "Demasiados intentos desde esta conexión. Espere unos minutos.")
    try:
        datos = await request.json()
        u, c = str(datos.get("usuario", ""))[:64], str(datos.get("clave", ""))[:256]
    except Exception:                                    # noqa: BLE001
        raise HTTPException(400, "petición inválida")
    token, msg = usuarios.iniciar_sesion(u, c, ip)
    if token is None:
        return JSONResponse({"detail": msg}, status_code=401)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, token, max_age=8 * 3600, httponly=True, secure=COOKIE_SEGURA, samesite="strict", path="/")
    return resp


@app.post("/api/logout")
def logout(request: Request):
    usuarios.cerrar_sesion(request.cookies.get(COOKIE))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


# ---------------------------------------------------------------- trabajos

def _carpeta(tid: str, request: Request) -> Path:
    if not ID_RE.match(tid):
        raise HTTPException(404)
    if (EJEMPLOS / tid / "resumen.json").exists():
        return EJEMPLOS / tid
    st = estado.get(tid)
    if st is None or st["dueno"] != request.state.usuario:      # un usuario no ve trabajos de otro
        raise HTTPException(404, "trabajo no encontrado")
    return TRABAJOS / tid


def _trabajador() -> None:
    while True:
        tid = cola.get()
        st = estado[tid]
        st.update(estado_="procesando", inicio=time.time())

        def avance(msg: str, frac: float, st=st) -> None:
            st.update(mensaje=msg, progreso=frac)

        try:
            with candado:
                servicio.procesar_volumen(modelo, Path(st["ruta"]), TRABAJOS / tid, avance)
            st.update(estado_="listo")
        except Exception:                                # noqa: BLE001 - no se muestran rutas ni trazas
            st.update(estado_="error", mensaje="No se pudo procesar el volumen. Verifique que sea un CT de pelvis.")
        finally:
            Path(st["ruta"]).unlink(missing_ok=True)


def _limpieza() -> None:
    while True:
        ahora = time.time()
        for d in TRABAJOS.iterdir():
            if d.is_dir() and ahora - d.stat().st_mtime > VIDA_S:
                shutil.rmtree(d, ignore_errors=True)
                estado.pop(d.name, None)
        time.sleep(600)


threading.Thread(target=_trabajador, daemon=True).start()
threading.Thread(target=_limpieza, daemon=True).start()


@app.get("/")
def inicio():
    return FileResponse(AQUI / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/static/{nombre}")
def estatico(nombre: str):
    if nombre not in ("app.js", "login.js"):
        raise HTTPException(404)
    return FileResponse(AQUI / "static" / nombre, headers={"Cache-Control": "no-cache"})


@app.get("/api/info")
def info(request: Request):
    ejemplos = []
    for d in sorted(EJEMPLOS.glob("*/resumen.json")):
        r = json.loads(d.read_text(encoding="utf-8"))
        ejemplos.append({"id": d.parent.name, "titulo": r.get("titulo", d.parent.name),
                         "fragmentos": len(r["fragmentos"]), "cortes": r["cortes"]})
    return {"modelo": modelo.nombre, "dispositivo": str(modelo.device).upper(), "ejemplos": ejemplos,
            "aviso": servicio.AVISO, "en_cola": cola.qsize(), "usuario": request.state.usuario}


@app.post("/api/subida")
def subida_inicio(request: Request, nombre: str = Form(...), tamano: int = Form(...)):
    try:
        ext = extension(nombre)
    except ArchivoInvalido as e:
        raise HTTPException(400, str(e))
    if not 0 < tamano <= MAX_BYTES:
        raise HTTPException(413, "el volumen supera 700 MB")
    activos = [t for t, s in estado.items() if s["dueno"] == request.state.usuario and s["estado_"] in ("subiendo", "en cola", "procesando")]
    if activos:
        raise HTTPException(429, "ya tiene un volumen en proceso; espere a que termine")
    if cola.qsize() >= 3:
        raise HTTPException(429, "hay 3 volúmenes en cola; intente en unos minutos")
    tid = uuid.uuid4().hex[:12]
    (TRABAJOS / tid).mkdir()
    estado[tid] = {"estado_": "subiendo", "ruta": str(TRABAJOS / tid / f"entrada{ext}"), "tamano": tamano,
                   "recibido": 0, "nombre": nombre[:120], "mensaje": "Subiendo", "progreso": 0.0,
                   "dueno": request.state.usuario}
    return {"id": tid}


@app.put("/api/subida/{tid}")
async def subida_trozo(tid: str, request: Request):
    st = estado.get(tid)
    if not st or st["estado_"] != "subiendo" or st["dueno"] != request.state.usuario:
        raise HTTPException(404)
    datos = await request.body()
    if len(datos) > TROZO_MAX or st["recibido"] + len(datos) > st["tamano"]:
        raise HTTPException(400, "trozo inválido")
    with open(st["ruta"], "ab") as f:
        f.write(datos)
    st["recibido"] += len(datos)
    return {"recibido": st["recibido"]}


@app.post("/api/subida/{tid}/fin")
def subida_fin(tid: str, request: Request):
    st = estado.get(tid)
    if not st or st["estado_"] != "subiendo" or st["dueno"] != request.state.usuario:
        raise HTTPException(404)
    if st["recibido"] != st["tamano"]:
        raise HTTPException(400, "faltan trozos")
    try:
        st["validacion"] = validar_volumen(st["ruta"], st["nombre"])      # antes de tocar la cola o el modelo
    except ArchivoInvalido as e:
        shutil.rmtree(TRABAJOS / tid, ignore_errors=True)
        estado.pop(tid, None)
        raise HTTPException(400, f"Archivo rechazado: {e}")
    st.update(estado_="en cola", mensaje=f"En cola ({cola.qsize()} antes)", progreso=0.0)
    cola.put(tid)
    return {"id": tid}


@app.get("/api/trabajo/{tid}")
def trabajo(tid: str, request: Request):
    _carpeta(tid, request)
    if tid in estado:
        st = estado[tid]
        return {k.rstrip("_"): v for k, v in st.items() if k in ("estado_", "mensaje", "progreso", "recibido", "tamano")}
    return {"estado": "listo", "progreso": 1.0}


# ---------------------------------------------------------------- resultados

@app.get("/api/trabajo/{tid}/resumen")
def resumen(tid: str, request: Request):
    return FileResponse(_carpeta(tid, request) / "resumen.json")


@app.get("/api/trabajo/{tid}/malla")
def malla(tid: str, request: Request):
    return FileResponse(_carpeta(tid, request) / "malla.json")


@app.get("/api/trabajo/{tid}/mip/{vista}")
def mip(tid: str, vista: str, request: Request):
    c = _carpeta(tid, request)
    if vista in ("axial", "coronal", "sagital"):
        return FileResponse(c / f"mip_{vista}.png")
    if vista.isdigit() and int(vista) < 24:
        return FileResponse(c / "mip_giro" / f"{int(vista):02d}.png")
    raise HTTPException(404)


@app.get("/api/trabajo/{tid}/corte/{z}")
def corte(tid: str, z: int, request: Request, mascaras: int = 1, cajas: int = 1, distancias: int = 1):
    img = servicio.dibujar_corte(_carpeta(tid, request), z, {"mascaras": bool(mascaras), "cajas": bool(cajas),
                                                             "distancias": bool(distancias)})
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=False, compress_level=3)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/imagen")
async def imagen(archivo: UploadFile = File(...), mm_px: float = Form(1.3)):
    datos = await archivo.read(20 * 1024 ** 2 + 1)
    if len(datos) > 20 * 1024 ** 2:
        raise HTTPException(413, "la imagen supera 20 MB")
    try:
        img = validar_imagen(datos)
    except ArchivoInvalido as e:
        raise HTTPException(400, f"Archivo rechazado: {e}")
    with candado:
        out, fr = servicio.procesar_imagen(modelo, img, float(min(max(mm_px, 0.2), 5.0)))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return JSONResponse({"png": base64.b64encode(buf.getvalue()).decode(), "fragmentos": fr})
