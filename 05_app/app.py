"""app.py.

Dashboard PENGWIN: FastAPI + una página (static/index.html) con los tres visualizadores.

    uvicorn app:app --host 127.0.0.1 --port 8780         (desde 05_app/)

Variables de entorno:
    PENGWIN_CKPT       checkpoint (por defecto checkpoints/v2_last/best.pth)
    PENGWIN_CONFIG     config de inferencia (por defecto configs/tuned.yaml)
    PENGWIN_TRABAJOS   carpeta de trabajos (por defecto 05_app/trabajos, no se versiona)
    PENGWIN_EJEMPLOS   carpeta con casos de ejemplo ya procesados (preparar_ejemplos.py)
    PENGWIN_USUARIOS   base SQLite de usuarios (por defecto 05_app/datos/usuarios.db, no se versiona)
    PENGWIN_CTS        carpeta con PENGWIN_CT_train_images_part1/2: si alguien sube la MÁSCARA de un caso
                       (PENGWIN_CT_train_labels), se analiza el CT de ese caso y la máscara queda como referencia
    PENGWIN_COOKIE_INSEGURA=1   solo para probar en http://localhost (la cookie normal exige HTTPS)

Seguridad (seguridad.py, validacion.py):
    - Todo exige sesión salvo la página de ingreso. Usuarios en SQLite con scrypt; 3 claves
      equivocadas seguidas bloquean el perfil (se desbloquea con usuarios.py).
    - Cada trabajo subido pertenece a su usuario; los ejemplos los ven todos los usuarios.
    - Los archivos se validan por contenido antes de llegar al modelo (solo volúmenes de un archivo
      y cortes PNG/JPEG); el nombre del archivo nunca se usa como ruta.
    - Peticiones que cambian algo (POST/PUT) deben venir del mismo origen (protección CSRF además
      de la cookie SameSite=Strict).

Subida de volúmenes (un CT de PENGWIN pesa 30-430 MB): trozos de 8 MB que el navegador comprime con
gzip (un CT baja a la mitad o menos) y manda 4 a la vez por WebSocket. Cada WebSocket es una conexión
TCP propia, porque nginx no lleva WebSocket sobre HTTP/2; por HTTP/2 todo va en una sola conexión y
con los ~180 ms de ida y vuelta Cali-Europa-Bogotá un trozo subía a 0,3 MB/s. Si el WebSocket no abre
(algún proxy lo bloquea), la página manda los mismos trozos por PUT, de a uno. Cada trozo trae su
número, así que puede llegar en cualquier orden o repetirse tras un corte sin dañar el archivo.

Un trabajo a la vez en la GPU; los trabajos subidos se borran a las 6 h.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import queue
import re
import shutil
import struct
import threading
import time
import uuid
import zlib
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
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
CTS = os.environ.get("PENGWIN_CTS")
COOKIE_SEGURA = os.environ.get("PENGWIN_COOKIE_INSEGURA") != "1"
COOKIE = "pengwin_sesion"
MAX_BYTES = 700 * 1024 ** 2
TROZO = 8 * 1024 ** 2                       # tamaño de cada trozo sin comprimir (el último puede ser menor)
TROZO_MAX = 9 * 1024 ** 2                   # lo máximo que puede pesar un trozo al llegar, comprimido o no
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
                try:
                    servicio.procesar_volumen(modelo, Path(st["ruta"]), TRABAJOS / tid, avance)
                except servicio.EsMascara:
                    # subieron la máscara de PENGWIN: si su CT está en el servidor, se analiza ese CT
                    if not servicio.analizar_mascara(modelo, Path(st["ruta"]), TRABAJOS / tid, avance, indice_ct):
                        raise
            st.update(estado_="listo")
        except servicio.NoEsCT as e:                     # mensaje escrito para el usuario (p. ej. subió la máscara)
            msg = str(e)
            caso = re.fullmatch(r"(\d{3})\.mha", st["nombre"])
            if isinstance(e, servicio.EsMascara) and caso and 1 <= int(caso[1]) <= 100:
                msg += f" Para este archivo: PENGWIN_CT_train_images_part{1 if int(caso[1]) <= 50 else 2}/{caso[1]}.mha"
            st.update(estado_="error", mensaje=msg)
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


indice_ct: dict = {}


def _indexar_cts() -> None:
    # solo cabeceras: ~1 s para los 100 CT
    if CTS and Path(CTS).is_dir():
        indice_ct.update(servicio.indice_cts(Path(CTS)))


threading.Thread(target=_trabajador, daemon=True).start()
threading.Thread(target=_limpieza, daemon=True).start()
threading.Thread(target=_indexar_cts, daemon=True).start()


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
    ruta = TRABAJOS / tid / f"entrada{ext}"
    with open(ruta, "wb") as f:                  # el archivo nace con su tamaño final: cada trozo va a su lugar
        f.truncate(tamano)
    estado[tid] = {"estado_": "subiendo", "ruta": str(ruta), "tamano": tamano, "recibido": 0,
                   "trozos": set(), "n_trozos": -(-tamano // TROZO), "nombre": nombre[:120],
                   "mensaje": "Subiendo", "progreso": 0.0, "dueno": request.state.usuario}
    return {"id": tid, "trozo": TROZO}


def _escribir_trozo(st: dict, i: int, n: int, comprimido: bool, datos) -> None:
    """Valida el trozo i (n bytes sin comprimir) y lo escribe en su lugar del archivo.

    Corre en un hilo: descomprimir 8 MB no debe frenar el resto de peticiones."""
    if not 0 <= i < st["n_trozos"] or n != min(TROZO, st["tamano"] - i * TROZO) or len(datos) > TROZO_MAX:
        raise ValueError("trozo inválido")
    if comprimido:
        d = zlib.decompressobj(wbits=31)                     # gzip, como lo produce CompressionStream
        try:
            crudo = d.decompress(datos, n + 1)               # nunca más de lo declarado: no hay bomba posible
        except zlib.error:
            raise ValueError("trozo comprimido dañado")
        if len(crudo) != n or not d.eof or d.unused_data:
            raise ValueError("trozo comprimido dañado")
    else:
        crudo = datos
        if len(crudo) != n:
            raise ValueError("trozo inválido")
    with open(st["ruta"], "r+b") as f:
        f.seek(i * TROZO)
        f.write(crudo)


def _anotar_trozo(st: dict, i: int, n: int) -> None:
    # solo desde el hilo del servidor (no desde el hilo que escribe): no hay carreras en la cuenta
    if i not in st["trozos"]:
        st["trozos"].add(i)
        st["recibido"] += n


@app.put("/api/subida/{tid}")
async def subida_trozo(tid: str, request: Request, i: int, n: int, z: int = 0):
    """Vía de respaldo, un trozo por petición, cuando el WebSocket no abre."""
    st = estado.get(tid)
    if not st or st["estado_"] != "subiendo" or st["dueno"] != request.state.usuario:
        raise HTTPException(404)
    datos = await request.body()
    try:
        await asyncio.to_thread(_escribir_trozo, st, i, n, bool(z), datos)
    except ValueError as e:
        raise HTTPException(400, str(e))
    _anotar_trozo(st, i, n)
    return {"i": i, "recibido": st["recibido"]}


@app.websocket("/api/subida/{tid}/ws")
async def subida_ws(ws: WebSocket, tid: str):
    """Vía rápida: el navegador abre varios de estos y manda un trozo por mensaje binario.

    Mensaje = 9 bytes de cabecera (número de trozo u32, bytes sin comprimir u32, gzip sí/no u8,
    little-endian) + el trozo. Se responde {"i", "recibido"} o {"i", "error"}.
    El middleware HTTP no corre en WebSocket: sesión, dueño y origen se revisan aquí."""
    origen = ws.headers.get("origin")
    usuario = usuarios.usuario_de(ws.cookies.get(COOKIE))
    st = estado.get(tid)
    if ((origen and urlparse(origen).netloc != ws.headers.get("host")) or usuario is None or not st
            or st["dueno"] != usuario or st["estado_"] != "subiendo"):
        await ws.close(code=1008)                            # antes de aceptar: el navegador recibe un 403
        return
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            datos = msg.get("bytes")
            if not datos or len(datos) < 9 or st["estado_"] != "subiendo":
                await ws.close(code=1008)
                return
            i, n, z = struct.unpack_from("<IIB", datos)
            try:
                await asyncio.to_thread(_escribir_trozo, st, i, n, bool(z), memoryview(datos)[9:])
            except (ValueError, OSError) as e:
                await ws.send_json({"i": i, "error": str(e) if isinstance(e, ValueError) else "no se pudo guardar"})
                continue
            _anotar_trozo(st, i, n)
            await ws.send_json({"i": i, "recibido": st["recibido"]})
    except WebSocketDisconnect:                              # el navegador se fue a mitad de una respuesta
        return


@app.post("/api/subida/{tid}/fin")
def subida_fin(tid: str, request: Request):
    st = estado.get(tid)
    if not st or st["estado_"] != "subiendo" or st["dueno"] != request.state.usuario:
        raise HTTPException(404)
    if len(st["trozos"]) != st["n_trozos"]:
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
