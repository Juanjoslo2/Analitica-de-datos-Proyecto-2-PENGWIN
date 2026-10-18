"""seguridad.py.

Usuarios del dashboard en SQLite, sesiones y bloqueo por intentos fallidos.

- Claves con scrypt (``hashlib.scrypt``, n = 2^14, r = 8, p = 1) y sal aleatoria de 16 bytes por
  usuario; se comparan con ``hmac.compare_digest`` (tiempo constante).
- **3 claves equivocadas seguidas bloquean el perfil.** Solo se desbloquea desde el servidor con
  ``python usuarios.py desbloquear <usuario>``. Un acierto antes del tercer fallo reinicia el conteo.
- Todas las consultas son parametrizadas (``?``): ningún dato del usuario se pega en el texto SQL.
  Además el nombre de usuario se valida con una expresión regular antes de llegar a la base.
- La sesión es un token aleatorio de 256 bits que viaja en una cookie HttpOnly, Secure y
  SameSite=Strict; en la base solo se guarda su SHA-256, así que robar la base no da sesiones.
- Un usuario inexistente gasta el mismo tiempo de scrypt que uno real y recibe el mismo mensaje:
  no se puede averiguar qué usuarios existen.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

MAX_INTENTOS = 3
VIDA_SESION_S = 8 * 3600
USUARIO_RE = re.compile(r"^[a-z0-9._-]{3,32}$")
CLAVE_MIN = 10
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 64, "maxmem": 64 * 1024 * 1024}
_SAL_FALSA = secrets.token_bytes(16)

MSG_MAL = "Usuario o clave incorrectos. Tras 3 intentos fallidos seguidos el perfil se bloquea."
MSG_BLOQUEADO = "Este perfil está bloqueado por intentos fallidos. Pida al administrador que lo desbloquee."


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_clave(clave: str, sal: bytes) -> bytes:
    return hashlib.scrypt(clave.encode("utf-8"), salt=sal, **_SCRYPT)


class Usuarios:
    def __init__(self, ruta: Path | str):
        self.ruta = str(ruta)
        Path(self.ruta).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._con() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS usuarios (
                    id INTEGER PRIMARY KEY,
                    usuario TEXT NOT NULL UNIQUE,
                    sal BLOB NOT NULL,
                    hash BLOB NOT NULL,
                    intentos INTEGER NOT NULL DEFAULT 0,
                    bloqueado INTEGER NOT NULL DEFAULT 0,
                    creado TEXT NOT NULL,
                    ultimo_acceso TEXT
                );
                CREATE TABLE IF NOT EXISTS sesiones (
                    token_sha256 TEXT PRIMARY KEY,
                    usuario_id INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
                    expira REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS eventos (
                    fecha TEXT NOT NULL, usuario TEXT, ip TEXT, evento TEXT NOT NULL
                );
            """)
        try:
            os.chmod(self.ruta, 0o600)
        except OSError:
            pass

    def _con(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.ruta, timeout=10, isolation_level=None)
        c.execute("PRAGMA foreign_keys = ON")
        c.execute("PRAGMA journal_mode = WAL")
        return c

    def _evento(self, c, usuario, ip, evento) -> None:
        c.execute("INSERT INTO eventos (fecha, usuario, ip, evento) VALUES (?, ?, ?, ?)",
                  (_ahora(), usuario, ip, evento))

    # ------------------------------------------------------------ administración (CLI)
    def crear(self, usuario: str, clave: str) -> None:
        if not USUARIO_RE.match(usuario):
            raise ValueError("usuario: 3-32 caracteres, solo minúsculas, números, punto, guion y guion bajo")
        if len(clave) < CLAVE_MIN:
            raise ValueError(f"la clave debe tener al menos {CLAVE_MIN} caracteres")
        sal = secrets.token_bytes(16)
        with self._lock, self._con() as c:
            c.execute("INSERT INTO usuarios (usuario, sal, hash, creado) VALUES (?, ?, ?, ?)",
                      (usuario, sal, hash_clave(clave, sal), _ahora()))
            self._evento(c, usuario, "cli", "creado")

    def cambiar_clave(self, usuario: str, clave: str) -> bool:
        if len(clave) < CLAVE_MIN:
            raise ValueError(f"la clave debe tener al menos {CLAVE_MIN} caracteres")
        sal = secrets.token_bytes(16)
        with self._lock, self._con() as c:
            n = c.execute("UPDATE usuarios SET sal = ?, hash = ?, intentos = 0 WHERE usuario = ?",
                          (sal, hash_clave(clave, sal), usuario)).rowcount
            c.execute("DELETE FROM sesiones WHERE usuario_id = (SELECT id FROM usuarios WHERE usuario = ?)", (usuario,))
            self._evento(c, usuario, "cli", "clave cambiada")
        return n == 1

    def fijar_bloqueo(self, usuario: str, bloqueado: bool) -> bool:
        with self._lock, self._con() as c:
            n = c.execute("UPDATE usuarios SET bloqueado = ?, intentos = 0 WHERE usuario = ?",
                          (1 if bloqueado else 0, usuario)).rowcount
            if bloqueado:
                c.execute("DELETE FROM sesiones WHERE usuario_id = (SELECT id FROM usuarios WHERE usuario = ?)", (usuario,))
            self._evento(c, usuario, "cli", "bloqueado" if bloqueado else "desbloqueado")
        return n == 1

    def eliminar(self, usuario: str) -> bool:
        with self._lock, self._con() as c:
            n = c.execute("DELETE FROM usuarios WHERE usuario = ?", (usuario,)).rowcount
            self._evento(c, usuario, "cli", "eliminado")
        return n == 1

    def listar(self):
        with self._con() as c:
            return c.execute("SELECT usuario, intentos, bloqueado, creado, ultimo_acceso FROM usuarios ORDER BY usuario").fetchall()

    def eventos(self, n: int = 30):
        with self._con() as c:
            return c.execute("SELECT fecha, usuario, ip, evento FROM eventos ORDER BY rowid DESC LIMIT ?", (n,)).fetchall()

    # ------------------------------------------------------------ inicio de sesión
    def iniciar_sesion(self, usuario: str, clave: str, ip: str) -> tuple[str | None, str]:
        """Devuelve (token, mensaje). token None = rechazado."""
        usuario = (usuario or "").strip().lower()
        clave = clave or ""
        if not USUARIO_RE.match(usuario) or len(clave) > 256:
            hash_clave(clave[:256], _SAL_FALSA)              # mismo costo: no revela nada
            return None, MSG_MAL
        with self._lock, self._con() as c:
            fila = c.execute("SELECT id, sal, hash, intentos, bloqueado FROM usuarios WHERE usuario = ?",
                             (usuario,)).fetchone()
            if fila is None:
                hash_clave(clave, _SAL_FALSA)
                self._evento(c, usuario, ip, "usuario inexistente")
                return None, MSG_MAL
            uid, sal, h, intentos, bloqueado = fila
            if bloqueado:
                hash_clave(clave, _SAL_FALSA)
                self._evento(c, usuario, ip, "intento sobre perfil bloqueado")
                return None, MSG_BLOQUEADO
            if not hmac.compare_digest(hash_clave(clave, sal), h):
                intentos += 1
                bloquear = intentos >= MAX_INTENTOS
                c.execute("UPDATE usuarios SET intentos = ?, bloqueado = ? WHERE id = ?",
                          (intentos, 1 if bloquear else 0, uid))
                self._evento(c, usuario, ip, "bloqueado por 3 fallos" if bloquear else f"clave incorrecta ({intentos})")
                # mismo mensaje que para un usuario inexistente: no se dice cuántos intentos quedan
                return None, MSG_BLOQUEADO if bloquear else MSG_MAL
            token = secrets.token_urlsafe(32)
            c.execute("UPDATE usuarios SET intentos = 0, ultimo_acceso = ? WHERE id = ?", (_ahora(), uid))
            c.execute("DELETE FROM sesiones WHERE expira < ?", (time.time(),))
            c.execute("INSERT INTO sesiones (token_sha256, usuario_id, expira) VALUES (?, ?, ?)",
                      (hashlib.sha256(token.encode()).hexdigest(), uid, time.time() + VIDA_SESION_S))
            self._evento(c, usuario, ip, "inicio de sesión")
        return token, "ok"

    def usuario_de(self, token: str | None) -> str | None:
        if not token or len(token) > 100:
            return None
        with self._con() as c:
            fila = c.execute(
                "SELECT u.usuario FROM sesiones s JOIN usuarios u ON u.id = s.usuario_id "
                "WHERE s.token_sha256 = ? AND s.expira > ? AND u.bloqueado = 0",
                (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
        return fila[0] if fila else None

    def cerrar_sesion(self, token: str | None) -> None:
        if token:
            with self._con() as c:
                c.execute("DELETE FROM sesiones WHERE token_sha256 = ?", (hashlib.sha256(token.encode()).hexdigest(),))


class LimiteIntentos:
    """Ventana deslizante por IP para /api/login: frena que alguien recorra usuarios desde una IP."""

    def __init__(self, maximo: int = 10, ventana_s: float = 600):
        self.maximo, self.ventana = maximo, ventana_s
        self._d: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def permitido(self, ip: str) -> bool:
        ahora = time.time()
        with self._lock:
            v = [t for t in self._d.get(ip, []) if ahora - t < self.ventana]
            ok = len(v) < self.maximo
            if ok:
                v.append(ahora)
            self._d[ip] = v
            return ok
