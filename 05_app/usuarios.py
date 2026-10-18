"""usuarios.py.

Administración de usuarios del dashboard (se corre en el servidor, nunca desde la web).

    python usuarios.py crear <usuario>            pide la clave dos veces (o --generar)
    python usuarios.py clave <usuario>            cambia la clave y cierra sus sesiones
    python usuarios.py desbloquear <usuario>      tras 3 claves equivocadas
    python usuarios.py bloquear <usuario>
    python usuarios.py eliminar <usuario>
    python usuarios.py listar
    python usuarios.py eventos [n]                últimos ingresos, fallos y bloqueos

En el servidor del dashboard, dentro del contenedor:
    docker exec -it pengwin-demo python usuarios.py desbloquear profesor
"""

import argparse
import getpass
import os
import secrets
import string
from pathlib import Path

from seguridad import Usuarios

AQUI = Path(__file__).resolve().parent


def pedir_clave(generar: bool) -> str:
    if generar:
        alfabeto = string.ascii_letters + string.digits
        clave = "".join(secrets.choice(alfabeto) for _ in range(16))
        print("clave generada:", clave)
        return clave
    a = getpass.getpass("clave nueva: ")
    if a != getpass.getpass("repítala: "):
        raise SystemExit("las claves no coinciden")
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["crear", "clave", "desbloquear", "bloquear", "eliminar", "listar", "eventos"])
    ap.add_argument("usuario", nargs="?")
    ap.add_argument("--generar", action="store_true", help="genera una clave aleatoria de 16 caracteres")
    ap.add_argument("--db", default=os.environ.get("PENGWIN_USUARIOS", AQUI / "datos" / "usuarios.db"))
    a = ap.parse_args()
    db = Usuarios(a.db)
    if a.accion == "listar":
        for u, intentos, bloq, creado, ultimo in db.listar():
            print(f"{u:20s} {'BLOQUEADO' if bloq else 'activo':10s} fallos {intentos}  creado {creado}  último {ultimo or '-'}")
        return
    if a.accion == "eventos":
        for fila in db.eventos(int(a.usuario or 30)):
            print(*fila, sep="  ")
        return
    if not a.usuario:
        raise SystemExit("falta el usuario")
    if a.accion == "crear":
        db.crear(a.usuario, pedir_clave(a.generar))
        print("creado:", a.usuario)
    elif a.accion == "clave":
        print("ok" if db.cambiar_clave(a.usuario, pedir_clave(a.generar)) else "no existe")
    elif a.accion in ("desbloquear", "bloquear"):
        print("ok" if db.fijar_bloqueo(a.usuario, a.accion == "bloquear") else "no existe")
    elif a.accion == "eliminar":
        print("ok" if db.eliminar(a.usuario) else "no existe")


if __name__ == "__main__":
    main()
