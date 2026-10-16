"""Semana 12 · tablero en Markdown de la corrida en ANTON (lo publica orquestar.sh como ESTADO.md).

Solo biblioteca estándar: corre en el host, fuera del contenedor.
"""
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S = REPO / "reports" / "tuning" / "s12"
J = Path.home() / "pengwin_s12_jobs"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return f"({e})"


def ultima_epoca(log: Path):
    if not log.exists():
        return ""
    lineas = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if re.match(r"\s*época", l)]
    return lineas[-1].strip() if lineas else ""


print(f"# Semana 12 · estado en ANTON\n\nActualizado: {time.strftime('%Y-%m-%d %H:%M:%S')} (hora de ANTON)\n")
etapa = (S / "ETAPA").read_text(encoding="utf-8").strip() if (S / "ETAPA").exists() else "sin iniciar"
print(f"**Etapa:** {etapa}\n")
print("| carril | estado | trabajo | detalle |\n|---|---|---|---|")
for carril in sorted(p.name for p in J.iterdir() if p.is_dir()) if J.exists() else []:
    for est in ("running", "pending", "failed", "done"):
        for f in sorted((J / carril / est).glob("*.sh")):
            det = ultima_epoca(f.with_suffix(".sh.log")) if est == "running" else ""
            if est == "failed":
                log = f.with_suffix(".sh.log")
                det = (log.read_text(encoding="utf-8", errors="replace").strip().splitlines() or [""])[-1][:160] if log.exists() else ""
            print(f"| {carril} | {est} | `{f.stem}` | {det} |")

for n in ("etapa1", "etapa2"):
    p = S / f"{n}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        print(f"\n## {n}\n")
        if n == "etapa1":
            print(f"Referencia: media {d['media_ref']} · σ {d['sigma_ref']} · umbral {d['umbral']} · IoU ref {d['iou_ref']}\n")
            for c in d["candidatos"]:
                print(f"- `{c['candidato']}`: Dice {c.get('dice', '-')} · Δ vs media {c.get('delta_vs_media_ref', '-')} "
                      f"(z = {c.get('z', '-')}) · IoU {c.get('iou', '-')} · borde {c.get('borde')} · **{'pasa' if c['pasa'] else 'no pasa'}**")
            print(f"\nA confirmar: {d['a_confirmar'] or 'ninguno'}")
        else:
            for c in d["candidatos"]:
                print(f"- `{c['candidato']}`: Δ Dice {c.get('delta_dice', '-')} · t {c.get('t', '-')} · "
                      f"{c.get('folds_a_favor', '-')} folds · Δ IoU {c.get('delta_iou', '-')} · **{'pasa' if c['pasa'] else 'no pasa'}**")
            print(f"\nGanador: {d['ganador'] or 'ninguno'}")

print("\n## Máquina\n\n```")
print(sh("nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader"))
print(sh("free -g | head -2"))
print(sh("docker ps --format '{{.Names}} {{.Status}}' | grep pengwin-s12"))
print("```")
print("\nBitácora (últimas 15 líneas):\n\n```")
print(sh(f"tail -15 {S / 'bitacora.log'}"))
print("```")
