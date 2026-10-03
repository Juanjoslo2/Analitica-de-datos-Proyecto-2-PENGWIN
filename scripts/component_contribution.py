"""component_contribution.py.

¿Qué aporta cada parte de la arquitectura? Tres medidas sobre un modelo YA entrenado:

1. γ aprendidos (``PengwinNet.gammas``): cuánto decidió usar la red cada componente.
2. Cambio que produce cada componente en las features, en val:
       cambio relativo   ‖y − x‖ / ‖x‖
       cambio de forma   1 − cos(x, y)   (no cuenta un simple reescalado; el CBAM sin γ
                                          reduce todo a la mitad y eso infla la medida 1)
3. Knockout: se apaga UN componente en inferencia y se mide cuánto caen las métricas.
   Es la medida más directa de "qué aporta", sin reentrenar:
       residual bN       el bloque residual se salta (identidad)
       CBAM bN           la atención se salta
       contexto C4       P3 se calcula sin el aporte de C4 (solo C3)
       contexto 2.5D     los cortes vecinos se reemplazan por el corte central

Uso:
    python scripts/component_contribution.py --ckpt checkpoints/base/best.pth --cache-dir D:/PENGWIN/data_processed
Salida: reports/contribucion/<modelo>.json + tabla por consola.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.dataset import PengwinSlices, read_split  # noqa: E402
from pengwin.losses.losses import MultiTaskLoss  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402
from pengwin.training.engine import evaluate  # noqa: E402

METRICS = ["mAP@0.50", "mAP@[.50:.95]", "dice_hueso", "dice_SA", "cls_f1_macro"]


@contextlib.contextmanager
def swap(module: nn.Module, name: str, new: nn.Module):
    old = getattr(module, name)
    setattr(module, name, new)
    try:
        yield
    finally:
        setattr(module, name, old)


@contextlib.contextmanager
def no_c4_context(model):
    neck = model.neck
    old = neck.forward
    neck.forward = lambda c3, c4: neck.smooth(neck.lat3(c3))
    try:
        yield
    finally:
        neck.forward = old


class CentreOnly(torch.utils.data.Dataset):
    """Envuelve el dataset y copia el canal central en los 3 canales (sin contexto 2.5D)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = dict(self.ds[i])
        item["image"] = item["image"][1:2].repeat(3, 1, 1)
        return item


def feature_change(model, batch):
    stats = {}

    def hook(name):
        def f(_, inp, out):
            x, y = inp[0].float().flatten(1), out.float().flatten(1)
            rel = ((y - x).norm(dim=1) / x.norm(dim=1).clamp(min=1e-6)).mean()
            cos = torch.nn.functional.cosine_similarity(x, y, dim=1).mean()
            stats[name] = {"cambio_relativo": float(rel), "cambio_forma": float(1 - cos)}
        return f

    hs = []
    for k, st in enumerate(model.backbone.stages, start=1):
        if not isinstance(st.res, nn.Identity):
            hs.append(st.res.register_forward_hook(hook(f"residual b{k}")))
        if not isinstance(st.cbam, nn.Identity):
            hs.append(st.cbam.register_forward_hook(hook(f"CBAM b{k}")))
    with torch.no_grad():
        model(batch)
    for h in hs:
        h.remove()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--n-cases", type=int, default=6, help="casos de val usados (los knockouts son inferencias completas)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = PengwinNet(cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])
    loss_fn = MultiTaskLoss(cfg).to(device)
    ds = PengwinSlices(args.cache_dir, read_split(REPO / cfg["data"]["splits"], "val")[:args.n_cases], cfg, train=False)

    def run(dataset):
        dl = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
        m = evaluate(model, dl, loss_fn, device, cfg)
        return {k: m[k] for k in METRICS}

    ref = run(ds)
    rows = {"modelo completo": ref}
    for k, st in enumerate(model.backbone.stages, start=1):
        if not isinstance(st.res, nn.Identity):
            with swap(st, "res", nn.Identity()):
                rows[f"sin residual b{k}"] = run(ds)
        if not isinstance(st.cbam, nn.Identity):
            with swap(st, "cbam", nn.Identity()):
                rows[f"sin CBAM b{k}"] = run(ds)
    with no_c4_context(model):
        rows["sin contexto C4 en P3"] = run(ds)
    rows["sin contexto 2.5D"] = run(CentreOnly(ds))

    idx = np.linspace(0, len(ds) - 1, 64).astype(int)
    batch = torch.stack([ds[i]["image"] for i in idx]).to(device)
    change = feature_change(model, batch)

    table = pd.DataFrame(rows).T
    delta = (table - table.loc["modelo completo"]).drop(index="modelo completo")
    name = args.ckpt.parent.name
    out = {"checkpoint": str(args.ckpt), "casos_val": ds.meta and list(ds.meta), "n_cortes": len(ds),
           "gammas": model.gammas(), "cambio_features": change,
           "metricas": table.to_dict(orient="index"), "caida_por_knockout": delta.to_dict(orient="index")}
    out_dir = REPO / "reports" / "contribucion"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"\n{name}: {len(ds)} cortes de val")
    print("\nγ aprendidos:", {k: round(v, 3) for k, v in model.gammas().items()})
    print("\nCambio en las features:")
    print(pd.DataFrame(change).T.round(3).to_string())
    print("\nCaída de cada métrica al apagar el componente (negativo = el componente ayudaba):")
    print(delta.round(4).to_string())


if __name__ == "__main__":
    main()
