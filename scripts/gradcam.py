"""gradcam.py.

Dónde mira la red, con y sin atención (pedido del profesor en el avance del 2 de octubre).

    Grad-CAM (Selvaraju et al., 2017) sobre la salida del bloque 4 del backbone (C4, stride 16,
    después del CBAM): para la presencia de un hueso c,
        α_k = media espacial de ∂y_c / ∂A_k,     CAM = ReLU(Σ_k α_k A_k)   (reescalado a 256 px)
    Se compara el modelo de la semana 9 CON CBAM (`base`) contra el entrenado SIN CBAM
    (`sin_cbam`), que comparten todo lo demás, y el modelo final (`v2_last`).
    Además se guarda el mapa de atención ESPACIAL del CBAM (M_s) de los bloques 3 y 4 de v2_last.

Uso:
    python scripts/gradcam.py --cache-dir D:/PENGWIN/data_processed --cases 075 069 019
Salida: reports/figures/semana11/gradcam_<caso>.png y reports/eval/gradcam.json (resumen numérico)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.dataset import context_stack  # noqa: E402
from pengwin.data.slice_cache import load_case_cache  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402

MODELOS = {"base": "con CBAM (semana 9)", "sin_cbam": "sin CBAM (semana 9)", "v2_last": "final (v2, γ en CBAM)"}
REG = {0: "SA", 1: "LI", 2: "RI"}


def cargar(nombre, dev):
    ck = torch.load(REPO / "checkpoints" / nombre / "best.pth", map_location="cpu", weights_only=False)
    m = PengwinNet(ck["cfg"]["model"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m


def gradcam(model, x, clase):
    feats = {}
    h = model.backbone.stages[3].register_forward_hook(lambda mod, i, o: feats.__setitem__("A", o))
    x = x.clone().requires_grad_(False)
    out = model(x)
    h.remove()
    A = feats["A"]
    g, = torch.autograd.grad(out["cls_logits"][0, clase], A)
    cam = F.relu((g.mean((2, 3), keepdim=True) * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), float(torch.sigmoid(out["cls_logits"][0, clase]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", default=["075", "069", "019"])
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelos = {k: cargar(k, dev) for k in MODELOS}
    fig_dir = REPO / "reports" / "figures" / "semana11"
    fig_dir.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for cid in args.cases:
        image, label, meta = load_case_cache(args.cache_dir / cid)
        lab = np.asarray(label)
        reg = np.where(lab == 0, 0, (lab.astype(np.int64) - 1) // 10 + 1)
        # corte con los tres huesos y la mayor área de sacro: el caso que más exige separar regiones
        cand = [z for z in range(reg.shape[0]) if all((reg[z] == c).any() for c in (1, 2, 3))]
        z = max(cand, key=lambda k: (reg[k] == 1).sum()) if cand else int(np.argmax((reg > 0).sum((1, 2))))
        x = torch.from_numpy(context_stack(np.asarray(image), z, meta["context_offset"]).astype(np.float32) / 255.0)[None].to(dev)
        hueso = reg[z] > 0
        fig, axes = plt.subplots(3, 5, figsize=(16, 9.6))
        resumen[cid] = {"z": int(z)}
        for i, c in enumerate((0, 1, 2)):
            axes[i, 0].imshow(np.asarray(image[z]), cmap="gray")
            axes[i, 0].contour(reg[z] == c + 1, levels=[0.5], colors="yellow", linewidths=0.8)
            axes[i, 0].set_ylabel(f"presencia de {REG[c]}", fontsize=10)
            axes[i, 0].set_title("corte y hueso real" if i == 0 else "", fontsize=9)
            for j, (k, titulo) in enumerate(MODELOS.items(), start=1):
                cam, prob = gradcam(modelos[k], x, c)
                # fracción de la "masa" del Grad-CAM que cae sobre el hueso correcto y sobre cualquier hueso
                en_hueso = float(cam[reg[z] == c + 1].sum() / (cam.sum() + 1e-8))
                en_pelvis = float(cam[hueso].sum() / (cam.sum() + 1e-8))
                resumen[cid].setdefault(k, {})[REG[c]] = {"prob": prob, "masa_en_su_hueso": en_hueso, "masa_en_pelvis": en_pelvis}
                axes[i, j].imshow(np.asarray(image[z]), cmap="gray")
                axes[i, j].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[i, j].set_title(f"{titulo}\nP = {prob:.2f} · en su hueso {100 * en_hueso:.0f} %" if i == 0 else
                                     f"P = {prob:.2f} · en su hueso {100 * en_hueso:.0f} %", fontsize=8.5)
        # atención espacial del CBAM de v2 (M_s), igual para las tres filas
        m = modelos["v2_last"]
        with torch.no_grad():
            m(x)
        for i, b in enumerate((2, 3)):
            ms = m.backbone.stages[b].cbam.last_spatial_map
            ms = F.interpolate(ms.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            gam = float(m.backbone.stages[b].cbam.gamma.detach())
            axes[i, 4].imshow(np.asarray(image[z]), cmap="gray")
            axes[i, 4].imshow(ms, cmap="magma", alpha=0.55)
            axes[i, 4].set_title(f"v2: atención espacial CBAM b{b + 1}\n(γ = {gam:+.2f})", fontsize=8.5)
            resumen[cid][f"Ms_b{b + 1}_hueso_vs_fondo"] = float(ms[hueso].mean() / (ms[~hueso].mean() + 1e-8))
        axes[2, 4].axis("off")
        for a in axes.flat:
            a.set_xticks([])
            a.set_yticks([])
        fig.suptitle(f"Paciente {cid}, corte {z}: Grad-CAM de la cabeza de clasificación sobre C4", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"gradcam_{cid}.png", dpi=85, bbox_inches="tight")
        plt.close(fig)
        print(cid, z, flush=True)
    (REPO / "reports" / "eval" / "gradcam.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
