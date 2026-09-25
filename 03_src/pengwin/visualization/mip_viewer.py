"""mip_viewer.py.

Visualizador 1 obligatorio (MIP crudo, sin modelo).
Proyecciones de Máxima Intensidad (MIP) del volumen filtrado por umbral óseo en HU:
    - 3 vistas ortogonales (axial, coronal, sagital) con ejes en milímetros.
    - MIP rotacional alrededor del eje cráneo-caudal (sensación 3D para el dashboard).

Cambios respecto a la primera versión (ver IA_USAGE.md):
    1. Se elimina la camilla del CT y objetos externos con la silueta del paciente
       (``body_mask_2d``), calculada solo con la imagen.
    2. Ventana de visualización FIJA [umbral, 1800 HU]: con autoescala, los casos con
       metal (hasta ~47 000 HU) dejaban el hueso casi negro.
    3. Ejes en mm usando el spacing del header (el enunciado prohíbe píxeles como medida).
    4. Las funciones devuelven la figura/arreglos y no llaman a ``plt.show()``,
       para reutilizarlas en Streamlit (``st.pyplot(fig)``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

if __package__ in (None, ""):  # ejecutado como script: python 03_src/pengwin/visualization/mip_viewer.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pengwin.data.preprocessing import HU_BONE_MIP, body_mask_2d  # noqa: E402

DISPLAY_MAX_HU = 1800.0   # hueso cortical denso ~1200-1800 HU; por encima es metal


def filter_bone(
    volume_hu: np.ndarray,
    bone_threshold_hu: float = HU_BONE_MIP,
    remove_table: bool = True,
) -> np.ndarray:
    """Devuelve un volumen float32 donde todo lo que no es hueso vale ``bone_threshold_hu``."""
    filtered = np.maximum(volume_hu, bone_threshold_hu).astype(np.float32, copy=False)
    if remove_table:
        body = body_mask_2d(volume_hu)
        filtered = np.where(body[None], filtered, bone_threshold_hu).astype(np.float32, copy=False)
    return filtered


def compute_bone_mip(
    volume_hu: np.ndarray,
    bone_threshold_hu: float = HU_BONE_MIP,
    remove_table: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MIP en los tres ejes de un volumen (Z, Y, X) en LPS.

    Devuelve (axial (Y,X), coronal (Z,X), sagital (Z,Y)); coronal y sagital ya
    volteados para que superior quede arriba.
    """
    vol = filter_bone(volume_hu, bone_threshold_hu, remove_table)
    return vol.max(axis=0), np.flipud(vol.max(axis=1)), np.flipud(vol.max(axis=2))


def rotating_mip(
    volume_hu: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    angles_deg: List[float] | None = None,
    bone_threshold_hu: float = HU_BONE_MIP,
    target_mm: float = 2.0,
) -> Tuple[List[np.ndarray], Tuple[float, float]]:
    """MIP rotacional alrededor del eje Z (vista coronal girando 360°).

    El volumen se submuestrea a ~``target_mm`` por vóxel para que cada ángulo tarde
    < 0,5 s en CPU. Devuelve la lista de imágenes (Z, ancho) y el tamaño de píxel
    (mm_z, mm_horizontal) para dibujarlas con la proporción física correcta.
    """
    angles_deg = angles_deg if angles_deg is not None else list(range(0, 360, 15))
    dz, dy, dx = spacing_zyx
    vol = filter_bone(volume_hu, bone_threshold_hu)
    fz = max(1, int(round(target_mm / dz)))
    fxy = max(1, int(round(target_mm / min(dy, dx))))
    vol = vol[::fz, ::fxy, ::fxy]
    frames = []
    for ang in angles_deg:
        rot = ndi.rotate(vol, ang, axes=(1, 2), reshape=False, order=1, cval=bone_threshold_hu)
        frames.append(np.flipud(rot.max(axis=1)))
    return frames, (dz * fz, dx * fxy)


def plot_orthogonal_mip(
    volume_hu: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    case_id: str,
    bone_threshold_hu: float = HU_BONE_MIP,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura con las 3 vistas MIP; ejes en mm y proporción física correcta."""
    mip_ax, mip_cor, mip_sag = compute_bone_mip(volume_hu, bone_threshold_hu)
    dz, dy, dx = spacing_zyx
    Z, H, W = volume_hu.shape
    style = dict(cmap="bone", vmin=bone_threshold_hu, vmax=DISPLAY_MAX_HU, interpolation="antialiased")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor="black")
    fig.suptitle(
        f"Visualizador 1 · MIP del volumen crudo · Caso {case_id}\n"
        f"Umbral óseo ≥ {bone_threshold_hu:.0f} HU · sin modelo · camilla removida",
        color="white", fontsize=13,
    )
    # extent = (izq, der, abajo, arriba) en mm -> matplotlib respeta la proporción física
    panels = [
        (mip_ax, (0, W * dx, H * dy, 0), "Axial (vista desde los pies)",
         "x (mm)  derecha del paciente → izquierda", "y (mm)  anterior → posterior"),
        (mip_cor, (0, W * dx, 0, Z * dz), "Coronal (vista frontal)",
         "x (mm)  derecha del paciente → izquierda", "z (mm)  inferior → superior"),
        (mip_sag, (0, H * dy, 0, Z * dz), "Sagital (vista lateral)",
         "y (mm)  anterior → posterior", "z (mm)  inferior → superior"),
    ]
    for ax, (img, extent, title, xl, yl) in zip(axes, panels):
        ax.imshow(img, extent=extent, aspect="equal", **style)
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel(xl, color="#c3c2b7", fontsize=9)
        ax.set_ylabel(yl, color="#c3c2b7", fontsize=9)
        ax.tick_params(colors="#898781", labelsize=8)
        ax.grid(False)  # sin cuadrícula aunque el estilo global de matplotlib la active
        for sp in ax.spines.values():
            sp.set_color("#383835")
    fig.text(0.5, 0.005,
             "Uso académico. No es un dispositivo médico ni apoya decisiones quirúrgicas reales.",
             ha="center", color="#898781", fontsize=8)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    from pengwin.data.data_loader import get_dataset_pairs, load_and_standardize_mha

    case_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pairs = get_dataset_pairs(repo_root / "01_data")
    pilot = next((p for p in pairs if p["case_id"] == case_arg), pairs[0])
    print(f"Generando MIP para el caso: {pilot['case_id']}")
    raw_vol, spacing, _ = load_and_standardize_mha(pilot["image_path"], is_label=False)
    out = repo_root / "reports" / "figures" / "mip_viewer1" / f"mip_visualizer1_{pilot['case_id']}.png"
    plot_orthogonal_mip(raw_vol, spacing, pilot["case_id"], save_path=out)
    print(f"MIP guardado en: {out}")
    plt.show()
