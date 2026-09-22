import json
from pathlib import Path

# Ruta destino solicitada
target_dir = Path(
    r"C:\Users\USER\OneDrive - Universidad Autonoma de Occidente\8 vo semestre\Analitica de datos\Proyecto\Corte 2\EDA"
)
target_dir.mkdir(parents=True, exist_ok=True)
notebook_path = target_dir / "EDA_PENGWIN_Entrada.ipynb"

# Construcción de la estructura estándar de Jupyter Notebook
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PENGWIN Dataset - Análisis Exploratorio de Datos de Entrada (EDA)\n",
            "**Proyecto Integrador Corte 2 - Analítica de Datos**  \n",
            "**Universidad Autónoma de Occidente (Semestre 2026-2)**\n",
            "\n",
            "Este notebook audita la calidad, distribución de intensidades y consistencia de etiquetas del dataset PENGWIN.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import nibabel as nib\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from tqdm import tqdm\n",
            "\n",
            'sns.set_theme(style="whitegrid", palette="muted")\n',
            'plt.rcParams["figure.figsize"] = (12, 6)\n',
            "\n",
            'DATA_DIR = Path(r"C:\\Users\\USER\\OneDrive - Universidad Autonoma de Occidente\\8 vo semestre\\Analitica de datos\\Proyecto\\Corte 2\\data")\n',
            'EDA_DIR = Path(r"C:\\Users\\USER\\OneDrive - Universidad Autonoma de Occidente\\8 vo semestre\\Analitica de datos\\Proyecto\\Corte 2\\EDA")\n',
            "EDA_DIR.mkdir(parents=True, exist_ok=True)\n",
            'print(f"Directorio de datos: {DATA_DIR}")\n',
            'print(f"Directorio EDA: {EDA_DIR}")',
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'nii_files = sorted(list(DATA_DIR.rglob("*.nii*")) + list(DATA_DIR.rglob("*.mha*")))\n',
            "images_dict = {}\n",
            "labels_dict = {}\n",
            "\n",
            "for f in nii_files:\n",
            "    fname = f.name.lower()\n",
            '    if "mask" in fname or "label" in fname or "seg" in fname:\n',
            (
                '        case_id = fname.replace("_mask",'
                ' "").replace("_label", "").replace("_seg", "").split(".")[0]\n'
            ),
            "        labels_dict[case_id] = f\n",
            "    else:\n",
            '        case_id = fname.split(".")[0]\n',
            "        images_dict[case_id] = f\n",
            "\n",
            (
                "common_cases = sorted(list(set(images_dict.keys())"
                " & set(labels_dict.keys())))\n"
            ),
            'print(f"Imágenes: {len(images_dict)} | Máscaras: {len(labels_dict)}")\n',
            'print(f"Casos emparejados: {len(common_cases)}")',
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "metadata_records = []\n",
            'print("Extrayendo metadatos NIfTI...")\n',
            "for case_id in tqdm(common_cases):\n",
            "    img_path = images_dict[case_id]\n",
            "    lbl_path = labels_dict[case_id]\n",
            "    img_nii = nib.load(str(img_path))\n",
            "    lbl_nii = nib.load(str(lbl_path))\n",
            "    \n",
            "    dim_x, dim_y, dim_z = img_nii.shape[:3]\n",
            "    zooms = img_nii.header.get_zooms()[:3]\n",
            "    dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2])\n",
            "    voxel_vol = dx * dy * dz\n",
            "    \n",
            "    img_data = img_nii.get_fdata(dtype=np.float32)\n",
            "    lbl_data = np.asanyarray(lbl_nii.dataobj).astype(np.int16)\n",
            "    \n",
            "    unique_labels = np.unique(lbl_data)\n",
            "    unique_labels = unique_labels[unique_labels > 0]\n",
            "    \n",
            "    sacrum_frags = [l for l in unique_labels if 1 <= l <= 10]\n",
            "    left_frags = [l for l in unique_labels if 11 <= l <= 20]\n",
            "    right_frags = [l for l in unique_labels if 21 <= l <= 30]\n",
            "    unknown_frags = [l for l in unique_labels if l > 30]\n",
            "    \n",
            "    slices_bone = int(np.sum(np.any(lbl_data > 0, axis=(0, 1))))\n",
            "    \n",
            "    metadata_records.append({\n",
            '        "case_id": case_id,\n',
            '        "dim_x": dim_x, "dim_y": dim_y, "dim_z": dim_z,\n',
            '        "dx_mm": dx, "dy_mm": dy, "dz_mm": dz, "voxel_vol_mm3": voxel_vol,\n',
            '        "hu_min": float(np.min(img_data)), "hu_max": float(np.max(img_data)),\n',
            '        "hu_mean": float(np.mean(img_data)), "hu_p99": float(np.percentile(img_data, 99)),\n',
            '        "total_fragments": len(unique_labels),\n',
            '        "n_sacrum_frags": len(sacrum_frags),\n',
            '        "n_left_hip_frags": len(left_frags),\n',
            '        "n_right_hip_frags": len(right_frags),\n',
            '        "unknown_labels": len(unknown_frags),\n',
            '        "slices_with_bone": slices_bone,\n',
            '        "bone_slice_ratio": slices_bone / dim_z\n',
            "    })\n",
            "\n",
            "df_eda = pd.DataFrame(metadata_records)\n",
            (
                'df_eda.to_csv(EDA_DIR / "resumen_metadatos_pengwin.csv",'
                " index=False)\n"
            ),
            "df_eda.head()",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'print("=== ESTADÍSTICAS GLOBALES ===")\n',
            (
                'display(df_eda[["dim_z", "dx_mm", "dz_mm", "hu_max",'
                ' "total_fragments"]].describe())\n'
            ),
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            (
                'sns.histplot(df_eda["dim_z"], kde=True, ax=axes[0],'
                ' color="steelblue")\n'
            ),
            'axes[0].set_title("Cortes Axiales por Volumen (Dim Z)")\n',
            "\n",
            (
                'sns.histplot(df_eda["hu_max"], kde=True, ax=axes[1],'
                ' color="crimson")\n'
            ),
            'axes[1].axvline(2500, color="black", linestyle="--", label="Metal'
            ' >2500 HU")\n',
            'axes[1].set_title("Detección de Implantes Metálicos (HU Máximo)")\n',
            "axes[1].legend()\n",
            "\n",
            (
                'sns.countplot(x=df_eda["total_fragments"], ax=axes[2],'
                ' color="darkseagreen")\n'
            ),
            'axes[2].set_title("Cantidad Total de Fragmentos por Caso")\n',
            "plt.tight_layout()\n",
            'plt.savefig(EDA_DIR / "analisis_general_eda.png", dpi=300)\n',
            "plt.show()",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Diagnóstico de Casos Limpios vs Outliers\n",
            "audit = []\n",
            "for _, r in df_eda.iterrows():\n",
            "    obs = []\n",
            "    if r['hu_max'] > 2500: obs.append('Artefacto Metálico')\n",
            (
                "    if r['unknown_labels'] > 0: obs.append('Etiquetas fuera de"
                " taxonomía')\n"
            ),
            (
                "    if r['bone_slice_ratio'] < 0.20: obs.append('Poco tejido"
                " óseo en Z')\n"
            ),
            (
                "    audit.append({'case_id': r['case_id'], 'estado': 'Revisión'"
                " if obs else 'Limpio', 'motivos': '; '.join(obs) if obs else"
                " 'OK'})\n"
            ),
            "\n",
            "df_audit = pd.DataFrame(audit)\n",
            (
                'df_eda.merge(df_audit, on="case_id").to_csv(EDA_DIR /'
                ' "auditoria_calidad.csv", index=False)\n'
            ),
            'print(df_audit["estado"].value_counts())\n',
            "df_audit[df_audit['estado'] == 'Revisión']",
        ],
    },
]

notebook_content = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2, ensure_ascii=False)

print(f"Notebook creado exitosamente en: {notebook_path}")