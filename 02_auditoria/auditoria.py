import json
from pathlib import Path


def find_root(start: Path = None) -> Path:
  """Busca hacia arriba en el árbol de carpetas hasta encontrar '01_data'."""
  if start is None:
    start = Path.cwd()
  for p in [start, *start.parents]:
    if (p / "01_data" / "PENGWIN_CT_train_labels").is_dir():
      return p
  raise FileNotFoundError(
      f"No se encontró la carpeta '01_data/PENGWIN_CT_train_labels' partiendo"
      f" de: {start}"
  )


# --- Rutas Dinámicas del Repositorio ---
ROOT = find_root()
DATA_DIR = ROOT / "01_data"
AUDIT_DIR = ROOT / "02_auditoria"  # Todo se queda en esta carpeta
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

NOTEBOOK_PATH = AUDIT_DIR / "AUDITORIA_PENGWIN.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PENGWIN Dataset - Auditoría y Análisis Exploratorio de Entrada"
            " (EDA)\n",
            "**Proyecto Integrador Corte 2 - Analítica de Datos**  \n",
            "**Universidad Autónoma de Occidente (Semestre 2026-2)**\n",
            "\n",
            (
                "Este notebook ejecuta la auditoría de calidad de datos dentro"
                " de `02_auditoria`:\n"
            ),
            (
                "- Lectura de datos desde `01_data/` mediante búsqueda dinámica"
                " (`find_root`).\n"
            ),
            (
                "- Generación y almacenamiento de reportes CSV y figuras"
                " directamente en esta carpeta."
            ),
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Instalación de librerías necesarias\n",
            "!pip install --quiet SimpleITK nibabel seaborn tqdm",
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
            "import SimpleITK as sitk\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from tqdm import tqdm\n",
            "\n",
            "# Configuración estética\n",
            'sns.set_theme(style="whitegrid", palette="muted")\n',
            'plt.rcParams["figure.figsize"] = (12, 5)\n',
            'plt.rcParams["font.size"] = 10\n',
            "\n",
            "# Localización dinámica de la raíz partiendo de 02_auditoria\n",
            "def find_root(start: Path = None) -> Path:\n",
            "    if start is None:\n",
            "        start = Path.cwd()\n",
            "    for p in [start, *start.parents]:\n",
            (
                '        if (p / "01_data" /'
                ' "PENGWIN_CT_train_labels").is_dir():\n'
            ),
            "            return p\n",
            (
                '    raise FileNotFoundError(f"No se encontró \'01_data\' desde'
                ' {start}")\n'
            ),
            "\n",
            "ROOT      = find_root()\n",
            'DATA_DIR  = ROOT / "01_data"\n',
            'AUDIT_DIR = ROOT / "02_auditoria"\n',
            "AUDIT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            'print(f"Raíz del repositorio : {ROOT}")\n',
            'print(f"Origen de datos      : {DATA_DIR}")\n',
            'print(f"Carpeta de ejecución : {AUDIT_DIR}")',
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Indexación de imágenes y etiquetas\n",
            "image_dirs = [\n",
            '    DATA_DIR / "PENGWIN_CT_train_images_part1",\n',
            '    DATA_DIR / "PENGWIN_CT_train_images_part2"\n',
            "]\n",
            'labels_dir = DATA_DIR / "PENGWIN_CT_train_labels"\n',
            "\n",
            "images_dict = {}\n",
            "labels_dict = {}\n",
            "\n",
            "# 1. Indexar imágenes\n",
            "for img_folder in image_dirs:\n",
            "    if img_folder.exists():\n",
            "        for f in img_folder.rglob(\"*.*\"):\n",
            '            if f.is_file() and f.suffix.lower() in [".mha", ".nii", ".gz"]:\n',
            '                case_id = f.name.replace(".nii.gz", "").replace(".nii", "").replace(".mha", "")\n',
            "                images_dict[case_id] = f\n",
            "\n",
            "# 2. Indexar etiquetas\n",
            "if labels_dir.exists():\n",
            "    for f in labels_dir.rglob(\"*.*\"):\n",
            '        if f.is_file() and f.suffix.lower() in [".mha", ".nii", ".gz"]:\n',
            '            case_id = f.name.replace(".nii.gz", "").replace(".nii", "").replace(".mha", "")\n',
            "            labels_dict[case_id] = f\n",
            "\n",
            "# 3. Casos coincidentes\n",
            "common_cases = sorted(list(set(images_dict.keys()).intersection(set(labels_dict.keys()))))\n",
            'print(f"Total imágenes encontradas : {len(images_dict)}")\n',
            'print(f"Total máscaras encontradas : {len(labels_dict)}")\n',
            'print(f"Casos emparejados listos   : {len(common_cases)}")',
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Función de carga SimpleITK (formato Z, Y, X)\n",
            "def load_sitk_volume(path):\n",
            "    reader = sitk.ImageFileReader()\n",
            "    reader.SetFileName(str(path))\n",
            "    image = reader.Execute()\n",
            "    data = sitk.GetArrayFromImage(image)\n",
            "    dim_x, dim_y, dim_z = image.GetSize()\n",
            "    dx, dy, dz = image.GetSpacing()\n",
            "    return data, (dim_x, dim_y, dim_z), (dx, dy, dz)\n",
            "\n",
            "print(\"Cargador SimpleITK configurado.\")",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "metadata_records = []\n",
            'print(f"Extrayendo metadatos de {len(common_cases)} volúmenes...")\n',
            "for case_id in tqdm(common_cases):\n",
            "    img_data, (dim_x, dim_y, dim_z), (dx, dy, dz) = load_sitk_volume(images_dict[case_id])\n",
            "    lbl_data, (ldim_x, ldim_y, ldim_z), _ = load_sitk_volume(labels_dict[case_id])\n",
            "    \n",
            "    lbl_data = lbl_data.astype(np.int16)\n",
            "    unique_labels = np.unique(lbl_data)\n",
            "    unique_labels = unique_labels[unique_labels > 0]\n",
            "    \n",
            "    sacrum_frags = [l for l in unique_labels if 1 <= l <= 10]\n",
            "    left_hip_frags = [l for l in unique_labels if 11 <= l <= 20]\n",
            "    right_hip_frags = [l for l in unique_labels if 21 <= l <= 30]\n",
            "    anomalous = [l for l in unique_labels if l > 30]\n",
            "    \n",
            "    slices_with_bone = int(np.sum(np.any(lbl_data > 0, axis=(1, 2))))\n",
            "    voxel_vol = dx * dy * dz\n",
            "    bone_vol_cm3 = (int(np.sum(lbl_data > 0)) * voxel_vol) / 1000.0\n",
            "    \n",
            "    metadata_records.append({\n",
            '        "case_id": case_id,\n',
            '        "dim_x": dim_x, "dim_y": dim_y, "dim_z": dim_z,\n',
            '        "dx_mm": round(dx, 4), "dy_mm": round(dy, 4), "dz_mm": round(dz, 4),\n',
            '        "voxel_vol_mm3": round(voxel_vol, 4),\n',
            '        "shape_match": (dim_x == ldim_x and dim_y == ldim_y and dim_z == ldim_z),\n',
            '        "hu_min": round(float(np.min(img_data)), 1),\n',
            '        "hu_max": round(float(np.max(img_data)), 1),\n',
            '        "hu_mean": round(float(np.mean(img_data)), 1),\n',
            '        "hu_p99": round(float(np.percentile(img_data, 99)), 1),\n',
            '        "total_fragments": len(unique_labels),\n',
            '        "n_sacrum_frags": len(sacrum_frags),\n',
            '        "n_left_hip_frags": len(left_hip_frags),\n',
            '        "n_right_hip_frags": len(right_hip_frags),\n',
            '        "n_anomalous_labels": len(anomalous),\n',
            '        "bone_volume_cm3": round(bone_vol_cm3, 2),\n',
            '        "slices_with_bone": slices_with_bone,\n',
            '        "bone_slice_ratio": round(slices_with_bone / dim_z, 4)\n',
            "    })\n",
            "\n",
            "df_eda = pd.DataFrame(metadata_records)\n",
            'output_csv = AUDIT_DIR / "resumen_metadatos_pengwin.csv"\n',
            "df_eda.to_csv(output_csv, index=False)\n",
            'print(f"Archivo guardado en: {output_csv}")\n',
            "df_eda.head()",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Distribución espacial guardada en 02_auditoria\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            (
                'sns.histplot(df_eda["dim_z"], kde=True, ax=axes[0],'
                ' color="steelblue", bins=15)\n'
            ),
            'axes[0].set_title("Número de Cortes Axiales (Dim Z)")\n',
            'axes[0].set_xlabel("Cantidad de Cortes")\n',
            "\n",
            (
                'sns.scatterplot(data=df_eda, x="dx_mm", y="dz_mm",'
                ' hue="dim_z", palette="viridis", ax=axes[1])\n'
            ),
            'axes[1].set_title("Anisotropía: Espaciado en Plano vs Axial")\n',
            'axes[1].set_xlabel("dx / dy (mm)")\n',
            'axes[1].set_ylabel("dz (mm - Espesor de Corte)")\n',
            "\n",
            (
                'sns.boxplot(y=df_eda["bone_slice_ratio"], ax=axes[2],'
                ' color="lightgreen")\n'
            ),
            (
                'axes[2].set_title("Proporción de Cortes Útiles (con'
                ' Hueso)")\n'
            ),
            'axes[2].set_ylabel("Cortes con hueso / Dim Z")\n',
            "\n",
            "plt.tight_layout()\n",
            'plt.savefig(AUDIT_DIR / "distribucion_espacial.png", dpi=300)\n',
            "plt.show()",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Distribución de HU y detección de metal guardada en 02_auditoria\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
            (
                'sns.histplot(df_eda["hu_max"], kde=True, ax=axes[0],'
                ' color="crimson", bins=20)\n'
            ),
            (
                'axes[0].axvline(2500, color="black", linestyle="--",'
                ' label="Umbral Metal (>2500 HU)")\n'
            ),
            'axes[0].set_title("Valores Máximos de HU (Detección de Metal)")\n',
            'axes[0].set_xlabel("HU Máximo")\n',
            "axes[0].legend()\n",
            "\n",
            (
                'sns.boxplot(data=df_eda[["hu_min", "hu_mean", "hu_p99"]],'
                ' ax=axes[1], palette="Set2")\n'
            ),
            'axes[1].set_title("Distribución de Percentiles HU")\n',
            'axes[1].set_ylabel("Unidades Hounsfield (HU)")\n',
            "\n",
            "plt.tight_layout()\n",
            'plt.savefig(AUDIT_DIR / "distribucion_hu.png", dpi=300)\n',
            "plt.show()",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Distribución de fragmentos guardada en 02_auditoria\n",
            "df_eda['sacro_fracturado'] = df_eda['n_sacrum_frags'] > 1\n",
            "df_eda['coxal_izq_fracturado'] = df_eda['n_left_hip_frags'] > 1\n",
            "df_eda['coxal_der_fracturado'] = df_eda['n_right_hip_frags'] > 1\n",
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)\n",
            (
                'sns.countplot(data=df_eda, x="n_sacrum_frags", ax=axes[0],'
                ' color="teal")\n'
            ),
            'axes[0].set_title("Fragmentos en Sacro (1 = Sano)")\n',
            'axes[0].set_xlabel("N° Fragmentos")\n',
            "\n",
            (
                'sns.countplot(data=df_eda, x="n_left_hip_frags", ax=axes[1],'
                ' color="coral")\n'
            ),
            'axes[1].set_title("Fragmentos en Coxal Izquierdo")\n',
            'axes[1].set_xlabel("N° Fragmentos")\n',
            "\n",
            (
                'sns.countplot(data=df_eda, x="n_right_hip_frags", ax=axes[2],'
                ' color="mediumpurple")\n'
            ),
            'axes[2].set_title("Fragmentos en Coxal Derecho")\n',
            'axes[2].set_xlabel("N° Fragmentos")\n',
            "\n",
            "plt.tight_layout()\n",
            'plt.savefig(AUDIT_DIR / "distribucion_fragmentos.png", dpi=300)\n',
            "plt.show()",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Consolidación de auditoría final guardada en 02_auditoria\n",
            "auditoria = []\n",
            "for _, r in df_eda.iterrows():\n",
            "    alertas = []\n",
            "    if not r['shape_match']:\n",
            "        alertas.append('Dimensiones incompatibles')\n",
            "    if r['n_anomalous_labels'] > 0:\n",
            (
                "        alertas.append(f'Etiquetas fuera de taxonomía"
                " ({r[\"n_anomalous_labels\"]})')\n"
            ),
            "    if r['hu_max'] > 2500:\n",
            (
                "        alertas.append(f'Artefacto metálico (HU máx"
                " {r[\"hu_max\"]:.0f})')\n"
            ),
            "    if r['bone_slice_ratio'] < 0.25:\n",
            (
                "        alertas.append(f'Baja cobertura ósea axial"
                " ({r[\"bone_slice_ratio\"]*100:.1f}%)')\n"
            ),
            "    if r['total_fragments'] == 0:\n",
            "        alertas.append('Máscara vacía')\n",
            "        \n",
            (
                "    estado = 'Atípico / Desafiante' if len(alertas) > 0 else"
                " 'Limpio / Confiable'\n"
            ),
            "    auditoria.append({\n",
            "        'case_id': r['case_id'],\n",
            "        'estado': estado,\n",
            "        'n_alertas': len(alertas),\n",
            (
                "        'observaciones': '; '.join(alertas) if alertas else"
                " 'Conforme'\n"
            ),
            "    })\n",
            "\n",
            "df_auditoria = pd.DataFrame(auditoria)\n",
            "df_reporte = df_eda.merge(df_auditoria, on='case_id')\n",
            'audit_csv = AUDIT_DIR / "auditoria_calidad_casos.csv"\n',
            "df_reporte.to_csv(audit_csv, index=False)\n",
            "\n",
            'print(f"Auditoría guardada exitosamente en: {audit_csv}")\n',
            'print(df_reporte["estado"].value_counts())',
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

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
  json.dump(notebook_content, f, indent=2, ensure_ascii=False)

print(
    f"Notebook generado exitosamente en 02_auditoria:\n-> Arquitectura:"
    f" {NOTEBOOK_PATH}"
)