import json
from pathlib import Path

target_dir = Path(
    r"C:\Users\USER\OneDrive - Universidad Autonoma de Occidente\8 vo semestre\Analitica de datos\Proyecto\Corte 2\EDA"
)
target_dir.mkdir(parents=True, exist_ok=True)
notebook_path = target_dir / "EDA_PENGWIN_Entrada.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PENGWIN Dataset - Análisis Exploratorio de Datos de Entrada (EDA)\n",
            "**Proyecto Integrador Corte 2 - Analítica de Datos**  \n",
            "**Universidad Autónoma de Occidente (Semestre 2026-2)**\n",
            "\n",
            "Este notebook realiza la auditoría de calidad y extracción de metadatos de los volúmenes CT del dataset **PENGWIN**:\n",
            "1. Indexación robusta por carpetas (`PENGWIN_CT_train_images_part1`, `part2` y `PENGWIN_CT_train_labels`).\n",
            "2. Lectura universal con `SimpleITK` para compatibilidad nativa con `.mha` y `.nii.gz`.\n",
            "3. Extracción de metadatos: Dimensiones, Voxel Spacing físico ($dx, dy, dz$) y anisotropía.\n",
            "4. Análisis de intensidades Hounsfield (HU) y detección de artefactos metálicos (>2500 HU).\n",
            "5. Verificación de la taxonomía oficial (Sacro: 1-10, Coxal Izquierdo: 11-20, Coxal Derecho: 21-30).\n",
            "6. Auditoría de fragmentos sanos vs fracturados y reporte de calidad de datos limpios vs atípicos.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Instalación de librerías médicas necesarias si no están presentes\n",
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
            "import glob\n",
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
            "# Rutas exactas del proyecto\n",
            'BASE_DIR = Path(r"C:\\Users\\USER\\OneDrive - Universidad Autonoma de Occidente\\8 vo semestre\\Analitica de datos\\Proyecto\\Corte 2")\n',
            'DATA_DIR = BASE_DIR / "data"\n',
            'EDA_DIR = BASE_DIR / "EDA"\n',
            "EDA_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
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
            "# Indexación explícita según el árbol de carpetas\n",
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
            "# 3. Casos emparejados\n",
            "common_cases = sorted(list(set(images_dict.keys()).intersection(set(labels_dict.keys()))))\n",
            "\n",
            'print(f"Total Imágenes encontradas: {len(images_dict)}")\n',
            'print(f"Total Máscaras encontradas: {len(labels_dict)}")\n',
            'print(f"Casos perfectamente emparejados: {len(common_cases)}")\n',
            "\n",
            "if len(common_cases) > 0:\n",
            '    print(f"Ejemplo de ID emparejado: {common_cases[0]}")\n',
            '    print(f" -> Imagen: {images_dict[common_cases[0]].name}")\n',
            '    print(f" -> Máscara: {labels_dict[common_cases[0]].name}")\n',
            "else:\n",
            '    print("[ALERTA] Revise los nombres de archivos dentro de las carpetas.")',
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Función auxiliar para cargar volúmenes con SimpleITK\n",
            "def load_sitk_volume(path):\n",
            '    """Lee archivos .mha o .nii y entrega (array_zyx, dims_xyz, spacing_xyz)"""\n',
            "    reader = sitk.ImageFileReader()\n",
            "    reader.SetFileName(str(path))\n",
            "    image = reader.Execute()\n",
            "    \n",
            "    # SimpleITK GetArrayFromImage retorna dimensión en orden (Z, Y, X)\n",
            "    data = sitk.GetArrayFromImage(image)\n",
            "    dim_x, dim_y, dim_z = image.GetSize()      # Tamaño en píxeles (X, Y, Z)\n",
            "    dx, dy, dz = image.GetSpacing()            # Espaciado físico mm (dx, dy, dz)\n",
            "    return data, (dim_x, dim_y, dim_z), (dx, dy, dz)\n",
            "\n",
            "print(\"Función de carga inicializada correctamente.\")",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "metadata_records = []\n",
            "\n",
            'print(f"Extrayendo metadatos y estadísticas de {len(common_cases)} volúmenes...")\n',
            "for case_id in tqdm(common_cases):\n",
            "    img_path = images_dict[case_id]\n",
            "    lbl_path = labels_dict[case_id]\n",
            "    \n",
            "    img_data, (dim_x, dim_y, dim_z), (dx, dy, dz) = load_sitk_volume(img_path)\n",
            "    lbl_data, (ldim_x, ldim_y, ldim_z), _ = load_sitk_volume(lbl_path)\n",
            "    \n",
            "    # Validación de dimensiones consistentes\n",
            "    shape_match = (dim_x == ldim_x and dim_y == ldim_y and dim_z == ldim_z)\n",
            "    \n",
            "    # Estadísticas HU en el volumen\n",
            "    hu_min = float(np.min(img_data))\n",
            "    hu_max = float(np.max(img_data))\n",
            "    hu_mean = float(np.mean(img_data))\n",
            "    hu_p99 = float(np.percentile(img_data, 99))\n",
            "    \n",
            "    # Análisis de etiquetas conforme a la taxonomía oficial PENGWIN\n",
            "    lbl_data = lbl_data.astype(np.int16)\n",
            "    unique_labels = np.unique(lbl_data)\n",
            "    unique_labels = unique_labels[unique_labels > 0]\n",
            "    \n",
            "    sacrum_frags = [l for l in unique_labels if 1 <= l <= 10]\n",
            "    left_hip_frags = [l for l in unique_labels if 11 <= l <= 20]\n",
            "    right_hip_frags = [l for l in unique_labels if 21 <= l <= 30]\n",
            "    anomalous_labels = [l for l in unique_labels if l > 30]\n",
            "    \n",
            "    # Cortes en Z con presencia de tejido óseo\n",
            "    # En shape (Z, Y, X), el eje 0 es Z\n",
            "    slices_with_bone = int(np.sum(np.any(lbl_data > 0, axis=(1, 2))))\n",
            "    \n",
            "    voxel_vol = dx * dy * dz\n",
            "    bone_voxels = int(np.sum(lbl_data > 0))\n",
            "    bone_vol_cm3 = (bone_voxels * voxel_vol) / 1000.0\n",
            "    \n",
            "    metadata_records.append({\n",
            '        "case_id": case_id,\n',
            '        "dim_x": dim_x,\n',
            '        "dim_y": dim_y,\n',
            '        "dim_z": dim_z,\n',
            '        "dx_mm": round(dx, 4),\n',
            '        "dy_mm": round(dy, 4),\n',
            '        "dz_mm": round(dz, 4),\n',
            '        "voxel_vol_mm3": round(voxel_vol, 4),\n',
            '        "shape_match": shape_match,\n',
            '        "hu_min": round(hu_min, 1),\n',
            '        "hu_max": round(hu_max, 1),\n',
            '        "hu_mean": round(hu_mean, 1),\n',
            '        "hu_p99": round(hu_p99, 1),\n',
            '        "total_fragments": len(unique_labels),\n',
            '        "n_sacrum_frags": len(sacrum_frags),\n',
            '        "n_left_hip_frags": len(left_hip_frags),\n',
            '        "n_right_hip_frags": len(right_hip_frags),\n',
            '        "n_anomalous_labels": len(anomalous_labels),\n',
            '        "bone_volume_cm3": round(bone_vol_cm3, 2),\n',
            '        "slices_with_bone": slices_with_bone,\n',
            '        "bone_slice_ratio": round(slices_with_bone / dim_z, 4)\n',
            "    })\n",
            "\n",
            "df_eda = pd.DataFrame(metadata_records)\n",
            'df_eda.to_csv(EDA_DIR / "resumen_metadatos_pengwin.csv", index=False)\n',
            'print(f"Extracción completada: {len(df_eda)} registros procesados y guardados en resumen_metadatos_pengwin.csv.")\n',
            "df_eda.head()",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'print("=== ESTADÍSTICAS GLOBALES DEL DATASET ===")\n',
            (
                'display(df_eda[["dim_z", "dx_mm", "dz_mm", "hu_max",'
                ' "total_fragments", "bone_volume_cm3",'
                ' "bone_slice_ratio"]].describe())\n'
            ),
            "\n",
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
            'plt.savefig(EDA_DIR / "distribucion_espacial.png", dpi=300)\n',
            "plt.show()",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'print("=== AUDITORÍA DE INTENSIDADES (HU) Y ARTEFACTOS METÁLICOS ===")\n',
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
            (
                'sns.histplot(df_eda["hu_max"], kde=True, ax=axes[0],'
                ' color="crimson", bins=20)\n'
            ),
            (
                'axes[0].axvline(2500, color="black", linestyle="--",'
                ' label="Umbral Típico de Metal (>2500 HU)")\n'
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
            'plt.savefig(EDA_DIR / "distribucion_hu.png", dpi=300)\n',
            "plt.show()\n",
            "\n",
            "casos_metal = df_eda[df_eda['hu_max'] > 2500]\n",
            (
                'print(f"Casos con presencia de material quirúrgico / implante'
                ' (>2500 HU): {len(casos_metal)}")\n'
            ),
            "if len(casos_metal) > 0:\n",
            (
                "    display(casos_metal[['case_id', 'hu_max', 'hu_p99',"
                " 'bone_volume_cm3']])"
            ),
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'print("=== ANÁLISIS DE FRACTURAS SEGÚN LA TAXONOMÍA PENGWIN ===")\n',
            "# 1 fragmento = Hueso sano / no fracturado\n",
            "# >1 fragmentos = Hueso fracturado\n",
            "df_eda['sacro_fracturado'] = df_eda['n_sacrum_frags'] > 1\n",
            "df_eda['coxal_izq_fracturado'] = df_eda['n_left_hip_frags'] > 1\n",
            "df_eda['coxal_der_fracturado'] = df_eda['n_right_hip_frags'] > 1\n",
            "\n",
            (
                'print(f"Sacro con fractura:'
                " {df_eda['sacro_fracturado'].sum()} / {len(df_eda)}"
                " ({df_eda['sacro_fracturado'].mean()*100:.1f}%)\")\n"
            ),
            (
                'print(f"Coxal Izquierdo con fractura:'
                " {df_eda['coxal_izq_fracturado'].sum()} / {len(df_eda)}"
                " ({df_eda['coxal_izq_fracturado'].mean()*100:.1f}%)\")\n"
            ),
            (
                'print(f"Coxal Derecho con fractura:'
                " {df_eda['coxal_der_fracturado'].sum()} / {len(df_eda)}"
                " ({df_eda['coxal_der_fracturado'].mean()*100:.1f}%)\")\n"
            ),
            "\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)\n",
            (
                'sns.countplot(data=df_eda, x="n_sacrum_frags", ax=axes[0],'
                ' color="teal")\n'
            ),
            'axes[0].set_title("Fragmentos en Sacro (1 = Sano)")\n',
            'axes[0].set_xlabel("N° Fragmentos")\n',
            'axes[0].set_ylabel("Cantidad de Pacientes")\n',
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
            'plt.savefig(EDA_DIR / "distribucion_fragmentos.png", dpi=300)\n',
            "plt.show()",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'print("=== AUDITORÍA FINAL: CASOS LIMPIOS VS CASOS ATÍPICOS / ANOMALÍAS ===")\n',
            "auditoria = []\n",
            "\n",
            "for _, r in df_eda.iterrows():\n",
            "    alertas = []\n",
            "    if not r['shape_match']:\n",
            (
                "        alertas.append('Dimensiones incompatibles Imagen vs"
                " Máscara')\n"
            ),
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
            (
                'df_reporte.to_csv(EDA_DIR / "auditoria_calidad_casos.csv",'
                " index=False)\n"
            ),
            "\n",
            'print(f"Resumen de auditoría de calidad:")\n',
            'print(df_reporte["estado"].value_counts())\n',
            "\n",
            "casos_atípicos = df_reporte[df_reporte['estado'] != 'Limpio / Confiable']\n",
            "if len(casos_atípicos) > 0:\n",
            '    print(f"\\nCasos marcados para control previo:")\n',
            (
                "    display(casos_atípicos[['case_id', 'estado',"
                " 'observaciones']].head(15))\n"
            ),
            "else:\n",
            (
                '    print("\\nTodos los casos cumplen perfectamente con los'
                ' criterios de consistencia.")'
            ),
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

print(f"Notebook corregido generado exitosamente en:\n{notebook_path}")
