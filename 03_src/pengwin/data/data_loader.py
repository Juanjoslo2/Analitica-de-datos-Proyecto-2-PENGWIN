"""data_loader.py.

Cargador de datos médicos con reorientación forzada a LPS,
extracción de espaciado físico (mm), ventaneo óseo HU y mapeo taxonómico PENGWIN.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import SimpleITK as sitk

# Taxonomía anatómica fija PENGWIN
ANATOMICAL_RANGES = {
    "sacro": {"range": (1, 10), "main_id": 1, "class_idx": 0},
    "coxal_izquierdo": {"range": (11, 20), "main_id": 11, "class_idx": 1},
    "coxal_derecho": {"range": (21, 30), "main_id": 21, "class_idx": 2},
}


def get_orientation_code(itk_image: sitk.Image) -> str:
    """Extrae el código de orientación anatómica (ej.

    'LPS', 'RAS') de los cosenos directores.
    """
    direction = itk_image.GetDirection()
    try:
        # Método estándar en SimpleITK
        return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
            direction
        )
    except AttributeError:
        try:
            # Variante en algunas versiones de bindings
            return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                direction
            )
        except Exception:
            return "DESCONOCIDA"


def load_and_standardize_mha(
    file_path: Path | str,
    is_label: bool = False,
    image_dtype: type = np.float32,
) -> Tuple[np.ndarray, Tuple[float, float, float], str]:
    """Carga un volumen .mha, lo reorienta estrictamente a 'LPS' y devuelve:

    - Matriz NumPy en formato (Z, Y, X) -> (Cortes, Alto, Ancho).
      En LPS: x crece hacia la IZQUIERDA del paciente, y hacia POSTERIOR, z hacia SUPERIOR.
    - Espaciado físico en mm calibrado a (sz, sy, sx)  -> usar tal cual como
      ``sampling`` de ``scipy.ndimage.distance_transform_edt``.
    - Orientación original del archivo antes de reorientar ('LPS' o 'RAS' en PENGWIN).

    ``image_dtype``: float32 por defecto. Para cachear, ``np.int16`` ahorra la mitad
    de memoria, pero exige recortar antes a [-1024, 3071] HU (hay casos con metal
    de hasta ~47 000 HU que desbordarían int16).
    """
    path_str = str(file_path)
    itk_image = sitk.ReadImage(path_str)

    # Detectar orientación anatómica original
    original_orientation = get_orientation_code(itk_image)

    # Reorientar forzosamente a LPS (Left-Posterior-Superior)
    # Permuta ejes y voltea según sea necesario sin interpolación (mantiene etiquetas enteras intactas)
    itk_image = sitk.DICOMOrient(itk_image, "LPS")

    # Espaciado físico: SimpleITK entrega (sx, sy, sz) -> Invertir a (sz, sy, sx)
    spacing_xyz = itk_image.GetSpacing()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    # Obtener matriz NumPy en orden (Z, Y, X)
    volume_np = sitk.GetArrayFromImage(itk_image)

    if is_label:
        # Las etiquetas PENGWIN van de 0 a 30: uint8 basta y ocupa 1/2 de int16
        volume_np = volume_np.astype(np.uint8)
    else:
        volume_np = volume_np.astype(image_dtype, copy=False)

    return volume_np, spacing_zyx, original_orientation


def apply_bone_window(
    volume: np.ndarray,
    window_center: float = 400.0,
    window_width: float = 1800.0,
) -> np.ndarray:
    """Aplica ventaneo de hueso en Unidades Hounsfield (HU) y escala a [0.0, 1.0].

    Corta valores por fuera de [-500, 1300] HU.
    """
    min_hu = window_center - (window_width / 2.0)
    max_hu = window_center + (window_width / 2.0)

    windowed = np.clip(volume, min_hu, max_hu)
    normalized = (windowed - min_hu) / (max_hu - min_hu)
    return normalized.astype(np.float32)


def get_dataset_pairs(data_dir: Path | str) -> List[Dict[str, Path]]:
    """Empareja imágenes y máscaras de las carpetas de entrenamiento."""
    base_path = Path(data_dir)
    labels_dir = base_path / "PENGWIN_CT_train_labels"
    img_dirs = [
        base_path / "PENGWIN_CT_train_images_part1",
        base_path / "PENGWIN_CT_train_images_part2",
    ]

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de etiquetas: {labels_dir}")
    label_files = {f.name: f for f in labels_dir.glob("*.mha")}

    matched_pairs = []
    seen = set()
    for d in img_dirs:
        if not d.exists():
            warnings.warn(f"Carpeta de imágenes no encontrada: {d}")
            continue
        for img_file in d.glob("*.mha"):
            if img_file.name in seen:
                raise ValueError(f"Caso duplicado entre carpetas de imágenes: {img_file.name}")
            seen.add(img_file.name)
            if img_file.name in label_files:
                matched_pairs.append(
                    {
                        "case_id": img_file.stem,
                        "image_path": img_file,
                        "label_path": label_files[img_file.name],
                    }
                )

    orphan_labels = sorted(set(label_files) - seen)
    orphan_images = sorted(seen - set(label_files))
    if orphan_labels or orphan_images:
        warnings.warn(
            f"Sin pareja -> etiquetas: {orphan_labels[:5]} | imágenes: {orphan_images[:5]}"
        )
    return sorted(matched_pairs, key=lambda x: x["case_id"])


def decompose_labels_by_region(label_mask: np.ndarray) -> Dict[str, Dict]:
    """Analiza la máscara 3D y descompone las instancias por macro-hueso."""
    unique_ids = np.unique(label_mask)
    regions_info = {}

    for region_name, meta in ANATOMICAL_RANGES.items():
        low, high = meta["range"]
        main_id = meta["main_id"]

        present_fragments = [
            int(lbl) for lbl in unique_ids if low <= lbl <= high
        ]

        if not present_fragments:
            continue

        has_main = main_id in present_fragments
        conminuted = [lbl for lbl in present_fragments if lbl != main_id]

        regions_info[region_name] = {
            "class_idx": meta["class_idx"],
            "present_fragments": present_fragments,
            "main_fragment_id": main_id if has_main else None,
            "conminuted_ids": conminuted,
            "num_fragments": len(present_fragments),
            "is_fractured": len(present_fragments) > 1,
        }

    return regions_info


def verify_left_right_consistency(
    label_mask: np.ndarray,
) -> Tuple[bool, str]:
    """Test de sanidad anatómica para corroborar la corrección LPS.

    En LPS el eje x del arreglo crece hacia la IZQUIERDA del paciente, por lo que
    el centroide del coxal izquierdo (11-20) debe tener x MAYOR que el del derecho
    (21-30). Si la reorientación falla (p. ej. un caso RAS sin corregir), el orden
    se invierte y el test falla. Comparar solo ``!=`` no detecta ese error.
    """
    left_mask = (label_mask >= 11) & (label_mask <= 20)
    right_mask = (label_mask >= 21) & (label_mask <= 30)

    if not np.any(left_mask) or not np.any(right_mask):
        return True, "Falta alguno de los coxales para contrastar lateralidad."

    # Centroide exacto en X a partir del perfil de vóxeles por columna
    # (evita materializar las coordenadas de cada vóxel con np.where)
    x = np.arange(label_mask.shape[2])
    w_left = left_mask.sum(axis=(0, 1))
    w_right = right_mask.sum(axis=(0, 1))
    left_x_center = float((w_left * x).sum() / w_left.sum())
    right_x_center = float((w_right * x).sum() / w_right.sum())

    is_consistent = left_x_center > right_x_center
    msg = (
        f"Centro X Izquierdo: {left_x_center:.1f} vs Centro X Derecho: {right_x_center:.1f} "
        f"(esperado izquierdo > derecho en LPS)"
    )
    return is_consistent, msg


if __name__ == "__main__":
    # Ubicar la raíz del proyecto sin importar desde dónde se ejecute el script
    script_dir = Path(__file__).resolve()
    # pengwin -> data -> 03_src -> repo root
    repo_root = script_dir.parents[3]
    base_data_path = repo_root / "01_data"

    if not base_data_path.exists():
        base_data_path = Path("01_data")

    pairs = get_dataset_pairs(base_data_path)
    print(f"Total casos emparejados encontrados: {len(pairs)}")

    if pairs:
        pilot = pairs[0]
        print(f"\n--- Probando caso: {pilot['case_id']} ---")

        raw_img, spacing, orig_ori_img = load_and_standardize_mha(
            pilot["image_path"], is_label=False
        )
        mask, _, orig_ori_mask = load_and_standardize_mha(
            pilot["label_path"], is_label=True
        )

        print(
            f"Orientación original en disco: Imagen={orig_ori_img} | Máscara={orig_ori_mask}"
        )
        print("Orientación actual: Estandarizada forzosamente a LPS")
        print(f"Dimensiones NumPy (Z, Y, X): {raw_img.shape}")
        print(
            f"Espaciado físico calibrado (dz, dy, dx): {spacing[0]:.3f} mm, {spacing[1]:.3f} mm, {spacing[2]:.3f} mm"
        )

        # Ventaneo óseo
        bone_img = apply_bone_window(raw_img)
        print(
            f"Imagen con Ventana Ósea: Min={bone_img.min():.2f}, Max={bone_img.max():.2f}"
        )

        # Descomposición anatómica
        regions = decompose_labels_by_region(mask)
        print("\nRegiones anatómicas detectadas en este caso:")
        for reg, data in regions.items():
            print(
                f"  - {reg.upper()}: {data['num_fragments']} fragmento(s) (Principal: {data['main_fragment_id']}, Conminutos: {data['conminuted_ids']})"
            )

        # Sanity check de lateralidad
        ok, msg = verify_left_right_consistency(mask)
        print(f"\nTest de consistencia lateral: {'CORRECTO' if ok else 'FALLO'}")
        print(f"Detalle: {msg}")