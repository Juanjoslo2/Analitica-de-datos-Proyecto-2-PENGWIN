from pathlib import Path

import numpy as np

from pengwin.data.data_loader import (
    get_dataset_pairs,
    load_and_standardize_mha,
    label_id_to_class_idx,
)

from pengwin.data.preprocessing import (
    compute_bone_crop,
    transform_boxes_to_crop,
)


def mask_to_boxes(label_slice):
    """Convierte una máscara 2D en bounding boxes."""

    boxes = []
    classes = []

    values = np.unique(label_slice)
    values = values[values != 0]

    for label_id in values:

        ys, xs = np.where(label_slice == label_id)

        if len(xs) == 0:
            continue

        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()

        boxes.append([x1, y1, x2, y2])
        classes.append(label_id)

    return (
        np.array(boxes, dtype=np.float32),
        np.array(classes, dtype=np.int64),
    )



def make_25d_slice(
    image,
    z,
    spacing,
    context_mm=2.0,
):
    """Construye una imagen 2.5D [3, H, W]."""

    dz = spacing[0]

    delta = round(context_mm / dz)

    z_indices = [
        z - delta,
        z,
        z + delta,
    ]

    # Comprobar límites del volumen
    if z_indices[0] < 0 or z_indices[-1] >= image.shape[0]:
        raise ValueError(
            f"No hay suficiente contexto alrededor del corte {z}. "
            f"Índices solicitados: {z_indices}"
        )

    image_25d = np.stack(
        [image[idx] for idx in z_indices],
        axis=0,
    )

    return image_25d


# --------------------------------------------------
# 1. Buscar un caso
# --------------------------------------------------

base_data_path = Path("01_data")

pairs = get_dataset_pairs(base_data_path)

print("Casos encontrados:", len(pairs))

case = pairs[0]

print("\nCaso:", case["case_id"])


# --------------------------------------------------
# 2. Cargar imagen y label
# --------------------------------------------------

image, spacing, orientation = load_and_standardize_mha(
    case["image_path"],
    is_label=False,
)

label, _, _ = load_and_standardize_mha(
    case["label_path"],
    is_label=True,
)

print("\nImagen:", image.shape)
print("Label:", label.shape)
print("Spacing:", spacing)
print("Orientación original:", orientation)


# --------------------------------------------------
# 3. Calcular crop óseo
# --------------------------------------------------

crop = compute_bone_crop(
    image,
    spacing,
    margin_mm=15.0,
)

print("\n===== BONE CROP =====")
print("y0:", crop.y0)
print("y1:", crop.y1)
print("x0:", crop.x0)
print("x1:", crop.x1)
print("side:", crop.side_px)


# --------------------------------------------------
# 4. Elegir un corte con etiquetas
# --------------------------------------------------

slices_with_labels = np.where(
    np.any(label > 0, axis=(1, 2))
)[0]

z = slices_with_labels[len(slices_with_labels) // 2]

print("\nCorte seleccionado:", z)


image_25d = make_25d_slice(
    image,
    z,
    spacing,
    context_mm=2.0,
)

print("\n===== 2.5D =====")
print("Shape:", image_25d.shape)
print("Min:", image_25d.min())
print("Max:", image_25d.max())


# --------------------------------------------------
# 5. Obtener bounding boxes originales
# --------------------------------------------------

boxes, classes = mask_to_boxes(label[z])

# Convertir IDs PENGWIN → clases del detector
classes = np.array(
    [label_id_to_class_idx(label_id) for label_id in classes],
    dtype=np.int64,
)

print("\n===== BOXES ORIGINALES =====")

print("Boxes:")
print(boxes)

print("Classes:")
print(classes)

# --------------------------------------------------
# 6. Transformar boxes al crop + 256x256
# --------------------------------------------------

boxes_256 = transform_boxes_to_crop(
    boxes,
    crop,
    output_size=256,
)

print("\n===== BOXES 256x256 =====")

print(boxes_256)
print("Classes:")
print(classes)