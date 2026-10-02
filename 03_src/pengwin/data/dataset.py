from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset

from pengwin.data.data_loader import (
    get_dataset_pairs,
    load_and_standardize_mha,
    label_id_to_class_idx,
    apply_bone_window,
)

from pengwin.data.preprocessing import (
    compute_bone_crop,
    apply_crop,
    transform_boxes_to_crop,
)


class PENGWINDataset(Dataset):
    """
    Dataset para detección de estructuras óseas en PENGWIN.

    Cada muestra contiene:

        image:
            Tensor [3, 256, 256]

        target:
            {
                "boxes": Tensor [N, 4],
                "labels": Tensor [N],
            }
    """

    def __init__(
        self,
        data_dir,
        output_size=256,
        context_mm=2.0,
        crop_margin_mm=15.0,
        window_center=400,
        window_width=1800,
    ):
        self.data_dir = Path(data_dir)

        self.output_size = output_size
        self.context_mm = context_mm
        self.crop_margin_mm = crop_margin_mm
        self.window_center = window_center
        self.window_width = window_width

        # Buscar los pares imagen-label
        self.pairs = get_dataset_pairs(self.data_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        # ================================================
        # 1. Obtener el caso
        # ================================================

        case = self.pairs[index]

        image_path = case["image_path"]
        label_path = case["label_path"]

        # ================================================
        # 2. Cargar imagen y label
        # ================================================

        image, spacing, orientation = load_and_standardize_mha(
            image_path
        )

        label, label_spacing, label_orientation = load_and_standardize_mha(
            label_path
        )

        # ================================================
        # 3. Calcular bone crop
        # ================================================

        bone_crop = compute_bone_crop(
            volume_hu=image,
            spacing_zyx=spacing,
            margin_mm=self.crop_margin_mm,
        )

        # ================================================
        # 4. Buscar cortes que tengan labels
        # ================================================

        slices_with_labels = np.where(
            np.any(label > 0, axis=(1, 2))
        )[0]

        if len(slices_with_labels) == 0:
            raise RuntimeError(
                f"El caso {case['case_id']} no contiene labels."
            )

        # Por ahora usamos el corte central
        z = slices_with_labels[len(slices_with_labels) // 2]

        # ================================================
        # 5. Construir entrada 2.5D
        # ================================================

        dz = spacing[0]

        delta = round(self.context_mm / dz)

        z_indices = [
            z - delta,
            z,
            z + delta,
        ]

        if (
            z_indices[0] < 0
            or z_indices[-1] >= image.shape[0]
        ):
            raise RuntimeError(
                f"Contexto 2.5D fuera de rango en "
                f"caso {case['case_id']}, z={z}"
            )

        image_25d = np.stack(
            [
                image[z_indices[0]],
                image[z_indices[1]],
                image[z_indices[2]],
            ],
            axis=0,
        )

        # ================================================
        # 6. Bone window
        # ================================================

        image_25d = apply_bone_window(
            image_25d,
            window_center=self.window_center,
            window_width=self.window_width,
        )

        # ================================================
        # 7. Aplicar bone crop
        # ================================================

        image_25d = apply_crop(
            image_25d,
            bone_crop,
            fill=0.0,
        )

        # ================================================
        # 8. Resize a 256x256
        # ================================================

        scale = self.output_size / bone_crop.side_px

        image_25d = ndi.zoom(
            image_25d,
            zoom=(1, scale, scale),
            order=1,
        )



        # ================================================
        # 9. Obtener cajas macroanatómicas y máscara
        # ================================================

        label_slice = label[z]

        # Cada grupo de IDs representa una región anatómica.
        region_ids = {
            0: range(1, 11),    # Sacro
            1: range(11, 21),   # Coxal izquierdo
            2: range(21, 31),   # Coxal derecho
        }

        boxes_original = []
        classes = []

        for class_idx, valid_ids in region_ids.items():

            # Unir los fragmentos de la misma región
            region_mask = np.isin(
                label_slice,
                list(valid_ids),
            )

            ys, xs = np.where(region_mask)

            if len(xs) == 0:
                continue

            # Coordenadas con límite superior exclusivo
            x1 = xs.min()
            y1 = ys.min()
            x2 = xs.max() + 1
            y2 = ys.max() + 1

            boxes_original.append(
                [x1, y1, x2, y2]
            )

            classes.append(class_idx)

        boxes_original = np.asarray(
            boxes_original,
            dtype=np.float32,
        ).reshape(-1, 4)

        classes = np.asarray(
            classes,
            dtype=np.int64,
        )

        # ================================================
        # 10. Transformar cajas a 256x256
        # ================================================

        boxes = transform_boxes_to_crop(
            boxes=boxes_original,
            crop=bone_crop,
            output_size=self.output_size,
        )

        # ================================================
        # 11. Transformar máscara de fragmentos
        # ================================================

        mask_crop = apply_crop(
            label_slice,
            bone_crop,
            fill=0,
        )

        scale = self.output_size / bone_crop.side_px

        instance_mask = ndi.zoom(
            mask_crop,
            zoom=(scale, scale),
            order=0,
        )

        # Ajustar posibles diferencias de tamaño por redondeo
        mask_resized = np.zeros(
            (self.output_size, self.output_size),
            dtype=instance_mask.dtype,
        )

        height = min(self.output_size, instance_mask.shape[0])
        width = min(self.output_size, instance_mask.shape[1])

        mask_resized[:height, :width] = instance_mask[:height, :width]

        # ================================================
        # 12. Convertir imagen a Tensor
        # ================================================

        image_tensor = torch.from_numpy(
            image_25d.astype(np.float32)
        )

        # ================================================
        # 13. Crear target
        # ================================================

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(classes),
            "mask": torch.from_numpy(
                mask_resized.astype(np.int64)
            ),
            "case_id": case["case_id"],
            "z": int(z),
        }

        return image_tensor, target