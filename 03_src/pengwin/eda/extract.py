"""extract.py.

Extrae, caso por caso, TODAS las estadísticas que necesita el EDA
(04_notebook/01_eda_pengwin.ipynb) y las guarda en un JSON por caso.
El notebook solo lee tablas: así se ejecuta en segundos y el cálculo pesado
(~20-40 s por caso) se hace una sola vez y es reanudable.

Niveles de análisis:
    caso       -> geometría, orientación, HU, metal, recorte del cuerpo
    fragmento  -> volumen, forma, HU, conectividad 3D, distancia al principal, contacto
    corte (2D) -> presencia por región, cajas, islas, fragmentos visibles, píxeles de borde
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from pengwin.data.data_loader import get_orientation_code
from pengwin.data.preprocessing import HU_AIR, HU_BODY, HU_BONE_MIP, HU_METAL, compute_body_crop, compute_bone_crop

REGIONS = {"SA": (1, 10), "LI": (11, 20), "RI": (21, 30)}
TARGET_SIZE = 256                       # resolución de entrada al modelo
HIST_BINS = np.arange(-1024, 3072 + 16, 16)  # histograma HU común a todos los casos
STRUCT_26 = np.ones((3, 3, 3), bool)    # conectividad 3D completa (la que usará el post-proceso)


def _region_of(label_arr: np.ndarray) -> np.ndarray:
    """0 = fondo, 1 = SA, 2 = LI, 3 = RI (vectorizado)."""
    return np.where(label_arr > 0, (label_arr.astype(np.int16) - 1) // 10 + 1, 0).astype(np.uint8)


def _read_lps(path: Path) -> sitk.Image:
    return sitk.DICOMOrient(sitk.ReadImage(str(path)), "LPS")


def _bbox_union(slices_a, slices_b, pad: int, shape) -> tuple:
    out = []
    for sa, sb, n in zip(slices_a, slices_b, shape):
        out.append(slice(max(min(sa.start, sb.start) - pad, 0), min(max(sa.stop, sb.stop) + pad, n)))
    return tuple(out)


def _pairwise_min_distance(mask_a: np.ndarray, mask_b: np.ndarray, sampling, step: int = 1) -> float:
    """Distancia mínima (mm, centro a centro de vóxel) entre dos máscaras, vía EDT.

    ``step`` > 1 submuestrea (aproximación para estructuras grandes región↔región).
    """
    if step > 1:
        mask_a, mask_b = mask_a[::step, ::step, ::step], mask_b[::step, ::step, ::step]
        sampling = tuple(s * step for s in sampling)
    if not mask_a.any() or not mask_b.any():
        return float("nan")
    za, ya, xa = np.where(mask_a | mask_b)
    sl = (slice(za.min(), za.max() + 1), slice(ya.min(), ya.max() + 1), slice(xa.min(), xa.max() + 1))
    dt = ndi.distance_transform_edt(~mask_a[sl], sampling=sampling)
    return float(dt[mask_b[sl]].min())


def connectivity_stats(mask: np.ndarray, vox_mm3: float):
    """Componentes conexas 3D (26-vecindad) de un fragmento del GT."""
    cc, n = ndi.label(mask, structure=STRUCT_26)
    if n <= 1:
        return int(n), 1.0, int(n)
    sizes = np.bincount(cc.ravel())[1:]
    return int(n), float(sizes.max() / sizes.sum()), int((sizes * vox_mm3 >= 100).sum())


def bone_crop_stats(img: np.ndarray, spacing_zyx, yb: np.ndarray, xb: np.ndarray) -> Dict:
    """Recorte al esqueleto por HU (válido en inferencia) y si contiene todo el hueso etiquetado."""
    bc = compute_bone_crop(img, spacing_zyx)
    return {
        "bonecrop_side_px": bc.side_px,
        "bonecrop_side_mm": bc.side_px * max(spacing_zyx[1], spacing_zyx[2]),
        "px_mm_bonecrop_256": bc.pixel_mm_at(TARGET_SIZE),
        "bone_inside_bonecrop": bool(yb.min() >= bc.y0 and yb.max() < bc.y1 and xb.min() >= bc.x0 and xb.max() < bc.x1),
        "bonecrop_margin_min_mm": float(min(yb.min() - bc.y0, bc.y1 - 1 - yb.max(), xb.min() - bc.x0, bc.x1 - 1 - xb.max())
                                        * min(spacing_zyx[1], spacing_zyx[2])),
    }


def extract_case(case_id: str, image_path: Path, label_path: Path) -> Dict:
    t0 = time.time()
    img_itk_raw = sitk.ReadImage(str(image_path))
    lab_itk_raw = sitk.ReadImage(str(label_path))
    header_match = (
        np.allclose(img_itk_raw.GetSpacing(), lab_itk_raw.GetSpacing())
        and np.allclose(img_itk_raw.GetOrigin(), lab_itk_raw.GetOrigin())
        and np.allclose(img_itk_raw.GetDirection(), lab_itk_raw.GetDirection())
        and img_itk_raw.GetSize() == lab_itk_raw.GetSize()
    )
    orientation = get_orientation_code(img_itk_raw)
    img_itk = sitk.DICOMOrient(img_itk_raw, "LPS")
    lab_itk = sitk.DICOMOrient(lab_itk_raw, "LPS")
    del img_itk_raw, lab_itk_raw

    sx, sy, sz = img_itk.GetSpacing()
    sampling = (sz, sy, sx)                      # orden (Z, Y, X) para scipy
    vox_mm3 = sx * sy * sz
    img = sitk.GetArrayViewFromImage(img_itk)    # int32 (Z, Y, X), sin copia
    lab = sitk.GetArrayFromImage(lab_itk).astype(np.uint8)
    Z, H, W = lab.shape

    # ------------------------------------------------------------------ caso
    case: Dict = {
        "case_id": case_id, "orientation_orig": orientation, "header_match": bool(header_match),
        "pixel_type": img_itk.GetPixelIDTypeAsString(),
        "dim_x": W, "dim_y": H, "dim_z": Z, "dx": sx, "dy": sy, "dz": sz,
        "fov_x_mm": W * sx, "fov_y_mm": H * sy, "fov_z_mm": Z * sz,
    }
    flat_img = img.ravel()
    q = np.percentile(flat_img[:: max(1, flat_img.size // 2_000_000)], [0.1, 1, 50, 99, 99.9])
    case.update({"hu_min": float(img.min()), "hu_max": float(img.max()),
                 "hu_p001": q[0], "hu_p01": q[1], "hu_p50": q[2], "hu_p99": q[3], "hu_p999": q[4],
                 "frac_below_air": float((img < HU_AIR).mean())})

    bone = lab > 0
    bone_hu = img[bone]
    bq = np.percentile(bone_hu, [1, 5, 25, 50, 75, 95, 99])
    case.update({f"bone_hu_p{p:02d}": float(v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], bq)})
    case.update({
        "bone_frac_below_mip_thr": float((bone_hu < HU_BONE_MIP).mean()),
        "bone_frac_below_window": float((bone_hu < -500).mean()),
        "bone_frac_above_window": float((bone_hu > 1300).mean()),
        "bone_volume_cm3": float(bone.sum() * vox_mm3 / 1000),
    })
    hist_bone, _ = np.histogram(np.clip(bone_hu, -1024, 3071), bins=HIST_BINS)
    hist_all, _ = np.histogram(np.clip(flat_img[::4], -1024, 3071), bins=HIST_BINS)
    del bone_hu

    metal = img > HU_METAL
    case.update({
        "metal_cm3": float(metal.sum() * vox_mm3 / 1000),
        "metal_in_bone_cm3": float((metal & bone).sum() * vox_mm3 / 1000),
        "vox_above_2500_cm3": float((img > 2500).sum() * vox_mm3 / 1000),
        "above_mip_thr_outside_labels_cm3": float(((img >= HU_BONE_MIP) & ~bone).sum() * vox_mm3 / 1000),
    })
    if metal.any():
        # distancia (mm) del metal al hueso etiquetado (0 = dentro del hueso)
        case["metal_to_bone_mm"] = _pairwise_min_distance(bone, metal, sampling, step=2)
    else:
        case["metal_to_bone_mm"] = float("nan")
    del metal

    # Recorte del cuerpo (solo imagen) y su efecto en la resolución efectiva
    crop = compute_body_crop(np.asarray(img), (sz, sy, sx))
    yb, xb = np.where(bone.any(0))
    case.update({
        "crop_side_px": crop.side_px, "crop_side_mm": crop.side_px * max(sy, sx),
        "px_mm_no_crop_256": max(H * sy, W * sx) / TARGET_SIZE,
        "px_mm_crop_256": crop.pixel_mm_at(TARGET_SIZE),
        "bone_bbox_w_mm": float((xb.max() - xb.min() + 1) * sx), "bone_bbox_h_mm": float((yb.max() - yb.min() + 1) * sy),
        "bone_inside_crop": bool(yb.min() >= crop.y0 and yb.max() < crop.y1 and xb.min() >= crop.x0 and xb.max() < crop.x1),
    })
    case.update(bone_crop_stats(np.asarray(img), (sz, sy, sx), yb, xb))
    case["bonecrop_v"] = 3  # v3: componentes óseas grandes + cercanas (sin manos)
    zs_bone = np.where(bone.any(axis=(1, 2)))[0]
    case.update({"z_bone_first": int(zs_bone.min()), "z_bone_last": int(zs_bone.max()),
                 "slices_with_bone": int(zs_bone.size), "bone_z_extent_mm": float((zs_bone.max() - zs_bone.min() + 1) * sz)})
    scale256 = TARGET_SIZE / crop.side_px        # factor nativo -> entrada del modelo

    # ------------------------------------------------------------ fragmentos
    counts = np.bincount(lab.ravel(), minlength=31)
    objs = ndi.find_objects(lab)
    shape_f = sitk.LabelShapeStatisticsImageFilter()
    shape_f.ComputeFeretDiameterOff()
    shape_f.ComputePerimeterOff()
    shape_f.Execute(sitk.Cast(lab_itk, sitk.sitkUInt8))

    fragments: List[Dict] = []
    region_masks = {}
    for reg, (lo, hi) in REGIONS.items():
        ids = [i for i in range(lo, hi + 1) if counts[i] > 0]
        case[f"n_frag_{reg}"] = len(ids)
        if not ids:
            continue
        main_id = max(ids, key=lambda i: counts[i])      # principal = mayor volumen
        case[f"main_is_lowest_{reg}"] = bool(main_id == min(ids))
        region_masks[reg] = (lab >= lo) & (lab <= hi)

        # EDT del complemento del principal UNA vez por región (sobre el bbox de la región)
        reg_obj = ndi.find_objects(region_masks[reg].astype(np.uint8))[0]
        reg_obj = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, n)) for s, n in zip(reg_obj, lab.shape))
        lab_r = lab[reg_obj]
        main_r = lab_r == main_id
        dt_main = ndi.distance_transform_edt(~main_r, sampling=sampling) if len(ids) > 1 else None
        dil6 = ndi.binary_dilation  # alias

        for i in ids:
            sl = objs[i - 1]
            m = lab[sl] == i
            n_cc, cc_frac, n_cc_big = connectivity_stats(m, vox_mm3)
            surf = m & ~ndi.binary_erosion(m)
            # vecinos 6-conectados que pertenecen a OTRO fragmento de la misma región / de otra región
            pad = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, n)) for s, n in zip(sl, lab.shape))
            lab_p = lab[pad]
            m_p = lab_p == i
            ring = dil6(m_p) & ~m_p
            neigh = lab_p[ring]
            reg_n = (neigh.astype(np.int16) - 1) // 10
            same = (neigh > 0) & (reg_n == (i - 1) // 10)
            other = (neigh > 0) & (reg_n != (i - 1) // 10)
            # superficie de contacto: vóxeles de superficie del fragmento que tocan otro fragmento
            touch_same = m_p & dil6(np.isin(lab_p, [j for j in ids if j != i]))
            hu_f = img[sl][m]
            rec = {
                "case_id": case_id, "region": reg, "label": i, "is_main": i == main_id,
                "n_vox": int(counts[i]), "volume_cm3": counts[i] * vox_mm3 / 1000,
                "vol_frac_region": counts[i] / sum(counts[j] for j in ids),
                "ext_z_mm": (sl[0].stop - sl[0].start) * sz, "ext_y_mm": (sl[1].stop - sl[1].start) * sy,
                "ext_x_mm": (sl[2].stop - sl[2].start) * sx,
                "n_slices": sl[0].stop - sl[0].start,
                "z_center_norm": ((sl[0].start + sl[0].stop) / 2 - zs_bone.min()) / max(np.ptp(zs_bone), 1),
                "n_cc_3d": int(n_cc),
                "cc_largest_frac": cc_frac,          # fracción del volumen en la componente 3D mayor
                "n_cc_ge_100mm3": n_cc_big,          # componentes de tamaño no trivial (>= 0,1 cm3)
                "surface_vox": int(surf.sum()),
                "contact_same_region_vox": int(touch_same.sum()),
                "contact_frac_surface": float(touch_same.sum() / max(int(surf.sum()), 1)),
                "touches_other_fragment": bool(same.any()),
                "touches_other_region": bool(other.any()),
                "hu_mean": float(hu_f.mean()), "hu_p05": float(np.percentile(hu_f, 5)),
                "hu_p95": float(np.percentile(hu_f, 95)),
                "elongation": shape_f.GetElongation(i), "flatness": shape_f.GetFlatness(i),
                "ellipsoid_diam_max_mm": max(shape_f.GetEquivalentEllipsoidDiameter(i)),
                "ellipsoid_diam_min_mm": min(shape_f.GetEquivalentEllipsoidDiameter(i)),
            }
            if dt_main is not None and i != main_id:
                d = float(dt_main[lab_r == i].min())
                rec["dist_to_main_mm"] = d
                rec["contact_with_main"] = bool(d <= np.sqrt(sx**2 + sy**2 + sz**2) + 1e-6)
            fragments.append(rec)
        del dt_main, lab_r, main_r

    # Distancias región↔región (articulaciones sacroilíacas y sínfisis), aproximadas con step=2
    for a, b in [("SA", "LI"), ("SA", "RI"), ("LI", "RI")]:
        if a in region_masks and b in region_masks:
            case[f"dist_{a}_{b}_mm"] = _pairwise_min_distance(region_masks[a], region_masks[b], sampling, step=2)

    # ---------------------------------------------------------------- cortes
    reg_map = _region_of(lab)
    slices: List[Dict] = []
    for z in range(Z):
        l2 = lab[z]
        if not l2.any():
            continue
        r2 = reg_map[z]
        # píxeles 4-vecinos con etiqueta distinta (ambos hueso): borde de fractura o interfaz entre regiones
        frac_edge = np.zeros_like(l2, bool)
        inter_edge = np.zeros_like(l2, bool)
        for axis in (0, 1):
            for shift in (1, -1):
                ln = np.roll(l2, shift, axis=axis)
                rn = np.roll(r2, shift, axis=axis)
                diff = (l2 > 0) & (ln > 0) & (l2 != ln)
                frac_edge |= diff & (rn == r2)
                inter_edge |= diff & (rn != r2)
        row = {"case_id": case_id, "z": z,
               "z_norm": (z - zs_bone.min()) / max(np.ptp(zs_bone), 1),
               "fg_px_256": float((l2 > 0).sum() * scale256**2),
               "frac_edge_px_256": float(frac_edge.sum() * scale256**2),
               "inter_edge_px_256": float(inter_edge.sum() * scale256**2)}
        for k, (reg, (lo, hi)) in enumerate(REGIONS.items(), start=1):
            m = r2 == k
            row[f"has_{reg}"] = bool(m.any())
            if not m.any():
                continue
            ys, xs = np.where(m)
            frag_ids = np.unique(l2[m])
            island_counts = [ndi.label(l2 == f)[1] for f in frag_ids]
            areas = [int((l2 == f).sum()) for f in frag_ids]
            row.update({
                f"{reg}_w_256": (xs.max() - xs.min() + 1) * scale256, f"{reg}_h_256": (ys.max() - ys.min() + 1) * scale256,
                f"{reg}_w_mm": (xs.max() - xs.min() + 1) * sx, f"{reg}_h_mm": (ys.max() - ys.min() + 1) * sy,
                f"{reg}_cx_rel": ((xs.min() + xs.max()) / 2 - crop.x0) / crop.side_px,
                f"{reg}_cy_rel": ((ys.min() + ys.max()) / 2 - crop.y0) / crop.side_px,
                f"{reg}_area_256": float(m.sum() * scale256**2),
                f"{reg}_n_frag": int(frag_ids.size),
                f"{reg}_n_islands": int(ndi.label(m)[1]),
                f"{reg}_max_islands_per_frag": int(max(island_counts)),
                f"{reg}_min_frag_area_256": float(min(areas) * scale256**2),
            })
        slices.append(row)

    case["seconds"] = time.time() - t0
    return {"case": case, "fragments": fragments, "slices": slices,
            "hist_bone": hist_bone.tolist(), "hist_all": hist_all.tolist()}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def save_case(result: Dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['case']['case_id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, default=_json_default), encoding="utf-8")
    tmp.replace(path)          # escritura atómica: un caso a medias nunca queda como "hecho"
    return path
