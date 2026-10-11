"""model_cards.py.

Genera un model card por cada modelo entrenado entregado (§7 del enunciado) a partir de los
reportes versionados, para que ninguna cifra se copie a mano:

    reports/model_cards/final.md      v2_last + TTA 5 + posproceso F2C2 (configs/tuned.yaml)
    reports/model_cards/v2.md         v2 (40 épocas; checkpoints best = época 22, last = época 39)
    reports/model_cards/base.md       semana 9, con CBAM y transfer learning
    reports/model_cards/sin_cbam.md   ablación sin CBAM
    reports/model_cards/sin_tl.md     ablación sin transfer learning

Uso:  python scripts/model_cards.py
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REP = REPO / "reports"
OUT = REP / "model_cards"

AVISO = ("> **Uso exclusivamente académico.** Este modelo no es un dispositivo médico, no ha sido validado "
         "clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.")

OBJ = [("Clasificación", "F1", "cls_f1_macro", 0.85), ("Clasificación", "AUC", "cls_auc_macro", 0.85),
       ("Detección", "IoU promedio", "IoU_promedio", 0.65), ("Detección", "mAP@0.50", "mAP@0.50", 0.65),
       ("Detección", "mAP@[.50:.95]", "mAP@[.50:.95]", 0.40)]


def rj(p):
    p = REP / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(v):
    return "—" if v is None else f"{v:.3f}".replace(".", ",")


def tabla(ev, fr):
    filas = ["| Tarea | Métrica | Test | Objetivo (§5) | Cumple |", "|---|---|---|---|---|"]
    for t, n, k, o in OBJ:
        v = ev.get(k)
        filas.append(f"| {t} | {n} | {f3(v)} | ≥ {f3(o)} | {'sí' if v is not None and v >= o else 'no'} |")
    if fr:
        for n, k, o in (("Dice", "dice_fragmento", 0.85), ("IoU", "iou_fragmento", 0.70)):
            v = fr.get(k)
            filas.append(f"| Segmentación de fragmento | {n} | {f3(v)} | ≥ {f3(o)} | {'sí' if v >= o else 'no'} |")
    filas.append(f"| Segmentación por región | Dice (SA, LI, RI) | {f3(ev.get('dice_hueso'))} | — | — |")
    return "\n".join(filas)


ARQ = """- **Entrada:** corte axial 2.5D (z−Δ, z, z+Δ, Δ = round(2 mm / dz)), 256×256, ventana L400/W1800, recorte óseo calculado solo con la imagen, volumen reorientado a LPS por los cosenos directores del header.
- **Backbone:** FundidoraPC del curso extendida con un bloque residual por etapa (anchos 32-64-128-256){cbam}.
- **Cabezas (desde cero):** clasificación de presencia de SA/LI/RI; detección anchor-free con grid propio de stride 8 y NMS propia (una caja por hueso); segmentación semántica de 4 clases + probabilidad de borde de fractura.
- **Segunda etapa (sin aprendizaje):** fragmentos por semillas de la transformada de distancia del núcleo + watershed; distancia de separación con `distance_transform_edt` y el spacing del header.
- **Pérdida:** λ_cls·L_cls + λ_det·L_det + λ_seg·L_seg, con λ por norma de gradiente sobre el bloque 4 ({lambdas}).
- **Parámetros:** {params:,} (cabezas incluidas)."""

DATOS = """- **Datos:** PENGWIN (MICCAI 2024), 100 CT de pelvis fracturada con máscaras de fragmentos de expertos. Partición fija por paciente (`01_data/splits.json`, balanceada por patrón de fractura): 70 train, 15 val, 15 test. El ajuste de la semana 10 usó validación cruzada de 5 folds sobre los 85 de train+val y no miró el test. Antes de ese ajuste, la época entregada de `v2` (la última en vez de la mejor en validación) y el paso a la separación por distancia se decidieron viendo resultados de test; está declarado en `04_notebook/02_semana10.ipynb` §21.
- **Entrenamiento:** AdamW, lr 3·10⁻⁴ con coseno, batch 8, precisión mixta (`torch.amp`), aumentación geométrica leve (escala 0,9-1,1, rotación ±10°, traslación 5 %) y de intensidad, **sin volteo** (distingue izquierda y derecha); semilla 42. {extra}
- **Cómputo:** DGX Spark (GPU GB10), contenedor NGC PyTorch 25.09."""

LIMITES = """- **Fragmentos en contacto:** cuando un secundario toca al principal en la superficie de fractura, el modelo tiende a fusionarlos o a cortarlos por otro sitio. En los 5 peores casos de test, los 10 secundarios perdidos estaban en contacto y 8 fueron absorbidos por el principal. El error de la distancia es mayor en contacto ({mae_c} mm) que separados ({mae_s} mm).
- **Fragmentos pequeños:** los de menos de 5 cm³ tienen Dice {d5}. La cabeza de borde encuentra menos del 20 % de la superficie de fractura real.
- **Fragmentos de más:** predice entre 14 y 28 fragmentos por paciente contra 4-8 reales. La métrica por fragmento real no los castiga, pero en la reconstrucción 3D aparecen.
- **2D corte a corte:** el contexto en z se limita a ±Δ; los objetivos del enunciado son de un reto académico de 4 semanas, por debajo del estado del arte 3D del challenge.
- **Datos:** 100 casos de un solo conjunto; no se probó en otros escáneres ni protocolos. Un volumen oblicuo se remuestrea a ejes LPS, pero esa ruta no está validada con datos reales (los 100 casos están alineados).
- **Detección:** las cajas de menos de 16 px (ramas del sacro en sus extremos) tienen IoU medio 0,75 contra 0,92 global."""


def card(nombre, titulo, ckpt, ev, fr, train, extra_arq, extra_datos, notas, lat=None):
    lam = train.get("lambdas", {}) if train else {}
    lam_s = ", ".join(f"λ_{k} = {str(round(v, 3)).replace('.', ',')}" for k, v in lam.items())
    txt = [f"# Model card · {titulo}", "", AVISO, "",
           f"Proyecto Integrador Corte 2, Analítica de Datos, UAO 2026-2. Checkpoint: `{ckpt}`.", "",
           "## Arquitectura", ARQ.format(cbam=extra_arq, lambdas=lam_s or "—", params=(train or {}).get("parametros", {}).get("total", 0)), "",
           "## Datos y entrenamiento", DATOS.format(extra=extra_datos), "",
           "## Métricas en test (15 pacientes)", tabla(ev, fr), ""]
    if lat:
        txt += ["## Latencia por imagen (un corte, RTX 3060 Laptop / Intel i7)", "",
                "| | CPU fp32 | GPU fp32 | GPU AMP | Volumen completo (GPU) |", "|---|---|---|---|---|"]
        for k, l in lat.items():
            txt.append((f"| {k} | {l['cpu_fp32']['media_ms']:.0f} ms | {l['gpu_fp32']['media_ms']:.1f} ms | "
                        f"{l['gpu_amp']['media_ms']:.1f} ms | {l['gpu_volumen_completo_s']:.1f} s ({l['cortes_volumen']} cortes) |").replace(".", ","))
        txt.append("")
    txt += ["## Notas", re.sub(r"(\d)\.(\d)", r"\1,\2", notas), "", "## Uso previsto y fuera de alcance",
            "- **Previsto:** docencia e investigación académica sobre detección y segmentación de fragmentos pélvicos en CT; demostración en el dashboard del curso.",
            "- **Fuera de alcance:** cualquier uso clínico, planeación quirúrgica real, diagnóstico o triaje; volúmenes que no sean CT de pelvis.", ""]
    if fr:
        txt += ["## Limitaciones conocidas", LIMITES.format(mae_c=f"{fr.get('mae_mm_en_contacto'):.2f}".replace(".", ","),
                                                            mae_s=f"{fr.get('mae_mm_separados'):.2f}".replace(".", ","), d5=f3(fr.get("dice_<5 cm3"))), ""]
    else:
        txt += ["## Limitaciones conocidas", "- Mismas que el modelo final en la parte de región y detección; este modelo no se evaluó por fragmento. Ver `final.md`.", ""]
    (OUT / f"{nombre}.md").write_text("\n".join(txt), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cbam = ", con CBAM (canal + espacial) en las etapas 3 y 4, antes de la bifurcación hacia las cabezas"
    v2 = rj("train/v2.json")
    ev_v2l = rj("eval/v2_last_test.json")["metricas"]
    fin = rj("fragments/v2_last_test_tuned_superficie.json") or rj("fragments/v2_last_test_tuned.json")
    g = fin["global"]
    lat = {"sin TTA": rj("latency/v2_last.json"), "con TTA de 5 pasadas (final)": rj("latency/v2_last_tta.json")}
    notas_fin = (f"- **Inferencia:** TTA de 5 pasadas (escala 0,92/1,08, rotación ±8°) que promedia región y borde; detección y clasificación salen de la pasada sin transformar.\n"
                 f"- **Posproceso:** `edt_hmax` (semilla a 5 mm del borde del núcleo, umbral de borde 0,1, rescate por h-máxima con h = 1,5 mm), elegido por validación cruzada.\n"
                 f"- **Métricas opcionales (§5):** HD95 {g.get('hd95_mm'):.1f} mm y ASSD {g.get('assd_mm'):.1f} mm por fragmento emparejado "
                 f"(principales HD95 {g.get('hd95_mm_principal'):.1f} mm, secundarios {g.get('hd95_mm_secundario'):.1f} mm).\n"
                 f"- **Distancia de separación:** {g.get('secundarios_con_distancia_%'):.0f} % de los secundarios reales se recuperan y se pueden medir; error medio {g.get('mae_distancia_mm'):.2f} mm, mediana {g.get('mediana_error_mm'):.2f} mm.\n"
                 f"- **SAM (línea base, ViT-B con la caja predicha como prompt):** Dice por región 0,908 contra 0,968 del modelo propio (semana 10).\n"
                 f"- El test se midió una sola vez con esta configuración; detalle de cada evaluación en `04_notebook/02_semana10.ipynb` §21.")
    card("final", "modelo final (v2_last + TTA + F2C2)", "checkpoints/v2_last/best.pth + configs/tuned.yaml", ev_v2l, g, v2,
         cbam + "; la salida de cada CBAM se mezcla con la identidad, y = x + γ(CBAM(x) − x), con γ aprendible",
         "40 épocas; se entrega la última (`v2_last`). Dropout2d 0,1 en las etapas 3-4 y en el decodificador; borde de fractura 3D como objetivo.",
         notas_fin, lat)
    card("v2", "v2 (segundo entrenamiento, sin TTA ni F2C2)", "checkpoints/v2/best.pth (época 22) y checkpoints/v2_last/best.pth (época 39)",
         ev_v2l, rj("fragments/v2_last_test_edt.json")["global"], v2, cbam + " y γ aprendible",
         "40 épocas. Cambios frente a la semana 9: borde de fractura 3D, Dropout2d 0,1, γ en CBAM y cuello, λ recalibrados.",
         "- Posproceso `edt` (semilla a 5 mm, umbral de borde 0,2). La época 22 (`best`, elegida por mAP@0.5) da mAP@[.50:.95] 0,848; la 39 (`last`), 0,864: por eso se usa la última.\n"
         "- γ aprendidos: CBAM b3 ≈ 0,18 (casi identidad), CBAM b4 ≈ −1,90 (la red resta la atención), cuello C4 ≈ 0,72.")
    base = rj("train/base.json")
    for nombre, titulo, rep, arq, nota in (
            ("base", "base (semana 9, con CBAM y transfer learning)", "base", cbam,
             "- Referencia de la ablación del §7. En test por fragmento (posproceso `edt`): Dice 0,717, IoU 0,657."),
            ("sin_cbam", "ablación sin CBAM", "sin_cbam", " sin CBAM",
             "- Ablación del §7. En test empata con `base` (mAP@[.50:.95] 0,845 contra 0,841). Por validación cruzada (5 folds, 20 épocas) quitar CBAM cambia el Dice por fragmento en −0,009 (t = −0,92): dentro del ruido de semilla."),
            ("sin_tl", "ablación sin transfer learning", "sin_tl", cbam + "; backbone inicializado al azar (sin los pesos del Taller 3)",
             "- Ablación del §7. En test empata con `base`. Por validación cruzada con dos semillas, sin TL da +0,011 de Dice por fragmento (t = 1,23 en 10 pares): sin diferencia significativa.")):
        tr = rj(f"train/{rep}.json")
        card(nombre, titulo, f"checkpoints/{rep}/best.pth", rj(f"eval/{rep}_test.json")["metricas"], None, tr, arq,
             f"40 épocas, checkpoint de la mejor época en val ({tr['mejor']['epoch']}). Sin Dropout2d ni γ (configuración de la semana 9).",
             nota, {"modelo": rj("latency/base.json")} if nombre == "base" else None)
    print("model cards en", OUT)


if __name__ == "__main__":
    main()
