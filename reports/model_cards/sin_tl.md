# Model card · ablación sin transfer learning

> **Uso exclusivamente académico.** Este modelo no es un dispositivo médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

Proyecto Integrador Corte 2, Analítica de Datos, UAO 2026-2. Checkpoint: `checkpoints/sin_tl/best.pth`.

## Arquitectura
- **Entrada:** corte axial 2.5D (z−Δ, z, z+Δ, Δ = round(2 mm / dz)), 256×256, ventana L400/W1800, recorte óseo calculado solo con la imagen, volumen reorientado a LPS por los cosenos directores del header.
- **Backbone:** FundidoraPC del curso extendida con un bloque residual por etapa (anchos 32-64-128-256), con CBAM (canal + espacial) en las etapas 3 y 4, antes de la bifurcación hacia las cabezas; backbone inicializado al azar (sin los pesos del Taller 3).
- **Cabezas (desde cero):** clasificación de presencia de SA/LI/RI; detección anchor-free con grid propio de stride 8 y NMS propia (una caja por hueso); segmentación semántica de 4 clases + probabilidad de borde de fractura.
- **Segunda etapa (sin aprendizaje):** fragmentos por semillas de la transformada de distancia del núcleo + watershed; distancia de separación con `distance_transform_edt` y el spacing del header.
- **Pérdida:** λ_cls·L_cls + λ_det·L_det + λ_seg·L_seg, con λ por norma de gradiente sobre el bloque 4 (λ_cls = 1,306, λ_det = 0,667, λ_seg = 1,027).
- **Parámetros:** 2,921,883 (cabezas incluidas).

## Datos y entrenamiento
- **Datos:** PENGWIN (MICCAI 2024), 100 CT de pelvis fracturada con máscaras de fragmentos de expertos. Partición fija por paciente (`01_data/splits.json`, balanceada por patrón de fractura): 70 train, 15 val, 15 test. El ajuste de la semana 10 usó validación cruzada de 5 folds sobre los 85 de train+val y no miró el test. Antes de ese ajuste, la época entregada de `v2` (la última en vez de la mejor en validación) y el paso a la separación por distancia se decidieron viendo resultados de test; está declarado en `04_notebook/02_semana10.ipynb` §21.
- **Entrenamiento:** AdamW, lr 3·10⁻⁴ con coseno, batch 8, precisión mixta (`torch.amp`), aumentación geométrica leve (escala 0,9-1,1, rotación ±10°, traslación 5 %) y de intensidad, **sin volteo** (distingue izquierda y derecha); semilla 42. 40 épocas, checkpoint de la mejor época en val (31). Sin Dropout2d ni γ (configuración de la semana 9).
- **Cómputo:** DGX Spark (GPU GB10), contenedor NGC PyTorch 25.09.

## Métricas en test (15 pacientes)
| Tarea | Métrica | Test | Objetivo (§5) | Cumple |
|---|---|---|---|---|
| Clasificación | F1 | 0,992 | ≥ 0,850 | sí |
| Clasificación | AUC | 0,999 | ≥ 0,850 | sí |
| Detección | IoU promedio | 0,911 | ≥ 0,650 | sí |
| Detección | mAP@0.50 | 0,975 | ≥ 0,650 | sí |
| Detección | mAP@[.50:.95] | 0,840 | ≥ 0,400 | sí |
| Segmentación por región | Dice (SA, LI, RI) | 0,971 | — | — |

## Notas
- Ablación del §7. En test empata con `base`. Por validación cruzada con dos semillas, sin TL da +0,011 de Dice por fragmento (t = 1,23 en 10 pares): sin diferencia significativa.

## Uso previsto y fuera de alcance
- **Previsto:** docencia e investigación académica sobre detección y segmentación de fragmentos pélvicos en CT; demostración en el dashboard del curso.
- **Fuera de alcance:** cualquier uso clínico, planeación quirúrgica real, diagnóstico o triaje; volúmenes que no sean CT de pelvis.

## Limitaciones conocidas
- Mismas que el modelo final en la parte de región y detección; este modelo no se evaluó por fragmento. Ver `final.md`.
