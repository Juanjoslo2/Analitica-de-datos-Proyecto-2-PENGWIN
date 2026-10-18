# Model card · modelo final (v2_last + TTA + F2C2)

> **Uso exclusivamente académico.** Este modelo no es un dispositivo médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

Proyecto Integrador Corte 2, Analítica de Datos, UAO 2026-2. Checkpoint: `checkpoints/v2_last/best.pth + configs/tuned.yaml`.

## Arquitectura
- **Entrada:** corte axial 2.5D (z−Δ, z, z+Δ, Δ = round(2 mm / dz)), 256×256, ventana L400/W1800, recorte óseo calculado solo con la imagen, volumen reorientado a LPS por los cosenos directores del header.
- **Backbone:** FundidoraPC del curso extendida con un bloque residual por etapa (anchos 32-64-128-256), con CBAM (canal + espacial) en las etapas 3 y 4, antes de la bifurcación hacia las cabezas; la salida de cada CBAM se mezcla con la identidad, y = x + γ(CBAM(x) − x), con γ aprendible.
- **Cabezas (desde cero):** clasificación de presencia de SA/LI/RI; detección anchor-free con grid propio de stride 8 y NMS propia (una caja por hueso); segmentación semántica de 4 clases + probabilidad de borde de fractura.
- **Segunda etapa (sin aprendizaje):** fragmentos por semillas de la transformada de distancia del núcleo + watershed; distancia de separación con `distance_transform_edt` y el spacing del header.
- **Pérdida:** λ_cls·L_cls + λ_det·L_det + λ_seg·L_seg, con λ por norma de gradiente sobre el bloque 4 (λ_cls = 0,704, λ_det = 0,942, λ_seg = 1,354).
- **Parámetros:** 2,921,886 (cabezas incluidas).

## Datos y entrenamiento
- **Datos:** PENGWIN (MICCAI 2024), 100 CT de pelvis fracturada con máscaras de fragmentos de expertos. Partición fija por paciente (`01_data/splits.json`, balanceada por patrón de fractura): 70 train, 15 val, 15 test. El ajuste de la semana 10 usó validación cruzada de 5 folds sobre los 85 de train+val y no miró el test. Antes de ese ajuste, la época entregada de `v2` (la última en vez de la mejor en validación) y el paso a la separación por distancia se decidieron viendo resultados de test; está declarado en `04_notebook/02_semana10.ipynb` §21.
- **Entrenamiento:** AdamW, lr 3·10⁻⁴ con coseno, batch 8, precisión mixta (`torch.amp`), aumentación geométrica leve (escala 0,9-1,1, rotación ±10°, traslación 5 %) y de intensidad, **sin volteo** (distingue izquierda y derecha); semilla 42. 40 épocas; se entrega la última (`v2_last`). Dropout2d 0,1 en las etapas 3-4 y en el decodificador; borde de fractura 3D como objetivo.
- **Cómputo:** DGX Spark (GPU GB10), contenedor NGC PyTorch 25.09.

## Métricas en test (15 pacientes)
| Tarea | Métrica | Test | Objetivo (§5) | Cumple |
|---|---|---|---|---|
| Clasificación | F1 | 0,993 | ≥ 0,850 | sí |
| Clasificación | AUC | 0,999 | ≥ 0,850 | sí |
| Detección | IoU promedio | 0,919 | ≥ 0,650 | sí |
| Detección | mAP@0.50 | 0,980 | ≥ 0,650 | sí |
| Detección | mAP@[.50:.95] | 0,864 | ≥ 0,400 | sí |
| Segmentación de fragmento | Dice | 0,768 | ≥ 0,850 | no |
| Segmentación de fragmento | IoU | 0,697 | ≥ 0,700 | no concluyente |
| Segmentación por región | Dice (SA, LI, RI) | 0,971 | — | — |

## Latencia por imagen (un corte, RTX 3060 Laptop / Intel i7)

| | CPU fp32 | GPU fp32 | GPU AMP | Volumen completo (GPU) |
|---|---|---|---|---|
| sin TTA | 66 ms | 7,7 ms | 10,3 ms | 2,8 s (401 cortes) |
| con TTA de 5 pasadas (final) | 309 ms | 33,8 ms | 37,9 ms | 7,4 s (401 cortes) |

## Notas
- **Inferencia:** TTA de 5 pasadas (escala 0,92/1,08, rotación ±8°) que promedia región y borde; detección y clasificación salen de la pasada sin transformar.
- **Posproceso:** `edt_hmax` (semilla a 5 mm del borde del núcleo, umbral de borde 0,1, rescate por h-máxima con h = 1,5 mm), elegido por validación cruzada.
- **Métricas opcionales (§5):** HD95 11,7 mm y ASSD 3,3 mm por fragmento emparejado (principales HD95 6,9 mm, secundarios 18,1 mm).
- **Distancia de separación:** 56 % de los secundarios reales se recuperan y se pueden medir; error medio 1,96 mm, mediana 0,00 mm.
- **SAM (línea base, ViT-B con la caja predicha como prompt):** Dice por región 0,908 contra 0,968 del modelo propio (semana 10).
- **Incertidumbre:** con 15 pacientes, IC95 por bootstrap de pacientes: Dice por fragmento 0,768 [0,702; 0,827], IoU por fragmento 0,697 [0,636; 0,753]. El objetivo de IoU (0,70) queda dentro del intervalo: no se puede afirmar si se cumple.
- El test se midió una sola vez con esta configuración; detalle de cada evaluación en `04_notebook/02_semana10.ipynb` §21 y auditoría de fuga en `03_semana11.ipynb` §10.

## Uso previsto y fuera de alcance
- **Previsto:** docencia e investigación académica sobre detección y segmentación de fragmentos pélvicos en CT; demostración en el dashboard del curso.
- **Fuera de alcance:** cualquier uso clínico, planeación quirúrgica real, diagnóstico o triaje; volúmenes que no sean CT de pelvis.

## Limitaciones conocidas
- **Fragmentos en contacto:** cuando un secundario toca al principal en la superficie de fractura, el modelo tiende a fusionarlos o a cortarlos por otro sitio. En los 5 peores casos de test, los 10 secundarios perdidos estaban en contacto y 8 fueron absorbidos por el principal. El error de la distancia es mayor en contacto (2,32 mm) que separados (0,75 mm).
- **Fragmentos pequeños:** los de menos de 5 cm³ tienen Dice 0,526. La cabeza de borde encuentra menos del 20 % de la superficie de fractura real.
- **Fragmentos de más:** predice entre 14 y 28 fragmentos por paciente contra 4-8 reales. La métrica por fragmento real no los castiga, pero en la reconstrucción 3D aparecen.
- **2D corte a corte:** el contexto en z se limita a ±Δ; los objetivos del enunciado son de un reto académico de 4 semanas, por debajo del estado del arte 3D del challenge.
- **Datos:** 100 casos de un solo conjunto; no se probó en otros escáneres ni protocolos. Un volumen oblicuo se remuestrea a ejes LPS, pero esa ruta no está validada con datos reales (los 100 casos están alineados).
- **Detección:** las cajas de menos de 16 px (ramas del sacro en sus extremos) tienen IoU medio 0,75 contra 0,92 global.
