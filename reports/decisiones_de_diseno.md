# Decisiones de diseño — punto de partida de la semana 9

Este documento es la referencia común del equipo para implementar el modelo. Cada decisión cita su evidencia en el EDA (`04_notebook/01_eda_pengwin.ipynb`, §n). Los parámetros concretos viven en `configs/base.yaml`; si una decisión cambia, se cambia aquí y en el YAML dentro del mismo PR.

## 1. Datos y entrada al modelo

| Decisión | Valor | Evidencia |
|---|---|---|
| Orientación | todo a LPS con `sitk.DICOMOrient` | 34/100 casos vienen en RAS (§1) |
| Recorte | recorte óseo por volumen (`compute_bone_crop`), el mismo para todos sus cortes | 1,27 vs 1,56 mm/px; nunca corta hueso (§2) |
| Tamaño | 256 × 256; spacing efectivo = lado del recorte (mm) / 256, guardado por caso | §2 |
| Contexto | 2.5D: canales (z−Δ, z, z+Δ), Δ = round(2 mm / dz) | Δ = 2 (78 casos) / 3 (22) (§2) |
| Intensidad | recorte a [−1024, 3071] HU; ventana L400 / W1800 → [0, 1] | 99,6 % del hueso sin saturar (§3) |
| Muestreo | cortes con hueso + ~12 % de cortes vacíos | coxales en 96 % de los cortes con hueso (§8) |
| Partición | `01_data/splits.json` v2 (por paciente, SMD máx 0,155, `sha256`) | §9 |
| Aumentación | **sin flip horizontal** (o con intercambio de etiquetas izq ↔ der) | la lateralidad depende de la posición |

## 2. Detección (cabeza 2)

- **Una caja por región anatómica y corte, no por fragmento.** El enunciado pide "bounding boxes de cada región anatómica".
- **La caja GT es la envolvente de todas las islas de la región en ese corte.** Una región aparece partida en islas en 28–42 % de los cortes (§8). Como el GT y la predicción usan la misma definición, el espacio vacío entre islas no penaliza el IoU. Las cajas no se fusionan entre cortes: la reconstrucción 3D se hace con las máscaras (§3 de este documento).
- Grid propio anchor-free de **stride 8** (32 × 32 celdas): el 39 % de las cajas del sacro mide < 16 px de lado (§8).
- Las cajas GT con lado < 4 px se marcan como "ignorar": no cuentan como positivos ni como negativos.
- NMS propio y vectorizado por clase; se conserva como máximo 1 caja por clase y corte.

## 3. Segmentación de instancias y contacto (cabeza 3)

- Salidas: semántica de 4 clases (fondo, SA, izq., der.) + **mapa de borde de fractura**.
  - El 88,7 % de los fragmentos secundarios toca al principal (§7).
  - El borde es 1 de cada ~446 píxeles de hueso: el objetivo se dilata 2 px y se entrena con BCE (`pos_weight` ≈ 21) + Dice.
- Instancias = (región − borde) → **componentes conexas 3D tras apilar los cortes** → watershed para devolver los píxeles de borde.
  - Las componentes < 0,1 cm³ se reasignan a la instancia vecina.
  - Un mismo fragmento aparece partido en islas 2D en 18–40 % de los cortes, así que las componentes 2D no equivalen a instancias (§8).
- El fragmento principal de la predicción es la componente 3D de mayor volumen en la región. En el GT coincide con la etiqueta 1/11/21 en 300/300 regiones (§1).
- **Modo de falla principal:** si dos fragmentos que se tocan se fusionan, el secundario **desaparece**: no hay distancia que medir (no es que se mida 0 mm). Por eso se reportan:
  - **fragmentos recuperados** (recall de instancias con IoU ≥ 0,5 frente al GT);
  - Dice por fragmento **estratificado por volumen** (< 5, 5–20, > 20 cm³): el 14,5 % de los secundarios mide < 5 cm³ (§6);
  - error de distancia (pred − GT) **por separado** para fragmentos en contacto y separados.

## 4. Función de pérdida

- **Tres términos, como pide el enunciado:** L = λ_cls·L_cls + λ_det·L_det + λ_seg·L_seg.
  - **La distancia de separación no es un término de pérdida**: es una medición posterior sobre las máscaras.
  - L_seg = CE (pesos ∝ 1/√frecuencia: 1 / 10,8 / 8,7 / 8,7) + Dice + borde.
- **Calibración de λ:**
  1. Entrenar ~200 iteraciones con λ = 1 y registrar G_k = ‖∇_θs L_k‖₂ sobre la última capa compartida (la salida del CBAM).
  2. Fijar λ_k = mean(G) / G_k, normalizado para que Σλ = 3.
  3. Reportar la tabla de G_k y λ_k, y una prueba de sensibilidad multiplicando cada λ por 0,5 y por 2.

## 5. Prueba de overfit (entregable de la semana 9)

- **8 cortes fijos de train** (que incluyan sacro fracturado y al menos un fragmento separado), sin aumentación, semilla fija, hasta 500 pasos.
- **Criterios de aceptación:**
  - la pérdida cae ≥ 95 %;
  - Dice ≥ 0,95;
  - mAP@0.5 ≥ 0,95 sobre esos mismos 8 cortes.
- **Si falla**, revisar en este orden:
  1. forma de los tensores;
  2. asignación de celdas del grid;
  3. decodificación de cajas;
  4. IoU;
  5. NMS;
  6. alineación máscara ↔ imagen tras el recorte;
  7. pérdidas;
  8. que el gradiente llegue a las tres cabezas.
- Ubicación: `06_tests/test_overfit_batch.py`, con la marca `slow` (no corre en CI).

## 6. Política de pruebas

- Todo módulo nuevo entra con tests sobre **phantoms sintéticos** con respuesta conocida (IoU, NMS, decodificación del grid, EDT con spacing anisotrópico, Dice). No deben depender del dataset (~32 GB).
- Los tests que necesitan los `.mha` llevan la marca `data` y se omiten solos si no están.
- GitHub Actions (`.github/workflows/tests.yml`) corre `pytest` en cada PR. Hoy: 50 tests sin datos + 5 con datos (incluida la prueba de overfit, `slow`).

## 7. Revisión externa de la semana 8 — qué se tomó y qué no

| Observación | Resolución |
|---|---|
| Sin PRs y commits directos a `main` | Flujo de ramas + PR con 1 aprobación + CI obligatorio (ver README) |
| Islas 2D: ¿una caja o varias? | Una por región, envolvente de las islas (§2); instancias en 3D (§3) |
| Fragmentos en contacto se fusionan | Salida de borde + métrica de fragmentos recuperados + análisis por grupo (§3) |
| Tests dependientes del dataset | Ya había 8 tests sin datos; ahora son 16, y el CI corre sin dataset (§6) |
| Prueba de overfit | Protocolo y criterios en §5 |
| Calibrar λ | Método en §4. **No** se añade un término de pérdida para la distancia |
| Cifras citadas (86 %, 33 %/15 %, 8 GB) | Eran preliminares o erróneas: usar 88,7 %, 28–42 % / 18–40 % y ~32 GB |

## 8. Backbone compartido, CBAM y transfer learning (semana 9)

Implementado en `03_src/pengwin/models/` (`backbone.py`, `cbam.py`, `heads.py`, `pengwin_net.py`); parámetros en `configs/base.yaml → model`.

```
entrada (B, 3, 256, 256) = cortes (z−Δ, z, z+Δ)
  FundidoraBackbone: 4 bloques [conv3×3 → BN → ReLU → residual → MaxPool (→ CBAM en 3 y 4)]
     C1 32 ch s2 · C2 64 ch s4 · C3 128 ch s8 (CBAM) · C4 256 ch s16 (CBAM)
  ── bifurcación ──
  C4 → pooling global → Linear            → presencia SA / izq. / der.   (clasificación)
  C3 + up(C4) → P3 (s8, 32×32) → grid     → puntaje + (l, t, r, b) por región (detección)
  P3 + C2 + C1 → decodificador → 256×256  → semántica 4 clases + borde      (segmentación)
```

| Decisión | Valor | Por qué |
|---|---|---|
| Base | Los 4 bloques de FundidoraPC del curso, con los mismos canales (32 → 256), kernels y pooling | Es la "extensión propia del backbone del curso" que pide el enunciado §4.1, y conserva la forma de sus pesos |
| Extensión 1: residual | Tras la conv de cada bloque, un bloque básico conv-BN-ReLU-conv-BN + identidad. La última BN arranca en 0, así que el bloque empieza como identidad | Da más profundidad sin degradar el gradiente. Al cargar pesos, la red arranca dando la misma salida que la FundidoraPC original (lo verifica `test_course_weights_load_and_reproduce_course_output`) |
| Extensión 2: multiescala | Se exponen C1–C4 en lugar del vector del pooling global | La detección necesita stride 8 [EDA §8] y la segmentación necesita resolución completa |
| CBAM | Canal → espacial, reducción 16, kernel 7, en C3 y C4. Con `cbam: false` se reemplaza por identidad y el resto de la red no cambia | Queda "en el backbone, antes de la bifurcación hacia las tres cabezas": todas las cabezas leen C3/C4 o mapas derivados de ellos. La ablación solo resta los 2 CBAM (`test_cbam_ablation_changes_only_cbam`) |
| Transfer learning | Solo en el backbone: las conv+BN de la FundidoraPC entrenada en el **Taller 3** (RPN híbrido, MVTec Screws, 120 épocas) → `checkpoints/fundidora_taller3.pth`. Residuales, CBAM, cuello y cabezas arrancan desde cero | El enunciado permite "la fundidora de pc en datasets previamente usados, si almacenaron los pesos". El cargador también acepta el formato de S6 (`conv1…bn4`, pliega el sesgo de la conv en la media de la BN) y primeras capas de 1 canal |
| Cuello | P3 = conv(lat(C3) + up×2(lat(C4))), 128 canales | La detección a stride 8 recibe así el contexto del nivel más profundo |
| Detección | Anchor-free, 12 canales de caja (4 por región) + 3 de puntaje. Positivos: centro de la celda dentro de la caja y a ≤ 1,5 celdas de su centro; las cajas diminutas reciben siempre ≥ 1 positivo. Pérdida focal (α 0,25, γ 2) + (1 − GIoU) | Cajas por clase porque las envolventes del sacro y de cada coxal se solapan en la articulación sacroilíaca; ~1 % de las celdas son positivas → focal |
| NMS | Propia (`detection/nms.py`), voraz, por clase, `max_per_class = 1` | Enunciado §3.1; una región aparece a lo sumo una vez por corte [DD §2] |
| Tamaño | 2,92 M parámetros (backbone 1,97 M) | Con batch 8 a 256 px y AMP entra en una GPU de 6 GB |
| AMP | `torch.amp.autocast("cuda")` + `torch.amp.GradScaler`, la API vigente de `torch.cuda.amp` (los nombres viejos dan aviso de obsolescencia en torch 2.10). Las pérdidas se calculan en float32 | Enunciado §4.3 |

**Caché de cortes** (`data/slice_cache.py`, `scripts/build_slice_cache.py`):
- Cada caso se lee una vez y se guarda con recorte óseo, ventana L400/W1800 cuantizada a uint8 (≈ 7 HU por nivel) y etiqueta con vecino más cercano, a 256 × 256.
- Pesa ~45 MB por caso y se lee con `mmap`.
- `meta.json` guarda el recorte y el spacing para volver al volumen nativo y medir en mm (`model_to_native_xy`).

**Prueba de overfit (§5) — APROBADA en el primer intento** (`scripts/overfit_batch.py`, caso 024, 8 cortes con ≥ 2 fragmentos en una región, 500 pasos, AdamW 1e-3, sin aumentación):

| Criterio | Umbral | Resultado |
|---|---|---|
| Caída de la pérdida | ≥ 95 % | 98,8 % |
| Dice (hueso) | ≥ 0,95 | 0,992 |
| mAP@0.5 | ≥ 0,95 | 1,00 |
| Gradiente en backbone y 3 cabezas | sí | sí |

Reporte y curvas: `reports/overfit/overfit_base.json|png`.

## 9. Ablación: CBAM y transfer learning (semana 9)

Tres modelos de 40 épocas cada uno, entrenados en ANTON (DGX Spark). Comparten datos, semilla, λ (`reports/lambdas.json`) y configuración; cada ablación cambia una sola cosa. El mejor checkpoint se elige en val (promedio de mAP@0.5, Dice y F1) y se evalúa en **test** (15 pacientes nunca vistos, 4279 cortes) con `scripts/evaluate.py`.

| Test | Base (CBAM + TL) | Sin CBAM | Sin TL (desde cero) | Objetivo §5 |
|---|---|---|---|---|
| F1 clasificación | 0,993 | 0,992 | 0,992 | ≥ 0,85 |
| AUC | 0,999 | 0,999 | 0,999 | ≥ 0,85 |
| mAP@0.5 | 0,977 | 0,980 | 0,976 | ≥ 0,65 |
| mAP@[.5:.95] | 0,841 | 0,845 | 0,840 | ≥ 0,40 |
| IoU promedio de caja | 0,911 | 0,912 | 0,911 | ≥ 0,65 |
| AP@0.5 sacro | 0,941 | 0,948 | 0,936 | — |
| Dice por región | 0,968 | 0,970 | 0,971 | ≥ 0,85* |
| Dice sacro | 0,954 | 0,960 | 0,961 | — |

\* El objetivo del enunciado es Dice **por fragmento**, que llega en la semana 10. Este es el Dice por región anatómica.

Detalle: `reports/eval/*_test.json`, `reports/train/*.json`; curvas en `runs/<nombre>/history.csv` (no versionado).

**Lectura:**
- Los tres modelos quedan prácticamente empatados (diferencias ≤ 0,01). Ni CBAM ni el transfer learning desde MVTec mejoran de forma medible el resultado final.
- Con una sola semilla por modelo, diferencias de este tamaño no se pueden separar del azar. Afirmar una ventaja exigiría repetir con varias semillas.
- El transfer learning **sí acelera la convergencia**: en las épocas 0–2, el mAP@[.5:.95] en val fue 0,664 / 0,706 / 0,730 con TL contra 0,650 / 0,672 / 0,679 desde cero. Con 19 000 cortes, el modelo desde cero alcanza al otro antes de la época 40.
- **Interpretación:** el hueso es la estructura más brillante del corte en la ventana L400/W1800, así que la atención espacial tiene poco que "señalar". Y los pesos de MVTec solo aportan detectores genéricos de bordes, que la red aprende sola con este volumen de datos.

## 10. Semana 10: qué aporta cada componente, dropout espacial y fragmentos

### 10.1 ¿Qué aporta cada parte? (`scripts/component_contribution.py`)

Sobre el modelo base entrenado (val, 6 casos, 1575 cortes) se midieron tres cosas: el γ aprendido, cuánto cambia cada componente las features y un **knockout** (apagar el componente en inferencia y medir la caída).

| Componente apagado | Δ mAP@[.5:.95] | Δ Dice | Δ Dice sacro |
|---|---|---|---|
| residual bloque 1 | −0,042 | −0,006 | −0,007 |
| residual bloque 2 | −0,032 | −0,003 | −0,005 |
| residual bloque 3 | −0,021 | 0,000 | +0,002 |
| residual bloque 4 | −0,528 | −0,163 | −0,116 |
| CBAM bloque 3 | −0,142 | −0,081 | −0,190 |
| CBAM bloque 4 | −0,106 | −0,098 | −0,251 |
| contexto C4 en P3 | −0,802 | −0,496 | −0,543 |
| contexto 2.5D (vecinos = corte central) | −0,045 | −0,022 | −0,020 |

**Knockout ≠ ablación.** El knockout mide cuánto **depende** la red entrenada de un componente. La ablación mide cuánto **aporta** si se reentrena sin él. El CBAM es el mejor ejemplo: apagarlo hunde el sacro (−0,25), pero el modelo entrenado sin CBAM llega al mismo resultado (§9). La red se apoya en lo que tiene, y si no lo tiene, compensa.

**Por qué el γ.** La medida "cambio relativo" (‖y − x‖/‖x‖) exagera el CBAM: su sigmoide reduce todo el mapa a ~la mitad, y eso parece cambio aunque no lo sea. Por eso, desde `v2`:
- el CBAM se mezcla como y = x + γ·(CBAM(x) − x), con γ que arranca en 0;
- el contexto profundo entra en P3 como C3 + γ·up(C4), con γ que arranca en 1.

Los γ finales dicen directamente cuánto decidió usar la red cada componente (`PengwinNet.gammas()`).

### 10.2 Dropout espacial

`Dropout2d` (apaga canales completos), p = 0,1:
- a la salida de los bloques 3 y 4 (128 y 256 canales), después del CBAM;
- en el decodificador de segmentación a stride 4 (64 canales).

Por qué así:
- **Dropout2d y no dropout por píxel:** los píxeles vecinos están correlacionados, así que apagar píxeles sueltos casi no regulariza.
- **No en los bloques 1–2 ni en el decodificador a stride 2/1:** hay pocos canales y llevan el detalle fino del borde de fractura.
- **No en la torre de detección:** la regresión de cajas es sensible y su mAP@[.5:.95] no mostraba sobreajuste.
- **Motivación:** la pérdida de validación se aplanaba hacia la época 17 y la del borde subía (BCE de val 0,025 → 0,089).

### 10.3 Separación de fragmentos: el borde tiene que ser 3D

Pipeline (`inference/volume.py`, `postprocess/instances.py`):
1. El modelo predice región y borde corte a corte, y se apila (Z, 256, 256).
2. Por región: núcleo = región sin borde.
3. Componentes conexas 3D del núcleo. Las menores de 0,1 cm³ no son semilla.
4. Watershed sobre el borde, para devolver los píxeles de borde a su fragmento.
5. Se ordena por volumen: el mayor es el principal.
6. Se vuelve a la grilla nativa del .mha (`to_native`).

**Primer resultado con el modelo base:** el principal de cada hueso queda bien (Dice 0,90), pero **ningún secundario se separaba**. Hubo dos causas:

1. **La cabeza de borde sobreajustó.** Pérdida Dice del borde en train 0,10 contra 0,39 en val; en test predice P(borde) ≈ 0 sobre bordes reales.
2. **Diseño:** el borde objetivo se calculaba en 2D, dentro de cada corte, y no veía los contactos **entre cortes**. Con el borde real (oráculo) en test, sobre la etiqueta en la grilla del modelo:

| Borde usado para separar (oráculo GT) | Secundarios recuperados (IoU ≥ 0,5) | Dice secundario |
|---|---|---|
| 2D, dilatación 2 (semana 9) | 35 % | 0,34 |
| 3D, dilatación 1 | 38 % | 0,37 |
| **3D, dilatación 2** | **71 %** | **0,69** |

**Cambios para `v2`:**
- El objetivo de borde ahora es 3D (6-vecindad incluyendo z, dilatación 2 dentro del hueso), precalculado en el caché como `edge.npy` (`build_slice_cache.py --edges-only`). El modelo sigue siendo 2D: aprende a marcar también los contactos con los cortes vecinos, que ve por la entrada 2.5D.
- `edge_pos_weight` = 9. Antes era 21, calculado como √(1:446) con el borde **sin** dilatar, pero lo que se entrenaba era el borde dilatado. La regla √(hueso:borde) aplicada al objetivo real (1:83) da 9.
- **λ recalibrados** con la pérdida nueva: cls 0,70, det 0,94, seg 1,35 (antes 1,31 / 0,67 / 1,03; los anteriores quedan en `reports/lambdas_semana9.json`).

### 10.4 Latencia (`scripts/latency.py`, batch 1, modelo + decodificación + NMS)

| Dispositivo | Media | p95 |
|---|---|---|
| CPU Intel 11.ª gen (Tiger Lake-H), 8 hilos, fp32 | 102 ms/corte | 114 ms |
| GPU RTX 3060 Laptop, fp32 | 10 ms/corte | 13 ms |
| GPU RTX 3060 Laptop, AMP | 13 ms/corte | 16 ms |

Un volumen completo (401 cortes, batch 16) tarda 3,4 s en GPU. A batch 1, AMP es un poco más lento que fp32 por el costo de convertir tipos; el AMP rinde en entrenamiento.

### 10.5 SAM zero-shot como línea base (`scripts/sam_baseline.py`)

SAM ViT-B (`sam_vit_b_01ec64.pth`, sha256 `ec2df627…`) recibe el mismo corte que ve el modelo y, como prompt, la caja que predijo nuestro detector para cada región. Test: 15 pacientes, 3 765 cortes con hueso. Se compara contra el GT por región, sumando todos los píxeles del conjunto.

| Método | Dice | IoU | Dice SA | Dice LI | Dice RI |
|---|---|---|---|---|---|
| **Modelo propio (base)** | **0,968** | **0,938** | **0,954** | **0,975** | **0,974** |
| SAM + caja predicha | 0,908 | 0,832 | 0,863 | 0,927 | 0,932 |
| SAM + caja GT (cota de SAM) | 0,908 | 0,832 | 0,866 | 0,925 | 0,931 |

- El modelo específico le gana a SAM por 6 puntos de Dice y 11 de IoU. La diferencia más grande está en el sacro (+9), cuya forma irregular con agujeros (forámenes) SAM no sigue bien.
- SAM rinde igual con nuestras cajas que con las reales: el detector no es lo que limita a SAM.
- Costo: SAM tarda ~417 ms por corte en la RTX 3060, contra ~12 ms del modelo propio.

### 10.6 Fragmentos con el modelo base (antes de `v2`)

`scripts/evaluate_fragments.py`, test, 84 fragmentos GT (45 principales y 39 secundarios), medido en la grilla nativa del .mha:

| Métrica | Valor |
|---|---|
| Dice por fragmento (todos) | 0,545 |
| IoU por fragmento | 0,505 |
| Dice principal | 0,921 |
| Dice secundario | 0,111 |
| Secundarios recuperados (IoU ≥ 0,5) | 12,8 % (5 de 39) |
| Error de distancia en los recuperados (MAE) | 1,06 mm |

Esta es la línea de partida: el modelo base no separa los secundarios, por las causas descritas en §10.3. `v2` se evalúa con el mismo script.

### 10.7 `v2` y la separación por distancia

**`v2`** = borde 3D + `pos_weight` 9 + λ recalibrados + γ (CBAM y cuello) + dropout espacial. Mismos 40 épocas en ANTON. En test, con las métricas de la semana 9, la última época mejora al modelo base en las cajas finas:

| Test | base | v2 (época 22, "mejor") | v2 (época 39) |
|---|---|---|---|
| mAP@0.5 | 0,977 | 0,979 | 0,980 |
| mAP@[.5:.95] | 0,841 | 0,848 | **0,864** |
| IoU de caja | 0,911 | 0,914 | **0,919** |
| Dice por región | 0,968 | 0,970 | **0,971** |
| Dice sacro | 0,954 | 0,961 | **0,961** |

El criterio de selección elegía la época 22 porque usaba mAP@0.5, que se satura en ~0,98. Desde ahora usa mAP@[.5:.95] (`engine.selection_score`).

**Pero `v2` sola no separó mejor los fragmentos** (10 % de secundarios). El diagnóstico en test, cruzando región y borde predichos con los reales:

| Región \ borde usados para separar | Secundarios recuperados |
|---|---|
| predicha + predicho | 8 % |
| predicha + real | 11 % |
| real + predicho | 12 % |
| real + real | 71 % |

Hay dos cuellos de botella:
- el borde predicho solo encuentra el 15 % del borde real;
- la región predicha "rellena" las grietas, y eso reconecta los fragmentos aunque el borde fuera perfecto.

**Solución en el posproceso (ajustada en val, sin reentrenar):** semillas por distancia (`instance_method: edt`).
- d = `distance_transform_edt` del núcleo (región sin borde).
- Las semillas son las zonas con d > h. Los cuellos y las grietas nunca son profundos, así que separan aunque la red los haya rellenado.
- Watershed sobre −d + 5·borde.

Barrido en val (v2, 15 casos):

| Método | h (mm) | Secundarios recuperados | Dice fragmento |
|---|---|---|---|
| solo borde, umbral 0,05–0,5 | — | 1,8 % | 0,531 |
| EDT, umbral 0,2 | 2 | 31,0 % | 0,638 |
| EDT, umbral 0,2 | 3 | 46,4 % | 0,705 |
| EDT, umbral 0,2 | 4 | 61,3 % | 0,765 |
| **EDT, umbral 0,2** | **5** | **67,3 %** | **0,790** |
| EDT, umbral 0,2 | 6 | 59,5 % | 0,764 |
| EDT, umbral 0,2 | 8 | 33,3 % | 0,622 |

**Resultado en test** (`evaluate_fragments.py`, grilla nativa, 84 fragmentos):

| Por fragmento | base, posproceso semana 10 inicial | base + EDT | **v2 + EDT** | Objetivo §5 |
|---|---|---|---|---|
| Dice | 0,545 | 0,717 | **0,725** | ≥ 0,85 |
| IoU | 0,505 | 0,657 | **0,671** | ≥ 0,70 |
| Dice principal | 0,921 | 0,920 | **0,930** | — |
| Dice secundario | 0,111 | 0,483 | **0,489** | — |
| Secundarios recuperados | 12,8 % | 51,3 % | **51,3 %** | — |
| Dice secundarios < 5 cm³ (7) | 0,22 | 0,00 | 0,00 | — |
| Error de distancia (MAE, recuperados) | 1,1 mm (5) | 0,3 mm (20) | 2,2 mm (20) | — |

Lectura honesta:
- Los objetivos por fragmento **todavía no se cumplen** (Dice 0,725 contra 0,85). Los principales sí (0,93); lo que baja el promedio son los secundarios.
- Los menores de 5 cm³ no reciben semilla con h = 5 mm. Es el precio del umbral que maximiza el total.
- El error de distancia se calcula sobre apenas 20 secundarios recuperados, y cambia mucho entre modelos, porque unos pocos fragmentos mal delimitados lo dominan.

### 10.8 ¿Qué aporta cada parte en `v2`? (γ + knockout, val)

| Componente | γ aprendido | Δ mAP@[.5:.95] al apagarlo | Δ Dice sacro |
|---|---|---|---|
| residual b1 / b2 / b3 | 0,25 / 0,22 / 0,21 | −0,042 / −0,027 / −0,033 | ≈ 0 |
| residual b4 | 0,32 | −0,513 | −0,248 |
| CBAM b3 | **0,18** | −0,002 | −0,001 |
| CBAM b4 | **−1,90** | −0,068 | −0,020 |
| contexto C4 en P3 | 0,72 | −0,793 | −0,474 |
| contexto 2.5D | — | −0,043 | −0,021 |

- Con γ, la red puede **ignorar** un componente, y el knockout deja de exagerar.
- La red casi no usa el CBAM del bloque 3: γ = 0,18, y apagarlo no cambia nada.
- El CBAM del bloque 4 sí se usa, con γ negativo: y = 2,9·x − 1,9·CBAM(x), es decir, resalta lo que la atención atenuaría (un realce de contraste).
- Lo indispensable es el bloque residual 4 y el contexto profundo que entra al cuello.

## 11. Ajuste de hiperparámetros (semana 11)

Campaña de ajuste objetivo en ANTON (DGX Spark GB10). Detalle completo y CSV/JSON por experimento en
`reports/tuning/` (`RESUMEN.md`, `PROGRESO.md`).

**Protocolo.** Selección **solo** por validación cruzada por paciente: k = 5 sobre los 85 casos de
train+val (`reports/tuning/cv_folds.json`, `scripts/cv_folds.py`, estratificado por nº de
secundarios). **El test de 15 pacientes no se usó para elegir nada** y se miró una sola vez al final.
Regla fijada antes de ver resultados: si |Δ de la media entre folds| < la desviación entre folds, no
hay ganador y se queda la opción actual. λ recalibrados con `scripts/calibrate_lambdas.py` en cada
candidato y cada fold; se evaluó siempre la última época, igual que `v2_last`.

**Escala de ruido** (`c00_base` @20 épocas, 5 folds, media ± desv. entre folds):

| Métrica | Media | Desv. entre folds |
|---|---|---|
| Dice por fragmento | 0,7429 | **0,0449** |
| mAP@0.50 | 0,9784 | 0,0034 |
| mAP@[.50:.95] | 0,8439 | 0,0096 |
| IoU de caja | 0,9111 | 0,0046 |
| Dice de región | 0,9655 | 0,0026 |
| F1 / AUC | 0,9905 / 0,9987 | 0,0011 / 0,0004 |

La desviación del Dice por fragmento (0,045) es la barra de toda la campaña: la métrica la domina si
los secundarios se recuperan o no, y eso varía mucho por paciente.

### 11.1 Posproceso: los cinco parámetros se quedan como estaban

Tres etapas sobre los 5 folds (grilla gruesa → refinamiento), sin reentrenar, sobre las predicciones
fuera de fold (`scripts/cv_predict.py`).

| Etapa | Mejor | Segundo | Actual | Δ mejor−actual | ¿Gana? |
|---|---|---|---|---|---|
| `seed_depth_mm` × `edge_threshold` | 5 / 0,1 → 0,7453 | 5 / 0,3 → 0,7430 | **5 / 0,2 → 0,7429** | +0,0024 | no (0,05 σ) |
| `seed_depth_mm` × `edge_weight` | 5,5 → 0,7474 | 5 / 20 → 0,7430 | **5 / 5 → 0,7430** | +0,0044 | no (0,10 σ) |
| `seed_min_cm3` | 0,002 → 0,7441 | 0,01 → 0,7430 | **0,02 → 0,7430** | +0,0011 | no (0,02 σ) |

En las etapas 2 y 3 el primer puesto por Dice **pierde en los dos desempates** (IoU y % de
secundarios recuperados): no hay señal. Lo que sí quedó medido:

- **`instance_method: edt` vale +0,245 de Dice por fragmento** frente a `edge` (0,7430 contra 0,4983)
  y 59,5 % contra 4,9 % de secundarios recuperados. Confirma por CV lo que §10.7 vio en val, y es la
  mejora más grande del proyecto en esta métrica.
- `edge_threshold` es irrelevante (0,0055 entre 0,1 y 0,5) y **`edge_weight` es inerte** (≤ 0,0002
  entre 0 y 20): el watershed no puede rescatar una cabeza de borde floja, toda la separación la hace
  la geometría (−d).

### 11.2 Dónde está el techo, y de quién es la culpa

Alimentando el mismo posproceso con región y/o borde del *ground truth* (5 folds):

| Región | Borde | Dice fragmento | `seed_depth_mm` óptimo | rec. secundarios |
|---|---|---|---|---|
| predicha | predicho | 0,7430 ± 0,0449 | 5 | 59,5 % |
| **GT** | predicho | 0,7697 ± 0,0368 (+0,027) | 5 | 59,6 % |
| predicha | **GT** | **0,9024 ± 0,0110 (+0,159)** | **3** | **90,6 %** |
| GT | GT | 0,9952 ± 0,0014 (+0,252) | 1,5 | 100 % |

- **Arreglar el borde vale 6 veces más que arreglar la región.** "La región rellena las grietas" es
  real pero menor: con región perfecta la recuperación de secundarios no se mueve (59,6 % vs 59,5 %).
- El método `edge` tiene **techo 0,847** incluso con borde perfecto, por debajo del objetivo 0,85 del
  enunciado: otra razón para `edt`.
- `seed_depth_mm` **no es un umbral de ruido, es un detector geométrico de grietas que compensa al
  borde**: su óptimo pasa de 5 (borde predicho) a 3 (borde GT) a 1,5 (todo GT).
- **Modo de fallo del borde: cobertura, no calibración.** Marca ~20 % de la superficie de fractura
  (recall 0,193 a umbral 0,2) y bajar el umbral a 0,05 solo lo lleva a 0,213: el 80 % que falta está
  en ≈ 0. Esto explica las tres anomalías del §11.1.

### 11.3 Entrenamiento: ningún candidato supera el ruido

- **Cribado:** 11 candidatos a 12 épocas en el fold 0 (pos_weight del borde, dilatación del borde,
  sobremuestreo, lr, dropout espacial, aumentación fuerte, `backbone_lr_factor`). Suelo de ruido con
  4 semillas: σ = 0,0097.
  - Ninguno lo supera con margen.
  - `c10` (`backbone_lr_factor` 0,3) **empeora 4,3σ**: valida el afinado completo del backbone.
- **Confirmación (P4, 5 folds @20, emparejado por fold):**
  - `c05` (sobremuestreo ×3 de cortes con secundarios): +0,0093 = 1,61σ, 3/5 folds. No se adopta.
  - El control de semilla da Δ −0,0000 con σ = 0,0130.
- **Resultado en test:** sin configuración ganadora, sigue el de `v2_last` (0,725 / 0,671).
- El cuello de botella no es un hiperparámetro, es la cobertura de la cabeza de borde (§11.2).

## 12. Fase 2: arquitectura y transfer learning

Detalle en `reports/tuning/fase2/` (`RESUMEN_FASE2.md`, `REGISTRO.md` con el pre-registro,
`EXPERIMENTOS.csv`). Comparaciones emparejadas por fold contra `c00_base`@20 (σ = 0,0130) y réplica
de semilla antes de adoptar cualquier cosa.

| experimento | Dice frag | Δ | prueba | decisión |
|---|---|---|---|---|
| sin transfer learning (2 semillas, 10 pares) | 0,759 / 0,752 | +0,0105 | t = 1,19, 6/10 | sin efecto |
| núcleo/borde de 3 clases (`core3`) | 0,673 | −0,072 | — | rechazado |
| regresión de distancia al borde (`dist`) | 0,504 | −0,241 | — | rechazado |
| sin CBAM (ablación en CV) | 0,737 | −0,0088 | t = −0,92, 1/5 | sin efecto medible |
| posproceso h-máxima | 0,760 | +0,0147 | principal −0,043 | rechazado |
| posproceso híbrido `edt` + h-máxima | 0,772 / 0,764 | +0,027 / +0,020 | principal −0,0093 / **−0,0112** | no confirma la réplica |

- **Transfer learning del Taller 3:** ni ayuda ni estorba en fragmentos. El efecto aparente de la
  primera semilla lo aportaba un solo fold, y ese mismo fold se repite en la segunda semilla:
  heterogeneidad entre pacientes. Es la ablación con/sin TL que pide el enunciado, ahora en CV.
- **Representaciones nuevas del borde:** las dos fallan por la representación, no por el
  entrenamiento. Evaluados con el `edt` viejo, sus mismos checkpoints quedan iguales o algo peores
  que el base. El oráculo explica por qué: incluso con núcleo perfecto hace falta erosión
  geométrica, porque una superficie de fractura de 2 px no desconecta del todo dos fragmentos que
  se tocan en 3D.
- **Híbrido `edt_hmax`:** recupera secundarios (Dice secundario +0,05/+0,07 en 5/5 folds en los dos
  OOF), pero cuesta ≈ −0,010 de Dice principal y +0,5-0,7 mm de MAE de distancia. Falló el
  guardarraíl pre-registrado en la réplica y no se adopta. Queda implementado
  (`instance_method: edt_hmax`) como opción documentada.
- **Arquitectura final = la de la semana 10.** Las métricas de fragmento no alcanzan el objetivo
  (test 0,725 / 0,671 contra 0,85 / 0,70). Clasificación y detección siguen cumpliendo con margen.
