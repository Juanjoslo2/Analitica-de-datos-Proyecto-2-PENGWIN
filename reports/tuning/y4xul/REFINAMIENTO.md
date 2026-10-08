# Refinamiento por región: segunda pasada por hueso a resolución nativa

Fecha: 2026-10-07. Rama `y4xul`. Ejecución en SparkLab (DGX Spark, GPU NVIDIA GB10).
Antecedentes: `RESUMEN.md` (cribado y confirmación) y `REGISTRO.md` (pre-registros), en esta carpeta.

**Estado: implementado, probado y evaluado en validación cruzada (§6).** Mejora el control en
+0,037 de Dice por fragmento (t = 3,13, 5/5 folds) y deja el IoU en 0,725, pero el Dice por
fragmento queda en 0,789 contra el objetivo de 0,85. El test no se ha evaluado.

## 1. Por qué

Lo que dejan las rondas anteriores:

| hallazgo | dónde se midió |
|---|---|
| El cuello de botella es la separación de fragmentos secundarios: Dice principal 0,92-0,94, secundario 0,57-0,63 | confirmación en 5 folds |
| Con el borde real el Dice por fragmento subiría a 0,90; con el papel real, a 0,93 | oráculos (Fase 2 y `oracle_role.py`) |
| Cambiar la **salida** no basta: `core3`, `dist` y el Dice ponderado dejan el modelo igual | Fase 2 y segunda ronda |
| Reponderar la pérdida no mueve los fragmentos (5 intentos) | Fases 1-2 y segunda ronda |
| Llevar características de resolución completa al decodificador sí ayuda, poco (+0,013, t = 4,23) | confirmación |

Todo apunta a la **entrada**: a 256 px el píxel mide 1,2-1,6 mm y una grieta sin desplazamiento
ocupa 1-2 px. El CT original tiene el doble de resolución en el plano (0,6-0,8 mm) y hasta ahora
se descartaba al construir el caché.

## 2. Qué se implementó

Un **único modelo que se ejecuta dos veces**, sobre el mismo backbone, el mismo CBAM y las mismas
tres cabezas. No hay una segunda red.

```
pasada 1   corte completo, 256 px (1,2-1,6 mm/px)         4.º canal de entrada = 0
           → clasificación, cajas (grid + NMS propios), región por hueso
pasada 2   por cada hueso que encontró la pasada 1:
           recorte de 256 px del caché de 512 px (0,6-0,7 mm/px) centrado en ese hueso
           4.º canal de entrada = máscara de ESE hueso según la pasada 1
           → papel (principal / secundario) y borde, con el doble de resolución
vuelta     se promedia por bloque a la grilla de 256 y sustituye al papel y al borde de la
           pasada 1 solo dentro del hueso; el posproceso y las métricas no cambian
```

Frente al enunciado: sigue siendo 2D corte a corte, un backbone propio con CBAM antes de la
bifurcación, tres cabezas, pérdida de tres términos, grid y NMS propios y ningún framework de alto
nivel. La resolución del recorte (256 px) queda dentro de la recomendada. Es la "lógica de dos
etapas: primero las regiones anatómicas y después los fragmentos dentro de cada región" del §3.2.

| pieza | archivo |
|---|---|
| 4.º canal de entrada (`model.in_channels: 4`); partir de un checkpoint de 3 canales con el canal nuevo en cero | `models/pengwin_net.py` (`load_expanding_input`) |
| Recortes por hueso mezclados con los cortes completos | `data/two_pass.py` (`TwoPassSlices`) |
| Caché de alta resolución (512 px, 16 GB) | `scripts/build_slice_cache.py --image-size 512 --no-extras` |
| Máscara previa desde predicciones fuera de fold | `scripts/build_prior.py` |
| Inferencia en dos pasadas | `inference/volume.py` (`predict_case_two_pass`) |
| EMA de pesos (`train.ema_decay`) | `scripts/train.py` |
| Pasos en el laboratorio | `scripts/lab/refinar.sh` |
| Configuración | `configs/tuning/y7_refine.yaml` |
| Pruebas con phantoms (12) | `06_tests/test_two_pass.py` |

## 3. Tres decisiones de diseño

**La máscara previa nunca es el ground truth.** Si la segunda pasada se entrena con máscaras
perfectas aprende a fiarse de ellas y falla en inferencia, donde la máscara la predice el modelo.
Aquí cada caso lleva la región que predijo un modelo **que no lo vio**: las predicciones fuera de
fold de `y4` (caso del fold k ← modelo entrenado sin el fold k). Vale para los 85 casos sea cual
sea el fold que se esté entrenando. `TwoPassSlices` se niega a arrancar si falta ese archivo, y
`build_prior.py --from-labels` existe solo para probar el código y avisa al usarse. En inferencia la
máscara es la de la pasada 1 del propio modelo.

**Un modelo, no dos.** Los recortes son para la red una imagen más, con todos sus objetivos
(región, cajas, presencia, papel, borde), así que entran en el mismo lote que los cortes completos y
la pérdida es la misma. El coste es un canal de entrada: 288 parámetros sobre 2,93 millones.

**Se parte de lo ya entrenado.** Cada fold arranca del `y4` de ese mismo fold (entrenado sin sus
casos de validación) con el canal nuevo en cero: en el paso 0 el modelo da exactamente la salida de
`y4` (lo verifica `test_ampliar_los_canales_de_entrada_no_cambia_el_modelo_de_origen`). Se afina 10
épocas a lr 1e-4, la mitad de GPU que entrenar desde cero.

**EMA de pesos, decaimiento 0,999.** Se evalúa y se guarda la media móvil; el modelo sin promediar
queda en `last_raw.pth` y en el fold 0 se evalúa también, para medir qué aporta la EMA por separado.
Es un estabilizador: no recupera información que la entrada no trae.

## 4. Cómo se mide

Tres comparaciones, todas emparejadas por fold sobre los mismos 85 pacientes:

| comparación | qué aísla |
|---|---|
| dos pasadas contra **el mismo modelo con solo la pasada 1** | el efecto de la segunda pasada, sin ruido de entrenamiento |
| pasada 1 del modelo afinado contra `y4` | el efecto de afinar 10 épocas más con EMA y recortes |
| EMA contra sin EMA (fold 0) | el efecto de la EMA |

Regla, fijada antes de tener resultados: la segunda pasada **aporta** si mejora el Dice por
fragmento frente a la pasada 1 del mismo modelo en la mayoría de los folds, sin empeorar el IoU por
fragmento y sin que el Dice del principal caiga más de 0,01. Para adoptar el modelo frente a `y4` se
pide además superar 2σ = 0,026 (σ emparejada de la Fase 2) y no romper los guardarraíles de
detección y clasificación. Se reportan también el MAE de distancia y la latencia, que con dos
pasadas sube: cada corte con hueso añade hasta tres inferencias.

## 5. Verificación antes de gastar GPU

- 12 pruebas con phantoms: geometría de recortar y pegar, canales de entrada, mezcla del dataset,
  rechazo sin máscara previa, EMA. Suite completa: 155 en verde; las 10 que fallan son de la Fase 2
  (dos configuraciones que no se versionaron).
- Prueba de humo local de punta a punta en 3 casos (caché de 512, máscara previa, entrenamiento
  con EMA, predicción en dos pasadas).
- Sobre un caso real, la segunda pasada recorre los recortes, cambia el 100 % de los vóxeles dentro
  del hueso predicho y ninguno fuera.
- En el laboratorio: caché de 512 px de los 100 casos (16 GB, 0,6-0,7 mm/px) y máscara previa de
  los 85 casos de train + val.

## 6. Resultados (2026-10-08)

5 folds, 85 pacientes, validación cruzada. Una sola combinación de posproceso por modelo, elegida
por la media de los 5 folds (`scripts/lab/confirmacion.py`). **No es test.**

### 6.1 Tabla principal

| modelo | posproceso | Dice frag | desv. | IoU frag | Dice principal | Dice secundario | sec. recuperados | MAE distancia |
|---|---|---|---|---|---|---|---|---|
| y0 control | edt 5 / 0,3 | 0,7520 | 0,0423 | 0,6892 | 0,9227 | 0,5688 | 59,8 % | 1,01 mm |
| y1 salto | edt 5 / 0,3 | 0,7651 | 0,0409 | 0,7032 | 0,9257 | 0,5925 | 63,0 % | 0,92 mm |
| y4 salto + role3 | role 2 / 0,3 / 3 | 0,7874 | 0,0224 | 0,7101 | 0,9376 | 0,6252 | 55,3 % | 1,84 mm |
| y7, solo pasada 1 | edt 5 / 0,1 | 0,7784 | 0,0415 | 0,7139 | 0,9257 | 0,6200 | 66,1 % | 1,64 mm |
| y7, solo pasada 1 | role 2 / 0,3 / 3 | 0,7874 | 0,0239 | 0,7104 | 0,9370 | 0,6254 | 57,6 % | 1,97 mm |
| **y7, dos pasadas** | **edt 5 / 0,2** | **0,7889** | 0,0534 | **0,7252** | 0,9269 | **0,6412** | **68,3 %** | **0,57 mm** |
| y7, dos pasadas | role 1 / 0,3 / 3 | 0,7816 | 0,0195 | 0,7021 | 0,9346 | 0,6163 | 56,4 % | 1,71 mm |

Contra el control, emparejado por fold (t crítico 2,776):

| modelo | Δ Dice | t | folds a favor | Δ IoU | Δ principal |
|---|---|---|---|---|---|
| y1 [edt] | +0,0131 | 4,23 | 5/5 | +0,0140 | +0,0031 |
| y4 [role] | +0,0354 | 2,05 | 4/5 | +0,0209 | +0,0150 |
| y7 pasada 1 [edt] | +0,0264 | 3,91 | 5/5 | +0,0247 | +0,0031 |
| **y7 dos pasadas [edt]** | **+0,0369** | **3,13** | **5/5** | **+0,0360** | +0,0042 |
| y7 dos pasadas [role] | +0,0296 | 2,70 | 5/5 | +0,0129 | +0,0120 |

Dice por fragmento en cada fold:

| modelo | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| y0 [edt] | 0,6805 | 0,7777 | 0,7871 | 0,7510 | 0,7637 |
| y4 [role] | 0,7646 | 0,8149 | 0,7673 | 0,8056 | 0,7847 |
| y7 pasada 1 [edt] | 0,7098 | 0,7788 | 0,8166 | 0,7812 | 0,8055 |
| y7 dos pasadas [edt] | 0,7057 | 0,7778 | 0,8373 | 0,7897 | 0,8339 |
| y7 dos pasadas [role] | 0,7491 | 0,8002 | 0,7914 | 0,7862 | 0,7809 |

Detección y clasificación de y7 (corte completo, media de 5 folds): mAP@0.5 0,9746,
mAP@[.5:.95] 0,7997, IoU de caja 0,8932, Dice de región 0,9665, F1 0,9893, AUC 0,9982.

### 6.2 Qué aporta cada pieza

| comparación | Δ Dice frag | lectura |
|---|---|---|
| segunda pasada: y7 dos pasadas contra y7 pasada 1, `edt` | +0,0105 (3/5 folds) | mejora media pequeña y no consistente: gana claro en los folds 2 y 4 (+0,021 y +0,028), empata o pierde por milésimas en 0 y 1 |
| segunda pasada, con `role` | −0,0058 | no ayuda al método por papel |
| afinado + EMA: y7 pasada 1 contra y4, `edt` | +0,0026 | sin efecto medible |
| afinado + EMA, con `role` | 0,0000 | idéntico |
| EMA contra sin EMA (fold 0, dos pasadas, `edt`) | +0,023 | un solo fold: indicio, no medida |

- **Donde la segunda pasada sí es clara es en la distancia.** El MAE de separación baja de 1,64 a
  0,57 mm con `edt` (en el fold 0, de 1,62 a 0,21 mm). Es la medida que pide el §3.3 del enunciado, y
  es el mejor valor de toda la campaña.
- También sube el IoU por fragmento (+0,011) y los secundarios recuperados (66 → 68 %).
- La cabeza de borde mejora al ver los recortes: su Dice en validación pasa de 0,53-0,63 a
  0,57-0,63. Por eso `edt`, que depende del borde, vuelve a superar a `role`.
- `role` no se beneficia: su fallo (inconsistencia entre cortes, que el suavizado en z compensa) no
  es de resolución.

### 6.3 Frente a los objetivos del enunciado (§5)

Validación cruzada, **no test**.

| métrica | objetivo | y0 control | y7 dos pasadas [edt] | ¿cumple? |
|---|---|---|---|---|
| Dice por fragmento | ≥ 0,85 | 0,752 | 0,789 | **no** (faltan 0,061) |
| IoU por fragmento | ≥ 0,70 | 0,689 | 0,725 | sí; lo supera en los 5 folds salvo el 0 (0,66 con la combinación común) |
| F1 clasificación | ≥ 0,85 | 0,987 | 0,989 | sí |
| AUC | ≥ 0,85 | 0,998 | 0,998 | sí |
| IoU de caja | ≥ 0,65 | 0,907 | 0,893 | sí |
| mAP@0.50 | ≥ 0,65 | 0,977 | 0,975 | sí |
| mAP@[.50:.95] | ≥ 0,40 | 0,834 | 0,800 | sí |

### 6.4 Veredicto

- **La regla del §4 se cumple a medias.** Frente al control, y7 con dos pasadas y `edt` es el primer
  candidato que supera 2σ (0,026), gana en los 5 folds, es significativo (t = 3,13) y no empeora el
  IoU ni el principal. Pero la segunda pasada, aislada, gana solo en 3 de 5 folds: la condición de
  "mayoría de folds sin empeorar" se cumple por la mínima y con una media de +0,010.
- **Guardarraíl de detección: no pasa.** mAP@[.5:.95] baja 0,034 e IoU de caja 0,014 respecto al
  control (0,010 y 0,004 respecto a y4). Siguen al doble del objetivo, pero el coste es real y viene
  de añadir `role3` y de afinar con recortes, no de la segunda pasada en sí.
- **En Dice por fragmento empata con y4** (0,7889 contra 0,7874); lo que gana es IoU (+0,015),
  secundarios recuperados (68 % contra 55 %) y distancia (0,57 contra 1,84 mm).
- **El objetivo de 0,85 no se alcanza.** La resolución no era toda la respuesta: duplicarla mueve la
  métrica principal un punto. Lo que queda es lo que el §7 anticipaba, fragmentos que se tocan sin
  grieta visible, y la inconsistencia entre cortes de una red 2D.
- **Coste de inferencia.** La segunda pasada añade ≈ 2,2 recortes por corte (≈ 10 600 recortes en
  los 17 casos de un fold, 4 796 cortes): unas 3,2 veces el cómputo del modelo por corte. La
  latencia real no se ha medido; hay que hacerlo antes de reportarla.

### 6.5 Recomendación

1. Para la entrega, **y7 con dos pasadas y `edt`** es la configuración con mejor IoU, mejor
   recuperación de secundarios y, con diferencia, mejor distancia. Si la latencia en CPU resulta
   inaceptable, **y7 con solo la pasada 1** conserva casi todo el Dice (0,778) al coste de siempre.
2. Evaluar en test una sola vez. Hace falta entrenar y7 sobre la partición oficial partiendo de
   `y4_fullres_role_final`, con la máscara previa de los casos de test predicha por ese modelo.
3. Medir la latencia en GPU y CPU con dos pasadas (`scripts/latency.py` necesita soportarlas).
4. Reportar el Dice por fragmento como no cumplido, con el análisis de causas: es lo que el
   enunciado pide en la galería de peores casos y en las limitaciones del model card.

## 7. Limitaciones conocidas de antemano

- La salida refinada vuelve a la grilla de 256 antes del posproceso: la segunda pasada decide con
  más resolución, pero las máscaras finales no son más finas.
- La máscara previa de entrenamiento viene de `y4` y la de inferencia del modelo afinado; son
  parecidas, no idénticas.
- Si dos fragmentos se tocan sin grieta visible ni a resolución nativa, ninguna resolución los
  separa. Los oráculos dicen que hay margen, no cuánto es alcanzable.
- Una sola semilla.
