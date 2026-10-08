# Refinamiento por región: segunda pasada por hueso a resolución nativa

Fecha: 2026-10-07. Rama `y4xul`. Ejecución en SparkLab (DGX Spark, GPU NVIDIA GB10).
Antecedentes: `RESUMEN.md` (cribado y confirmación) y `REGISTRO.md` (pre-registros), en esta carpeta.

**Estado: implementado, probado y entrenando.** Los resultados se añaden en el §6 cuando terminen
los 5 folds. El test no se ha evaluado.

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

## 6. Resultados

Pendientes. En curso: `y7_refine`, 5 folds, 10 épocas, 37 072 muestras por época (cortes completos
y recortes a partes iguales), en tres carriles.

## 7. Limitaciones conocidas de antemano

- La salida refinada vuelve a la grilla de 256 antes del posproceso: la segunda pasada decide con
  más resolución, pero las máscaras finales no son más finas.
- La máscara previa de entrenamiento viene de `y4` y la de inferencia del modelo afinado; son
  parecidas, no idénticas.
- Si dos fragmentos se tocan sin grieta visible ni a resolución nativa, ninguna resolución los
  separa. Los oráculos dicen que hay margen, no cuánto es alcanzable.
- Una sola semilla.
