# Cribado y4xul — pre-registro

Escrito el 2026-10-06 a las 12:25 (hora de Colombia), con las corridas en su primera época y antes
de tener ningún Dice por fragmento. Sigue el protocolo de `reports/tuning/fase2/REGISTRO.md`.

## Por qué

La Fase 2 cerró con el cuello de botella medido: la cabeza de borde cubre el 19 % de la superficie de
fractura y con el borde real el Dice por fragmento subiría de 0,743 a 0,902. Probó dos
representaciones nuevas de la **salida** (`core3`, `dist`) y las dos fallaron. Dejó sin ejecutar la
hipótesis de un decodificador a resolución completa.

Tres cosas del código que la Fase 2 no tocó:

1. Cada bloque del backbone termina en `MaxPool2d`, y lo más fino que sale es C1, a stride 2.
2. El decodificador pasa de stride 2 a 1 con una interpolación: no recibe ninguna característica a
   resolución completa. Una grieta mide 1-2 px a 256.
3. Ninguna salida dice qué vóxeles son del fragmento principal y cuáles de un secundario; eso se
   decide después, por volumen.

## Qué se compara

Fold 0 de `reports/tuning/cv_folds.json`, 12 épocas, semilla 42, sin transfer learning (la Fase 2
midió que no aporta y el laboratorio no tiene esos pesos), λ recalibrados en cada corrida, última
época. Es la misma escala en la que la Fase 1 midió su suelo de ruido: **σ = 0,0097** con 4 semillas.

| id | config | cambia respecto al control |
|---|---|---|
| y0 | `y0_sintl.yaml` | control |
| y1 | `y1_fullres.yaml` | `seg_fullres_skip`: C0 (bloque 1 antes del pooling) entra al decodificador |
| y2 | `y2_fullres_avg.yaml` | y1 + `pool: avg` en el backbone |
| y4 | `y4_fullres_role.yaml` | y1 + salida `role3` (fondo / principal / secundario) |

`y3_role.yaml` (role3 sin el salto) se lanzó y se detuvo en la primera época para liberar GPU: el
efecto de `role3` se lee igual en y4 contra y1, y en y4 evaluado con sus dos posprocesos.

Cada modelo se evalúa en su mejor posproceso `edt` (grilla de la Fase 2: `seed_depth_mm` 3/4/5/6 ×
`edge_threshold` 0,1/0,2/0,3/0,5). y4 se evalúa además con `role` (`role_seed_depth_mm` 1/2 ×
`role_threshold` 0,3/0,5/0,7 × `role_smooth_mm` 0/1/2/3).

Permitido por el enunciado: sigue siendo 2D corte a corte, un backbone propio, CBAM antes de la
bifurcación, tres cabezas y pérdida de tres términos (`role3` va dentro de `seg`). Sin frameworks de
alto nivel. El pooling del backbone no tiene pesos; cambiarlo solo es posible sin transfer learning.

## Regla de decisión (no se cambia después de ver resultados)

Métrica primaria: Dice por fragmento en el fold 0.

Un candidato **pasa el cribado** si:
- supera al control y0 por más de **2σ = 0,0194**,
- no empeora el IoU por fragmento,
- y el Dice del fragmento principal no cae más de 0,01.

Guardarraíl: F1, AUC, mAP@[.50:.95], IoU de caja y Dice de región no caen más de 2σ de su escala
(desviaciones entre folds de la Fase 1: 0,0011 / 0,0004 / 0,0096 / 0,0046 / 0,0026).

Pasar el cribado **no es adoptar**: como en la Fase 2, hace falta la confirmación en 5 folds a 20
épocas, emparejada por fold, antes de tocar `base.yaml` o mirar el test. Un solo fold a 12 épocas
solo decide en qué gastar esa confirmación.

Señales secundarias, sin regla: Dice de la cabeza de borde en validación (mide directamente si el
salto mejora la cobertura del borde) y Dice por vóxel de la clase secundaria de `role3`.

## Techo medido antes de entrenar (`scripts/oracle_role.py`, fold 0, 17 casos)

Con el papel real de cada vóxel y sin borde:

| erosión entre secundarios | suavizado en z | Dice fragmento | IoU | Dice principal | Dice secundario |
|---|---|---|---|---|---|
| 2 mm | 0 | 0,901 | 0,885 | 1,000 | 0,809 |
| 4 mm | 0 | 0,928 | 0,892 | 1,000 | 0,861 |
| 4 mm | 1 mm | 0,929 | 0,892 | 1,000 | 0,863 |

Con el papel borrado en el 20 % de los cortes que tienen secundario (el fallo esperable de una red
que decide corte a corte) baja a 0,67 sin suavizar y a 0,72-0,77 con suavizado de 1-3 mm.

Lectura: el techo supera el objetivo (0,85 / 0,70) y no parte el principal, pero queda por debajo
del 0,995 del oráculo de borde porque dos secundarios que se tocan siguen necesitando erosión. Y es
sensible a la consistencia entre cortes. La hipótesis de `role3` es más débil que la del salto.
