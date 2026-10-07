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

## Resultado del cribado (2026-10-06, 15:54)

Fold 0, 12 épocas, última época, cada modelo en su mejor posproceso.

| id | posproceso | Dice frag | IoU frag | Dice principal | Dice secundario | rec. sec. | mAP@[.5:.95] | IoU caja | Dice región | F1 | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| y0 control | edt 5 / 0,1 | 0,6842 | 0,6164 | 0,9110 | 0,4739 | 49,1 % | 0,8127 | 0,8968 | 0,9597 | 0,9854 | 0,9970 |
| y1 salto | edt 4 / 0,2 | 0,7292 | 0,6663 | 0,9327 | 0,5405 | 56,4 % | 0,8057 | 0,8947 | 0,9617 | 0,9830 | 0,9960 |
| y2 salto + avg | edt 4 / 0,1 | 0,7128 | 0,6542 | 0,9295 | 0,5118 | 54,5 % | 0,8030 | 0,8921 | 0,9619 | 0,9863 | 0,9970 |
| y4 salto + role3 | edt 4 / 0,2 | 0,7084 | 0,6507 | 0,9362 | 0,4972 | 52,7 % | 0,7890 | 0,8864 | 0,9630 | 0,9801 | 0,9957 |
| y4 salto + role3 | role 2 / 0,3 / 3 | 0,7574 | 0,6739 | 0,9293 | 0,5979 | 56,4 % | (mismo modelo) | | | | |

Aplicando la regla sin cambiarla (2σ = 0,0194):

- **y1 pasa** en la métrica primaria: +0,0450 (4,6σ), IoU +0,050, principal +0,022. Guardarraíl: F1
  −0,0024 y AUC −0,0010 quedan fuera de su margen por poco (0,0022 y 0,0008); el resto dentro.
- **y2 pasa** (+0,0286, 2,9σ) pero queda por debajo de y1: el pooling promedio no aporta sobre el
  máximo. No se confirma.
- **y4 con `role` pasa** en la métrica primaria: +0,0732 (7,5σ), IoU +0,058, principal +0,018.
  **Falla el guardarraíl** de detección: mAP@[.5:.95] −0,0237 (margen 0,0192), IoU de caja −0,0104
  (margen 0,0092), F1 −0,0053. Siguen muy por encima de los objetivos del enunciado (0,40 / 0,65 /
  0,85), pero es un coste real que la confirmación debe medir.
- El Dice de la cabeza de borde en validación es igual en las cuatro corridas (0,524-0,529): el
  salto no mejora la cobertura del borde. La ganancia de y1 llega por otra vía (región y fragmento
  principal más limpios), no por la hipótesis con la que se propuso.

Ningún candidato alcanza el objetivo (0,85 / 0,70). Se lanza la confirmación de y0, y1 e y4: 5
folds, 20 épocas, emparejada por fold. El test no se ha tocado.

## Segunda ronda: Dice ponderado de la clase secundaria (pre-registro, 2026-10-07)

Escrito antes de lanzar. Motivo: en la confirmación el método `role` recupera solo el 55 % de los
secundarios; su debilidad es la cobertura de la clase secundaria.

| id | config | cambia respecto a y4 |
|---|---|---|
| y5 | `y5_role_dw3.yaml` | Dice de `role3` ponderado (principal 1, secundario 3) |
| y6 | `y6_role_dw6.yaml` | ídem con secundario 6 |

    L_papel = 1 − (w_p·D_principal + w_s·D_secundario) / (w_p + w_s)
    D_c = (2·Σ p_c·g_c + ε) / (Σ p_c + Σ g_c + ε)

Sigue dentro del término `seg`; la pérdida es de tres términos y los λ se recalibran.

Además, sin coste de entrenamiento: las dos corridas eligen su mejor checkpoint con
`train.selection_metrics` = mAP@[.50:.95], Dice de región, F1 **y Dice por vóxel de la clase
secundaria**, y se evalúan en esa época y en la última.

Diseño: fold 0, 12 épocas, semilla 42. Control: `y4_fullres_role_f0_e12` de la primera ronda (Dice
por fragmento 0,7574 con `role`, 0,7084 con `edt`).

Regla: un candidato pasa si supera a y4 en más de 2σ = 0,0194 de Dice por fragmento (cada modelo en
su mejor posproceso), sin empeorar el IoU por fragmento y sin que el Dice del principal caiga más de
0,01. Expectativa declarada: efecto entre 0 y +0,02; la Fase 1 probó cuatro reponderaciones de
pérdida y sobremuestreo y ninguna superó el ruido.
