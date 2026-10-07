# Cribado y confirmación y4xul — informe de resultados

Fecha: 2026-10-07. Rama `y4xul`. Ejecutado en SparkLab (DGX Spark, GPU NVIDIA GB10).
Pre-registro, regla de decisión y resultado del cribado: `REGISTRO.md`, en esta misma carpeta.

**Veredicto.** Hay avance medible, pero el objetivo de Dice por fragmento sigue sin cumplirse. El
salto a resolución completa es la única mejora confirmada con significancia estadística y sin
coste. La salida principal/secundario (plan A) da la media más alta, pero no alcanza significancia
con 5 folds y cuesta precisión de detección y de la medida de distancia. **El test no se ha
evaluado**: todas las cifras son de validación cruzada.

## 1. Qué se probó

Punto de partida: la Fase 2 de la rama `Nicolas` (`../fase2/RESUMEN_FASE2.md`), que cerró en
0,745 de Dice por fragmento en validación cruzada y 0,725 en test, con el cuello de botella medido
en la cabeza de borde.

| id | qué cambia | origen de la idea |
|---|---|---|
| y0 | control: `base.yaml` sin transfer learning | la Fase 2 midió que el transfer learning no aporta |
| y1 | salto a resolución completa: el decodificador recibe el bloque 1 antes del pooling | el decodificador pasaba de stride 2 a 1 solo interpolando |
| y2 | y1 + pooling promedio en vez de máximo | propuesta del equipo |
| y4 | y1 + salida `role3` (fondo / principal / secundario) y método de separación `role` | plan A |

El plan B completo (segunda pasada por hueso a mayor resolución) no se implementó.

## 2. Cribado (fold 0, 12 épocas)

| id | posproceso | Dice frag | IoU frag | Δ Dice vs y0 | decisión |
|---|---|---|---|---|---|
| y0 | edt | 0,684 | 0,616 | — | control |
| y1 | edt | 0,729 | 0,666 | +0,045 (4,6σ) | a confirmación |
| y2 | edt | 0,713 | 0,654 | +0,029 (2,9σ) | descartado: peor que y1 |
| y4 | edt | 0,708 | 0,651 | +0,024 | — |
| y4 | role | 0,757 | 0,674 | +0,073 (7,5σ) | a confirmación |

σ = 0,0097 (suelo de ruido de la Fase 1 a esta escala). El pooling promedio no mejora al máximo.

## 3. Confirmación (5 folds, 20 épocas, emparejada por fold)

Una sola combinación de posproceso por modelo, elegida por la media de los 5 folds
(`scripts/lab/confirmacion.py`). 85 pacientes, cada uno predicho por un modelo que no lo vio.

| modelo | posproceso | Dice frag | desv. | IoU frag | Dice principal | Dice secundario | sec. recuperados | MAE distancia |
|---|---|---|---|---|---|---|---|---|
| y0 control | edt 5 / 0,3 | 0,7520 | 0,0423 | 0,6892 | 0,9227 | 0,5688 | 59,8 % | 1,01 mm |
| y1 salto | edt 5 / 0,3 | 0,7651 | 0,0409 | 0,7032 | 0,9257 | 0,5925 | 63,0 % | 0,92 mm |
| y4 salto + role3 | edt 5 / 0,2 | 0,7758 | 0,0494 | 0,7127 | 0,9255 | 0,6149 | 66,5 % | 1,39 mm |
| y4 salto + role3 | role 2 / 0,3 / 3 | **0,7874** | 0,0224 | 0,7101 | **0,9376** | **0,6252** | 55,3 % | 1,84 mm |

Diferencia contra el control, fold a fold (t crítico 2,776 con 4 grados de libertad):

| modelo | Δ Dice | t | folds a favor | Δ IoU | Δ principal |
|---|---|---|---|---|---|
| y1 [edt] | +0,0131 | **4,23** | 5/5 | +0,0140 | +0,0031 |
| y4 [edt] | +0,0238 | 1,71 | 4/5 | +0,0235 | +0,0028 |
| y4 [role] | +0,0354 | 2,05 | 4/5 | +0,0209 | +0,0150 |

Dice por fragmento en cada fold:

| modelo | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| y0 [edt] | 0,6805 | 0,7777 | 0,7871 | 0,7510 | 0,7637 |
| y1 [edt] | 0,6946 | 0,7801 | 0,8014 | 0,7727 | 0,7767 |
| y4 [edt] | 0,7041 | 0,7548 | 0,8135 | 0,7783 | 0,8282 |
| y4 [role] | 0,7646 | 0,8149 | 0,7673 | 0,8056 | 0,7847 |

Detección y clasificación (media de 5 folds, última época):

| modelo | mAP@0.5 | mAP@[.5:.95] | IoU caja | Dice región | F1 | AUC |
|---|---|---|---|---|---|---|
| y0 | 0,9773 | 0,8341 | 0,9068 | 0,9660 | 0,9871 | 0,9975 |
| y1 | 0,9764 | 0,8286 | 0,9047 | 0,9679 | 0,9876 | 0,9975 |
| y4 | 0,9749 | 0,8096 | 0,8969 | 0,9683 | 0,9867 | 0,9973 |

## 4. Lectura

- **y1 (salto a resolución completa) es una mejora real y gratuita.** +0,013 de Dice y +0,014 de IoU
  en los 5 folds, t = 4,23. No toca la detección ni la clasificación (diferencias dentro del ruido)
  y mejora algo la distancia. Es pequeña: no pasa el umbral de 2σ = 0,026 de la Fase 2, que está
  pensado para tamaños de efecto mayores; lo que la sostiene es la consistencia entre folds.
- **y4 con `role` tiene la mejor media (0,787) y la menor variación entre folds (0,022 contra
  0,042), pero no es concluyente.** t = 2,05 con 4/5 folds: pierde en el fold 2. Su ventaja está en
  el fragmento principal (+0,015) y en el fold más difícil (fold 0: 0,765 contra 0,681).
- **y4 tiene costes medidos:**
  - mAP@[.5:.95] baja 0,0245 e IoU de caja 0,0099, fuera del margen fijado en el pre-registro. Se
    repite en los 5 folds. Ambos siguen muy por encima de los objetivos del enunciado.
  - El MAE de distancia sube de 1,0 a 1,8 mm con `role` (1,4 mm con `edt`).
  - `role` recupera menos secundarios que `edt` (55 % contra 66 %) aunque los que recupera los
    delimita mejor: gana Dice en el principal y pierde fragmentos pequeños.
- **El cribado sobrestimó los efectos.** En el fold 0 a 12 épocas y1 daba +0,045 y y4 +0,073; en 5
  folds a 20 épocas son +0,013 y +0,035. El fold 0 es el más difícil y el más sensible.
- **La hipótesis con la que se propuso el salto no se cumplió.** El Dice de la cabeza de borde en
  validación es igual con y sin salto. La mejora llega por otra vía, no por una mejor cobertura del
  borde; el cuello de botella que midió la Fase 2 sigue ahí.
- El control de esta campaña (0,752) queda por encima de la referencia de la Fase 2 con transfer
  learning (0,745) y a la par de su corrida sin él (0,759 / 0,752): coherente con que el transfer
  learning no aporta.

## 5. Frente a los objetivos del enunciado (§5)

Validación cruzada, **no test**. El test de 15 pacientes se evalúa una sola vez con la
configuración que el equipo adopte.

| métrica | objetivo | y0 control | y1 salto | y4 + role | ¿cumple el mejor? |
|---|---|---|---|---|---|
| Dice por fragmento | ≥ 0,85 | 0,752 | 0,765 | 0,787 | **no** (faltan 0,063) |
| IoU por fragmento | ≥ 0,70 | 0,689 | 0,703 | 0,710 | sí, por poco |
| F1 clasificación | ≥ 0,85 | 0,987 | 0,988 | 0,987 | sí |
| AUC | ≥ 0,85 | 0,998 | 0,998 | 0,997 | sí |
| IoU de caja | ≥ 0,65 | 0,907 | 0,905 | 0,897 | sí |
| mAP@0.50 | ≥ 0,65 | 0,977 | 0,976 | 0,975 | sí |
| mAP@[.50:.95] | ≥ 0,40 | 0,834 | 0,829 | 0,810 | sí |

El IoU por fragmento cruza el objetivo por primera vez, pero con 0,003-0,010 de margen y una
desviación entre folds mayor que ese margen: en test puede quedar a cualquier lado.

## 6. Recomendación

1. **Adoptar y1** (`seg_fullres_skip: true`, sin transfer learning). Es la única decisión que los
   datos respaldan sin reservas.
2. **y4 queda como candidato, no como decisión.** Antes de adoptarlo haría falta una réplica con
   otra semilla, como exigió la Fase 2 para sus candidatos. Si el equipo prioriza el Dice por
   fragmento sobre la distancia y acepta el coste en detección, es la opción con mejor media.
3. Probar `role` y `edt` combinados: `role` protege el principal y `edt` recupera más secundarios;
   fallan en sitios distintos.
4. El objetivo de 0,85 no está al alcance con cambios de este tamaño. El cuello de botella sigue
   siendo la cobertura del borde; la vía que queda sin probar es el plan B (segunda pasada por
   hueso a resolución nativa).

## 7. Limitaciones

- Una sola semilla por modelo; sin réplica.
- El posproceso de cada modelo se eligió sobre los mismos 5 folds en los que se reporta: sesgo
  optimista pequeño, igual para todos los modelos.
- El guardarraíl del pre-registro usa desviaciones entre folds de la Fase 1, no emparejadas.
- La GPU del laboratorio estuvo compartida durante el cribado; no afecta a los resultados, solo a
  los tiempos.
- Los archivos crudos (CSV por combinación y fold, checkpoints, predicciones fuera de fold) están
  en el laboratorio, en el repositorio clonado y en `/workspace/mio/pengwin/`. El laboratorio no
  tiene credenciales para subirlos; las tablas de este informe se transcribieron de la salida de
  `scripts/lab/confirmacion.py`.

## 8. Cómo repetirlo

```bash
bash scripts/lab/y4xul.sh cache                       # caché de cortes + borde 3D
bash scripts/lab/y4xul.sh oraculo                     # techo del método role con el GT
bash scripts/lab/y4xul.sh lanzar y1_fullres 0 12      # una corrida del cribado
for f in 0 1 2 3 4; do bash scripts/lab/y4xul.sh corrida y1_fullres $f 20; done
bash scripts/lab/y4xul.sh confirmar                   # posproceso consolidado + tabla emparejada
```

En curso al escribir esto: entrenamiento final de y1 e y4 sobre la partición oficial (train → val,
40 épocas), `checkpoints/y1_fullres_final` y `checkpoints/y4_fullres_role_final`. No tocan el test.
