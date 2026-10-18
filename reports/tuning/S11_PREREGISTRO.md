# Semana 11 · pre-registro del afinado por etapas y de la EMA

Escrito el 2026-10-11 a las 09:50, con los entrenamientos en curso y **antes de ver cualquier resultado**.

## Por qué

En el avance del 9 de octubre el profesor propuso entrenar por etapas: una vez el extractor y el detector
funcionan, afinar solo la parte de segmentación con el detector "apagado", porque las tres pérdidas
juntas interactúan. También advirtió que la EMA de pesos puede funcionar como un enmascarante si las
imágenes de entrada no siempre llegan en las mismas condiciones.

## Qué se corre (DGX Spark, cola de trabajos)

| brazo | qué se entrena | desde | épocas | lr | EMA |
|---|---|---|---|---|---|
| `ftseg` | solo la cabeza de segmentación (backbone, cuello, clasificación y detección congelados, BatchNorm en modo eval) | `cv_c00_base_e20_f{k}/last.pth` | 8 | 1e-4 | 0,999 |
| `fttodo` (control) | todo el modelo, igual que `ftseg` | ídem | 8 | 1e-4 | 0,999 |

- 5 folds por paciente (`reports/tuning/cv_folds.json`), los mismos λ de cada fold.
- Cada corrida guarda la EMA (`last.pth`) y el modelo sin promediar (`last_raw.pth`): la comparación EMA
  contra sin EMA sale emparejada dentro de la misma corrida, sin ruido de semilla.
- Predicción fuera de fold con TTA de 5 pasadas y **un solo** posproceso fijo, el final: `edt_hmax`,
  profundidad 5 mm, umbral de borde 0,1, h = 1,5. No se barre nada.
- Referencia: el mismo `cv_c00_base_e20_f{k}` con TTA 5 y ese mismo posproceso (`pp_tta_base_hmax`).

## Regla de adopción (fijada ahora)

Un brazo se adopta solo si, contra la referencia y emparejado por fold:

1. Δ Dice por fragmento ≥ +0,026 (2 × 0,013, la desviación por fold del control de semilla a 20 épocas),
2. t > 2,776 (5 folds, α = 0,05 a dos colas) y al menos 4 de 5 folds a favor,
3. Δ IoU por fragmento > 0,
4. el Dice del fragmento principal no baja más de 0,010.

Además:

- `ftseg` solo se atribuye al **congelamiento** si también supera a `fttodo` por el mismo criterio; si no,
  la ganancia es de "más épocas", y se reporta así.
- La EMA se adopta solo si `last` supera a `last_raw` con el mismo criterio. Por la advertencia del
  profesor, si se adopta se mide también en volúmenes perturbados (orientación por header, ruido, contraste)
  antes de llevarla al modelo final.
- Si un brazo cumple, se afina igual el modelo final (`v2_last`, mismas 8 épocas sobre los 85 casos de
  train+val) y el test se mide **una sola vez** más, declarado como tal. Si ninguno cumple, el modelo
  final no cambia y el test no se vuelve a mirar.

## Además, para lo que pidió el profesor el 2 de octubre

`v2_repro`: el entrenamiento `v2` repetido con la misma semilla, configuración y λ (40 épocas),
registrando por época los γ de CBAM y del cuello y la misma métrica en 2000 cortes de train sin aumento
(curva train contra val con la misma métrica). No compite por ser el modelo final: sirve para mostrar la
evolución de γ y para verificar la reproducibilidad del entrenamiento (sus métricas de val deben seguir a
las de `v2`).
