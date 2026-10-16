# Semana 11 · pre-registro: sobreajuste de la cabeza de borde

Escrito el 2026-10-16 a las 08:50, **antes de lanzar cualquier corrida**. Las reglas de abajo las aplica
`scripts/s11_borde/decidir.py` sin intervención; el orquestador (`scripts/s11_borde/orquestar.sh`) avanza de etapa solo.

## Por qué

La cabeza de borde sobreajusta: en `v2_repro` la BCE del borde en train baja de 0,064 a 0,005 en 40
épocas mientras la de validación sube de 0,018 a 0,034 (`reports/train/v2_repro_history.csv`). Es el
patrón "train baja, validación sube". Para ese caso, la receta habitual es subir la dificultad de la
tarea, reducir el modelo o regularizar (L2, dropout).

Lo que ya estaba probado y **no se repite**:

| Remedio | Dónde | Resultado |
|---|---|---|
| Congelar el backbone y entrenar solo la segmentación | `ftseg`, semana 11 | empeora: −0,015, t = −5,2, 0/5 folds |
| Frenar el backbone (lr × 0,3) | `c10`, Fase 1 | empeora 4,3σ |
| Subir la dificultad (aumentación fuerte) | `c09` | peor: subajusta |
| Dropout espacial 0-0,40 | `c07`, `c08`, `sd025-040` | sin efecto |
| Tasa de aprendizaje 1e-3 | `c06` | sin efecto |
| Otras representaciones del borde | `core3`, `dist` (Fase 2) | peores |

Lo que **no** estaba probado, y se prueba ahora:

| Candidato | Qué cambia (lo demás es `c00_base`) | Pregunta |
|---|---|---|
| `c11_wd1e2` | `weight_decay` 1e-4 → 1e-2 (L2 ×100; con AdamW, 1e-4 es casi nulo) | ¿regularizar con L2 cierra la brecha? |
| `c12_segw32` | decodificador de segmentación de 64 a 32 canales (−55 % de sus parámetros) | ¿reducir el modelo donde sobreajusta? |
| `c13_unetpp` | decodificador U-Net++ (sesión 2 de segmentación): C0 a resolución completa (bloque 1 antes del MaxPool), skips anidadas y supervisión profunda | ¿le faltaba detalle fino al decodificador? Es la hipótesis que la Fase 2 dejó sin probar ("decodificador a resolución completa, supervisión profunda") |

Reglas del enunciado que se respetan: un solo backbone (C0 sale del mismo bloque 1 de la FundidoraPC; no
hay rama nueva), cabezas desde cero, CBAM antes de la bifurcación (las skips de C1 y C2 ya lo saltaban
antes; C0 igual), tres términos de pérdida (la supervisión profunda promedia L_seg, no añade un
término), λ recalibrados en cada corrida, implementación propia de U-Net++ con la idea citada (Zhou et
al., 2018). Nada de YOLO, Detectron2 ni Mask R-CNN. SAM no se toca.

## Etapa 1 · cribado (fold 0, 12 épocas)

Mismo protocolo que el cribado de la Fase 1: λ recalibrados en el fold, `last.pth`, predicción fuera de fold
sin TTA y la misma grilla de posproceso para todos (`edt`, profundidad 1,5-6 mm × umbral 0,1-0,5); cada
candidato se mide en su mejor combinación.

Corren también **`s11_ref`** (= `c00_base`, mismo código y semilla) como control: el código de la semana 11
toca el backbone y la pérdida, y el control comprueba que el modelo de siempre no cambió.

**Referencia:** las cinco corridas con la configuración base (`c00_base`, semillas 42, 1337, 2024 y 7 de la
Fase 1, más `s11_ref`). μ y σ son la media y la desviación muestral de su Dice por fragmento.

**Pasa a la etapa 2** un candidato con Dice por fragmento ≥ μ + 2σ **y** IoU por fragmento ≥ la media de
las referencias. Se confirman como máximo los dos mejores. Con las cuatro referencias de la Fase 1 el
umbral sería 0,6785 + 2 × 0,0097 ≈ 0,698.

Es un filtro deliberadamente permisivo: comparar contra la media, y no contra la semilla 42 (la más
alta de las cuatro, 0,6925), evita descartar por mala suerte de la referencia. La decisión de verdad se
toma en la etapa 2.

## Etapa 2 · confirmación (5 folds, 20 épocas)

Igual que el afinado de la semana 11: entrenamiento desde cero en cada fold (λ recalibrados por fold),
predicción fuera de fold con TTA de 5 pasadas y **un solo** posproceso fijo, el final (`edt_hmax`, 5 mm,
umbral 0,1, h = 1,5). Referencia emparejada por fold: `pp_tta_base_hmax` (`cv_c00_base_e20_f{k}` con lo
mismo; copia en `referencia/`).

**Se adopta** solo si, contra la referencia y emparejado por fold:

1. Δ Dice por fragmento ≥ +0,026 (2 × 0,013, la desviación por fold del control de semilla a 20 épocas);
2. t > 2,776 (4 g.l., α = 0,05 a dos colas) y al menos 4 de 5 folds a favor;
3. Δ IoU por fragmento > 0;
4. el Dice del fragmento principal no baja más de 0,010;
5. guardarraíl: el mAP@[.50:.95] de validación (media de los 5 folds, última época) no baja más de 0,005.

Si los dos candidatos cumplen, gana el de mayor Δ Dice.

## Etapa 3 · modelo final y test

Solo si hay ganador. Se entrena como `v2` (split de siempre, 40 épocas, λ recalibrados en train) con el
cambio ganador, y el test se mide **una sola vez** con la inferencia final (`configs/tuned.yaml`: TTA 5 +
`edt_hmax`). Se reporta contra `v2_last` + `tuned.yaml` (Dice 0,768, IoU 0,697) tal cual salga, sea
mejor o peor. Si no hay ganador, el modelo final no cambia y el test no se vuelve a mirar.

## Qué cuenta como resultado

Los tres desenlaces sirven, y se reportan con la misma visibilidad:

- **Ninguno pasa el cribado:** L2, una cabeza más pequeña y U-Net++ quedan descartados con evidencia,
  igual que `ftseg`. Es lo más probable: el dropout, que es regularización de la misma familia, no movió
  nada.
- **Pasa el cribado pero no confirma:** se reporta como señal que no sobrevivió a 5 folds (como `c05`).
- **Confirma:** nuevo modelo final, con una medición de test declarada.

## Aparte: diagnóstico del hueco del ala ilíaca (no es una decisión de modelo)

El hueco que se ve en la reconstrucción 3D del caso 075 (test) se diagnosticó mirando la etiqueta de los
expertos de ese caso y de los 15 de test. Lo que se encontró es que la grilla de 256 px abre túneles en
hueso delgado. Ninguna decisión de esta semana se toma con esas cifras de test: el único cambio que
motiva es cómo se dibuja la malla en el dashboard (visualizador 3), que no entra en ninguna métrica.
