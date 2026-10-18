# Fase 2 — pre-registro de experimentos

Cada experimento se escribe AQUÍ **antes** de lanzarlo: id, hipótesis, qué cambia y por qué el
enunciado lo permite (con cita), métrica primaria y regla de decisión. La regla no se cambia después
de ver resultados; si se cambia, se explica y la conclusión queda marcada como **exploratoria**.

Reglas comunes (Fase 2 §4):
- Selección solo por CV por paciente (`reports/tuning/cv_folds.json`). **Test prohibido** hasta la
  comparación final única.
- Comparación **emparejada por fold** contra la referencia del **mismo número de épocas**.
- Suelo de ruido medido en Fase 1: **σ = 0,0097** (Dice por fragmento, 12 épocas, 4 semillas, cada
  modelo en su posproceso óptimo). A 20 épocas se mide con el control `c00b_seed1337`@20 de P4.
- Un cambio **aporta** solo si: (a) supera **2σ**, (b) gana en la **mayoría de los folds** y (c) **no
  empeora el IoU por fragmento**.
- Se reportan TODAS las métricas del §5 en cada decisión, también las que empeoran, y los
  experimentos negativos con la misma visibilidad que los positivos.

---

## F2A — Sin transfer learning

- **id:** `F2A`  ·  **línea:** A (Fase 2 §3.A)  ·  **carpeta:** `reports/tuning/fase2/F2A/`
- **Hipótesis:** el transfer learning del backbone (`checkpoints/fundidora_taller3.pth`, pesos del
  Taller 3) **no mejora** el Dice por fragmento. La semana 9 lo midió por región, en test y con una
  sola semilla (`reports/decisiones_de_diseno.md` §9); nunca se midió su efecto sobre la separación
  de fragmentos ni por CV.
- **Qué cambia:** `model.pretrained: false`. Config ya existente: `configs/ablation_scratch.yaml`
  (hereda `base.yaml` y cambia solo esa clave). **Nada más cambia.**
- **Por qué lo permite el enunciado:** §4.1 dice "Se permite transfer learning **únicamente** en el
  backbone"; es un permiso, no una obligación. La Fase 2 §2 lo confirma: "No es obligatorio; probar
  sin él está permitido y se pide". Además el enunciado §7 exige la ablación con/sin TL en el informe.
- **Diseño:** 5 folds × 20 épocas, semilla 42, λ recalibrados en cada fold. Emparejado contra
  `c00_base`@20 × 5 folds (P1), que es exactamente la misma config con `pretrained: true`.
- **Métrica primaria:** Dice por fragmento (cada modelo en su propio óptimo de posproceso, grilla
  idéntica `seed_depth_mm` 3/4/5/6 × `edge_threshold` 0,1/0,2/0,3/0,5).
- **Regla de decisión:** el TL se declara prescindible si |Δ| < 2σ (es decir, sin TL ≈ con TL) o si
  Δ > 0 (sin TL mejor). Si sin TL es **peor** por más de 2σ, el TL se queda y la línea A se cierra.
- **Qué desbloquea si sale favorable:** variantes de arquitectura que no tienen que respetar la forma
  de los pesos de FundidoraPC (5 canales de contexto 2.5D en vez de 3, anchos distintos), que es lo
  que la Fase 2 §3.A pide justificar con esta medición.
- **Coste:** 5 corridas × ~1,75 h = ~8,8 h de GPU (~4,4 h de reloj con los 2 carriles).
- **Estado:** pre-registrado 2026-10-04 08:55. **Encolado cuando P4 libere los carriles.**

---

## F2B1 — Representación núcleo/borde predicha directamente (en vez de borde binario)

- **id:** `F2B1`  ·  **línea:** B (Fase 2 §3.B)  ·  **carpeta:** `reports/tuning/fase2/F2B1/`
- **Hipótesis y el número de la Fase 1 que la motiva:** hoy el núcleo se **deriva** restando el borde
  predicho a la región (`core = región & (P(borde) < umbral)`), así que la separación de fragmentos
  **depende del recall del borde, que es 0,193**. Con el borde real el Dice por fragmento sube de
  0,743 a 0,902 (+0,159; `pp_hib_predsem_gtedge`). La hipótesis es que **predecir el núcleo como
  clase propia elimina esa dependencia**: la red aprende a dejar hueco entre fragmentos en vez de
  tener que marcar una superficie fina de 1 vóxel de cada 83.
- **Qué cambia:** una salida nueva de 3 clases (fondo / núcleo / borde) en la cabeza de segmentación,
  entrenada con CE ponderada + Dice, y un método de posproceso que toma las **componentes conexas del
  núcleo predicho** como semillas (en vez de erosionar por distancia). La semántica de 4 clases, la
  cabeza de clasificación y la de detección **no se tocan**.
- **Por qué lo permite el enunciado:** §3.2 pide "combinar una salida semántica con una salida de
  instancia" y seguir "una lógica de dos etapas: primero las regiones anatómicas y después los
  fragmentos, **de acuerdo con la lógica descrita en la literatura del challenge**". La
  representación núcleo/borde es exactamente la del ganador de PENGWIN 2024 (MIC-DKFZ), que es esa
  literatura. Sigue siendo 2D corte a corte, un solo backbone, CBAM antes de la bifurcación y tres
  cabezas. No usa YOLO/Detectron2/Mask R-CNN ni ningún framework de alto nivel.
- **Métrica primaria:** Dice por fragmento, emparejado por fold contra `c00_base` al mismo número de
  épocas.
- **Regla de decisión:** aporta solo si Δ > 2σ, gana en la mayoría de los folds y no empeora el IoU
  por fragmento. Guardarraíl: F1, AUC, mAP y Dice de región no caen más de 2σ de su propia escala.
- **Requisitos antes de entrenar en serio** (Fase 2 §4.4): justificación escrita (arriba), prueba de
  overfit de 8 cortes en verde, tests con phantoms del código nuevo y suite completa en verde.
- **Estado:** pre-registrado 2026-10-04 09:00. Implementación delegada a un subagente; el
  entrenamiento lo encola el coordinador cuando P4 libere los carriles.

---

## Nota sobre F2B1 tras el oráculo (2026-10-04 21:30)

El oráculo de `core3` con núcleo del GT (`pp_oraculo_core3`) **refuta la hipótesis que motivaba
F2B1** antes de que terminen sus 5 folds: sin erosión (`core_seed_depth_mm: 0`) el techo es
**0,8464**, por debajo del objetivo 0,85 del enunciado **incluso con un núcleo perfecto**, y coincide
con el techo del método `edge` (0,847). Con 1,5 mm de erosión sube a 0,9952.

**Qué queda en pie y qué no:**
- **Cae:** "predecir el núcleo como clase propia hace innecesaria la erosión por distancia".
- **Sigue en pie:** que un núcleo predicho mejor que el derivado del borde permita **bajar** la
  erosión de 5 mm (lo que necesita hoy) hacia 1,5-2 mm, que es donde está el Dice alto. Esa es la
  hipótesis que de verdad mide el barrido `D/350`, cuya grilla recorre `core_seed_depth_mm` 0/1/2/3.
- **No se cambia la regla de decisión** de F2B1 (2σ + mayoría de folds + no empeorar IoU). Lo que se
  corrige es la expectativa, no el criterio. El experimento sigue siendo confirmatorio, no
  exploratorio: la métrica primaria y la regla eran y siguen siendo las mismas.

---

## F2B2 — Regresión del mapa de distancia a la superficie de fractura

- **id:** `F2B2`  ·  **línea:** B (Fase 2 §3.B)  ·  **carpeta:** `reports/tuning/fase2/F2B2/`
- **Hipótesis, con los números que la motivan:** el oráculo muestra que la separación funciona si se
  umbraliza una **geometría** correcta: con el borde/núcleo del GT y 1,5 mm de erosión el Dice por
  fragmento es **0,9952**, y sin erosión solo **0,8464**. Con el borde **predicho** hace falta 5 mm
  de erosión (y ahí mata los fragmentos pequeños) porque la cabeza de borde solo cubre el **19 %** de
  la superficie de fractura. La causa raíz de ese recall bajo es el desbalance: el borde dilatado es
  **1 vóxel de cada 83** del hueso. Un **mapa de distancia** a la superficie de fractura es un
  objetivo **denso** —cada vóxel de hueso tiene valor— así que elimina el desbalance de clases en
  vez de compensarlo con `pos_weight`, y en inferencia el núcleo es exactamente la operación que el
  oráculo demuestra que funciona: `núcleo = distancia_predicha > umbral`.
- **Qué cambia:** la cabeza de borde pasa de clasificar un borde binario a **regresar** la distancia
  (en mm, recortada a un máximo configurable, p. ej. 8 mm) de cada vóxel de hueso a la superficie de
  fractura más cercana. Pérdida: L1 o Huber enmascarada al hueso, **dentro del término `seg`** (la
  pérdida sigue siendo de 3 términos). Posproceso: `method="dist"`, con semillas =
  componentes 3D de `distancia_predicha > seed_depth_mm`. La semántica de 4 clases, la cabeza de
  clasificación y la de detección **no se tocan**.
- **Por qué lo permite el enunciado:** §3.2 pide dos etapas (región → fragmentos) "de acuerdo con la
  lógica descrita en la literatura del challenge"; la regresión de distancia al borde es
  explícitamente una de las representaciones de esa literatura y la Fase 2 §3.B la lista como opción.
  Sigue siendo 2D corte a corte, un backbone, CBAM antes de la bifurcación, tres cabezas, grid y NMS
  propios, y nada de frameworks de alto nivel.
- **Métrica primaria:** Dice por fragmento, emparejado por fold contra `c00_base`@20.
- **Regla de decisión:** aporta solo si Δ > 2σ (σ por fold emparejado = **0,0130** a 20 épocas,
  medido con el control de P4), gana en la mayoría de los folds y no empeora el IoU por fragmento.
  Guardarraíl: F1, AUC, mAP y Dice de región no caen más de 2σ.
- **Requisitos antes de entrenar:** overfit de 8 cortes en verde, tests con phantoms y suite completa
  en verde (Fase 2 §4.4).
- **Relación con F2B1:** son **alternativas** para el mismo cuello de botella. Se implementa ahora en
  paralelo (es trabajo de CPU, no compite por GPU) para no dejar los carriles parados cuando F2B1
  entregue su resultado. La decisión de entrenarlo se toma **después** de ver F2B1.
- **Estado:** pre-registrado 2026-10-05 00:35. Implementación delegada a un subagente.

---

## F2C1 — Semillas por h-máxima en vez de umbral global de profundidad

- **id:** `F2C1`  ·  **línea:** posproceso guiado por distancia (TAREA §6; Fase 2 §3 permite priorizar
  por ganancia esperada y coste)  ·  **carpeta:** `reports/tuning/fase2/F2C1/`
- **Hipótesis, con los números que la motivan:** el oráculo dice que la separación casi perfecta
  (0,9952) se logra erosionando **1,5 mm**, y que a **5 mm** —lo que el modelo real necesita— se cae
  a 0,880. El problema es que `seed_depth_mm` es **un umbral global** y tiene que servir a la vez a
  dos casos incompatibles:
  - dos fragmentos grandes fusionados, que **solo** se separan erosionando profundo;
  - un fragmento de < 5 cm³, cuya profundidad máxima interior es de pocos mm y que **muere** a 5 mm
    (medido en Fase 1: Dice 0,000 en el estrato < 5 cm³).
  Las **h-máximas** de la transformada de distancia dan **una semilla por cada "montículo"**,
  independientemente de su profundidad absoluta: el fragmento pequeño conserva su semilla y los dos
  grandes siguen separándose. Es la misma familia de técnicas que ya usa el proyecto (watershed
  guiado por distancia, componentes conexas) y que TAREA §6 señala como práctica de nnU-Net y de los
  ganadores de PENGWIN.
- **Qué cambia:** solo el **posproceso**. Un `method="hmax"` que calcula `d = EDT(núcleo)` y toma como
  semillas las h-máximas de `d` con dinámica `h_mm`, en vez de las componentes de `d > seed_depth_mm`.
  No cambia el modelo, ni la pérdida, ni las métricas, ni los splits. **Cuesta 0 de GPU**: se evalúa
  sobre las predicciones OOF que ya existen.
- **Por qué lo permite el enunciado:** §3.2 pide la lógica de dos etapas "de acuerdo con la lógica
  descrita en la literatura del challenge"; el watershed por h-máximas de la transformada de
  distancia es exactamente eso y ya es el método del repositorio, solo cambia cómo se eligen las
  semillas. No toca arquitectura ni introduce frameworks externos.
- **Métrica primaria:** Dice por fragmento, emparejado por fold contra el mejor posproceso actual
  (`edt`, dep 5 / thr 0,1) sobre **el mismo** OOF de `c00_base`@20. Al ser el mismo modelo, la
  comparación **no tiene ruido de entrenamiento**: el único término que varía es el posproceso.
- **Regla de decisión:** aporta si Δ > 0 de forma consistente (mayoría de folds) y no empeora el IoU.
  Al no haber ruido de semilla, el umbral de 2σ no aplica: se usa la desviación entre folds del
  propio Δ emparejado. Guardarraíl: el Dice del fragmento principal no cae más de 0,01 (una
  sobre-segmentación partiría el principal).
- **Primer paso, antes de evaluar sobre el modelo real:** medir su **techo** con el oráculo (GT), como
  en los demás casos. Si con geometría perfecta no supera al `edt` de 1,5 mm, no hay nada que ganar.
- **Estado:** pre-registrado 2026-10-05 17:00. Implementación propia (los subagentes agotan la cuota).

---

## F2C2 — Híbrido: semillas `edt` + rescate por h-máxima donde `edt` no encontró ninguna

- **id:** `F2C2`  ·  **línea:** posproceso  ·  **carpeta:** `reports/tuning/fase2/F2C2/`
- **Hipótesis, con los números de F2C1 que la motivan:** las h-máximas **sí** recuperan secundarios
  (+7,10 pp, t = +2,29; Dice secundario +0,0775, t = +2,61, ambos en 4/5 folds) pero **parten el
  fragmento principal** (Dice principal −0,0434, t = **−9,59**, 0/5 folds) porque meten varias
  semillas dentro de un hueso grande con relieve interno. El umbral profundo de `edt` no tiene ese
  problema: da **una** semilla por zona profunda, respeta el principal, y lo que falla es que deja
  **sin** semilla a los fragmentos de pocos mm de profundidad.
  Los dos métodos fallan en sitios **complementarios**. El híbrido toma las semillas de `edt` y
  **añade** h-máximas solo en las componentes del núcleo que no contienen ninguna semilla de `edt`:
  el principal queda intacto y los fragmentos pequeños recuperan la suya.
- **Qué cambia:** solo el posproceso (`method="edt_hmax"`). **Cero GPU**: se evalúa sobre el OOF que
  ya existe de `c00_base`@20.
- **Por qué lo permite el enunciado:** sigue siendo watershed guiado por distancia sobre componentes
  conexas (§3.2, "la lógica descrita en la literatura del challenge"); no toca arquitectura, pérdida,
  métricas ni splits.
- **Métrica primaria:** Dice por fragmento, emparejado por fold sobre el mismo OOF contra el mejor
  `edt` actual (dep 5 / thr 0,1).
- **Regla de decisión (la misma de F2C1, sin cambios):** aporta si Δ > 0 en la mayoría de los folds,
  **no empeora el IoU** y **el Dice del fragmento principal no cae más de 0,01**. Ese guardarraíl es
  el que rechazó F2C1 y se mantiene idéntico: es lo que impide comprar secundarios a costa del hueso.
- **Primer paso:** techo con el oráculo. Si no supera a `edt` 1,5 mm (0,9952), se cierra.
- **Estado:** pre-registrado 2026-10-05 18:30. Si falla, **se cierra la Fase 2** (§6: dos líneas
  seguidas sin nada por encima del ruido y sin hipótesis con ganancia esperada medible).

### F2C2 — resultado sobre el OOF de `c00_base`@20 y réplica pre-registrada (2026-10-05 22:30)

Resultado (emparejado contra `edt` dep 5 / thr 0,1, mismo OOF, 5 folds). La grilla fue la del
trabajo `C/420` (dep 4/5/6 × h 1/1,5/2), fijada antes de ver resultados:

| combo | ΔDice frag | t | folds | ΔIoU | ΔDice principal | ΔMAE dist |
|---|---|---|---|---|---|---|
| dep 5 / h 1,0 | +0,0270 | +2,27 | 5/5 | +0,0157 | −0,0093 | +0,72 mm |
| dep 5 / h 1,5 | +0,0291 | +2,19 | 4/5 | +0,0165 | **−0,0100** (roza el guardarraíl) | +0,75 mm |
| dep 5 / h 2,0 | +0,0252 | +2,24 | 4/5 | +0,0134 | −0,0098 | +0,76 mm |
| dep 6 (cualquier h) | ≈ +0,022 | ≈ +1,3 | 3/5 | **< 0** | −0,037 | +1,1 mm |

Aplicando la regla tal cual (sin cambiarla): dep 5 / h 1,0 es la mejor combinación que cumple las
tres condiciones (Δ > 0 en 5/5, IoU no empeora, principal −0,0093 > −0,01). h 1,5 queda fuera por
4·10⁻⁵ en el guardarraíl. Coste reportado: el MAE de distancia sube +0,72 mm (5/5 folds).
Techo con oráculo: edt_hmax 0,9938 frente a edt 0,9893 (supera; cumple el primer paso).

**Réplica (pre-registrada ahora, antes de correrla):** como la combinación salió de una grilla de 9
sobre un único OOF, se confirma con el OOF de **otro entrenamiento** (`c00b_seed1337`@20, misma
config, semilla 1337) con la combinación fija dep 5 / h 1,0, contra su propio `edt` dep 5 / thr 0,1.
**Se adopta si:** ΔDice frag > 0 en ≥ 4/5 folds, ΔIoU ≥ 0 y ΔDice principal ≥ −0,01. Si falla,
F2C2 queda como exploratorio y no se adopta.
