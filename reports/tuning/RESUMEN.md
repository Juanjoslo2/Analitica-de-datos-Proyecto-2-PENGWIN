# Ajuste objetivo de hiperparámetros — semana 10/11

Objetivo: subir las métricas del §5 del enunciado, sobre todo la **segmentación por fragmento**
(hoy Dice 0,725 / IoU 0,671 en test contra los objetivos 0,85 / 0,70), sin cambiar la arquitectura
ni ninguna restricción del enunciado.

## 0. Protocolo

- **El test (15 pacientes) no se usó para elegir nada.** Se mira una sola vez al final (§5 de este
  documento): configuración actual contra configuración ajustada.
- **Selección por validación cruzada por paciente**: `reports/tuning/cv_folds.json`, k = 5 sobre los
  85 casos de train+val, estratificado por número de fragmentos secundarios (0-1 / 2 / 3+).
  Generado con `scripts/cv_folds.py` (semilla 42).
- **Regla de decisión fijada antes de ver resultados:** si |Δ de la media entre folds| es menor que
  la desviación entre folds, **no hay ganador** y se queda la opción actual (la más simple).
  Métrica primaria: Dice por fragmento. Desempates: IoU por fragmento, luego % de secundarios
  recuperados. Guardarraíl: F1, AUC y mAP@[.50:.95] no pueden caer más de 0,005.
- **λ recalibrados** con `scripts/calibrate_lambdas.py` en cada candidato y cada fold (los candidatos
  que tocan L_seg cambian las normas de gradiente). Nunca se copió un λ.
- Se evalúa siempre `last.pth`, no `best.pth`: `selection_score` no incluye el Dice por fragmento y la
  configuración de referencia (`v2_last`) también es el último checkpoint.
- Todo el cómputo en ANTON (DGX Spark GB10), contenedor `nvcr.io/nvidia/pytorch:25.09-py3`, AMP activa,
  256 px, batch 8.

## 1. Punto de partida (ya medido, test, `v2_last` + posproceso actual)

| Métrica | Valor | Objetivo §5 |
|---|---|---|
| Dice por fragmento | 0,725 | ≥ 0,85 |
| IoU por fragmento | 0,671 | ≥ 0,70 |
| Dice principal / secundario | 0,930 / 0,489 | — |
| Secundarios recuperados | 51,3 % | — |
| Dice por volumen: < 5 / 5-20 / > 20 cm³ | 0,000 / 0,506 / 0,849 | — |
| mAP@0.50 / mAP@[.50:.95] / IoU de caja | 0,980 / 0,864 / 0,919 | ≥ 0,65 / ≥ 0,40 / ≥ 0,65 |
| F1 / AUC | 0,993 / 0,999 | ≥ 0,85 / ≥ 0,85 |
| MAE de distancia (separados / en contacto) | 1,08 / 2,40 mm | — |

Los objetivos de clasificación y detección ya están cumplidos con holgura: el margen de mejora está
en la **separación de fragmentos**, y dentro de ella en los secundarios de menos de 20 cm³.

## 2. Techo del posproceso (estudio oráculo)

Antes de gastar GPU conviene saber **de quién es la culpa** del Dice por fragmento bajo: ¿del modelo
(región y borde imperfectos) o del método de separación? Se construyó un "OOF oráculo"
(`~/pengwin_tuning/oracle_npz.py` → `/data/oof/oracle_gt`) con la **región y el borde tomados del
ground truth** del caché, y se corrió el mismo posproceso sobre los 85 casos de CV
(`reports/tuning/pp_oraculo_gt.{csv,json}`).

| Método | `seed_depth_mm` | Dice fragmento | IoU fragmento | Dice secundario | Secundarios recuperados |
|---|---|---|---|---|---|
| edt | 1,0 | 0,975 ± 0,006 | 0,970 | 0,955 | 96,1 % |
| edt | **1,5** | **0,995 ± 0,001** | **0,991** | **0,991** | **100,0 %** |
| edt | 2,0 | 0,986 ± 0,009 | 0,982 | 0,972 | 98,2 % |
| edt | 2,5 | 0,981 ± 0,011 | 0,975 | 0,962 | 96,9 % |
| edt | 3,0 | 0,972 ± 0,006 | 0,963 | 0,947 | 96,2 % |
| edt | 4,0 | 0,919 ± 0,038 | 0,901 | 0,849 | 87,7 % |
| edt | **5,0 (actual)** | **0,880 ± 0,049** | 0,850 | 0,785 | 81,0 % |
| edt | 6,0 | 0,815 ± 0,035 | 0,770 | 0,680 | 69,4 % |
| edt | 8,0 | 0,653 ± 0,030 | 0,598 | 0,390 | 35,0 % |
| **edge** | — (sin erosión) | **0,847 ± 0,027** | 0,833 | 0,712 | 71,4 % |

(`seed_min_cm3` fijo en 0,02; con el borde perfecto 0,005 vs 0,02 cambia ≤ 0,005, dentro de la
desviación. El método `edge` es insensible a `min_fragment_cm3` entre 0,02 y 0,1.)

Lecturas:

1. **El método `edt` no es el techo: el parámetro sí.** Con `seed_depth_mm = 1,5` el posproceso
   reproduce el GT casi exacto (0,995, 100 % de secundarios). No hace falta implementar otro método
   para llegar al objetivo de 0,85.
   - El óptimo en 1,5 mm no es monótono: por debajo (1,0 mm, menos que el píxel de 1,27 mm) los
     puentes finos del núcleo sobreviven y dos fragmentos reales comparten semilla (0,975); por
     encima de 2,5 mm empiezan a morir los fragmentos delgados.
   - **El método `edge` sí tiene techo: 0,847, por debajo del objetivo de 0,85 incluso con borde
     perfecto.** Esto justifica con cifras y por CV el `instance_method: edt` que ya estaba elegido:
     el watershed guiado por distancia (−d + w·borde) reparte mucho mejor que el guiado solo por el
     borde.
2. **El `seed_depth_mm = 5` que hay hoy cuesta 0,11 de Dice por fragmento incluso con un modelo
   perfecto**, y 18 puntos de secundarios recuperados. Fue elegido en val con el modelo real, donde
   un borde ruidoso penaliza las semillas pequeñas; pero nunca se probó por CV ni por debajo de 2 mm.
3. `seed_min_cm3` casi no mueve nada con el borde perfecto (Δ ≤ 0,005, dentro de la desviación): su
   papel es filtrar semillas espurias, así que solo puede importar con el borde **predicho**.
4. Esto fija la grilla del §3: hay que barrer `seed_depth_mm` en 1-6 mm y no dar por bueno el 5
   actual. Con el borde **predicho** el óptimo será mayor que 1,5 (un borde incompleto deja puentes
   que solo rompe una erosión más profunda, y un borde ruidoso crea semillas espurias), así que la
   grilla se barre completa en vez de saltar al óptimo del oráculo.

El error de distancia del oráculo es 0,0 mm (entra GT, sale GT): confirma que la medición con
`distance_transform_edt` y el spacing no añade error propio.

## 3. Posproceso

Sin reentrenar: todas las combinaciones se evalúan sobre las mismas predicciones fuera de fold
(`scripts/cv_predict.py` → `/data/oof/...`), en la grilla del modelo y contra el `label.npy` del caché.

### 3.1 El `seed_depth_mm` no es lo que parecía

El §2 decía que con borde perfecto el óptimo está en 1,5 mm. Con el borde **predicho** el óptimo se
**invierte** y se va a 4-6 mm (lectura temprana, folds 0-1, 28 combinaciones,
`reports/tuning/pp1_base_e20_f01.{csv,json}`):

| `seed_depth_mm` | Dice fragmento (borde predicho) | Dice fragmento (borde del GT, §2) |
|---|---|---|
| 1,5 | 0,558 ± 0,060 | **0,995** |
| 2 | 0,619 ± 0,023 | 0,986 |
| 3 | 0,673 ± 0,006 | 0,972 |
| **4** | **0,714 ± 0,006** | 0,919 |
| 5 (actual) | 0,698 ± 0,030 | 0,880 |
| 6 | 0,708 ± 0,043 | 0,815 |

**Por qué se invierte.** La erosión por distancia *separa*: `d > seed_depth_mm` rompe el núcleo en
trozos. Con el borde perfecto el núcleo ya viene partido por el propio borde, así que erosionar solo
puede hacer daño (mata los fragmentos delgados). Con el borde predicho el núcleo llega **fusionado**
—la región rellena la grieta y el borde no se detecta—, así que la erosión es lo **único** que
separa, y hace falta mucha más profundidad.

Es decir: **`seed_depth_mm` no es un umbral de ruido, es un detector geométrico de grietas que está
compensando una cabeza de borde floja.** Dos corroboraciones del mismo diagnóstico:

- `edge_threshold` es casi irrelevante (≤ 0,004 entre 0,1 y 0,5): el borde predicho casi nunca pasa
  de 0,1, así que el umbral no llega a recortar el núcleo.
- Con profundidad 1 mm solo se recupera el 12 % de los secundarios: el núcleo entero queda como una
  sola semilla por región.

**Consecuencia para el plan:** mejorar el borde (§4) y bajar `seed_depth_mm` son la **misma** palanca.
Los 0,11 de Dice que el §2 deja sobre la mesa no se cobran ajustando el posproceso por separado; por
eso el posproceso se reajusta sobre el modelo ganador antes de tocar el test, y cada candidato de §4
se compara con su propio `seed_depth_mm` óptimo (comparar todos a 5 mm penalizaría justo a los que
mejoran el borde).

### 3.2 Decisión del posproceso: NO HAY GANADOR, no se cambia nada

Tres etapas sobre los 5 folds (`pp1_base_e20`, `pp2_base_e20_refina`, `pp3_base_e20_seedmin`).
La desviación entre folds del Dice por fragmento es **0,045**: esa es la barra.

| Etapa | Parámetro | Mejor | Segundo | Actual | Δ mejor−actual | ¿Gana? |
|---|---|---|---|---|---|---|
| 1 | `seed_depth_mm` × `edge_threshold` | 5 / 0,1 → 0,7453 | 5 / 0,3 → 0,7430 | **5 / 0,2 → 0,7429** | +0,0024 | no (0,05 σ) |
| 2 | `seed_depth_mm` × `edge_weight` | 5,5 / cualq. → 0,7474 | 5 / 20 → 0,7430 | **5 / 5 → 0,7430** | +0,0044 | no (0,10 σ) |
| 3 | `seed_min_cm3` | 0,002 → 0,7441 | 0,01 → 0,7430 | **0,02 → 0,7430** | +0,0011 | no (0,02 σ) |

En las etapas 2 y 3 el "mejor" por Dice **pierde en los dos criterios de desempate**: la etapa 2 da
IoU 0,6783 contra 0,6807 y 57,9 % de secundarios contra 59,5 %; la etapa 3 da IoU 0,6755 contra
0,6807. Cuando el primer puesto cae en ambos desempates, no hay señal: es ruido de muestreo.

**Los cinco parámetros se quedan como estaban**, ahora con respaldo de CV en vez de una elección en
val:

| Parámetro | Valor | Qué dice la CV |
|---|---|---|
| `instance_method` | `edt` | **+0,245** de Dice contra `edge` (0,7430 vs 0,4983); 59,5 % vs 4,9 % de secundarios |
| `seed_depth_mm` | 5 | óptimo entre 1 y 7; mejor/segundo/actual dentro de 0,0024 |
| `edge_threshold` | 0,2 | irrelevante: 0,0055 entre 0,1 y 0,5 |
| `edge_weight` | 5 | **inerte**: ≤ 0,0002 entre 0 y 20 |
| `seed_min_cm3` | 0,02 | en lo alto de la meseta; degrada de forma monótona por encima |

**El ajuste del posproceso no produjo ninguna mejora.** Lo que aporta es (a) evidencia por validación
cruzada de que las elecciones del equipo eran correctas y (b) la medición de que `edt` vale +0,245,
diez veces más que cualquier hiperparámetro de esta campaña.

Aviso metodológico: la lectura temprana con folds 0-1 daba `seed_depth_mm` 4 como ganador
(0,714 ± 0,006 contra 0,698 del 5). Con los 5 folds, el 4 queda **0,040 por debajo** del 5. La
desviación pequeña con 2 folds era un artefacto de tener un grado de libertad, no precisión.

### 3.3 ¿De quién es la culpa? Región contra borde

Mezclando predicción y *ground truth* se reparten los 0,252 que separan el modelo del oráculo
(5 folds, mejor profundidad de cada fila; `pp_hib_*`):

| Región | Borde | Dice fragmento | Ganancia | `seed_depth_mm` óptimo | rec. secundarios |
|---|---|---|---|---|---|
| predicha | predicho | 0,7430 ± 0,0449 | — (real) | 5 | 59,5 % |
| **GT** | predicho | 0,7697 ± 0,0368 | +0,027 | 5 | 59,6 % |
| predicha | **GT** | **0,9024 ± 0,0110** | **+0,159** | **3** | **90,6 %** |
| GT | GT | 0,9952 ± 0,0014 | +0,252 | 1,5 | 100 % |

- **Arreglar el borde vale 6 veces más que arreglar la región.**
- La debilidad conocida "la región rellena las grietas" es real pero **menor**: con región perfecta el
  Dice del principal sube de 0,920 a 0,947, pero los secundarios se quedan en 0,578 y la recuperación
  en 59,6 %, idéntica al OOF real. La región afecta al contorno, no a si los fragmentos se separan.
- Las dos correcciones son **superaditivas** (0,027 + 0,159 = 0,186 contra 0,252 juntas): un buen
  borde solo rinde del todo con una buena región, porque permite bajar la erosión y dejar de
  sacrificar fragmentos delgados.

**Cómo falla el borde** (85 casos, contra `edge.npy` dentro del hueso): P(borde) media sobre los
vóxeles de borde reales = 0,2016; máxima por caso = 0,957.

| umbral | precisión | recall |
|---|---|---|
| 0,05 | 0,474 | **0,213** |
| 0,20 | 0,493 | 0,193 |
| 0,50 | 0,509 | 0,176 |

**El fallo es de cobertura, no de calibración:** la cabeza marca ~20 % de la superficie de fractura, y
bajar el umbral de 0,5 a 0,05 sube el recall solo de 0,176 a 0,213 — el 80 % que falta está en ≈ 0,
no "justo por debajo del umbral". Esto explica de una vez las tres anomalías del §3.2:
`edge_threshold` irrelevante (probabilidad bimodal), `edge_weight` inerte (mapa casi vacío) y
`seed_depth_mm` 5 necesario (hay que separar sin ayuda del borde).

## 4. Entrenamiento

### 4.1 Diseño del cribado

11 candidatos (`configs/tuning/c00..c10*.yaml`) a 12 épocas sobre el **fold 0**, con λ recalibrados en
cada uno. Cada candidato se evalúa con **su propio** posproceso óptimo, barriendo una grilla
**idéntica** para todos (`seed_depth_mm` 1,5/2/3/4/5/6 × `edge_threshold` 0,1/0,2/0,3/0,5): comparar
todos a 5 mm penalizaría a los que mejoran el borde, porque la profundidad óptima depende de la
calidad del borde (§3.1).

Coste de cribar a 12 en vez de 20 épocas (misma config, mismo fold): mAP@[.50:.95] 0,828 vs 0,839,
Dice de región 0,9595 vs 0,9619, pérdida de borde 0,565 vs 0,536. El cribado parte de un borde algo
peor, de forma **uniforme** para todos.

### 4.2 El control que decide: cambiar solo la semilla

`c00b_seed1337` y `c00c_seed2024` son `base.yaml` con otra `seed`. **Nada más cambia.**

| semilla | Dice con posproceso FIJO (dep 4 / thr 0,3) | Dice en su propio óptimo | profundidad óptima |
|---|---|---|---|
| **42** (la del base) | **0,6925** | **0,6925** | **4** |
| 1337 | 0,6626 | 0,6700 | 6 |
| 2024 | 0,6568 | 0,6770 | 6 |
| 7 | 0,6617 | 0,6746 | 6 |
| media | 0,6684 | 0,6785 | — |
| **σ (n = 4)** | **0,0163** | **0,0097** | — |

Tres cosas, y una obliga a matizar el resto del informe:

1. **Reoptimizar el posproceso absorbe ruido:** σ baja de 0,0163 a 0,0097 cuando cada modelo usa su
   propia profundidad. El σ relevante para la tabla del §4.3 (cada candidato en su óptimo) es
   **0,0097**; afirmar una diferencia pediría ~2 σ = **+0,019**.
2. **Tres de las cuatro semillas prefieren profundidad 6; solo la 42 prefiere 4.** El `seed_depth_mm`
   "óptimo" del modelo base es él mismo un accidente de la semilla, lo que relativiza la decisión del
   §3.2: el 5 que se conservó era indistinguible del 4 y del 6 por folds, y ahora también por semilla.
3. **La semilla 42 —la del base— es la más alta de las cuatro** (+0,0140 sobre la media en el óptimo
   propio). La referencia contra la que se midieron los 11 candidatos es un sorteo afortunado. La
   comparación emparejada sigue siendo válida (todos comparten esa semilla), pero **los Δ negativos
   de la tabla del §4.3 están inflados**: parte de ellos es la suerte del base, no un candidato peor.

Nota de método: una estimación previa con **un solo par** de semillas daba |Δ| = 0,0225-0,0299 y se
interpretó como σ. Es incorrecto: E|X₁−X₂| ≈ 1,13 σ, así que un |Δ| suelto sobreestima σ. Las cifras
de arriba (n = 3) son las que se usan.

### 4.3 Resultados del cribado (fold 0, mejor posproceso de cada uno)

σ = 0,0097 (§4.2) → para afirmar una diferencia hace falta ~2 σ = **0,0195**.

| candidato | Dice frag | Δ vs base | **Δ/σ** | IoU frag | rec. sec. |
|---|---|---|---|---|---|
| `c05` sobremuestreo ×3 | **0,7031** | +0,0106 | +1,09 | **0,6447** | 52,7 % |
| `c06` lr 1e-3 | 0,6963 | +0,0038 | +0,39 | 0,6197 | 52,7 % |
| `c03` borde dil. 1 | 0,6939 | +0,0014 | +0,14 | 0,6320 | 52,7 % |
| **`c00_base` (referencia)** | **0,6925** | — | — | 0,6296 | 47,3 % |
| `c01` `edge_pos_weight` 4 | 0,6908 | −0,0017 | −0,17 | 0,6270 | 47,3 % |
| `c08` dropout 0,2 | 0,6905 | −0,0020 | −0,21 | 0,6294 | 50,9 % |
| `c07` sin dropout espacial | 0,6891 | −0,0034 | −0,35 | 0,6222 | 50,9 % |
| `c04` borde dil. 3 | 0,6875 | −0,0049 | −0,51 | 0,6247 | 49,1 % |
| `c02` `edge_pos_weight` 16 | 0,6835 | −0,0090 | −0,92 | 0,6217 | 45,5 % |
| `c09` aumentación fuerte | 0,6805 | −0,0120 | −1,23 | 0,6182 | 49,1 % |
| **`c10` `backbone_lr_factor` 0,3** | **0,6503** | **−0,0421** | **−4,32** | 0,5743 | 41,8 % |
| réplica semilla 2024 | 0,6770 | −0,0155 | −1,59 | 0,6052 | 50,9 % |
| réplica semilla 7 | 0,6746 | −0,0179 | −1,83 | 0,5981 | 45,5 % |
| réplica semilla 1337 | 0,6700 | −0,0225 | −2,31 | 0,5914 | 45,5 % |

**Un único resultado significativo, y es negativo: `c10` empeora 0,0421 = 4,3 σ** y pierde en las tres
métricas. Frenar el backbone preentrenado a 0,3× del lr de las cabezas lo perjudica claramente: el
backbone necesita adaptarse por completo a esta tarea. El valor actual (`backbone_lr_factor: 1.0`)
queda **validado con evidencia**, no por omisión — un dato para la discusión de transfer learning.

**Los otros 10 candidatos caben en 0,0226** (0,6805-0,7031), prácticamente el mismo rango que cubren
las 4 semillas de la configuración **idéntica** (0,0225). Ninguno alcanza 2 σ.

Tres candidatos (`c05`, `c06`, `c03`) superan a las 4 réplicas de semilla. Bajo la hipótesis nula eso
le ocurre a ~1/5 de los candidatos (≈ 2 de 10 por azar), así que **que sean 3 no es evidencia**. Solo
`c05` añade coherencia: es el único que gana en la métrica primaria y en los dos desempates a la vez.

### 4.4 Por qué el ajuste de hiperparámetros no puede llegar al objetivo

Del oráculo (§3.3): subir el recall del borde de 0,164 a 1,0 compra +0,159 de Dice, o sea
**dDice/dRecall ≈ 0,19**. Con eso:

| candidato | Δ recall borde | Dice predicho | Observado | σ(semilla) |
|---|---|---|---|---|
| `c04` dil. 3 | +0,024 | +0,0046 | −0,0049 | 0,0097 |
| `c05` sobremuestreo | +0,022 | +0,0042 | +0,0106 | 0,0097 |

- El efecto esperado de los candidatos es **≈ 0,4 σ** (0,0042-0,0046 contra σ = 0,0097): harían
  falta del orden de **40-50 corridas emparejadas** para detectarlo con 80 % de poder.
- Para ganar los **+0,05** de Dice que acercarían el modelo al objetivo 0,85 desde 0,743, el recall
  del borde tendría que subir de 0,164 a **~0,42 (×2,6)**. El mejor candidato lo sube un **14 %**.
- `c05` mejora el **F1 del borde un 18 %** (precisión 0,4544 vs 0,3522 y recall 0,1866 vs 0,1643,
  mejorando ambas) **y eso aún no basta**.

**Conclusión: el cuello de botella no es un hiperparámetro mal puesto.** Es la capacidad de la cabeza
de borde para cubrir la superficie de fractura, y eso pide un cambio de diseño (más resolución de
salida, otra pérdida, supervisión 3D más rica), que queda fuera de lo que este encargo permite.

### 4.4b Precisión y recall del borde contra el Dice: asociación débil, no explicación

Con las 11 corridas que tienen borde y Dice medidos (fold 0, umbral 0,2 contra `edge.npy`):

| correlación de rangos con el Dice por fragmento | Spearman (n = 11) |
|---|---|
| precisión del borde | **+0,636** (marginal: el crítico al 5 % es ≈ 0,618) |
| recall del borde | −0,227 |

| corrida | precisión | recall | Dice frag |
|---|---|---|---|
| `c05` sobremuestreo ×3 | **0,4544** | 0,1866 | **0,7031** |
| `c06` lr 1e-3 | 0,3988 | 0,1711 | 0,6963 |
| `c03` borde dil. 1 | 0,3908 | 0,1152 | 0,6939 |
| `c00_base` | 0,3522 | 0,1643 | 0,6925 |
| `c01` posw 4 | 0,3652 | 0,1571 | 0,6908 |
| `c08` dropout 0,2 | 0,3233 | 0,1844 | 0,6905 |
| `c07` sin dropout | **0,4098** | 0,1650 | 0,6891 |
| `c04` borde dil. 3 | 0,3047 | 0,1882 | 0,6875 |
| `c02` posw 16 | 0,3436 | 0,1532 | 0,6835 |
| `c09` aumentación fuerte | 0,3179 | **0,1868** | 0,6805 |
| **`c00b` igual config, otra semilla** | **0,3627** | 0,1745 | **0,6700** |

**Dos contraejemplos que impiden leer esto como causa:**

1. **`c00b_seed1337` es la MISMA configuración con otra semilla**: tiene **más** precisión que el base
   (0,3627 vs 0,3522) y el **peor** Dice de la tabla (0,6700 vs 0,6925). Si la precisión fuese el
   motor, eso no podría ocurrir.
2. **`c07`** tiene la segunda mejor precisión (0,4098) y es séptimo en Dice.

**Conclusión honesta:** la precisión del borde se asocia con el Dice por fragmento bastante mejor que
el recall (+0,64 contra −0,23), pero **ninguna de las dos lo predice**: lo que domina es el ruido entre
corridas (§4.2). El mecanismo plausible —un falso positivo de borde se come el núcleo del fragmento y
la erosión por distancia no puede arreglarlo, mientras un falso negativo sí se compensa— es coherente
con el oráculo del §3.3 y con la irrelevancia de `edge_threshold`, pero **estos datos no lo
demuestran**.

Nota de método: una versión anterior de esta sección, calculada con 8 corridas, daba ρ = +0,905 y lo
presentaba como hallazgo firme. Añadir 3 corridas (una de ellas una réplica de semilla de la misma
configuración) lo bajó a +0,636 y aportó el contraejemplo. **La lección es recalcular con cada corrida
nueva antes de escribir una conclusión, no después.**

### 4.5 Hipótesis propias que fallaron

1. **`edge_pos_weight` 16 subiría el recall del borde.** Lo bajó (0,1532 vs 0,1643). La explicación de
   repuesto —que la recalibración de λ cancelaba el cambio— se midió y también es falsa (λ_seg se
   mueve ±0,01). Queda sin explicación con evidencia; no se inventa una.
2. **El recall del borde serviría para ordenar candidatos** (métrica de menos ruido que el Dice).
   Falso: `c03` tiene el peor recall (0,1152) y el mejor Dice de su grupo; `c04` el mejor recall y
   peor Dice. Sirve como evidencia del mecanismo, no como sustituto de la métrica primaria.
3. **La aumentación fuerte (`c09`) arreglaría el sobreajuste del borde.** A igual época, su pérdida de
   borde en val es **peor** que la del base (0,59 vs 0,52 en la época 7) con una brecha train-val algo
   menor: reduce el sobreajuste **empeorando el ajuste**, que es subajuste por aumentación excesiva.

## 5. Resultado final en test (una sola vez)

**Fase 1 cerrada sin configuración ganadora** (posproceso §3.2: ningún parámetro > desviación entre
folds; entrenamiento: el mejor, `c05`, queda en 1,61σ en P4 emparejado con 3/5 folds). La
configuración ajustada es la actual, así que el test no se volvió a evaluar: el resultado es el de
`v2_last` + `edt` ya medido (Dice por fragmento 0,725, IoU 0,671, mAP@0.5 0,980, mAP@[.5:.95]
0,864, IoU de caja 0,919, F1 0,993, AUC 0,999). La Fase 2 siguió desde aquí y tampoco produjo una
ganadora: ver `fase2/RESUMEN_FASE2.md` §5, con la distancia al objetivo de cada métrica.
