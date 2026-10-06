# Fase 2 — arquitectura y transfer learning (CERRADA 2026-10-05)

Objetivo: acercar o cumplir las métricas de **fragmento** del §5 del enunciado (Dice ≥ 0,85,
IoU ≥ 0,70) partiendo de 0,7453 ± 0,0451 (CV) y 0,725 (test), sin perder lo ya cumplido y **sin
enmascarar nada**.

**Veredicto: ningún cambio de la Fase 2 cumple su regla pre-registrada. La configuración final es la
actual (`base.yaml`, `v2_last`, `edt` dep 5 / thr 0,2). No hay `configs/tuned.yaml` ni
`checkpoints/tuned/`.** El objetivo de fragmento **no se alcanza**: faltan 0,125 de Dice y 0,029 de
IoU en test.

Protocolo (Fase 2 §4): pre-registro en `REGISTRO.md` antes de lanzar; selección solo por CV por
paciente; **test prohibido** para elegir; comparaciones **emparejadas por fold**; un cambio de
entrenamiento aporta solo si **supera 2σ, gana la mayoría de los folds y no empeora el IoU**, con
réplica de semilla antes de adoptarlo. Fuente de verdad: `EXPERIMENTOS.csv`.

## 0. Suelo de ruido

| escala | σ | cómo se midió |
|---|---|---|
| 12 épocas, 1 fold, por semilla | **0,0097** | 4 semillas de la config idéntica (Fase 1) |
| 20 épocas, emparejado por fold | **0,0130** | control `c00b_seed1337`@20 × 5 folds, Δ medio **−0,0000** |

Emparejar por fold baja el ruido de 0,045 (sin emparejar) a 0,0130: **3,5× más sensibilidad**.

## 1. Qué se probó, en qué orden y por qué

Orden por ganancia esperada medida en la Fase 1 (oráculo: con el borde real el Dice sube a 0,90) y
por coste.

| id | línea | qué cambia | Dice frag | IoU frag | Δ Dice vs ref | estadístico | folds | decisión |
|---|---|---|---|---|---|---|---|---|
| P1-base | ref | `base.yaml` (con TL), 20 ép. | 0,7453 | 0,6828 | — | — | — | referencia |
| P4-ctrl | control | misma config, semilla 1337 | 0,7446 | 0,6687 | −0,0000 | — | — | suelo de ruido |
| P4-c05 | D | sobremuestreo ×3 de cortes con secundarios | 0,7540 | 0,6939 | +0,0093 | 1,61σ | 3/5 | no se adopta |
| F2A | A | **sin transfer learning** (s42) | 0,7594 | 0,6969 | +0,0141 | t = 1,04 | 3/5 | réplica obligatoria |
| F2A-rep | A | sin TL, semilla 1337 | 0,7522 | 0,6903 | +0,0069 | t = 0,54 | 3/5 | **no confirma** |
| F2B1 | B | núcleo/borde 3 clases (`core3`) | 0,6729 | 0,6253 | **−0,072** | — | — | rechazado |
| F2B1-ctrl | B | mismo checkpoint, `edt` viejo | 0,7461 | 0,6737 | +0,0005 | 0,09σ | 4/5 | el modelo = base |
| F2B2 | B | regresión del mapa de distancia (`dist`) | 0,5039 | 0,4680 | **−0,241** | — | — | rechazado |
| F2B2-ctrl | B | mismo checkpoint, `edt` viejo | 0,7298 | 0,6686 | −0,0155 | t = −1,07 | 1/5 | modelo algo peor |
| ABL-noCBAM | ablación | sin CBAM | 0,7365 | 0,6744 | −0,0088 | t = −0,92 | 1/5 | ablación del informe |
| F2C1 | C (pp) | semillas por h-máxima | 0,7600 | 0,6839 | +0,0147 | t = 1,01 | 4/5 | rechazado: principal −0,043 |
| F2C2 | C (pp) | híbrido `edt` + h-máxima (dep 5, h 1,0) | 0,7723 | 0,6985 | +0,0270 | t = 2,27 | 5/5 | pasa; réplica |
| F2C2-rep | C (pp) | ídem sobre el OOF de semilla 1337 | 0,7635 | 0,6908 | +0,0199 | t = 2,35 | 4/5 | **no confirma**: principal −0,0112 |

Detección y clasificación (media de 5 folds, última época) para los entrenamientos:

| id | mAP@0.5 | mAP@[.5:.95] | IoU caja | F1 | AUC |
|---|---|---|---|---|---|
| P1-base | 0,9785 | 0,8438 | 0,9111 | 0,9905 | 0,9987 |
| P4-ctrl | 0,9772 | 0,8362 | 0,9079 | 0,9896 | 0,9984 |
| P4-c05 | 0,9812 | 0,8718 | 0,9217 | 0,9926 | 0,9990 |
| F2A | 0,9767 | 0,8312 | 0,9059 | 0,9881 | 0,9975 |
| F2A-rep | 0,9771 | 0,8254 | 0,9040 | 0,9900 | 0,9984 |
| F2B1 | 0,9791 | 0,8587 | 0,9168 | 0,9917 | 0,9989 |
| F2B2 | 0,9823 | **0,8925** | **0,9301** | 0,9934 | 0,9992 |
| ABL-noCBAM | 0,9788 | 0,8377 | 0,9094 | 0,9891 | 0,9986 |

Las variantes de posproceso (F2C*) no tocan detección ni clasificación: son las del modelo de origen.

## 2. Qué aportó de verdad y qué fue ruido

- **Línea A (sin TL): sin efecto detectable.** Agrupando las dos semillas (10 pares): Δ +0,0105,
  t = 1,19 (crítico 2,262), 6/10 folds. El efecto es específico de un fold y se repite entre
  semillas (fold 1: +0,063 y +0,050; folds 3-4 negativos en las dos): heterogeneidad entre
  pacientes, no una mejora global. **El TL del Taller 3 ni ayuda ni estorba en fragmentos.** En la
  primera lectura publiqué Δ/σ = +2,43 mezclando combos de posproceso; corregido en `PROGRESO.md`.
- **Línea B (borde/separación): dos negativos.** `core3` y `dist` empeoran con su propio método, y
  sus controles con `edt` muestran que el modelo entrenado es igual o algo peor que el base. Lo que
  falla es la representación: la erosión geométrica hace falta siempre (§3), y la regresión de
  distancia se sesga hacia "lejos de la fractura", el valor mayoritario.
  Efecto secundario: F2B2 sube mAP@[.5:.95] +0,049 e IoU de caja +0,019 (sin prueba emparejada ni
  réplica; no se usó para decidir).
- **Ablación sin CBAM: sin efecto medible** en fragmentos (−0,0088, t = −0,92). Se mantiene CBAM
  porque el enunciado lo exige.
- **Línea C (posproceso h-máxima):**
  - F2C1 recupera secundarios (+7,1 pp) pero parte el principal (−0,043, 0/5).
  - F2C2 es lo más cercano a una mejora real de toda la campaña: +0,020/+0,027 de Dice por
    fragmento en dos OOF independientes, con Dice secundario +0,05/+0,07 en 5/5 folds.
  - **No se adopta:** el guardarraíl pre-registrado (Dice principal ≥ −0,01) falla en la réplica
    (−0,0112).
  - Coste: el MAE de distancia sube +0,54/+0,72 mm en 5/5 folds.
  - Lectura exploratoria (no cuenta como decisión): h 1,5 pasa los tres criterios en la réplica
    (+0,0281, 5/5, principal −0,0096), pero falló el guardarraíl por 4·10⁻⁵ en el OOF original. El
    coste en el principal es estable en ≈ −0,010, justo en el umbral. Queda como pista documentada
    para el equipo.
- **Línea D (sobremuestreo c05):** 1,61σ en la Fase 1. Se debía combinar con lo que ganara en B, y
  B no dio nada.

Cierre por la Fase 2 §6: A y B consecutivas sin nada por encima del ruido, C sin confirmar, y no
queda ninguna hipótesis con ganancia esperada medible y barata. Las que quedan (decodificador de
borde a resolución completa, supervisión profunda) apuntan al mismo cuello de botella en el que ya
fallaron dos representaciones, y no hay un número de la Fase 1 que prediga que una tercera vaya a
funcionar.

## 3. Techos medidos antes de gastar GPU (oráculos)

| método | con GT | detalle |
|---|---|---|
| `edt`, erosión 1,5 mm | **0,9952** | el óptimo absoluto |
| `edt`, erosión 5 mm (lo que hoy hace falta) | 0,880 | |
| `edge` (sin erosión) | 0,847 | por debajo de 0,85 con datos perfectos |
| `core3`, sin erosión / 1,5 mm | 0,846 / 0,995 | misma lógica que `edge` / `edt` |
| `edt_hmax` (F2C2) | 0,9938 | contra 0,9893 de `edt` en la misma grilla |

Reparto de culpa (Fase 1): región GT +0,027, borde GT **+0,159**, ambos +0,252. El borde marca
~20 % de la superficie de fractura y el 80 % restante está en probabilidad ≈ 0.

## 4. Arquitectura final (sin cambios respecto a la semana 10)

```
corte 2.5D (3 canales, 256×256)
   │
FundidoraPC-R (residual, 4 etapas 32-64-128-256, TL del Taller 3)
   │   CBAM (canal + espacial) en las etapas 3 y 4  ← antes de la bifurcación
   ▼
cuello FPN (128 canales)
   ├── cabeza 1: clasificación (SA / coxal izq. / coxal der.)
   ├── cabeza 2: detección, grid propio stride 8 + NMS propio
   └── cabeza 3: segmentación en dos etapas
         región semántica (4 clases) + borde de fractura
         → posproceso `edt`: núcleo = región − borde, semillas d > 5 mm, watershed sobre −d
         → fragmentos coloreados por hueso (1-10 / 11-20 / 21-30)
pérdida = λcls·Lcls + λdet·Ldet + λseg·Lseg, λ por norma de gradiente (recalibrados por fold)
```

## 5. Resultado final en test (configuración actual = configuración final)

Como no hay configuración ganadora, la comparación "actual contra final" es la misma configuración y
**no se volvió a evaluar el test**: se reporta la única medición existente (`reports/eval/v2_last_test.json`,
`reports/fragments/v2_last_test_edt.json`).

| métrica §5 | test | objetivo | distancia |
|---|---|---|---|
| Dice por fragmento | 0,725 | ≥ 0,85 | **−0,125 (no se cumple)** |
| IoU por fragmento | 0,671 | ≥ 0,70 | **−0,029 (no se cumple)** |
| F1 (macro) | 0,993 | ≥ 0,85 | +0,143 |
| AUC (macro) | 0,999 | ≥ 0,85 | +0,149 |
| IoU de caja | 0,919 | ≥ 0,65 | +0,269 |
| mAP@0.5 | 0,980 | ≥ 0,65 | +0,330 |
| mAP@[.5:.95] | 0,864 | ≥ 0,40 | +0,464 |
| Dice principal / secundario | 0,930 / 0,489 | — | — |
| secundarios recuperados | 51,3 % | — | — |
| MAE de distancia | 2,20 mm | — | — |

## 6. Lo que NO funcionó, con la misma visibilidad

- 11 hiperparámetros de la Fase 1 (ninguno > ruido; `c10` empeora 4,3σ).
- Sobremuestreo `c05` (1,61σ).
- Sin TL (no confirma con réplica).
- Núcleo/borde `core3` (−0,072).
- Regresión de distancia `dist` (−0,241).
- h-máxima pura (parte el principal).
- Híbrido `edt_hmax` (falla el guardarraíl en la réplica).
- Correcciones mías publicadas: Δ/σ de F2A mal emparejado (§2) y la afirmación "F2A cumple la regla"
  (cumplía la letra; la prueba F mostró que el 2σ no le aplicaba).
