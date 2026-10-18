# Semana 11 · resumen

**Pregunta.** La cabeza de borde sobreajusta: en `v2_repro` la BCE del borde en train baja de 0,064 a
0,005 y la de validación sube de 0,018 a 0,034. ¿La cierra L2, un decodificador más pequeño o uno U-Net++?

**Cómo.** Reglas fijadas antes de entrenar (`PREREGISTRO.md`). Corrió solo en la DGX Spark
(`scripts/s11_borde/orquestar.sh`), cuatro carriles en paralelo, 09:22-12:30 del 16 de octubre.
Cribado en el fold 0 con 12 épocas, misma grilla de posproceso (`edt`, 24 combinaciones, sin TTA).

**Cribado** (Dice e IoU por fragmento, mejor combinación de la grilla):

| Corrida | Dice frag | z | IoU frag | Dice principal | Sec. recuperados | Borde P / R @0,2 | Pasa |
|---|---|---|---|---|---|---|---|
| referencias (4 semillas + `s11_ref`) | μ 0,6792 · σ 0,0086 | — | 0,6087 | — | — | — | umbral 0,6964 |
| `s11_ref` (control) | 0,6819 | 0,31 | 0,6193 | 0,921 | 45 % | 0,370 / 0,157 | — |
| `c11_wd1e2` (L2 ×100) | 0,6829 | 0,43 | 0,6204 | 0,921 | 47 % | 0,355 / 0,156 | no |
| `c12_segw32` (32 canales) | 0,6899 | 1,25 | 0,6345 | 0,926 | 51 % | 0,371 / 0,155 | no |
| `c13_unetpp` (U-Net++) | 0,6934 | 1,66 | 0,6356 | 0,930 | 53 % | 0,433 / 0,157 | no |

**Curvas del borde** (época final, fold 0): BCE train 0,007-0,008 en los cuatro; validación 0,031-0,034,
con mínimo en la época 0-1. La brecha baja como mucho 0,003 frente al control.

**Etapas 2 y 3.** No corrieron: nadie pasó el cribado. El test no se volvió a mirar.

**Decisión.** Los tres quedan descartados con evidencia. Modelo final sin cambios: `v2_last` +
`configs/tuned.yaml` (test: Dice por fragmento 0,768, IoU 0,697).

**Verificación.** Las cifras de `etapa1.json` se recalcularon con pandas desde los CSV y JSON crudos y
cuadran. Los cuatro historiales tienen 12 épocas y ningún NaN.
