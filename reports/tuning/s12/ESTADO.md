# Semana 12 · estado en ANTON

Actualizado: 2026-10-16 12:22:35 (hora de ANTON)

**Etapa:** 1 · cribado (fold 0, 12 épocas)

| carril | estado | trabajo | detalle |
|---|---|---|---|
| A | running | `110_scr_c13_unetpp` | `s12_c13_unetpp_e12_f0`: 11 épocas hechas · 366 s/época · val mAP@[.50:.95] 0.815 · Dice región 0.962 · BCE borde train/val 0.007/0.030 |
| B | done | `110_scr_ref` |  |
| C | done | `110_scr_c11_wd1e2` |  |
| D | done | `110_scr_c12_segw32` |  |

## Máquina

```
94 %, 66, 26.43 W
total       usado       libre  compartido   búf/caché  disponible
Mem:             121          14          46           0          62         107
pengwin-s12-D Up 3 hours
pengwin-s12-C Up 3 hours
pengwin-s12-B Up 3 hours
pengwin-s12-A Up 3 hours
```

Bitácora (últimas 15 líneas):

```
2026-10-16 09:22:12 orquestador arriba (DRY=0) en 744b375
carril A arriba (pengwin-s12-A)
carril B arriba (pengwin-s12-B)
carril C arriba (pengwin-s12-C)
carril D arriba (pengwin-s12-D)
2026-10-16 09:22:13 == 1 · cribado (fold 0, 12 épocas)
2026-10-16 09:22:13 encolado 110_scr_c13_unetpp en el carril A
2026-10-16 09:22:13 encolado 110_scr_ref en el carril B
2026-10-16 09:22:13 encolado 110_scr_c11_wd1e2 en el carril C
2026-10-16 09:22:13 encolado 110_scr_c12_segw32 en el carril D
```
