# Semana 12 · estado en ANTON

Actualizado: 2026-10-16 10:52:24 (hora de ANTON)

**Etapa:** 1 · cribado (fold 0, 12 épocas)

| carril | estado | trabajo | detalle |
|---|---|---|---|
| A | running | `110_scr_c13_unetpp` | `s12_c13_unetpp_e12_f0`: 4 épocas hechas · 1294 s/época · val mAP@[.50:.95] 0.760 · Dice región 0.959 · BCE borde train/val 0.013/0.027 |
| B | running | `110_scr_ref` | `s12_ref_e12_f0`: 7 épocas hechas · 713 s/época · val mAP@[.50:.95] 0.789 · Dice región 0.958 · BCE borde train/val 0.011/0.031 |
| C | running | `110_scr_c11_wd1e2` | `s12_c11_wd1e2_e12_f0`: 7 épocas hechas · 713 s/época · val mAP@[.50:.95] 0.794 · Dice región 0.954 · BCE borde train/val 0.010/0.033 |
| D | running | `110_scr_c12_segw32` | `s12_c12_segw32_e12_f0`: 7 épocas hechas · 659 s/época · val mAP@[.50:.95] 0.796 · Dice región 0.955 · BCE borde train/val 0.011/0.025 |

## Máquina

```
95 %, 66, 26.42 W
total       usado       libre  compartido   búf/caché  disponible
Mem:             121          26          34           1          62          95
pengwin-s12-D Up 2 hours
pengwin-s12-C Up 2 hours
pengwin-s12-B Up 2 hours
pengwin-s12-A Up 2 hours
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
