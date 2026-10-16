# Semana 12 · estado en ANTON

Actualizado: 2026-10-16 11:52:31 (hora de ANTON)

**Etapa:** 1 · cribado (fold 0, 12 épocas)

| carril | estado | trabajo | detalle |
|---|---|---|---|
| A | running | `110_scr_c13_unetpp` | `s12_c13_unetpp_e12_f0`: 7 épocas hechas · 1188 s/época · val mAP@[.50:.95] 0.772 · Dice región 0.957 · BCE borde train/val 0.010/0.028 |
| B | running | `110_scr_ref` | `s12_ref_e12_f0`: 11 épocas hechas · 749 s/época · val mAP@[.50:.95] 0.820 · Dice región 0.958 · BCE borde train/val 0.008/0.033 |
| C | running | `110_scr_c11_wd1e2` | `s12_c11_wd1e2_e12_f0`: 11 épocas hechas · 749 s/época · val mAP@[.50:.95] 0.824 · Dice región 0.958 · BCE borde train/val 0.008/0.030 |
| D | running | `110_scr_c12_segw32` | `s12_c12_segw32_e12_f0`: 12 épocas hechas · 677 s/época · val mAP@[.50:.95] 0.826 · Dice región 0.959 · BCE borde train/val 0.008/0.031 |

## Máquina

```
93 %, 66, 24.55 W
total       usado       libre  compartido   búf/caché  disponible
Mem:             121          24          35           0          62          96
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
