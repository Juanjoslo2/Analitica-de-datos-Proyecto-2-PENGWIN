# Semana 12 · estado en ANTON

Actualizado: 2026-10-16 11:22:27 (hora de ANTON)

**Etapa:** 1 · cribado (fold 0, 12 épocas)

| carril | estado | trabajo | detalle |
|---|---|---|---|
| A | running | `110_scr_c13_unetpp` | `s12_c13_unetpp_e12_f0`: 5 épocas hechas · 1076 s/época · val mAP@[.50:.95] 0.756 · Dice región 0.957 · BCE borde train/val 0.012/0.025 |
| B | running | `110_scr_ref` | `s12_ref_e12_f0`: 9 épocas hechas · 818 s/época · val mAP@[.50:.95] 0.811 · Dice región 0.957 · BCE borde train/val 0.009/0.033 |
| C | running | `110_scr_c11_wd1e2` | `s12_c11_wd1e2_e12_f0`: 9 épocas hechas · 818 s/época · val mAP@[.50:.95] 0.805 · Dice región 0.956 · BCE borde train/val 0.009/0.030 |
| D | running | `110_scr_c12_segw32` | `s12_c12_segw32_e12_f0`: 9 épocas hechas · 724 s/época · val mAP@[.50:.95] 0.812 · Dice región 0.957 · BCE borde train/val 0.009/0.031 |

## Máquina

```
93 %, 66, 25.80 W
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
