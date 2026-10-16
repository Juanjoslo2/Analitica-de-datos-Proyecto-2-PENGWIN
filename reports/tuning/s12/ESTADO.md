# Semana 12 · estado en ANTON

Actualizado: 2026-10-16 09:52:17 (hora de ANTON)

**Etapa:** 1 · cribado (fold 0, 12 épocas)

| carril | estado | trabajo | detalle |
|---|---|---|---|
| A | running | `110_scr_c13_unetpp` | `s12_c13_unetpp_e12_f0`: 1 épocas hechas · 1293 s/época · val mAP@[.50:.95] 0.668 · Dice región 0.951 · BCE borde train/val 0.106/0.016 |
| B | running | `110_scr_ref` | `s12_ref_e12_f0`: 2 épocas hechas · 680 s/época · val mAP@[.50:.95] 0.719 · Dice región 0.947 · BCE borde train/val 0.014/0.021 |
| C | running | `110_scr_c11_wd1e2` | `s12_c11_wd1e2_e12_f0`: 2 épocas hechas · 680 s/época · val mAP@[.50:.95] 0.732 · Dice región 0.941 · BCE borde train/val 0.014/0.018 |
| D | running | `110_scr_c12_segw32` | `s12_c12_segw32_e12_f0`: 2 épocas hechas · 649 s/época · val mAP@[.50:.95] 0.733 · Dice región 0.939 · BCE borde train/val 0.013/0.021 |

## Máquina

```
94 %, 65, 26.39 W
total       usado       libre  compartido   búf/caché  disponible
Mem:             121          26          34           1          62          95
pengwin-s12-D Up 30 minutes
pengwin-s12-C Up 30 minutes
pengwin-s12-B Up 30 minutes
pengwin-s12-A Up 30 minutes
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
