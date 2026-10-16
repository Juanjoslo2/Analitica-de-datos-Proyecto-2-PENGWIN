# Semana 12 · estado en ANTON

Actualizado: 2026-10-16 12:30:38 (hora de ANTON)

**Etapa:** 1 · cribado (fold 0, 12 épocas)

| carril | estado | trabajo | detalle |
|---|---|---|---|
| A | done | `110_scr_c13_unetpp` |  |
| B | done | `110_scr_ref` |  |
| C | done | `110_scr_c11_wd1e2` |  |
| D | done | `110_scr_c12_segw32` |  |

## etapa1

Referencia: media 0.6792 · σ 0.0086 · umbral 0.6964 · IoU ref 0.6087

- `c11_wd1e2`: Dice 0.6829279425858424 · Δ vs media 0.0037 (z = 0.43) · IoU 0.6204106521882091 · borde {'precision@0.2': 0.3551, 'recall@0.2': 0.1561} · **no pasa**
- `c12_segw32`: Dice 0.689911429456661 · Δ vs media 0.0107 (z = 1.25) · IoU 0.6345035981689588 · borde {'precision@0.2': 0.3714, 'recall@0.2': 0.1546} · **no pasa**
- `c13_unetpp`: Dice 0.6934261259484794 · Δ vs media 0.0142 (z = 1.66) · IoU 0.6356218786473106 · borde {'precision@0.2': 0.4332, 'recall@0.2': 0.1567} · **no pasa**

A confirmar: ninguno

## Máquina

```
1 %, 54, 9.36 W
total       usado       libre  compartido   búf/caché  disponible
Mem:             121           8          52           0          61         113
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
2026-10-16 12:30:38 etapa 1 decidida: a confirmar = 'ninguno'
```
