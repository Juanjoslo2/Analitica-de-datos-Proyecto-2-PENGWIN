# Encargo para la sesión en la nube · semana 12

Lo cumple una tarea programada en la nube (routine de claude.ai) cuando ANTON, la DGX Spark del equipo,
termina `scripts/s12/orquestar.sh`, es decir, cuando `reports/tuning/s12/ETAPA` de la rama `s12-spark`
empieza por «terminado». El usuario no está pendiente: trabaja solo y deja todo listo para que él lo revise. Proyecto: PENGWIN (fracturas pélvicas en
CT), Proyecto Integrador Corte 2, Analítica de Datos, UAO. La sustentación es el miércoles 21 de octubre.

## 0. Antes de empezar

```bash
git fetch origin s12-spark
git merge --no-edit origin/s12-spark          # trae los resultados de ANTON (solo toca reports/tuning/s12/)
```

Lee, en este orden: `reports/tuning/s12/PREREGISTRO.md` (qué se probó y con qué reglas, escrito antes
de los resultados), `reports/tuning/s12/ESTADO.md` y `bitacora.log` (qué pasó), `etapa1.json` y
`etapa2.json` si existe (lo que decidió `scripts/s12/decidir.py`), `resultados/`, `historiales/` y
`logs/`. Para el estilo de los notebooks y del informe, mira `04_notebook/03_semana11.ipynb` y
`reports/decisiones_de_diseno.md`.

Aquí no hay GPU ni datos de CT: todo se hace con lo que hay en `reports/`.

## 1. Verifica las decisiones (independiente de decidir.py)

Recalcula con pandas, desde los archivos crudos de `resultados/` y `referencia/`, lo que dice
`etapa1.json` (μ, σ, umbral, quién pasa) y, si existe, `etapa2.json` (Δ por fold, t, folds a favor,
criterios). Si algo no cuadra, dilo en el informe y explica cuál cifra es la correcta. No cambies la
decisión para que cuadre con lo que esperabas: la regla está escrita de antemano.

Revisa también en `logs/` que ninguna corrida haya fallado en silencio. Por ejemplo: épocas que faltan,
NaN, un `s12_ref` que se aleje de las cuatro referencias de la Fase 1 más de lo que explica el ruido
(σ ≈ 0,01) o tiempos por época absurdos.

## 2. Notebook nuevo: `04_notebook/04_semana12_borde.ipynb`

Corto y en español, con el mismo estilo de `03_semana11.ipynb` (misma función `find_root`, mismos
colores y solo lectura de `reports/`). Secciones:

1. **Qué preguntamos.** La cabeza de borde sobreajusta (curva train vs validación de `v2_repro`). Qué se
   había probado ya y qué faltaba: tabla corta tomada del pre-registro.
2. **Curvas de pérdida del borde, train contra validación**, para `s12_ref` y los tres candidatos
   (`historiales/*_history.csv`: `train_edge_bce`, `val_edge_bce`, `train_edge_dice`, `val_edge_dice`).
   Ojo: en `c13_unetpp`, `train_seg` es el promedio de la supervisión profunda; para comparar usa las
   columnas `*_edge_*`, que son siempre de la salida final.
3. **Cribado:** tabla y figura de Dice por fragmento contra la banda μ ± 2σ de las referencias, IoU,
   secundarios recuperados, precisión y recall del borde (`borde_*.json`). Di quién pasa y por qué.
4. **Confirmación** (si hubo): Δ por fold, t y criterios.
5. **Modelo final y test** (si hubo): comparación con `v2_last` + `tuned.yaml` (Dice 0,768 / IoU 0,697;
   `reports/fragments/v2_last_test_tuned.json`), declarada como la segunda medición de test.
6. **Lectura y decisión:** qué hicimos y qué salió, en pocas líneas. Si nada ganó, dilo así: queda
   descartado con evidencia, igual que `ftseg`. No suavices ni infles: cifras con su ruido.

Ejecútalo para que queden las salidas: `jupyter nbconvert --to notebook --execute --inplace`, con
pandas, matplotlib y nbconvert instalados por pip. Guarda las figuras en `reports/figures/semana12/`.

## 3. Documentos

- `reports/decisiones_de_diseno.md`: sección nueva **§15. Semana 12: sobreajuste del borde**, corta, con
  la tabla de candidatos y la decisión.
- `reports/tuning/s12/RESUMEN.md`: resumen de una página con las cifras.
- `04_notebook/03_semana11.ipynb`: en la tabla de "Decisiones de la semana y uso del test" (sección 11)
  añade una fila por candidato que apunte al notebook nuevo. Ejecuta solo si lo cambias.
- **Solo si hubo modelo final nuevo:** model card (`scripts/model_cards.py` o a mano en
  `reports/model_cards/`) y el README. Deja escrito que el checkpoint está en ANTON, en
  `~/pengwin_s12/checkpoints/s12_final_<candidato>/last.pth`, y que hay que copiarlo a mano para
  usarlo en el dashboard. Si no hubo modelo nuevo, no toques el model card ni el README.
- `IA_USAGE.md` falta en esta rama (se borró en el commit a0fe7f2) y el enunciado lo exige. No lo
  recrees. Deja un borrador de la entrada de esta semana en `reports/tuning/s12/IA_USAGE_entrada_s12.md`
  para que el equipo lo pegue: modelo usado (Claude), qué se pidió, qué propuso la IA, qué se probó y
  qué salió, y qué se decidió a mano.

## 4. Reglas

- **No elijas nada mirando el test.** El test solo aparece si la etapa 3 corrió, y se reporta tal cual.
- No toques el código de entrenamiento, `configs/base.yaml` ni `configs/tuned.yaml`.
- Textos cortos y directos ("qué hicimos y cómo"), sin relleno ni frases que suenen a IA, en español.
- Commits pequeños en tu rama, con mensajes en español. No abras PR contra otros repos.

## 5. Al final

Escribe `reports/tuning/s12/INFORME_NUBE.md` con lo esencial en 10-15 líneas:

- qué se probó y qué salió en cada etapa;
- si verificaste las decisiones y cuadraron;
- qué archivos cambiaste;
- qué queda pendiente para el usuario (por ejemplo, llevar los cambios a la rama `Nicolas` del repo del
  equipo, copiar un checkpoint, actualizar el informe en Word).

Sube todo a la rama `claude/nube-s12` con un archivo `reports/tuning/s12/NUBE_HECHO` (una línea con la
fecha): es la marca que hace que las siguientes corridas de la tarea programada salgan sin repetir el trabajo.
