# Borrador · entrada de IA_USAGE para la semana 12

`IA_USAGE.md` no está en esta rama (se borró en a0fe7f2). Este texto es para pegarlo como nueva entrada
con el formato del archivo. Las partes entre corchetes las completa el equipo: la IA no las sabe.

---

## Entrada [NNN] — Semana 12: sobreajuste de la cabeza de borde

- **Modelo:** Claude (Anthropic), en Claude Code. [Confirmar la versión que se usó en la sesión local.]
- **Fecha:** 16 de octubre de 2026.
- **Contexto adjunto:** el repositorio, `reports/train/v2_repro_history.csv` y la lista de lo ya probado
  (semanas 10 y 11).

**Prompt (resumen):** [pegar el prompt exacto]. Se pidió proponer remedios no probados contra el
sobreajuste de la cabeza de borde, implementarlos sin romper las reglas del enunciado, dejar un
pre-registro con reglas de decisión, un orquestador que corriera solo en ANTON y una tarea programada
en la nube que verificara los resultados y redactara notebook y documentos al terminar.

**Qué produjo la IA:**
1. Tres candidatos: `weight_decay` 1e-2 (`c11_wd1e2`), decodificador de 32 canales (`c12_segw32`) y
   decodificador U-Net++ con C0 y supervisión profunda (`c13_unetpp`, implementación propia citando
   Zhou et al., 2018).
2. `PREREGISTRO.md` con las reglas (μ + 2σ en el cribado; Δ ≥ 0,026, t > 2,776 y 4/5 folds en la
   confirmación) y `scripts/s12/orquestar.sh` + `decidir.py`.
3. Al terminar: verificación de `etapa1.json` recalculando desde los CSV crudos, el notebook
   `04_semana12_borde.ipynb`, la §15 de decisiones de diseño y `RESUMEN.md`.

**Qué salió:** ninguno pasó el cribado (Dice por fragmento 0,683 / 0,690 / 0,693 contra un umbral de
0,696). Quedan descartados; el modelo final no cambia y el test no se volvió a medir.

**Qué decidió el equipo a mano:** [completar: qué candidatos se aceptaron para probar, si se cambió
alguno, los umbrales del pre-registro, la revisión del código de U-Net++ antes de lanzar].

**Análisis crítico:**
- **Aciertos:** las reglas quedaron escritas antes de ver resultados y se aplicaron sin intervención; la
  verificación independiente cuadró con lo que decidió `decidir.py`.
- **Limitaciones:** la IA no ve los datos de CT ni la GPU; todo lo que afirma sobre las corridas sale de
  los archivos de `reports/`. La interpretación de que el límite es la etiqueta a 256 px es una
  hipótesis, no algo que esta semana demuestre.
- [Completar: errores de la IA que el equipo haya tenido que corregir.]
