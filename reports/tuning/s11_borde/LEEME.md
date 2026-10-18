# Cierre de la semana 11 · sobreajuste de la cabeza de borde

Corridas del 16 de octubre en la DGX Spark para el sobreajuste del borde (`PREREGISTRO.md`, `RESUMEN.md`,
`04_notebook/04_semana11_borde.ipynb`). Son parte de la semana 11: el curso no tiene semana 12.

- `resultados/`, `historiales/`, `lambdas/`: lo que dejó cada corrida. `referencia/`: las corridas de la
  Fase 1 contra las que se comparó.
- `etapa1.json` y `ETAPA`: lo que decidió `scripts/s11_borde/decidir.py` con las reglas del pre-registro y
  la etapa en la que terminó `scripts/s11_borde/orquestar.sh`.
- `logs/` y `bitacora.log`: registros crudos del orquestador, sin editar.

**Nombres.** Durante las corridas el lote se llamó `s12` (`s12_ref`, `s12_c11_wd1e2`...). Aquí se
renombró a `s11_` para que coincida con la semana del curso. Los logs crudos conservan el nombre
original, y las rutas `~/pengwin_s12` de los scripts son las de la máquina donde corrió.
