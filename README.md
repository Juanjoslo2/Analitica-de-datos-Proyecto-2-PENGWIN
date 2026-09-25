# Analitica-de-datos-Proyecto-2-PENGWIN
Sistema multitarea para detección, segmentación y medición de separación en fracturas pélvicas usando TC (dataset PENGWIN).

> *El sistema desarrollado tiene exclusivamente fines académicos. No constituye un dispositivo médico, no ha sido validado clínicamente y no debe utilizarse para apoyar decisiones quirúrgicas reales.*

## Entorno

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .          # hace importable el paquete `pengwin` (03_src/pengwin)
python -m pytest          # 06_tests (los tests con datos se omiten si 01_data está vacío)
```

## Datos y partición

Los `.mha` de PENGWIN van en `01_data/PENGWIN_CT_train_images_part1|part2` y `01_data/PENGWIN_CT_train_labels` (no se versionan).
`01_data/splits.json` (v2, balanceada por patrón de fractura, `sha256` incluido) se regenera con:

```powershell
python -m pengwin.data.create_splits
```

## EDA (semana 8)

```powershell
python scripts/run_eda_extract.py      # ~30-60 s por caso, reanudable -> reports/eda/cache/*.json
jupyter lab 04_notebook/01_eda_pengwin.ipynb
```

El notebook consolida el caché en `reports/eda/eda_*.csv` (versionados), guarda las figuras en `reports/figures/eda/` y termina con la tabla de decisiones de diseño (§12).

## Visualizador 1 (MIP)

```powershell
python 03_src/pengwin/visualization/mip_viewer.py 001
```
