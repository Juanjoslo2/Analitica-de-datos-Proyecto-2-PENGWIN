# Analitica-de-datos-Proyecto-2-PENGWIN
**Integrantes**
> * Yáxul Santiago Cárdenas Hincapié
> * Nicolas Santiago Cuaran Sotelo
> * Juan Jose Orozco Lopez
> * David Felipe Martinez Viracacha

Sistema multitarea para detección, segmentación y medición de separación en fracturas pélvicas usando TC (dataset PENGWIN).

## Entorno

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .          # hace importable el paquete `pengwin` (03_src/pengwin)
python -m pytest          # 06_tests: 16 tests con phantoms sintéticos + 4 con datos (se omiten si 01_data está vacío)
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

## Decisiones de diseño

`reports/decisiones_de_diseno.md` es la referencia para implementar el modelo (entrada, detección, instancias, pérdida y prueba de overfit). `configs/base.yaml` contiene los mismos valores en formato ejecutable; las ablaciones (`configs/ablation_*.yaml`) heredan de él y cambian una sola cosa.

## Flujo de trabajo en Git (obligatorio desde la semana 9)

El enunciado exige historial de commits y **Pull Requests por integrante**, `main` protegida y merge solo tras revisión.

**1. Proteger `main`** (lo hace el dueño del repositorio, una vez): *Settings → Branches → Add branch ruleset* sobre `main`:
- *Require a pull request before merging*, con **1 aprobación** y *Dismiss stale approvals*.
- *Require status checks to pass*: seleccionar `tests / pytest`, disponible tras el primer PR.
- *Block force pushes*.
- **Sin bypass para administradores**: si no, los commits directos siguen entrando.

**2. Una rama por tarea**, creada desde `main` actualizado:

```powershell
git switch main; git pull
git switch -c feat/grid-detector      # feat/…, fix/…, docs/…, test/…, exp/…
# … trabajo y commits pequeños …
git push -u origin feat/grid-detector
```

Luego se abre el PR en GitHub.

**3. Reglas del PR**
- Lo aprueba **otro** integrante.
- El check `tests` debe estar en verde.
- El código nuevo trae sus tests sintéticos.
- Si se usó IA, se añade la entrada correspondiente en `IA_USAGE.md` dentro del mismo PR.
- Se integra con *Squash and merge*, con un mensaje del tipo `feat(det): grid anchor-free stride 8`.

**4. Nunca subir** datos, pesos `.pth` (van a GitHub Releases) ni notas `.md` personales. El `.gitignore` solo deja pasar `README.md`, `IA_USAGE.md` y `reports/**/*.md`.

**Reparto sugerido para la semana 9** (una rama y un PR cada uno):

| Rama | Contenido |
|---|---|
| `feat/slice-cache` | caché de cortes 2.5D y `Dataset` de PyTorch |
| `feat/backbone-cbam` | backbone con interfaz FundidoraPC + CBAM |
| `feat/grid-detector` | cabeza de detección, asignación de celdas y decodificación |
| `feat/nms-metrics` | NMS propio, IoU y mAP con casos de control, prueba de overfit |
