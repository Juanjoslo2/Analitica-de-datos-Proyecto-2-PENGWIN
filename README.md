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
python -m pytest          # 06_tests: 61 tests con phantoms sintéticos + 5 con datos (se omiten si no están)
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

## Sustentación: punto de entrada

`04_notebook/00_sustentacion_pengwin.ipynb` reúne todo lo hecho en el orden en que se presenta:
- estado frente al enunciado (hecho / parcial / pendiente);
- datos y su carga desde los `.mha`;
- arquitectura, grid y NMS;
- λ, overfit y curvas;
- resultados en test, ablación y predicciones;
- limitaciones y próximos pasos.

Corre en segundos sin datos ni GPU, porque lee lo versionado en `reports/`. Con `PENGWIN_CACHE_DIR` y los checkpoints presentes regenera las figuras en vivo. El EDA detallado sigue en `01_eda_pengwin.ipynb`.

## Modelo (semana 9)

Backbone FundidoraPC extendida (residual + CBAM en los bloques 3 y 4) → clasificación, detección por grid propio de stride 8 con NMS propio y segmentación semántica + borde. Diseño y evidencia en `reports/decisiones_de_diseno.md` §8.

Si el repositorio está en OneDrive, conviene dejar los datos (~32 GB) y el caché (~5 GB) **fuera** de la carpeta sincronizada; todos los scripts aceptan `--data-dir` / `--cache-dir`.

```powershell
# 1. Caché de cortes 256×256 (una vez, reanudable; ~10-30 s por caso; --shard k/n para paralelizar)
python scripts/build_slice_cache.py --data-dir D:/PENGWIN/01_data --cache-dir D:/PENGWIN/data_processed

# 2. Prueba de correctitud: overfit de 8 cortes (criterios en DD §5)
python scripts/overfit_batch.py --cache-dir D:/PENGWIN/data_processed

# 3. Calibración de λ -> reports/lambdas.json (la usa el entrenamiento con `lambdas: auto`)
python scripts/calibrate_lambdas.py --cache-dir D:/PENGWIN/data_processed

# 4. Entrenamiento y ablaciones
python scripts/train.py --cache-dir D:/PENGWIN/data_processed --name base
python scripts/train.py --cache-dir D:/PENGWIN/data_processed --name sin_cbam  --config configs/ablation_no_cbam.yaml
python scripts/train.py --cache-dir D:/PENGWIN/data_processed --name sin_tl    --config configs/ablation_scratch.yaml

# 5. Evaluación en test, curvas y galería de predicciones
python scripts/evaluate.py --cache-dir D:/PENGWIN/data_processed --ckpt checkpoints/base/best.pth checkpoints/sin_cbam/best.pth checkpoints/sin_tl/best.pth
python scripts/plot_training_curves.py
python scripts/predict_slices.py --ckpt checkpoints/base/best.pth --cache-dir D:/PENGWIN/data_processed
```

## Semana 10: fragmentos, distancia, SAM y latencia

Resumen y resultados en `04_notebook/02_semana10_fragmentos.ipynb`; decisiones y evidencia en `reports/decisiones_de_diseno.md` §10.

```powershell
# borde de fractura 3D en el caché (una vez; no relee los .mha)
python scripts/build_slice_cache.py --cache-dir D:/PENGWIN/data_processed --edges-only

# pipeline completo: modelo -> fragmentos 3D -> grilla nativa -> Dice por fragmento + distancia en mm
python scripts/evaluate_fragments.py --ckpt checkpoints/v2_last/best.pth --cache-dir D:/PENGWIN/data_processed --data-dir D:/PENGWIN/01_data --tag _edt

# qué aporta cada componente (γ, cambio en las features y knockout)
python scripts/component_contribution.py --ckpt checkpoints/v2/best.pth --cache-dir D:/PENGWIN/data_processed

# SAM zero-shot como línea base (checkpoint sam_vit_b_01ec64.pth aparte) y latencia CPU/GPU
python scripts/sam_baseline.py --ckpt checkpoints/v2/best.pth --cache-dir D:/PENGWIN/data_processed --sam-ckpt D:/PENGWIN/sam/sam_vit_b_01ec64.pth
python scripts/latency.py --ckpt checkpoints/v2/best.pth --cache-dir D:/PENGWIN/data_processed
```

Los entrenamientos largos se corren en ANTON (DGX Spark), unas 2 h por modelo; la laptop se usa para tests e inferencia. Los pesos de transfer learning (`checkpoints/fundidora_taller3.pth`, FundidoraPC del Taller 3) y los checkpoints entrenados se publican en GitHub Releases, no en git. Para los tests con datos: `$env:PENGWIN_DATA_DIR="D:/PENGWIN/01_data"; $env:PENGWIN_CACHE_DIR="D:/PENGWIN/data_processed"; python -m pytest -m "data or slow"`.

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
