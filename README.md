# Analitica-de-datos-Proyecto-2-PENGWIN
**Integrantes**
> * Yáxul Santiago Cárdenas Hincapié
> * Nicolas Santiago Cuaran Sotelo
> * Juan Jose Orozco Lopez
> * David Felipe Martinez Viracacha

Sistema multitarea para detección, segmentación, clasificación y medición de separación de fragmentos en fracturas pélvicas a partir de CT (dataset PENGWIN). Proyecto Integrador Corte 2, Analítica de Datos, UAO 2026-2.

> **Uso exclusivamente académico.** No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Resultado final (test, 15 pacientes, medido una vez)

Modelo `v2_last` + TTA de 5 pasadas + posproceso `edt_hmax` (`configs/tuned.yaml`).

| Tarea | Métrica | Test | Objetivo | Cumple |
|---|---|---|---|---|
| Clasificación | F1 | 0,993 | ≥ 0,85 | sí |
| Clasificación | AUC | 0,999 | ≥ 0,85 | sí |
| Detección | IoU promedio | 0,919 | ≥ 0,65 | sí |
| Detección | mAP@0.50 | 0,980 | ≥ 0,65 | sí |
| Detección | mAP@[.50:.95] | 0,864 | ≥ 0,40 | sí |
| Segmentación de fragmento | Dice | 0,768 (IC95 0,702-0,827) | ≥ 0,85 | no |
| Segmentación de fragmento | IoU | 0,697 (IC95 0,636-0,753) | ≥ 0,70 | no concluyente |
| Opcional | HD95 / ASSD por fragmento | 11,7 / 3,3 mm | — | — |

Distancia de separación: error medio 1,96 mm (0,75 mm en fragmentos separados, 2,32 mm en fragmentos en contacto). Latencia por corte (RTX 3060 Laptop): 34 ms en GPU y 309 ms en CPU con TTA; 8 ms y 66 ms sin TTA. Intervalos por bootstrap de pacientes (15 en test); auditoría de fuga de datos y sobreajuste en `04_notebook/03_semana11.ipynb` §10 (`scripts/auditoria_fugas.py`). Limitaciones en `reports/model_cards/final.md`.

## Dónde está cada entregable

| Entregable (§7 del enunciado) | Dónde |
|---|---|
| Dashboard en vivo (3 visualizadores) | **https://pengwin.80-241-217-130.nip.io** · código en `05_app/` |
| Pesos `.pth` | GitHub Releases (`checkpoints/` no se versiona) |
| Pipeline reproducible | esta guía, `requirements.txt`, `configs/`, semilla 42 |
| Latencia CPU/GPU | `reports/latency/`, `scripts/latency.py` |
| Model cards | `reports/model_cards/` (final, v2, base, sin_cbam, sin_tl) |
| Ablación con/sin CBAM y con/sin TL | `04_notebook/02_semana10.ipynb` §16 (CV) y `reports/eval/` (test) |
| Matriz de confusión, 5 peores casos, distancia predicha vs real | `04_notebook/03_semana11.ipynb` |
| Decisiones de diseño y su evidencia | `reports/decisiones_de_diseno.md` |
| Uso de IA | `IA_USAGE.md` |

Notebooks, en orden: `01_eda_pengwin` (semana 8), `00_sustentacion_pengwin` (semana 9), `02_semana10` (pipeline completo y ajuste), `03_semana11` (entrega). Todos leen los resultados guardados en `reports/` y corren sin GPU.

## Entorno

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .          # hace importable el paquete `pengwin` (03_src/pengwin)
python -m pytest          # 06_tests: 136 tests con phantoms sintéticos; 4 con datos se omiten si no están
```

Los entrenamientos se corrieron en una DGX Spark (GPU GB10, aarch64) dentro del contenedor `nvcr.io/nvidia/pytorch:25.09-py3`; el mismo código corre en Windows con CUDA o en CPU.

## Reproducir desde cero

Los `.mha` de PENGWIN (Zenodo) van en `01_data/PENGWIN_CT_train_images_part1|part2` y `01_data/PENGWIN_CT_train_labels`. Si el repositorio está en OneDrive, conviene dejar los datos (~32 GB) y el caché (~5 GB) fuera; todos los scripts aceptan `--data-dir` / `--cache-dir`. Abajo, `D:/PENGWIN/...` es ese lugar.

```powershell
# 0. Partición fija por paciente (ya versionada en 01_data/splits.json, con su sha256)
python -m pengwin.data.create_splits

# 1. EDA (semana 8) y visualizador 1
python scripts/run_eda_extract.py --data-dir D:/PENGWIN/01_data
python 03_src/pengwin/visualization/mip_viewer.py 001

# 2. Caché de cortes 256×256 con borde de fractura 3D (una vez, reanudable)
python scripts/build_slice_cache.py --data-dir D:/PENGWIN/01_data --cache-dir D:/PENGWIN/data_processed

# 3. Prueba de overfit (semana 9) y calibración de λ por norma de gradiente
python scripts/overfit_batch.py --cache-dir D:/PENGWIN/data_processed
python scripts/calibrate_lambdas.py --cache-dir D:/PENGWIN/data_processed

# 4. Entrenamientos (40 épocas, ~2 h cada uno en la GB10)
python scripts/train.py --cache-dir D:/PENGWIN/data_processed --name base
python scripts/train.py --cache-dir D:/PENGWIN/data_processed --name sin_cbam --config configs/ablation_no_cbam.yaml
python scripts/train.py --cache-dir D:/PENGWIN/data_processed --name sin_tl   --config configs/ablation_scratch.yaml
python scripts/train.py --cache-dir D:/PENGWIN/data_processed --name v2       # modelo final: checkpoints/v2/last.pth -> checkpoints/v2_last/best.pth

# 5. Evaluación del modelo final en test (detección/clasificación y fragmentos con HD95/ASSD)
python scripts/evaluate.py --cache-dir D:/PENGWIN/data_processed --ckpt checkpoints/v2_last/best.pth
python scripts/evaluate_fragments.py --ckpt checkpoints/v2_last/best.pth --cache-dir D:/PENGWIN/data_processed `
    --data-dir D:/PENGWIN/01_data --split test --config configs/tuned.yaml --tag _tuned_superficie --surface --save-dir D:/PENGWIN/pred_tuned

# 6. Línea base SAM, latencia, matriz de confusión, peores casos, Grad-CAM, model cards
python scripts/sam_baseline.py --ckpt checkpoints/base/best.pth --cache-dir D:/PENGWIN/data_processed --sam-ckpt D:/PENGWIN/sam/sam_vit_b_01ec64.pth
python scripts/latency.py --ckpt checkpoints/v2_last/best.pth --cache-dir D:/PENGWIN/data_processed --tta
python scripts/confusion.py --ckpt checkpoints/v2_last/best.pth --cache-dir D:/PENGWIN/data_processed
python scripts/peores_casos.py --pred-dir D:/PENGWIN/pred_tuned --data-dir D:/PENGWIN/01_data --report reports/fragments/v2_last_test_tuned_superficie
python scripts/gradcam.py --cache-dir D:/PENGWIN/data_processed --cases 075 069 019
python scripts/model_cards.py
```

Validación cruzada del ajuste (semana 10): `scripts/cv_folds.py` arma los 5 folds por paciente, `scripts/train.py --folds-file reports/tuning/cv_folds.json --fold k` entrena cada uno, `scripts/cv_predict.py` guarda las predicciones fuera de fold y `scripts/tune_postprocess.py` barre el posproceso sobre ellas. `scripts/consolidar_ajuste.py` reúne los resultados en `reports/tuning/ajuste_*` (lo que lee el notebook).

## Dashboard

`05_app/` es una app FastAPI con una sola página: visualizador 1 (MIP del volumen crudo con umbral de 250 HU, sin modelo), visualizador 2 (corte a corte con slider: cajas, máscaras por fragmento y la distancia de separación escrita sobre cada secundario) y visualizador 3 (marching cubes + Plotly Mesh3d con la distancia junto a cada fragmento). Acepta un volumen `.mha`, `.nii`, `.nii.gz` o `.nrrd` (subido por trozos de 8 MB) o la imagen de un corte suelto, y trae tres casos de test ya procesados.

```powershell
# local
cd 05_app
uvicorn app:app --host 127.0.0.1 --port 8780
python preparar_ejemplos.py --data-dir D:/PENGWIN/01_data --cases 075 097 069    # casos de ejemplo (opcional)

# servidor con GPU (como está desplegado en ANTON)
docker build -f 05_app/Dockerfile -t pengwin-demo .
docker run -d --name pengwin-demo --gpus all -p 127.0.0.1:8780:8780 -v $PWD/checkpoints:/w/checkpoints:ro `
    -v $PWD/datos:/datos -e PENGWIN_USUARIOS=/datos/usuarios.db pengwin-demo
docker exec -it pengwin-demo python usuarios.py crear <usuario>     # primer usuario (la página exige ingreso)
```

Seguridad: el dashboard exige usuario y clave (SQLite con scrypt; `05_app/seguridad.py`). Tres claves equivocadas seguidas bloquean el perfil; se administra desde el servidor con `docker exec pengwin-demo python usuarios.py crear|clave|desbloquear|listar|eventos`. Las consultas son parametrizadas. Los archivos se validan por contenido antes de llegar al modelo (`05_app/validacion.py`): solo volúmenes de un único archivo (un `.mha` o `.nrrd` cuya cabecera apunte a otro archivo del servidor se rechaza), dimensiones acotadas antes de descomprimir y cortes PNG/JPEG reales. nginx limita los intentos de ingreso por IP.

Despliegue: el contenedor corre en ANTON y un túnel SSH inverso lo publica en el VPS del equipo (nginx + HTTPS de Let's Encrypt). El enunciado propone un túnel de Cloudflare; la red de ANTON bloquea su puerto de salida (7844), por eso se usó el VPS. Se abre igual desde un computador o un celular.

## Flujo de trabajo en Git

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
git push -u origin feat/grid-detector
```

**3. Reglas del PR:** lo aprueba **otro** integrante; el check `tests` debe estar en verde; el código nuevo trae sus tests sintéticos; si se usó IA, se añade la entrada en `IA_USAGE.md` dentro del mismo PR; se integra con *Squash and merge*.

**4. Nunca subir** datos, pesos `.pth` (van a GitHub Releases) ni notas `.md` personales. El `.gitignore` solo deja pasar `README.md`, `IA_USAGE.md` y `reports/**/*.md`.
