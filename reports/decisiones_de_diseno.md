# Decisiones de diseño — punto de partida de la semana 9

Este documento es la referencia común del equipo para implementar el modelo. Cada decisión cita su evidencia en el EDA (`04_notebook/01_eda_pengwin.ipynb`, §n). Los parámetros concretos viven en `configs/base.yaml`; si una decisión cambia, se cambia aquí y en el YAML dentro del mismo PR.

## 1. Datos y entrada al modelo

| Decisión | Valor | Evidencia |
|---|---|---|
| Orientación | todo a LPS con `sitk.DICOMOrient` | 34/100 casos vienen en RAS (§1) |
| Recorte | recorte óseo por volumen (`compute_bone_crop`), el mismo para todos sus cortes | 1,27 vs 1,56 mm/px; nunca corta hueso (§2) |
| Tamaño | 256 × 256; spacing efectivo = lado del recorte (mm) / 256, guardado por caso | §2 |
| Contexto | 2.5D: canales (z−Δ, z, z+Δ), Δ = round(2 mm / dz) | Δ = 2 (78 casos) / 3 (22) (§2) |
| Intensidad | recorte a [−1024, 3071] HU; ventana L400 / W1800 → [0, 1] | 99,6 % del hueso sin saturar (§3) |
| Muestreo | cortes con hueso + ~12 % de cortes vacíos | coxales en 96 % de los cortes con hueso (§8) |
| Partición | `01_data/splits.json` v2 (por paciente, SMD máx 0,155, `sha256`) | §9 |
| Aumentación | **sin flip horizontal** (o con intercambio de etiquetas izq ↔ der) | la lateralidad depende de la posición |

## 2. Detección (cabeza 2)

- **Una caja por región anatómica y corte, no por fragmento.** El enunciado pide "bounding boxes de cada región anatómica".
- **La caja GT es la envolvente de todas las islas de la región en ese corte.** Una región aparece partida en islas en 28–42 % de los cortes (§8). Como el GT y la predicción usan la misma definición, el espacio vacío entre islas no penaliza el IoU. Las cajas no se fusionan entre cortes: la reconstrucción 3D se hace con las máscaras (§3 de este documento).
- Grid propio anchor-free de **stride 8** (32 × 32 celdas): el 39 % de las cajas del sacro mide < 16 px de lado (§8).
- Las cajas GT con lado < 4 px se marcan como "ignorar": no cuentan como positivos ni como negativos.
- NMS propio y vectorizado por clase; se conserva como máximo 1 caja por clase y corte.

## 3. Segmentación de instancias y contacto (cabeza 3)

- Salidas: semántica de 4 clases (fondo, SA, izq., der.) + **mapa de borde de fractura**.
  - El 88,7 % de los fragmentos secundarios toca al principal (§7).
  - El borde es 1 de cada ~446 píxeles de hueso: el objetivo se dilata 2 px y se entrena con BCE (`pos_weight` ≈ 21) + Dice.
- Instancias = (región − borde) → **componentes conexas 3D tras apilar los cortes** → watershed para devolver los píxeles de borde.
  - Las componentes < 0,1 cm³ se reasignan a la instancia vecina.
  - Un mismo fragmento aparece partido en islas 2D en 18–40 % de los cortes, así que las componentes 2D no equivalen a instancias (§8).
- El fragmento principal de la predicción es la componente 3D de mayor volumen en la región. En el GT coincide con la etiqueta 1/11/21 en 300/300 regiones (§1).
- **Modo de falla principal:** si dos fragmentos que se tocan se fusionan, el secundario **desaparece**: no hay distancia que medir (no es que se mida 0 mm). Por eso se reportan:
  - **fragmentos recuperados** (recall de instancias con IoU ≥ 0,5 frente al GT);
  - Dice por fragmento **estratificado por volumen** (< 5, 5–20, > 20 cm³): el 14,5 % de los secundarios mide < 5 cm³ (§6);
  - error de distancia (pred − GT) **por separado** para fragmentos en contacto y separados.

## 4. Función de pérdida

- **Tres términos, como pide el enunciado:** L = λ_cls·L_cls + λ_det·L_det + λ_seg·L_seg.
  - **La distancia de separación no es un término de pérdida**: es una medición posterior sobre las máscaras.
  - L_seg = CE (pesos ∝ 1/√frecuencia: 1 / 10,8 / 8,7 / 8,7) + Dice + borde.
- **Calibración de λ:**
  1. Entrenar ~200 iteraciones con λ = 1 y registrar G_k = ‖∇_θs L_k‖₂ sobre la última capa compartida (la salida del CBAM).
  2. Fijar λ_k = mean(G) / G_k, normalizado para que Σλ = 3.
  3. Reportar la tabla de G_k y λ_k, y una prueba de sensibilidad multiplicando cada λ por 0,5 y por 2.

## 5. Prueba de overfit (entregable de la semana 9)

- **8 cortes fijos de train** (que incluyan sacro fracturado y al menos un fragmento separado), sin aumentación, semilla fija, hasta 500 pasos.
- **Criterios de aceptación:**
  - la pérdida cae ≥ 95 %;
  - Dice ≥ 0,95;
  - mAP@0.5 ≥ 0,95 sobre esos mismos 8 cortes.
- **Si falla**, revisar en este orden:
  1. forma de los tensores;
  2. asignación de celdas del grid;
  3. decodificación de cajas;
  4. IoU;
  5. NMS;
  6. alineación máscara ↔ imagen tras el recorte;
  7. pérdidas;
  8. que el gradiente llegue a las tres cabezas.
- Ubicación: `06_tests/test_overfit_batch.py`, con la marca `slow` (no corre en CI).

## 6. Política de pruebas

- Todo módulo nuevo entra con tests sobre **phantoms sintéticos** con respuesta conocida (IoU, NMS, decodificación del grid, EDT con spacing anisotrópico, Dice). No deben depender del dataset (~32 GB).
- Los tests que necesitan los `.mha` llevan la marca `data` y se omiten solos si no están.
- GitHub Actions (`.github/workflows/tests.yml`) corre `pytest` en cada PR. Hoy: 16 tests sin datos + 4 con datos.

## 7. Revisión externa de la semana 8 — qué se tomó y qué no

| Observación | Resolución |
|---|---|
| Sin PRs y commits directos a `main` | Flujo de ramas + PR con 1 aprobación + CI obligatorio (ver README) |
| Islas 2D: ¿una caja o varias? | Una por región, envolvente de las islas (§2); instancias en 3D (§3) |
| Fragmentos en contacto se fusionan | Salida de borde + métrica de fragmentos recuperados + análisis por grupo (§3) |
| Tests dependientes del dataset | Ya había 8 tests sin datos; ahora son 16, y el CI corre sin dataset (§6) |
| Prueba de overfit | Protocolo y criterios en §5 |
| Calibrar λ | Método en §4. **No** se añade un término de pérdida para la distancia |
| Cifras citadas (86 %, 33 %/15 %, 8 GB) | Eran preliminares o erróneas: usar 88,7 %, 28–42 % / 18–40 % y ~32 GB |
