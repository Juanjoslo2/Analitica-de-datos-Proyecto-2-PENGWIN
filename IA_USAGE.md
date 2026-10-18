# IA_USAGE — Bitácora de uso de IA generativa

Formato por entrada: modelo · fecha · prompt exacto · qué produjo la IA · qué se aceptó/modificó manualmente · análisis crítico (aciertos, errores, limitaciones).

---

## Entrada 001 — Consulta de arquitectura, reproducibilidad y validación de pipeline

- **Modelo:** Claude (Anthropic), configurado como `claude-sonnet-3-5`.
- **Contexto adjunto:** Resumen de restricciones del documento `Proyecto_Corte2_PENGWIN.pdf` (prohibición de YOLO/Detectron2, uso de grid y NMS propios, backbone compartido con CBAM), estructura de carpetas preliminar planteada por el equipo (`00_docs` a `06_tests`) y hallazgos preliminares de la inspección de archivos `.mha`.

**Prompt exacto:**
> Estamos iniciando el Proyecto Integrador de detección, segmentación y medición de fracturas pélvicas en CT (dataset PENGWIN). Como equipo ya configuramos la estructura base del repositorio (`00_docs`, `01_data`, `02_auditoria`, `03_src`, `04_app`, `05_notebook`, `06_tests`).
> Necesitamos consultarte como apoyo técnico los siguientes puntos para contrastar con nuestro criterio:
> 1. Para asegurar reproducibilidad en Windows entre 4 integrantes con soporte PyTorch + CUDA, ¿qué ventajas concretas tiene usar `requirements.txt` frente a un `environment.yml` en este tipo de proyecto?
> 2. Revisa la viabilidad técnica de nuestro pipeline multitarea (Backbone con CBAM hacia 3 cabezas: clasificación multiclase, detección con grid propio y segmentación de fragmentos). ¿Cómo recomiendas plantear matemáticamente el balanceo inicial de los pesos λ de las pérdidas multitarea para evitar colapso de gradientes?
> 3. En la literatura ortopédica se menciona la morfología de fragmentos conminutos. ¿Tiene sentido metodológico medir elongación/esfericidad de fragmentos dentro del alcance del reto, o dispersa los objetivos frente a la distancia euclidiana mínima (`distance_transform_edt`) exigida?
> 4. Qué consideraciones debemos tener en cuenta sobre la modularidad del código fuente en `03_src` para facilitar tanto el entrenamiento como el despliegue posterior en un dashboard interactivo.

**Qué produjo la IA:**
1. Comparativa técnica entre `requirements.txt` y `environment.yml`, recomendando `requirements.txt` fijando versiones exactas de PyTorch con wheels de CUDA específicos para simplificar la sincronización en Windows.
2. Esbozo teórico para la función de pérdida multitarea: sugerencia de normalizar los términos de pérdida mediante factores de escala fijos iniciales (λcls,λdet,λseg) basados en la magnitud de salida de cada cabeza, recomendando registrar gradientes en las primeras épocas para calibrarlos.
3. Análisis crítico sobre la medición de elongación: desaconsejó enfocar esfuerzos de modelado en ella, argumentando que no forma parte del ground truth ni de la evaluación de PENGWIN, recomendando limitarla a un descriptor complementario en el EDA si el equipo lo deseaba.
4. Sugerencia de modularización dentro de `03_src` separando explícitamente: carga de volúmenes médicos, módulos de atención (CBAM), lógica de inferencia e implementación matemática de NMS y transformadas de distancia.

**Qué debe verificar/modificar el equipo (completar):**
* **Aceptado**: Se tomó la recomendación de usar `requirements.txt` con versiones fijadas para facilitar la compatibilidad cruzada del equipo. Se adoptó el criterio de descartar la medición de elongación en el pipeline del modelo para priorizar estrictamente el cálculo de separación euclidiana en milímetros exigido.
* **Modificado/Descartado**:
   - Las versiones de librerías sugeridas por la IA se probaron en local y se ajustaron manualmente; se corrigió la versión de CUDA y PyTorch para hacerla compatible con las GPUs reales del equipo.
   -  La propuesta de arquitectura de la IA incluía sugerencias genéricas de detección tipo anchor boxes clásicos; el equipo descartó ese enfoque para implementar un grid detector propio adaptado a las 3 regiones anatómicas específicas del challenge.
   - Toda la inspección real de datos (detección de orientaciones LPS/RAS en los `.mha`, verificación de espaciado físico y fragmentos en contacto) fue codificada y ejecutada localmente por el equipo en notebooks de Python, sin delegación en la IA.

**Análisis crítico (completar por el equipo):**
* **Aciertos**: La IA fue útil para contrastar decisiones de diseño de software y confirmar que implementar métricas complejas no solicitadas (como elongación) ponía en riesgo el tiempo de entrega de los hitos obligatorios. La formulación matemática sugerida para el balanceo de λ sirvió como buen punto de partida teórico para la discusión del equipo.
* **Errores / Supuestos no verificados**: La IA asumió inicialmente formatos NIfTI tradicionales (`.nii.gz`) basados en el estándar del PDF, desconociendo que la distribución real de PENGWIN en Zenodo utiliza MetaImage (`.mha`). El equipo tuvo que ajustar manualmente el pipeline hacia `SimpleITK` para resolver la inversión de ejes `(X,Y,Z)` a `(Z,Y,X)`.
* **Limitaciones**: El modelo no tiene visibilidad del entorno de hardware real ni de la distribución real de los vóxeles; cualquier recomendación de hiperparámetros o pesos de pérdida debe ser necesariamente validada y calibrada experimentalmente en código propio.


---

## Entrada 002 — Auditoría de código de la Semana 8, optimización de preprocesamiento y diseño de EDA


- **Modelo:** Claude (Anthropic), configurado como `claude-sonnet-3-5`.
- **Contexto suministrado en el prompt:** Módulos desarrollados por el equipo para la Semana 8 (`data_loader.py`, `create_splits.py`, `mip_viewer.py`), planteamiento del test de orientación anatómica y estructura preliminar para el análisis exploratorio orientado a decisiones de diseño de la red.

**Prompt exacto:**
> "Como equipo completamos los componentes clave de la Semana 8: cargador de datos con orientación forzada a LPS, partición de splits por paciente y el Visualizador 1 (MIP). Comparto los fragmentos de código para realizar una revisión por pares (code review) enfocada en robustez, eficiencia y prevención de fallos silenciosos.
> 
> Necesitamos retroalimentación técnica puntual sobre:
> 1. Validar la lógica matemática del test de lateralidad en `data_loader.py` bajo coordenadas LPS y el tipo de datos óptimo para las máscaras.
> 2. Mejorar el Visualizador MIP en `mip_viewer.py` para eliminar la camilla del tomógrafo, evitar que artefactos de alta densidad (metal) saturen la escala ósea y asegurar que las dimensiones se reporten en mm y no en píxeles.
> 3. En `create_splits.py`, evaluar si una estratificación simple por número de fragmentos introduce sesgos en la distribución de casos complejos (conminutos) y sugerir un criterio estadístico más riguroso (ej. minimizar la diferencia de medias estandarizada, SMD).
> 4. Estructurar un notebook de EDA con hipótesis claras que justifiquen técnicamente las decisiones de la arquitectura: dimensiones del recorte corporal, desbalance de clases y pesos para la función de pérdida."

### Hallazgos de la auditoría y refactorizaciones implementadas

| Archivo | Problema detectado por el equipo / revisión | Corrección técnica aplicada |
|---|---|---|
| `data_loader.py` | `verify_left_right_consistency` solo evaluaba `x_izq != x_der`, lo cual permitía falsos positivos en casos RAS no reorientados. | Se ajustó la aserción estricta a `x_izq > x_der` según el estándar de coordenadas LPS. Se formuló un test unitario que falla deliberadamente si no se reorienta. |
| `data_loader.py` | Máscaras cargadas en `int16` consumían memoria innecesaria; no se alertaban posibles archivos huérfanos. | Se redujo el tipo de dato a `uint8` (las etiquetas van de 0 a 30) y se añadieron advertencias explícitas para prevenir inconsistencias de nombres. |
| `mip_viewer.py` | Presencia de la camilla del escáner; artefactos metálicos (>30 000 HU) distorsionaban el mapa de color; la escala estaba en píxeles; llamada rígida a `plt.show()`. | Se implementó segmentación de silueta corporal, ventana fija de visualización ósea [250, 1800] HU, escalado de ejes en mm usando el espaciado físico y retorno del objeto `matplotlib.figure.Figure`. |
| `__init__.py` | Lista `__all__` desactualizada e importación prematura de `matplotlib` al cargar el paquete. | Se desacoplaron las dependencias gráficas del núcleo de datos. |
| `data/test.py` | Script de prueba manual y no estandarizado dentro de la carpeta fuente. | Se migró formalmente a la suite de pruebas automatizadas en `06_tests/test_data_integrity.py` bajo `pytest`. |
| `create_splits.py` | Estratificación inicial ingenua (≤6 vs. >6 fragmentos) generaba sesgo: el split de test quedó con 67 % de sacros fracturados frente a 43 % en train (SMD máx de 0.44). | Se implementó un algoritmo de búsqueda de partición que minimiza la Diferencia Media Estandarizada máxima ($\text{SMD} \le 0.155$) y se añadió verificación por `sha256` en `splits.json`. |

### Sugerencias conceptuales de la IA evaluadas por el equipo:
- **Concepto estadístico de balanceo:** La IA sugirió evaluar el balance de los splits usando la Diferencia Media Estandarizada (SMD). El equipo adoptó la sugerencia matemática y programó su propio script de optimización.
- **Sintaxis de pruebas unitarias:** Se consultó la estructura recomendada para parametrizar casos de prueba en `pytest` (`@pytest.mark.parametrize`), la cual el equipo utilizó para estructurar sus tests en `06_tests/test_data_integrity.py`.
- **Fórmula de ponderación de clases:** La IA propuso un esquema teórico de frecuencias para mitigar el desbalance entre fondo y fragmentos óseos.

### Errores y limitaciones de las propuestas de la IA (identificados y corregidos por el equipo):
1. **Fallo en propuesta de recorte anatómico (Riesgo clínico grave):** Para aislar el cuerpo, la IA sugirió un snippet basado en conservar únicamente la mayor componente conexa en 2D (`label_with_largest_area`). Al auditar visualmente los datos, **el equipo descubrió que esta lógica amputaba hueso en 9 de los 100 casos**, debido a que las fracturas graves de sínfisis o sacro separan la pelvis en dos masas desconectadas. El equipo descartó el algoritmo de la IA y programó una solución propia que fusiona componentes conexas relativas ($\ge 25\%$ del área mayor) y agrupa fragmentos por proximidad euclidiana ($\le 30\text{ mm}$).
2. **Ponderación de pérdida numéricamente inestable:** La sugerencia de la IA de usar frecuencia inversa pura ($1/f$) asignaba un peso desproporcionadamente bajo al fondo ($0.01$) y valores extremos a fragmentos milimétricos, provocando inestabilidad en el gradiente. El equipo corrigió la formulación aplicando un amortiguamiento por raíz cuadrada: $w_c = \frac{1}{\sqrt{f_c}}$.
3. **Mala práctica de encapsulamiento gráfico:** La IA sugirió configurar `matplotlib.rcParams` de forma global, lo que activaba cuadrículas indeseadas sobre las tomografías. El equipo rechazó esa implementación y encapsuló los estilos en manejadores de contexto locales (`plt.rc_context`).

### Verificaciones experimentales ejecutadas por el equipo:
- [x] Ejecución y aprobación de la suite completa de pruebas locales (`pytest 06_tests/`).
- [x] Inspección visual de la ventana ósea en casos complejos con prótesis/osteosíntesis metálica (casos 027 y 100).
- [x] Congelación y verificación criptográfica (`sha256`) del archivo `01_data/splits.json` definitivo.

### Análisis crítico:
- **Aciertos de la consulta:** La sugerencia de emplear SMD permitió formular una métrica matemática clara para validar que los splits no tuvieran sesgos de severidad.
- **Errores / Supuestos no verificados de la IA:** La IA asumió erróneamente que una pelvis siempre constituye una sola estructura conexa continua en el plano 2D, ignorando la realidad de los pacientes politraumáticos del dataset. Aceptar su sugerencia de código habría corrompido el ground truth del 9% de los casos.
- **Limitaciones del modelo:** Las respuestas de la IA carecen de validación empírica sobre datos médicos reales. Toda decisión geométrica, de recorte o de estabilidad numérica tuvo que ser verificada, depurada y programada directamente por los integrantes del equipo.

---

## Entrada 003 — Auditoría de arquitectura, CI y preparación para Semana 9

- **Modelo:** Claude 3.5 Sonnet (Anthropic)
- **Fecha:** 2026-09-26

**Prompt exacto:**
> Actúa como auditor técnico. Revisa la transición a Semana 9 del proyecto PENGWIN (detección por macro-región anatómica, NMS propio y distancia borde a borde como post-proceso con scipy). Contrasta las observaciones del equipo sobre gobernanza en Git, diseño de CI en GitHub Actions sin GPU y generación de phantoms sintéticos para validar la convención de espaciado físico en la EDT. Señala inconsistencias frente a la rúbrica y propón únicamente la plantilla base de pruebas desacopladas.

**Sugerencias del modelo y filtrado técnico:**
- **Rechazado:** La IA propuso incluir un término `loss_distance` en el entrenamiento y delimitar bounding boxes por fragmento. Se descartó por contradecir la rúbrica: la distancia es un cálculo determinista de post-procesamiento y la detección es por macro-región anatómica (Sacro, Coxal Izq, Coxal Der).
- **Aceptado y corregido:** Plantilla base para GitHub Actions (`.github/workflows/tests.yml`) y phantoms sintéticos para pruebas unitarias.

**Desarrollo manual del equipo:**
- **Corrección de la EDT:** Ajuste manual del orden de ejes en `scipy.ndimage.distance_transform_edt` para usar `sampling=(dz, dy, dx)` conforme a la lectura de SimpleITK, evitando errores en el cálculo milimétrico.
- **Suite de pruebas:** Implementación de 20 tests unitarios en `06_tests/` (16 ejecutables sin necesidad del dataset real).
- **Gobernanza Git:** Activación de reglas de protección en la rama `main` (bloqueo de commits directos y requisito obligatorio de PR con revisión por pares).
- **Diseño técnico:** Redacción de `reports/decisiones_de_diseno.md` justificando la pérdida de 3 términos y el protocolo de overfit en batch pequeño.

**Análisis crítico:**
- **Aciertos:** Aceleró la estructura inicial del pipeline de CI y la sintaxis de pruebas desacopladas de datos pesados.
- **Errores de la IA:** Confusión en la orientación espacial de volúmenes médicos $(Z, Y, X)$ frente a matrices estándar $(X, Y, Z)$, e insistencia en optimizar por gradiente una métrica clínica no derivable (distancia física).
- **Limitaciones:** El modelo carece de comprensión sobre tensores en memoria GPU y dependencias clínicas reales; requiere validación y refactorización humana estricta.
