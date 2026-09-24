# IA_USAGE — Bitácora de uso de IA generativa

Formato por entrada: modelo · fecha · prompt exacto · qué produjo la IA · qué se aceptó/modificó manualmente · análisis crítico (aciertos, errores, limitaciones).

---

## Entrada 001 — Consulta de arquitectura, reproducibilidad y validación de pipeline

- **Modelo:** Claude (Anthropic), configurado como `claude-sonnet-3-5`.
- **Fecha:** 2026-09-18
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

## Entrada 002 — 
