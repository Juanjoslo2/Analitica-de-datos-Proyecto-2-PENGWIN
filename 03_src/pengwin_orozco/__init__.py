"""pengwin_orozco.

Intento propio de la semana 9 (rama ``orozco``): detección por grid anchor-free
con DFL (Distribution Focal Loss), implementado desde cero sobre la base común.

Punto de vista distinto al pipeline compartido:
  * backbone FundidoraPC pura del curso (sin residuales), con CBAM propio;
  * sin caché de cortes en disco: el dataset carga los .mha directamente;
  * sin cabezas de clasificación/segmentación: la semana 9 es solo detección
    (overfit sobre un batch pequeño y cajas razonables) [enunciado, semana 9];
  * cada celda predice una distribución de ``reg_max`` bins por borde (DFL):
    precisión subpixel en la regresión l, t, r, b (principio de YOLOv8 visto en
    clase, implementación 100 % propia);
  * pérdida de detección = focal + DFL + (1 − GIoU), NMS propio por clase.
"""