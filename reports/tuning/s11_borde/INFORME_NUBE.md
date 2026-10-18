# Informe de la sesión en la nube · semana 11

- **Qué se probó:** contra el sobreajuste de la cabeza de borde, L2 ×100 (`c11_wd1e2`), un decodificador
  de 32 canales (`c12_segw32`) y uno U-Net++ (`c13_unetpp`), más el control `s11_ref`.
- **Etapa 1 (cribado, fold 0, 12 épocas):** umbral μ + 2σ = 0,6964 (μ 0,6792, σ 0,0086, cinco
  referencias). Dice por fragmento 0,6829 / 0,6899 / 0,6934 (z = 0,43 / 1,25 / 1,66). **Ninguno pasa.**
- **Etapas 2 y 3:** no corrieron. Sin modelo nuevo y sin otra medición de test.
- **Verificación:** recalculé con pandas μ, σ, umbral, IoU de referencia y la mejor combinación de cada
  corrida desde los CSV/JSON crudos: todo cuadra con `etapa1.json`. Logs e historiales: 12 épocas, sin
  NaN; `s11_ref` cae dentro de las referencias de la Fase 1 (z = +0,35). `c13_unetpp` tardó ~1 200 s por
  época con los otros carriles activos y ~365 s cuando quedó sola: reparto de la GPU, no un error.
- **Curvas:** los cuatro sobreajustan igual (BCE de validación con mínimo en la época 0-1); ningún cambio
  cierra la brecha.
- **Archivos:** `04_notebook/04_semana11_borde.ipynb` (ejecutado), `reports/figures/semana11/`,
  §15 en `reports/decisiones_de_diseno.md`, filas 22-24 en la sección 11 de `03_semana11.ipynb` (solo
  markdown, no se re-ejecutó), `reports/tuning/s11_borde/RESUMEN.md`, `IA_USAGE_entrada_s11_borde.md` y este informe.
  Model card y README sin tocar (no hubo modelo nuevo).
- **Pendiente para el usuario:** revisar y llevar estos cambios a la rama `Nicolas` del repo del equipo;
  completar y pegar la entrada de IA_USAGE (prompt exacto y decisiones manuales); añadir la semana 11 al
  informe en Word; apagar los contenedores `pengwin-s12-A..D` en ANTON si siguen arriba.
