# Registro de uso de inteligencia artificial generativa

## Información del proyecto

- **Institución:** Universidad Autónoma de Occidente
- **Proyecto:** Sistema de detección, segmentación y medición de fragmentos en fracturas pélvicas a partir de tomografías con el dataset PENGWIN
- **Profesor:** Carlos Ferro
- **Integrantes:** Nicolas Cuaran, Juan Jose Orozco, Yáxul Santiago Cárdenas, David Felipe Martinez
- **Herramientas de inteligencia artificial utilizadas:** Claude Opus 5.5 (investigación y formas de aplicación), ChatGPT (explicaciones y contexto), Gemini (explicaciones y contexto)

---

## Resumen de decisiones implementadas

### 1. Separación de fragmentos

- **Qué aportó la IA:** Idea general del método watershed con puntos de partida.
- **Qué adaptamos:** Se ajustaron los parámetros con los datos de validación y se agregó un filtro de tamaño mínimo.

### 2. Preparación de imágenes

- **Qué aportó la IA:** Ventana de intensidades para resaltar el hueso.
- **Qué cambiamos:** Se probaron varias ventanas y se escogió la más adecuada con validación; también se cambió la forma de redimensionar las máscaras.

### 3. Modelo

- **Qué aportó la IA:** La propuesta de dónde agregar CBAM al modelo y de qué manera, para las pruebas con y sin CBAM.
- **Qué cambiamos:** Realizamos una prueba con y sin CBAM y vimos que la diferencia en los resultados era pequeña, es decir, que su aporte fue más un requerimiento del proyecto que algo que impulsara o ayudara al modelo.

### 4. Comparación

- **Qué aportó la IA:** Una forma justa de comparar los resultados con SAM.
- **Qué cambiamos:** Utilizamos la forma de SAM que nos dio únicamente para realizar comparaciones, sin emplearlo para crear las etiquetas del conjunto de datos.

### 5. Dashboard

- **Qué aportó la IA:** Recomendaciones para mejorar la latencia, la visualización y la carga del archivo `.mha` externo debido a la ubicación de la página; recomendaciones para la visualización de las imágenes y la decoración.
- **Qué cambiamos:** Agregamos un aviso de que este es un desarrollo totalmente académico, se organizó mejor la página y se aplicó la mejora de la carga del archivo `.mha`, debido a que el servidor se encuentra en Europa.

---

## Prompts

### Investigación: preguntas y artículos

#### Prompt 1

> dame papers que hayan tratado con el dataset de pengwin, en el que hablen sobre como trataron los archivos, que valores HU se tomaron para tomar como ejemplo esa se paracion, (ejemplo si fue de -200 a 1200, de 400 a 3000, etc)

**Respuesta:**

- [Benchmark of Segmentation Techniques for Pelvic Fracture in CT and X-ray: Summary of the PENGWIN 2024 Challenge (arXiv)](https://arxiv.org/html/2504.02382v2) · [versión IEEE](https://ieeexplore.ieee.org/document/11322550/)
- [Automatic pelvic fracture segmentation: a deep learning approach and benchmark dataset (Frontiers in Medicine, 2025)](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1511487/full)
- [FracSegNet – repositorio con código y modelos](https://github.com/Ren-ling/FracSegNet) · [paper MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_30)
- [Transformer-based and CNN-based models for clinically effective 2D and 3D pelvic bone segmentation in CT imaging (BMC, 2026)](https://link.springer.com/article/10.1186/s12891-025-09444-8)
- [Multi-step segmentation of pelvic fractures… (Jurgas et al., ficha AGH)](https://badap.agh.edu.pl/publikacja/163136)
- [Dataset PENGWIN en Zenodo](https://zenodo.org/records/10927452)

#### Prompt 2

> Investiga los equipos ganadores  o con métodos que dieron mejores resultados publicados en PENGWIN, dame los enlaces a los papers que estos equipos para poder observar como trataron las imagenes, como trabajaron el spacing, la ventana hu,orientacion etc.

**Respuesta:**

- [Benchmark of Segmentation Techniques for Pelvic Fracture in CT and X-ray: Summary of the PENGWIN 2024 Challenge (arXiv)](https://arxiv.org/html/2504.02382v2) · [versión IEEE](https://ieeexplore.ieee.org/document/11322550/)
- [SMILE – PENGWIN_Challenge](https://github.com/yuepeiyan/PENGWIN_Challenge)
- [MedIG – MICCAI-Challenge-PENGWIN2024](https://github.com/pzhhhhh2263/MICCAI-Challenge-PENGWIN2024) · [NexToU (arXiv 2305.15911)](https://arxiv.org/abs/2305.15911)
- [MedApp-AGH – multi-step-pelvic-fractures](https://github.com/Jarartur/multi-step-pelvic-fractures) · [ficha del paper en AGH](https://badap.agh.edu.pl/publikacja/163136)
- [FracSegNet (base de MedApp-AGH)](https://github.com/Ren-ling/FracSegNet) · [paper MICCAI 2023](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_30)

#### Prompt 3

> busca qué métodos utilizaron los equipos destacados de PENGWIN para separar los fragmentos individuales dentro de una misma región anatómica. Quiero saber si utilizaron watershed, transformadas de distancia, componentes conexas, modelos 3D, postprocesamiento o cualquier otra técnica.

**Respuesta:**

- [Summary of the PENGWIN 2024 Challenge (arXiv)](https://arxiv.org/html/2504.02382v2) · [IEEE](https://ieeexplore.ieee.org/document/11322550/)
- [Código SMILE](https://github.com/yuepeiyan/PENGWIN_Challenge) (posprocesamiento en `Inference/CT/utils/utils.py`)
- [Código MedIG](https://github.com/pzhhhhh2263/MICCAI-Challenge-PENGWIN2024) (posprocesamiento en `task1/inference.py`) · [NexToU](https://arxiv.org/abs/2305.15911)
- [Código MedApp-AGH](https://github.com/Jarartur/multi-step-pelvic-fractures) · [paper Jurgas et al., MICCAI 2025](https://doi.org/10.1007/978-3-032-05825-6_10)
- [Código MIC-DKFZ](https://codebase.helmholtz.cloud/hi-dkfz/applied-computer-vision-lab/challenges/abbc)
- [FracSegNet](https://github.com/Ren-ling/FracSegNet)

Conclusiones del paper resumen:

- La representación borde-núcleo (MIC-DKFZ) fue la que mejor separó fragmentos que se tocan.
- La representación primario-secundario falla cuando el principal es ambiguo y puede fusionar secundarios vecinos.
- Borde-núcleo también tiene un punto débil: es sensible a errores pequeños en el borde, sobre todo en fracturas finas.

#### Prompt 4

> Dame una explicacion completa del algoritmo watershed como funciona, sus partes etc, y como podria aplicarse en este caso.
> tambien explicame que es un nucleo de mascara, como se obtiene el nucleo al quitar o excluir el borde, como podemos calcular la transformada de distancia y como podemos crear las semillas.

**Respuesta (fragmento):**

Watershed es un algoritmo de segmentación que trata la imagen como un relieve topográfico, donde el valor de cada vóxel es su altura. Desde cada semilla o marcador se inunda el relieve: el agua avanza primero por los valles y, cuando dos cuencas están por encontrarse, se levanta un dique que queda como frontera entre regiones. Sus partes son el relieve (la imagen que se inunda), los marcadores (las semillas, una por objeto), la máscara (la zona donde puede avanzar el agua) y la conectividad entre vóxeles (6, 18 o 26 vecinos en 3D). Se implementa con una cola de prioridad: siempre se toma el vóxel no etiquetado más bajo y recibe la etiqueta del vecino que lo alcanzó. Sin marcadores, cada mínimo local genera una región y el resultado queda sobresegmentado. Por eso se usa la variante con marcadores, en la que las semillas fijan cuántos objetos hay y watershed solo decide dónde va la frontera.

#### Prompt 5

> Explícame paso a paso cómo se calculan las semillas para watershed, quisiera que me explicaras como se usa la transformada de distancia en esre caso, que significa que un punto este lejos del borde, como se escogen los maximos y como podemos decidir si una semilla pertence a un fragmento distinto

**Respuesta (fragmento):**

Las semillas se calculan por separado en cada hueso. Primero se obtiene el núcleo: la región del hueso sin los vóxeles que la red marca como borde de fractura (probabilidad de borde menor que 0,1). Después se calcula la transformada de distancia euclídea del núcleo con `scipy.ndimage.distance_transform_edt`, usando el spacing del header en el parámetro `sampling` para que los valores queden en milímetros. Para cada vóxel, la transformada da la distancia al punto más cercano fuera del núcleo, ya sea la superficie del hueso o una grieta predicha. Ese valor es el radio de la esfera más grande que cabe ahí sin salir del núcleo, es decir, el medio grosor local del hueso. Que un punto esté lejos del borde significa que está en el centro de una parte gruesa. Un puente de grosor w nunca supera w/2, así que los cuellos entre fragmentos siempre tienen valores bajos.

#### Prompt 6

> quisiera saber como podemos obtener el nucleo, y porque este nos puede ayudar a separar fragmentos, que ocurre cuando la region rellena la frieta, y como watersehd devuelve el borde a cada fragmento

**Respuesta (fragmento):**

**1. Cómo se obtiene el núcleo**

El núcleo es la región del hueso sin la franja donde la red cree que hay fractura:

```python
nucleo = region & (p_borde < 0.1)
```

- `region`: máscara de un hueso (SA, LI o RI) que predice la cabeza de segmentación.
- `p_borde`: probabilidad de borde de fractura que predice la segunda salida. Se entrenó con la superficie entre fragmentos calculada en 3D y dilatada 2 px.

El umbral de 0,1 es conservador: quita un vóxel aunque la red tenga solo un 10 % de sospecha de que hay fractura. Conviene quitar de más, porque watershed después devuelve esos vóxeles.

Hay dos alternativas que no dependen de la red: la erosión morfológica de la región, o un umbral sobre su transformada de distancia (`EDT(region) > d`). Las dos encogen el hueso por todos lados, pero no saben dónde está la fractura.

**2. Por qué ayuda a separar fragmentos**

Dos fragmentos que se tocan forman una sola componente conexa, así que `label()` sobre la región los ve como un solo hueso. El borde de fractura es justamente la superficie donde se tocan. Si lo quitas, rompes el contacto y cada fragmento queda como una isla separada en el núcleo. Es como bajar la marea: el agua cubre el istmo y quedan dos islas.

Cada isla, o el centro profundo de cada isla, se vuelve la identidad de un fragmento. El núcleo no es el resultado final, porque es más chico que el hueso: es la forma de saber cuántos fragmentos hay y dónde está cada uno.

#### Prompt 7

> porque watershed puede fallar cuando dos fragmentos se tocan en la superficie de fractura?

**Respuesta (fragmento):**

Watershed no descubre fragmentos: solo traza la frontera entre las semillas que le das. Si dos fragmentos pegados salen con una sola semilla, watershed devuelve una sola región. Y si salen con dos, el dique queda donde se encuentran las dos aguas, no necesariamente sobre la fractura. Corrí tres casos pequeños en 2D que muestran los tres modos de falla.

**1. Contacto plano: no hay cuello y sale una sola semilla**

Las semillas dependen de que la EDT forme un cuello (una silla baja) entre los fragmentos. Hay casos en que dos fragmentos se tocan por una superficie ancha y plana, como una fractura sin desplazamiento o una impactada. Si además la red no detecta el borde, la forma de los dos juntos es igual a la de un hueso sano: no se estrecha en ningún punto.

Esta es la EDT a lo largo de dos fragmentos pegados, con la grieta entre las columnas 13 y 14:

```
1 2 3 4 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 5 4 3 2 1
                          ↑ grieta
```

No hay valle en la grieta. El máximo es una meseta continua que pasa justo por encima de la fractura. El umbral da 1 semilla, h-máximos da 1 semilla, y watershed devuelve un solo fragmento. La EDT solo mide la forma, y la forma no cambia cuando el hueso se parte sin separarse.

**2. Dos semillas, pero el dique cae en el lugar equivocado**

Con el relieve −EDT, el dique queda donde se encuentran las dos aguas. Ese lugar depende de la geometría y de la posición de las semillas, no de dónde está la fractura. Si la fractura es oblicua, o si las semillas no están a la misma distancia de ella, la frontera se desplaza:

```
aaaaaaaaa!!!!!!bbbbbbbbbbbbbb
aaaaaaaaaa!!!!!bbbbbbbbbbbbbb
aaaaaaaaaaa!!!!bbbbbbbbbbbbbb
aaaaaaaaaaaaaa!bbbbbbbbbbbbbb
aaaaaaaaaaaaaaabbbbbbbbbbbbbb
aaaaaaaaaaaaaaa!bbbbbbbbbbbbb
aaaaaaaaaaaaaaa!!!!bbbbbbbbbb      ! = vóxel asignado al fragmento equivocado
```

La fractura real es diagonal, pero watershed cortó casi en vertical por la mitad: el 10 % de los vóxeles quedó en el fragmento equivocado. En el CT, la línea de contacto entre dos fragmentos pegados mide menos de 1 mm. A 1,27 mm/px se promedia con el hueso vecino (volumen parcial) y el relieve no tiene ninguna cresta que la marque.

### Gráficas (EDA e información relevante)

#### Prompt 1

> estamos trabajando con el dataset del reto penwin que contiene 100 tomografias computarizadas, en archivos mha, quisiera que para una mejor vizualizacion de estos datos me dieras el codigo para el siguiente tipo de graficas:
>
> graficas para el total de pacientes, el numero de casos, el numero de corte por paciente, las dimensiones de los volumenes, regiones presentes osea las regiones sacro, coxal izquierdo y derecho yel numero de fragmentos por pacientes, define que tipo de grafica debe de ser para un entendimiento mas claro.

**Respuesta (fragmento):**

Para que cada gráfica se entienda rápido, conviene elegir el tipo según la clase de dato: conteos discretos con barras, distribuciones continuas con histogramas o boxplots, y relaciones entre dos variables con dispersión. También conviene usar siempre el mismo color para cada región: sacro azul, coxal izquierdo naranja y coxal derecho verde.

Spacing por caso, cortes por volumen y resolución por corte (boxplot de las dimensiones y dispersión del spacing):

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.scatter(cases.dx, cases.dz, s=28, color=INK2, alpha=0.7, edgecolor=SURF, linewidth=0.8)
ax.set_xlabel("spacing en el plano dx = dy (mm)")
ax.set_ylabel("espesor de corte dz (mm)")
ax.set_title("Spacing por caso")
ax = axes[1]
ax.hist(cases.dim_z, bins=20, color=INK2, alpha=0.35, label="cortes totales")
ax.hist(cases.slices_with_bone, bins=20, color=INK2, label="cortes con hueso etiquetado")
ax.set_xlabel("cortes axiales por volumen")
ax.set_ylabel("casos")
ax.set_title("Cortes por volumen")
ax.legend()
ax = axes[2]
cols = {"sin recorte": "px_mm_no_crop_256", "recorte al cuerpo": "px_mm_crop_256", "recorte óseo": "px_mm_bonecrop_256"}
data = [cases[c].values for c in cols.values()]
bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, medianprops=dict(color=INK, linewidth=2))
for b in bp["boxes"]:
    b.set(facecolor=GRID, edgecolor=AXIS)
ax.set_yticks([1, 2, 3], list(cols))
ax.set_xlabel("mm por píxel en la entrada 256×256 (menor = más detalle)")
ax.set_title("Resolución efectiva según el recorte")
ax.grid(axis="y", visible=False)
fig.tight_layout()
savefig(fig, "02_geometria")
med = {k: cases[c].median() for k, c in cols.items()}
print("Mediana mm/píxel a 256:", {k: round(v, 2) for k, v in med.items()})
print(f"Ganancia del recorte óseo frente a sin recorte: {100 * (1 - med['recorte óseo'] / med['sin recorte']):.0f} % menos mm por píxel")
print(f"El recorte óseo contiene todo el hueso etiquetado en {int(cases.bone_inside_bonecrop.sum())}/{n} casos "
      f"(margen mínimo {cases.bonecrop_margin_min_mm.min():.1f} mm)")
print(f"El recorte al cuerpo contiene todo el hueso etiquetado en {int(cases.bone_inside_crop.sum())}/{n} casos")
```

#### Prompt 2

> a continuacion como antes se dijo quisiera que esta vez me dieras graficas sobre las tomografias en si, como por ejemplo, la orientacion de los 100 volumenes, el notero de cortes por caso, el spacinng de cada caso, la diferencia entre la resolucion de tanto ejes z y x/y:

**Respuesta (fragmento):**

Cada aspecto de las tomografías conviene mostrarlo con un tipo de gráfica distinto:

- **Orientación de los volúmenes:** es una categoría con dos valores (LPS o RAS), así que va en barras horizontales con el número de casos escrito al lado. Así se ve de inmediato cuántos volúmenes hay que reorientar.
- **Número de cortes por caso:** es una variable continua, así que va en un histograma. Los cortes totales y los cortes con hueso etiquetado se superponen para ver cuántos cortes de cada volumen aportan información.
- **Spacing de cada caso:** va en un diagrama de dispersión del spacing en el plano (dx = dy) contra el espesor de corte (dz), con un punto por paciente. Muestra los grupos de escáneres y qué tan lejos está cada volumen de un vóxel isotrópico.
- **Diferencia de resolución entre z y x/y:** se ve en la misma dispersión y se completa de dos formas. Un boxplot muestra los milímetros por píxel que le llegan al modelo en el plano x/y según el recorte. El cálculo de cuántos cortes equivalen a 2 mm en z traduce la resolución en z a la separación de canales de la entrada 2.5D.

Spacing por caso, cortes por volumen y resolución efectiva en x/y: el código es el mismo de la respuesta anterior (figura `02_geometria`).

#### Prompt 3

> quisiera que graficaras sobre lo siguiente,
> cuantos fragmentos tiene cada region (coxal izq -dere y sacro)
> cuantos fragmentos principales y secundarios hay, el numero total de fragmentos por paciente, la diferencia entre fragmentos grandes y pequeños, la distancia entre fragmentos, el tamaño de estos fragmentos.

**Resultado (gráficas de `04_notebook/01_eda_pengwin.ipynb`, §5 a §7):**

- **Fragmentos por región:** barras, un panel por región con su color y el número de casos sobre cada barra (1 fragmento = hueso sano).
- **Principales, secundarios y total por paciente:** conteo de cada tipo y total por caso (mediana 6, rango 3-9); el patrón de regiones fracturadas va como tabla de frecuencias.
- **Tamaño y diferencia entre grandes y pequeños:** histograma del volumen en escala logarítmica, principales y secundarios superpuestos; los cortes en 1, 5 y 10 cm³ dicen cuántos secundarios son pequeños (14,5 % mide menos de 5 cm³).
- **Distancia entre fragmentos:** distribución acumulada de la distancia de cada secundario a su principal, con una línea en la diagonal del vóxel que separa los fragmentos en contacto (88,7 %) de los separados.

Fragmentos del código (tal como está en el notebook):

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
kmax = int(cases[[f"n_frag_{r}" for r in REG]].max().max())
for ax, r in zip(axes, REG):
    vc = cases[f"n_frag_{r}"].value_counts().reindex(range(1, kmax + 1), fill_value=0)
    ax.bar(vc.index, vc.values, color=C[r], width=0.7)
    for x, v in vc.items():
        if v:
            ax.text(x, v + 0.8, str(v), ha="center", fontsize=8, color=INK)
    ax.set_title(f"{REG_NAME[r]} · fracturado en {pct((cases[f'n_frag_{r}'] > 1).mean())}")
    ax.set_xlabel("fragmentos en la región (1 = sano)")
    ax.set_xticks(range(1, kmax + 1))
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("casos")
fig.tight_layout()
savefig(fig, "05_fragmentos_region")

patt = cases.apply(lambda r: " + ".join(REG_NAME[x] for x in REG if r[f"n_frag_{x}"] > 1) or "ninguna", axis=1)
cases["patron"] = patt
cases["total_fragments"] = cases[[f"n_frag_{r}" for r in REG]].sum(axis=1)
print("Patrones de regiones fracturadas (casos):")
print(patt.value_counts().to_string())
print(f"\nFragmentos totales por caso: mediana {cases.total_fragments.median():.0f}, rango {cases.total_fragments.min()}-{cases.total_fragments.max()}")
```

```python
frags["tipo"] = np.where(frags.is_main, "principal", "secundario")
sec = frags[~frags.is_main].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax = axes[0]
bins = np.logspace(np.log10(max(frags.volume_cm3.min(), 0.05)), np.log10(frags.volume_cm3.max()), 30)
ax.hist(frags.loc[frags.is_main, "volume_cm3"], bins=bins, color=MAIN_C, alpha=0.85, label="principal")
ax.hist(sec.volume_cm3, bins=bins, color=SEC_C, alpha=0.85, label="secundario")
ax.set_xscale("log")
ax.set_xlabel("volumen del fragmento (cm³, escala log)")
ax.set_ylabel("fragmentos")
ax.set_title("Tamaño de los fragmentos")
ax.legend()
```

```python
sec["contacto"] = sec.contact_with_main.astype(bool)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax = axes[0]
ecdf(ax, sec.dist_to_main_mm.values, INK2)
diag = np.sqrt(2 * cases.dx**2 + cases.dz**2)
ax.axvline(diag.median(), color=REF, ls="--", lw=1.2)
ax.text(diag.median() * 1.08, 0.08, f"diagonal del vóxel\n≈ {diag.median():.2f} mm", color=REF, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("distancia al fragmento principal (mm, escala log)")
ax.set_ylabel("fracción acumulada de secundarios")
ax.set_title("Distancia de separación (GT)")
```

### Correcciones y mejoras

#### Prompt 1

> ayudanos a mejorar la subida del archivo mha ya que se etsa demorando demasiado debido a la ubicacion de la direccion ip de la web ya qu esta se encuentra en europa entonces debe de salir de cali, llegar  bogota y luego a europa y viceversa.

**Respuesta (fragmento):**

La demora no es solo la distancia sino cómo viaja el archivo. Desde Cali hasta el servidor en Europa hay unos 180 ms de ida y vuelta, y por HTTP/2 todos los trozos de la subida van por una sola conexión TCP: con ese retardo un trozo de 8 MB sube a unos 0,3 MB/s. Se puede atacar por dos lados:

1. **Mandar menos bytes:** el navegador comprime cada trozo con gzip (`CompressionStream`) antes de enviarlo. Un CT baja a la mitad o menos, porque gran parte del volumen es aire con el mismo valor.
2. **Usar varias conexiones:** mandar 4 trozos a la vez por WebSocket. nginx no lleva WebSocket sobre HTTP/2, así que cada WebSocket abre su propia conexión TCP y las cuatro suben en paralelo. Si algún proxy bloquea el WebSocket, la página manda los mismos trozos por `PUT`, de a uno.

Cada trozo lleva su número, así que puede llegar en cualquier orden o repetirse tras un corte sin dañar el archivo. En nginx hace falta una `location` para el WebSocket (`Upgrade` y `Connection "upgrade"`) y `client_body_buffer_size 8m`; en uvicorn, `--ws-per-message-deflate false`, porque los trozos ya llegan comprimidos. Medido con un CT de 420 MB: la subida pasó de unos 19 minutos a 33 segundos.

#### Prompt 2

> a modificar esa misma parte para que reciba mas de un corte.

**Respuesta (fragmento):**

Para recibir varios cortes, la página puede juntar los archivos elegidos (PNG, JPEG o DICOM, una carpeta o un `.zip`) en un solo archivo de lote con un índice, y subirlo por la misma vía de trozos. En el servidor se arma el volumen; lo difícil es el orden de los cortes, que se toma de lo más confiable que traigan los archivos:

1. la posición DICOM (`ImagePositionPatient` proyectada sobre la normal del plano);
2. el `SliceLocation` del PNG, si lo trae;
3. el número del nombre (`img_2` va antes que `img_10`), confirmado con el contenido;
4. si nada de eso sirve, el parecido entre cortes vecinos (orden espectral más 2-opt sobre miniaturas).

Un PNG no dice qué extremo es la cabeza, así que eso lo decide el modelo: el sacro queda por encima de los coxales. Se quitan los cortes repetidos, se interpolan los que faltan, la página muestra cómo se armó el volumen y deja descargarlo como `.mha`. Por seguridad se acotan el número de archivos (2000), el peso de cada uno (64 MB) y el total descomprimido (1,5 GB). Medido en casos reales: el orden solo por contenido recuperó exactas 39 de 40 pilas; una serie DICOM desordenada da los mismos fragmentos que el `.mha` (Dice 1,000) y los PNG de 8 bits, Dice 0,999.
