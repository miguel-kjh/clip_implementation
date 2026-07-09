# Evaluación: ¿recupera bien? ¿han colapsado los embeddings?

La val loss de este proyecto **no sirve** para saber si el modelo funciona. La loss de
`CLIPModel._compute_losses` construye sus targets a partir de los propios embeddings,
así que "todos los embeddings son iguales" la satisface trivialmente y se queda clavada
en `ln(batch_size)` (3.466 con `--batch-size 32`). Un modelo colapsado puede tener una
loss aparentemente razonable y un Recall de puro azar.

De ahí este módulo. Responde dos preguntas **separadas**:

1. **¿Recupera bien?** → métricas de retrieval (Recall@K y compañía).
2. **¿Ha colapsado el espacio?** → métricas geométricas sobre los embeddings.

---

## Comandos

Todo lo demás (dataset, tamaño de imagen, `max_length`, `val_split`, semilla) se lee del
`hparams.json` del experimento, así que solo hace falta el directorio.

**Evaluación normal** (los dos splits, galería igualada a 1000 imágenes):

```bash
python src/eval/evaluate.py \
  --experiment-dir checkpoints/resnet50_distilbert-base-uncased_imgmlp_txtmlp_bs32_ep4_flickr8k_20260709-184330 \
  --gallery-size 1000
```

Imprime las tablas y escribe `metrics.json` dentro del directorio del experimento.

**Un solo split:**

```bash
python src/eval/evaluate.py --experiment-dir checkpoints/<exp> --split val-disjoint --gallery-size 1000
```

**Control de azar** — el mismo modelo con pesos aleatorios, sin entrenar. Sirve para
saber qué cifras significan "no ha aprendido nada". No escribe `metrics.json`:

```bash
python src/eval/evaluate.py --experiment-dir checkpoints/<exp> --gallery-size 1000 --untrained
```

**Otros flags:**

| Flag | Qué hace |
|---|---|
| `--gallery-size N` | Recorta cada split a N imágenes. **Úsalo siempre** para comparar (ver más abajo). 1000 es el estándar de Flickr. |
| `--ks 1 5 10 50` | Qué valores de K reportar en el Recall@K. |
| `--split` | `val` o `val-disjoint`. Repetible. Por defecto, ambos. |
| `--batch-size`, `--num-workers`, `--device` | Rendimiento. |
| `--output` | Nombre del JSON de salida (por defecto `metrics.json`). |

---

## Las métricas de retrieval

La **galería** son las imágenes únicas del subconjunto. Cada caption tiene exactamente
una imagen correcta; cada imagen tiene ~5 captions correctas.

### Recall@K

De todas las consultas, qué porcentaje encuentra la respuesta correcta **entre los K
primeros resultados**. R@1 = "acierta a la primera". R@10 = "la correcta está en el
top 10". Más alto es mejor.

Se calcula en las dos direcciones:

- **texto→imagen**: doy una caption, busco su imagen entre las N de la galería. Es el
  caso de uso real (buscar imágenes escribiendo).
- **imagen→texto**: doy una imagen, busco sus captions. Acierta si **cualquiera** de las
  ~5 captions correctas entra en el top-K (es el protocolo estándar de Flickr).

**Con qué comparar.** El azar es `K/N`. Con una galería de 1000 imágenes, R@1 de azar es
0.1%. Si tu R@1 ronda esa cifra, el modelo no ha aprendido nada — probablemente colapsó.
El script imprime siempre el valor del azar en la cabecera.

### Mediana del rank (`medRank`) y rank medio

La **posición** en la que aparece la respuesta correcta. Mediana 3 significa que en la
mitad de las consultas la correcta salió entre las 3 primeras. Más bajo es mejor; 1 es
perfecto.

Es útil porque el Recall@K **satura y esconde información**: dos modelos pueden tener
ambos R@10 = 0%, pero uno deja la correcta en la posición 11 y el otro en la 800. El
Recall no los distingue; la mediana del rank sí.

Se usa la mediana y no la media porque unas pocas consultas catastróficas (rank 900)
disparan la media sin que el modelo sea malo en general.

### MRR (Mean Reciprocal Rank)

La media de `1 / posición_de_la_correcta`. Vale 1.0 si siempre acierta a la primera,
0.5 si siempre queda segunda, y tiende a 0 si se va al fondo. Resume en un único número
"cómo de arriba" suele quedar la respuesta correcta. Más alto es mejor.

### Empates

Si el modelo colapsa, **todas las similitudes son iguales**. Los empates se resuelven
en contra del acierto: la respuesta correcta se considera detrás de todas las que
empatan con ella. Sin esta precaución un modelo colapsado sacaría **Recall@1 = 100%**
(ningún distractor puntúa *estrictamente* más que la correcta), justo el fallo que
queremos detectar disfrazado de perfección.

---

## Las métricas de colapso

"Colapso" = todos los embeddings acaban apuntando prácticamente al mismo sitio, o
viviendo en un subespacio diminuto. El modelo deja de distinguir unas imágenes de otras.
Se calculan por separado para imagen y para texto.

### Rango efectivo (`rango efec`) — **la métrica importante**

Cuántas dimensiones está usando **de verdad** el espacio de embeddings, de las 256
disponibles (`projection_dims`). Técnicamente es `exp(entropía)` del espectro de valores
singulares (métrica *RankMe*). Más alto es mejor.

- Colapso total → **≈ 1.0** (todo cabe en una dirección).
- Modelo sano en este repo → **≈ 40–75**.
- Aleatorio sin entrenar → **≈ 240** (usa todo el espacio, pero sin estructura útil).

Es la métrica **fiable** para detectar colapso, por las dos razones de abajo.

### Coseno medio (`cos medio`) — engañoso por sí solo

El coseno medio entre pares de embeddings distintos. La intuición dice "1.0 = colapso,
0 = sano". **En CLIP esa intuición falla.**

Un modelo perfectamente sano de este repo da **0.81–0.91**, porque todos los embeddings
comparten una componente media grande: es el **cone effect** (viven en un cono estrecho,
pero se distinguen bien *dentro* del cono). Un modelo colapsado da ~1.0. Los dos "parecen
altos".

Y al revés: con colapso *dimensional parcial* (los embeddings ocupan 8 de 256
dimensiones) el coseno medio sale **≈ 0.00**, que parece sanísimo, mientras el rango
efectivo lo delata con un 8.

Conclusión: el coseno medio solo es señal de alarma pasado ~0.99. Por eso el script
imprime un aviso de *cone effect* (informativo) en vez de una alarma de COLAPSO cuando el
coseno es alto pero el rango efectivo está sano.

### PCA top-1 (`PCA top1`)

Qué fracción de la varianza explica la primera componente principal. Cerca de 1.0 =
casi toda la variación va en una única dirección = mal. En un modelo sano de aquí ronda
0.15. Es la misma idea que el rango efectivo, más fácil de leer.

### Desviación típica por dimensión (`std/dim`)

Cuánto varían los embeddings (ya normalizados) en cada dimensión, promediado. Un espacio
sano y bien repartido tiende a `1/√256 = 0.0625`. Colapso → 0.0000. En este repo un
modelo bueno da ~0.025: hay variación real, pero concentrada (otra vez el cone effect).

### Norma (`norma`)

La longitud media de los embeddings **sin normalizar**. La loss no normaliza L2, así que
la norma puede dispararse. Sirve para cazar la divergencia de la cabeza `swiglu`, la
única que no termina en `LayerNorm`: allí la norma crece sin control y la loss explota
(de 3.8 a 40 en una época). El script avisa por encima de 50.

### Uniformity

`log E exp(-2·distancia²)` entre pares (Wang & Isola 2020). Mide cómo de bien repartidos
están los embeddings por la esfera. **Más negativo = mejor repartido.** Colapso total → 0.0.
Aleatorio → ≈ -3.97. Modelo sano de aquí → ≈ -0.7.

### Alignment

`E‖e_imagen − e_texto‖²` sobre los **pares correctos**. Mide si la imagen y su caption
acaban cerca. Más bajo es mejor. Sin entrenar → ≈ 2.0 (vectores ortogonales).

Se lee **junto con uniformity**, y ahí está la gracia: un modelo colapsado tiene
alignment **excelente** (todo está pegado a todo, incluidos los pares correctos) y
uniformity **pésima**. Solo mirando las dos a la vez se distingue "ha aprendido" de "lo
ha aplastado todo".

### Modality gap

Distancia entre el centroide de las imágenes y el de los textos. Si es grande, las dos
ramas viven en conos separados del espacio aunque cada una internamente esté sana.
Importa aquí más de lo normal porque la loss no normaliza L2. Sin entrenar → ≈ 1.37;
entrenado → ≈ 0.17.

---

## Los dos splits, y por qué sus cifras no se comparan sin cuidado

`trainer.py` parte el dataset **a nivel de caption**. Como hay ~5 captions por imagen,
unas 4 caen en train y una en validación: **la imagen de validación ya se vio durante el
entrenamiento**. Por eso hay dos splits:

- **`val`** — reproduce exactamente el subconjunto sobre el que se eligió `best.pt`.
  Es coherente con el entrenamiento, pero optimista.
- **`val-disjoint`** — agrupa por imagen antes de partir. Ninguna imagen de validación
  aparece en train. Es la cifra honesta y la comparable con resultados publicados.

Hay **dos trampas** al comparar sus Recalls, y el script avisa de ambas:

**1. El tamaño de la galería domina el Recall.** Cuantas más imágenes compiten, más
difícil acertar. Sin igualar, `val` compite contra 5421 imágenes y `val-disjoint` contra
1618, y eso solo ya mueve el R@1 de 4.96% a 14.70%. Eso **no mide la fuga de datos**,
mide el tamaño de la galería. Por eso hay que pasar `--gallery-size`.

**2. Las captions por imagen difieren** (1.5 en `val`, 5.0 en `val-disjoint`). Como
imagen→texto acierta si *cualquiera* de las captions correctas entra en el top-K, tener
5 candidatas correctas en vez de 1.5 lo hace estructuralmente más fácil. **Solo
texto→imagen es comparable entre splits.**

Con la galería igualada a 1000, resulta que la fuga a nivel de caption **apenas infla**
el Recall: 16.0% (`val`) vs 18.6% (`val-disjoint`) a 1 época.

---

## Cifras de referencia

Checkpoints de este repo, `--gallery-size 1000`, split `val-disjoint`:

| Modelo | R@1 texto→imagen | R@1 imagen→texto | medRank | rango efec (img/txt) | cos medio | ¿colapso? |
|---|---|---|---|---|---|---|
| Sin entrenar (azar) | 0.06% | 0.20% | 497 | 10.8 / 72.9 | 0.97 / 0.90 | — (control) |
| 1 época | 18.58% | 25.30% | 7 | 63.7 / 40.2 | 0.91 / 0.91 | no (cone effect) |
| 4 épocas | 40.28% | 55.40% | 2 | 75.4 / 58.6 | 0.81 / 0.83 | no |
| Colapsado (`--dropout 0`) | 0.00% | 0.00% | último | 1.0 / 1.0 | 1.00 / 1.00 | **sí** |

El azar con 1000 imágenes de galería es R@1 = 0.10%.

---

## Cómo leer un resultado en 10 segundos

1. Mira el **rango efectivo**. Si es ~1, el modelo ha colapsado; nada más importa.
   ¿Entrenaste con `--dropout 0`? Ese es el motivo (ver `CLAUDE.md`).
2. Mira el **R@1 de texto→imagen** contra el azar que imprime la cabecera. Si están al
   mismo nivel, no ha aprendido.
3. Si el coseno medio es ~0.9 pero el rango efectivo está sano, **no es colapso**, es el
   cone effect. Normal en CLIP.
4. Para comparar dos modelos entre sí, asegúrate de usar el **mismo `--gallery-size` y el
   mismo split**. Si no, las cifras no significan nada.

---

## Ficheros

| Fichero | Contenido |
|---|---|
| `evaluate.py` | CLI, tablas, diagnóstico y `metrics.json`. |
| `retrieval.py` | Recall@K, ranks y MRR en ambas direcciones. |
| `collapse.py` | Rango efectivo, coseno, PCA, normas, uniformity, alignment, modality gap. |
| `embeddings.py` | Extrae los embeddings del subconjunto (cada imagen se codifica una vez, no 5). |
| `splits.py` | Reproduce el val split del trainer, construye el disjunto por imagen, iguala galerías. |
