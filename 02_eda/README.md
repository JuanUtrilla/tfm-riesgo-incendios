# 2 · Análisis exploratorio: el cubo y el dataset

Hay dos exploraciones. La del cubo mira qué hay en las 498,530 celdas × día
antes de muestrear nada. La del dataset mira el conjunto ya muestreado de la
iteración 1, solo en el split de entrenamiento (2015-2018), para no tomar
decisiones mirando el test.

## Parte A. El cubo

El documento completo está en [`ANALISIS_CUBO.md`](ANALISIS_CUBO.md).

### Entrenar con el cubo y servir con ERA5-Land

Se comprobó si la meteorología del cubo IberFire servía para entrenar o si
hacía falta bajar 72 meses de ERA5-Land (unas 30 horas de cola en el CDS, el
*Climate Data Store* de Copernicus). El cubo es ERA5-Land reprocesado.
Temperatura máxima, humedad y viento coinciden en los nodos con correlación
0.987-0.998 y sesgo cero. Solo difiere la precipitación (la del cubo es la de
ERA5-Land dividida por 2.02).

### La prevalencia real

Celdas-día con `is_fire` (EFFIS, *European Forest Fire Information System*;
celda con al menos 5 ha quemadas) por cada 100,000:

| Año | Anual | Verano (jun-sep) | Días con alguna celda | Máximo de celdas en un día |
|---|---|---|---|---|
| 2018 | 0.40 | 0.74 | 90 | 60 |
| 2020 | 3.08 | 6.40 | 190 | 361 |
| 2022 | 14.16 | 37.96 | 223 | 1,678 |
| 2024 | 1.82 | 2.40 | 190 | 112 |

En un verano normal arden entre 2 y 6 celdas de cada 100,000 al día. El
conjunto de la iteración 1 tenía un 25 % de positivos por construcción. Esa
distancia anticipa el problema de especificación del capítulo 4.

Solo el 2.97 % de las celdas tuvo alguna ignición EGIF (Estadística General de
Incendios Forestales) entre 2015 y 2020, y el 2.68 % tuvo superficie quemada
según EFFIS entre 2021 y 2024.

![Prevalencia real del fuego y persistencia de la etiqueta](../docs/MEMORIA/figs/f7_prevalencia.png)

*Celdas-día quemadas por cada 100,000 y persistencia de `is_fire`, año a año.*

### La etiqueta persiste

La probabilidad de que una celda con fuego hoy siga con fuego mañana es 0.51;
a los 3 días, 0.23; a los 7, 0.04. `is_fire` marca todos los días en que una
celda sigue ardiendo, y no solo el de la ignición. Por eso las evaluaciones
con EFFIS usan `primer_dia` (fuego hoy y no ayer): 7,301 de las 28,135
celdas-día de los veranos de evaluación. Contar los días de arrastre sería
puntuar el mismo incendio varias veces.

### La autocorrelación decide cómo se parten los datos

Índice de Moran con vecindad reina:

| Campo | 1 km | Bloques de 10 km |
|---|---|---|
| nº de igniciones EGIF 2015-20 por celda | 0.28 | 0.63 |
| nº de celdas EFFIS 2021-24 por celda | 0.81 | 0.40 |
| FWI medio de verano | 0.99 | 0.95 |

EFFIS a 1 km es casi todo perímetro: celdas contiguas del mismo incendio.
Partir al azar entre entrenamiento y prueba infla el AUC (*Area Under the ROC
Curve*) en +0.036 frente a partir por bloques de 100 km. Por eso todas las
particiones del trabajo son espaciales o temporales.

### Qué script produce cada resultado

| Script | Qué mide | Dónde está |
|---|---|---|
| `dos_00_cubo_etiquetas.py` | Prevalencia por año, persistencia, EGIF frente a EFFIS | `04_bisagra/44_es_la_especificacion/` |
| `dos_01_celdas.py` | La tabla por celda: censo de las 498,530 | `05_iteracion2/51_celdas/` |
| `dos_04_analisis.py` | Autocorrelación, coste de la partición aleatoria, susceptibilidad estática | `05_iteracion2/54_analisis/` |
| `malla_02_vs_cubo.py` | El cubo frente a ERA5-Land nativo | `04_bisagra/42_no_es_la_meteo/prototipo_TFM_fuego/` |

Cada script está en la carpeta del capítulo donde se ejecuta; aquí se reúnen
sus resultados.

## Parte B. El dataset de la iteración 1

`eda_dataset.py`, el único script de esta carpeta, calcula distribuciones por
clase, correlaciones, estacionalidad y la normalización local del FWI (*Fire
Weather Index*) sobre el split de entrenamiento. Ese split tiene 53,563 filas
(13,577 positivos y 39,986 negativos). El resumen numérico está en
[`figuras/eda_resumen.log`](figuras/eda_resumen.log).

Medianas de positivos frente a negativos:

| Variable | Positivo | Negativo |
|---|---|---|
| `fwi` | 12.41 | 1.77 |
| `fwi_pctl_local` | 84.33 | 47.00 |
| `rh_min` | 39.88 | 58.73 |
| `dias_sin_lluvia` | 9.00 | 2.00 |
| `precip_30d` | 16.56 | 36.17 |
| `popdens` | 9.67 | 9.67 |
| `dist_carreteras` | 0.63 | 0.63 |

De la tabla salen dos lecturas. La primera: las variables estáticas no separan
nada en este diseño, porque población y distancia a carreteras tienen la misma
mediana en los dos grupos. Es lo esperable con negativos tomados de la misma
celda en otros días, y el capítulo 4 mide esa consecuencia.

La segunda: el percentil local del FWI separa mejor que el FWI absoluto. En
Galicia un FWI de 3.1 ya es el percentil 91.7 de su historia, y en Canarias un
41.4 se queda en el 72.8. Como ablación
(`03_iteracion1/35_ablaciones/ablacion_features.py`), añadir el percentil local
al FWI absoluto aporta +0.042 de AUC-PR (de 0.668 a 0.710).

![El FWI absoluto frente al percentil local](../docs/MEMORIA/figs/f8_normalizacion.png)

*El FWI absoluto de los días de incendio va de 3 en Asturias a 41 en Andalucía; en percentil local todos se concentran arriba.*

![Qué separa un positivo de un negativo en el conjunto de la iteración 1](../docs/MEMORIA/figs/f3_separacion.png)

*Medianas de positivos y negativos: las variables meteorológicas separan los dos grupos y las estáticas coinciden.*

### Figuras

| Figura | Qué enseña |
|---|---|
| [`eda_distribuciones.png`](figuras/eda_distribuciones.png) | Distribución de cada variable, positivos frente a negativos |
| [`eda_correlaciones.png`](figuras/eda_correlaciones.png) | Correlación entre variables |
| [`eda_estacionalidad.png`](figuras/eda_estacionalidad.png) | El ciclo anual del fuego |
| [`eda_normalizacion_local.png`](figuras/eda_normalizacion_local.png) | Por qué el FWI hay que normalizarlo por sitio |

## Lo que se dejó fuera

El bloque de visión por satélite (Sentinel-2, D-Fire, timelapses y los casos de
Sotalvo, Luna y Ponteareas) se desarrolló, pero no entra en la memoria. Se
conserva en el repositorio original.
