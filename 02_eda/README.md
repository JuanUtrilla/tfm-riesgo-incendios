# 2 · Análisis exploratorio: el cubo y el dataset

> Dos exploraciones distintas y las dos necesarias: **el cubo** —qué hay
> realmente en las 498.530 celdas × día antes de muestrear nada— y **el
> dataset** ya muestreado de la iteración 1, restringido al split de
> entrenamiento (2015-2018) para que ninguna decisión se tome mirando el test.

## Parte A · El cubo

Documento completo: [`ANALISIS_CUBO.md`](ANALISIS_CUBO.md).

### La decisión previa: entrenar con el cubo, servir ERA5-Land

Antes de nada había que saber si la meteorología del cubo IberFire servía para
entrenar o hacía falta bajar 72 meses de ERA5-Land (≈30 h de cola en CDS).
**Sirve**: el cubo *es* ERA5-Land reprocesado, y tmax, humedad y viento
coinciden en los nodos con correlación 0,987-0,998 y sesgo 0. Lo único que
difiere es la precipitación (cubo = ERA5-Land / 2,02).

### La prevalencia real, que es el número que lo cambia todo

Celdas-día con `is_fire` (EFFIS ≥5 ha) por cada 100.000:

| Año | Anual | Verano (jun-sep) | Días con ≥1 celda | Máx. celdas/día |
|---|---|---|---|---|
| 2018 | 0,40 | 0,74 | 90 | 60 |
| 2020 | 3,08 | 6,40 | 190 | 361 |
| **2022** | **14,16** | **37,96** | 223 | **1.678** |
| 2024 | 1,82 | 2,40 | 190 | 112 |

**2-6 por 100.000 en un verano normal**, frente al **25 % de positivos** que
tenía el conjunto de la iteración 1 por construcción: cinco órdenes de
magnitud. Este dato, medido aquí y no en el capítulo 4, es el que anticipa todo
el problema de especificación.

Solo el **2,97 %** de las celdas tuvo alguna ignición EGIF en 2015-20, y el
**2,68 %** algún EFFIS en 2021-24. El fuego es un fenómeno concentrado.

### La etiqueta persiste, y eso obliga a definirla mejor

`P(fuego en t+1 | fuego en t) = 0,51`, a los 3 días 0,23, a los 7 días 0,04.
`is_fire` marca **días ardiendo**, no igniciones. Por eso la evaluación con
EFFIS usa `primer_dia` —fuego hoy y no ayer—: 7.301 de 28.135 celdas-día en los
veranos de evaluación. Contar los días de arrastre sería puntuarse el mismo
incendio siete veces.

### La autocorrelación decide cómo se parte

Moran's I con vecindad reina:

| Campo | 1 km | Bloques 10 km |
|---|---|---|
| nº EGIF 2015-20 por celda | 0,28 | 0,63 |
| nº EFFIS 2021-24 por celda | **0,81** | 0,40 |
| FWI medio de verano | 0,99 | 0,95 |

EFFIS a 1 km es casi todo perímetro: celdas contiguas del mismo incendio. La
consecuencia es directa y está medida: partir al azar **infla el AUC en
+0,036** frente a partir por bloques de 100 km. Todas las particiones del
trabajo son espaciales o temporales, nunca aleatorias.

### Qué script produce cada cosa

| Script | Qué mide | Dónde vive |
|---|---|---|
| `dos_00_cubo_etiquetas.py` | Prevalencia, persistencia, EGIF vs EFFIS | `04_bisagra/44_es_la_especificacion/` |
| `dos_01_celdas.py` | La tabla por celda: censo de las 498.530 | `05_iteracion2/51_celdas/` |
| `dos_04_analisis.py` | Autocorrelación, partición, susceptibilidad, deriva | `05_iteracion2/54_analisis/` |
| `malla_02_vs_cubo.py` | El cubo contra ERA5-Land nativo | `04_bisagra/42_no_es_la_meteo/prototipo_TFM_fuego/` |

Están repartidos por capítulos a propósito: cada uno se ejecuta donde hace
falta en el hilo. Este capítulo es donde se leen juntos sus resultados.

## Parte B · El dataset de la iteración 1

`eda_dataset.py`, solo sobre train (2015-2018). Resumen numérico en
[`figuras/eda_resumen.log`](figuras/eda_resumen.log): 53.563 filas (13.577
positivos / 39.986 negativos).

Medianas de positivo contra negativo, que enseñan qué separa de verdad:

| Feature | Positivo | Negativo |
|---|---|---|
| `fwi` | 12,41 | 1,77 |
| `fwi_pctl_local` | **84,33** | **47,00** |
| `rh_min` | 39,88 | 58,73 |
| `dias_sin_lluvia` | 9,00 | 2,00 |
| `precip_30d` | 16,56 | 36,17 |
| `popdens` | 9,67 | 9,67 |
| `dist_carreteras` | 0,63 | 0,63 |

Dos lecturas. Las **estáticas no separan nada** en este diseño —población y
distancia a carreteras tienen la misma mediana en positivos y negativos—, lo
cual es coherente con lo que el capítulo 4 acabará demostrando: con negativos
tomados de la misma celda en otros días, las variables de la celda **no pueden**
discriminar, porque son idénticas en los dos lados.

Y el **percentil local separa mejor que el FWI absoluto**, que es la
contribución del TFM: en Galicia un FWI de 3,1 es percentil 91,7, mientras en
Canarias un 41,4 se queda en el 72,8. Cuantificado como ablación en
`03_iteracion1/35_ablaciones/ablacion_features.py`: **+0,042 de AUC**.

### Figuras

| Figura | Qué enseña |
|---|---|
| [`eda_distribuciones.png`](figuras/eda_distribuciones.png) | Distribución de cada feature, positivos contra negativos |
| [`eda_correlaciones.png`](figuras/eda_correlaciones.png) | Correlación entre features |
| [`eda_estacionalidad.png`](figuras/eda_estacionalidad.png) | El ciclo anual del fuego |
| [`eda_normalizacion_local.png`](figuras/eda_normalizacion_local.png) | Por qué el FWI hay que normalizarlo por sitio |

## Lo que se dejó fuera

El bloque de visión por satélite —Sentinel-2, D-Fire, timelapses y las
historias de los incendios de Sotalvo, Luna y Ponteareas— fue exploración real
pero no entra en el hilo de la memoria. Sigue en `TFM_fuego`.
