# Borrador de la memoria — bloque de Machine Learning

> **Dos ficheros con papeles distintos.**
> Este `.md` es el **borrador de trabajo completo**: todo el material disponible,
> con más tablas y más detalle del que cabe en la memoria.
> `BORRADOR_memoria_ML.tex` es el **recorte que se entrega**, ajustado al límite
> de 7 páginas: 5 de texto más el hueco de las figuras. Lo que se cae de ahí no
> se pierde — va al repositorio, que es exactamente su papel (cap. 8).
>
> Los `[PENDIENTE]` marcan lo que falta; los `[FIG]`/`[TAB]`, el hueco de cada
> figura o tabla.

**Hilo narrativo:** un primer ciclo completo (datos → EDA → features → train →
evaluación) que da buenas métricas en val/test pero rinde mal en producción; una
bisagra de diagnóstico que apunta a la **definición de la target**; y un segundo
ciclo que reentrena con otra etiqueta y otro muestreo y sí mejora en producción.
No se narran los errores de implementación corregidos por el camino: la memoria
cuenta el razonamiento, no el registro de incidencias.

---

## 1. Extracción de datos  [~1,5–2 páginas]

Nueve fuentes heterogéneas —un NetCDF de 30 GB, una API con clave, un PDF
diario— convertidas en una fila de 46 números para una celda y un día. Lo que
cambia entre las dos iteraciones del trabajo no son los datos: es qué se toma
como positivo, qué como negativo y qué como verdad.

### 1.1 IberFire — el cubo de datos

Datacubo ML-ready publicado por Tekniker y la UPV/EHU (Erzibengoa,
Gómez-Omella y Goienetxea; arXiv:2505.00837; Zenodo `10.5281/zenodo.15798999`,
**CC-BY 4.0**). Un único NetCDF de **30,3 GB** con **261 variables**.

| Propiedad | Valor |
|---|---|
| Proyección y rejilla | EPSG:3035, **1.188 × 920 celdas de 1 km** |
| Cobertura | España peninsular + Baleares (excluye Canarias, Ceuta y Melilla) |
| Celdas útiles | **498.530** peninsulares |
| Periodo | 01-dic-2007 → 31-dic-2024 (**6.241 días**) |
| Granularidad | 1 km × 1 día |

**De dónde salen por dentro sus variables** (importante para citar bien): la
meteorología es **ERA5-Land** (~9 km) remuestreada a 1 km por **vecino más
próximo**; la vegetación, Copernicus CLMS (capas decadales interpoladas a
diario); el terreno, EU-DEM (~30 m); distancias y población, OSM y WorldPop; el
uso del suelo, CORINE con cortes 2006/2012/2018.

El cubo aporta 45 de las 46 variables del modelo. Lo que **no** trae y se añade
aparte: los componentes sueltos del FWI, el VPD, los rayos y —sobre todo— la
etiqueta: su `is_fire` (celda quemada EFFIS ≥5 ha) no se usa como *target* en la
primera iteración.

### 1.2 EGIF — la etiqueta de la primera iteración

Registro oficial español de incendios forestales (MAPA, publicado por **Civio**,
CC BY-SA 3.0). Una fila por incendio, con fecha y **punto** GPS —no perímetro.

- De 292.181 registros (1968–2023) se usan **19.561 incendios de 2015–2020** con
  coordenadas válidas, proyectados a EPSG:3035, asignados a su celda por vecino
  más próximo y deduplicados por (celda, fecha) conservando el de mayor
  superficie.
- **La limitación que fija la ventana temporal de todo el trabajo:** el registro
  está incompleto desde 2021 — 888, 226 y 23 incendios en 2021, 2022 y 2023. Es
  un artefacto del retraso de consolidación, no la realidad: 2022 fue el peor año
  real de la década. Usar 2021+ contaminaría los positivos (faltan) *y* las
  pseudo-ausencias (podrían ser incendios reales sin registrar).
- **Uso doble:** etiqueta, y cinco variables de historial de fuego.
- **Descartadas por fuga:** `time_ctrl`/`time_ext` (solo se conocen *después* del
  incendio) y `personal`/`medios`/`gastos` (consecuencia, no predictor).

### 1.3 ERA5-Land — la meteorología

Reanálisis de Copernicus/ECMWF (~9 km). Entra **dos veces y de forma distinta**,
y esa asimetría reaparece en el capítulo 7:
- **al entrenar**, a través del cubo (que *es* ERA5-Land reprocesado);
- **al servir**, descargado directamente del CDS sobre **5.605 nodos**, más la
  previsión **IFS** de ECMWF vía Open-Meteo para D y D+1, donde el reanálisis
  todavía no existe.

Se verificó que el cubo y ERA5-Land nativo coinciden en los nodos: correlación
**0,987–0,998** y sesgo cero en temperatura, humedad y viento. Lo único que
difiere es la precipitación (cubo = ERA5-Land / 2,02, ver §2).

### 1.4 FIRMS — detecciones de fuego activo

Anomalías térmicas del sensor **VIIRS** (NASA/LANCE), píxel de 375 m, varias
pasadas al día. **~256.000 detecciones** sobre Iberia 2015–2024.

Se usan como **variable predictora**, nunca como etiqueta de entrenamiento, en
ventana **[D−7, D−1]** y radio de **50 km**: `frp_max_50km_7d` (potencia
radiativa máxima) y `n_detec_50km_7d` (número de detecciones). El día D queda
fuera a propósito: sería circular con la ignición que se quiere predecir.

*Limitación:* FIRMS incluye fuentes térmicas industriales fijas (fundiciones,
papeleras) que se filtran por recurrencia de píxel en la validación externa,
pero no en la extracción de esta variable.

### 1.5 EFFIS — la etiqueta de la segunda iteración

Perímetros de superficie quemada de Copernicus EMS. En el cubo aparece como
`is_fire` (celda quemada **≥5 ha**, diario, 2008–2024); para 2026 se descarga el
GeoJSON de la temporada: **1.895 polígonos, 307.850 ha**, mediana de 8 ha.

Es la etiqueta del segundo ciclo y uno de los tres jueces. Al ser MODIS, es
ciego a lo pequeño y sitúa el fuego en el **perímetro**, no en el punto de
ignición.

### 1.6 Las demás fuentes

| Fuente | Papel | Nota |
|---|---|---|
| **WGLC** (rayos) | Proxy de causa natural | Rejilla 0,5° (~55 km): la más gruesa de todas |
| **AEMET** | Meteorología de la producción de la iteración 1 | Observación horaria + climatología diaria |
| **MITECO** | El juez rápido: parte diario D+1 | Se parsea de un PDF |
| **Extra** | Carreteras, censo ganadero | Variables humanas y estructurales |

### 1.7 Dos avisos que condicionan todo lo demás

1. **El FWI del cubo son las 13 UTC.** Verificado antes de construir nada.
   Servir un FWI de otra hora introduce un factor ~2 respecto al entrenamiento.
2. **ERA5-Land llueve de más.** Frente a AEMET, **+8,00 mm en 30 días** (sesgo
   húmedo conocido del reanálisis: llovizna difusa). De ahí se propaga todo:
   menos días sin lluvia, combustible más húmedo, FWI diez puntos más bajo. Es
   la razón de que entrenar y servir con la misma fuente importe tanto.

---

## 2. Análisis exploratorio de datos  [~1,5 páginas]

Dos exploraciones distintas y las dos necesarias: **el cubo** —qué hay realmente
en las 498.530 celdas × día antes de muestrear nada— y **el conjunto ya
muestreado**, restringido al *split* de entrenamiento para que ninguna decisión
se tome mirando el test. Cada resultado de este capítulo justifica una decisión
posterior, y uno de ellos **anticipa el problema entero del trabajo**.

### 2.1 Las dos etiquetas no son la misma cosa

|  | EGIF (Civio) | EFFIS en el cubo (`is_fire`) |
|---|---|---|
| Qué es | Ignición registrada, punto, ≥1 ha | Celda quemada, ≥5 ha, diaria |
| Años útiles | 2015–2020 | 2008–2024 |
| n en 2015–20 | 19.866 igniciones → **19.561** celda-día | **21.785** celdas-día · 9.942 celdas |
| Tamaño | p50 3,6 ha · p90 33 · p99 343 · **41 % ≥5 ha** | Por definición ≥5 ha |

**Y se solapan poco.** Solo el **2,6 %** de los incendios EGIF de 5–25 ha tienen
`is_fire`=1 en su celda ese día; el 14,6 % de los de 25–100 ha, el 34 % de los
de 100–500 ha y el **59 %** de los de más de 500 ha.

**Estacionalidad distinta**, que es el hallazgo más contraintuitivo del EDA: en
el EGIF **el 38 % del año va de febrero a abril** (marzo 15 %, abril 11 %,
febrero 10 %) y junio-septiembre suma otro 38 %. EFFIS, en cambio, se concentra
en julio y agosto. → **Justifica no entrenar solo con verano**, decisión que
luego se ablaciona y se confirma (cap. 6.4).

**Sesgo geográfico:** Galicia 18,6 %, Asturias 15,8 %, Castilla y León 15,4 %,
Cantabria 13,4 %, Extremadura 10,9 %. → Justifica el bloqueo espacial (§2.3).

**Causa:** intencionado **68 %**, negligencia 12 %, rayo **2,4 %**. → Es la cota
superior de cualquier modelo meteorológico: el 97 % de las igniciones tienen
causa humana, y la meteorología solo explica si el combustible estaba listo, no
si alguien iba a prender fuego ese día. Es el marco honesto para los días de
fallo del cap. 7.5.

### 2.2 La prevalencia real: el número que lo cambia todo

Celdas-día con `is_fire` por cada 100.000, sobre las 498.530 celdas:

| Año | Anual | Verano (jun–sep) | Días con ≥1 celda | Máx. celdas/día |
|---|---|---|---|---|
| 2018 | 0,40 | 0,74 | 90 | 60 |
| 2020 | 3,08 | 6,40 | 190 | 361 |
| **2022** | **14,16** | **37,96** | 223 | **1.678** |
| 2024 | 1,82 | 2,40 | 190 | 112 |

**2–6 por 100.000 en un verano normal, frente al 25 % de positivos que tiene por
construcción el conjunto de la primera iteración: cinco órdenes de magnitud.**
Medido aquí, en el capítulo 2, y no en el 5 — el EDA ya contenía el diagnóstico
que costó un ciclo entero entender.

El fuego además está **concentrado**: solo el **2,97 %** de las celdas tuvo
alguna ignición EGIF en 2015–20, y el 2,68 % algún EFFIS en 2021–24.

### 2.3 La etiqueta persiste, y la señal está autocorrelada

`P(fuego en t+1 | fuego en t) = 0,51`; a los 3 días 0,23; a los 7 días 0,04.
`is_fire` marca **días ardiendo, no igniciones** → por eso la evaluación con
EFFIS usa `primer_dia` (fuego hoy y no ayer): 7.301 de 28.135 celdas-día en los
veranos de evaluación. Contar el arrastre sería puntuar el mismo incendio siete
veces.

Autocorrelación espacial (Moran's I, vecindad reina):

| Campo | 1 km | Bloques 10 km |
|---|---|---|
| nº EGIF 2015–20 por celda | 0,28 | 0,63 |
| nº EFFIS 2021–24 por celda | **0,81** | 0,40 |
| FWI medio de verano | 0,99 | 0,95 |

**Y la fuga que provoca, medida:** con las mismas variables y la misma etiqueta,
partir al azar en vez de por bloques de 100 km **infla el AUC en +0,036** — y
hasta **+0,078** si el modelo es solo de clima. → **Todas las particiones del
trabajo son espaciales o temporales, nunca aleatorias.**

### 2.4 La cota de lo que puede dar un mapa estático

Un XGBoost sobre las 498.530 celdas con etiqueta «≥1 EGIF en 2015-18»,
validado *out-of-fold* por bloques de 100 km:

| Variables | OOF | 2020 | EFFIS 21–24 | Lift decil 2020 |
|---|---|---|---|---|
| Estáticas | 0,749 | 0,757 | 0,711 | ×4,1 |
| + clima de la celda | 0,799 | 0,805 | 0,732 | ×5,2 |
| + densidad EGIF 2008-14 | **0,843** | 0,843 | 0,778 | **×6,0** |

**Un mapa que no mira el día ya alcanza AUC 0,82–0,84**, y su decil superior
concentra el **54–60 %** de las igniciones del año. Cualquier modelo que no
contenga esa información parte perdiendo. → Es la línea base que justifica el
diseño del segundo ciclo.

### 2.5 El conjunto muestreado: qué separa de verdad

Sobre el *train* de la primera iteración (53.563 filas: 13.577 positivos /
39.986 negativos), medianas de positivo contra negativo:

| Variable | Positivo | Negativo |
|---|---|---|
| `fwi` | 12,41 | 1,77 |
| `fwi_pctl_local` | **84,33** | **47,00** |
| `rh_min` | 39,88 | 58,73 |
| `dias_sin_lluvia` | 9,00 | 2,00 |
| `precip_30d` | 16,56 | 36,17 |
| `popdens` | 9,67 | 9,67 |
| `dist_carreteras` | 0,63 | 0,63 |

Dos lecturas, y las dos importan:

1. **Las estáticas no separan nada.** Población y distancia a carreteras tienen
   la **misma mediana exacta** en positivos y negativos. No es un defecto de los
   datos: es aritmética del diseño. Con pseudo-ausencias tomadas de la **misma
   celda**, las variables de la celda son idénticas a los dos lados y **no
   pueden** discriminar. El modelo estima riesgo temporal condicionado a la
   localización; la componente espacial la aporta el muestreo, no las columnas.
   → Es la primera evidencia visible de lo que el capítulo 5 acabará
   diagnosticando.
2. **El percentil local separa mejor que el FWI absoluto**, que es la
   contribución propia del trabajo (cap. 3.2).

### 2.6 Deriva entre entrenar y servir

PSI con los *bins* del cubo, jul–sep 2024, 300 nodos:

| Variable servida | PSI | Media cubo → servido |
|---|---|---|
| `t2m_max`, `rh_min`, `viento_max` | **0,000** | Sin deriva: es el mismo dato |
| `precip_dia` cruda | **0,593** | 0,59 → 1,19 |
| `precip_dia` / 2,02 | 0,194 | 0,59 → 0,59 |
| `precip_30d` cruda | 0,329 | 17,6 → 35,8 |
| `precip_30d` / 2,02 | **0,006** | 17,6 → 17,7 |
| `fwi` (13 UTC) | 0,012 | 28,3 → 28,2 |
| `fwi` (proxy tmax/rh_min) | 0,237 | 28,3 → 34,8 |

Temperatura, humedad y viento no derivan. La precipitación **sí**, y se corrige
sirviéndola dividida por 2,02: es reproducir a propósito un valor probablemente
erróneo del cubo —el raro es el cubo, no ERA5-Land— porque el modelo aprendió
con ese número.

### 2.7 Qué se lleva el EDA al resto de la memoria

- No entrenar solo con verano (§2.1) · Ninguna partición aleatoria (§2.3) ·
  Evaluar con `primer_dia` (§2.3) · Métricas ponderadas por superficie (§2.1) ·
  Servir la precipitación escalada y el FWI a 13 UTC (§2.6).
- Y sobre todo: **la prevalencia de diseño está cinco órdenes de magnitud por
  encima de la real, y las variables de la celda no separan nada.** Los dos
  hechos estaban medidos antes de entrenar el primer modelo. El capítulo 5
  cuenta lo que costó leerlos.

### Figuras

| Figura | Qué enseña |
|---|---|
| `eda_estacionalidad.png` | El ciclo anual del fuego |
| `eda_distribuciones.png` | Cada variable, positivos contra negativos |
| `eda_correlaciones.png` | Correlación entre variables |
| `eda_normalizacion_local.png` | Por qué el FWI hay que normalizarlo por sitio |

---

## 3. Extracción e ingeniería de variables  [~1 página]

Las **46 variables** del modelo de producción, agrupadas por familia. (El
conjunto completo tiene 50; producción excluye cuatro, §3.4.)

### 3.1 Las familias

| Familia | n | Ejemplos |
|---|---|---|
| Meteorología y peligro | 21 | `fwi`, `t2m_max`, `rh_min`, `viento_max`, `vpd_max`, las ventanas |
| Vegetación / combustible | 4 | `ndvi`, `ndvi_med_30d`, `lai`, `swi010` |
| Estáticas de la celda | 12 | `elevacion`, `pendiente`, `dist_carreteras`, `popdens`, 6 clases CORINE |
| Historial de fuego | 5 | `n_fuegos_10km_mismomes_hist`, `n_fuegos_10km_90d`… |
| Fuego activo y rayos | 4 | `frp_max_50km_7d`, `n_detec_50km_7d`, `rayos_dia`, `rayos_7d` |
| Calendario | 4 | `mes`, `dia_anio`, `es_festivo`, `ccaa` |

### 3.2 Ingeniería propia

**Las ventanas móviles son 7, 15 y 30 días** *(no 7/14/30)*:

| Ventana | Variables |
|---|---|
| 7 días | `precip_7d`, `fwi_med_7d`, `fwi_max_7d`, `rh_min_med_7d`, `t2m_max_med_7d`, `viento_max_med_7d` |
| 15 días | `precip_15d`, `fwi_med_15d` |
| 30 días | `precip_30d`, `fwi_med_30d`, `ndvi_med_30d` |

Capturan la sequedad acumulada, no solo el día concreto: el combustible tiene
memoria. Y **todas son [D−w, D−1], excluyendo explícitamente el día D** — regla
anti-fuga aplicada sin excepción, incluidas las ventanas de 90 y 365 días del
historial.

**Derivadas puntuales:**

| Variable | Fórmula | Sentido físico |
|---|---|---|
| `vpd_max` | Magnus: `0,6108·exp(17,27·T/(T+237,3))·(1−HR/100)`, con T = `t2m_max`, HR = `rh_min` | Demanda evaporativa del aire: cuanto mayor, más seca el combustible |
| `dias_sin_lluvia` | Racha de días consecutivos con `precip_dia` < 1 mm hacia atrás desde D (tope 120) | Sequía reciente acumulada |

**La normalización local del peligro — la aportación propia.** Un FWI de 15 es
«moderado» en la escala EFFIS, pero en Galicia (FWI mediano 3–14) es un día
excepcionalmente seco, mientras que en el Mediterráneo (mediano 38–41) es un día
normal de mayo. Aplicar el mismo umbral a toda España **penaliza sistemáticamente
a las zonas húmedas**, que son justo donde más arde (§2.1: Galicia y Asturias
suman el 34 % de las igniciones). Lo relevante no es «¿supera un umbral fijo?»
sino «¿es un día raro **para esta zona**?».

Implementación: climatología del FWI por (celda, mes) calculada **solo con
2008–2014** —estrictamente anterior al conjunto 2015–2020, sin fuga— de la que
salen `fwi_pctl_local` (percentil 0–100) y `fwi_anom_sigma` (anomalía
estandarizada).

**Evidencia empírica propia:** las comunidades atlánticas arden con FWI absoluto
mediano de 3–14 y las mediterráneas con 38–41, pero el **percentil local converge
a 73–92 en todas** cuando el día es realmente anómalo para su zona.

### 3.3 Cuánto aporta cada capa

Escalera de líneas base, test 2020 (prevalencia 26,28 %):

| Modelo | AUC-PR | AUC-ROC | n vars |
|---|---|---|---|
| A — FWI absoluto como *score*, sin ML | 0,488 | 0,729 | 1 |
| B — XGBoost solo-FWI | 0,491 | 0,736 | 1 |
| C — XGBoost meteo + FWI | 0,716 | 0,882 | 21 |
| **D — XGBoost completo** | **0,843** | **0,931** | 50 |

**El salto grande no es del FWI al *machine learning*** (B apenas mejora sobre
A): es de meteorología puntual a meteorología con ventanas (+0,23), y de ahí a
incorporar historial, estáticas y calendario (+0,13). Cada peldaño de
complejidad queda justificado con su medida.

Ablaciones de diseño que conviene reportar por honestidad:

| Ablación | AUC-PR | Lectura |
|---|---|---|
| `solo_calendario` (4 vars) | 0,476 | **La estacionalidad pura ya rinde como el FWI solo**: parte del resultado es «cuándo y dónde», no física de combustible |
| `calendario + fwi_local` (6 vars) | 0,749 | El 89 % del AUC-PR completo con seis columnas |
| `M_abs` — meteo + FWI absoluto | 0,668 | |
| `M_pctl` — meteo + FWI local | 0,650 | |
| **`M_ambos`** — los dos | **0,710** | **+0,042 sobre el absoluto**: la normalización local es real y *complementaria*, no sustitutiva |

### 3.4 Qué variables pesan, y una que hubo que quitar

Importancia SHAP (top 8, modelo completo): `n_fuegos_10km_mismomes_hist` 0,612 ·
**`fwi_anom_sigma` 0,516** · `n_fuegos_10km_90d` 0,418 · `n_fuegos_1km_hist`
0,385 · `rh_min` 0,332 · `dia_anio` 0,289 · `fwi` 0,278 · `lst` 0,269.

Entre las variables meteorológicas puras, **`fwi_anom_sigma` (la normalización
local) supera al FWI absoluto**: evidencia directa, con SHAP, de que la
contribución propia no es solo razonable, sino la que el modelo prioriza.

**Cuatro variables autorregresivas se eliminan de producción.** Con
pseudo-ausencias de celda fija, el incendio del positivo entra en el historial de
los negativos posteriores de esa misma celda, así que el modelo puede aprender
«orden temporal» en vez de física. `n_fuegos_10km_mismomes_hist` se libra —solo
cuenta años estrictamente anteriores— y es la única que se conserva. Coste
medido: **AUC-PR 0,843 → 0,828**. Se paga a sabiendas.

En el segundo ciclo, ya con negativos del mismo día, el reparto por *gain* es:
`n_fuegos_10km_mismomes_hist` **29 %** (domina: el fuego es recurrente donde ya
hubo fuego), CORINE ~10 %, pendiente 5,8 %, y NDVI/LST/LAI/SWI del día 5,5 %.

---

## 4. Entrenamiento  [~1–1,5 páginas]

### 4.1 Formulación
Clasificación binaria supervisada sobre la unidad muestral **(celda 1 km × día)**:
probabilidad de que ese día se **inicie** un incendio en esa celda. Se predice
la **ignición**, no la propagación ni la severidad.

### 4.2 Construcción del conjunto (primer ciclo)
- Positivos: EGIF 2015–2020.
- **Pseudo-ausencias**: no existe un registro de «días sin incendio», hay que
  construirlos. Ratio **1:3**, dentro del rango 1:1–1:10 respaldado por la
  literatura (Barbet-Massin *et al.*, 2012), tomando la **misma celda** en otras
  fechas.
- Consecuencia (se cobra en el cap. 6): con la celda emparejada, el modelo
  aprende sobre todo el **cuándo**, no el **dónde**.

### 4.3 Particiones
Train / validación / test **por años**, no aleatorias, para no filtrar
información temporal: `[PENDIENTE fijar los años del primer ciclo]`.

### 4.4 Algoritmo e hiperparámetros

**Algoritmo.** XGBoost (*gradient boosting* sobre árboles), con
`tree_method="hist"`, `enable_categorical=True`, `eval_metric="aucpr"`,
`random_state=42`. Justificación: datos tabulares, variables heterogéneas en
escala y naturaleza, no linealidades e interacciones, robusto a la falta de
escalado e importancia interpretable. *(Fuente: `entrenar_modelo.py:61-65`.)*

**Preprocesado que NO se hace, y por qué.** Sin escalado, sin winsorización, sin
imputación y **sin SMOTE**. Y de forma deliberada **sin `scale_pos_weight`**: la
salida se quiere como probabilidad, así que se conserva la prevalencia real del
diseño y se corrige después con **calibración isotónica ajustada en validación**
(van den Goorbergh *et al.*, 2022). Reponderar la clase positiva habría
mejorado el aspecto de las métricas de umbral a costa de destruir la
calibración.

#### Configuración base — procedencia: valores conservadores de referencia

| Hiperparámetro | Valor | Procedencia |
|---|---|---|
| `n_estimators` | 3000 (tope) | No se fija: lo decide el *early stopping* |
| `early_stopping_rounds` | 100 | Paciencia sobre AUC-PR en validación |
| `learning_rate` | 0,05 | Valor conservador de referencia |
| `max_depth` | 6 | Por defecto de XGBoost; profundidad moderada |
| `min_child_weight` | 5 | Por encima del defecto (1): frena el sobreajuste |
| `subsample` | 0,9 | Rango habitual 0,8–1,0 |
| `colsample_bytree` | 0,8 | Rango habitual 0,8–1,0 |
| `reg_lambda` | 1,0 | Por defecto de XGBoost |

**El número de árboles no es un hiperparámetro elegido a mano**: se fija un tope
de 3000 y para el *early stopping* con paciencia de 100 rondas sobre el AUC-PR
de validación. Los árboles que resultan: **v1 → 489**, **v2 (producción) → 240**,
**v3 → 590** (variante ligera 212), **v4 → 732**.
`[VERIFICAR]` el `.ubj` de v4 contiene 832 árboles frente a los 732 que declara
su metadata: comprobar cuál es el servido antes de publicar la cifra.

#### Búsqueda propia — Optuna, y por qué el presupuesto fue corto

`tuning_optuna.py`: optimización bayesiana (TPE, semilla 42), **40 *trials***,
objetivo **AUC-PR sobre la validación de 2019**; el test de 2020 no se toca.

| Hiperparámetro | Espacio explorado | Óptimo hallado |
|---|---|---|
| `learning_rate` | [0,01 – 0,15] log | 0,0170 |
| `max_depth` | [4 – 10] | 10 |
| `min_child_weight` | [1 – 30] | 3 |
| `subsample` | [0,6 – 1,0] | 0,892 |
| `colsample_bytree` | [0,5 – 1,0] | 0,503 |
| `reg_lambda` | [0,1 – 20] log | 0,716 |
| `gamma` | [0,0 – 5,0] | 0,0106 |

**El resultado es la parte interesante:** el mejor de los 40 *trials* da
AUC-PR **0,8666** en validación frente a **0,8647** de la configuración base —
una ganancia de **+0,0019**. El presupuesto fue corto **a propósito**, con la
hipótesis previa de que *el retorno está en las variables, no en el tuning*; el
experimento la confirma. Es un resultado que se reporta tal cual: en este
problema **el ajuste de hiperparámetros es irrelevante frente al diseño del
conjunto de entrenamiento**, que es justamente la tesis de los capítulos 5 y 6.

#### Cuándo se usa cada configuración, y un coste de despliegue

Se arrastran las dos configuraciones (*base* y *tuned*) y se elige en
validación, no en test (`entrenar_modelo_v3.py`, `entrenar_modelo_v4.py`).
La *tuned* gana **+0,0014** de AUC-PR en validación… y **multiplica por ~13 el
tamaño del fichero del modelo (11 MB frente a 840 KB)**, porque `max_depth` pasa
de 6 a 10. Ese fichero viaja dentro del repositorio que ejecuta el trabajo
diario en GitHub Actions, donde cada MB se paga en clonado y en cuota. Por eso
se produce además una **variante ligera** con los parámetros base y **se
documenta el intercambio para que la decisión de despliegue sea explícita**: una
milésima de AUC-PR no compensa trece veces el peso en una cadena que corre a
diario.

#### Particiones y disciplina de selección

| Versión | Train | Val | Test | Hiperparámetros | Árboles |
|---|---|---|---|---|---|
| v1 / v2 (producción) | 2015–2018 | 2019 | 2020 | base | 489 / 240 |
| v3 | 2015–2020 completo | — (mediana de CV) | — | tuned (+ *lite* base) | 590 / 212 |
| v4 | 2015–2020 | 2021 | 2022 | tuned | 732 |

- La selección de hiperparámetros, del número de árboles y del **umbral de
  alerta (por TSS)** se hace **siempre en validación**; el test se toca una vez.
- **v3 no tiene validación**: al reentrenar con 2015–2020 completo (todos los
  años disponibles) el número de árboles se fija por la **mediana de las
  iteraciones de una validación cruzada**, no con un conjunto reservado, y las
  métricas que lo amparan son las de un modelo gemelo entrenado bajo el
  protocolo congelado.
- **Validación cruzada espacial** adicional: `GroupKFold(5)` agrupando por
  **bloque de 100 km** sobre train+val, para medir generalización a **zonas no
  vistas** — el eje que la literatura señala como el más débil en modelos
  geoespaciales (Ploton *et al.*, 2020).

#### Segundo ciclo: los hiperparámetros se congelan a propósito

`dos_05_modelos.py:72-75` **hereda literalmente la configuración base** del
primer ciclo. Es una decisión de diseño experimental, no un descuido: al cambiar
la etiqueta (EGIF → EFFIS) y el muestreo (misma celda → mismo día), mantener los
hiperparámetros garantiza que **la diferencia medida sea atribuible al diseño
muestral y no a un tuning distinto**. Split: train 2015–2022 / val 2023 /
test 2024, con *early stopping* en 2023; `donde_dia_effis` paró en **441
árboles**.

El mismo principio se lleva al extremo en la ablación del ratio (cap. 6.3):
allí los **441 árboles se congelan para todos los peldaños**, porque con más
negativos cambia la prevalencia de la validación y el *early stopping* pararía
en otro punto — la comparación mediría entonces el ratio *y* el número de
árboles a la vez.

---

## 5. Evaluación del primer modelo, y el diagnóstico  [~1 página]

### 5.1 En test las métricas son buenas
- AUC-ROC **0,931** y AUC-PR 0,843 (v1/v2, test 2020); **0,891** [0,864, 0,914]
  en v4 sobre test 2022.
- **Control anti-fuga:** un modelo entrenado con las etiquetas barajadas al azar
  da AUC-PR 0,231 ≈ prevalencia base. No hay fuga de información.
- **Advertencia obligatoria:** ese AUC-PR **no es comparable con la literatura**
  — corresponde a una prevalencia de diseño del **26,28 %** frente a una
  prevalencia real del orden de 10⁻⁵ (cap. 2). Solo el AUC-ROC admite
  comparación aproximada.

### 5.2 En operación, el mismo modelo rinde 0,56–0,64
La distancia entre 0,89 y 0,64 es el verdadero objeto del trabajo. Se
persiguieron **tres explicaciones, en este orden, y las dos primeras se
descartaron con números** — esa es la razón de que la tercera sea creíble.

| # | Hipótesis | Veredicto |
|---|---|---|
| a | ¿Está mal medido? | **No.** Tres verdades independientes dicen lo mismo |
| b | ¿Es la meteorología? | **No.** Hay desplazamiento de fuente, pero no basta |
| c | ¿Es el desajuste train/serve? | **No.** Los *proxis* operativos cuestan 0,0045 |
| d | **Es la especificación** | La etiqueta y el muestreo definían otra pregunta |

**(a) ¿Está mal medido?** Cinco *scores* contra tres verdades-terreno
independientes (focos FIRMS, perímetros EFFIS, partes del MITECO) sobre 20 días
sellados: **ninguna diferencia es significativa**. Y corrigió un error propio: la
línea base dura no es el FWI crudo sino el **percentil local**, que al acumular
días subió de 0,540 a **0,702** mientras el modelo apenas se movía. La conclusión
anterior —«el modelo bate al FWI», con 2-3 días de muestra— era un artefacto del
tamaño muestral.

**(b) ¿Es la meteorología?** El bloque más grande del trabajo, 24 *scripts*,
porque era la hipótesis más plausible y hubo que agotarla. Firme: la
precipitación no coincide (+8,00 mm/30 días) y la climatología del percentil
cruzaba fuentes, saturando la variable (12,6 % de las filas en percentil ≥99,9
en producción frente al 1,0 % en entrenamiento). Ninguna de las dos explica el
hueco. *(Lo que no cerró también se dice: la comparación de AUC entre brazos
falló su control — ver `docs/LIMITACIONES.md`.)*

**(c) ¿Es el train/serve?** Auditoría de las 46 variables una a una y ablación de
los *proxis* de producción:

| Proxy operativo | Δ AUC test 2020 |
|---|---|
| Satélite → climatología mensual congelada | −0,004 |
| Rayos → 0 | −0,000 |
| **Los dos, como hoy en producción** | **−0,0045** |

No explica el hueco. De regalo: las cinco variables satelitales no aportan nada
**ni siendo reales**.

### 5.3 La bisagra: es la especificación
Dos medidas, y con ellas el trabajo cambia de dirección:

1. El **98,8 %** de las celdas del conjunto tenía exactamente un 25 % de
   positivos. El modelo **no podía** aprender qué celda es más peligrosa que
   otra: solo qué día lo es.
2. El **AUC caso-control de los dos diseños es idéntico (0,92)**, y el **AUC
   dentro del día los separa (0,744 frente a 0,828)**. **La métrica de
   desarrollo era ciega a la diferencia que importaba.**

No es un problema de ajuste. **La pregunta que se entrenó —«¿es hoy peligroso en
esta celda?»— no es la que se evalúa en operación —«¿cuál de las 498.530 celdas
arde hoy?».** Y la evidencia estaba a la vista desde el capítulo 2: la
prevalencia cinco órdenes de magnitud por encima de la real, y las variables de
la celda con mediana idéntica en positivos y negativos.

---

## 6. Segunda iteración: reentrenar cambiando la pregunta  [~1 página]

### 6.1 Dos cambios, y nada más

| | Iteración 1 | Iteración 2 |
|---|---|---|
| Negativos | La misma celda, otros días | **Otras celdas, el mismo día** |
| Etiqueta | Ignición EGIF | **Superficie quemada EFFIS** (`is_fire`) |

**Las 46 variables son las mismas a propósito**: si algo mejora, no puede
atribuirse a información nueva. **Los hiperparámetros también se heredan
literalmente**, por la misma razón — así la diferencia medida es del diseño
muestral y no de un ajuste distinto.

Positivos: primer día EFFIS, tope de 30 por día × bloque de 100 km para que un
episodio grande no domine. Split: train 2015–2022 / val 2023 / test 2024, con
**2026 como conjunto externo**; *early stopping* en 2023 → **441 árboles**.

### 6.2 La ablación del ratio de pseudo-ausencias
El 1:3 venía heredado de la literatura y nunca se había justificado con datos
propios. Se barre 1:3 → 1:10 → 1:30 → 1:60 → 1:100 sobre **los mismos positivos
y con los 441 árboles congelados** — con más negativos cambia la prevalencia de
la validación y el *early stopping* pararía en otro punto, de modo que la
comparación mediría el ratio *y* el número de árboles a la vez.

- Óptimo **interior**, en **1:10–1:30**; desde 1:60 empeora (1:100 pierde −0,013
  en 2026).
- **Pero la ganancia en AUC medio (+0,006 a +0,017) es del orden del ruido del
  sorteo de negativos** (0,005, medido re-sorteando la misma receta). No se vende
  como mejora.
- Lo que **sí** mejora de forma consistente es el fuego grande: **percentil
  ponderado por hectáreas 81,7 → 87,6–87,9**. Es lo que justifica servir el 1:10.

### 6.3 Dos ablaciones más, ambas negativas y las dos útiles
- **Entrenar solo con verano empeora el DÓNDE** — coherente con el 38 % de
  igniciones de febrero a abril del capítulo 2.
- **Añadir una capa de «ya quemado» cuesta −0,02**: la etiqueta EFFIS penaliza
  bajar el riesgo sobre la cicatriz reciente.

### 6.4 Los tres modelos que salen
- **único** (`donde_dia_effis`): 46 variables, muestreo del mismo día, etiqueta
  EFFIS.
- **dónde** (`donde_effis_c`): una fila por celda, sin meteorología — su mapa es
  el mismo todos los días.
- **cuándo**: solo las 29 dinámicas servibles.
- La **pareja** es el producto de los dos últimos.

---

## 7. Evaluación del segundo ciclo y producción  [~1,5–2 páginas — el capítulo con más peso]

### 7.1 En validación y test

AUC **dentro del día** sobre el banco `eval_dia` (celdas EFFIS de primer día
contra 1.000 celdas al azar del mismo día), con IC95 por *bootstrap* de días y
pareado contra producción:

| Modelo | Val 2023 (37 días, 223 pos.) | Test 2024 (72 días, 943 pos.) |
|---|---|---|
| Producción (`xgb_v2_prototipo`) | 0,723 | 0,792 |
| **`donde_dia_effis` (el único)** | **0,836** (Δ **+0,113** [+0,051, +0,176], gana 68 %) | **0,809** (Δ +0,017 [−0,024, +0,063], gana 38 %) |
| `cuando_effis` | 0,628 (Δ −0,094) | 0,687 (Δ −0,105) |
| `donde_effis_c` (por celda) | 0,786 | 0,746 |
| `fwi_pctl_local` (línea base sin ML) | 0,680 | 0,636 |

**Hay que leerlo con honestidad, y da un argumento mejor que esconderlo:** en
val 2023 el modelo único gana con claridad y su intervalo no toca el cero; en
test 2024 la ventaja se encoge a +0,017 y el intervalo sí lo cruza. 2024 fue un
año tranquilo —1,82 celdas-día por 100.000 frente a las 14,16 de 2022 (§2.2)— y
72 días de verano con 943 positivos no bastan para separar dos modelos que se
llevan dos centésimas.

La conclusión metodológica es que **el test de 2024 no tiene potencia
suficiente**, y por eso el trabajo no se cierra ahí: la prueba de verdad es la
temporada 2026 completa, externa y sellada (§7.2), y la validación prospectiva
día a día (§7.3). Un modelo de riesgo se juzga en la temporada, no en un
conjunto reservado.

### 7.2 En 2026 (conjunto externo)
- Único `donde_dia_effis`: **AUC medio 0,751** frente a **0,644** de producción;
  Δ **+0,106**, IC [+0,067, +0,148]; gana el **70 %** de los días.
- Variante **1:10** (`donde_dia_effis_r10`), la mejor: **0,758**, Δ **+0,114**
  [+0,074, +0,154], 73 % de los días y el mejor percentil ponderado por
  hectáreas (81,8).
- Modelo de pareja (dónde/cuándo separados): 0,702.
- `[TAB 7.1]` comparativa de los cuatro modelos.
- `[FIG 7.1]` distribución diaria de Δ AUC con su intervalo.

### 7.3 Validación prospectiva: tres jueces independientes
Sellado el modelo, se puntúa **día a día, hacia delante**, contra tres etiquetas
que no se usaron en entrenamiento:
- **EFFIS** (perímetros, 45 días de latencia),
- **MITECO** (parte diario D+1),
- **por estación meteorológica**, contra el ranking sellado del sistema en
  producción.

`[TAB 7.2]` resultados acumulados por juez. `[PENDIENTE actualizar a la fecha de
cierre — veredicto esperado a mediados de septiembre]`.

### 7.4 Comparación justa: previsión contra previsión
El experimento sin retrovisor para nadie (19 días, 22-jul → 15-ago):
ranking sellado 0,568, **único 0,605** (Δ +0,036 [−0,020, +0,085], gana 14/19
días), malla 0,547, pareja 0,548. Por celda: único 0,755 vs. 0,669
(Δ +0,086 [+0,022, +0,156]).
Es la primera vez que el modelo nuevo bate a producción **por estación**, la
geometría donde siempre perdía.

### 7.5 Limitaciones
- La probabilidad **ordena, pero no es una probabilidad real**: está referida a
  la prevalencia de diseño.
- Ninguna de las etiquetas de validación coincide con la de entrenamiento.
- Intervalos anchos en la validación prospectiva: resultado indicativo.
- **Desajuste train/serve** conocido: al servir, cuatro variables de vegetación
  del día se sustituyen por climatología mensual; pesan un 5,5 % del *gain*.
- Días de fallo documentados (3, 8 y 12 de julio de 2026): incendios sin señal
  meteorológica previa — el límite de un modelo alimentado por meteorología
  cuando la causa es humana e inmediata.

---

## 8. El repositorio  [~0,5 página]

`github.com/JuanUtrilla/tfm-riesgo-incendios` — acompaña a la memoria.
- Árbol espejo de los capítulos (`00_marco` … `07_produccion` + `docs` +
  `muestras`).
- **Trazabilidad número → script**: cada cifra de la memoria remite al script que
  la produce (`docs/PROCEDENCIA.md`).
- `muestras/` con datos reducidos para que la rama de servicio corra sin el cubo.
- Contiene **más gráficas y más análisis de los que caben en la memoria**: el EDA
  completo, las ablaciones enteras y los mapas diarios.
- `[PENDIENTE]` `environment.yml`, demo ejecutable, CI con badge, licencia y
  autoría (ver `docs/PENDIENTE.md`).

---

## 9. Conclusión  [~0,5 página]

La lección del trabajo no es qué modelo gana, sino que **una target mal alineada
con la pregunta de producción produce métricas excelentes y un sistema inútil**.
El primer ciclo era metodológicamente correcto y aun así no servía; lo que lo
arregló no fue un algoritmo mejor ni más hiperparámetros, sino **volver a
formular la unidad muestral**.

---

## Presupuesto total

| Capítulo | Páginas | Estado |
|---|---|---|
| 1. Extracción de datos | 1,5–2 | **Escrito** |
| 2. EDA | 1,5 | **Escrito** |
| 3. Features | 1 | **Escrito** |
| 4. Entrenamiento | 1–1,5 | **Escrito** |
| 5. Evaluación del primer modelo | 1 | Esqueleto |
| 6. Reentrenamiento | 1,5 | Esqueleto |
| 7. Evaluación y producción | 1,5–2 | 7.1 escrito; resto esqueleto |
| 8. Repositorio | 0,5 | Esqueleto |
| 9. Conclusión | 0,5 | Esqueleto |
| **Total** | **≈ 10–11,5** | |
