# Modelo de riesgo diario de incendio forestal: datos, entrenamiento y puesta en producción

> Capítulo de aprendizaje automático (6-7 páginas), autocontenido. Borrador del 21/08/2026.
> Cada cifra sale de un JSON o CSV de `TFM_fuego_malla/salida/` producido por los scripts
> `dos_00` … `dos_16`; el pipeline completo se reproduce en ~15 minutos desde el cubo IberFire.

## 1. Objetivo

Producir cada mañana un mapa de España peninsular a 1 km que ordene las celdas por su riesgo de
quemarse ese día y el siguiente. La pregunta operativa es **"¿cuál de estas 498.530 celdas arde
hoy?"**; el modelo se juzga por cómo ordena las celdas **dentro de cada día**, no por su
probabilidad absoluta. El capítulo recorre los datos, las variables, el análisis exploratorio, la
ingeniería de datos, el modelo, su entrenamiento y evaluación, y el pipeline que lo sirve a diario
con fuentes distintas de las de entrenamiento.

## 2. Datos

| fuente | qué aporta | resolución / periodo | papel |
|---|---|---|---|
| **IberFire** (datacubo, Ercibengoa 2025) | meteorología diaria (ERA5-Land reescalado: tmax, tmin, HR, viento, precipitación, FWI), vegetación (NDVI, LAI, SWI, LST), 232 capas estáticas (orografía, usos del suelo CLC, distancias a vías y ríos, población) y la etiqueta `is_fire` (celda quemada EFFIS ≥5 ha) | 1 km, diario, 2007-2024 | **entrenamiento**: variables y etiqueta |
| **EGIF** (estadística oficial, vía Civio) | igniciones ≥1 ha con punto, fecha y superficie | 2015-2020 útiles (2021+ incompleto: 226 incendios en 2022) | etiqueta del primer modelo y del componente "cuándo"; densidad histórica |
| **FIRMS** (VIIRS) | detecciones térmicas con potencia radiativa | 375 m, diario, 2015- y tiempo real | variable de entorno [D−5, D−1]; **nunca** juez (sería circular) |
| **WGLC** | densidad de rayos | 0,5°, diario, hasta 2023 | variable; en producción no existe en tiempo real (se sirve a cero) |
| **ERA5-Land** (CDS) e **IFS** (Open-Meteo) | reanálisis horario y previsión diaria | 0,1° / 0,25°, 5.605 nodos peninsulares | **producción**: la meteorología del día |
| **EFFIS** (WFS) y **MITECO** (partes) | perímetros quemados; incidentes con medios del Estado | temporada 2026 | jueces en operación |

La decisión que organiza todo el trabajo: **la etiqueta de entrenamiento es la misma que la de
validación**, superficie quemada EFFIS, y no la ignición EGIF. El EGIF es mejor etiqueta de
ignición, pero se consolida con 2-4 años de retraso y nunca podrá juzgar un sistema en operación;
el primer modelo del trabajo (entrenado con EGIF) daba 0,886 de AUC en su test y 0,56-0,64 servido,
en parte porque medía otra cosa: solo el 15 % de los incendios EGIF de 25-100 ha coincide celda a
celda con un perímetro EFFIS ese día.

## 3. Variables

Se usan **46 variables**, las mismas del sistema anterior, para que la comparación con él sea del
diseño y no de las variables:

| bloque | n | variables | en producción |
|---|---|---|---|
| meteorología del día | 9 | `fwi`, `fwi_pctl_local`, `fwi_anom_sigma`, `t2m_max`, `t2m_min`, `rh_min`, `viento_max`, `precip_dia`, `vpd_max` | ERA5-Land (D−7) + IFS (D−6→D+1) |
| ventanas [D−w, D−1] | 11 | `precip_7/15/30d`, `fwi_med_7/15/30d`, `fwi_max_7d`, `rh_min_med_7d`, `t2m_max_med_7d`, `viento_max_med_7d`, `dias_sin_lluvia` | ídem, recursivo |
| vegetación | 5 | `ndvi`, `ndvi_med_30d`, `lai`, `swi010`, `lst` | climatología mensual 2020-24 (no hay satélite a tiempo) |
| estáticas | 12 | `elevacion`, `pendiente`, `rugosidad`, `dist_carreteras`, `dist_rios`, `popdens`, `clc_bosque/matorral/agricola/artificial/abierto/agric_hetero` | capas del cubo (CLC 2018, población 2020) |
| historial y entorno | 3 | `n_fuegos_10km_mismomes_hist` (EGIF ≥2008, mismo mes, años anteriores), `frp_max_50km_7d`, `n_detec_50km_7d` (FIRMS) | EGIF congelado; FIRMS en tiempo real |
| rayos | 2 | `rayos_dia`, `rayos_7d` | a cero (como el 10 % de NaN del entrenamiento) |
| calendario | 4 | `mes`, `dia_anio`, `es_festivo`, `ccaa` | trivial |

**Qué se descartó y por qué.** Cuatro autorregresivas de corto plazo del sistema anterior
(`n_fuegos_1km_90d`, `n_fuegos_10km_90d`, `n_fuegos_10km_365d`, `n_fuegos_1km_hist`) quedan fuera
por dos razones independientes: introducían fuga (con el muestreo original, el incendio de un
positivo entraba en el historial de sus propios negativos) y **no se pueden servir** (el EGIF llega
con años). La regla es estricta: una variable que no existe en tiempo real no entra en el
entrenamiento, y una que en producción se sirve constante (vegetación, rayos) se deja solo si su
ablación cuesta poco (−0,004 de AUC medido). La precipitación del cubo es la mitad de la de
ERA5-Land (ratio 2,02, medido), así que en producción se sirve escalada.

**Variables derivadas (ingeniería de características).** `vpd_max` (déficit de presión de vapor,
Magnus sobre `t2m_max` y `rh_min`); `fwi_pctl_local` y `fwi_anom_sigma`, la contribución del
trabajo: el FWI del día frente a la climatología 2008-2014 **del mismo mes en la misma celda**
(percentil y anomalía en desviaciones). Un FWI de 25 es extremo en Asturias y rutinario en Murcia;
el percentil local lo hace comparable. La climatología es previa al periodo de entrenamiento, así
que no hay fuga. Las ventanas excluyen el día D.

## 4. Análisis exploratorio

**Prevalencia.** El problema es extremadamente desbalanceado: en un verano normal arden 2-6 celdas
por cada 100.000 celdas-día (2022, el peor año: 38). Solo el 2,97 % de las celdas tuvo alguna
ignición EGIF en 2015-2020 y el 2,68 % alguna celda quemada EFFIS en 2021-2024. La etiqueta
`is_fire` persiste (51 % de probabilidad de seguir quemando al día siguiente): marca días ardiendo,
no igniciones, por lo que el positivo se define en su **primer día**.

**Estacionalidad y geografía.** Las igniciones EGIF tienen dos picos —febrero-abril (38 %, Cantábrico
y Galicia, quemas) y junio-septiembre (38 %)—; la superficie quemada EFFIS se concentra en julio-
agosto y, en 2026, en el centro-este peninsular (Ávila, Guadalajara, Zaragoza, Huelva, Huesca).
Las dos etiquetas no viven en el mismo sitio, y ese fue el hallazgo que obligó a cambiar la del
modelo de susceptibilidad (§7).

**Separación de clases** (medianas, positivos frente a negativos del mismo día en otra celda):

| variable | quemada | no quemada | lectura |
|---|---|---|---|
| `fwi` (absoluto) | 18,1 | 22,4 | **al revés**: las celdas que arden tienen FWI absoluto menor |
| `fwi_pctl_local` | 90,8 | 78,3 | el percentil local sí separa: arde lo anómalo para su sitio |
| `pendiente` (°) | 15,1 | 5,8 | relieve |
| `clc_matorral` / `clc_agricola` | 0,54 / 0,03 | 0,07 / 0,49 | combustible continuo frente a cultivo |
| `ndvi` | 0,50 | 0,39 | vegetación |
| `n_fuegos_10km_mismomes_hist` | 12 | 0 | persistencia del dónde |
| `n_detec_50km_7d` (FIRMS) | 8 | 1 | fuego activo cerca |
| `precip_30d` (mm) | 16,1 | 9,8 | contraintuitivo: llueve más donde hay combustible |

La primera fila resume el argumento de todo el trabajo: sin normalización local, la variable
canónica de peligro **ordena mal** entre regiones.

**Autocorrelación y partición.** El fuego está muy autocorrelacionado en el espacio (Moran's I de
las celdas quemadas 2021-24: 0,81 a 1 km) y poco en el tiempo a una semana (anomalía de tmax: 0,71 a
un día, −0,15 a siete). Partir al azar filtra información del test al train: el mismo modelo
estático evaluado con validación aleatoria infla el AUC +0,04-0,08 frente a la validación por
bloques de 100 km. Por eso toda partición aquí es **temporal** y la métrica se calcula **dentro de
cada día**, lo que es automáticamente un control espacial. La densidad histórica de fuego a 10 km
predice por sí sola "arde en 2019" con AUC 0,84: el dónde es muy persistente, y es la cota natural
de cualquier mapa estático.

**Colinealidad.** Hay 51 pares de variables con |ρ de Spearman| ≥ 0,8: `pendiente`/`rugosidad`
(1,00), `mes`/`dia_anio` (0,99), `fwi_pctl_local`/`fwi_anom_sigma` (0,98), las ventanas del FWI
entre sí (0,94-0,98), `t2m_max`/`vpd_max`/`lst` (0,89-0,96), `frp_max`/`n_detec` (0,92). **No se
eliminan**: el modelo es de árboles, que no sufren la colinealidad como los lineales, y cada par
aporta una forma distinta (media frente a máximo de la ventana, nivel frente a anomalía). El precio
es que la importancia se reparte entre variables redundantes y hay que leerla por bloques, no por
columnas. Lo que sí se eliminó fueron las variables con fuga o no servibles (arriba). Los NaN se
dejan como NaN (≤1,2 % en meteorología, 10 % en rayos) porque XGBoost los trata de forma nativa y
producción garantiza no producir NaN que no hubiera en entrenamiento.

## 5. Ingeniería de datos: cómo se construye el conjunto de entrenamiento

1. **Unidad.** Celda de 1 km × día.
2. **Positivos.** Celdas con `is_fire`=1 en su primer día (19.552 en 2015-2024), con un tope de 30
   por día y bloque de 100 km para que un megaincendio de 400 celdas no domine el conjunto.
3. **Negativos (1:3): mismo día, otra celda de España al azar.** Es la pregunta operativa. Se
   sortean 4 y se descartan los que caen a <12,5 km y ±10 días de un fuego (`is_near_fire` del
   cubo) —0,45 % de ambigüedad— y se recorta a 3. Se midieron también el diseño del sistema
   anterior (misma celda, otro día: aprende "cuándo") y un mixto (mitad y mitad): §7.
4. **Partición temporal congelada antes de extraer nada:** entrenamiento 2015-2022 (incluye a
   propósito el año récord), validación 2023 (*early stopping*), prueba **verano 2024** y, como
   prueba externa, la temporada 2026 reconstruida y la operación real.
5. **Extracción** con el mismo código que el sistema anterior (78.208 filas; 100 % idénticas en
   las filas comunes), por bloques del cubo en paralelo (4 min).
6. **Banco de evaluación aparte (`eval_dia`):** para cada día de verano de 2023 y 2024, 1.000
   celdas al azar de España más todas las celdas EFFIS de primer día (hasta 60). Ningún modelo lo ve.

## 6. Modelo y entrenamiento

**XGBoost (clasificación binaria)**, por cuatro razones medidas en este problema: datos tabulares
heterogéneos con interacciones no lineales (el FWI importa según el combustible); NaN nativos y
sin escalado; tolerancia a la colinealidad; e interpretabilidad por ganancia. Se descartaron
modelos lineales (la primera fila de la tabla de EDA ya los descalifica sin la normalización
local) y redes neuronales (78k filas, sin beneficio esperable y con coste de servicio). Dentro del
mismo marco se compararon **cuatro arquitecturas**: (a) un modelo único con las 46 variables,
(b) el modelo del sistema anterior reentrenado (control), (c) **dos modelos** —susceptibilidad por
celda con 26 estáticas + clima + coordenadas, y peligro diario con 29 variables dinámicas— combinados
por producto de probabilidades, y (d) el modelo único con muestreo mixto.

Hiperparámetros heredados del sistema anterior (profundidad 6, tasa 0,05, submuestreo 0,9,
`colsample` 0,8, `min_child_weight` 5), 3.000 árboles como máximo con parada a 100 rondas sin mejora
de AUC-PR en validación; el modelo final paró en **541 árboles**. Sin `scale_pos_weight` ni SMOTE:
la prevalencia del diseño (25 %) es artificial y la salida se usa como **orden**, no como
probabilidad. **Qué se priorizó:** la discriminación dentro del día y la robustez entre fuentes
(mismas variables en entrenamiento y servicio); **qué no:** la calibración absoluta —que se
resuelve aguas abajo con niveles por percentil— y el ajuste fino de hiperparámetros, que en el
sistema anterior (Optuna, 0,865 → 0,867 de AUC-PR) no movió nada.

**Métricas.** En el test del propio diseño (verano 2024, prevalencia 25 %) el modelo da AUC-ROC
0,916 y AUC-PR 0,820; son los números "de laboratorio" y **engañan**: el diseño anterior daba 0,92
en el suyo y 0,74 en operación. La métrica honesta, la única que se reporta como resultado, es el
**AUC dentro de cada día** (celdas que arden contra celdas al azar del mismo día), la media sobre
días, el *lift* del decil superior (fracción de lo quemado que cae en el 10 % de celdas mejor
puntuadas ÷ 0,1) y el intervalo por bootstrap de **días** (las celdas de un incendio no son
independientes: por celdas el intervalo sale cuatro veces más estrecho). La línea base obligatoria
es el modelo de producción con las mismas variables y el mismo protocolo; las de referencia, el
percentil del FWI y la densidad histórica.

## 7. Resultados

**Tabla. AUC dentro del día.** Test con etiqueta EGIF (2020) para los modelos EGIF; test 2024 para
los EFFIS; temporada 2026 (74 días con fuego EFFIS, 1-jun→15-ago, 13.567 celdas quemadas, mismas
variables de reanálisis para todos); juez MITECO (14 días, 49 incidentes, radio 10 km).
Δ = diferencia con producción, IC95 por bootstrap de días.

| modelo | etiqueta · muestreo | test | 2026 | 2026 megaincendios (11 d) | 2026 resto (63 d) | MITECO |
|---|---|---|---|---|---|---|
| producción `xgb_v2` | EGIF · cuándo | 0,744 | 0,737 | 0,883 | 0,711 | 0,693 |
| producción **sin FIRMS** | | — | 0,611 | 0,689 | 0,597 | **0,520** |
| único, muestreo dónde | EGIF · dónde | 0,828 | 0,733 | 0,728 | 0,734 | 0,698 |
| único, muestreo mixto | EGIF · mixto | 0,805 | 0,731 | — | — | — |
| dónde × cuándo | EGIF | 0,831 (+0,087 [+0,065, +0,108]) | 0,698 (−0,038) | 0,615 | 0,713 | 0,680 |
| dónde(EFFIS) × cuándo(EGIF) | híbrido | 0,840 | 0,754 (+0,017) | 0,830 | 0,740 | 0,696 |
| **único, muestreo dónde** | **EFFIS · dónde** | 0,809 | **0,773 (+0,037 [−0,004, +0,076])** | 0,794 | **0,770** | **0,731** |
| el mismo **sin FIRMS** | | — | **0,744** | 0,739 | 0,745 | 0,706 |
| percentil del FWI (sin modelo) | — | 0,627 | 0,653 | — | — | 0,629 |

**Lo que se descubrió, en orden.**

1. *El muestreo era el problema.* Con las mismas 46 variables, pasar de "misma celda, otro día" a
   "mismo día, otra celda" sube el AUC dentro del día de 0,744 a 0,828 en el test de 2020; el
   diseño mixto pierde frente a los dos puros (−0,026 [−0,040, −0,012]); el producto de dos modelos
   y el único empatan (±0,01); sumar logits no mejora el producto.
2. *La etiqueta también.* Con área quemada real (2026), el modelo de susceptibilidad entrenado con
   igniciones EGIF se hunde en los megaincendios (AUC 0,51; su decil superior contenía el 0-7 % de
   lo quemado): las igniciones se concentran en el noroeste y la superficie quemada de 2026 en el
   centro-este. Reentrenado con etiqueta EFFIS, el modelo único es el mejor de la temporada (0,773),
   el mejor en MITECO y el mejor de largo en los días normales (0,770 frente a 0,711).
3. *EFFIS ayuda al dónde y perjudica al cuándo.* El componente de peligro diario con etiqueta EFFIS
   empeora (0,661 frente a 0,697): sus positivos se concentran en pocos días extremos y aprende
   "qué día es extremo". Si se quiere la pareja, la buena es híbrida.
4. *La ventaja de producción es FIRMS.* Sin las detecciones de los cinco días previos cae a 0,611
   (EFFIS) y a 0,52 —aleatorio— (MITECO): su habilidad en megaincendios es persistencia del fuego
   activo, legítima en operación pero circular como anticipación. El modelo nuevo sin FIRMS
   iguala a producción con FIRMS (0,744 frente a 0,737).
5. *Dónde sigue ganando producción.* En la cola alta: con niveles por percentil del día
   (p30/p90/p98), el EXTREMO (2 % del territorio) de producción concentra el 28 % de lo quemado
   (lift ×14) y el del nuevo el 12 % (×6). El intervalo global aún roza el cero.

**Validación en condiciones reales (previsión contra previsión).** La tabla anterior usa
reanálisis para todos los modelos. Para los 19 días con fuego entre el 22-jul y el 15-ago de 2026
se reconstruyó lo que la malla habría servido de verdad —reanálisis hasta D−7 y la previsión IFS
*archivada tal como se emitió*, mapeada— y se comparó con el ranking que producción **publicó y
selló** esas mañanas (687 estaciones, estación positiva si hay perímetro EFFIS a ≤25 km):

| sistema | AUC por estación | Δ vs producción publicada | gana |
|---|---|---|---|
| producción publicada (AEMET + previsión municipal) | 0,568 | — | — |
| modelo de producción sobre la malla | 0,602 | +0,034 [−0,015, +0,082] | 13/19 |
| **único (etiqueta EFFIS, muestreo dónde)** | **0,625** | **+0,057 [−0,004, +0,108]** | **16/19** |
| pareja dónde(EFFIS) × cuándo | 0,595 | +0,027 [−0,030, +0,075] | 13/19 |

Por celda, contra los perímetros EFFIS de esos días, el único da 0,776 y el modelo de producción
sobre la malla 0,765 (empate). Es el número que más se parece a lo que se publicará: el candidato
gana 16 de 19 días con previsión real, con un intervalo que roza el cero.

**Importancia (ganancia) del modelo final, por bloques:** historial y FIRMS 34 % (la densidad
EGIF del mismo mes sola, 29 %), meteorología 26 % (`fwi_pctl_local` y `vpd_max` a la cabeza; el
FWI absoluto, residual), estáticas 25 % (`pendiente`, `clc_agricola`, `clc_matorral`), vegetación
7 %, calendario 6 %, rayos 2 %. Coherente con el EDA: el dónde lo ponen relieve, combustible e
historia; el cuándo, la anomalía meteorológica local.

Resultados negativos que se dejan escritos: entrenar solo con verano empeora el dónde (−0,03);
descargar 72 meses de ERA5-Land para entrenar era innecesario (la deriva cubo/ERA5-Land es
despreciable salvo la precipitación); la validación aleatoria infla +0,04-0,08.

## 8. Producción: el pipeline diario y por qué rinde menos que en entrenamiento

**El problema.** IberFire es un cubo cerrado (2007-2024) que no existe para hoy. Lo que hay cada
mañana es otra cosa: reanálisis ERA5-Land con 6 días de retraso, previsión IFS para los días
siguientes, detecciones FIRMS de anoche, y capas estáticas que no cambian. El pipeline tiene que
reconstruir las 46 variables a partir de eso, **con la misma receta** que en el cubo.

**La cadena** (GitHub Actions, 05:30 y 15:45 hora peninsular; el estado entre corridas en un
*release* del repositorio):

1. *Reanálisis* ERA5-Land horario del mes en curso desde el CDS (≈25 MB), agregado a diario con
   las reglas del cubo (máx/mín de temperatura, mínimo de HR por Magnus, máximo de viento medio
   horario, precipitación acumulada) y fundido con el acumulado desde el 1 de mayo.
2. *Previsión* IFS en los 5.605 nodos (Open-Meteo, 29 lotes con ritmo de 23 por hora: el tope es
   5.000 unidades/hora). El FWI se reanuda desde el estado del último día de reanálisis y **la rama
   de previsión se mapea por cuantiles** a la escala de ERA5-Land: cruda dispara la saturación del
   percentil de 0,23 % a 2,05 % (con tres días de previsión ya está el 83 % del daño: el FFMC
   responde en horas).
3. *46 variables por celda*: meteorología del nodo (cada celda de 1 km hereda su nodo de 9 km: sin
   interpolación, sin huecos, sin depender de que una estación reporte), `fwi_pctl_local` con
   numerador y climatología de la **misma fuente** (la climatología del cubo reconstruida por nodo,
   84 meses), estáticas desde 16 capas 2D exportadas del cubo (16 MB), vegetación como climatología
   mensual, EGIF del mismo mes congelado, FIRMS de la API, rayos a cero.
4. *Inferencia* del modelo único y de la pareja; publicación como **percentil del día** y niveles
   p30/p90/p98, con una alerta global aparte (FWI medio del día frente a su climatología).
5. *Tres jueces automáticos*: EFFIS (perímetros por fecha de inicio, ventana de 45 días que se
   reescribe por la latencia de 6-9 días), MITECO (parte del día previo a las 14 h, municipio,
   10/25 km) y el ranking sellado de producción por estación (687 estaciones, previsión real contra
   previsión real). Cada paso tiene tope de tiempo y reintentos; el estado no se pierde si un día
   falla.

**Por qué en producción se rinde menos que en entrenamiento** —y cuánto, medido:

- *Previsión en vez de reanálisis.* Los tres días con mapa sellado de producción muestran 0,80 /
  0,60 / 0,20 de AUC con previsión frente a 0,90 / 0,89 / 0,53 con reanálisis del mismo día: el
  "retrovisor" vale 0,1-0,3. Afecta a todos los modelos por igual.
- *Deriva de variables.* Medida con PSI (>0,25 = sospechosa): nula en temperatura, humedad y viento;
  el FWI calculado con el proxy `tmax`/`hr_min` da 0,24 frente a 0,012 con las 13 UTC; el percentil
  con numerador y climatología de recetas distintas, 0,34 frente a 0,004 con la misma; la
  precipitación cruda, 0,59, escalada 0,19. Lo que no se pudo medir en 2024 se sirve como se
  entrenó.
- *Variables congeladas.* Vegetación como climatología, población de 2020, EGIF a 2020, rayos a
  cero: −0,004 de AUC en ablación, pero ciegas a un año anómalo.
- *La etiqueta.* EFFIS ve lo grande y con días de retraso; MITECO ve lo grave y con error de
  municipio. Ninguna es la ignición que el modelo "vio" en el cubo, y por eso la métrica de
  laboratorio (0,92) no es comparable con la operativa (0,7-0,8).

**Criterio de sustitución.** El sistema anterior no se apaga: el candidato lo sustituye cuando el
IC95 de su diferencia acumulada en la temporada quede por encima de cero en EFFIS y no por debajo en
estaciones y MITECO, y cuando sus niveles tengan tabla de calibración empírica. Con ~25 días por
juez a mediados de septiembre de 2026 el intervalo debería cerrarse en un sentido u otro.

## 9. Conclusión

El modelo no estaba mal entrenado; estaba entrenado para otra pregunta y con otra etiqueta. Tres
cambios, cada uno medido —negativos del mismo día para aprender el **dónde**, la misma malla y la
misma fuente meteorológica en entrenamiento y servicio, y la misma etiqueta (superficie quemada)
en entrenamiento y validación— convierten las mismas 46 variables y el mismo XGBoost en un sistema
que pasa de 0,737 a 0,773 de AUC dentro del día en la temporada 2026 y que ya no depende de las
detecciones satelitales del día anterior. La ventaja es consistente en tres jueces y todavía no
significativa; la cadena está montada para que la propia temporada lo decida sin intervención, y
para que este capítulo pueda cerrar con el intervalo, sea el que sea.
