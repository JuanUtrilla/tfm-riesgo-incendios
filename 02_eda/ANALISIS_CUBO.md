<!-- Redactado el 25/08/2026.
     Es el documento de resultados de la fase 1 del encargo de los dos
     modelos: lo producen dos_00_cubo_etiquetas.py, dos_01_celdas.py y
     dos_04_analisis.py, que viven en 04_bisagra/44 y 05_iteracion2/. -->

# Análisis de datos — fase 1 del encargo de los dos modelos

Todo está medido la noche del 21/08/2026; los scripts que producen cada número son `dos_00` … `dos_06` de
este repo y los JSON de `salida/`. Los datos pesados viven en el disco de
expansión (`config_expansion.py`).

---

## 0. La decisión previa: entrenar con el cubo, servir ERA5-Land

El encargo pedía valorar si hacía falta bajar 72 meses de ERA5-Land (≈30 h de
cola CDS) o si la meteo del cubo IberFire servía para entrenar. **Sirve, y se
adopta.** Razones, todas medidas antes de esta noche:

- El cubo **es** ERA5-Land reprocesado: tmax, hr_min y viento en los nodos
  coinciden con corr 0.987-0.998 y sesgo 0 (`malla_02_vs_cubo.json`).
- El FWI del cubo se reproduce con las 13 UTC (`verificar_hora_fwi.py`).
- Lo que difiere es la precipitación (cubo = ERA5-Land / 2.02) y se mide en
  §5 qué hace eso a las features.

Se arrancó la descarga (`malla_06_descarga_historico.py`, conserva el horario
en expansión) y se paró a los dos minutos sin bajar nada. El script queda
documentado y reanudable por si algún día se reentrena con ERA5-Land nativo.

## 1. Etiquetas: EGIF contra EFFIS

| | EGIF (Civio) | EFFIS en el cubo (`is_fire`) | EFFIS 2026 (geojson) |
|---|---|---|---|
| qué es | ignición registrada, punto, ≥1 ha | celda quemada, ≥5 ha, diario | polígono de área quemada |
| años útiles | **2015-2020** | 2008-2024 (un solo día por fuego hasta 2014) | temporada 2026 |
| n en 2015-20 | 19,866 igniciones → 19,561 celda-día | 21,785 celdas-día · 9,942 celdas | — |
| tamaño | p50 3.6 ha · p90 33 · p99 343 · **41 % ≥5 ha** | por definición ≥5 ha | 1,895 polígonos · 307,850 ha · p50 8 ha · 62 % ≥5 ha |

**Se solapan poco a nivel celda-día** (`dos_03`, flag `is_fire_dia` de los
positivos EGIF): solo el 2.6 % de los EGIF de 5-25 ha, el 14.6 % de 25-100 ha,
el 34 % de 100-500 ha y el 59 % de >500 ha tienen `is_fire`=1 en su celda ese
día. A 12.5 km y ±10 días (`is_near_fire`) sube a 34/53/75/97 %. EFFIS, al
ser MODIS, es ciego a lo pequeño y sitúa el fuego en el perímetro, no en el
punto de ignición; el EGIF es la ignición pero se corta en 2020.

**Estacionalidad distinta.** EGIF 2015-20: marzo 15 %, febrero 10 %, abril
11 % — **el 38 % del año va de febrero a abril**, y jun-sep suma el 38 %.
EFFIS (celdas-día 2008-24): julio 18,784, agosto 15,481, y marzo 7,702. El
EFFIS 2026 lo confirma en extremo: 1,321 de sus 1,895 polígonos son de
feb-abr, en Cantabria (530), Asturias (410) y Navarra (230).

**Sesgo geográfico EGIF:** Galicia 18.6 %, Asturias 15.8 %, Castilla y León
15.4 %, Cantabria 13.4 %, Extremadura 10.9 %. Causa: intencionado 68 %,
negligencia 12 %, rayo 2.4 %.

**Decisión:** se entrena con EGIF 2015-2020 (es la ignición, que es lo que se
modela) y se evalúa con **las dos** verdades-terreno por separado, porque
ninguna contiene a la otra. EFFIS además permite una prueba externa en
2021-2024, fuera del EGIF, que resultó decisiva en 2022.

## 2. Prevalencia real

498,530 celdas peninsulares × día. Celdas-día con `is_fire` (EFFIS ≥5 ha),
por 100,000 (`dos_00`):

| año | anual | verano (jun-sep) | días con ≥1 celda | máx. celdas/día |
|---|---|---|---|---|
| 2015 | 1.56 | 4.24 | 90 | 194 |
| 2017 | 4.01 | 5.30 | 165 | 920 |
| 2018 | 0.40 | 0.74 | 90 | 60 |
| 2019 | 1.71 | 1.89 | 142 | 155 |
| 2020 | 3.08 | 6.40 | 190 | 361 |
| 2021 | 3.46 | 6.69 | 179 | 385 |
| **2022** | **14.16** | **37.96** | 223 | **1,678** |
| 2023 | 3.80 | 0.50 | 171 | 909 |
| 2024 | 1.82 | 2.40 | 190 | 112 |

Es decir, **2-6 por 100,000 en un verano normal** frente al 25 % del diseño
caso-control: cinco órdenes de magnitud. Con EGIF: 9 igniciones/día de verano
(mediana), p90 19, máx. 170; 286 días de 2,192 sin ninguna. Solo el **2.97 %**
de las celdas tuvo algún EGIF en 2015-20 y el 2.68 % algún EFFIS en 2021-24.

`is_fire` **persiste**: P(fuego en t+1 | fuego en t) = 0.51; t+3 0.23; t+7
0.04. Marca *días ardiendo*, no igniciones. Por eso en la evaluación EFFIS se
usa `primer_dia` (fuego hoy y no ayer: 7,301 de 28,135 celdas-día en los
veranos de evaluación).

## 3. Autocorrelación y partición

**Espacial** (`dos_04`, Moran's I, vecindad reina):

| campo | 1 km | bloques 10 km |
|---|---|---|
| nº EGIF 2015-20 por celda | 0.28 (binario 0.18) | 0.63 |
| nº EFFIS 2021-24 por celda | **0.81** | 0.40 |
| FWI medio de verano | 0.99 | 0.95 |

EFFIS a 1 km es casi todo perímetro (celdas contiguas del mismo fuego); EGIF
es puntual pero se agrupa a 10 km. **Fuga por partir al azar**, medida con el
modelo de susceptibilidad (mismas features, misma etiqueta):

| features | AUC OOF bloques 100 km | AUC OOF aleatorio | inflado |
|---|---|---|---|
| estáticas | 0.749 | 0.785 | +0.036 |
| estáticas + clima | 0.799 | 0.835 | +0.036 |
| solo clima | 0.734 | 0.812 | **+0.078** |
| solo lat/lon | 0.709 | 0.773 | +0.064 |

**Temporal**, anomalías respecto a la media móvil de 31 días, nodos ERA5-Land
2010-14: tmax lag-1 0.71, lag-3 0.21, lag-7 −0.15; hr_min 0.45 / 0.08; prec
0.26 / 0.01. A una semana la meteo ya no se parece; a un día sí.

**El DÓNDE es persistente entre periodos.** La densidad EGIF 2015-18 a 10 km
predice «arde en 2019» con AUC 0.840, «en 2020» 0.822 y «EFFIS 2021-24» 0.805.
El 98.6 % de las celdas que arden en 2019-20 están a <10 km de una que ardió en
2015-18 (pero solo el 26 % es la misma celda). La densidad de **2008-14**
predice 2015-18 con 0.821: la estructura no cambia en una década.

**Partición adoptada:** temporal (train 2015-18 / val 2019 / test 2020, la
del repo), y la evaluación se hace **dentro de cada día**, que es
automáticamente un control espacial: compara celdas distintas del mismo día,
no la misma celda en días distintos. El CV por `bloque_100km` se usa en el
modelo por celda (§4).

## 4. Susceptibilidad: la cota del DÓNDE

XGBoost sobre las 498,530 celdas, etiqueta «≥1 EGIF en 2015-18» (prevalencia
1.7 %), OOF por bloques de 100 km, evaluado en años posteriores (`dos_04`):

| features | OOF | 2019 | 2020 | EFFIS 21-24 | lift decil 2020 |
|---|---|---|---|---|---|
| estáticas | 0.749 | 0.776 | 0.757 | 0.711 | ×4.1 |
| + clima de la celda | 0.799 | 0.822 | 0.805 | 0.732 | ×5.2 |
| + lat/lon | 0.811 | 0.825 | 0.822 | 0.751 | ×5.4 |
| + densidad EGIF 2008-14 | 0.843 | 0.861 | 0.843 | 0.778 | ×6.0 |
| solo densidad 2008-14 | 0.828 | 0.843 | 0.818 | **0.808** | ×5.6 |

Lectura: **las estáticas + clima explican casi tanto como el historial**
(0.805 contra 0.818 en 2020), y sobre EFFIS 2021-24 el historial solo gana
porque EFFIS está dominado por grandes perímetros repetidos. Juntarlo todo
da 0.84, y el historial añade +0.02-0.04 sobre estáticas+clima+geo. Es la
cota de lo que puede dar un mapa estático: AUC ≈0.82-0.84 y **el 10 % de
celdas más susceptible concentra el 54-60 % de las igniciones del año**.

## 5. Deriva de features entre entrenar (cubo) y servir (ERA5-Land)

PSI con bins del cubo, jul-sep 2024, 300 nodos (`dos_06`, `salida/dos_06_deriva.json`):

| feature servida | PSI | media cubo → servido | p95 cubo → servido | corr |
|---|---|---|---|---|
| t2m_max | 0.000 | 28.2 → 28.1 | 37.5 → 37.5 | 0.999 |
| rh_min | 0.000 | 36.1 → 36.3 | 69.1 → 69.4 | 0.999 |
| viento_max | 0.000 | 3.67 → 3.68 | 5.86 → 5.88 | 0.994 |
| precip_dia cruda | **0.593** | 0.59 → 1.19 | 3.1 → 6.7 | 0.914 |
| precip_dia / 2.02 | 0.194 | 0.59 → 0.59 | 3.1 → 3.3 | 0.914 |
| **fwi (13 UTC)** | **0.012** | 28.3 → 28.2 | 57.8 → 58.7 | 0.957 |
| fwi (proxy tmax/hr_min) | 0.237 | 28.3 → 34.8 | 57.8 → **70.3** | 0.930 |
| fwi_pctl_local (proxy / clim proxy) | **0.004** | 45.8 → 44.4 | 93.1 → 92.6 | 0.849 |
| fwi_pctl_local (13 UTC / clim proxy) | **0.339** | 45.8 → 29.9 | 93.1 → 79.3 | 0.793 |
| precip_30d cruda | 0.329 | 17.6 → 35.8 | 56.6 → 114.9 | 0.979 |
| precip_30d / 2.02 | **0.006** | 17.6 → 17.7 | 56.6 → 56.9 | 0.979 |
| fwi_med_7d (13 UTC) | 0.013 | 28.2 → 28.0 | 54.4 → 55.2 | 0.979 |
| fwi_med_7d (proxy) | 0.278 | 28.2 → 34.6 | 54.4 → 67.6 | 0.965 |
| dias_sin_lluvia (sobre prec cruda) | 0.129 | 17.0 → 10.4 | 62 → 45 | 0.732 |

Conclusiones:

1. **Temperatura, humedad y viento no tienen deriva**: es el mismo dato.
2. **El FWI de nivel se sirve a las 13 UTC** y queda en PSI ≈0.03. El proxy
   tmax/hr_min (lo que sirve producción hoy) está en 0.24 y sube el p95 de
   58 a 70: inflaría las alertas.
3. **El percentil tiene que tener numerador y denominador de la misma
   receta.** Proxy/clim-proxy: PSI 0.01. Meter el FWI de las 13 UTC sobre la
   climatología del proxy (lo que pasaría si se cambia el numerador sin
   rehacer `malla_04`) da PSI 0.34 y hunde la mediana de 46 a 30:
   infraalertaría. Así que en producción: o se
   sirve `fwi` a 13 UTC **y** `fwi_pctl_local` con proxy/proxy (dos recetas,
   cada una consistente con su uso), o se rehace la climatología a 13 UTC
   (84 meses de CDS). La primera opción es gratis y se recomienda.
4. **La precipitación se sirve dividida por 2.02.** Cruda, `precip_30d` sale
   con PSI 0.33 y media doble; escalada, 0.006. La diaria escalada se queda
   en 0.19 porque ERA5-Land llueve *más días* (no solo más cantidad) y el
   factor no arregla la masa en cero. Es reproducir a propósito un
   valor probablemente erróneo del cubo (el raro es el cubo, no ERA5-Land),
   pero el modelo aprendió con ese número. Queda escrito y desaparecería si
   se reentrenase con ERA5-Land nativo.
5. `dias_sin_lluvia` (PSI 0.13, media 17 → 10 días) hereda esa mayor
   frecuencia de lluvia de ERA5-Land. Queda por medir si el umbral de 1 mm
   sobre la precipitación escalada lo cierra; si no, es la feature con más
   deriva residual del CUÁNDO.

## 6. Qué se puede servir de verdad

| feature | en tiempo real | cómo se sirve | veredicto |
|---|---|---|---|
| meteo del día y ventanas (21) | sí, ERA5-Land D−7 + IFS D−6→D+1 | `riesgo_hoy.py`; FWI 13 UTC, prec/2.02 | **entra** |
| `lst` | no (MODIS, latencia) | climatología mensual 2020-24 | entra como clima; PSI no medido |
| `ndvi, ndvi_med_30d, lai, swi010` | no en tiempo real | climatología mensual 2020-24 | entra como clima (ablación: −0.004 AUC) |
| estáticas + CLC 2018 + popdens 2020 | constantes | del cubo | **entra** (son el DÓNDE) |
| `n_fuegos_1km_90d, _10km_90d, _10km_365d` | **no** (EGIF llega con años) | — | **fuera**, como ya están |
| `n_fuegos_1km_hist, _10km_mismomes_hist` | como constante a 2020 | del cubo/EGIF | entra como ESTÁTICA del DÓNDE; no como dinámica |
| `frp_max_50km_7d, n_detec_50km_7d` | sí (FIRMS NRT) | `.env` + API | **entra**; ojo a la circularidad si FIRMS juzga |
| `rayos_dia, rayos_7d` | no (WGLC no es NRT) | se sirven a 0 | entra a 0: es lo que ve el modelo en test y en operación |
| `mes, dia_anio, es_festivo, ccaa` | sí | calendario | entra |

Regla aplicada en la fase 2: **el modelo CUÁNDO lleva solo dinámicas
servibles; el modelo DÓNDE, solo constantes.** Nada que se sirva a cero
distinto de como se entrenó.

## 7. Lo que esto cambia para la fase 2

- El problema operativo es el DÓNDE (prevalencia 10⁻⁵, 500,000 celdas el
  mismo día) y un mapa estático ya llega a 0.82 de AUC. Cualquier modelo que
  no lo contenga parte perdiendo.
- La métrica es AUC dentro del día con las dos verdades-terreno y bootstrap
  por días; el banco `eval_dia` (verano 2019, 2020 y 2022, 373,540 filas) se
  construyó para eso y no lo ve ningún entrenamiento.
- La línea base obligatoria (`xgb_v2_prototipo`) se mide con ese protocolo
  en `dos_05`, junto con el «mapa tonto» de densidad a 10 km y el percentil
  del FWI, que en la iteración 1 ganaba al modelo. Los resultados están en
  `05_iteracion2/README.md`.
