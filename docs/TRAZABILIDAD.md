# Trazabilidad: de cada número de la memoria a su script

Esta tabla es la que permite comprobar el trabajo. Para cada cifra que aparece
en la memoria: qué script la produjo, en qué fichero quedó escrita y con qué
datos hace falta ejecutarlo.

**Cómo leer la columna «Se ejecuta con»:**

- **muestra** — corre con `muestras/` recién clonado el repo.
- **cubo** — necesita IberFire (29 GB) en `TFM_DATOS`.
- **disco** — necesita además los datasets de entrenamiento en `TFM_USB`.
- **sellado** — el número lo produjo la cadena diaria en su repositorio; aquí
  está el código, pero la serie no se puede regenerar sin romper el sellado.

---

## Capítulo 2 · Análisis exploratorio

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| **2-6 por 100.000** en verano | Prevalencia real de celda-día quemada, frente al 25 % del diseño caso-control | `04_bisagra/44_es_la_especificacion/dos_00_cubo_etiquetas.py` | `dos_00_etiquetas.json` · [`02_eda/ANALISIS_CUBO.md`](../02_eda/ANALISIS_CUBO.md) §2 | cubo |
| 14,16 (2022) vs 0,40 (2018) | El rango entre un año malo y uno bueno | ídem | ídem | cubo |
| 2,97 % · 2,68 % | Celdas con algún EGIF (2015-20) / algún EFFIS (2021-24) | ídem | ídem | cubo |
| P(t+1\|t) = **0,51** | `is_fire` persiste: marca días ardiendo, no igniciones → se evalúa con `primer_dia` | ídem | ídem | cubo |
| Moran **0,81** (EFFIS, 1 km) | Autocorrelación espacial: casi todo es perímetro del mismo fuego | `05_iteracion2/54_analisis/dos_04_analisis.py` | `dos_04_analisis.json` | disco |
| **+0,036** | Lo que infla el AUC partir al azar en vez de por bloques de 100 km | ídem | ídem | disco |
| corr 0,987-0,998, sesgo 0 | El cubo *es* ERA5-Land reprocesado (salvo precipitación, /2,02) | `04_bisagra/42_no_es_la_meteo/prototipo_TFM_fuego/malla_02_vs_cubo.py` | `malla_02_vs_cubo.json` | cubo |
| fwi_pctl_local 84,3 vs 47,0 | Lo que separa positivos de negativos en train (medianas) | `02_eda/eda_dataset.py` | [`02_eda/figuras/eda_resumen.log`](../02_eda/figuras/eda_resumen.log) | disco |
| popdens y dist_carreteras: **misma mediana** | Las estáticas no discriminan en el diseño caso-control — el aviso temprano del capítulo 4 | ídem | ídem | disco |

## Capítulo 3 · Primera iteración

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| **0,8906** [0,8644, 0,9136] | AUC-ROC de v4 en test 2022 — **el «0,89» de la memoria** | `03_iteracion1/33_train/entrenar_modelo_v4.py` | `dataset/metricas_v4.json` | disco |
| 0,931 · 0,843 | AUC-ROC y AUC-PR de v1/v2 en test 2020 | `03_iteracion1/33_train/entrenar_modelo.py` | `dataset/metricas.json` | disco |
| 0,848 ± 0,028 | GroupKFold espacial a 100 km: no colapsa fuera de zona vista | `03_iteracion1/33_train/entrenar_modelo.py` | ídem | disco |
| 0,231 | Etiquetas barajadas ≈ azar: no hay fuga de datos | `03_iteracion1/33_train/entrenar_modelo.py` | ídem | disco |
| +0,042 | Ablación FWI absoluto vs percentil local — la contribución del TFM | `03_iteracion1/35_ablaciones/ablacion_features.py` | `dataset/ablaciones_v1.json` | disco |
| +0,007 | Lo que aportó el tuning bayesiano (40 trials) | `03_iteracion1/33_train/tuning_optuna.py` | `dataset/optuna.json` | disco |
| 46 = 50 − 4 | `xgb_v2_prototipo` (el modelo SERVIDO) es v1 sin las 4 autorregresivas intra-celda contaminadas por el muestreo misma-celda | `03_iteracion1/33_train/verificar_v2.py` | `muestras/modelos/xgb_v2_prototipo_features.json` | muestra |
| 0,828 (−0,015 vs 0,843) | AUC-PR de ese reentrenamiento del 15/07 — interactivo, sin script versionado; la decisión y su racional, en la bitácora | [`03_iteracion1/MODELO_B_BITACORA.md`](../03_iteracion1/MODELO_B_BITACORA.md) §16 | ídem | — |

## Capítulo 4 · La bisagra

### 41 · La validación operativa

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| 0,642 D0 · 0,618 D1 · 0,707 ranking · 0,596 retro | El modelo en operación, 20 días sellados, etiqueta EFFIS | `03_iteracion1/34_produccion/validar_modelo.py` | `rankings/validacion_estacion_dia.csv` | sellado |
| **0,702** D0 | El baseline duro: FWI percentil local — bate al modelo | ídem | ídem | sellado |
| 0,540 → 0,702 | El baseline al pasar de 3 a 20 días: por qué se retiró la conclusión del 28/07 | ídem | ídem | sellado |
| 0,715 [0,659, 0,764] | Con etiqueta FIRMS filtrada (FRP≥20 MW) | ídem | ídem | sellado |
| 0,723 | La **persistencia** (`n_detec_50km_7d`) empatando con el modelo en D0 | ídem | ídem | sellado |
| 0,644 · pctl 72,3 | Contra 114 incidentes de los partes oficiales | `04_bisagra/41_validacion_operativa/validar_miteco.py` | `validacion_miteco_*.csv/json` | cubo |
| p85,5 · 58 % >p80 | 1.710 eventos FIRMS 2021-24 (y la pseudo-replicación que lo inflaba) | `04_bisagra/41_validacion_operativa/validar_eventos_firms.py` | `validacion_operativa.csv` | cubo |

### 42 · No es la meteorología

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| **+8,00 mm** en `precip_30d` | El sesgo húmedo de ERA5-Land frente a AEMET | `04_bisagra/42_no_es_la_meteo/experimento_b_era5.py` | `experimento_b_fase1.json` | cubo |
| −10,04 de FWI · −16,50 pctl | Lo que se propaga de ese sesgo | ídem | ídem | cubo |
| 12,6 % vs 1,0 % en pctl ≥99,9 | La saturación por cruce de fuentes en el percentil | ídem | ídem | cubo |
| −0,113 (control fallido) | Por qué la comparación de AUC entre brazos **no se reporta** | `04_bisagra/42_no_es_la_meteo/experimento_b_fase2.py` | `experimento_b_fase2.json` | cubo |
| factor ~2 en el FWI | Diagnóstico de la diferencia entrenamiento vs producción | `04_bisagra/42_no_es_la_meteo/diagnostico_fwi.py` | `diagnostico_fwi.json` | cubo |
| 13 UTC, estable en 3 meses | La hora real del FWI del cubo | `04_bisagra/42_no_es_la_meteo/verificar_hora_fwi.py` | `verificar_hora_fwi.json` | cubo |
| ECMWF IFS mejor en viento (1,30) | Qué fuente predice mejor el tiempo | `04_bisagra/42_no_es_la_meteo/verificar_fuentes_meteo.py` | `verificacion_meteo.json` | cubo |

### 43 · No es el train/serve

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| **−0,0045** | Los dos proxis operativos juntos (satélite congelado + rayos a 0) | `04_bisagra/43_no_es_el_train_serve/ablacion_proxies_operativos.py` | `dataset/ablacion_proxies.json` | disco |
| −0,004 | Eliminar las cinco satelitales cuesta lo mismo que congelarlas | ídem | ídem | disco |
| −0,023 a −0,040 | Las 36 features desviadas, acumuladas | `04_bisagra/43_no_es_el_train_serve/auditoria_train_serve.py` | `auditoria_train_serve.json` | cubo |
| deriva train→serve | Features del cubo frente a ERA5-Land en los nodos | `04_bisagra/43_no_es_el_train_serve/dos_06_deriva.py` | `dos_06_deriva.json` | cubo |

### 44 · Es la especificación

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| **98,8 %** de celdas con 25 % de positivos | El muestreo caso-control no enseña a distinguir celdas | `04_bisagra/44_es_la_especificacion/dos_00_cubo_etiquetas.py` | `dos_00_etiquetas.json` | cubo |
| **0,92 = 0,92** (caso-control) | La métrica de desarrollo es ciega a los dos diseños | `04_bisagra/44_es_la_especificacion/dos_02_muestrear.py` + `05_iteracion2/55_train/dos_05_modelos.py` | `dos_05_metricas.json` | disco |
| **0,744 vs 0,828** (dentro del día) | La misma comparación con la métrica correcta | `05_iteracion2/55_train/dos_05_modelos.py` | ídem | disco |

## Capítulo 5 · Segunda iteración

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| 0,828 · +0,083 [+0,063, +0,105] | `donde_dia` (único, muestreo del mismo día), test 2020 EGIF | `05_iteracion2/55_train/dos_05_modelos.py` | `dos_05_metricas.json` | disco |
| 0,831 · +0,087 [+0,065, +0,108] | `donde_cel_hist × cuando` (la pareja) | ídem | ídem | disco |
| **0,809** test 2024 · **0,752** en 2026 | `donde_dia_effis` — el único con etiqueta EFFIS | `05_iteracion2/55_train/dos_13_modelos_effis.py` | `dos_13_metricas.json` | disco |
| 0,840 [+0,015, +0,085] test 2024 | `donde_effis_c × cuando_egif` — la pareja definitiva | ídem | ídem | disco |
| 0,744 | El único **sin FIRMS** (producción, 0,647, cae a 0,611) | ídem | ídem | disco |
| p30/p90/p98 | Los cortes BAJO/MODERADO/ALTO/EXTREMO | `05_iteracion2/56_calibracion/dos_14_cortes.py` | `dos_14_cortes.json` | disco |
| **1:10-1:30** óptimo · 0,759 en 2026 | La escalera del ratio de negativos | `05_iteracion2/57_ablaciones/dos_18_ratio.py` | `dos_18_ratio.json` | disco |
| **79,5 → 81,9-80,6** | Percentil ponderado por hectáreas: lo que sí mejora | ídem | ídem | disco |
| 0,005 | El **ruido del sorteo de negativos**, midiendo el mismo diseño otra vez | ídem | ídem | disco |
| entrenar solo verano: peor | Resultado negativo | `05_iteracion2/57_ablaciones/dos_08_verano.py` | `dos_08_verano.json` | disco |
| −0,02 | La capa «ya quemado» empeora | `05_iteracion2/57_ablaciones/dos_17_capa_quemado.py` | `dos_17_capa_quemado.csv/json` | disco |

## Capítulo 6 · Comparación

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| 0,647 prod · 0,752 único · **0,759** único 1:10 · 0,705 pareja | Temporada 2026 contra EFFIS | `06_comparacion/dos_09_temporada2026.py` | `dos_09_temporada2026.csv/json` | disco |
| **0,611** | Producción sin FIRMS (0,520 con juez MITECO): la caída real es de 0,036, no de 0,126 | ídem | ídem | disco |
| 0,692 vs 0,763 en los 11 días grandes | Los megaincendios: los acierta el único 1:10, no producción | ídem | ídem | disco |
| 0,568 · 0,547 · **0,605** (14/19 días) | Cara a cara justo: previsión contra previsión, 19 días | `06_comparacion/comparar_rankings_justo.py` | `rankings_justo.csv/json` | sellado |
| 17 días de julio 2026 | La primera comparación seria con área quemada | `06_comparacion/comparar_julio2026.py` | `julio2026_effis.json` | cubo |
| **0,759 · Δ+0,112 [+0,074, +0,151]** | **El veredicto de la memoria** | `06_comparacion/dos_19_veredicto.py` | `dos_19_veredicto.json/png` | disco |
| — | **Todas las cifras de 2026 y MITECO de esta página se recalcularon el 31/08/2026** tras corregir la fuga de futuro de FIRMS (`docs/BITACORA.md`). Las de test 2020/2024 no dependían de ella y no cambian. | `06_comparacion/comparar_rankings.py` | — | — |

## Capítulo 7 · Producción y jueces

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| pctl 39,2 / 73,5 / 90,6 | Juez EFFIS, por día | `07_produccion/puntuar_effis.py` | `puntuacion_effis.csv` | sellado |
| 0,598 / 0,807 / 0,771 (AUC10) | Juez MITECO acumulado | `07_produccion/dos_15_veredicto_miteco.py` | `veredicto_miteco.csv/json` | sellado |
| 0,507 / 0,572 / 0,600 / 0,607 | Juez por estación, 688 estaciones | `07_produccion/dos_16_juez_estaciones.py` | `veredicto_estaciones.csv/json` | sellado |
| «con 17 días no se distingue del ruido» | El veredicto acumulado | `07_produccion/gh_estado.py` + `06_comparacion/dos_19_veredicto.py` | `veredicto_acumulado.json` | sellado |

---

## Los números que **no** se pueden reproducir, y por qué

| Número | Por qué |
|---|---|
| Todo lo marcado **sellado** | Lo produjo la cadena diaria contra APIs en vivo (IFS, FIRMS, EFFIS) en una fecha concreta. Reejecutarlo hoy da otra cosa: el reanálisis se corrige, EFFIS cartografía tarde. Los resultados están congelados en el Release `estado` de `tfm-fuego-malla` |
| La comparación de AUC entre brazos del Experimento B | El control falló. Ver [`LIMITACIONES.md`](LIMITACIONES.md) |
| El resultado con etiqueta FIRMS de julio | Requiere `FIRMS_MAP_KEY`, que solo existe como secreto del repositorio de producción |
