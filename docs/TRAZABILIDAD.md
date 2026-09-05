# Trazabilidad: de cada número de la memoria a su script

Esta tabla es la que permite comprobar el trabajo. Para cada cifra que aparece
en la memoria: qué script la produjo, en qué fichero quedó escrita y con qué
datos hace falta ejecutarlo.

> **Cifras selladas el 31/08/2026**, tras corregir la fuga de futuro de FIRMS
> y relanzar la evaluación retrospectiva. El juego oficial es el de **ventana
> FIRMS de 7 días** (`dos_09_temporada2026.json`, que es lo que lee `dos_19`);
> la corrida con ventana de 5 (`dos_09_ventana5.json`) es solo la ablación del
> desajuste train/serve. Solo la sección «Capítulo 7 · Producción y jueces» se
> actualiza hasta el cierre de septiembre. A partir de aquí, un número que no
> coincida con esta tabla es un error del documento, no una duda sobre el
> resultado.

**Cómo leer la columna «Se ejecuta con»:**

- **muestra** — corre con `muestras/` recién clonado el repo.
- **cubo** — necesita IberFire (29 GB) en `TFM_DATOS`.
- **disco** — necesita además los datasets de entrenamiento en `TFM_USB`.
- **sellado** — el número lo produjo la cadena diaria en su repositorio; aquí
  está el código, pero la serie no se puede regenerar sin romper el sellado.

---

## Reproducción verificada (02/09/2026)

Todos los entrenamientos citados se reejecutaron en copia aislada desde sus
parquets sellados, con `environment.yml` (xgboost 3.2.0), y se compararon con
los artefactos originales: **cero discrepancias en todas las métricas y `.ubj`
idénticos por md5** para v1 (50 métricas), v4 (39 + 46 metadatos), `dos_05`
(645), `dos_13` (144) y `dos_18` (111, los cinco modelos de la escalera). El
modelo servido, `xgb_v2_prototipo`, que no tenía script, se reconstruye con
`03_iteracion1/33_train/reconstruir_v2.py` (receta de v1 sin las cuatro
autorregresivas): 240/240 árboles idénticos, mejor iteración 139, AUC-PR de
validación 0,8528 y de test 0,828 con 140 árboles. Fuera del alcance, por
diseño: la temporada 2026 (`dos_09`), cuya verdad EFFIS se reescribe a diario.

## Prerregistro del replay 2025/2026 (escrito el 02/09/2026 a las 18:10 (commit ed7931f), ANTES de correr nada)

**Modelos (6, todos entrenados con datos ≤ 2024):** producción sobre la malla
(`xgb_v2_prototipo`, EGIF) · único 1:3 (`donde_dia_effis`) · único 1:10
(`donde_dia_effis_r10`) · pareja (`donde_effis_c` × `cuando`, EGIF) · dónde solo
(`donde_effis_c`) · cuándo solo (`cuando`). Fuera: producción con datos AEMET
(no replicable), v3/v4, los EGIF de dos_05, r30-r100.

**Temporadas y condiciones:** 2025 (25-may→01-nov) y 2026 (25-may→ último día con
reanálisis D−7 disponible), cada una con IFS (servicio) y reanálisis del día
(cota superior). Entradas: `archivo_ifs/` (md5 en `MD5_archivo_replay.txt`).

**Métrica principal:** AUC medio por día (días con ≥1 celda EFFIS quemada, EFFIS
= perímetros con FIREDATE ese día), IC95 por bootstrap de días (2.000, semilla
42), diferencia contra producción-malla. **Operativas:** captura en el 2 % más
alto (celdas, hectáreas, incendios ≥100 ha con alguna celda), percentil mediano
y ponderado por hectáreas.

**Regla:** se reportan los seis siempre. «Mejor» = mayor AUC medio en 2025 con IC
que no toque el de producción. 2026 completo es cobertura, no prueba. Ningún
resultado de 2025 cambia el conjunto de modelos ni las métricas.

**Pregunta secundaria («si», no «dónde»):** para producción, cuándo y pareja,
umbral absoluto = probabilidad que deja de media el 2 % de celdas por encima en
los veranos 2015-2024 (calibrado en el cubo), aplicado igual todos los días; se
evalúa con TODOS los días, también los de cero fuego (tabla aviso × fuego). El
único queda fuera por construcción (negativos del mismo día).

**Resultado del replay (02/09/2026 20:10, tras el prerregistro):** ver
[`REPLAY_VEREDICTO.md`](REPLAY_VEREDICTO.md) y [`REPLAY_SI.md`](REPLAY_SI.md).
Scripts: `~/Desktop/Master/archivo_ifs/replay_{dia,temporada,verdad,veredicto,figuras,si}.py`
(pendiente copiarlos al repo); salidas `archivo_ifs/replay/` (md5 en `MD5_replay.txt`).

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| **0,806 · 0,807 · 0,794** vs 0,743 (2025, IFS) | Único 1:3, único 1:10 y pareja contra producción-malla en la temporada 2025 completa, en condiciones de servicio; Δ +0,063 [+0,026, +0,102] | `replay_temporada.py` + `replay_veredicto.py` | `replay_2025_ifs.csv`, `replay_veredicto.json` | archivo |
| 0,757 · 0,762 · 0,716 vs 0,658 (2026, IFS) | Lo mismo en 2026 completo (cobertura, no prueba) | ídem | `replay_2026_ifs.csv` | archivo |
| −0,004…+0,001 | Coste de la previsión: AUC con IFS − con reanálisis, IC cruzan el cero | ídem | `replay_veredicto.json` | archivo |
| 32 % · 26-29 % · 22 % (2025) / 6-7 % vs 11,5 % (2026) | Hectáreas capturadas en el top-2 % del mapa: pareja, únicos, producción | ídem | ídem | archivo |

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
| 0,828 (−0,015 vs 0,843) | AUC-PR de ese reentrenamiento del 15/07 — fue interactivo, pero `reconstruir_v2.py` lo reproduce árbol a árbol (02/09/2026); la decisión y su racional, en la bitácora | `03_iteracion1/33_train/reconstruir_v2.py` · [`MODELO_B_BITACORA.md`](../03_iteracion1/MODELO_B_BITACORA.md) §16 | `xgb_v2_reconstruido.ubj` | disco |

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
| 0,004 | El **ruido del sorteo de negativos**, midiendo el mismo diseño otra vez (0,752 vs 0,748 en 2026; 0,809 vs 0,804 en test 2024) | ídem | ídem | disco |

### Calibración del «si» y de la celda (05/09/2026)

Partición fija en los cuatro scripts: **calibración 2015-2021 · validación
2022-2023 · test 2024**. El umbral se elige solo en calibración.

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| 33,9 % · **85,6 %** en el NO | Los **dos regímenes**: días con fuego en nov-mar, y fuego de dic-abr concentrado en el noroeste (14,4 % del territorio) | `05_iteracion2/56_calibracion/dos_25_barrido_historico.py` | `dos_25_hist.npz` | cubo |
| 3.653 × 5 × 4.096 | Histograma diario por modelo (97 bloques) y `quem`, 64.088 celdas quemadas con su puntuación | ídem | ídem | cubo |
| **+0,297 [+0,198, +0,381]** | BSS del «si» en invierno-primavera, test 2024, r10 (los cinco modelos con IC que excluye el cero) | `05_iteracion2/56_calibracion/dos_26_calibra_si.py` | `dos_26_calibracion.csv` | cubo |
| −0,107 [−0,230, **+0,022**] | BSS en verano: **cruza el cero**. El «si» no se distingue de la climatología en jun-sep | ídem | ídem | cubo |
| 0,794 [0,65, 0,89] | SEDI del «si» en invierno, r10 | ídem | ídem | cubo |
| 1,08×-1,18× | Separación de umbrales verano/invierno: **un solo umbral vale todo el año** | ídem | ídem | cubo |
| 1,35× | Cuánto **ensancha** el bootstrap por bloques de 14 días frente al iid usado en el resto del TFM | ídem | ídem | cubo |
| **0,1259 % · lift 35,6×** | Qué significa EXTREMO: frecuencia observada de quema en la banda | `05_iteracion2/56_calibracion/dos_27_escala_absoluta.py` | `dos_27_bandas.csv` | cubo |
| 5 celdas en 10 años | Lo que ardió en BAJO, que es el 30 % de España | ídem | ídem | cubo |
| 0,0 % de días sin rojo | `r10` con corte **absoluto** nunca se apaga (es estático) → hace falta la puerta del «si» | ídem | `dos_27_dias.csv` | cubo |
| 44,0 % · **6,2 %** | Días de invierno sin rojo con la puerta puesta, y días grandes que se pierden | ídem | ídem | cubo |
| **1,565 %** | Techo: probabilidad calibrada de la celda más roja que llega a existir (r10) | `05_iteracion2/56_calibracion/dos_28_calibra_celda.py` | `dos_28_niveles.csv` | cubo |
| beta descartada | La beta da coeficiente **negativo** en −log(1−s) y no es monótona (techo 0,000 %); se usa isotónica | ídem | `dos_28_beta.json` | cubo |
| **0,28 % → 10,36 %** | Probabilidad de la peor celda, de diciembre a julio (factor 37) | `05_iteracion2/56_calibracion/dos_29_calibra_estacional.py` | `dos_29_punta.csv` | cubo |
| 1,8× contra 2,4× | Error típico de la calibración por mes contra la global; en julio, 7× contra 52× | ídem | ídem | cubo |
| 0,453 (todo) contra **0,947** (3 años) | **Ventana móvil: hipótesis refutada.** Cuanta menos historia, peor | `05_iteracion2/56_calibracion/dos_31_ventana_movil.py` | `dos_31_ventanas.csv` | cubo |
| 0,00 % → **12,76 %** | La punta de julio, año a año (mediana 0,53 %): la variación no está en la puntuación de la celda sino en la intensidad del año | ídem | ídem | cubo |
| 1,7× contra **3,4×** | El **producto de dos capas empeora**: hereda la infraconfianza del «si» y la multiplica | `05_iteracion2/56_calibracion/dos_32_producto_dos_capas.py` | `dos_32_punta.csv` | cubo |
| 2,05 % contra **14,95 %** | Julio 2022 condicionado a día grande, contra el máximo de calibración: **7× fuera del envolvente**. No es calibración, es extrapolación | ídem | ídem | cubo |
| 11,5 % → **4,8 %** | **La «anomalía de la punta 2026» es UN día** (23-jul, 21.348 ha): quitándolo, el orden vuelve al de 2025 | `06_comparacion/dos_33_anomalia_punta.py` | `dos_33_sensibilidad.csv` | replay |
| [1,6, 27,1] contra [2,9, 14,7] | Los IC de producción y r10 en la punta de 2026 **se solapan por completo** | ídem | ídem | replay |
| **1,391** contra 1,404 teórico | Caja/círculo a 10 km medido sobre la misma rejilla; a 50 km, 1,300 contra 1,299 | `05_iteracion2/57_ablaciones/dos_34_auditoria_geometria.py` | `dos_34_geometria.csv` | cubo |
| Spearman 0,92-0,98 | El desajuste caja/radio es de **escala, no de orden**: por eso el AUC apenas se resiente | ídem | ídem | cubo |
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

## Prerregistro: r10 con todos los años (escrito el 05/09/2026 a las 11:30 UTC, ANTES de correr nada)

Hipótesis: entrenar `donde_dia_effis_r10` con 2015-**2024** en vez de
2015-**2022** (+32 % de positivos) mejora en 2025-2026, que son externos al cubo.
Métrica primaria fijada de antemano: **AUC medio por día** en los días del
replay con IFS, con IC bootstrap de la diferencia pareada. Criterio de éxito:
que el IC no toque el cero. Prerregistro íntegro en
`calibracion_si/PRERREGISTRO_r10_todo.md`.

| Número | Qué es | Script | Salida | Se ejecuta con |
|---|---|---|---|---|
| Δ máx 0,00e+00 · Spearman 1,000000 | `r10_base` reproduce **exactamente** el `donde_dia_effis_r10.ubj` servido | `05_iteracion2/57_ablaciones/dos_30_r10_todo.py` | `dos_30_info.json` | disco |
| Δ máx 0,00e+00 en 224 días | Control: los 4 modelos no tocados reproducen el replay original | `06_comparacion/replay_dia_todo.py` | `replay_<año>_ifs_r10todo.csv` | archivo IFS |
| **+0,0033 [−0,0042, +0,0104]** | Diferencia pareada de AUC, 224 días. **El IC cruza el cero: el criterio prerregistrado NO se cumple** | ídem | ídem | archivo IFS |
| 0,7898 → 0,7931 | AUC medio, base contra todos los años. Gana el 54,5 % de días | ídem | ídem | archivo IFS |
| +0,079 contra +0,0033 | La distancia r10→producción es **24 veces** la ganancia de reentrenar con todo | ídem | ídem | archivo IFS |

Lectura: el modelo está **saturado de datos**. La partición
`train 2015-2022 / val 2023 / test 2024` se mantiene **sin coste de
rendimiento**, conservando un test honesto reservado.

## Los números que **no** se pueden reproducir, y por qué

| Número | Por qué |
|---|---|
| Todo lo marcado **sellado** | Lo produjo la cadena diaria contra APIs en vivo (IFS, FIRMS, EFFIS) en una fecha concreta. Reejecutarlo hoy da otra cosa: el reanálisis se corrige, EFFIS cartografía tarde. Los resultados están congelados en el Release `estado` de `tfm-fuego-malla` |
| La comparación de AUC entre brazos del Experimento B | El control falló. Ver [`LIMITACIONES.md`](LIMITACIONES.md) |
| El resultado con etiqueta FIRMS de julio | Requiere `FIRMS_MAP_KEY`, que solo existe como secreto del repositorio de producción |
