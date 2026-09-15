# 3 · Primera iteración: etiqueta EGIF y muestreo caso-control

Esta iteración es el punto de partida del trabajo y la referencia con la que
se comparó el rediseño posterior. Sin ella no se podría cuantificar lo que
aportó cambiar la pregunta.

## Qué pregunta responde

«¿Es hoy un día peligroso en esta celda?». Es una manera razonable de plantear
el problema, pero no coincide con la pregunta que se hace en operación. El
capítulo 4 lo muestra con números.

## El diseño

Cada positivo es una ignición registrada en la EGIF (Estadística General de
Incendios Forestales), en su celda y su fecha (19,561 entre 2015 y 2020). Cada
negativo es la misma celda en otro día en que no ardió, tres por positivo. El
modelo es XGBoost y usa 46 variables; la lista completa tiene 50 y la
diferencia se explica más abajo.

## Qué hace cada script

Las tablas siguientes recogen los scripts de cada carpeta. En ellas aparecen
varias fuentes: AEMET (Agencia Estatal de Meteorología), el FWI (*Fire Weather
Index*), los focos FIRMS (*Fire Information for Resource Management System*) y
los rayos de WGLC (*WWLLN Global Lightning Climatology*).

### `31_muestreo/`, la tabla maestra

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `muestrear_dataset.py` | Proyecta las igniciones EGIF a la celda de 1 km, sortea los negativos en la misma celda otros días y asigna la partición temporal | Tabla maestra v1 (2015-2020) |
| `ensamblar_dataset.py` | Une la tabla maestra con las variables del cubo y las de historia, con controles de calidad | `dataset_modelo_v1.parquet` |
| `muestrear_dataset_v4.py`, `ensamblar_dataset_v4.py` | Lo mismo con el EGIF que Civio republicó en julio de 2026, con 2021 y 2022 consolidados | Dataset v4 (test 2022) |

### `32_features/`, historia de fuego y entorno

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `extraer_features_historia.py` | Historial de incendios EGIF por celda y entorno (1 y 10 km, mismo mes en años anteriores), focos FIRMS a 50 km en la última semana y rayos WGLC. Todas las ventanas terminan antes del día | Variables de historia de v1 |
| `extraer_features_historia_v4.py` | La misma receta sobre la tabla v4 | Dataset v4 |

### `33_train/`, entrenamiento

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `entrenar_modelo.py` | Entrena v1 (train 2015-2018, val 2019, test 2020) con una escalera de baselines, validación espacial por bloques de 100 km y un control con las etiquetas barajadas | El modelo base y `metricas_v1.json` |
| `entrenar_modelo_v3.py` | Reajuste con 2015-2020 completo | Explorado; no se desplegó |
| `entrenar_modelo_v4.py` | Entrena con el EGIF consolidado y prueba sobre 2022, el peor año de la década | El 0.89 de la memoria |
| `tuning_optuna.py` | Ajuste bayesiano acotado (40 pruebas) sobre la validación | Medir cuánto aporta el ajuste: +0.007 |
| `modelo_nativo_estacion.py` | Entrena directamente en la geometría de estaciones donde se sirve | Alternativa de servicio, explorada |
| `reconstruir_v2.py` | Reconstruye el modelo servido (`xgb_v2_prototipo`) con la receta de v1 sin cuatro variables; los 240 árboles salen idénticos | Reproducibilidad del modelo en producción |
| `verificar_v2.py` | Comprueba con las muestras que las 46 variables del modelo servido son las 50 menos esas cuatro, en el mismo orden | Se ejecuta nada más clonar el repositorio |

### `34_produccion/`, la cadena que se sirvió a diario

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `tiempo_real.py` | De la estación AEMET (observación, predicción municipal y FWI propio) a las 46 variables y la probabilidad | Núcleo de la producción de la iteración 1 |
| `preparar_prototipo.py` | Precálculo por estación: celda del cubo, estáticas y climatologías | Activos que usa `tiempo_real.py` |
| `ranking_diario.py` | Ranking nacional por estación para hoy y mañana, preparado para GitHub Actions | La serie de producción |
| `ranking_diario_cron.py` | Versión de cron local del ranking, con archivo histórico | Antes de pasar a Actions |
| `mapa_diario.py` | Mapa nacional de hoy y mañana sin el cubo (capas congeladas por `exportar_malla_gh.py`) | Los mapas publicados a diario en la iteración 1 |
| `mapa_riesgo_hoy.py` | Mapa nacional en tiempo real interpolando las estaciones a la malla (IDW, *inverse distance weighting*) | Primera versión del mapa |
| `mapa_riesgo_dia.py` | Mapa de un día pasado desde el cubo | Demostración y comprobación visual (por defecto, 17-jul-2022) |
| `predecir_punto.py` | Riesgo en un (lat, lon, fecha) con explicación local SHAP (*SHapley Additive exPlanations*) | Consulta puntual |
| `publicar_mapas_gh.py` | Publica los PNG del día en el repositorio del colector | Consultar los mapas desde el móvil |
| `retro_julio.py` | Mapas de julio de 2026 con meteorología observada, comprobados contra FIRMS | Primera medida sobre un mes entero |
| `validar_modelo.py` | Validación por estación y día de la serie de producción, con baselines (FWI, percentil local, persistencia) | Los números del capítulo 4 |

### `35_ablaciones/`

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `ablacion_features.py` | Modelo meteorológico con FWI absoluto, con percentil local y con los dos | Medir lo que aporta el percentil local: +0.042 de AUC-PR al añadirlo |
| `ablacion_v3_firms.py` | Sustituir el historial EGIF (que caduca) por su equivalente FIRMS | Decidir la variable autorregresiva de servicio |

## Los números de test

La tabla recoge los resultados en el conjunto de prueba, medidos con el
AUC-ROC (*Area Under the Receiver Operating Characteristic curve*) y con el
AUC-PR (área bajo la curva precisión-exhaustividad).

| Métrica | Valor |
|---|---|
| AUC-ROC de v4, test 2022 | 0.8906 [0.8644, 0.9136] |
| AUC-ROC de v1, test 2020 | 0.931 |
| AUC-PR de v1, test 2020 | 0.843 (azar 0.263) |
| AUC-PR de v1 en validación espacial por bloques de 100 km | 0.848 ± 0.028 |
| AUC-PR de v1 con las etiquetas barajadas | 0.231, igual que la prevalencia |
| Ajuste de hiperparámetros (Optuna, 40 pruebas) | +0.007 de AUC-PR |

El 0.89 es la cifra con la que abre la memoria. El diagnóstico posterior
explica por qué no medía lo que hacía falta en operación.

## La puesta en producción

`34_produccion/` es lo que se sirvió desde el 22 de julio de 2026 y sigue
sirviéndose: un ranking nacional por estación (`ranking_diario.py`) y los mapas
de hoy y mañana (`mapa_diario.py`), publicados por GitHub Actions desde el
repositorio del colector de AEMET. El modelo que corre ahí es
`xgb_v2_prototipo`: la receta de v1 sin cuatro variables autorregresivas de la
propia celda (`n_fuegos_1km_hist`, `n_fuegos_1km_90d`, `n_fuegos_10km_90d`,
`n_fuegos_10km_365d`). Se retiraron porque, con negativos de la misma celda en
otros días, el incendio del positivo entra en el historial de los negativos
posteriores de su celda ([`MODELO_B_BITACORA.md`](MODELO_B_BITACORA.md) §16).
`reconstruir_v2.py` lo reproduce árbol a árbol y `verificar_v2.py` comprueba la
lista de variables. v3 y v4 se entrenaron y se midieron, pero no se desplegaron
para no romper la serie a mitad de temporada.

## Lo que se midió en operación

Servido a diario y evaluado contra la superficie quemada real, el modelo bajó
del 0.89 a 0.57-0.64. De dónde sale esa diferencia se explica en el capítulo 4.
