# 1 · Datos: nueve fuentes, un cubo y una malla

Este capítulo es común a las dos iteraciones. Entre la primera y la segunda no
cambian los datos; cambia qué se toma como positivo, qué como negativo y con
qué verdad se evalúa.

## Las fuentes

| Carpeta | Fuente | Para qué se usa |
|---|---|---|
| `cubo/` | IberFire, datacubo diario de 1 km (29 GB, 261 variables, 2008-2024) | Las 46 variables de entrenamiento de las dos iteraciones |
| `era5land/` | ERA5-Land (Copernicus/ECMWF), ~9 km | La meteorología: la del cubo al entrenar y la de los 5,605 nodos al servir |
| `ifs/` | IFS de ECMWF vía Open-Meteo | La previsión de hoy y mañana, donde el reanálisis aún no existe |
| `aemet/` | AEMET: observación horaria y climatología diaria por estación | La meteorología de la producción de la iteración 1 |
| `effis/` | EFFIS (Copernicus EMS) | Perímetros de superficie quemada: etiqueta de la iteración 2 y verdad para evaluar |
| `miteco/` | Parte diario del MITECO | Incidentes con medios del Estado, usados como segunda verdad durante el desarrollo |
| `firms/` | FIRMS (VIIRS, NASA) | Focos activos de la última semana como variables de entorno |
| `egif/` | EGIF (vía Civio) | Registro oficial de igniciones: etiqueta de la iteración 1 ([README propio](egif/README.md)) |
| `extra/` | Carreteras, población, usos del suelo, censo ganadero, rayos WGLC | Variables humanas y estructurales |
| `comun/` | `config`, `features`, `fwi_canadiense` | Rutas, el orden de las 46 variables y el cálculo del FWI |

## Qué hace cada script

### `comun/`, módulos que importa todo el repositorio

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `config.py` | Rutas y constantes del pipeline de malla. Cada entrada se busca en `salida/`, luego en `TFM_DATOS` y por último en `muestras/` | Lo importan unos 50 scripts |
| `config_expansion.py` | Rutas del disco externo con los datasets de entrenamiento (`TFM_USB`) | Iteración 2 y ablaciones |
| `features.py` | La lista de variables y su orden, con una comprobación de que no diverge | Entrenamiento y servicio de los dos modelos |
| `fwi_canadiense.py` | El sistema canadiense FWI (Van Wagner, 1987) en numpy: FFMC, DMC, DC, ISI, BUI y FWI | Todo cálculo de FWI propio (AEMET, ERA5-Land, IFS) |
| `exportar_limites.py` | Capa administrativa (comunidades, provincias, capitales) proyectada a la malla de 1 km y guardada en un `.npz` | El fondo de los mapas |

### `cubo/`, IberFire

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `extraer_features_cubo.py` | Extrae del cubo las variables de cada (celda, día) de la tabla maestra, por bloques y reanudable | Dataset de la iteración 1. La iteración 2 importa sus funciones para usar la misma receta |
| `extraer_features_cubo_v4.py` | Ejecuta el extractor anterior sobre la tabla maestra v4 | Dataset v4 (EGIF consolidado) |
| `exportar_malla_gh.py` | Saca del cubo lo que el mapa nacional necesita (unos 35 MB: estáticas y climatologías) | Permite que la cadena corra en GitHub Actions sin los 29 GB |

### `era5land/`, reanálisis

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `malla_01_nodos.py` | Define los 5,605 nodos de ERA5-Land sobre España y el mapeo de cada celda a su nodo | Base de la rama de malla |
| `malla_02_descarga.py` | Descarga ERA5-Land horario del CDS y lo agrega a diario (tmax, HR mínima, viento máximo, precipitación) | Meteorología de servicio y de evaluación |
| `malla_04_climatologia.py` | Climatología del FWI por nodo y mes (2008-2014) | El denominador del percentil local en servicio |
| `malla_06_descarga_historico.py` | ERA5-Land horario 2015-2020 para entrenar con la misma fuente que sirve | Se descartó: el cubo ya es ERA5-Land reprocesado (ver `02_eda`) |
| `descargar_era5_2026.py` | ERA5-Land horario de 2026 para el experimento de fuentes meteorológicas | `04_bisagra/42` |
| `descargar_cape_era5.py` | CAPE, precipitación convectiva y K-index diarios de ERA5 | Proxy de rayo seco, explorado como variable |
| `bajar_era5land_2025.py` | ERA5-Land de mayo a noviembre de 2025 | El replay de la temporada 2025 (`06_comparacion`) |

### `ifs/`, previsión

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `malla_02b_ifs.py` | Previsión IFS en los mismos 5,605 nodos, de D−6 a D+1, ajustada a la escala de ERA5-Land | La rama de previsión de la cadena diaria |
| `rellenar_ifs.py` | Recupera del archivo de Open-Meteo las pasadas IFS de días pasados, tal como se emitieron | Entradas del replay de 2026 |
| `bajar_ifs_2025.py` | Lo mismo para la temporada 2025 | Entradas del replay de 2025 |

### `aemet/`, observación y climatología por estación

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `aemet_horario_collector.py` | Recoge las observaciones horarias de todas las estaciones cada 6 h (GitHub Actions) | La meteorología en tiempo real de la producción de la iteración 1 |
| `observacion_horaria.py` | Consulta la observación horaria de una estación | Utilidad de desarrollo del colector |
| `aemet_descarga_historico.py`, `_key2.py`, `_key3.py` | Descargan el histórico climatológico diario de todas las estaciones, repartido entre tres claves de la API | Climatología de FWI por estación |
| `descargar_historico_aemet.py` | Histórico diario 2015-2025 consolidado en un parquet | Base de la climatología de FWI propia |
| `descargar_datos_2025_2026.py` | Climatologías diarias de 2025-2026 con arranque en noviembre de 2024, para que el FWI tenga rodaje | Validación por estaciones |

### `effis/`, `firms/`, `miteco/`, `extra/`

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `effis/descargar_effis.py` | Perímetros de superficie quemada de EFFIS para España (WFS) | Etiqueta de la iteración 2 y verdad de las evaluaciones |
| `effis/bajar_effis_2025.py` | La capa anual de 2025 completa | Verdad del replay de 2025 |
| `firms/firms_descarga.py` | Detecciones VIIRS 2015-2024 para la Península | Variables de entorno de fuego reciente en entrenamiento |
| `firms/firms_api.py` | Descarga FIRMS en tiempo casi real, con reintentos, distinguiendo un fallo de la API de «no hay focos» | Las mismas variables en servicio |
| `firms/extraer_historia_firms_v3.py` | Propensión local de fuego a partir de FIRMS, alternativa al historial EGIF | Ablación de la iteración 1 |
| `firms/focos_activos.py` | Mapa de focos activos sin modelo | Capa de situación junto al mapa de riesgo |
| `miteco/parseo_y_chuncking.py` | Convierte el PDF del parte del MITECO en una lista de incidentes con municipio | Copia literal del módulo del RAG; la usa `validar_miteco.py` |
| `miteco/validar_miteco.py` | Geocodifica los incidentes y puntúa un mapa contra ellos | Segunda verdad durante el desarrollo |
| `extra/descargar_features_extra.py` | Carreteras (Natural Earth), población (WorldPop) y usos del suelo (WorldCover) por punto | Variables humanas y estructurales |
| `extra/ganaderia_descarga.py` | Censo ganadero armonizado de Europa, recortado a España | Variable explorada |

## En qué orden se ejecutan

```
malla_01_nodos.py        los nodos de ERA5-Land y el mapeo celda -> nodo
malla_02_descarga.py     descarga y agregación diaria desde el CDS
malla_04_climatologia.py climatología del FWI por nodo
extraer_features_cubo.py variables del cubo
malla_02b_ifs.py         la rama de previsión, en los mismos nodos
```

Las descargas de AEMET, FIRMS, EGIF y las fuentes extra no dependen unas de
otras.

## Dos cosas de los datos que condicionan el diseño

El FWI del cubo corresponde a las 13 UTC. Se comprobó antes de construir nada
(`04_bisagra/42_no_es_la_meteo/verificar_hora_fwi.py`) y toda la rama de malla
calcula el FWI a esa hora. Servirlo a otra hora introduce un factor de
alrededor de 2 respecto al entrenamiento.

ERA5-Land da más lluvia que las estaciones: +8.00 mm en 30 días frente a AEMET
(`04_bisagra/42_no_es_la_meteo/experimento_b_era5.py`). De ahí salen menos días
sin lluvia, un combustible más húmedo y un FWI 10 puntos más bajo. Es la razón
de que entrenar y servir con la misma fuente importe tanto.

## Ejecutar

Las descargas de ERA5-Land necesitan cuenta en el CDS (`CDSAPI_KEY`), FIRMS una
clave de mapa (`FIRMS_MAP_KEY`) y AEMET su API key. El cubo IberFire se indica
con `TFM_DATOS`. Sin nada de eso, `muestras/` trae julio de 2026 ya procesado.
