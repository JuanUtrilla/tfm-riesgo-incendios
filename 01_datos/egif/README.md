# EGIF, el registro oficial de igniciones

No hay script de descarga. El fichero `egif_civio.csv` (292,181 incendios,
1968-2023, 30 MB) se bajó a mano del portal de datos de Civio el 30/06/2026.
Procede del conjunto «Todos los incendios forestales» (CC BY-SA 3.0, elaborado
a partir de la EGIF, Estadística General de Incendios Forestales, del MAPA,
Ministerio de Agricultura, Pesca y Alimentación). Se volvió a bajar el
09/07/2026, cuando Civio republicó el conjunto con 2021 y 2022 consolidados
(`egif_civio_2026-08.csv`). Se guarda en `TFM_DATOS/`.

Columnas usadas: `id, fecha, lat, lng, superficie, causa, municipio,
idprovincia`. Se descartaron `time_ctrl` y `time_ext`, que se conocen después
del incendio, y `personal`, `medios`, `gastos` y `perdidas`, que son
consecuencia del incendio y no predictores.

## Dónde se procesa

| Paso | Script |
|---|---|
| Proyección a EPSG:3035, asignación de celda y deduplicación por (celda, fecha) | `03_iteracion1/31_muestreo/muestrear_dataset.py` (v1) y `muestrear_dataset_v4.py` (Civio 09/07) |
| Historial de fuego por celda (`n_fuegos_*`) | `03_iteracion1/32_features/extraer_features_historia.py` |
| Cruce con EFFIS: solo el 15 % de las igniciones de 25-100 ha coincide celda a celda con un perímetro del mismo día | `05_iteracion2/54_analisis/dos_04_analisis.py` |

## Lo que hay que saber antes de usarlo

La descarga de junio estaba incompleta desde 2021 (888, 226 y 23 incendios en
2021, 2022 y 2023) por retraso de consolidación. Por eso la iteración 1 (v1-v3)
se entrenó con 2015-2020. La republicación de julio consolida 2021 y casi todo
2022 (2,897 y 2,520 incendios), y v4 se probó sobre 2022.

A 2022 le faltan Navarra y Cantabria. `muestrear_dataset_v4.py` excluye ese
año las celdas a menos de 15 km de esas dos comunidades.

Cada registro es un punto de ignición, mientras que EFFIS (*European Forest
Fire Information System*) da perímetros de superficie quemada. Esa diferencia
es el origen del capítulo 4.
