# EGIF — el registro oficial de igniciones

**No hay script de descarga, y no es un olvido.** El fichero `egif_civio.csv`
(292.181 incendios, 1968-2023, 30 MB) se bajó a mano del portal de datos de
Civio —conjunto *«Todos los incendios forestales»*, publicado bajo CC BY-SA 3.0
a partir de la Estadística General de Incendios Forestales del MAPA— el
30/06/2026, y se volvió a bajar el 09/07/2026 cuando Civio republicó el
conjunto consolidando 2021 y 2022 (`egif_civio_2026-08.csv`). Se guarda en
`TFM_DATOS/`.

Columnas usadas: `id, fecha, lat, lng, superficie, causa, municipio,
idprovincia`. Descartadas por fuga: `time_ctrl`, `time_ext` (se conocen después
del incendio) y `personal`, `medios`, `gastos`, `perdidas` (consecuencia, no
predictor).

## Dónde se procesa

| Paso | Script |
|---|---|
| Proyección a EPSG:3035, celda por vecino más próximo, dedup por (celda, fecha) | `03_iteracion1/31_muestreo/muestrear_dataset.py` (v1) y `muestrear_dataset_v4.py` (Civio 09/07) |
| Historial de fuego por celda (`n_fuegos_*`) | `03_iteracion1/32_features/extraer_features_historia.py` |
| Cruce con EFFIS (solo el 15 % de las igniciones de 25-100 ha coincide celda a celda) | `05_iteracion2/54_analisis/dos_04_analisis.py` |

## Lo que hay que saber antes de usarlo

- **Incompleto desde 2021 en la descarga de junio** (888 / 226 / 23 incendios en
  2021 / 2022 / 2023): retraso de consolidación, no realidad. Por eso la
  iteración 1 (v1-v3) entrena con 2015-2020. La republicación de julio
  consolida 2021 y casi todo 2022 (2.897 y 2.520), y v4 testea sobre 2022.
- A 2022 le faltan Navarra y Cantabria; `muestrear_dataset_v4.py` excluye las
  celdas a menos de 15 km de esas dos comunidades ese año.
- Es un **punto** de ignición, no un perímetro. Esa diferencia con EFFIS es la
  semilla del capítulo 4.
