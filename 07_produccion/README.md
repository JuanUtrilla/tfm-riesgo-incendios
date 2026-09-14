# 7 · Producción: el mapa de cada mañana

La cadena corre en GitHub Actions desde este repositorio
(`.github/workflows/mapa_diario.yml`), una vez al día, hacia las 03:03 en verano
y las 02:02 en invierno (hora de Madrid). Publica en `publicado/`
el mapa de hoy y el de mañana con el modelo elegido, el r10, de dos formas: en
escala absoluta, con la cifra del día, y por percentil del día. El estado que hace falta entre corridas (el reanálisis
acumulado y los mapas de los últimos días) se guarda en el Release `estado`
del repositorio, porque no cabe en git.

Durante la temporada de 2026 la misma cadena corrió en un repositorio aparte
(`tfm-fuego-malla`) y publicaba, además, los mapas de los otros candidatos.
Aquí queda su código completo, incluidos los scripts que ya no se ejecutan.

![Los mapas publicados hoy](../publicado/mapas_hoy.png)

*El producto de la cadena: r10 en escala absoluta con la cifra del día, y el mismo modelo por percentil del día.*

## Cómo se genera un día

```
gh_estado.py pull          baja el estado de la corrida anterior (Release)
gh_reanalisis.py           ERA5-Land del mes en curso, incremental, hasta D-7 (CDS)
malla_02b_ifs.py           previsión IFS en los 5,605 nodos, de D-6 a D+1 (Open-Meteo)
riesgo_hoy.py              mapa de referencia: el modelo de producción sobre la malla
dos_riesgo_hoy.py          mapas de los candidatos; de aquí sale la puntuación del r10
mapas_hoy_manana.py       el producto: r10 en escala absoluta + cifra del día + percentil, PNG y JSON
gh_estado.py push          guarda el estado; publicado/ se sube a git
```

La corrida completa tarda alrededor de una hora y cuarto; casi todo es la
espera de la descarga de ERA5-Land del CDS y la previsión IFS nodo a nodo.
Necesita dos secretos en el repositorio: `CDSAPI_KEY` (Copernicus) y
`FIRMS_MAP_KEY` (NASA FIRMS, para los focos de la última semana).

## Los dos mapas y la cifra del día

La cadena calcula para cada celda una puntuación del r10 y la dibuja de dos
formas.

- Por percentil del día: el 2 % más alto en EXTREMO. Indica dónde mirar, pero
  siempre hay un primero: un día de febrero sale con tanto rojo como el 15 de
  agosto.
- En escala absoluta: cortes fijos en la puntuación (MODERADO ≥ 0.0060,
  ALTO ≥ 0.1125, EXTREMO ≥ 0.4058). La cantidad de rojo cambia con el riesgo del
  día. En EXTREMO ardió una celda-día de cada 760, 13 veces la media.

Junto al mapa absoluto va la cifra del día: el porcentaje de España en EXTREMO
y su posición entre los 250 días de referencia.

Los cortes y la referencia están en `escala_servicio.json` y se calibraron con
los mapas servidos del replay de 2025 y 2026. Los del cubo
(`05_iteracion2/56_calibracion/dos_27_escala_absoluta.py`) no sirven para el mapa
servido: con ellos marcaba entre 2 y 10 veces más EXTREMO que el cubo en el mismo
mes. La referencia no tiene días de diciembre a abril y hay que ampliarla con
los días publicados.

Hasta el 14/09/2026 el script aplicaba un semáforo nacional que retiraba el
nivel EXTREMO los días de puntuación baja. Se quitó porque en verano lo ocultaba
en uno de cada tres días con incendio grande.

## Qué hace cada script

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `gh_reanalisis.py` | Descarga ERA5-Land del mes en curso de forma incremental y lo agrega a diario | Meteorología observada hasta D−7 |
| `riesgo_hoy.py` | Construye las 46 variables en los nodos (reanálisis, IFS, climatologías, FIRMS), puntúa el modelo de producción sobre las 498,530 celdas y dibuja el mapa | El mapa de referencia |
| `dos_riesgo_hoy.py` | Lo mismo con los modelos de etiqueta EFFIS: único, r10, pareja y dónde. Con `--pasada <fecha>` repite un día pasado | La puntuación del r10 de cada día |
| `mapas_hoy_manana.py` | Lee la puntuación del r10, aplica la escala absoluta y el percentil del día, calcula la cifra del día y escribe el PNG y el JSON que se publican | El producto final |
| `escala_servicio.json` | Cortes de la escala absoluta y serie de referencia de la cifra del día | Lo lee `mapas_hoy_manana.py` |
| `gh_estado.py` | Baja y sube el estado de la cadena a un Release de GitHub | Persistencia entre corridas |
| `gh_exportar_estado.py` | Exporta desde el portátil lo que la cadena necesita del cubo y del disco externo (capas estáticas, climatologías, modelos) | Se ejecutó una vez, y cada vez que cambió un modelo |
| `gh_mensual_rapido.py` | Cachés mensuales de vegetación, LST e historial EGIF para los meses fuera de verano | Ampliar la cadena a todo el año |
| `capa_base.py` | Provincias, comunidades y ciudades sobre los mapas | Dibujo |
| `capa_verdad.py` | Perímetros EFFIS y focos FIRMS del día sobre los mapas | Dibujo |
| `redibujar.py` | Regenera un mapa ya publicado con el formato actual sin recalcular | Figuras |
| `dos_07_mapa_hoy.py` | Primer prototipo del mapa DÓNDE × CUÁNDO junto al de producción | Antecedente de `dos_riesgo_hoy.py` |
| `puntuar_effis.py`, `dos_15_veredicto_miteco.py`, `dos_16_juez_estaciones.py` | Puntuaban cada día los mapas de la temporada 2026 contra los perímetros EFFIS, los partes del MITECO y el ranking por estación de producción | Seguimiento durante el desarrollo. La evaluación de la memoria es el replay de `06_comparacion/`; estos scripts ya no forman parte de la cadena |

## Ejecutar con la muestra

Esta es la parte del repositorio que corre con `muestras/`, que es lo que hace
también GitHub Actions (allí tampoco está el cubo de 29 GB):

```bash
source entorno.sh
python 07_produccion/riesgo_hoy.py
python 07_produccion/dos_riesgo_hoy.py
python 07_produccion/mapas_hoy_manana.py
```

Para un día pasado del que se tenga la pasada IFS guardada,
`dos_riesgo_hoy.py --pasada 2026-08-20` y después
`mapas_hoy_manana.py --fecha 2026-08-20`.
