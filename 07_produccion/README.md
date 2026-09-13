# 7 · Producción: el mapa de cada mañana

La cadena corre en GitHub Actions desde este repositorio
(`.github/workflows/mapa_diario.yml`), una vez al día. Publica en `publicado/`
el mapa de hoy y el de mañana con el modelo elegido, el r10, y al lado el
modelo de la primera iteración sobre la misma malla, que es la referencia con
la que se compara. El estado que hace falta entre corridas (el reanálisis
acumulado y los mapas de los últimos días) se guarda en el Release `estado`
del repositorio, porque no cabe en git.

Durante la temporada de 2026 la misma cadena corrió en un repositorio aparte
(`tfm-fuego-malla`) y publicaba, además, los mapas de los otros candidatos.
Aquí queda su código completo, incluidos los scripts que ya no se ejecutan.

## Cómo se genera un día

```
gh_estado.py pull          baja el estado de la corrida anterior (Release)
gh_reanalisis.py           ERA5-Land del mes en curso, incremental, hasta D-7 (CDS)
malla_02b_ifs.py           previsión IFS en los 5.605 nodos, de D-6 a D+1 (Open-Meteo)
riesgo_hoy.py              mapa de referencia: el modelo de producción sobre la malla
dos_riesgo_hoy.py          mapas de los candidatos; de aquí sale la puntuación del r10
mapa_r10_semaforo.py       el producto: r10 en escala absoluta + semáforo, PNG y JSON
gh_estado.py push          guarda el estado; publicado/ se sube a git
```

La corrida completa tarda alrededor de una hora y cuarto; casi todo es la
espera de la descarga de ERA5-Land del CDS y la previsión IFS nodo a nodo.
Necesita dos secretos en el repositorio: `CDSAPI_KEY` (Copernicus) y
`FIRMS_MAP_KEY` (NASA FIRMS, para los focos de la última semana).

## El semáforo y la escala absoluta

La cadena calcula para cada celda una puntuación del r10 y, para dibujarla, hay
dos opciones. La primera es pintar por percentil del día: el 2 % más alto en
rojo. Así se sirvió durante la temporada, y tiene un problema: el ranking
siempre tiene un primero, y un martes de febrero sale con tanto rojo como el
15 de agosto.

La segunda, que es la que publica `mapa_r10_semaforo.py`, usa dos piezas
calibradas sobre los diez años del cubo (`05_iteracion2/56_calibracion/`):

- Una escala fija en la puntuación del modelo. Los cortes MODERADO, ALTO y
  EXTREMO se leyeron de la climatología 2015-2024 (`dos_27_escala_absoluta.py`)
  de forma que EXTREMO sea el 2 % de la historia, no del día. En esa banda ardió
  una celda de cada 800, 36 veces la tasa media.
- Un semáforo nacional. Se toma el percentil 98 de la puntuación del día en
  España, se pasa por la calibración del aviso (`dos_26_calibra_si.py`) y se
  obtiene una probabilidad de «día con incendio grande». Si no llega al umbral,
  el día no lleva nivel EXTREMO: esas celdas se pintan como ALTO y el mapa
  queda apagado. Con esta regla, en invierno y primavera el 44 % de los días se
  quedan sin rojo y se pierde el 6,2 % de los días grandes. En verano el
  semáforo casi siempre está encendido y manda el mapa.

Las constantes están copiadas en el script con el JSON del que salen.

## Qué hace cada script

| Script | Qué hace | Para qué se usa |
|---|---|---|
| `gh_reanalisis.py` | Descarga ERA5-Land del mes en curso de forma incremental y lo agrega a diario | Meteorología observada hasta D−7 |
| `riesgo_hoy.py` | Construye las 46 variables en los nodos (reanálisis, IFS, climatologías, FIRMS), puntúa el modelo de producción sobre las 498.530 celdas y dibuja el mapa | El mapa de referencia |
| `dos_riesgo_hoy.py` | Lo mismo con los modelos de etiqueta EFFIS: único, r10, pareja y dónde. Con `--pasada <fecha>` repite un día pasado | La puntuación del r10 de cada día |
| `mapa_r10_semaforo.py` | Lee las puntuaciones del día, aplica la escala absoluta y el semáforo, y escribe el PNG y el JSON que se publican | El producto final |
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
python 07_produccion/mapa_r10_semaforo.py
```

Para un día pasado del que se tenga la pasada IFS guardada,
`dos_riesgo_hoy.py --pasada 2026-08-20` y después
`mapa_r10_semaforo.py --fecha 2026-08-20`.
