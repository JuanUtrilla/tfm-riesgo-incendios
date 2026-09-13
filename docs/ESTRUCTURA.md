# Cómo está montado este repositorio

Decisiones de construcción, y por qué. Si vas a tocar el repo, lee esto antes.

## 1. El árbol es el orden de lectura de la memoria

No es el orden en que ocurrieron las cosas ni una estructura de software al
uso. Cada carpeta responde a un capítulo, y dentro van los scripts que
produjeron sus números. La consecuencia buena es que un lector encuentra la
evidencia de cada afirmación; la mala es la que se explica en el punto 3.

## 2. El código no se ha reescrito

Los 120 scripts son copias literales de los repositorios donde se ejecutaron,
con su procedencia y su hash en [`PROCEDENCIA.md`](PROCEDENCIA.md). Se valoró
reescribirlos para que el repo luciera mejor y se descartó: una reescritura
obliga o a reejecutar el pipeline entero para volver a generar los resultados
—semanas de cómputo sobre 29 GB de datos, a semanas de entregar— o a publicar
código que no produjo los números que documenta. Sin tests que fijen las
cifras, un refactor que convierta un 0,773 en un 0,771 no se detectaría.

Lo único adaptado son las **rutas**, y solo en los cuatro módulos de
`01_datos/comun/`:

| Módulo | Qué es | Qué se cambió |
|---|---|---|
| `config.py` | rutas y constantes del pipeline de malla | la raíz pasa a ser la del repo; las rutas personales salen a variables de entorno; añade `muestras/` como último recurso |
| `config_expansion.py` | disco externo con los datasets de entrenamiento | nada: ya buscaba por `TFM_USB` |
| `features.py` | las 46 features y su orden | nada |
| `fwi_canadiense.py` | el FWI canadiense (Van Wagner, 1987) | nada |

Ninguna constante del pipeline se ha tocado.

## 3. Los imports son planos, y por eso hace falta `entorno.sh`

Los scripts se importan entre sí con nombres planos (`import config`,
`from dos_05_modelos import FEATS_CUANDO`) porque así estaban escritos. Al
repartirlos por capítulos, Python deja de encontrarlos: `dos_18_ratio.py` vive
en `05_iteracion2/57_ablaciones/` e importa `dos_05_modelos`, que vive en
`55_train/`. Y no es un caso aislado — `config` lo importan 50 scripts y
`dos_05_modelos` otros 8.

Había dos salidas: convertir el repo en un paquete y reescribir los imports de
124 ficheros —que es exactamente la reescritura que se descartó en el punto 2—
o poner las carpetas en `PYTHONPATH`. Se eligió lo segundo:

```bash
source entorno.sh
python 05_iteracion2/57_ablaciones/dos_18_ratio.py
```

`entorno.sh` recorre el repo, mete en `PYTHONPATH` toda carpeta que contenga
`.py` y declara dónde están los datos. El precio es esa línea; el beneficio es
que el código es idéntico al que generó los resultados.

## 4. Dónde están los datos

Ninguna ruta personal vive ya en el código. Tres variables de entorno, todas
con valor por defecto:

| Variable | Para qué | Por defecto |
|---|---|---|
| `TFM_DATOS` | fuentes crudas: cubo IberFire (29 GB), EGIF, FIRMS, rayos WGLC | `./datos` (ignorado por git) |
| `TFM_USB` | disco con los datasets de entrenamiento de la iteración 2 | se busca el USB montado |
| `TFM_SALIDA` | dónde escribir | `./salida` |

La cadena de resolución de `config.entrada()` es: `salida/` propio →
`TFM_DATOS` → **`muestras/`**. Ese último eslabón es el que hace que el repo
arranque recién clonado, sin descargar nada.

## 5. `muestras/`: 48 MB para no descargar 44 GB

Recorte de **julio de 2026**: capas estáticas del cubo, climatología de FWI del
mes, reanálisis ERA5-Land del 20/06 al 31/07 (los diez días de margen son para
las ventanas móviles de 30 días… que por eso mismo quedan incompletas al
principio), los cuatro modelos entrenados, los perímetros EFFIS de julio (259
incendios) y las capas de dibujo.

**Lo que sí se puede ejecutar con esto**: la rama de servicio —construir las
features en los 5.605 nodos, puntuar los modelos sobre las 498.530 celdas y
dibujar los mapas— que es justamente lo que corre en GitHub Actions, que
tampoco tiene el cubo.

**Lo que no**: todo lo que lee el cubo día a día (`dos_00`…`dos_13`, los
entrenamientos, las ablaciones). Necesita los 29 GB de IberFire. La muestra
lleva `cubo_estaticas.nc`, que son solo las capas 2D que no cambian con el
tiempo.

## 6. La cadena diaria

Corre en GitHub Actions desde este repositorio
(`.github/workflows/mapa_diario.yml`), una vez al día, y publica en
`publicado/` el mapa del r10 con el semáforo y el mapa de referencia. El
estado entre corridas (reanálisis acumulado, mapas de los últimos días) vive
en el Release `estado` del repositorio, que `07_produccion/gh_estado.py` baja
al empezar y sube al terminar. Hacen falta dos secretos: `CDSAPI_KEY` y
`FIRMS_MAP_KEY`. Durante la temporada de 2026 la misma cadena corrió en el
repositorio `tfm-fuego-malla`, con más pasos; su código está íntegro en
`07_produccion/`.
