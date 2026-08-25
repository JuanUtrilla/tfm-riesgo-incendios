# 1 · Datos: nueve fuentes, un cubo y una malla

> **Marco común a las dos iteraciones.** Lo que cambia entre la iteración 1 y
> la 2 no son los datos: es qué se toma como positivo, qué como negativo y qué
> como verdad. Por eso este capítulo se lee una sola vez.

## Qué pregunta responde

De dónde sale cada uno de los números que ve el modelo, y cómo se convierte una
fuente heterogénea —un NetCDF de 29 GB, una API con clave, un PDF diario— en
una fila de features para una celda y un día.

## Las fuentes, y para qué sirve cada una

| Carpeta | Fuente | Papel |
|---|---|---|
| `cubo/` | **IberFire**, datacubo diario de la Península (29 GB) | Las 46 features de entrenamiento de las dos iteraciones |
| `era5land/` | **ERA5-Land** (Copernicus/ECMWF), ~9 km | La meteorología: la del cubo al entrenar, y la de los 5.605 nodos al servir |
| `ifs/` | **IFS** de ECMWF vía Open-Meteo | La previsión de D y D+1, donde el reanálisis aún no existe |
| `aemet/` | **AEMET**: observación horaria, climatología diaria | La meteorología de la producción de la iteración 1 |
| `effis/` | **EFFIS** (Copernicus EMS) | Perímetros de superficie quemada: etiqueta de la iteración 2 y juez |
| `miteco/` | **Parte diario del MITECO** | El juez rápido: sale al día siguiente |
| `firms/` | **FIRMS** (VIIRS, NASA) | Focos activos: features de entorno y capa de situación |
| `egif/` | **EGIF** (vía Civio) | Registro oficial de igniciones: etiqueta de la iteración 1 |
| `extra/` | Carreteras, censo ganadero, rayos WGLC | Features humanas y estructurales |
| `comun/` | `config`, `features`, `fwi_canadiense` | Rutas, el orden de las 46 features y el FWI canadiense |

## El orden

```
malla_01_nodos.py        los 5.605 nodos de ERA5-Land y el mapeo celda -> nodo
malla_02_descarga.py     descarga y agregación horaria -> diaria desde CDS
malla_04_climatologia.py climatología del FWI por nodo (7 años)
extraer_features_cubo.py las features estáticas y dinámicas del cubo
malla_02b_ifs.py         la rama de previsión, en los mismos nodos
```

Las descargas de AEMET, FIRMS, EGIF y las fuentes extra son independientes
entre sí y se pueden lanzar en cualquier orden.

## Dos avisos que ahorran horas

**El FWI del cubo son las 13 UTC.** Se verificó antes de construir nada
(`04_bisagra/42_no_es_la_meteo/verificar_hora_fwi.py`): es estable en tres
meses y no es un proxy de extremos. Servir un FWI de otra hora introduce un
factor ~2 respecto al entrenamiento.

**ERA5-Land llueve de más.** Frente a AEMET, +8,00 mm en 30 días (sesgo húmedo
conocido del reanálisis, llovizna difusa). De ahí se propaga todo: menos días
sin lluvia, combustible más húmedo, FWI 10 puntos más bajo. Está medido en
`04_bisagra/42_no_es_la_meteo/experimento_b_era5.py` y es la razón de que
entrenar y servir con la misma fuente importe tanto.

## Ejecutar

Las descargas de ERA5-Land necesitan cuenta en CDS (`CDSAPI_KEY`), FIRMS una
clave de mapa (`FIRMS_MAP_KEY`) y AEMET su API key. El cubo IberFire se apunta
con `TFM_DATOS`. Sin nada de eso, `muestras/` lleva julio de 2026 ya procesado.
