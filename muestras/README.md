# Recorte de julio de 2026

Este directorio ocupa 48 MB y permite que el repositorio funcione recién
clonado, sin bajar los 44 GB de fuentes originales. `config.entrada()` cae aquí
cuando no encuentra el dato ni en `salida/` ni en `TFM_DATOS`.

| Fichero | Qué es | Tamaño |
|---|---|---|
| `cubo_estaticas.nc` | capas 2D del cubo IberFire que no cambian con el tiempo: elevación, pendiente, rugosidad, distancias, población, CLC 2018, máscara de España | 16 MB |
| `era5land_diario.nc` | reanálisis ERA5-Land agregado a diario en los 5,605 nodos, 20/06 → 31/07/2026 | 9.9 MB |
| `effis_ba_season_ES.geojson` | perímetros de EFFIS (*European Forest Fire Information System*) con `FIREDATE` en julio de 2026 (259 incendios) | 5.5 MB |
| `malla_mensual_m7.npz` | caché mensual de julio: vegetación y LST por celda (climatología 2020-24) | 5.2 MB |
| `clim_fwi_nodos.npz` | climatología de FWI por nodo, solo julio (del original de 50 MB con los doce meses) | 4.2 MB |
| `municipios.json` | maestro de municipios, para situar los incidentes del parte del MITECO (Ministerio para la Transición Ecológica y el Reto Demográfico) | 2.8 MB |
| `dos_13_mapa_donde_effis_c.npz` | mapa del DÓNDE precalculado por celda | 1.2 MB |
| `modelos/` | `xgb_v2_prototipo` (producción), `donde_dia_effis` (único), `donde_effis_c` y `cuando` (pareja) | 4.2 MB |
| `estaciones_prototipo.parquet`, `nodos.npz`, `limites.npz`, `mapeo_ifs_a_era5land.npz` | estaciones, malla de nodos, capa base administrativa y mapeo IFS→ERA5-Land | < 0.5 MB |

## Por qué julio de 2026

Es el mes con el que se hizo la primera comparación entre la producción y la
malla (17 días con área quemada de EFFIS). Los números que salgan del recorte
se pueden contrastar con los de la memoria.

## Sus límites

Las ventanas móviles empiezan incompletas. `precip_30d` o `dias_sin_lluvia`
necesitan 30 días de historia; con el reanálisis arrancando el 20/06, los
primeros días de julio salen sesgados. Los números publicados se reproducen
con la segunda quincena.

El recorte no permite entrenar. Los scripts `dos_00`…`dos_13` leen el cubo día
a día (29 GB) y aquí solo están sus capas estáticas.

Tampoco lleva la rama de previsión. El IFS (*Integrated Forecasting System*)
se descarga en vivo de Open-Meteo, sin clave, y para un día de julio ya pasado
lo que aplica es el reanálisis.

## Procedencia y licencias

ERA5-Land y el reanálisis son de Copernicus/ECMWF (licencia abierta, requiere
atribución). Los perímetros son de EFFIS (Copernicus EMS). Las capas estáticas
derivan del datacubo IberFire. Los modelos `.ubj` son producto de este trabajo.
