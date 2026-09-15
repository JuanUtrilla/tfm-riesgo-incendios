#!/usr/bin/env python3
"""
Pipeline de malla, módulo 1: nodos ERA5-Land y mapeo celda → nodo.

No toca producción. Todo el pipeline nuevo lleva prefijo `malla_` y escribe en
`malla_data/`. El sistema que corre (`tiempo_real.py`, `mapa_riesgo_hoy.py`)
queda intacto: la serie de previsiones selladas por commit no se parte.

Qué hace y por qué
------------------
Producción hoy calcula las 19 features meteo por estación AEMET (~690 puntos
irregulares, 16 % sin viento) y las interpola a la malla de 1 km con IDW k=8.
Ese paso nunca se validó y es el eslabón frágil.

Aquí se sustituye por algo sin interpolación: la meteo se calcula en los nodos
de la malla nativa de ERA5-Land (0,1° ≈ 9 km) y cada celda de 1 km toma el
valor de su nodo. No hay IDW, no hay huecos y no depende de que una
estación concreta reporte.

Que esto no pierde nada está medido: la meteo del cubo nunca fue de 1 km
(ERA5-Land es de ~9 km reescalado), y lo genuinamente fino (terreno, usos del
suelo) no se toca.

Decisiones
----------
· Rejilla de 0,1° alineada con la de ERA5-Land (múltiplos exactos de 0,1),
  no una rejilla propia. Así el nodo que se pide a la API es el que existe y
  no hay una interpolación encubierta en el proveedor.
· Asignación por redondeo al nodo más cercano en lat/lon, no por vecino más
  cercano en metros. Es lo mismo a esta escala y es reproducible sin árboles.
· Se conservan solo los nodos con al menos una celda peninsular (`is_spain`):
  no se descarga meteo de mar ni de Francia.

Salida: malla_data/nodos.npz
    nodo_lat, nodo_lon  (n_nodos,)   coordenadas a pedir a la API
    idx_nodo            (920, 1188)  índice de nodo por celda, −1 fuera
    n_celdas            (n_nodos,)   celdas peninsulares que cuelgan del nodo

Uso: python malla_01_nodos.py
"""

import os

import numpy as np
import xarray as xr
from pyproj import Transformer

import config

SALIDA = config.SALIDA
PASO = config.PASO


def main():
    os.makedirs(SALIDA, exist_ok=True)
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    xs, ys = ds["x"].values, ds["y"].values
    es_esp = ds["is_spain"].values.astype(bool)
    ny, nx = es_esp.shape
    ds.close()

    # centros de celda EPSG:3035 → lat/lon
    gx, gy = np.meshgrid(xs, ys)
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(gx[es_esp], gy[es_esp])

    # snap a la rejilla de 0,1° (múltiplos exactos, como ERA5-Land)
    ilat = np.rint(lat / PASO).astype(np.int32)
    ilon = np.rint(lon / PASO).astype(np.int32)
    clave = ilat.astype(np.int64) * 100_000 + ilon.astype(np.int64)
    _, inv, cuenta = np.unique(clave, return_inverse=True, return_counts=True)

    # coordenadas de cada nodo (desde la clave, no promediando celdas)
    uniq = np.unique(clave)
    nodo_lat = (uniq // 100_000).astype(np.float64) * PASO
    nodo_lon = (uniq % 100_000).astype(np.float64) * PASO
    # el módulo pierde el signo de lon negativas: se reconstruye desde ilon
    orden = {int(k): i for i, k in enumerate(uniq)}
    lat_r = np.zeros(len(uniq)); lon_r = np.zeros(len(uniq))
    for a, o, k in zip(ilat, ilon, clave):
        i = orden[int(k)]
        lat_r[i] = a * PASO
        lon_r[i] = o * PASO
    nodo_lat, nodo_lon = lat_r, lon_r

    idx = np.full((ny, nx), -1, dtype=np.int32)
    idx[es_esp] = inv.astype(np.int32)

    np.savez_compressed(f"{SALIDA}/nodos.npz", nodo_lat=nodo_lat,
                        nodo_lon=nodo_lon, idx_nodo=idx, n_celdas=cuenta)

    print(f"celdas peninsulares : {int(es_esp.sum()):,}")
    print(f"nodos ERA5-Land     : {len(nodo_lat):,}")
    print(f"celdas por nodo     : mediana {np.median(cuenta):.0f} · "
          f"min {cuenta.min()} · max {cuenta.max()}")
    print(f"extensión           : lat {nodo_lat.min():.1f}–{nodo_lat.max():.1f} · "
          f"lon {nodo_lon.min():.1f}–{nodo_lon.max():.1f}")
    print(f"peticiones a la API (10 nodos/petición): {int(np.ceil(len(nodo_lat)/10)):,}")
    print(f"\nGuardado: {SALIDA}/nodos.npz")

    # comprobaciones que deben pasar
    assert idx[es_esp].min() >= 0, "hay celdas peninsulares sin nodo"
    assert (idx[~es_esp] == -1).all(), "hay nodos asignados fuera de la península"
    assert cuenta.sum() == es_esp.sum(), "las celdas por nodo no suman el total"
    print("comprobaciones OK")


if __name__ == "__main__":
    main()
