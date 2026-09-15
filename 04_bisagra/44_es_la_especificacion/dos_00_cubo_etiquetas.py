#!/usr/bin/env python3
"""
Dos modelos, paso 0: una pasada por el cubo para medir la etiqueta real.

No toca producción. Lee `IberFire.nc` en solo lectura; escribe en expansión.

Por qué

La fase 1 del encargo (`PROMPT_DOS_MODELOS.md` §3) pide tres números que nadie
ha medido todavía sobre la malla completa:

  · prevalencia real de celda-día con fuego (frente al 25 % del diseño);
  · autocorrelación espacial (cuánto se parecen celdas vecinas) y temporal
    (cuánto persiste el fuego de un día al siguiente);
  · densidad histórica por celda, que es la etiqueta natural del modelo dónde.

El cubo trae `is_fire` (celda quemada EFFIS ≥5 ha, diario, 2007-12 → 2024-12),
que es la misma etiqueta con la que se valida en operación. Así que las
tres cosas salen de una sola pasada por bloques de chunk [521,77,99].

Qué se acumula (todo agregado, para no tener 6,8 GB en RAM):
  · n_fuego_anio[anio, y, x]   días con is_fire por celda y año (uint16)
  · n_fuego_mes[mes, y, x]     ídem por mes del año, todos los años
  · serie_dia[t]               nº de celdas de España con is_fire cada día
  · serie_near[t]              ídem con is_near_fire (el buffer del muestreo)
  · persistencia[k]            nº de pares (celda, t) con fuego en t y en t+k,
                               k = 1..15, sobre toda España
  · fuego_total[y, x]          días con fuego por celda en todo el periodo

Paralelo por bloques (lectura, sin escribir en el cubo). ~3-5 min con 8 procesos.

Salida: EXPANSION/dataset/cubo_etiquetas.npz
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

import config
import config_expansion as ce

CH_Y, CH_X = 77, 99
LAGS = 15
SALIDA = f"{ce.DATASET}/cubo_etiquetas.npz"


def bloque(args):
    by, bx = args
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    sy = slice(by * CH_Y, min((by + 1) * CH_Y, ds.sizes["y"]))
    sx = slice(bx * CH_X, min((bx + 1) * CH_X, ds.sizes["x"]))
    esp = ds["is_spain"].isel(y=sy, x=sx).values.astype(bool)
    if not esp.any():
        ds.close()
        return None
    f = ds["is_fire"].isel(y=sy, x=sx).values.astype(bool)       # (t, y, x)
    nf = ds["is_near_fire"].isel(y=sy, x=sx).values.astype(bool)
    ds.close()
    f &= esp[None]
    nf &= esp[None]
    t = np.arange(f.shape[0])
    anios = ANIOS_T[t]
    meses = MESES_T[t]
    n_anio = np.zeros((len(ANIOS), f.shape[1], f.shape[2]), np.uint16)
    for i, a in enumerate(ANIOS):
        n_anio[i] = f[anios == a].sum(0)
    n_mes = np.zeros((12, f.shape[1], f.shape[2]), np.uint16)
    for m in range(12):
        n_mes[m] = f[meses == m + 1].sum(0)
    serie = f.sum((1, 2)).astype(np.int32)
    serie_near = nf.sum((1, 2)).astype(np.int32)
    pers = np.zeros(LAGS + 1, np.int64)
    pers[0] = f.sum()
    for k in range(1, LAGS + 1):
        pers[k] = (f[:-k] & f[k:]).sum()
    return by, bx, sy, sx, n_anio, n_mes, serie, serie_near, pers


def main():
    global ANIOS, ANIOS_T, MESES_T
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    tiempos = ds["time"].values.astype("datetime64[D]")
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    ds.close()
    ANIOS_T = tiempos.astype("datetime64[Y]").astype(int) + 1970
    MESES_T = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    ANIOS = np.arange(ANIOS_T.min(), ANIOS_T.max() + 1)

    n_anio = np.zeros((len(ANIOS), ny, nx), np.uint16)
    n_mes = np.zeros((12, ny, nx), np.uint16)
    serie = np.zeros(len(tiempos), np.int64)
    serie_near = np.zeros(len(tiempos), np.int64)
    pers = np.zeros(LAGS + 1, np.int64)

    tareas = [(by, bx) for by in range(-(-ny // CH_Y)) for bx in range(-(-nx // CH_X))]
    t0 = time.time()
    with ProcessPoolExecutor(8, initializer=_init,
                             initargs=(ANIOS, ANIOS_T, MESES_T)) as ex:
        for n, r in enumerate(ex.map(bloque, tareas), 1):
            if r is None:
                continue
            by, bx, sy, sx, na, nm, s, sn, p = r
            n_anio[:, sy, sx] = na
            n_mes[:, sy, sx] = nm
            serie += s
            serie_near += sn
            pers += p
            if n % 20 == 0:
                print(f"  {n}/{len(tareas)} bloques · {time.time()-t0:.0f}s",
                      flush=True)

    np.savez_compressed(SALIDA, anios=ANIOS, fechas=tiempos.astype(int),
                        n_fuego_anio=n_anio, n_fuego_mes=n_mes,
                        serie_dia=serie, serie_near=serie_near,
                        persistencia=pers)
    print(f"\nguardado {SALIDA}")
    print(f"  celdas-día con fuego 2008-2024: {serie.sum():,}")
    for a in ANIOS:
        print(f"  {a}: {n_anio[list(ANIOS).index(a)].sum():7,} celdas-día · "
              f"{(n_anio[list(ANIOS).index(a)] > 0).sum():6,} celdas")


def _init(a, at, mt):
    global ANIOS, ANIOS_T, MESES_T
    ANIOS, ANIOS_T, MESES_T = a, at, mt


if __name__ == "__main__":
    main()
