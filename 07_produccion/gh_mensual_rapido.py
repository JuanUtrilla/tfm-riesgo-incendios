#!/usr/bin/env python3
"""
Cachés mensuales (vegetación/LST 2020-24 + EGIF mismo-mes) para los meses que
producción no tenía (abril, mayo, octubre, noviembre), con lecturas CONTIGUAS
del cubo.

`riesgo_hoy.mensual` las calcula con `isel(time=idx)` sobre índices no
contiguos, lo que obliga a netCDF a descomprimir cada chunk de 521 días una
vez por día pedido: más de una hora por mes (medido el 21/08). Por año y mes
el rango es contiguo y tarda ~1 min. Mismas claves y mismo resultado (media
sobre los mismos días, NaN ignorados).

Uso: python gh_mensual_rapido.py 10 5 11 4
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import config
from malla_05_riesgo import suma_caja

ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
t = pd.to_datetime(ds["time"].values)
ny, nx = ds.sizes["y"], ds.sizes["x"]
xs, ys = ds["x"].values, ds["y"].values
tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
eg = pd.read_csv(config.EGIF, usecols=["fecha", "lat", "lng"]).dropna()
eg["fecha"] = pd.to_datetime(eg["fecha"], errors="coerce")
eg = eg.dropna()
eg = eg[eg["fecha"].dt.year >= 2008]
for mes in [int(m) for m in sys.argv[1:]] or (10, 5, 11, 4):
    out = {}
    for var, col in [("NDVI", "ndvi"), ("LAI", "lai"), ("SWI_010", "swi010"), ("LST", "lst")]:
        suma = np.zeros((ny, nx))
        n = np.zeros((ny, nx))
        for a in range(2020, 2025):
            k = np.where((t.year == a) & (t.month == mes))[0]
            v = ds[var].isel(time=slice(k[0], k[-1] + 1)).values
            ok = np.isfinite(v)
            suma += np.where(ok, v, 0).sum(0)
            n += ok.sum(0)
        out[col] = np.where(n > 0, suma / np.maximum(n, 1), np.nan).astype(np.float32)
        print(f"  m{mes} {col}", flush=True)
    e = eg[eg["fecha"].dt.month == mes]
    ex, ey = tr.transform(e["lng"].values, e["lat"].values)
    eix = np.rint((ex - xs[0]) / (xs[1] - xs[0])).astype(int)
    eiy = np.rint((ey - ys[0]) / (ys[1] - ys[0])).astype(int)
    ok = (eix >= 0) & (eix < nx) & (eiy >= 0) & (eiy < ny)
    g = np.zeros((ny, nx))
    np.add.at(g, (eiy[ok], eix[ok]), 1)
    out["n_fuegos_10km_mismomes_hist"] = suma_caja(g, 10)
    np.savez_compressed(config.salida(f"malla_mensual_m{mes}.npz"), **out)
    print(f"guardado m{mes}", flush=True)
