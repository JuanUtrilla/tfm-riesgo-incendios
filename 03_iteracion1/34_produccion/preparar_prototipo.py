#!/usr/bin/env python3
"""
Precálculo por estación AEMET para el prototipo en tiempo real (se ejecuta 1 vez).

Para cada estación del colector horario 2026:
1. Celda IberFire (EPSG:3035) — se descartan las fuera de la malla (Canarias).
2. Estáticas de la celda (elevación…CLC 2018, popdens 2020) — mismas defs que el pipeline.
3. Climatología FWI-propio 2008-2014 por mes: se corre `fwi_canadiense` sobre la
   meteo del CUBO en esa celda (spin-up desde 2007) y se guardan los valores por
   mes → el percentil en tiempo real compara nuestro-FWI con nuestro-FWI (el
   sesgo de implementación se cancela).
4. Climatología mensual de vegetación 2020-2024 (ndvi, lai, swi010, lst) — proxy
   de las features no observables en tiempo real.
5. Autorregresivas EGIF (hasta 2020, congeladas — caveat documentado).

Salida: prototipo/estaciones_prototipo.parquet (1 fila/estación, listas por mes
serializadas) + prototipo/clim_fwi/<idema>.npz (climatología FWI por mes).
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from fwi_canadiense import calcular_fwi_serie

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
DB_COLECTOR = "/home/charredgem/Desktop/Master/aemet_horario_verano2026/data/aemet_horario_verano2026.db"
DIR_OUT = f"{DIR}/prototipo"


def main():
    os.makedirs(f"{DIR_OUT}/clim_fwi", exist_ok=True)

    # --- estaciones del colector con coordenadas (del inventario AEMET en la BD histórica) ---
    con_h = sqlite3.connect(f"{DIR}/aemet_historico.db")
    inv = pd.read_sql("SELECT idema, nombre, latitud, longitud, altitud "
                      "FROM estaciones", con_h)
    con_c = sqlite3.connect(DB_COLECTOR)
    activas = pd.read_sql("SELECT DISTINCT idema FROM observacion_horaria", con_c)
    est = inv.merge(activas, on="idema")
    print(f"Estaciones del colector con coordenadas: {len(est)}")

    def dms_a_dec(v):
        # AEMET usa formato '404942N' (GGMMSS + hemisferio)
        if isinstance(v, (int, float)):
            return float(v)
        g, m, s = int(v[:2]), int(v[2:4]), int(v[4:6])
        dec = g + m / 60 + s / 3600
        return -dec if v[-1] in "WS" else dec

    est["lat"] = est["latitud"].map(dms_a_dec)
    est["lon"] = est["longitud"].map(dms_a_dec)

    ds = xr.open_dataset(f"{DIR}/iberfire/IberFire.nc", decode_timedelta=False)
    xs, ys = ds["x"].values, ds["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X3, Y3 = tr.transform(est["lon"].values, est["lat"].values)
    est["ix"] = np.rint((X3 - xs[0]) / (xs[1] - xs[0])).astype(int)
    est["iy"] = np.rint((Y3 - ys[0]) / (ys[1] - ys[0])).astype(int)
    dentro = ((est.ix >= 0) & (est.ix < len(xs)) & (est.iy >= 0) & (est.iy < len(ys)))
    est = est[dentro].copy()
    es_esp = ds["is_spain"].values.astype(bool)
    est = est[es_esp[est.iy, est.ix]].reset_index(drop=True)
    est["x3035"], est["y3035"] = xs[est.ix], ys[est.iy]
    print(f"Dentro de la malla peninsular: {len(est)} (Canarias/fuera descartadas)")

    # --- estáticas de la celda ---
    for k, v in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                 "rugosidad": "roughness_mean", "dist_carreteras": "dist_to_roads_mean",
                 "dist_rios": "dist_to_waterways_mean", "popdens": "popdens_2020",
                 "ccaa": "AutonomousCommunities"}.items():
        est[k] = ds[v].values[est.iy, est.ix]
    for k, suf in {"clc_bosque": "forest_proportion", "clc_matorral": "scrub_proportion",
                   "clc_agricola": "agricultural_proportion",
                   "clc_artificial": "artificial_proportion",
                   "clc_abierto": "open_space_proportion",
                   "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
        est[k] = ds[f"CLC_2018_{suf}"].values[est.iy, est.ix]

    # --- índices temporales del cubo ---
    tiempos = ds["time"].values.astype("datetime64[D]")
    meses_t = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    anios_t = tiempos.astype("datetime64[Y]").astype(int) + 1970
    m_clim = (anios_t >= 2008) & (anios_t <= 2014)          # FWI climatología
    m_spin = (anios_t >= 2007) & (anios_t <= 2014)          # con spin-up 2007-12
    idx_spin = np.where(m_spin)[0]
    m_veg = (anios_t >= 2020) & (anios_t <= 2024)           # vegetación reciente

    # --- bucle por estación (celda): FWI-propio 2008-14 + veg mensual + EGIF ---
    egif = pd.read_csv(f"{DIR}/egif_civio.csv", usecols=["fecha", "lat", "lng"]).dropna()
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna()
    egif = egif[egif["fecha"].dt.year >= 2008]
    ex, ey = tr.transform(egif["lng"].values, egif["lat"].values)
    e_mes = egif["fecha"].dt.month.values

    veg_cols, auto_cols = [], []
    for n, f in enumerate(est.itertuples()):
        iy, ix = int(f.iy), int(f.ix)
        # FWI propio sobre meteo del cubo (spin-up 2007-12→, guardamos 2008-14)
        t = ds["t2m_max"].isel(y=iy, x=ix, time=idx_spin).values
        h = ds["RH_min"].isel(y=iy, x=ix, time=idx_spin).values
        w = ds["wind_speed_max"].isel(y=iy, x=ix, time=idx_spin).values * 3.6
        p = ds["total_precipitation_mean"].isel(y=iy, x=ix, time=idx_spin).values
        fwi = calcular_fwi_serie(t, h, w, p, meses_t[m_spin])["fwi"]
        m_ok = m_clim[m_spin]
        np.savez_compressed(
            f"{DIR_OUT}/clim_fwi/{f.idema}.npz",
            **{f"m{m}": np.sort(fwi[m_ok & (meses_t[m_spin] == m)
                                     & ~np.isnan(fwi)])
               for m in range(1, 13)})

        # vegetación: media mensual 2020-2024
        veg = {}
        for var, col in [("NDVI", "ndvi"), ("LAI", "lai"),
                         ("SWI_010", "swi010"), ("LST", "lst")]:
            s = ds[var].isel(y=iy, x=ix, time=np.where(m_veg)[0]).values
            for m in range(1, 13):
                veg[f"{col}_m{m}"] = np.nanmean(s[meses_t[m_veg] == m])
        veg_cols.append(veg)

        # autorregresivas EGIF (congeladas a fin de 2020)
        dist = np.hypot(ex - f.x3035, ey - f.y3035)
        a = {"n_fuegos_1km_hist": int((dist <= 1500).sum()),
             "n_fuegos_10km_365d": int(((dist <= 10000)
                 & (egif["fecha"].dt.year.values == 2020)).sum())}
        for m in range(1, 13):
            a[f"n_mismomes_m{m}"] = int(((dist <= 10000) & (e_mes == m)).sum())
        auto_cols.append(a)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(est)}", flush=True)

    est = pd.concat([est.reset_index(drop=True),
                     pd.DataFrame(veg_cols), pd.DataFrame(auto_cols)], axis=1)
    est.drop(columns=["latitud", "longitud"]).to_parquet(
        f"{DIR_OUT}/estaciones_prototipo.parquet", index=False)
    print(f"\nGuardado {DIR_OUT}/estaciones_prototipo.parquet "
          f"({len(est)} estaciones × {len(est.columns)} cols) + clim_fwi/*.npz")


if __name__ == "__main__":
    main()
