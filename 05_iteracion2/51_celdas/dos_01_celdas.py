#!/usr/bin/env python3
"""
Dos modelos, paso 1: la tabla por celda (498.530 celdas peninsulares).

No toca producción. Lee el cubo y el EGIF; escribe en expansión.

Por qué
-------
El modelo dónde (susceptibilidad) se entrena sobre celdas, no sobre muestras
celda-día: una fila por celda, las estáticas como features y la densidad
histórica de fuego como etiqueta. Esta tabla es su dataset, y también lo que
necesita la fase 1 para medir la autocorrelación espacial y la cota de lo que
explican las estáticas solas.

Features por celda (todas servibles: son capas fijas del cubo, las mismas que
usa producción hoy):
  elevacion, pendiente, rugosidad, dist_carreteras, dist_rios, dist_ferrocarril,
  popdens_2020 (la última del cubo; producción la sirve congelada igual),
  clc_* (CLC_2018, las 6 proporciones + arable + urbano), aspect_1..8,
  is_natura2000, ccaa, lat, lon, x3035, y3035, bloque_100km, idx_nodo (ERA5).
  + climatología meteo de la celda (FWI medio jun-sep 2008-2014, tmax, hr_min,
    prec anual): el "clima", que es estático y servible.

Etiquetas / densidades (separadas en el tiempo para no hacer trampa):
  egif_0814          incendios EGIF ≥1 ha en la celda 2008-2014 (previos al
                     dataset: la única densidad histórica usable como feature)
  egif_0814_10km     ídem a <10 km (suavizado)
  egif_1518 / egif_19 / egif_20   incendios EGIF en la celda por periodo de split
  egif_1520_10km     a <10 km, 2015-2020 (etiqueta suavizada)
  effis_1520, effis_2124   días con is_fire (EFFIS ≥5 ha) por periodo
  effis_2124_10km    a <10 km (la verdad terreno operativa, fuera del EGIF)

Salida: EXPANSION/dataset/celdas.parquet
"""

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import uniform_filter

import config
import config_expansion as ce

SALIDA = f"{ce.DATASET}/celdas.parquet"
R10 = 10          # radio en celdas (≈10 km) para las densidades suavizadas
ANIOS_CLIM = (2008, 2014)


def suaviza_10km(campo):
    """Suma en una ventana 21×21 (≈10 km de radio). La ventana es cuadrada
    en vez de circular: para una densidad da igual y es 50× más rápida."""
    return uniform_filter(campo.astype(np.float32), size=2 * R10 + 1,
                          mode="constant") * (2 * R10 + 1) ** 2


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    esp = ds["is_spain"].values.astype(bool)
    iy, ix = np.where(esp)
    xs, ys = ds["x"].values, ds["y"].values
    print(f"celdas España: {len(iy):,}")

    out = {"iy": iy, "ix": ix, "x3035": xs[ix], "y3035": ys[iy]}
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    out["lon"], out["lat"] = tr.transform(out["x3035"], out["y3035"])
    out["bloque_100km"] = (np.floor(out["x3035"] / 1e5) * 1000
                           + np.floor(out["y3035"] / 1e5)).astype(int)
    n = np.load(config.NODOS)
    out["idx_nodo"] = n["idx_nodo"][iy, ix]

    est = {"elevacion": "elevation_mean", "pendiente": "slope_mean",
           "rugosidad": "roughness_mean", "dist_carreteras": "dist_to_roads_mean",
           "dist_rios": "dist_to_waterways_mean",
           "dist_ferrocarril": "dist_to_railways_mean",
           "elev_std": "elevation_stdev", "popdens": "popdens_2020",
           "is_natura2000": "is_natura2000", "ccaa": "AutonomousCommunities",
           "clc_bosque": "CLC_2018_forest_proportion",
           "clc_matorral": "CLC_2018_scrub_proportion",
           "clc_agricola": "CLC_2018_agricultural_proportion",
           "clc_artificial": "CLC_2018_artificial_proportion",
           "clc_abierto": "CLC_2018_open_space_proportion",
           "clc_agric_hetero": "CLC_2018_heterogeneous_agriculture_proportion",
           "clc_arable": "CLC_2018_arable_land_proportion",
           "clc_urbano": "CLC_2018_urban_fabric_proportion",
           "clc_perm": "CLC_2018_permanent_crops_proportion"}
    for k, v in est.items():
        out[k] = ds[v].values[iy, ix]
    for a in range(1, 9):
        out[f"aspect_{a}"] = ds[f"aspect_{a}"].values[iy, ix]
    # popdens de 2015 también, para ver cuánto cambia (producción sirve 2020)
    out["popdens_2015"] = ds["popdens_2015"].values[iy, ix]

    # --- clima de la celda: medias de verano 2008-2014 (estático y servible) ---
    print("  clima de verano 2008-2014 (lectura por bloques)...", flush=True)
    t = ds["time"].values.astype("datetime64[D]")
    anio = t.astype("datetime64[Y]").astype(int) + 1970
    mes = (t.astype("datetime64[M]").astype(int) % 12) + 1
    k_ver = np.where((anio >= ANIOS_CLIM[0]) & (anio <= ANIOS_CLIM[1])
                     & (mes >= 6) & (mes <= 9))[0]
    k_anio = np.where((anio >= ANIOS_CLIM[0]) & (anio <= ANIOS_CLIM[1]))[0]
    clim = {k: np.full(esp.shape, np.nan, np.float32) for k in
            ("fwi_clim_verano", "fwi_p90_verano", "tmax_clim_verano",
             "hrmin_clim_verano", "prec_clim_anual", "ndvi_clim_verano")}
    for by in range(0, esp.shape[0], 77):
        for bx in range(0, esp.shape[1], 99):
            sy, sx = slice(by, by + 77), slice(bx, bx + 99)
            if not esp[sy, sx].any():
                continue
            sub = ds[["FWI", "t2m_max", "RH_min", "total_precipitation_mean",
                      "NDVI"]].isel(y=sy, x=sx)
            fw = sub["FWI"].isel(time=k_ver).values
            clim["fwi_clim_verano"][sy, sx] = np.nanmean(fw, 0)
            clim["fwi_p90_verano"][sy, sx] = np.nanpercentile(fw, 90, axis=0)
            clim["tmax_clim_verano"][sy, sx] = np.nanmean(
                sub["t2m_max"].isel(time=k_ver).values, 0)
            clim["hrmin_clim_verano"][sy, sx] = np.nanmean(
                sub["RH_min"].isel(time=k_ver).values, 0)
            clim["ndvi_clim_verano"][sy, sx] = np.nanmean(
                sub["NDVI"].isel(time=k_ver).values, 0)
            pr = sub["total_precipitation_mean"].isel(time=k_anio).values
            clim["prec_clim_anual"][sy, sx] = np.nansum(pr, 0) / (
                ANIOS_CLIM[1] - ANIOS_CLIM[0] + 1)
        print(f"    fila de bloques {by//77+1}/12", flush=True)
    for k, v in clim.items():
        out[k] = v[iy, ix]

    # --- EGIF por celda y periodo -------------------------------------------
    e = pd.read_csv(config.EGIF, usecols=["fecha", "lat", "lng", "superficie"])
    e["fecha"] = pd.to_datetime(e["fecha"], errors="coerce")
    e = e.dropna(subset=["fecha", "lat", "lng"])
    e["anio"] = e["fecha"].dt.year
    e = e[(e["anio"] >= 2008) & (e["anio"] <= 2020)]
    tr2 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X, Y = tr2.transform(e["lng"].values, e["lat"].values)
    jx = np.rint((X - xs[0]) / (xs[1] - xs[0])).astype(int)
    jy = np.rint((Y - ys[0]) / (ys[1] - ys[0])).astype(int)
    ok = (jx >= 0) & (jx < len(xs)) & (jy >= 0) & (jy < len(ys))
    e, jx, jy = e[ok], jx[ok], jy[ok]
    ok = esp[jy, jx]
    e, jx, jy = e[ok], jx[ok], jy[ok]
    print(f"  EGIF 2008-2020 en malla: {len(e):,}")

    def campo(mask):
        c = np.zeros(esp.shape, np.float32)
        np.add.at(c, (jy[mask], jx[mask]), 1)
        return c
    a = e["anio"].values
    c0814 = campo((a >= 2008) & (a <= 2014))
    c1518 = campo((a >= 2015) & (a <= 2018))
    c19 = campo(a == 2019)
    c20 = campo(a == 2020)
    c1520 = c1518 + c19 + c20
    out["egif_0814"] = c0814[iy, ix]
    out["egif_0814_10km"] = suaviza_10km(c0814)[iy, ix]
    out["egif_1518"] = c1518[iy, ix]
    out["egif_19"] = c19[iy, ix]
    out["egif_20"] = c20[iy, ix]
    out["egif_1520"] = c1520[iy, ix]
    out["egif_1520_10km"] = suaviza_10km(c1520)[iy, ix]
    out["egif_1518_10km"] = suaviza_10km(c1518)[iy, ix]

    # --- EFFIS (is_fire del cubo) por celda y periodo ---------------------
    z = np.load(f"{ce.DATASET}/cubo_etiquetas.npz")
    an = list(z["anios"])
    nfa = z["n_fuego_anio"]
    def per(a0, a1):
        return nfa[an.index(a0):an.index(a1) + 1].sum(0).astype(np.float32)
    f0814, f1520, f2124 = per(2008, 2014), per(2015, 2020), per(2021, 2024)
    out["effis_0814"] = f0814[iy, ix]
    out["effis_1520"] = f1520[iy, ix]
    out["effis_2124"] = f2124[iy, ix]
    out["effis_1520_10km"] = suaviza_10km(f1520 > 0)[iy, ix]
    out["effis_2124_10km"] = suaviza_10km(f2124 > 0)[iy, ix]
    out["effis_2124_bin"] = (f2124[iy, ix] > 0).astype(np.int8)

    df = pd.DataFrame(out)
    df.to_parquet(SALIDA, index=False)
    print(f"\nguardado {SALIDA}: {df.shape}")
    print(df[["egif_0814", "egif_1520", "effis_1520", "effis_2124"]]
          .describe().T.to_string())
    print("\ncelda con ≥1 EGIF 2015-20: "
          f"{(df.egif_1520 > 0).mean()*100:.2f} % · "
          f"≥1 EFFIS 2021-24: {(df.effis_2124 > 0).mean()*100:.2f} %")


if __name__ == "__main__":
    main()
