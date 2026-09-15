#!/usr/bin/env python3
"""
Extractor de features del datacubo IberFire para la tabla maestra del Modelo B.

Entrada:  dataset/muestra_maestra_v1.parquet  (de muestrear_dataset.py)
Salida:   dataset/features_cubo_v1.parquet    (id_muestra + ~40 features)
          dataset/_features_parts/*.parquet   (partes por bloque, resumible)

Estrategia de acceso, que es lo que decide el tiempo de ejecución: el NetCDF
está chunked [521,77,99] (tiempo, y, x), así que se recorre la malla por bloques
espaciales de 77×99 celdas, cargando en RAM la serie temporal completa de las
variables dinámicas del bloque
(~2 GB transitorios), y se calculan las features de todas las muestras del bloque
con numpy. Evita 14k×10 lecturas aleatorias por celda.

Features (definición exacta, documentada también en MODELO_B_BITACORA.md):

Del día D (el FWI/meteo del día es información de "predicción a día vista",
disponible operacionalmente vía forecast):
  fwi, t2m_max, t2m_min, rh_min, viento_max, precip_dia, lst, ndvi, lai, swi010,
  es_festivo; vpd_max derivado: es(t2m_max)·(1−rh_min/100), es en kPa (Magnus).

Ventanas hacia atrás excluyendo el día D (t-w .. t-1), anti-leakage N6/V4:
  precip_7d/15d/30d (suma), fwi_med_7d, fwi_max_7d, fwi_med_15d, fwi_med_30d,
  rh_min_med_7d, t2m_max_med_7d, viento_max_med_7d, ndvi_med_30d,
  dias_sin_lluvia (días consecutivos con precip<1 mm contando hacia atrás desde
  D incluido, tope 120).

Normalización local (contribución del TFM):
  fwi_pctl_local  = percentil del FWI del día vs la climatología 2008-2014 del
                    mismo mes en la misma celda (años previos al dataset, sin fuga).
  fwi_anom_sigma  = (fwi − media_clim) / std_clim (misma climatología).

Estáticas por celda (una lectura global vectorizada):
  elevacion, pendiente, rugosidad, dist_carreteras, dist_rios, popdens (del año
  de la fila), proporciones CLC (bosque, matorral, agrícola, artificial, espacios
  abiertos, agric. heterogénea) del corte CLC más cercano sin mirar al futuro:
  CLC_2012 para filas ≤2017, CLC_2018 para ≥2018.

Flags informativos (no son features): is_near_fire_dia (para filtrar negativos
ambiguos en el ensamblado), is_fire_dia.
"""

import os
import time as _t
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAIZ = Path(__file__).resolve().parents[2]
DIR = os.environ.get("TFM_DATOS", str(RAIZ / "datos"))
RUTA_MAESTRA = f"{DIR}/dataset/muestra_maestra_v1.parquet"
RUTA_IBERFIRE = f"{DIR}/iberfire/IberFire.nc"
DIR_PARTES = f"{DIR}/dataset/_features_parts"
RUTA_SALIDA = f"{DIR}/dataset/features_cubo_v1.parquet"

CHUNK_Y, CHUNK_X = 77, 99          # chunking interno del NetCDF
ANIOS_CLIM = (2008, 2014)          # climatología local (previa al dataset 2015-2020)
UMBRAL_LLUVIA_MM = 1.0
TOPE_DIAS_SECOS = 120

VARS_DINAMICAS = ["FWI", "t2m_max", "t2m_min", "RH_min", "wind_speed_max",
                  "total_precipitation_mean", "LST", "NDVI", "LAI", "SWI_010",
                  "is_holiday", "is_near_fire", "is_fire"]


def vpd_kpa(t_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Déficit de presión de vapor (kPa), Magnus sobre T (°C) y HR (%)."""
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    return es * (1.0 - rh_pct / 100.0)


def extraer_estaticas(ds: xr.Dataset, df: pd.DataFrame) -> pd.DataFrame:
    """Features estáticas: lectura global 2D + fancy indexing vectorizado."""
    iy, ix = df["iy"].values, df["ix"].values
    out = {"id_muestra": df["id_muestra"].values}

    mapa = {"elevacion": "elevation_mean", "pendiente": "slope_mean",
            "rugosidad": "roughness_mean", "dist_carreteras": "dist_to_roads_mean",
            "dist_rios": "dist_to_waterways_mean"}
    for nombre, var in mapa.items():
        out[nombre] = ds[var].values[iy, ix]

    # popdens del año de la fila (el cubo trae 2008-2020)
    pop = np.full(len(df), np.nan)
    for anio in sorted(df["anio"].unique()):
        m = (df["anio"] == anio).values
        pop[m] = ds[f"popdens_{anio}"].values[iy[m], ix[m]]
    out["popdens"] = pop

    # CLC del corte más cercano sin mirar al futuro: ≤2017 → 2012; ≥2018 → 2018
    clases = {"clc_bosque": "forest_proportion", "clc_matorral": "scrub_proportion",
              "clc_agricola": "agricultural_proportion",
              "clc_artificial": "artificial_proportion",
              "clc_abierto": "open_space_proportion",
              "clc_agric_hetero": "heterogeneous_agriculture_proportion"}
    m2018 = (df["anio"] >= 2018).values
    for nombre, suf in clases.items():
        v12 = ds[f"CLC_2012_{suf}"].values
        v18 = ds[f"CLC_2018_{suf}"].values
        val = v12[iy, ix].astype(float)
        val[m2018] = v18[iy[m2018], ix[m2018]]
        out[nombre] = val
    return pd.DataFrame(out)


def procesar_bloque(ds, df_b, t_idx, clim_idx_mes):
    """Features dinámicas de todas las muestras de un bloque espacial 77x99."""
    by, bx = df_b["_by"].iloc[0], df_b["_bx"].iloc[0]
    sy = slice(by * CHUNK_Y, min((by + 1) * CHUNK_Y, ds.sizes["y"]))
    sx = slice(bx * CHUNK_X, min((bx + 1) * CHUNK_X, ds.sizes["x"]))
    sub = ds[VARS_DINAMICAS].isel(y=sy, x=sx).load()
    arr = {v: sub[v].values for v in VARS_DINAMICAS}

    filas = []
    for celda, grupo in df_b.groupby(["iy", "ix"], sort=False):
        yr, xr_ = celda[0] - sy.start, celda[1] - sx.start
        fwi_s = arr["FWI"][:, yr, xr_]
        pre_s = arr["total_precipitation_mean"][:, yr, xr_]
        rh_s = arr["RH_min"][:, yr, xr_]
        tmx_s = arr["t2m_max"][:, yr, xr_]
        vto_s = arr["wind_speed_max"][:, yr, xr_]
        ndvi_s = arr["NDVI"][:, yr, xr_]

        # climatología FWI por mes de esta celda (2008-2014), ordenada
        clim_cache = {}
        for fila in grupo.itertuples():
            t = t_idx[fila.id_muestra]
            mes = fila.mes
            if mes not in clim_cache:
                c = fwi_s[clim_idx_mes[mes]]
                c = c[~np.isnan(c)]
                clim_cache[mes] = (np.sort(c), np.nanmean(c), np.nanstd(c))
            csort, cmed, cstd = clim_cache[mes]

            fwi_d = fwi_s[t]
            pctl = (np.searchsorted(csort, fwi_d, side="right") / len(csort) * 100
                    if len(csort) else np.nan)
            anom = (fwi_d - cmed) / cstd if cstd and cstd > 0 else np.nan

            # días sin lluvia: hacia atrás desde D incluido
            secos, k = 0, t
            while k >= 0 and secos < TOPE_DIAS_SECOS:
                p = pre_s[k]
                if np.isnan(p) or p >= UMBRAL_LLUVIA_MM:
                    break
                secos += 1
                k -= 1

            v7, v15, v30 = slice(t - 7, t), slice(t - 15, t), slice(t - 30, t)
            filas.append({
                "id_muestra": fila.id_muestra,
                # día D
                "fwi": fwi_d, "t2m_max": tmx_s[t],
                "t2m_min": arr["t2m_min"][t, yr, xr_], "rh_min": rh_s[t],
                "viento_max": vto_s[t], "precip_dia": pre_s[t],
                "lst": arr["LST"][t, yr, xr_], "ndvi": ndvi_s[t],
                "lai": arr["LAI"][t, yr, xr_], "swi010": arr["SWI_010"][t, yr, xr_],
                "es_festivo": arr["is_holiday"][t, yr, xr_],
                # normalización local
                "fwi_pctl_local": pctl, "fwi_anom_sigma": anom,
                # ventanas (excluyen D)
                "precip_7d": np.nansum(pre_s[v7]), "precip_15d": np.nansum(pre_s[v15]),
                "precip_30d": np.nansum(pre_s[v30]),
                "fwi_med_7d": np.nanmean(fwi_s[v7]), "fwi_max_7d": np.nanmax(fwi_s[v7]),
                "fwi_med_15d": np.nanmean(fwi_s[v15]), "fwi_med_30d": np.nanmean(fwi_s[v30]),
                "rh_min_med_7d": np.nanmean(rh_s[v7]),
                "t2m_max_med_7d": np.nanmean(tmx_s[v7]),
                "viento_max_med_7d": np.nanmean(vto_s[v7]),
                "ndvi_med_30d": np.nanmean(ndvi_s[v30]),
                "dias_sin_lluvia": secos,
                # flags (no features)
                "is_near_fire_dia": arr["is_near_fire"][t, yr, xr_],
                "is_fire_dia": arr["is_fire"][t, yr, xr_],
            })
    return pd.DataFrame(filas)


def main() -> None:
    os.makedirs(DIR_PARTES, exist_ok=True)
    df = pd.read_parquet(RUTA_MAESTRA)
    ds = xr.open_dataset(RUTA_IBERFIRE, decode_timedelta=False)

    # índice temporal: el cubo es diario contiguo desde 2007-12-01 (se verifica)
    tiempos = ds["time"].values.astype("datetime64[D]")
    assert tiempos[0] == np.datetime64("2007-12-01") and \
        (np.diff(tiempos).astype(int) == 1).all(), "eje temporal no contiguo"
    t0 = tiempos[0].astype(int)
    t_of = df["fecha"].values.astype("datetime64[D]").astype(int) - t0
    t_idx = dict(zip(df["id_muestra"].values, t_of))

    # índices de climatología por mes (2008-2014), comunes a todas las celdas
    anios = tiempos.astype("datetime64[Y]").astype(int) + 1970
    meses = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    m_clim = (anios >= ANIOS_CLIM[0]) & (anios <= ANIOS_CLIM[1])
    clim_idx_mes = {m: np.where(m_clim & (meses == m))[0] for m in range(1, 13)}

    # 1) estáticas (rápido, global)
    ruta_est = os.path.join(DIR_PARTES, "estaticas.parquet")
    if not os.path.exists(ruta_est):
        t0_ = _t.time()
        extraer_estaticas(ds, df).to_parquet(ruta_est, index=False)
        print(f"[estáticas] {len(df)} filas en {_t.time()-t0_:.0f}s", flush=True)

    # 2) dinámicas por bloque de chunk (resumible)
    df["_by"], df["_bx"] = df["iy"] // CHUNK_Y, df["ix"] // CHUNK_X
    bloques = df.groupby(["_by", "_bx"])
    print(f"Bloques con muestras: {bloques.ngroups}", flush=True)
    for n, ((by, bx), df_b) in enumerate(bloques, 1):
        ruta = os.path.join(DIR_PARTES, f"bloque_{by:02d}_{bx:02d}.parquet")
        if os.path.exists(ruta):
            continue
        t0_ = _t.time()
        parte = procesar_bloque(ds, df_b, t_idx, clim_idx_mes)
        parte.to_parquet(ruta, index=False)
        print(f"[{n}/{bloques.ngroups}] bloque ({by},{bx}): {len(parte)} filas "
              f"en {_t.time()-t0_:.0f}s", flush=True)

    # 3) consolidar
    partes = [pd.read_parquet(os.path.join(DIR_PARTES, f))
              for f in sorted(os.listdir(DIR_PARTES)) if f.startswith("bloque_")]
    dinamicas = pd.concat(partes, ignore_index=True)
    estaticas = pd.read_parquet(ruta_est)
    final = dinamicas.merge(estaticas, on="id_muestra", validate="1:1")
    assert len(final) == len(df), f"filas: {len(final)} != {len(df)}"
    final.to_parquet(RUTA_SALIDA, index=False)
    print(f"\nGuardado {RUTA_SALIDA}: {len(final)} filas × {len(final.columns)} cols",
          flush=True)
    ds.close()


if __name__ == "__main__":
    main()
