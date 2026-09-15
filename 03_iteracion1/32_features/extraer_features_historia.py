#!/usr/bin/env python3
"""
Features de historial de fuego (EGIF), entorno de fuego activo (FIRMS) y rayos
(WGLC) para la tabla maestra del Modelo B.

Entrada:  dataset/muestra_maestra_v1.parquet
Salida:   dataset/features_historia_v1.parquet (id_muestra + features)

Todas las features miran estrictamente hacia atrás (≤ D-1, salvo rayos del
propio día D, que son observables antes de la ignición típica de tarde):

Historial de fuego (EGIF 2008-2020, coords proyectadas a EPSG:3035):
  n_fuegos_1km_90d    incendios a <1,5 km en [D-90, D-1]
  n_fuegos_10km_90d   incendios a <10 km  en [D-90, D-1]
  n_fuegos_10km_365d  incendios a <10 km  en [D-365, D-1]
  n_fuegos_1km_hist   incendios a <1,5 km desde 2008-01-01 hasta D-1
                      (susceptibilidad estructural de la celda; los papers de
                      ocurrencia agregada sitúan el término autorregresivo como
                      driver nº1, arXiv:2508.09896)
  n_fuegos_10km_mismomes_hist  incendios a <10 km en el mismo mes de años
                      anteriores (estacionalidad local de ignición)

Entorno de fuego activo (FIRMS VIIRS 2015+, nunca la detección del día D,
que sería circular con la propia ignición):
  frp_max_50km_7d     FRP máximo a <50 km en [D-7, D-1] (0 si no hay detección:
                      es un cero real, no un missing)
  n_detec_50km_7d     nº de detecciones a <50 km en [D-7, D-1]
  Caveat: para fechas de ene-2015 la ventana está truncada (FIRMS empieza
  2015-01-01); afecta a <1% de filas y se deja tal cual.

Rayos (WGLC/WWLLN, rejilla 0,5° diaria 2010-2023; celda ~50x40 km, se asigna
por vecino más cercano en lat/lon):
  rayos_dia           densidad de strokes (km-2 d-1) en el día D
  rayos_7d            suma [D-7, D-1] (tormentas secas recientes)
"""

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

RUTA_MAESTRA = "/home/charredgem/Desktop/Master/TFM_fuego/dataset/muestra_maestra_v1.parquet"
RUTA_EGIF = "/home/charredgem/Desktop/Master/TFM_fuego/egif_civio.csv"
RUTA_FIRMS = "/home/charredgem/Desktop/Master/TFM_fuego/firms_iberia_2015_2024.parquet"
RUTA_WGLC = "/home/charredgem/Desktop/Master/TFM_fuego/rayos_data/wglc/wglc_timeseries_30m_daily.nc"
RUTA_SALIDA = "/home/charredgem/Desktop/Master/TFM_fuego/dataset/features_historia_v1.parquet"

R_CELDA = 1500.0     # "misma celda" con tolerancia (m)
R_ZONA = 10_000.0    # zona local (m)
R_FIRMS = 50_000.0   # entorno de fuego activo (m)


def dia(vals) -> np.ndarray:
    return np.asarray(vals, dtype="datetime64[D]").astype(int)


def main() -> None:
    df = pd.read_parquet(RUTA_MAESTRA)
    d_muestra = dia(df["fecha"].values)

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    # ---------- EGIF 2008-2020 ----------
    egif = pd.read_csv(RUTA_EGIF, usecols=["fecha", "lat", "lng"])
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna()
    egif = egif[(egif["fecha"].dt.year >= 2008) & (egif["fecha"].dt.year <= 2020)]
    ex, ey = tr.transform(egif["lng"].values, egif["lat"].values)
    e_dia = dia(egif["fecha"].values.astype("datetime64[D]"))
    e_mes = egif["fecha"].dt.month.values
    arbol_egif = cKDTree(np.column_stack([ex, ey]))
    print(f"EGIF 2008-2020 para historial: {len(egif)} incendios", flush=True)

    # ---------- FIRMS ----------
    firms = pd.read_parquet(RUTA_FIRMS, columns=["latitude", "longitude",
                                                 "acq_date", "frp"])
    fx, fy = tr.transform(firms["longitude"].values, firms["latitude"].values)
    f_dia = dia(firms["acq_date"].values.astype("datetime64[D]"))
    f_frp = firms["frp"].values.astype(float)
    arbol_firms = cKDTree(np.column_stack([fx, fy]))

    # ---------- WGLC (slab Iberia en memoria) ----------
    wglc = xr.open_dataset(RUTA_WGLC)
    wglc_ib = wglc["density"].sel(lat=slice(35, 45), lon=slice(-10, 5)).load()
    w_dia0 = dia(wglc["time"].values.astype("datetime64[D]"))[0]
    print("WGLC slab Iberia:", dict(wglc_ib.sizes), flush=True)

    # ---------- bucle por celda única (las muestras comparten celda) ----------
    celdas = df.groupby(["ix", "iy"], sort=False)
    print(f"Celdas únicas: {celdas.ngroups}", flush=True)
    out = []
    for n, ((ix_, iy_), grupo) in enumerate(celdas, 1):
        cx, cy = grupo["x3035"].iloc[0], grupo["y3035"].iloc[0]
        lat_c, lon_c = grupo["lat_celda"].iloc[0], grupo["lon_celda"].iloc[0]

        # vecinos EGIF por radio (una consulta por celda, no por fila)
        vz = arbol_egif.query_ball_point([cx, cy], r=R_ZONA)
        vz_d, vz_m = e_dia[vz], e_mes[vz]
        dist_z = np.hypot(ex[vz] - cx, ey[vz] - cy)
        vc_d = vz_d[dist_z <= R_CELDA]

        vf = arbol_firms.query_ball_point([cx, cy], r=R_FIRMS)
        vf_d, vf_frp = f_dia[vf], f_frp[vf]

        serie_rayos = wglc_ib.sel(lat=lat_c, lon=lon_c, method="nearest").values

        for fila in grupo.itertuples():
            d = d_muestra[fila.Index]
            mes = fila.mes
            anio = fila.anio

            atras_90 = (vz_d >= d - 90) & (vz_d < d)
            atras_365 = (vz_d >= d - 365) & (vz_d < d)
            mismomes = (vz_m == mes) & (vz_d < dia([np.datetime64(f"{anio}-01-01")])[0])

            m_firms = (vf_d >= d - 7) & (vf_d < d)
            frp7 = vf_frp[m_firms]

            t_w = d - w_dia0
            rayos_d = float(serie_rayos[t_w]) if 0 <= t_w < len(serie_rayos) else np.nan
            r7 = serie_rayos[max(0, t_w - 7):t_w] if t_w > 0 else np.array([])
            rayos_7 = float(np.nansum(r7)) if len(r7) else np.nan

            out.append({
                "id_muestra": fila.id_muestra,
                "n_fuegos_1km_90d": int(((vc_d >= d - 90) & (vc_d < d)).sum()),
                "n_fuegos_10km_90d": int(atras_90.sum()),
                "n_fuegos_10km_365d": int(atras_365.sum()),
                "n_fuegos_1km_hist": int((vc_d < d).sum()),
                "n_fuegos_10km_mismomes_hist": int(mismomes.sum()),
                "frp_max_50km_7d": float(frp7.max()) if len(frp7) else 0.0,
                "n_detec_50km_7d": int(m_firms.sum()),
                "rayos_dia": rayos_d,
                "rayos_7d": rayos_7,
            })
        if n % 2000 == 0:
            print(f"  celda {n}/{celdas.ngroups}", flush=True)

    res = pd.DataFrame(out)
    assert len(res) == len(df)
    res.to_parquet(RUTA_SALIDA, index=False)
    print(f"Guardado {RUTA_SALIDA}: {len(res)} filas × {len(res.columns)} cols",
          flush=True)


if __name__ == "__main__":
    main()
