#!/usr/bin/env python3
"""
Consulta puntual del Modelo B: riesgo de incendio en (lat, lon, fecha) con
explicación SHAP local. Es el germen del prototipo.

Uso: python3 predecir_punto.py <lat> <lon> <AAAA-MM-DD>
Ej.:  python3 predecir_punto.py 40.545 -4.786 2021-08-14   (incendio de Sotalvo,
      año fuera del entrenamiento 2015-2018)

Reconstruye las features desde el cubo IberFire + EGIF + FIRMS + WGLC con las
mismas definiciones del pipeline (ver dataset/DATASET_CARD.md) para una sola
celda, predice con el modelo afinado y muestra las 10 contribuciones SHAP locales.
Rango de fechas servible: de 2011 (ventanas) a 2024-12-31 (fin del cubo).
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer
from scipy.spatial import cKDTree

from entrenar_modelo import (FEATS_METEO, FEATS_VEG, FEATS_ESTAT, FEATS_HIST,
                             FEATS_CAL)

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
FULL = FEATS_METEO + FEATS_VEG + FEATS_ESTAT + FEATS_HIST + FEATS_CAL


def main():
    lat, lon, fecha = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    d = np.datetime64(fecha, "D")
    anio, mes = int(fecha[:4]), int(fecha[5:7])

    ds = xr.open_dataset(f"{DIR}/iberfire/IberFire.nc", decode_timedelta=False)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X3, Y3 = tr.transform(lon, lat)
    xs, ys = ds["x"].values, ds["y"].values
    ix = int(np.rint((X3 - xs[0]) / (xs[1] - xs[0])))
    iy = int(np.rint((Y3 - ys[0]) / (ys[1] - ys[0])))
    tiempos = ds["time"].values.astype("datetime64[D]")
    t = int((d - tiempos[0]).astype(int))
    assert 0 < t < len(tiempos), "fecha fuera del cubo (2007-12→2024-12)"
    if not bool(ds["is_spain"].values[iy, ix]):
        print("⚠️ el punto cae fuera de la máscara peninsular del cubo")

    celda = ds.isel(y=iy, x=ix)
    F = {}

    def serie(var, t0, t1):
        return celda[var].isel(time=slice(t0, t1)).values

    fwi_s = serie("FWI", 0, t + 1)
    pre_s = serie("total_precipitation_mean", t - 120, t + 1)
    F["fwi"] = fwi_s[-1]
    F["t2m_max"] = float(celda["t2m_max"].isel(time=t))
    F["t2m_min"] = float(celda["t2m_min"].isel(time=t))
    F["rh_min"] = float(celda["RH_min"].isel(time=t))
    F["viento_max"] = float(celda["wind_speed_max"].isel(time=t))
    F["precip_dia"] = pre_s[-1]
    F["lst"] = float(celda["LST"].isel(time=t))
    ndvi_s = serie("NDVI", t - 30, t + 1)
    F["ndvi"], F["ndvi_med_30d"] = ndvi_s[-1], np.nanmean(ndvi_s[:-1])
    F["lai"] = float(celda["LAI"].isel(time=t))
    F["swi010"] = float(celda["SWI_010"].isel(time=t))
    F["es_festivo"] = float(celda["is_holiday"].isel(time=t))
    es = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
    F["vpd_max"] = es * (1 - F["rh_min"] / 100)

    F["precip_7d"] = np.nansum(pre_s[-8:-1]); F["precip_15d"] = np.nansum(pre_s[-16:-1])
    F["precip_30d"] = np.nansum(pre_s[-31:-1])
    F["fwi_med_7d"] = np.nanmean(fwi_s[t-7:t]); F["fwi_max_7d"] = np.nanmax(fwi_s[t-7:t])
    F["fwi_med_15d"] = np.nanmean(fwi_s[t-15:t]); F["fwi_med_30d"] = np.nanmean(fwi_s[t-30:t])
    rh_s = serie("RH_min", t - 7, t)
    F["rh_min_med_7d"] = np.nanmean(rh_s)
    F["t2m_max_med_7d"] = np.nanmean(serie("t2m_max", t - 7, t))
    F["viento_max_med_7d"] = np.nanmean(serie("wind_speed_max", t - 7, t))
    secos = 0
    for p in pre_s[:-1][::-1]:
        if np.isnan(p) or p >= 1.0 or secos >= 120:
            break
        secos += 1
    F["dias_sin_lluvia"] = secos + (0 if (np.isnan(pre_s[-1]) or pre_s[-1] >= 1) else 1)

    anios_t = tiempos.astype("datetime64[Y]").astype(int) + 1970
    meses_t = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    m_clim = (anios_t >= 2008) & (anios_t <= 2014) & (meses_t == mes)
    clim = fwi_s[m_clim[:t + 1]] if t + 1 >= m_clim.sum() else celda["FWI"].values[m_clim]
    clim = celda["FWI"].isel(time=np.where(m_clim)[0]).values
    clim = clim[~np.isnan(clim)]
    F["fwi_pctl_local"] = float((clim <= F["fwi"]).mean() * 100)
    F["fwi_anom_sigma"] = float((F["fwi"] - clim.mean()) / clim.std())

    for k, v in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                 "rugosidad": "roughness_mean", "dist_carreteras": "dist_to_roads_mean",
                 "dist_rios": "dist_to_waterways_mean"}.items():
        F[k] = float(ds[v].values[iy, ix])
    F["popdens"] = float(ds[f"popdens_{min(max(anio, 2008), 2020)}"].values[iy, ix])
    corte = 2018 if anio >= 2018 else 2012
    for k, suf in {"clc_bosque": "forest_proportion", "clc_matorral": "scrub_proportion",
                   "clc_agricola": "agricultural_proportion",
                   "clc_artificial": "artificial_proportion",
                   "clc_abierto": "open_space_proportion",
                   "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
        F[k] = float(ds[f"CLC_{corte}_{suf}"].values[iy, ix])
    v = ds["AutonomousCommunities"].values[iy, ix]
    F["ccaa"] = int(v) if np.isfinite(v) else -1
    F["mes"] = mes
    F["dia_anio"] = int((d - np.datetime64(f"{anio}-01-01", "D")).astype(int)) + 1

    # historial EGIF / FIRMS / rayos
    egif = pd.read_csv(f"{DIR}/egif_civio.csv", usecols=["fecha", "lat", "lng"]).dropna()
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna()
    egif = egif[egif["fecha"].dt.year >= 2008]
    ex, ey = tr.transform(egif["lng"].values, egif["lat"].values)
    dist = np.hypot(ex - X3, ey - Y3)
    ed = egif["fecha"].values.astype("datetime64[D]")
    em = egif["fecha"].dt.month.values
    antes = ed < d
    F["n_fuegos_1km_90d"] = int(((dist <= 1500) & antes & (ed >= d - 90)).sum())
    F["n_fuegos_10km_90d"] = int(((dist <= 10000) & antes & (ed >= d - 90)).sum())
    F["n_fuegos_10km_365d"] = int(((dist <= 10000) & antes & (ed >= d - 365)).sum())
    F["n_fuegos_1km_hist"] = int(((dist <= 1500) & antes).sum())
    F["n_fuegos_10km_mismomes_hist"] = int(((dist <= 10000) & (em == mes)
                                            & (egif["fecha"].dt.year.values < anio)).sum())

    firms = pd.read_parquet(f"{DIR}/firms_iberia_2015_2024.parquet",
                            columns=["latitude", "longitude", "acq_date", "frp"])
    fx, fy = tr.transform(firms["longitude"].values, firms["latitude"].values)
    fd = firms["acq_date"].values.astype("datetime64[D]")
    m = (np.hypot(fx - X3, fy - Y3) <= 50000) & (fd >= d - 7) & (fd < d)
    F["frp_max_50km_7d"] = float(firms.loc[m, "frp"].max()) if m.any() else 0.0
    F["n_detec_50km_7d"] = int(m.sum())

    wglc = xr.open_dataset(f"{DIR}/rayos_data/wglc/wglc_timeseries_30m_daily.nc")
    wd0 = wglc["time"].values.astype("datetime64[D]")[0]
    tw = int((d - wd0).astype(int))
    sr = wglc["density"].sel(lat=lat, lon=lon, method="nearest").values
    F["rayos_dia"] = float(sr[tw]) if 0 <= tw < len(sr) else np.nan
    F["rayos_7d"] = float(np.nansum(sr[max(0, tw-7):tw])) if tw > 0 else np.nan

    X = pd.DataFrame([{k: F[k] for k in FULL}])
    modelo = xgb.XGBClassifier()
    modelo.load_model(f"{DIR}/modelos/xgb_v1_tuned.ubj")
    prob = float(modelo.predict_proba(X)[0, 1])

    print(f"\n=== Riesgo en ({lat}, {lon}) el {fecha} ===")
    print(f"Probabilidad del modelo (prevalencia de diseño 25%): {prob:.3f}")
    print(f"FWI={F['fwi']:.1f} (percentil local {F['fwi_pctl_local']:.0f}) · "
          f"HRmin={F['rh_min']:.0f}% · Tmax={F['t2m_max']:.1f}°C · "
          f"días sin lluvia={F['dias_sin_lluvia']}")
    try:
        import shap
        sv = shap.TreeExplainer(modelo).shap_values(X)[0]
        top = pd.Series(sv, index=FULL).sort_values(key=abs, ascending=False).head(10)
        print("\nTop 10 contribuciones SHAP (log-odds):")
        for k, v in top.items():
            print(f"  {'+' if v > 0 else '−'} {k:30s} {v:+.3f}  (valor={F[k]:.2f})")
    except Exception as e:
        print(f"(SHAP no disponible: {e})")


if __name__ == "__main__":
    main()
