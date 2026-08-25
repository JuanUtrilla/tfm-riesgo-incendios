#!/usr/bin/env python3
"""
Mapa de riesgo de incendio de España para un día dado — demo/validación visual.

Uso: python3 mapa_riesgo_dia.py [AAAA-MM-DD]   (default: 2022-07-17, pico de la
     ola de incendios de 2022, AÑO NUNCA VISTO por el modelo: entrenado 2015-2018)

Reconstruye las MISMAS features del pipeline de entrenamiento para TODAS las
celdas peninsulares (~500k) y pinta la probabilidad del modelo junto a las
detecciones FIRMS reales del día (verdad-terreno independiente del EGIF, que
está incompleto en 2021+).

Adaptaciones documentadas respecto al extractor de entrenamiento:
- Autorregresivas EGIF: el registro llega a 2020 → para fechas posteriores los
  conteos usan lo disponible (caveat de demo, sesga esas features a la baja).
- n_fuegos por radio se calculan con convolución sobre la malla (box 3×3 para
  1,5 km, 21×21 para 10 km) — equivalente a los radios KDTree del extractor.
- frp_max_50km: maximum_filter 101×101 sobre la malla de FRP [D-7, D-1].
- popdens: años >2020 usan popdens_2020; CLC: corte 2018.
Salida: eda/mapa_riesgo_<fecha>.png + parquet con las probabilidades.
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer
from scipy.ndimage import maximum_filter, uniform_filter

from entrenar_modelo import (FEATS_METEO, FEATS_VEG, FEATS_ESTAT, FEATS_HIST,
                             FEATS_CAL)

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
FECHA = sys.argv[1] if len(sys.argv) > 1 else "2022-07-17"
FULL = FEATS_METEO + FEATS_VEG + FEATS_ESTAT + FEATS_HIST + FEATS_CAL
ANIOS_CLIM = (2008, 2014)


def suma_caja(campo, radio_celdas):
    k = 2 * radio_celdas + 1
    return uniform_filter(campo.astype(float), size=k, mode="constant") * k * k


def main():
    d = np.datetime64(FECHA, "D")
    ds = xr.open_dataset(f"{DIR}/iberfire/IberFire.nc", decode_timedelta=False)
    tiempos = ds["time"].values.astype("datetime64[D]")
    t = int((d - tiempos[0]).astype(int))
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    es_esp = ds["is_spain"].values.astype(bool)
    anio, mes = int(str(d)[:4]), int(str(d)[5:7])
    dia_anio = int((d - np.datetime64(f"{anio}-01-01", "D")).astype(int)) + 1
    print(f"Mapa para {FECHA} (t={t}, año {'FUERA' if anio > 2020 else 'dentro'} "
          f"del rango de entrenamiento)", flush=True)

    # --- slabs temporales de todo el país ---
    def slab(var, t0, t1):
        return ds[var].isel(time=slice(t0, t1)).values

    fwi_w = slab("FWI", t - 30, t + 1)
    pre_w = slab("total_precipitation_mean", t - 120, t + 1)
    rh_w = slab("RH_min", t - 7, t + 1)
    tmx_w = slab("t2m_max", t - 7, t + 1)
    vto_w = slab("wind_speed_max", t - 7, t + 1)
    ndvi_w = slab("NDVI", t - 30, t + 1)
    print("slabs meteo cargados", flush=True)

    F = {}
    F["fwi"] = fwi_w[-1]
    F["t2m_max"] = tmx_w[-1]
    F["t2m_min"] = slab("t2m_min", t, t + 1)[0]
    F["rh_min"] = rh_w[-1]
    F["viento_max"] = vto_w[-1]
    F["precip_dia"] = pre_w[-1]
    F["lst"] = slab("LST", t, t + 1)[0]
    F["ndvi"] = ndvi_w[-1]
    F["lai"] = slab("LAI", t, t + 1)[0]
    F["swi010"] = slab("SWI_010", t, t + 1)[0]
    F["es_festivo"] = slab("is_holiday", t, t + 1)[0].astype(float)
    es = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
    F["vpd_max"] = es * (1 - F["rh_min"] / 100)

    F["precip_7d"] = np.nansum(pre_w[-8:-1], axis=0)
    F["precip_15d"] = np.nansum(pre_w[-16:-1], axis=0)
    F["precip_30d"] = np.nansum(pre_w[-31:-1], axis=0)
    F["fwi_med_7d"] = np.nanmean(fwi_w[-8:-1], axis=0)
    F["fwi_max_7d"] = np.nanmax(fwi_w[-8:-1], axis=0)
    F["fwi_med_15d"] = np.nanmean(fwi_w[-16:-1], axis=0)
    F["fwi_med_30d"] = np.nanmean(fwi_w[:-1], axis=0)
    F["rh_min_med_7d"] = np.nanmean(rh_w[:-1], axis=0)
    F["t2m_max_med_7d"] = np.nanmean(tmx_w[:-1], axis=0)
    F["viento_max_med_7d"] = np.nanmean(vto_w[:-1], axis=0)
    F["ndvi_med_30d"] = np.nanmean(ndvi_w[:-1], axis=0)

    # días sin lluvia (ventana 120, vectorizado hacia atrás)
    lluvia = pre_w >= 1.0
    secos = np.zeros((ny, nx), dtype=float)
    activo = np.ones((ny, nx), dtype=bool)
    for k in range(pre_w.shape[0] - 1, -1, -1):
        seco_k = ~lluvia[k] & ~np.isnan(pre_w[k])
        activo &= seco_k
        secos += activo
    F["dias_sin_lluvia"] = np.minimum(secos, 120)
    print("ventanas calculadas", flush=True)

    # climatología local del FWI (mismo mes, 2008-2014)
    anios_t = tiempos.astype("datetime64[Y]").astype(int) + 1970
    meses_t = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    idx_clim = np.where((anios_t >= ANIOS_CLIM[0]) & (anios_t <= ANIOS_CLIM[1])
                        & (meses_t == mes))[0]
    fwi_clim = ds["FWI"].isel(time=idx_clim).values
    cmed = np.nanmean(fwi_clim, axis=0)
    cstd = np.nanstd(fwi_clim, axis=0)
    F["fwi_pctl_local"] = (np.nansum(fwi_clim <= F["fwi"], axis=0)
                           / fwi_clim.shape[0] * 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        F["fwi_anom_sigma"] = np.where(cstd > 0, (F["fwi"] - cmed) / cstd, np.nan)
    del fwi_clim
    print("climatología local calculada", flush=True)

    # estáticas
    for k, v in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                 "rugosidad": "roughness_mean",
                 "dist_carreteras": "dist_to_roads_mean",
                 "dist_rios": "dist_to_waterways_mean"}.items():
        F[k] = ds[v].values
    F["popdens"] = ds[f"popdens_{min(anio, 2020)}"].values
    for k, suf in {"clc_bosque": "forest_proportion",
                   "clc_matorral": "scrub_proportion",
                   "clc_agricola": "agricultural_proportion",
                   "clc_artificial": "artificial_proportion",
                   "clc_abierto": "open_space_proportion",
                   "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
        F[k] = ds[f"CLC_2018_{suf}"].values
    F["ccaa"] = np.nan_to_num(ds["AutonomousCommunities"].values, nan=-1).astype(int)
    F["mes"] = np.full((ny, nx), mes)
    F["dia_anio"] = np.full((ny, nx), dia_anio)

    # autorregresivas EGIF por convolución sobre la malla
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    xs, ys = ds["x"].values, ds["y"].values
    egif = pd.read_csv(f"{DIR}/egif_civio.csv", usecols=["fecha", "lat", "lng"]).dropna()
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna()
    egif = egif[egif["fecha"].dt.year >= 2008]
    ex, ey = tr.transform(egif["lng"].values, egif["lat"].values)
    eix = np.rint((ex - xs[0]) / (xs[1] - xs[0])).astype(int)
    eiy = np.rint((ey - ys[0]) / (ys[1] - ys[0])).astype(int)
    ed = egif["fecha"].values.astype("datetime64[D]")
    ok = (eix >= 0) & (eix < nx) & (eiy >= 0) & (eiy < ny) & (ed < d)

    def malla_fuegos(m):
        g = np.zeros((ny, nx))
        np.add.at(g, (eiy[ok & m], eix[ok & m]), 1)
        return g

    g90 = malla_fuegos(ed >= d - 90)
    g365 = malla_fuegos(ed >= d - 365)
    ghist = malla_fuegos(np.ones(len(ed), bool))
    gmm = malla_fuegos((egif["fecha"].dt.month.values == mes)
                       & (egif["fecha"].dt.year.values < anio))
    F["n_fuegos_1km_90d"] = suma_caja(g90, 1)
    F["n_fuegos_10km_90d"] = suma_caja(g90, 10)
    F["n_fuegos_10km_365d"] = suma_caja(g365, 10)
    F["n_fuegos_1km_hist"] = suma_caja(ghist, 1)
    F["n_fuegos_10km_mismomes_hist"] = suma_caja(gmm, 10)
    print("autorregresivas EGIF calculadas", flush=True)

    # FIRMS [D-7, D-1]
    firms = pd.read_parquet(f"{DIR}/firms_iberia_2015_2024.parquet",
                            columns=["latitude", "longitude", "acq_date", "frp"])
    fd = firms["acq_date"].values.astype("datetime64[D]")
    m7 = (fd >= d - 7) & (fd < d)
    fx, fy = tr.transform(firms.loc[m7, "longitude"].values,
                          firms.loc[m7, "latitude"].values)
    fix = np.rint((fx - xs[0]) / (xs[1] - xs[0])).astype(int)
    fiy = np.rint((fy - ys[0]) / (ys[1] - ys[0])).astype(int)
    okf = (fix >= 0) & (fix < nx) & (fiy >= 0) & (fiy < ny)
    gfrp = np.zeros((ny, nx)); gn = np.zeros((ny, nx))
    np.maximum.at(gfrp, (fiy[okf], fix[okf]), firms.loc[m7, "frp"].values[okf])
    np.add.at(gn, (fiy[okf], fix[okf]), 1)
    F["frp_max_50km_7d"] = maximum_filter(gfrp, size=101, mode="constant")
    F["n_detec_50km_7d"] = suma_caja(gn, 50)

    # rayos WGLC (día y 7d), regrid vecino más cercano
    wglc = xr.open_dataset(f"{DIR}/rayos_data/wglc/wglc_timeseries_30m_daily.nc")
    wd = wglc["time"].values.astype("datetime64[D]")
    tw = int((d - wd[0]).astype(int))
    lon2d, lat2d = np.meshgrid(xs, ys)
    tr_inv = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lons, lats = tr_inv.transform(lon2d.ravel(), lat2d.ravel())
    li = np.clip(np.searchsorted(wglc["lat"].values, lats), 0, 359)
    lj = np.clip(np.searchsorted(wglc["lon"].values, lons), 0, 719)
    if 0 <= tw < wglc.sizes["time"]:
        F["rayos_dia"] = wglc["density"].isel(time=tw).values[li, lj].reshape(ny, nx)
        r7 = wglc["density"].isel(time=slice(max(0, tw - 7), tw)).sum("time").values
        F["rayos_7d"] = r7[li, lj].reshape(ny, nx)
    else:
        F["rayos_dia"] = np.full((ny, nx), np.nan)
        F["rayos_7d"] = np.full((ny, nx), np.nan)
    print("FIRMS y rayos calculados", flush=True)

    # --- predicción ---
    X = pd.DataFrame({k: F[k].ravel() for k in FULL})
    modelo = xgb.XGBClassifier()
    modelo.load_model(f"{DIR}/modelos/xgb_v1.ubj")
    prob = modelo.predict_proba(X)[:, 1].reshape(ny, nx)
    prob[~es_esp] = np.nan
    print("predicción hecha", flush=True)

    # --- pintar ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(prob, origin="lower" if ys[1] > ys[0] else "upper",
                   cmap="YlOrRd", vmin=0, vmax=np.nanpercentile(prob, 99.5))
    fig.colorbar(im, label="probabilidad del modelo (prevalencia de diseño 25%)")
    # FIRMS del PROPIO día D como verdad-terreno
    mD = fd == d
    fx2, fy2 = tr.transform(firms.loc[mD, "longitude"].values,
                            firms.loc[mD, "latitude"].values)
    fix2 = np.rint((fx2 - xs[0]) / (xs[1] - xs[0]))
    fiy2 = np.rint((fy2 - ys[0]) / (ys[1] - ys[0]))
    ax.scatter(fix2, fiy2, s=10, marker="o", facecolors="none",
               edgecolors="blue", linewidths=0.7,
               label=f"detecciones FIRMS del {FECHA} (n={mD.sum()})")
    ax.legend(loc="lower right")
    ax.set_title(f"Riesgo de incendio del modelo — {FECHA}"
                 + ("  (año NO visto en entrenamiento)" if anio > 2020 else ""))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(f"{DIR}/eda/mapa_riesgo_{FECHA}.png", dpi=150)

    # métrica del día: percentil de riesgo en celdas con detección vs resto
    pr_flat = prob.ravel()
    det = np.zeros((ny, nx), bool)
    det[fiy2.astype(int).clip(0, ny - 1), fix2.astype(int).clip(0, nx - 1)] = True
    p_det = pr_flat[det.ravel() & ~np.isnan(pr_flat)]
    p_all = pr_flat[~np.isnan(pr_flat)]
    print(f"\nRiesgo mediano en celdas CON detección FIRMS: {np.median(p_det):.3f}")
    print(f"Riesgo mediano en España ese día:              {np.median(p_all):.3f}")
    print(f"Percentil del riesgo de las celdas con fuego vs España: "
          f"{(p_all < np.median(p_det)).mean()*100:.1f}")
    print(f"\nGuardado eda/mapa_riesgo_{FECHA}.png", flush=True)


if __name__ == "__main__":
    main()
