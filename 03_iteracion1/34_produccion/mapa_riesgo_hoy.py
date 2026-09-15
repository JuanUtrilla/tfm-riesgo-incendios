#!/usr/bin/env python3
"""
Mapa nacional de riesgo en tiempo real para hoy (D) y mañana (D+1).

Motor ("Ruta 1", diseño 16/07/2026): las 19 features meteo se calculan por
estación AEMET (tiempo_real.evaluar_todas con fecha objetivo: serie observada
+ predicción municipal AEMET + FWI propio propagado) y se interpolan a la
malla IberFire de 1 km (IDW k=8 en EPSG:3035, con corrección de altitud
−6,5 °C/km para las temperaturas; vpd_max se recalcula tras corregir). El
resto se calcula celda a celda igual que mapa_riesgo_dia.py: vegetación y LST
= climatología mensual 2020-24 del cubo (mismo proxy que el prototipo por
estación), estáticas y CLC del cubo, EGIF mismo-mes por convolución, FIRMS
NRT en malla, rayos=0 (moda). Modelo: xgb_v2_prototipo (el de producción).

Limitación (para la memoria): la resolución efectiva de la meteo es la densidad
de la red (~30-50 km); el 1 km lo aportan estáticas y vegetación. Mismo enfoque
que los mapas operativos de peligro de AEMET (interpolación de red + índice).

Uso:  python3 mapa_riesgo_hoy.py [--dias 0 1]
      0 = hoy, 1 = mañana (D+1), -1 = ayer observado (prueba sin forecast)
Salida: eda/mapa_riesgo_rt_<fecha>.png + prototipo/cache/malla_prob_<fecha>.npz
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.spatial import cKDTree

import tiempo_real as trm

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
FULL = trm.FULL
GAMMA = 0.0065                      # lapse rate °C/m para corrección de altitud
CORTES = [-1, 0.25, 0.55, 0.80, 2]
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]

# features meteo que se interpolan desde las estaciones (lst y vpd_max no:
# lst = clim. mensual del cubo como en el prototipo; vpd_max se recalcula)
FEATS_IDW = ["fwi", "fwi_pctl_local", "fwi_anom_sigma", "t2m_max", "t2m_min",
             "rh_min", "viento_max", "precip_dia", "precip_7d", "precip_15d",
             "precip_30d", "fwi_med_7d", "fwi_max_7d", "fwi_med_15d",
             "fwi_med_30d", "rh_min_med_7d", "t2m_max_med_7d",
             "viento_max_med_7d", "dias_sin_lluvia"]
FEATS_TEMP = {"t2m_max", "t2m_min", "t2m_max_med_7d"}   # llevan corrección altitud


def suma_caja(campo, radio_celdas):
    k = 2 * radio_celdas + 1
    return uniform_filter(campo.astype(float), size=k, mode="constant") * k * k


def malla_estaticas(ds):
    """Features estáticas 2D del cubo (una vez por ejecución)."""
    F = {}
    for k, v in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                 "rugosidad": "roughness_mean",
                 "dist_carreteras": "dist_to_roads_mean",
                 "dist_rios": "dist_to_waterways_mean"}.items():
        F[k] = ds[v].values
    F["popdens"] = ds["popdens_2020"].values
    for k, suf in {"clc_bosque": "forest_proportion",
                   "clc_matorral": "scrub_proportion",
                   "clc_agricola": "agricultural_proportion",
                   "clc_artificial": "artificial_proportion",
                   "clc_abierto": "open_space_proportion",
                   "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
        F[k] = ds[f"CLC_2018_{suf}"].values
    F["ccaa"] = np.nan_to_num(ds["AutonomousCommunities"].values, nan=-1).astype(int)
    return F


def malla_mensual(ds, mes):
    """Climatología mensual 2020-24 de vegetación/LST + EGIF mismo-mes por
    celda (caché npz por mes, con las mismas definiciones que el prototipo)."""
    ruta = f"{DIR}/prototipo/cache/malla_mensual_m{mes}.npz"
    if os.path.exists(ruta):
        return dict(np.load(ruta))
    print(f"  precomputando malla mensual m{mes} (veg 2020-24 + EGIF)...", flush=True)
    tiempos = ds["time"].values.astype("datetime64[D]")
    anios_t = tiempos.astype("datetime64[Y]").astype(int) + 1970
    meses_t = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    idx = np.where((anios_t >= 2020) & (anios_t <= 2024) & (meses_t == mes))[0]
    out = {}
    for var, col in [("NDVI", "ndvi"), ("LAI", "lai"),
                     ("SWI_010", "swi010"), ("LST", "lst")]:
        out[col] = np.nanmean(ds[var].isel(time=idx).values, axis=0)

    # EGIF mismo mes (todo el registro 2008-2020), radio 10 km ≈ caja 21×21
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    xs, ys = ds["x"].values, ds["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    egif = pd.read_csv(f"{DIR}/egif_civio.csv", usecols=["fecha", "lat", "lng"]).dropna()
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna()
    egif = egif[(egif["fecha"].dt.year >= 2008) & (egif["fecha"].dt.month == mes)]
    ex, ey = tr.transform(egif["lng"].values, egif["lat"].values)
    eix = np.rint((ex - xs[0]) / (xs[1] - xs[0])).astype(int)
    eiy = np.rint((ey - ys[0]) / (ys[1] - ys[0])).astype(int)
    ok = (eix >= 0) & (eix < nx) & (eiy >= 0) & (eiy < ny)
    g = np.zeros((ny, nx))
    np.add.at(g, (eiy[ok], eix[ok]), 1)
    out["n_fuegos_10km_mismomes_hist"] = suma_caja(g, 10)
    np.savez_compressed(ruta, **out)
    return out


def malla_firms(ds):
    """frp_max_50km_7d y n_detec_50km_7d en malla desde FIRMS NRT (caché 6 h)."""
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    xs, ys = ds["x"].values, ds["y"].values
    gfrp, gn = np.zeros((ny, nx)), np.zeros((ny, nx))
    f = trm.firms_nrt_df()
    if len(f):
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        fx, fy = tr.transform(f["longitude"].values, f["latitude"].values)
        fix = np.rint((fx - xs[0]) / (xs[1] - xs[0])).astype(int)
        fiy = np.rint((fy - ys[0]) / (ys[1] - ys[0])).astype(int)
        ok = (fix >= 0) & (fix < nx) & (fiy >= 0) & (fiy < ny)
        np.maximum.at(gfrp, (fiy[ok], fix[ok]), f["frp"].values[ok])
        np.add.at(gn, (fiy[ok], fix[ok]), 1)
    return (maximum_filter(gfrp, size=101, mode="constant"), suma_caja(gn, 50))


def interpolador_idw(rk, ds, es_esp, k=8):
    """Prepara pesos IDW de estaciones a celdas peninsulares (una vez por día).
    Devuelve una función que pasa de feature por estación (array n_est) a campo 2D."""
    est = trm._estaciones()
    rk = rk.merge(est.reset_index()[["idema", "x3035", "y3035"]], on="idema")
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    xs, ys = ds["x"].values, ds["y"].values
    gx, gy = np.meshgrid(xs, ys)
    cel = np.column_stack([gx[es_esp], gy[es_esp]])
    arbol = cKDTree(np.column_stack([rk["x3035"], rk["y3035"]]))
    dist, j = arbol.query(cel, k=min(k, len(rk)), workers=-1)
    w = 1.0 / np.maximum(dist, 1.0) ** 2

    def interpolar(vals):
        v = vals[j]
        m = np.isfinite(v)
        ww = np.where(m, w, 0.0)
        campo = np.full((ny, nx), np.nan)
        with np.errstate(invalid="ignore"):
            campo[es_esp] = (ww * np.where(m, v, 0)).sum(1) / ww.sum(1)
        return campo

    return rk, interpolar


def generar_mapa(objetivo, ds, es_esp, estat, previsto):
    """Genera el mapa nacional para la fecha objetivo. Con previsto=False usa el
    ranking observado (último día completo por estación, sin API forecast)."""
    fstr = str(objetivo.date())
    mes, dia_anio = objetivo.month, objetivo.dayofyear
    print(f"\n=== Mapa {fstr} ({'previsto' if previsto else 'observado'}) ===",
          flush=True)
    rk = (trm.evaluar_todas(fecha_objetivo=objetivo) if previsto
          else trm.evaluar_todas())
    rk = rk[np.isfinite(rk["fwi"]) & np.isfinite(rk["t2m_max"])
            & np.isfinite(rk["rh_min"]) & np.isfinite(rk["elevacion"])]
    rk = rk.reset_index(drop=True)
    print(f"estaciones con features válidas: {len(rk)}", flush=True)
    if len(rk) < 100:
        raise RuntimeError(f"solo {len(rk)} estaciones válidas — abortando")

    rk, interpolar = interpolador_idw(rk, ds, es_esp)
    F = dict(estat)
    z_est = rk["elevacion"].values
    for c in FEATS_IDW:
        if c in FEATS_TEMP:  # se baja al nivel del mar, se interpola y se devuelve a la altitud
            F[c] = interpolar(rk[c].values + GAMMA * z_est) - GAMMA * F["elevacion"]
        else:
            F[c] = interpolar(rk[c].values)
    F["fwi"] = np.clip(F["fwi"], 0, None)
    F["fwi_pctl_local"] = np.clip(F["fwi_pctl_local"], 0, 100)
    F["rh_min"] = np.clip(F["rh_min"], 0, 100)
    for c in ["precip_dia", "precip_7d", "precip_15d", "precip_30d",
              "viento_max", "dias_sin_lluvia", "fwi_med_7d", "fwi_max_7d",
              "fwi_med_15d", "fwi_med_30d", "viento_max_med_7d"]:
        F[c] = np.clip(F[c], 0, None)
    es = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
    F["vpd_max"] = es * (1 - F["rh_min"] / 100)
    print("meteo interpolada", flush=True)

    F.update(malla_mensual(ds, mes))
    F["ndvi_med_30d"] = F["ndvi"]        # mismo proxy que el prototipo (clim mensual)
    F["frp_max_50km_7d"], F["n_detec_50km_7d"] = malla_firms(ds)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    F["rayos_dia"] = np.zeros((ny, nx)); F["rayos_7d"] = np.zeros((ny, nx))
    import holidays
    fest = holidays.Spain(years=[objetivo.year])
    F["es_festivo"] = np.full((ny, nx), float(objetivo.weekday() >= 5
                                              or objetivo.date() in fest))
    F["mes"] = np.full((ny, nx), mes)
    F["dia_anio"] = np.full((ny, nx), dia_anio)

    X = pd.DataFrame({c: F[c].ravel() for c in FULL})
    modelo = xgb.XGBClassifier()
    modelo.load_model(f"{DIR}/modelos/xgb_v2_prototipo.ubj")
    prob = modelo.predict_proba(X)[:, 1].reshape(ny, nx)
    prob[~es_esp] = np.nan
    print("predicción hecha", flush=True)

    np.savez_compressed(f"{DIR}/prototipo/cache/malla_prob_{fstr}.npz",
                        prob=prob.astype(np.float32))

    # resumen por nivel (solo celdas peninsulares)
    p = prob[es_esp]
    cuenta = pd.cut(pd.Series(p), CORTES, labels=NIVELES).value_counts()
    pct = (cuenta / len(p) * 100).round(1)
    resumen = " · ".join(f"{n}: {pct[n]}%" for n in NIVELES)
    print(f"celdas por nivel — {resumen}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ys = ds["y"].values
    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(prob, origin="lower" if ys[1] > ys[0] else "upper",
                   cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(im, label="probabilidad del modelo (prevalencia de diseño 25%)")
    xs = ds["x"].values
    six = np.rint((rk["x3035"].values - xs[0]) / (xs[1] - xs[0]))
    siy = np.rint((rk["y3035"].values - ys[0]) / (ys[1] - ys[0]))
    ax.scatter(six, siy, s=2, c="gray", alpha=0.35,
               label=f"estaciones AEMET usadas (n={len(rk)})")
    ax.legend(loc="lower right")
    etiqueta = ("PREVISTO — forecast municipal AEMET + FWI propagado"
                if previsto else "observado (último día completo)")
    ax.set_title(f"Riesgo de incendio — {fstr} ({etiqueta})\n"
                 f"meteo interpolada de {len(rk)} estaciones (IDW + corr. "
                 f"altitud) · estáticas 1 km del cubo · {resumen}", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(f"{DIR}/eda/mapa_riesgo_rt_{fstr}.png", dpi=150)
    plt.close(fig)
    print(f"guardado eda/mapa_riesgo_rt_{fstr}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, nargs="+", default=[0, 1],
                    help="0=HOY, 1=MAÑANA, -1=ayer observado (prueba)")
    args = ap.parse_args()

    ds = xr.open_dataset(f"{DIR}/iberfire/IberFire.nc", decode_timedelta=False)
    es_esp = ds["is_spain"].values.astype(bool)
    estat = malla_estaticas(ds)
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    for h in args.dias:
        generar_mapa(hoy + pd.Timedelta(days=h), ds, es_esp, estat,
                     previsto=(h >= 0))


if __name__ == "__main__":
    main()
