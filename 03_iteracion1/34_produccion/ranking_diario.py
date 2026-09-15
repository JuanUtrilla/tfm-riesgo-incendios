#!/usr/bin/env python3
"""
Ranking diario de riesgo de incendio, job de GitHub Actions (TFM).

Autocontenido: usa solo activos del propio repo (modelo XGBoost, estáticas y
climatologías por estación en `modelo/`, BD del colector en `data/`) más la API
de AEMET (secret AEMET_API_KEY) y opcionalmente FIRMS (secret FIRMS_MAP_KEY).

Flujo:
1. Serie diaria de todas las estaciones desde hoy-80 días (spin-up del FWI;
   la memoria del DC es ~52 días): API `todasestaciones` + colector horario
   del repo para los últimos días sin consolidar.
2. FWI propio (fwi_canadiense) + features del modelo de producción
   (xgb_v2_prototipo) para el último día completo de cada estación.
3. Guarda `rankings/ranking_<fecha>.csv`; el commit de GitHub sella la fecha
   de la predicción, de modo que queda constancia de que se emitió antes del
   día (la validación del trabajo es el replay de 2025-2026).

Diseño y racional: TFM_fuego/MODELO_B_BITACORA.md §14-§19.
"""

import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import firms_api
from fwi_canadiense import calcular_fwi_serie

RAIZ = Path(__file__).parent
DIAS_SPINUP = 80
UMBRALES = [(0.80, "EXTREMO"), (0.55, "ALTO"), (0.25, "MODERADO"), (-1, "BAJO")]


def _dos_pasos(url, timeout=120):
    key = os.environ["AEMET_API_KEY"]
    r = requests.get(url, headers={"api_key": key}, timeout=30).json()
    if r.get("estado") != 200:
        return None
    return requests.get(r["datos"], timeout=timeout).json()


def serie_diaria_todas():
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    filas = []
    f0 = hoy - pd.Timedelta(days=DIAS_SPINUP)
    while f0 <= hoy:
        f1 = min(f0 + pd.Timedelta(days=13), hoy)
        url = (f"https://opendata.aemet.es/opendata/api/valores/climatologicos/"
               f"diarios/datos/fechaini/{f0:%Y-%m-%d}T00:00:00UTC/"
               f"fechafin/{f1:%Y-%m-%d}T23:59:59UTC/todasestaciones")
        for intento in range(5):
            datos = _dos_pasos(url)
            if datos:
                filas += datos
                break
            print(f"[{f0.date()}] reintento {intento+1}/5 en 70s", flush=True)
            time.sleep(70)
        time.sleep(3)
        f0 = f1 + pd.Timedelta(days=1)
    df = pd.DataFrame(filas)

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")

    velmedia, racha = num(df["velmedia"]), num(df["racha"])
    api = pd.DataFrame({
        "idema": df["indicativo"], "fecha": pd.to_datetime(df["fecha"]),
        "tmax": num(df["tmax"]), "tmin": num(df["tmin"]),
        "hr_min": num(df.get("hrMin", pd.Series(dtype=float))),
        # fmin y no minimum: `minimum` propaga NaN, así que si a la estación le
        # falta cualquiera de los dos campos el viento salía NaN, y muchas
        # automáticas no reportan `racha` en el diario. Medido el 18/08/2026:
        # NaN en el 16,3% de las estaciones-día y en 128 de 858 estaciones
        # (15%) durante toda la serie. El fallo era silencioso: fwi_canadiense
        # trata el viento NaN igual que viento 0 (FWI 35,7 frente a 58,0 con
        # 15 km/h), así que esas estaciones entraban al modelo con el FWI
        # hundido unos 22 puntos sin que nada avisara. `fmin` usa el valor que haya.
        "viento_max": np.fmin(velmedia * 1.5, racha),
        "prec": num(df["prec"].replace("Ip", "0")), "n_horas": 24})

    ult = api["fecha"].max()
    con = sqlite3.connect(RAIZ / "data" / "aemet_horario_verano2026.db")
    c = pd.read_sql("SELECT idema, fint, ta, tamax, tamin, hr, vv, prec "
                    "FROM observacion_horaria WHERE fint >= ?", con,
                    params=(str((ult + pd.Timedelta(days=1)).date()),))
    if len(c):
        c["fecha"] = (pd.to_datetime(c["fint"], utc=True)
                      .dt.tz_localize(None).dt.normalize())
        g = c.groupby(["idema", "fecha"])
        api = pd.concat([api, pd.DataFrame({
            "tmax": g["tamax"].max().combine_first(g["ta"].max()),
            "tmin": g["tamin"].min().combine_first(g["ta"].min()),
            "hr_min": g["hr"].min(), "viento_max": g["vv"].max(),
            "prec": g["prec"].sum(min_count=1),
            "n_horas": g["ta"].count()}).reset_index()], ignore_index=True)
    return (api.sort_values(["idema", "fecha"])
               .drop_duplicates(["idema", "fecha"], keep="first"))


def firms_frp(est):
    """FRP máx / nº detecciones a <50 km en [D-5, D-1] (opcional: sin key, 0).

    Si hay key, un fallo de la API ya no se traga como 0 detecciones: eso
    metía features falseadas al modelo con el run en verde. `descargar`
    reintenta y, si no hay respuesta, levanta FirmsCaido."""
    key = os.environ.get("FIRMS_MAP_KEY")
    vacio = pd.DataFrame({"idema": est.index,
                          "frp_max_50km_7d": 0.0, "n_detec_50km_7d": 0.0})
    if not key:
        return vacio
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    det = firms_api.descargar(key, "VIIRS_NOAA20_NRT", 5)
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    det = det[pd.to_datetime(det["acq_date"]) < hoy]
    if not len(det):                # respuesta válida sin detecciones
        return vacio
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    fx, fy = tr.transform(det["longitude"].values, det["latitude"].values)
    ex, ey = tr.transform(est["lon"].values, est["lat"].values)
    arbol = cKDTree(np.column_stack([fx, fy]))
    vec = arbol.query_ball_point(np.column_stack([ex, ey]), r=50000)
    frp = det["frp"].values
    return pd.DataFrame({
        "idema": est.index,
        "frp_max_50km_7d": [float(frp[v].max()) if v else 0.0 for v in vec],
        "n_detec_50km_7d": [float(len(v)) for v in vec]})


def main():
    import holidays
    import xgboost as xgb
    with open(RAIZ / "modelo" / "xgb_v2_prototipo_features.json") as fh:
        FEATS = json.load(fh)
    modelo = xgb.XGBClassifier()
    modelo.load_model(str(RAIZ / "modelo" / "xgb_v2_prototipo.ubj"))
    est = pd.read_parquet(RAIZ / "modelo" / "estaciones_prototipo.parquet") \
            .set_index("idema")
    festivos = holidays.Spain(years=[pd.Timestamp.now().year])

    todas = serie_diaria_todas()
    frp = firms_frp(est).set_index("idema")
    filas = []
    for idema, s in todas.groupby("idema"):
        if idema not in est.index or s["tmax"].notna().sum() < 45:
            continue
        e = est.loc[idema]
        cal = pd.date_range(s["fecha"].min(), s["fecha"].max(), freq="D")
        s = s.set_index("fecha").reindex(cal).rename_axis("fecha").reset_index()
        comp = s.index[(s["tmax"].notna()) & (s["n_horas"].fillna(24) >= 18)]
        if not len(comp):
            continue
        i = int(comp[-1])
        fecha, mes = s.loc[i, "fecha"], s.loc[i, "fecha"].month
        fwi_s = calcular_fwi_serie(s["tmax"].values, s["hr_min"].values,
                                   s["viento_max"].values * 3.6,
                                   s["prec"].values,
                                   s["fecha"].dt.month.values)["fwi"]
        F = {"fwi": fwi_s[i], "t2m_max": s.loc[i, "tmax"],
             "t2m_min": s.loc[i, "tmin"], "rh_min": s.loc[i, "hr_min"],
             "viento_max": s.loc[i, "viento_max"], "precip_dia": s.loc[i, "prec"]}
        es_ = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
        F["vpd_max"] = es_ * (1 - F["rh_min"] / 100)
        pre, rh, tmx, vto = (s[c].values for c in
                             ["prec", "hr_min", "tmax", "viento_max"])
        F.update(precip_7d=np.nansum(pre[i-7:i]),
                 precip_15d=np.nansum(pre[i-15:i]),
                 precip_30d=np.nansum(pre[i-30:i]),
                 fwi_med_7d=np.nanmean(fwi_s[i-7:i]),
                 fwi_max_7d=np.nanmax(fwi_s[i-7:i]),
                 fwi_med_15d=np.nanmean(fwi_s[i-15:i]),
                 fwi_med_30d=np.nanmean(fwi_s[i-30:i]),
                 rh_min_med_7d=np.nanmean(rh[i-7:i]),
                 t2m_max_med_7d=np.nanmean(tmx[i-7:i]),
                 viento_max_med_7d=np.nanmean(vto[i-7:i]))
        secos = 0
        for k in range(i, -1, -1):
            if np.isnan(pre[k]) or pre[k] >= 1.0 or secos >= 120:
                break
            secos += 1
        F["dias_sin_lluvia"] = secos
        clim = np.load(RAIZ / "modelo" / "clim_fwi" / f"{idema}.npz")[f"m{mes}"]
        F["fwi_pctl_local"] = (float((clim <= F["fwi"]).mean() * 100)
                               if len(clim) else np.nan)
        F["fwi_anom_sigma"] = (float((F["fwi"] - clim.mean()) / clim.std())
                               if len(clim) and clim.std() > 0 else np.nan)
        for c in ["ndvi", "lai", "swi010", "lst"]:
            F[c] = e[f"{c}_m{mes}"]
        F["ndvi_med_30d"] = e[f"ndvi_m{mes}"]
        for c in ["elevacion", "pendiente", "rugosidad", "dist_carreteras",
                  "dist_rios", "popdens", "clc_bosque", "clc_matorral",
                  "clc_agricola", "clc_artificial", "clc_abierto",
                  "clc_agric_hetero"]:
            F[c] = e[c]
        F["n_fuegos_10km_mismomes_hist"] = e[f"n_mismomes_m{mes}"]
        F["frp_max_50km_7d"] = frp.loc[idema, "frp_max_50km_7d"]
        F["n_detec_50km_7d"] = frp.loc[idema, "n_detec_50km_7d"]
        F["rayos_dia"], F["rayos_7d"] = 0.0, 0.0
        F["es_festivo"] = float(fecha.weekday() >= 5 or fecha.date() in festivos)
        F["mes"], F["dia_anio"] = mes, fecha.dayofyear
        F["ccaa"] = int(e["ccaa"]) if np.isfinite(e["ccaa"]) else -1
        F.update(idema=idema, nombre=e["nombre"], lat=e["lat"], lon=e["lon"],
                 fecha=str(fecha.date()))
        filas.append(F)

    df = pd.DataFrame(filas)
    df["prob"] = modelo.predict_proba(df[FEATS])[:, 1].round(4)
    df["nivel"] = df["prob"].map(
        lambda p: next(n for u, n in UMBRALES if p >= u))
    df = df.sort_values("prob", ascending=False)
    dia = df["fecha"].mode()[0]
    salida = RAIZ / "rankings" / f"ranking_{dia}.csv"
    salida.parent.mkdir(exist_ok=True)
    cols = ["idema", "nombre", "lat", "lon", "fecha", "prob", "nivel", "fwi",
            "fwi_pctl_local", "fwi_anom_sigma", "t2m_max", "rh_min",
            "viento_max", "dias_sin_lluvia", "precip_30d"]
    df[cols].round(3).to_csv(salida, index=False)
    print(f"{salida.name}: {len(df)} estaciones · "
          f"{df['nivel'].value_counts().to_dict()}")
    print(df.head(10)[["nombre", "prob", "nivel", "fwi",
                       "fwi_pctl_local"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
