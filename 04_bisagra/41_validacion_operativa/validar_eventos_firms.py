#!/usr/bin/env python3
"""
Validación out-of-time 2021-2024: eventos de incendio FIRMS vs riesgo del modelo.

El modelo entrenó con 2015-2018 (val 2019, test 2020). El cubo IberFire llega a
dic-2024, así que 2021-2024 es un banco de pruebas con las mismas features del
entrenamiento y una verdad-terreno independiente del EGIF (incompleto esos años).

Fase 1, eventos: DBSCAN espacio-temporal sobre las detecciones FIRMS 2021-2024
  (x,y en km; el tiempo escalado a 2,5 km/día; eps=6, es decir ~6 km / ~2,4 días),
  tras eliminar fuentes estáticas industriales (píxel de 0,01° con detecciones
  en >15 días distintos del cuatrienio) y detecciones de baja confianza ya
  filtradas aguas arriba. Se descartan eventos fuera de la malla peninsular.

Fase 2, modelo: para cada evento, serie diaria de probabilidad del modelo de
  producción (xgb_v2_prototipo) en su celda durante todo el año del evento,
  con las mismas definiciones de features del pipeline (ventanas excluyendo el
  día, percentil local 2008-14, FIRMS frp de archivo con ventana [D-7,D-1],
  rayos WGLC hasta 2023 y 0 en 2024, autorregresivas EGIF congeladas en 2020).
  Se guarda la prob del día de inicio, su percentil dentro del año y la serie anual.

Salidas: dataset/eventos_firms_2021_2024.parquet (1 fila/evento)
         viz_data/eventos_firms_series.json (series anuales para el visor)
         stdout: estadística agregada (histograma de percentiles)
"""

from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

RAIZ = Path(__file__).resolve().parents[2]
DIR = os.environ.get("TFM_DATOS", str(RAIZ / "datos"))
ESCALA_TIEMPO_KM_DIA = 2.5
EPS_KM = 6.0
MIN_DETECCIONES = 3          # eventos con menos, fuera (ruido/quemas mínimas)
UMBRAL_INDUSTRIAL = 15       # días distintos con detección en el mismo píxel

FAMOSOS = [
    ("Sierra Bermeja", 36.62, -5.22, "2021-09-08"),
    ("Sotalvo (Ávila)", 40.545, -4.786, "2021-08-14"),
    ("Losacio/Tábara (Zamora)", 41.75, -6.05, "2022-07-15"),
    ("Ateca (Zaragoza)", 41.33, -1.80, "2022-07-18"),
    ("Bejís (Castellón)", 39.90, -0.72, "2022-08-15"),
    ("Vall d'Ebo (Alicante)", 38.80, -0.17, "2022-08-13"),
    ("Las Hurdes (Cáceres)", 40.30, -6.33, "2023-05-17"),
    ("Oleada Asturias", 43.30, -6.50, "2023-03-30"),
]


def fase1_eventos():
    f = pd.read_parquet(f"{DIR}/firms_iberia_2015_2024.parquet",
                        columns=["latitude", "longitude", "acq_date", "frp"])
    f["fecha"] = pd.to_datetime(f["acq_date"])
    f = f[(f["fecha"].dt.year >= 2021) & (f["fecha"].dt.year <= 2024)].copy()
    print(f"Detecciones FIRMS 2021-2024: {len(f)}")

    # filtro industrial: píxel ~1 km con detecciones en muchos días distintos
    pix = (f["latitude"].round(2).astype(str) + ","
           + f["longitude"].round(2).astype(str))
    dias_por_pix = f.groupby(pix)["fecha"].transform("nunique")
    industrial = dias_por_pix > UMBRAL_INDUSTRIAL
    print(f"Eliminadas como fuente estática industrial: {industrial.sum()} "
          f"({industrial.mean()*100:.1f}%)")
    f = f[~industrial].copy()

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X, Y = tr.transform(f["longitude"].values, f["latitude"].values)
    t_km = (f["fecha"].values.astype("datetime64[D]").astype(int)
            * ESCALA_TIEMPO_KM_DIA)
    pts = np.column_stack([X / 1000, Y / 1000, t_km])
    lab = DBSCAN(eps=EPS_KM, min_samples=1, n_jobs=8).fit(pts).labels_
    f["evento"] = lab

    g = f.groupby("evento")
    ev = pd.DataFrame({
        "lat": g["latitude"].mean(), "lon": g["longitude"].mean(),
        "fecha_ini": g["fecha"].min(), "fecha_fin": g["fecha"].max(),
        "n_detec": g.size(), "frp_max": g["frp"].max(), "frp_sum": g["frp"].sum(),
    }).reset_index(drop=True)
    ev = ev[ev["n_detec"] >= MIN_DETECCIONES].reset_index(drop=True)
    ev["duracion_d"] = (ev["fecha_fin"] - ev["fecha_ini"]).dt.days + 1
    ev = ev[ev["duracion_d"] <= 30].reset_index(drop=True)   # >30d = sospechoso
    print(f"Eventos (≥{MIN_DETECCIONES} detecciones, ≤30 días): {len(ev)}")

    # etiqueta de incendios famosos (a <15 km y ±5 días)
    ev["nombre"] = ""
    for nombre, la, lo, fe in FAMOSOS:
        X0, Y0 = tr.transform(lo, la)
        Xe, Ye = tr.transform(ev["lon"].values, ev["lat"].values)
        d = np.hypot(Xe - X0, Ye - Y0)
        dt = (ev["fecha_ini"] - pd.Timestamp(fe)).dt.days.abs()
        m = (d < 15000) & (dt <= 5)
        if m.any():
            j = ev.loc[m, "frp_sum"].idxmax()
            ev.loc[j, "nombre"] = nombre
    return ev


def fase2_modelo(ev):
    import xgboost as xgb
    with open(f"{DIR}/modelos/xgb_v2_prototipo_features.json") as fh:
        FEATS = json.load(fh)
    modelo = xgb.XGBClassifier()
    modelo.load_model(f"{DIR}/modelos/xgb_v2_prototipo.ubj")

    ds = xr.open_dataset(f"{DIR}/iberfire/IberFire.nc", decode_timedelta=False)
    xs, ys = ds["x"].values, ds["y"].values
    tiempos = ds["time"].values.astype("datetime64[D]")
    meses_t = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    anios_t = tiempos.astype("datetime64[Y]").astype(int) + 1970
    es_esp = ds["is_spain"].values.astype(bool)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    Xe, Ye = tr.transform(ev["lon"].values, ev["lat"].values)
    ev["ix"] = np.rint((Xe - xs[0]) / (xs[1] - xs[0])).astype(int)
    ev["iy"] = np.rint((Ye - ys[0]) / (ys[1] - ys[0])).astype(int)
    dentro = ((ev.ix >= 0) & (ev.ix < len(xs)) & (ev.iy >= 0) & (ev.iy < len(ys)))
    ev = ev[dentro].copy()
    ev = ev[es_esp[ev.iy, ev.ix]].reset_index(drop=True)
    ev["anio"] = ev["fecha_ini"].dt.year
    print(f"Eventos en malla peninsular: {len(ev)}")

    # fuentes auxiliares
    egif = pd.read_csv(f"{DIR}/egif_civio.csv", usecols=["fecha", "lat", "lng"]).dropna()
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna()
    egif = egif[egif["fecha"].dt.year >= 2008]
    egx, egy = tr.transform(egif["lng"].values, egif["lat"].values)
    eg_dia = egif["fecha"].values.astype("datetime64[D]").astype(int)
    eg_mes = egif["fecha"].dt.month.values
    firms = pd.read_parquet(f"{DIR}/firms_iberia_2015_2024.parquet",
                            columns=["latitude", "longitude", "acq_date", "frp"])
    fx, fy = tr.transform(firms["longitude"].values, firms["latitude"].values)
    f_dia = firms["acq_date"].values.astype("datetime64[D]").astype(int)
    f_frp = firms["frp"].values.astype(float)
    arbol_f = cKDTree(np.column_stack([fx, fy]))
    wglc = xr.open_dataset(f"{DIR}/rayos_data/wglc/wglc_timeseries_30m_daily.nc")
    w_d0 = wglc["time"].values.astype("datetime64[D]").astype(int)[0]

    import holidays as hol
    series_json, filas = {}, []
    celdas = ev.groupby(["ix", "iy", "anio"])
    print(f"Celda-años a evaluar: {celdas.ngroups}")
    for n, ((ix, iy, anio), grupo) in enumerate(celdas, 1):
        celda = ds.isel(y=int(iy), x=int(ix))
        m_anio = anios_t == anio
        i0 = int(np.where(m_anio)[0][0])
        i1 = int(np.where(m_anio)[0][-1])
        sl = slice(max(0, i0 - 120), i1 + 1)          # margen para ventanas
        off = i0 - sl.start

        v = {k: celda[k].isel(time=sl).values for k in
             ["FWI", "t2m_max", "t2m_min", "RH_min", "wind_speed_max",
              "total_precipitation_mean", "LST", "NDVI", "LAI", "SWI_010",
              "is_holiday"]}
        nd = i1 - i0 + 1
        idx = np.arange(off, off + nd)

        F = pd.DataFrame(index=range(nd))
        F["fwi"] = v["FWI"][idx]
        F["t2m_max"] = v["t2m_max"][idx]; F["t2m_min"] = v["t2m_min"][idx]
        F["rh_min"] = v["RH_min"][idx]; F["viento_max"] = v["wind_speed_max"][idx]
        F["precip_dia"] = v["total_precipitation_mean"][idx]
        F["lst"] = v["LST"][idx]; F["ndvi"] = v["NDVI"][idx]
        F["lai"] = v["LAI"][idx]; F["swi010"] = v["SWI_010"][idx]
        F["es_festivo"] = v["is_holiday"][idx].astype(float)
        es_ = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
        F["vpd_max"] = es_ * (1 - F["rh_min"] / 100)

        pre, fwi = v["total_precipitation_mean"], v["FWI"]
        def roll(arr, w, fn):
            return np.array([fn(arr[j-w:j]) if j >= w else np.nan for j in idx])
        F["precip_7d"] = roll(pre, 7, np.nansum)
        F["precip_15d"] = roll(pre, 15, np.nansum)
        F["precip_30d"] = roll(pre, 30, np.nansum)
        F["fwi_med_7d"] = roll(fwi, 7, np.nanmean)
        F["fwi_max_7d"] = roll(fwi, 7, np.nanmax)
        F["fwi_med_15d"] = roll(fwi, 15, np.nanmean)
        F["fwi_med_30d"] = roll(fwi, 30, np.nanmean)
        F["rh_min_med_7d"] = roll(v["RH_min"], 7, np.nanmean)
        F["t2m_max_med_7d"] = roll(v["t2m_max"], 7, np.nanmean)
        F["viento_max_med_7d"] = roll(v["wind_speed_max"], 7, np.nanmean)
        F["ndvi_med_30d"] = roll(v["NDVI"], 30, np.nanmean)
        secos = np.zeros(nd)
        for k, j in enumerate(idx):
            s, q = 0, j
            while q >= 0 and s < 120:
                p = pre[q]
                if np.isnan(p) or p >= 1.0:
                    break
                s += 1; q -= 1
            secos[k] = s
        F["dias_sin_lluvia"] = secos

        mes_d = meses_t[i0:i1 + 1]
        fwi_full = celda["FWI"].values
        m_clim = (anios_t >= 2008) & (anios_t <= 2014)
        pctl = np.full(nd, np.nan); anom = np.full(nd, np.nan)
        for m in range(1, 13):
            c = fwi_full[m_clim & (meses_t == m)]
            c = c[~np.isnan(c)]
            sel = mes_d == m
            if len(c):
                pctl[sel] = np.searchsorted(np.sort(c), F["fwi"][sel]) / len(c) * 100
                if c.std() > 0:
                    anom[sel] = (F["fwi"][sel] - c.mean()) / c.std()
        F["fwi_pctl_local"] = pctl; F["fwi_anom_sigma"] = anom

        e0 = grupo.iloc[0]
        for k, var in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                       "rugosidad": "roughness_mean",
                       "dist_carreteras": "dist_to_roads_mean",
                       "dist_rios": "dist_to_waterways_mean"}.items():
            F[k] = float(ds[var].values[iy, ix])
        F["popdens"] = float(ds["popdens_2020"].values[iy, ix])
        for k, suf in {"clc_bosque": "forest_proportion",
                       "clc_matorral": "scrub_proportion",
                       "clc_agricola": "agricultural_proportion",
                       "clc_artificial": "artificial_proportion",
                       "clc_abierto": "open_space_proportion",
                       "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
            F[k] = float(ds[f"CLC_2018_{suf}"].values[iy, ix])
        cc = ds["AutonomousCommunities"].values[iy, ix]
        F["ccaa"] = int(cc) if np.isfinite(cc) else -1
        F["mes"] = mes_d
        F["dia_anio"] = np.arange(1, nd + 1)

        x0, y0 = xs[ix], ys[iy]
        dist_e = np.hypot(egx - x0, egy - y0)
        F["n_fuegos_10km_mismomes_hist"] = [
            int(((dist_e <= 10000) & (eg_mes == m)).sum()) for m in mes_d]

        vec = arbol_f.query_ball_point([x0, y0], r=50000)
        vf_d, vf_frp = f_dia[vec], f_frp[vec]
        dias_anio = tiempos[i0:i1 + 1].astype(int)
        frp7 = np.zeros(nd); nd7 = np.zeros(nd)
        for k, dd in enumerate(dias_anio):
            m7 = (vf_d >= dd - 7) & (vf_d < dd)
            nd7[k] = m7.sum()
            frp7[k] = vf_frp[m7].max() if m7.any() else 0.0
        F["frp_max_50km_7d"] = frp7; F["n_detec_50km_7d"] = nd7

        if anio <= 2023:
            sr = wglc["density"].sel(lat=e0["lat"], lon=e0["lon"],
                                     method="nearest").values
            tw = dias_anio - w_d0
            F["rayos_dia"] = sr[tw]
            F["rayos_7d"] = [np.nansum(sr[max(0, t-7):t]) for t in tw]
        else:
            F["rayos_dia"], F["rayos_7d"] = 0.0, 0.0

        prob = modelo.predict_proba(F[FEATS])[:, 1]
        for _, e in grupo.iterrows():
            d0 = int((np.datetime64(e["fecha_ini"], "D")
                      - tiempos[i0]).astype(int))
            p_dia = float(prob[d0])
            pctl_anio = float((prob < p_dia).mean() * 100)
            eid = len(filas)
            filas.append({**e[["lat", "lon", "fecha_ini", "fecha_fin", "n_detec",
                               "frp_max", "frp_sum", "duracion_d", "nombre",
                               "anio"]].to_dict(),
                          "id": eid, "prob_diaD": p_dia,
                          "pctl_en_anio": pctl_anio,
                          "fwi_diaD": float(F["fwi"][d0]),
                          "fwi_pctl_diaD": float(F["fwi_pctl_local"][d0])})
            series_json[str(eid)] = {
                "prob": [round(float(p), 3) for p in prob],
                "d0": d0, "dur": int(e["duracion_d"]), "anio": int(anio)}
        if n % 100 == 0:
            print(f"  {n}/{celdas.ngroups} celda-años", flush=True)

    res = pd.DataFrame(filas)
    res.to_parquet(f"{DIR}/dataset/eventos_firms_2021_2024.parquet", index=False)
    with open(f"{DIR}/viz_data/eventos_firms_series.json", "w") as fh:
        json.dump(series_json, fh)
    return res


def main():
    ev = fase1_eventos()
    res = fase2_modelo(ev)
    print(f"\n=== VALIDACIÓN OUT-OF-TIME 2021-2024 ({len(res)} eventos) ===")
    print("Percentil del riesgo del modelo el día de inicio, dentro de su año:")
    print(f"  mediana: {res.pctl_en_anio.median():.1f} | "
          f"p25: {res.pctl_en_anio.quantile(.25):.1f} | "
          f">80: {(res.pctl_en_anio > 80).mean()*100:.0f}% | "
          f">50: {(res.pctl_en_anio > 50).mean()*100:.0f}% "
          f"(azar = 20% y 50%)")
    print("\nIncendios famosos:")
    fam = res[res.nombre != ""].sort_values("fecha_ini")
    print(fam[["nombre", "fecha_ini", "n_detec", "prob_diaD",
               "pctl_en_anio", "fwi_diaD"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
