#!/usr/bin/env python3
"""
Validación 2025-2026 vía estaciones: incendios FIRMS recientes frente al riesgo
que daba el prototipo (misma tubería estación→features que producción).

Diferencia con la validación 2021-2024 (cubo): aquí las features salen de las
estaciones AEMET (FWI propio, vegetación climatológica, autorregresivas
congeladas), así que se valida a la vez el modelo y el puente
estación→features del prototipo, sobre fuegos posteriores al entrenamiento
(2025-2026).

1. Serie diaria por estación (aemet_diario_2025_2026.parquet, spin-up nov-2024)
   → FWI propio → features de producción → probabilidad diaria por estación.
2. Eventos: DBSCAN FIRMS 2025-2026 (mismos parámetros que 2021-24, filtro
   industrial >8 días/píxel en el periodo).
3. Cada evento se empareja con la estación válida más cercana (≤35 km) y se
   calcula el percentil del día de inicio dentro del año de esa estación
   (2026 es un año parcial, de enero a hoy; el caveat queda documentado).

Salidas: dataset/eventos_firms_2025_2026.parquet
         viz_data/eventos_firms_series_2025.json
         stdout: agregados + eventos grandes
"""

import json

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from fwi_canadiense import calcular_fwi_serie

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
INICIO_EVAL = pd.Timestamp("2025-01-01")


def probabilidades_por_estacion():
    import xgboost as xgb
    import holidays as hol
    with open(f"{DIR}/modelos/xgb_v2_prototipo_features.json") as fh:
        FEATS = json.load(fh)
    modelo = xgb.XGBClassifier()
    modelo.load_model(f"{DIR}/modelos/xgb_v2_prototipo.ubj")
    est = pd.read_parquet(f"{DIR}/prototipo/estaciones_prototipo.parquet") \
            .set_index("idema")
    diario = pd.read_parquet(f"{DIR}/dataset/aemet_diario_2025_2026.parquet")
    festivos = hol.Spain(years=[2024, 2025, 2026])

    series, trozos = {}, []
    for idema, s in diario.groupby("idema"):
        if idema not in est.index or s["tmax"].notna().sum() < 200:
            continue
        e = est.loc[idema]
        cal = pd.date_range(s["fecha"].min(), s["fecha"].max(), freq="D")
        s = s.set_index("fecha").reindex(cal).rename_axis("fecha").reset_index()
        fwi = calcular_fwi_serie(s["tmax"].values, s["hr_min"].values,
                                 s["viento_max"].values * 3.6, s["prec"].values,
                                 s["fecha"].dt.month.values)["fwi"]
        F = pd.DataFrame({"fecha": s["fecha"], "fwi": fwi,
                          "t2m_max": s["tmax"], "t2m_min": s["tmin"],
                          "rh_min": s["hr_min"], "viento_max": s["viento_max"],
                          "precip_dia": s["prec"]})
        es_ = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
        F["vpd_max"] = es_ * (1 - F["rh_min"] / 100)
        pr = s["prec"]
        for w in (7, 15, 30):
            F[f"precip_{w}d"] = pr.shift(1).rolling(w, min_periods=1).sum()
        fw = pd.Series(fwi)
        F["fwi_med_7d"] = fw.shift(1).rolling(7, min_periods=3).mean()
        F["fwi_max_7d"] = fw.shift(1).rolling(7, min_periods=3).max()
        F["fwi_med_15d"] = fw.shift(1).rolling(15, min_periods=5).mean()
        F["fwi_med_30d"] = fw.shift(1).rolling(30, min_periods=10).mean()
        F["rh_min_med_7d"] = s["hr_min"].shift(1).rolling(7, min_periods=3).mean()
        F["t2m_max_med_7d"] = s["tmax"].shift(1).rolling(7, min_periods=3).mean()
        F["viento_max_med_7d"] = s["viento_max"].shift(1).rolling(7, min_periods=3).mean()
        # días sin lluvia (incluye el día; NaN corta la cuenta, como en el pipeline)
        secos = np.zeros(len(s))
        c = 0
        for j, p in enumerate(pr.values):
            c = 0 if (np.isnan(p) or p >= 1.0) else min(c + 1, 120)
            secos[j] = c
        F["dias_sin_lluvia"] = secos

        clim = np.load(f"{DIR}/prototipo/clim_fwi/{idema}.npz")
        meses = s["fecha"].dt.month.values
        pctl = np.full(len(s), np.nan); anom = np.full(len(s), np.nan)
        for m in range(1, 13):
            cm = clim[f"m{m}"]
            sel = meses == m
            if len(cm):
                pctl[sel] = np.searchsorted(cm, fwi[sel]) / len(cm) * 100
                if cm.std() > 0:
                    anom[sel] = (fwi[sel] - cm.mean()) / cm.std()
        F["fwi_pctl_local"], F["fwi_anom_sigma"] = pctl, anom

        for c_ in ["ndvi", "lai", "swi010", "lst"]:
            F[c_] = [e[f"{c_}_m{m}"] for m in meses]
        F["ndvi_med_30d"] = F["ndvi"]
        for c_ in ["elevacion", "pendiente", "rugosidad", "dist_carreteras",
                   "dist_rios", "popdens", "clc_bosque", "clc_matorral",
                   "clc_agricola", "clc_artificial", "clc_abierto",
                   "clc_agric_hetero"]:
            F[c_] = e[c_]
        F["n_fuegos_10km_mismomes_hist"] = [e[f"n_mismomes_m{m}"] for m in meses]
        F["frp_max_50km_7d"], F["n_detec_50km_7d"] = 0.0, 0.0   # sin FIRMS: moda
        F["rayos_dia"], F["rayos_7d"] = 0.0, 0.0
        F["es_festivo"] = [float(d.weekday() >= 5 or d.date() in festivos)
                           for d in s["fecha"]]
        F["mes"] = meses
        F["dia_anio"] = s["fecha"].dt.dayofyear
        F["ccaa"] = int(e["ccaa"]) if np.isfinite(e["ccaa"]) else -1

        m_ok = (F["fecha"] >= INICIO_EVAL) & F["fwi"].notna()
        F = F[m_ok].reset_index(drop=True)
        if len(F) < 100:
            continue
        F["prob"] = modelo.predict_proba(F[FEATS])[:, 1]
        F["idema"] = idema
        trozos.append(F[["idema", "fecha", "prob", "fwi", "fwi_pctl_local"]])
    probs = pd.concat(trozos, ignore_index=True)
    print(f"Probabilidades diarias: {len(probs)} filas, "
          f"{probs.idema.nunique()} estaciones, hasta {probs.fecha.max().date()}",
          flush=True)
    return probs, est


def eventos_2025_2026():
    f = pd.read_parquet(f"{DIR}/dataset/firms_2025_2026.parquet")
    f["fecha"] = pd.to_datetime(f["acq_date"])
    pix = f["latitude"].round(2).astype(str) + "," + f["longitude"].round(2).astype(str)
    industrial = f.groupby(pix)["fecha"].transform("nunique") > 8
    f = f[~industrial]
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X, Y = tr.transform(f["longitude"].values, f["latitude"].values)
    t_km = f["fecha"].values.astype("datetime64[D]").astype(int) * 2.5
    lab = DBSCAN(eps=6.0, min_samples=1, n_jobs=8).fit(
        np.column_stack([X / 1000, Y / 1000, t_km])).labels_
    f = f.assign(evento=lab)
    g = f.groupby("evento")
    ev = pd.DataFrame({
        "lat": g["latitude"].mean(), "lon": g["longitude"].mean(),
        "fecha_ini": g["fecha"].min(), "fecha_fin": g["fecha"].max(),
        "n_detec": g.size(), "frp_max": g["frp"].max(),
        "frp_sum": g["frp"].sum()}).reset_index(drop=True)
    ev = ev[ev["n_detec"] >= 3]
    ev["duracion_d"] = (ev["fecha_fin"] - ev["fecha_ini"]).dt.days + 1
    ev = ev[ev["duracion_d"] <= 30].reset_index(drop=True)
    print(f"Eventos FIRMS 2025-2026: {len(ev)}", flush=True)
    return ev


def main():
    probs, est = probabilidades_por_estacion()
    ev = eventos_2025_2026()

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    validas = probs["idema"].unique()
    e_ok = est.loc[[i for i in validas if i in est.index]]
    ex, ey = tr.transform(e_ok["lon"].values, e_ok["lat"].values)
    arbol = cKDTree(np.column_stack([ex, ey]))
    Xe, Ye = tr.transform(ev["lon"].values, ev["lat"].values)
    # hasta 4 estaciones candidatas a ≤35 km: si a la más cercana le falta el
    # día (huecos en su serie), se prueba la siguiente; sin esto se perdían
    # en silencio ~40 eventos, incluido el gran incendio de Almería (jul-2026)
    d4, j4 = arbol.query(np.column_stack([Xe, Ye]), k=4)

    series_json, filas = {}, []
    probs_idx = {k: v.reset_index(drop=True) for k, v in probs.groupby("idema")}
    n_asignados = 0
    for idx_e, e in ev.iterrows():
        pa = None
        for kk in range(4):
            if d4[idx_e, kk] > 35000:
                break
            cand = e_ok.index.values[j4[idx_e, kk]]
            p = probs_idx[cand]
            anio = e["fecha_ini"].year
            pc = p[p["fecha"].dt.year == anio].reset_index(drop=True)
            m = pc.index[pc["fecha"] == e["fecha_ini"].normalize()]
            if len(m) and not np.isnan(pc.loc[int(m[0]), "prob"]):
                pa = pc
                e = e.copy()
                e["idema"], e["dist_est_km"] = cand, d4[idx_e, kk] / 1000
                d0 = int(m[0])
                break
        if pa is None:
            continue
        n_asignados += 1
        eid = len(filas)
        filas.append({
            "id": eid, "lat": e["lat"], "lon": e["lon"],
            "fecha_ini": e["fecha_ini"], "fecha_fin": e["fecha_fin"],
            "n_detec": int(e["n_detec"]), "frp_max": e["frp_max"],
            "frp_sum": e["frp_sum"], "duracion_d": int(e["duracion_d"]),
            "anio": anio, "nombre": "", "idema": e["idema"],
            "dist_est_km": e["dist_est_km"],
            "prob_diaD": float(pa.loc[d0, "prob"]),
            "pctl_en_anio": float((pa["prob"] < pa.loc[d0, "prob"]).mean() * 100),
            "fwi_diaD": float(pa.loc[d0, "fwi"]),
            "fwi_pctl_diaD": float(pa.loc[d0, "fwi_pctl_local"]),
        })
        series_json[str(eid)] = {
            "prob": [round(float(x), 3) for x in pa["prob"]],
            "d0": d0, "dur": int(e["duracion_d"]), "anio": int(anio)}

    res = pd.DataFrame(filas)
    res.to_parquet(f"{DIR}/dataset/eventos_firms_2025_2026.parquet", index=False)
    with open(f"{DIR}/viz_data/eventos_firms_series_2025.json", "w") as fh:
        json.dump(series_json, fh)

    print(f"\n=== VALIDACIÓN 2025-2026 VÍA ESTACIONES ({len(res)} eventos) ===")
    print(f"  mediana pctl: {res.pctl_en_anio.median():.1f} | "
          f">80: {(res.pctl_en_anio > 80).mean()*100:.0f}% | "
          f">50: {(res.pctl_en_anio > 50).mean()*100:.0f}% (azar: 20% / 50%)")
    print("\nEventos más grandes:")
    print(res.nlargest(10, "frp_sum")[["fecha_ini", "n_detec", "frp_sum",
          "idema", "dist_est_km", "prob_diaD", "pctl_en_anio", "fwi_diaD"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
