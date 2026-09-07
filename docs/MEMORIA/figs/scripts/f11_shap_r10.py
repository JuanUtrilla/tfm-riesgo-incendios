#!/usr/bin/env python3
"""F11 - Que mira el r10: valores SHAP sobre el banco de evaluacion.

Modelo `donde_dia_effis_r10.ubj` (escalera de `dos_18_ratio.py`, en el disco
externo `TFM_USB/modelos/`). Filas: el banco `eval_dia` de
`dataset_effis.parquet` (veranos 2023 y 2024, 1.000 celdas al azar por dia +
las celdas EFFIS quemadas), que el modelo no vio al entrenar, unido a
`celdas.parquet` como en dos_18. Se muestrean 6.000 filas (semilla 42) y se
calcula SHAP con TreeExplainer. Izquierda: importancia media |SHAP| de las 15
primeras; derecha: enjambre de las mismas (rojo = valor alto de la variable).
Uso:  python f11_shap_r10.py   [TFM_USB=<ruta al disco>]
"""
import json
import os
import pathlib

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from comun import ANCHO, AZUL, BERMELLON, REPO, guarda

USB = pathlib.Path(os.environ.get("TFM_USB", "/run/media/charredgem/3D97-F226/TFM_fuego_expansion"))
FULL = json.load(open(REPO / "muestras" / "modelos" / "xgb_v2_prototipo_features.json"))
N, SEED, TOP = 6000, 42, 15

NOMBRES = {
    "n_fuegos_10km_mismomes_hist": "incendios a 10 km, mismo mes, años previos",
    "n_fuegos_10km_365d": "incendios a 10 km, último año",
    "n_fuegos_10km_90d": "incendios a 10 km, últimos 90 días",
    "n_fuegos_1km_hist": "incendios en la celda, histórico",
    "n_fuegos_1km_90d": "incendios en la celda, 90 días",
    "frp_max_50km_7d": "foco térmico máximo a 50 km, 7 días",
    "n_detec_50km_7d": "focos térmicos a 50 km, 7 días",
    "fwi": "FWI del día", "fwi_pctl_local": "FWI, percentil de la celda",
    "fwi_anom_sigma": "FWI, anomalía local", "fwi_med_7d": "FWI medio 7 días",
    "fwi_max_7d": "FWI máximo 7 días", "fwi_med_15d": "FWI medio 15 días",
    "fwi_med_30d": "FWI medio 30 días", "t2m_max": "temperatura máxima",
    "t2m_min": "temperatura mínima", "rh_min": "humedad relativa mínima",
    "viento_max": "viento máximo", "precip_dia": "lluvia del día",
    "vpd_max": "déficit de presión de vapor", "lst": "temperatura superficie (satélite)",
    "precip_7d": "lluvia 7 días", "precip_15d": "lluvia 15 días", "precip_30d": "lluvia 30 días",
    "rh_min_med_7d": "humedad mínima media 7 días", "t2m_max_med_7d": "temp. máxima media 7 días",
    "viento_max_med_7d": "viento máximo medio 7 días", "dias_sin_lluvia": "días sin lluvia",
    "ndvi": "NDVI (verdor)", "ndvi_med_30d": "NDVI medio 30 días", "lai": "índice de área foliar",
    "swi010": "humedad del suelo", "elevacion": "altitud", "pendiente": "pendiente",
    "rugosidad": "rugosidad", "dist_carreteras": "distancia a carreteras",
    "dist_rios": "distancia a ríos", "popdens": "densidad de población",
    "clc_bosque": "% bosque", "clc_matorral": "% matorral", "clc_agricola": "% agrícola",
    "clc_artificial": "% artificial", "clc_abierto": "% espacios abiertos",
    "clc_agric_hetero": "% agrícola heterogéneo", "rayos_dia": "rayos del día",
    "rayos_7d": "rayos 7 días", "mes": "mes", "dia_anio": "día del año",
    "es_festivo": "festivo", "ccaa": "comunidad autónoma",
}


def main():
    cel = pd.read_parquet(USB / "dataset" / "celdas.parquet")
    ev = pd.read_parquet(USB / "dataset" / "dataset_effis.parquet")
    ev = ev[ev["disenio"] == "eval_dia"]
    faltan = [c for c in FULL if c not in ev.columns]
    if faltan:
        ev = ev.merge(cel[["iy", "ix"] + faltan], on=["iy", "ix"], how="left", validate="m:1")
    ev = ev.sample(min(N, len(ev)), random_state=SEED).reset_index(drop=True)
    X = ev[FULL].values.astype(np.float32)
    m = xgb.XGBClassifier(); m.load_model(str(USB / "modelos" / "donde_dia_effis_r10.ubj"))
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    imp = np.abs(sv).mean(0)
    orden = np.argsort(-imp)[:TOP]
    etq = [NOMBRES.get(FULL[i], FULL[i]) for i in orden]
    json.dump({FULL[i]: float(imp[i]) for i in np.argsort(-imp)},
              open(pathlib.Path(__file__).resolve().parents[1] / "f11_shap.json", "w"),
              indent=1, ensure_ascii=False)

    fig = plt.figure(figsize=(ANCHO, ANCHO * 0.62))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    y = np.arange(TOP)[::-1]
    ax.barh(y, imp[orden], color=AZUL, height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(etq, fontsize=6)
    ax.set_xlabel("|SHAP| medio (log-odds)", fontsize=6.5); ax.tick_params(axis="x", labelsize=6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax2 = fig.add_subplot(gs[1])
    plt.sca(ax2)
    shap.summary_plot(sv[:, orden], X[:, orden], feature_names=etq, max_display=TOP,
                      show=False, plot_size=None, color_bar=True, sort=False)
    ax2.set_yticklabels([]); ax2.set_ylabel("")
    ax2.set_xlabel("valor SHAP (log-odds)", fontsize=6.5); ax2.tick_params(axis="x", labelsize=6)
    cb = fig.axes[-1]
    cb.set_ylabel("valor de la variable", fontsize=6); cb.tick_params(labelsize=5.5)
    cb.set_yticklabels(["bajo", "alto"])
    guarda(fig, "f11_shap_r10")
    print("top:", list(zip(etq, np.round(imp[orden], 3))))


if __name__ == "__main__":
    main()
