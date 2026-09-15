#!/usr/bin/env python3
"""
Dos modelos, paso 4: las medidas de la fase 1 (`02_eda/ANALISIS_CUBO.md`).

No toca producción. Lee las tablas de expansión; escribe salida/dos_04_analisis.json.

Qué se mide y por qué
---------------------
§3.3 Autocorrelación.
  · Moran's I de la densidad EGIF 2015-20 por celda (vecindad reina 3×3) y
    agregada a 10 km. Dice cuánto se parecen celdas vecinas: si se parte al
    azar, el test queda en el vecindario del train.
  · Persistencia del dónde entre periodos: AUC de predecir «arde en 2019»
    (y 2020, y EFFIS 2021-24) solo con la densidad de 2015-18 a 10 km. Es la
    línea base más simple posible del modelo dónde, y hay que ganarla.
  · Autocorrelación temporal de la meteo (lag 1, 7, 30) sobre los nodos
    ERA5-Land 2008-2014: dice cuánto se parecen días consecutivos.
  · Fuga por partición: el mismo modelo de susceptibilidad con CV aleatorio y
    con CV por bloques de 100 km. La diferencia es la fuga.

§3.4 Susceptibilidad (cota del dónde).
  XGBoost sobre celdas con estáticas + clima para estimar P(≥1 EGIF en
  2015-18). OOF por bloques de 100 km, y evaluado en 2019, 2020 y EFFIS
  2021-24. Con y sin densidad histórica 2008-14 (la feature que «casi es la
  etiqueta»).

Todo sobre las 498.530 celdas peninsulares. Unos 5 min.
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.ndimage import uniform_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, KFold

import config
import config_expansion as ce

SALIDA = config.salida("dos_04_analisis.json")
ESTATICAS = ["elevacion", "pendiente", "rugosidad", "elev_std", "dist_carreteras",
             "dist_rios", "dist_ferrocarril", "popdens", "is_natura2000",
             "clc_bosque", "clc_matorral", "clc_agricola", "clc_artificial",
             "clc_abierto", "clc_agric_hetero", "clc_arable", "clc_urbano",
             "clc_perm"] + [f"aspect_{a}" for a in range(1, 9)]
CLIMA = ["fwi_clim_verano", "fwi_p90_verano", "tmax_clim_verano",
         "hrmin_clim_verano", "prec_clim_anual", "ndvi_clim_verano"]
GEO = ["lat", "lon"]
HIST = ["egif_0814", "egif_0814_10km"]
R = {}


def moran(campo, mask):
    """Moran's I con vecindad reina 3×3 sobre una malla (NaN fuera)."""
    z = np.where(mask, campo - campo[mask].mean(), 0.0)
    w = np.ones((3, 3)); w[1, 1] = 0
    from scipy.ndimage import convolve
    lag = convolve(z, w, mode="constant")
    nw = convolve(mask.astype(float), w, mode="constant")
    num = (z * lag)[mask].sum()
    den = (z ** 2)[mask].sum()
    S0 = nw[mask].sum()
    return float(mask.sum() / S0 * num / den)


def bloque_10km(campo, mask):
    """Agrega a bloques de 10×10 celdas (media sobre celdas de España)."""
    ny, nx = campo.shape
    by, bx = ny // 10, nx // 10
    c = np.where(mask, campo, 0)[:by * 10, :bx * 10].reshape(by, 10, bx, 10).sum((1, 3))
    m = mask[:by * 10, :bx * 10].reshape(by, 10, bx, 10).sum((1, 3))
    out = np.where(m > 0, c / np.maximum(m, 1), np.nan)
    return out, m > 50          # bloques con mayoría de tierra


def auc_seguro(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


def main():
    df = pd.read_parquet(f"{ce.DATASET}/celdas.parquet")
    ny, nx = df["iy"].max() + 1, df["ix"].max() + 1
    mask = np.zeros((ny, nx), bool)
    mask[df["iy"], df["ix"]] = True

    # ---------------- autocorrelación espacial ------------------------------
    esp = {}
    for col in ["egif_1520", "effis_2124", "fwi_clim_verano"]:
        c = np.full((ny, nx), np.nan)
        c[df["iy"], df["ix"]] = df[col].values
        m1 = mask & np.isfinite(c)
        esp[f"moran_1km_{col}"] = moran(np.nan_to_num(c), m1)
        b, mb = bloque_10km(np.nan_to_num(c), m1)
        esp[f"moran_10km_{col}"] = moran(np.nan_to_num(b), mb)
    # binaria: ¿arde alguna vez?
    c = np.zeros((ny, nx)); c[df["iy"], df["ix"]] = (df["egif_1520"] > 0)
    esp["moran_1km_egif_bin"] = moran(c, mask)
    R["autocorr_espacial"] = esp
    print("Moran:", {k: round(v, 3) for k, v in esp.items()})

    # ---------------- persistencia del dónde entre periodos -----------------
    per = {}
    y19, y20 = (df["egif_19"] > 0).values, (df["egif_20"] > 0).values
    yef = (df["effis_2124"] > 0).values
    y1518 = (df["egif_1518"] > 0).values
    for nombre, p in [("egif_1518_10km", df["egif_1518_10km"].values),
                      ("egif_0814_10km", df["egif_0814_10km"].values),
                      ("egif_1518_celda", df["egif_1518"].values)]:
        per[nombre] = {"auc_2019": auc_seguro(y19, p), "auc_2020": auc_seguro(y20, p),
                       "auc_effis_2124": auc_seguro(yef, p),
                       "ap_2019": float(average_precision_score(y19, p)),
                       "prev_2019": float(y19.mean())}
    per["egif_0814_10km"]["auc_1518"] = auc_seguro(y1518, df["egif_0814_10km"].values)
    # ¿cuántas celdas que arden en 2019-20 habían ardido en 2015-18?
    per["frac_2019_20_ya_ardido_1518_celda"] = float(
        y1518[(y19 | y20)].mean())
    per["frac_2019_20_ya_ardido_1518_10km"] = float(
        (df["egif_1518_10km"].values[(y19 | y20)] > 0).mean())
    per["frac_effis_2124_en_celda_egif_1520"] = float(
        (df["egif_1520"].values[yef] > 0).mean())
    per["frac_effis_2124_a_10km_egif_1520"] = float(
        (df["egif_1520_10km"].values[yef] > 0).mean())
    R["persistencia_donde"] = per
    print("persistencia:", json.dumps(per, indent=1))

    # ---------------- autocorrelación temporal de la meteo -----------------
    import glob
    fs = sorted(glob.glob(f"{config.DATA}/_clim/diario_201[0-4]*.npz"))
    ser = {v: [] for v in ("tmax", "hr_min", "prec")}
    for f in fs:
        z = np.load(f)
        for v in ser:
            ser[v].append(z[v])
    tmp = {}
    rng = np.random.default_rng(0)
    for v in ser:
        M = np.concatenate(ser[v], 0)               # (dias, nodos)
        sel = rng.choice(M.shape[1], 500, replace=False)
        M = M[:, sel]
        ok = np.isfinite(M).all(0)
        M = M[:, ok]
        # anomalía respecto a la media móvil de 31 días, para quitar estación
        ker = np.ones(31) / 31
        an = M - np.apply_along_axis(lambda s: np.convolve(s, ker, "same"), 0, M)
        for lag in (1, 3, 7, 30):
            r = [np.corrcoef(an[:-lag, j], an[lag:, j])[0, 1] for j in range(an.shape[1])]
            tmp[f"{v}_lag{lag}"] = float(np.nanmedian(r))
    R["autocorr_temporal_meteo_2010_14"] = tmp
    print("temporal:", {k: round(v, 3) for k, v in tmp.items()})

    # ---------------- susceptibilidad: cota del dónde -----------------------
    y = y1518.astype(int)
    grupos = df["bloque_100km"].values
    params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  tree_method="hist", n_jobs=20, eval_metric="auc")
    sus = {}
    for nombre, cols in [("estaticas", ESTATICAS), ("estaticas+clima", ESTATICAS + CLIMA),
                         ("estaticas+clima+geo", ESTATICAS + CLIMA + GEO),
                         ("estaticas+clima+hist0814", ESTATICAS + CLIMA + HIST),
                         ("solo_hist0814", HIST), ("solo_clima", CLIMA),
                         ("solo_geo", GEO)]:
        X = df[cols].values.astype(np.float32)
        res = {}
        for cv_nombre, cv in [("bloques100km", GroupKFold(5)), ("aleatorio", KFold(5, shuffle=True, random_state=0))]:
            oof = np.zeros(len(y))
            for tr, te in cv.split(X, y, grupos):
                m = xgb.XGBClassifier(**params).fit(X[tr], y[tr])
                oof[te] = m.predict_proba(X[te])[:, 1]
            res[f"auc_oof_{cv_nombre}"] = auc_seguro(y, oof)
            res[f"ap_oof_{cv_nombre}"] = float(average_precision_score(y, oof))
            if cv_nombre == "bloques100km":
                # el OOF por bloques, evaluado en los años posteriores
                res["auc_2019"] = auc_seguro(y19, oof)
                res["auc_2020"] = auc_seguro(y20, oof)
                res["auc_effis_2124"] = auc_seguro(yef, oof)
                k = int(len(oof) * 0.1)
                top = np.argsort(-oof)[:k]
                res["lift_decil_2020"] = float(y20[top].mean() / y20.mean())
                res["lift_decil_effis_2124"] = float(yef[top].mean() / yef.mean())
        sus[nombre] = res
        print(f"  {nombre:28s}", {k: round(v, 3) for k, v in res.items()}, flush=True)
    R["susceptibilidad"] = sus
    R["prevalencia_celdas"] = {"egif_1518": float(y.mean()), "egif_2019": float(y19.mean()),
                               "egif_2020": float(y20.mean()), "effis_2124": float(yef.mean())}

    json.dump(R, open(SALIDA, "w"), indent=1)
    print(f"\nguardado {SALIDA}")


if __name__ == "__main__":
    main()
