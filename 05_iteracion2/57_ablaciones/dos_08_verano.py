#!/usr/bin/env python3
"""
Dos modelos — paso 8: ¿entrenar solo con VERANO mejora el producto de verano?

NO TOCA PRODUCCIÓN. Lee expansión; escribe salida/dos_08_verano.json.

=============================================================================
POR QUÉ
=============================================================================
`ANALISIS_DATOS.md` §1: el 38 % de las igniciones EGIF son de febrero-abril
(Cantábrico), con otra geografía y otra meteo que las de julio-agosto. El
DÓNDE de `dos_05` aprendió «dónde arde en todo el año» y el CUÁNDO «cuándo
arde, con negativos de todo el año». El mapa se sirve en verano. Aquí se
reentrenan los dos con jun-sep (DÓNDE: etiqueta = ≥1 EGIF jun-sep 2015-18;
CUÁNDO: filas del diseño `cuando` con mes 5-10, para que el negativo siga
siendo «otro día» pero de la misma estación) y se comparan, pareado por
días, con los de todo el año sobre el mismo banco `eval_dia`.
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from pyproj import Transformer

import config
import config_expansion as ce
from dos_05_modelos import (FEATS_CUANDO, FEATS_DONDE, HIST, PARAMS, SEED,
                            entrena, metricas_dia)


def main():
    df = pd.read_parquet(f"{ce.DATASET}/dataset_dos.parquet")
    cel = pd.read_parquet(f"{ce.DATASET}/celdas.parquet")
    ev = df[df["disenio"] == "eval_dia"].reset_index(drop=True)
    sc = pd.read_parquet(f"{ce.DATASET}/scores_eval.parquet")
    assert (sc["id_muestra"].values == ev["id_muestra"].values).all()

    # --- etiqueta de verano por celda ------------------------------------------
    e = pd.read_csv(config.EGIF, usecols=["fecha", "lat", "lng"])
    e["fecha"] = pd.to_datetime(e["fecha"], errors="coerce")
    e = e.dropna()
    e = e[(e["fecha"].dt.year.between(2015, 2018)) & (e["fecha"].dt.month.between(6, 9))]
    import xarray as xr
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    xs, ys = ds["x"].values, ds["y"].values
    ds.close()
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X, Y = tr.transform(e["lng"].values, e["lat"].values)
    jx = np.rint((X - xs[0]) / (xs[1] - xs[0])).astype(int)
    jy = np.rint((Y - ys[0]) / (ys[1] - ys[0])).astype(int)
    campo = np.zeros((len(ys), len(xs)), np.int32)
    ok = (jx >= 0) & (jx < len(xs)) & (jy >= 0) & (jy < len(ys))
    np.add.at(campo, (jy[ok], jx[ok]), 1)
    y_ver = (campo[cel["iy"], cel["ix"]] > 0).astype(int)
    y_val = (cel["egif_19"] > 0).astype(int).values
    print(f"celdas con EGIF jun-sep 2015-18: {y_ver.sum():,} ({y_ver.mean()*100:.2f} %)")

    feats = FEATS_DONDE + HIST
    Xc = cel[feats].values.astype(np.float32)
    m = entrena(Xc, y_ver, Xc, y_val, "donde_cel_hist_verano")
    tabla = pd.DataFrame({"iy": cel["iy"], "ix": cel["ix"],
                          "p": m.predict_proba(Xc)[:, 1]})
    sc["donde_verano"] = ev[["iy", "ix"]].merge(tabla, how="left")["p"].values

    # --- cuándo de verano ------------------------------------------------------
    c = df[(df["disenio"] == "cuando") & (df["mes"].between(5, 10))]
    trn, va = c[c["split"] == "train"], c[c["split"] == "val"]
    print(f"cuando verano: train {len(trn):,} (pos {trn.label.mean()*100:.1f} %)")
    mc = entrena(trn[FEATS_CUANDO].values, trn["label"].values,
                 va[FEATS_CUANDO].values, va["label"].values, "cuando_verano")
    sc["cuando_verano"] = mc.predict_proba(ev[FEATS_CUANDO].values)[:, 1]

    sc["donde_verano×cuando"] = sc["donde_verano"] * sc["cuando"]
    sc["donde_cel_hist×cuando_verano"] = sc["donde_cel_hist"] * sc["cuando_verano"]
    sc["donde_verano×cuando_verano"] = sc["donde_verano"] * sc["cuando_verano"]

    rng = np.random.default_rng(SEED)
    R = {}
    ref = "donde_cel_hist×cuando"
    for anio, verdad in [(2020, "egif"), (2020, "effis"), (2022, "effis"), (2019, "egif")]:
        e_ = ev[ev["anio"] == anio]
        tr_ = metricas_dia(e_, sc[ref].values[e_.index], verdad).set_index("fecha")
        for n in ["donde_verano", "cuando_verano", "donde_verano×cuando",
                  "donde_cel_hist×cuando_verano", "donde_verano×cuando_verano",
                  "donde_cel_hist", "cuando", ref]:
            t = metricas_dia(e_, sc[n].values[e_.index], verdad).set_index("fecha")
            com = t.index.intersection(tr_.index)
            d = (t.loc[com, "auc"] - tr_.loc[com, "auc"]).values
            b = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
            R[f"{anio}_{verdad}_{n}"] = {"auc": float(t.auc.mean()), "lift": float(t.lift.mean()),
                                        "dif_vs_ref": float(d.mean()),
                                        "ic95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
            print(f"{anio} {verdad:5s} {n:30s} AUC {t.auc.mean():.3f} · lift {t.lift.mean():.2f}"
                  f" · Δ vs {ref} {d.mean():+.3f} [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}]")
    json.dump(R, open(config.salida("dos_08_verano.json"), "w"), indent=1)
    sc.to_parquet(f"{ce.DATASET}/scores_eval.parquet", index=False)


if __name__ == "__main__":
    main()
