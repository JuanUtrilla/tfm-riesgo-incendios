#!/usr/bin/env python3
"""
Dos modelos — paso 13: los modelos con ETIQUETA EFFIS, y su test en 2024.

NO TOCA PRODUCCIÓN. Lee dataset_effis.parquet y celdas.parquet del disco
externo; escribe modelos *_effis allí y salida/dos_13_metricas.json.

Misma receta que `dos_05` (features, XGBoost, protocolo dentro del día),
cambiando solo la etiqueta: superficie quemada EFFIS en su primer día.

  cuando_effis        diseño cuándo (misma celda, otro día), 29 dinámicas servibles
  donde_dia_effis     modelo único, 46 features, diseño dónde
  donde_effis_c       por celda, estáticas+clima+geo+densidad EGIF 2008-14,
                      etiqueta «≥1 is_fire en 2008-2022», early stop en 2023.
                      (El `donde_effis` de dos_10 usaba 2008-2020 y paraba
                      con 2021-24: contamina el test 2024; este no.)
  donde_effis_c × cuando_effis     el producto
Referencias: prod, y los modelos EGIF de dos_05 (cuando, donde_dia,
donde_cel_hist×cuando) puntuados sobre el mismo banco.

Banco: eval_dia verano 2024 (test; 2023 es val y se reporta aparte). Verdad:
celdas EFFIS de primer día contra 1.000 celdas al azar del día.
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb

import config
import config_expansion as ce
from dos_05_modelos import (CLIMA, FEATS_CUANDO, FEATS_DONDE, FULL, GEO, HIST,
                            SEED, entrena, metricas_dia)

N_BOOT = 2000


def resumen(ev, scores, nombres, anio, ref="prod"):
    e = ev[ev["anio"] == anio]
    por = {n: metricas_dia(e, scores[n][e.index.values], "effis").set_index("fecha")
           for n in nombres}
    rng = np.random.default_rng(SEED)
    out = {}
    for n in nombres:
        t, b = por[n], por[ref]
        com = t.index.intersection(b.index)
        d = (t.loc[com, "auc"] - b.loc[com, "auc"]).values
        bb = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
        out[n] = {"n_dias": int(len(t)), "n_pos": int(t.n_pos.sum()),
                  "auc_medio": float(t.auc.mean()), "lift_decil": float(t.lift.mean()),
                  "dif_vs_prod": float(d.mean()),
                  "ic95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
                  "dias_gana": float((d > 0).mean())}
    return out


def main():
    df = pd.read_parquet(f"{ce.DATASET}/dataset_effis.parquet")
    cel = pd.read_parquet(f"{ce.DATASET}/celdas.parquet")
    cols_cel = ["iy", "ix"] + CLIMA + GEO + HIST + ["elev_std", "dist_ferrocarril",
                "is_natura2000", "clc_arable", "clc_urbano", "clc_perm",
                "egif_1518_10km"] + [f"aspect_{a}" for a in range(1, 9)]
    df = df.merge(cel[cols_cel], on=["iy", "ix"], how="left", validate="m:1")
    ev = df[df["disenio"] == "eval_dia"].reset_index(drop=True)
    print(f"dataset {df.shape} · eval {ev.shape} · "
          f"positivos eval {int((ev.origen == 'effis').sum())}")
    scores = {}

    prod = xgb.Booster(); prod.load_model(f"{config.MODELOS}/xgb_v2_prototipo.ubj")
    scores["prod"] = prod.predict(xgb.DMatrix(ev[FULL].values.astype(np.float32),
                                              feature_names=FULL))
    scores["fwi_pctl_local"] = ev["fwi_pctl_local"].values

    # --- EFFIS: cuándo y único ---------------------------------------------
    c = df[df["disenio"] == "cuando"]
    tr, va = c[c["split"] == "train"], c[c["split"] == "val"]
    m_c = entrena(tr[FEATS_CUANDO].values, tr["label"].values,
                  va[FEATS_CUANDO].values, va["label"].values, "cuando_effis")
    scores["cuando_effis"] = m_c.predict_proba(ev[FEATS_CUANDO].values)[:, 1]
    d = df[df["disenio"] == "donde"]
    tr, va = d[d["split"] == "train"], d[d["split"] == "val"]
    m_d = entrena(tr[FULL].values, tr["label"].values,
                  va[FULL].values, va["label"].values, "donde_dia_effis")
    scores["donde_dia_effis"] = m_d.predict_proba(ev[FULL].values)[:, 1]

    # --- DÓNDE por celda, etiqueta EFFIS 2008-2022, val 2023 ---------------
    z = np.load(f"{ce.DATASET}/cubo_etiquetas.npz")
    an = list(z["anios"]); nfa = z["n_fuego_anio"]
    def per(a0, a1):
        return nfa[an.index(a0):an.index(a1) + 1].sum(0)[cel["iy"], cel["ix"]]
    y = (per(2008, 2022) > 0).astype(int); yv = (per(2023, 2023) > 0).astype(int)
    print(f"  celdas con EFFIS 2008-22: {y.sum():,} ({y.mean()*100:.2f} %) · 2023: {yv.sum():,}")
    feats = FEATS_DONDE + HIST
    X = cel[feats].values.astype(np.float32)
    m_cel = entrena(X, y, X, yv, "donde_effis_c")
    p = m_cel.predict_proba(X)[:, 1]
    np.savez_compressed(config.salida("dos_13_mapa_donde_effis_c.npz"),
                        iy=cel["iy"].values, ix=cel["ix"].values, p=p)
    tabla = pd.DataFrame({"iy": cel["iy"], "ix": cel["ix"], "p": p})
    scores["donde_effis_c"] = ev[["iy", "ix"]].merge(tabla, how="left")["p"].values
    scores["donde_effis_c×cuando_effis"] = scores["donde_effis_c"] * scores["cuando_effis"]
    scores["donde_effis_c×prod"] = scores["donde_effis_c"] * scores["prod"]

    # --- referencias EGIF (dos_05) -------------------------------------------
    for nom in ("cuando", "donde_dia"):
        m = xgb.XGBClassifier(); m.load_model(f"{ce.MODELOS}/{nom}.ubj")
        fe = FEATS_CUANDO if nom == "cuando" else FULL
        scores[f"{nom}_egif"] = m.predict_proba(ev[fe].values)[:, 1]
    md = np.load(config.salida("dos_05_mapa_donde.npz"))
    tabla = pd.DataFrame({"iy": md["iy"], "ix": md["ix"], "p": md["p_donde_hist"]})
    scores["donde_egif_hist"] = ev[["iy", "ix"]].merge(tabla, how="left")["p"].values
    scores["donde_egif_hist×cuando_egif"] = scores["donde_egif_hist"] * scores["cuando_egif"]
    scores["donde_effis_c×cuando_egif"] = scores["donde_effis_c"] * scores["cuando_egif"]

    nombres = list(scores)
    R = {}
    for anio in (2024, 2023):
        R[anio] = resumen(ev, scores, nombres, anio)
        print(f"\n== verano {anio} ({'TEST' if anio == 2024 else 'val'}) · verdad EFFIS primer día ==")
        for n, r in sorted(R[anio].items(), key=lambda kv: -kv[1]["auc_medio"]):
            print(f"  {n:30s} AUC {r['auc_medio']:.3f} · lift {r['lift_decil']:.2f} · "
                  f"Δprod {r['dif_vs_prod']:+.3f} [{r['ic95'][0]:+.3f}, {r['ic95'][1]:+.3f}]"
                  f" · gana {r['dias_gana']*100:.0f}% · n_pos {r['n_pos']}")
    json.dump(R, open(config.salida("dos_13_metricas.json"), "w"), indent=1)
    sc = pd.DataFrame(scores); sc.insert(0, "id_muestra", ev["id_muestra"].values)
    sc.to_parquet(f"{ce.DATASET}/scores_eval_effis.parquet", index=False)


if __name__ == "__main__":
    main()
