#!/usr/bin/env python3
"""
Dos modelos, paso 5: entrenar dónde y cuándo, combinarlos, y medirlos
dentro del día contra el modelo de producción.

No toca producción. Lee expansión y `modelos/xgb_v2_prototipo.ubj` (solo
lectura); escribe modelos en expansión y métricas en salida/.

Los modelos (todos XGBoost, mismos hiperparámetros base de `entrenar_modelo.py`)
--------------------------------------------------------------------------------
  prod        `xgb_v2_prototipo` tal cual, 46 features. Línea base obligatoria.
  cuando      diseño `cuando` (misma celda, otro día), solo features dinámicas
              y servibles: meteo del día y ventanas, FWI y derivados, vegetación,
              FIRMS [D-7,D-1], rayos, calendario. Sin estáticas, sin historia
              EGIF. Responde a «dado este sitio, ¿es hoy?».
  donde_cel   modelo por celda (498.530 filas): estáticas + clima + lat/lon
              para estimar P(≥1 EGIF en 2015-18). Sin meteo del día. Sin
              densidad histórica de fuego como feature (la variante _hist la
              lleva).
  donde_cel_hist  ídem + densidad EGIF 2008-14 (previa al dataset, servible
              congelada). Es la variante «mapa de dónde hubo incendios antes»
              que se quería descartar: se mide qué añade.
  donde_dia   modelo único con features completas entrenado sobre el diseño
              `donde` (mismo día, otra celda). Lo que pasa si solo se cambia
              el muestreo y nada más.
  mixto       modelo único con features completas, negativos mitad `cuando` y
              mitad `donde`. El contraste obligatorio del encargo (§5).
  donde_cel×cuando, donde_cel_hist×cuando   producto de probabilidades.
  donde_cel+cuando (logit)   suma de logits = producto de odds. Se reporta
              para ver si la forma de combinar importa.
  Líneas base sin modelo: fwi, fwi_pctl_local, densidad EGIF 2015-18 a 10 km
              (el «mapa tonto» del dónde), y su producto con fwi_pctl_local.

Las features «completas» son las 46 de producción, sin quitar ninguna, para
que `mixto`/`donde_dia` sean comparables con `prod` feature a feature. Las 4
autorregresivas que producción no lleva siguen fuera.

El protocolo (§6 del encargo)
-----------------------------
Banco: `eval_dia`, verano de 2019 (val) y 2020 (test): cada día 1.000 celdas
al azar + los incendios EGIF del día + celdas EFFIS ardiendo. Ningún modelo
lo ha visto (train 2015-18; val 2019 solo para early stopping del cuándo).

  · AUC dentro del día: por cada día con ≥1 positivo, AUC de positivos contra
    las 1.000 celdas al azar de ese mismo día. Se reporta la media sobre días.
  · Lift del decil superior: fracción de positivos del día que caen en el 10 %
    de celdas mejor puntuadas, dividida por 0,1.
  · Dos verdades terreno: EGIF (ignición en la celda) y EFFIS (`is_fire` con
    `primer_dia`=1, lo más parecido a una ignición que tiene EFFIS).
  · Bootstrap de 2.000 remuestreos por días para el IC95 de la diferencia
    contra `prod`. Nunca por celdas (`README.md` §6bis).
  · 2022 (verano récord, fuera del EGIF): solo EFFIS, y solo como prueba
    externa: las features EGIF de 2022 arrastran el EGIF incompleto de 2021.

Salidas: expansión/modelos/*.ubj, expansión/dataset/scores_eval.parquet,
salida/dos_05_metricas.json, salida/dos_05_mapa_donde.npz (P por celda).
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import config
import config_expansion as ce
import features as F

SEED = 42
PARAMS = dict(tree_method="hist", n_estimators=3000, learning_rate=0.05,
              max_depth=6, min_child_weight=5, subsample=0.9,
              colsample_bytree=0.8, reg_lambda=1.0, n_jobs=20,
              early_stopping_rounds=100, eval_metric="aucpr", random_state=SEED)
FULL = json.load(open(f"{config.MODELOS}/xgb_v2_prototipo_features.json"))
FEATS_CUANDO = (F.FEATS_METEO + F.FEATS_VEG + ["frp_max_50km_7d", "n_detec_50km_7d",
                                                "rayos_dia", "rayos_7d"]
                + ["mes", "dia_anio", "es_festivo"])
ESTATICAS = ["elevacion", "pendiente", "rugosidad", "elev_std", "dist_carreteras",
             "dist_rios", "dist_ferrocarril", "popdens", "is_natura2000",
             "clc_bosque", "clc_matorral", "clc_agricola", "clc_artificial",
             "clc_abierto", "clc_agric_hetero", "clc_arable", "clc_urbano",
             "clc_perm"] + [f"aspect_{a}" for a in range(1, 9)]
CLIMA = ["fwi_clim_verano", "fwi_p90_verano", "tmax_clim_verano",
         "hrmin_clim_verano", "prec_clim_anual", "ndvi_clim_verano"]
GEO = ["lat", "lon"]
HIST = ["egif_0814", "egif_0814_10km"]
FEATS_DONDE = ESTATICAS + CLIMA + GEO
N_BOOT = 2000


def entrena(X, y, Xv, yv, nombre):
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    m.save_model(f"{ce.MODELOS}/{nombre}.ubj")
    print(f"  {nombre}: {m.best_iteration+1} árboles · "
          f"val aucpr {m.best_score:.4f}", flush=True)
    return m


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metricas_dia(ev, score, verdad):
    """AUC y lift del decil, por día. Devuelve DataFrame por día."""
    if verdad == "egif":
        pos = ev["label_egif"].values == 1
        neg = (ev["origen"].values == "azar")
    else:
        pos = (ev["origen"].values == "effis") & (ev["primer_dia"].values == 1)
        neg = (ev["origen"].values == "azar") & (ev["label_effis"].values == 0)
    s = ev[["fecha"]].copy()
    s["s"], s["pos"], s["neg"] = score, pos, neg
    filas = []
    for d, g in s.groupby("fecha"):
        p, n = g[g.pos], g[g.neg]
        if len(p) == 0 or len(n) == 0:
            continue
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]
        x = np.r_[p.s.values, n.s.values]
        ok = np.isfinite(x)
        if ok.sum() < 10 or y[ok].sum() == 0:
            continue
        auc = roc_auc_score(y[ok], x[ok])
        k = max(1, int(round(ok.sum() * 0.1)))
        top = np.argsort(-x[ok])[:k]
        lift = y[ok][top].mean() / y[ok].mean()
        filas.append((d, len(p), auc, lift))
    return pd.DataFrame(filas, columns=["fecha", "n_pos", "auc", "lift"])


def resumen(ev, scores, nombres, anio, verdad):
    """Media por día + IC bootstrap por días de la diferencia contra prod."""
    e = ev[ev["anio"] == anio]
    por_dia = {n: metricas_dia(e, scores[n][e.index.values], verdad)
               for n in nombres}
    base = por_dia["prod"].set_index("fecha")
    out = {}
    rng = np.random.default_rng(SEED)
    for n in nombres:
        t = por_dia[n].set_index("fecha")
        comun = t.index.intersection(base.index)
        dif_auc = (t.loc[comun, "auc"] - base.loc[comun, "auc"]).values
        dif_lift = (t.loc[comun, "lift"] - base.loc[comun, "lift"]).values
        idx = rng.integers(0, len(comun), (N_BOOT, len(comun)))
        ba = dif_auc[idx].mean(1)
        bl = dif_lift[idx].mean(1)
        out[n] = {"n_dias": int(len(t)), "n_pos_total": int(t.n_pos.sum()),
                  "auc_medio": float(t.auc.mean()), "auc_mediana": float(t.auc.median()),
                  "lift_decil_medio": float(t.lift.mean()),
                  "dif_auc_vs_prod": float(dif_auc.mean()),
                  "ic95_dif_auc": [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))],
                  "dif_lift_vs_prod": float(dif_lift.mean()),
                  "ic95_dif_lift": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
                  "dias_gana_a_prod": float((dif_auc > 0).mean())}
    return out


def main():
    df = pd.read_parquet(f"{ce.DATASET}/dataset_dos.parquet")
    cel = pd.read_parquet(f"{ce.DATASET}/celdas.parquet")
    # las columnas de celda (clima, hist, geo) se pegan a todas las filas
    cols_cel = ["iy", "ix"] + CLIMA + GEO + HIST + ["elev_std", "dist_ferrocarril",
                "is_natura2000", "clc_arable", "clc_urbano", "clc_perm",
                "egif_1518_10km"] + [f"aspect_{a}" for a in range(1, 9)]
    df = df.merge(cel[cols_cel], on=["iy", "ix"], how="left", validate="m:1")
    ev = df[df["disenio"] == "eval_dia"].reset_index(drop=True)
    print(f"dataset {df.shape} · eval {ev.shape}")
    scores = {}

    # ---- prod -----------------------------------------------------------
    prod = xgb.Booster()
    prod.load_model(f"{config.MODELOS}/xgb_v2_prototipo.ubj")
    scores["prod"] = prod.predict(xgb.DMatrix(ev[FULL].values.astype(np.float32),
                                              feature_names=FULL))
    scores["fwi"] = ev["fwi"].values
    scores["fwi_pctl_local"] = ev["fwi_pctl_local"].values
    scores["densidad_1518_10km"] = ev["egif_1518_10km"].values
    scores["densidad×fwi_pctl"] = ev["egif_1518_10km"].values * ev["fwi_pctl_local"].values

    # ---- cuando -----------------------------------------------------------
    c = df[df["disenio"] == "cuando"]
    tr, va = c[c["split"] == "train"], c[c["split"] == "val"]
    m_cuando = entrena(tr[FEATS_CUANDO].values, tr["label"].values,
                       va[FEATS_CUANDO].values, va["label"].values, "cuando")
    scores["cuando"] = m_cuando.predict_proba(ev[FEATS_CUANDO].values)[:, 1]
    # y el mismo diseño con las 46 features (= prod reentrenado aquí, control)
    m_c46 = entrena(tr[FULL].values, tr["label"].values,
                    va[FULL].values, va["label"].values, "cuando_46")
    scores["cuando_46"] = m_c46.predict_proba(ev[FULL].values)[:, 1]

    # ---- donde por celda ----------------------------------------------------
    y_cel = (cel["egif_1518"] > 0).astype(int).values
    y_val_cel = (cel["egif_19"] > 0).astype(int).values
    mapa = {}
    for nombre, feats in [("donde_cel", FEATS_DONDE), ("donde_cel_hist", FEATS_DONDE + HIST)]:
        X = cel[feats].values.astype(np.float32)
        m = entrena(X, y_cel, X, y_val_cel, nombre)   # early stop sobre 2019
        p = m.predict_proba(X)[:, 1]
        mapa[nombre] = p
        tabla = pd.DataFrame({"iy": cel["iy"], "ix": cel["ix"], "p": p})
        scores[nombre] = ev[["iy", "ix"]].merge(tabla, how="left")["p"].values
        imp = pd.Series(m.feature_importances_, feats).sort_values(ascending=False)
        print(f"    top features {nombre}: {imp.head(8).round(3).to_dict()}")
    np.savez_compressed(config.salida("dos_05_mapa_donde.npz"),
                        iy=cel["iy"].values, ix=cel["ix"].values,
                        p_donde=mapa["donde_cel"], p_donde_hist=mapa["donde_cel_hist"])

    # ---- donde_dia (modelo único, diseño dónde, 46 features) -------------
    d = df[df["disenio"] == "donde"]
    tr, va = d[d["split"] == "train"], d[d["split"] == "val"]
    m_dd = entrena(tr[FULL].values, tr["label"].values,
                   va[FULL].values, va["label"].values, "donde_dia")
    scores["donde_dia"] = m_dd.predict_proba(ev[FULL].values)[:, 1]

    # ---- mixto ---------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    def mezcla(split):
        a = c[(c["split"] == split)]
        b = d[(d["split"] == split)]
        pos = a[a["label"] == 1]
        na = a[a["label"] == 0].sample(frac=0.5, random_state=SEED)
        nb = b[b["label"] == 0].sample(frac=0.5, random_state=SEED)
        return pd.concat([pos, na, nb])
    tr, va = mezcla("train"), mezcla("val")
    m_mix = entrena(tr[FULL].values, tr["label"].values,
                    va[FULL].values, va["label"].values, "mixto")
    scores["mixto"] = m_mix.predict_proba(ev[FULL].values)[:, 1]

    # ---- combinaciones -------------------------------------------------------
    scores["donde_cel×cuando"] = scores["donde_cel"] * scores["cuando"]
    scores["donde_cel_hist×cuando"] = scores["donde_cel_hist"] * scores["cuando"]
    scores["donde_cel+cuando_logit"] = logit(scores["donde_cel"]) + logit(scores["cuando"])
    scores["donde_cel×prod"] = scores["donde_cel"] * scores["prod"]
    scores["densidad×cuando"] = scores["densidad_1518_10km"] * scores["cuando"]

    # ---- métricas ---------------------------------------------------------
    nombres = list(scores)
    R = {"features_cuando": FEATS_CUANDO, "features_donde": FEATS_DONDE,
         "n_eval": int(len(ev))}
    for anio in (2019, 2020, 2022):
        for verdad in ("egif", "effis"):
            if anio == 2022 and verdad == "egif":
                continue
            R[f"{anio}_{verdad}"] = resumen(ev, scores, nombres, anio, verdad)
            print(f"\n== {anio} · verdad {verdad} · AUC dentro del día ==")
            for n, r in sorted(R[f"{anio}_{verdad}"].items(),
                               key=lambda kv: -kv[1]["auc_medio"]):
                print(f"  {n:26s} AUC {r['auc_medio']:.3f} · lift {r['lift_decil_medio']:.2f}"
                      f" · Δprod {r['dif_auc_vs_prod']:+.3f} "
                      f"[{r['ic95_dif_auc'][0]:+.3f}, {r['ic95_dif_auc'][1]:+.3f}]"
                      f" · gana {r['dias_gana_a_prod']*100:.0f}% días")
    # AUC global clásico de cada diseño en su test (el número que engaña)
    glob = {}
    for dis, m, feats in [("cuando", m_c46, FULL), ("donde", m_dd, FULL)]:
        t = df[(df["disenio"] == dis) & (df["split"] == "test")]
        glob[f"{dis}_46_en_test_{dis}"] = float(roc_auc_score(
            t["label"], m.predict_proba(t[feats].values)[:, 1]))
        glob[f"prod_en_test_{dis}"] = float(roc_auc_score(
            t["label"], prod.predict(xgb.DMatrix(t[FULL].values.astype(np.float32),
                                                 feature_names=FULL))))
    R["auc_global_test_caso_control"] = glob
    print("\nAUC global caso-control:", {k: round(v, 3) for k, v in glob.items()})

    json.dump(R, open(config.salida("dos_05_metricas.json"), "w"), indent=1)
    sc = pd.DataFrame(scores)
    sc.insert(0, "id_muestra", ev["id_muestra"].values)
    sc.to_parquet(f"{ce.DATASET}/scores_eval.parquet", index=False)
    print(f"\nguardado {config.salida('dos_05_metricas.json')}")


if __name__ == "__main__":
    main()
