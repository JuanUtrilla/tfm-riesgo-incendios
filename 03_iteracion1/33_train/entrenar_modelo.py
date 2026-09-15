#!/usr/bin/env python3
"""
Entrenamiento y evaluación del Modelo B (riesgo de incendio, XGBoost).

Entrada:  dataset/dataset_modelo_v1.parquet
Salidas:  modelos/xgb_v1.ubj              modelo final (UBJSON, conserva categóricas)
          modelos/calibrador_isotonico.pkl
          dataset/metricas_v1.json        todas las métricas de la escalera
          eda/pr_curves_test.png, eda/calibracion_test.png, eda/shap_summary.png

Protocolo (ESTADO_ARTE §7 + PROXIMOS_PASOS §3.1):
- Split temporal congelado: train 2015-2018 / val 2019 (early stopping y
  calibración) / test 2020 (solo evaluación final).
- Escalera de baselines (cada peldaño tiene que ganar al anterior):
    A  FWI absoluto como score (sin ML)
    A2 percentil local de FWI como score (sin ML)
    B  XGBoost solo-FWI (sanity check del ensamblado)
    C  XGBoost meteo+FWI (sin estáticas/historial/calendario)
    D  XGBoost completo
- Métricas: AUC-PR (principal, prevalencia del test fijada), AUC-ROC, Brier,
  TSS (umbral elegido en val, nunca en test).
- CV espacial adicional del modelo D: GroupKFold(5) por bloque_100km sobre
  train+val (el test 2020 no se toca) → mide generalización a zonas no vistas,
  eje que la literatura señala como el más débil (Ploton 2020, Cheerala 2025).
- Sin escalado/winsorización/imputación/SMOTE (§7.5). Sin scale_pos_weight:
  la salida se quiere como probabilidad → prevalencia real + calibración
  isotónica post-hoc ajustada en val (§7.3-D3, van den Goorbergh 2022).
- Interpretabilidad: SHAP (TreeExplainer) sobre muestra del test, no gain.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
SEED = 42

FEATS_METEO = ["fwi", "fwi_pctl_local", "fwi_anom_sigma", "t2m_max", "t2m_min",
               "rh_min", "viento_max", "precip_dia", "vpd_max", "lst",
               "precip_7d", "precip_15d", "precip_30d", "fwi_med_7d",
               "fwi_max_7d", "fwi_med_15d", "fwi_med_30d", "rh_min_med_7d",
               "t2m_max_med_7d", "viento_max_med_7d", "dias_sin_lluvia"]
FEATS_VEG = ["ndvi", "ndvi_med_30d", "lai", "swi010"]
FEATS_ESTAT = ["elevacion", "pendiente", "rugosidad", "dist_carreteras",
               "dist_rios", "popdens", "clc_bosque", "clc_matorral",
               "clc_agricola", "clc_artificial", "clc_abierto",
               "clc_agric_hetero"]
FEATS_HIST = ["n_fuegos_1km_90d", "n_fuegos_10km_90d", "n_fuegos_10km_365d",
              "n_fuegos_1km_hist", "n_fuegos_10km_mismomes_hist",
              "frp_max_50km_7d", "n_detec_50km_7d", "rayos_dia", "rayos_7d"]
FEATS_CAL = ["mes", "dia_anio", "es_festivo", "ccaa"]

PARAMS = dict(tree_method="hist", n_estimators=3000, learning_rate=0.05,
              max_depth=6, min_child_weight=5, subsample=0.9,
              colsample_bytree=0.8, reg_lambda=1.0, eval_metric="aucpr",
              early_stopping_rounds=100, enable_categorical=True,
              random_state=SEED, n_jobs=20)


def tss_umbral(y_val, p_val, y_test, p_test):
    """TSS en test con umbral elegido maximizando TSS en val (nunca en test)."""
    fpr, tpr, umbrales = roc_curve(y_val, p_val)
    u = umbrales[np.argmax(tpr - fpr)]
    pred = (p_test >= u).astype(int)
    tp = ((pred == 1) & (y_test == 1)).sum(); fn = ((pred == 0) & (y_test == 1)).sum()
    fp = ((pred == 1) & (y_test == 0)).sum(); tn = ((pred == 0) & (y_test == 0)).sum()
    return tp / (tp + fn) + tn / (tn + fp) - 1, float(u)


def evaluar(nombre, y_test, score, y_val=None, score_val=None, es_prob=False):
    m = {"auc_pr": float(average_precision_score(y_test, score)),
         "auc_roc": float(roc_auc_score(y_test, score))}
    if es_prob:
        m["brier"] = float(brier_score_loss(y_test, score))
    if y_val is not None:
        tss, u = tss_umbral(y_val, score_val, y_test, score)
        m["tss"], m["umbral_val"] = float(tss), u
    print(f"  {nombre:32s} AUC-PR={m['auc_pr']:.4f}  ROC={m['auc_roc']:.4f}"
          + (f"  Brier={m['brier']:.4f}" if es_prob else "")
          + (f"  TSS={m['tss']:.3f}" if "tss" in m else ""), flush=True)
    return m


def entrenar_xgb(feats, Xtr, ytr, Xv, yv):
    modelo = xgb.XGBClassifier(**PARAMS)
    modelo.fit(Xtr[feats], ytr, eval_set=[(Xv[feats], yv)], verbose=False)
    return modelo


def main():
    df = pd.read_parquet(f"{DIR}/dataset/dataset_modelo_v1.parquet")
    # ccaa como label encoding entero (como IberFire; la categórica nativa de
    # pandas con valores float dispara un bug de serialización en xgboost 3.2)
    df["ccaa"] = df["ccaa"].fillna(-1).astype(int)
    df["es_festivo"] = df["es_festivo"].astype(float)

    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
    ytr, yv, yte = tr["label"].values, va["label"].values, te["label"].values
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} "
          f"(prevalencia test: {yte.mean()*100:.2f}%)\n", flush=True)

    feats_full = FEATS_METEO + FEATS_VEG + FEATS_ESTAT + FEATS_HIST + FEATS_CAL
    metricas = {"prevalencia_test": float(yte.mean()),
                "n_train": len(tr), "n_val": len(va), "n_test": len(te)}

    print("== ESCALERA DE BASELINES (test 2020) ==", flush=True)
    metricas["A_fwi_score"] = evaluar("A  FWI absoluto (sin ML)", yte,
                                      te["fwi"].values, yv, va["fwi"].values)
    metricas["A2_pctl_score"] = evaluar("A2 FWI pctl local (sin ML)", yte,
                                        te["fwi_pctl_local"].values, yv,
                                        va["fwi_pctl_local"].values)

    mB = entrenar_xgb(["fwi"], tr, ytr, va, yv)
    metricas["B_xgb_solo_fwi"] = evaluar(
        "B  XGB solo-FWI", yte, mB.predict_proba(te[["fwi"]])[:, 1],
        yv, mB.predict_proba(va[["fwi"]])[:, 1], es_prob=True)

    mC = entrenar_xgb(FEATS_METEO, tr, ytr, va, yv)
    metricas["C_xgb_meteo"] = evaluar(
        "C  XGB meteo+FWI", yte, mC.predict_proba(te[FEATS_METEO])[:, 1],
        yv, mC.predict_proba(va[FEATS_METEO])[:, 1], es_prob=True)

    mD = entrenar_xgb(feats_full, tr, ytr, va, yv)
    p_te = mD.predict_proba(te[feats_full])[:, 1]
    p_va = mD.predict_proba(va[feats_full])[:, 1]
    metricas["D_xgb_completo"] = evaluar("D  XGB completo", yte, p_te, yv, p_va,
                                         es_prob=True)
    metricas["D_mejor_iter"] = int(mD.best_iteration)

    # --- calibración isotónica (ajustada en val, evaluada en test) ---
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_va, yv)
    p_te_cal = iso.predict(p_te)
    metricas["D_calibrado"] = evaluar("D  calibrado (isotónico val)", yte,
                                      p_te_cal, es_prob=True)

    # --- CV espacial del modelo D (train+val, GroupKFold por bloque_100km) ---
    print("\n== CV espacial (5 folds por bloque_100km, años 2015-2019) ==",
          flush=True)
    trva = df[df.split.isin(["train", "val"])].reset_index(drop=True)
    aucs = []
    params_cv = {**PARAMS, "n_estimators": mD.best_iteration + 1}
    params_cv.pop("early_stopping_rounds")
    for k, (i_tr, i_te) in enumerate(GroupKFold(5).split(
            trva, groups=trva["bloque_100km"])):
        f_tr, f_te = trva.iloc[i_tr], trva.iloc[i_te]
        m = xgb.XGBClassifier(**params_cv)
        m.fit(f_tr[feats_full], f_tr["label"], verbose=False)
        p = m.predict_proba(f_te[feats_full])[:, 1]
        a = average_precision_score(f_te["label"], p)
        aucs.append(a)
        print(f"  fold {k}: AUC-PR={a:.4f} (prev={f_te['label'].mean()*100:.1f}%, "
              f"n={len(f_te)})", flush=True)
    metricas["D_cv_espacial_aucpr"] = {"media": float(np.mean(aucs)),
                                       "std": float(np.std(aucs)),
                                       "folds": [float(a) for a in aucs]}
    print(f"  media: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)

    # --- persistencia ---
    os.makedirs(f"{DIR}/modelos", exist_ok=True)
    mD.save_model(f"{DIR}/modelos/xgb_v1.ubj")
    with open(f"{DIR}/modelos/calibrador_isotonico.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(f"{DIR}/dataset/metricas_v1.json", "w") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    # --- gráficos: PR + calibración ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    from sklearn.calibration import calibration_curve
    os.makedirs(f"{DIR}/eda", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for nombre, score in [("FWI absoluto", te["fwi"].values),
                          ("FWI pctl local", te["fwi_pctl_local"].values),
                          ("XGB meteo", mC.predict_proba(te[FEATS_METEO])[:, 1]),
                          ("XGB completo", p_te)]:
        pr, rc, _ = precision_recall_curve(yte, score)
        ax.plot(rc, pr, label=f"{nombre} (AP={average_precision_score(yte, score):.3f})")
    ax.axhline(yte.mean(), ls="--", c="gray", label=f"prevalencia={yte.mean():.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Curvas PR — test 2020"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{DIR}/eda/pr_curves_test.png", dpi=130)

    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, p in [("sin calibrar", p_te), ("isotónico (val)", p_te_cal)]:
        fr, mp = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", label=nombre)
    ax.plot([0, 1], [0, 1], ls="--", c="gray")
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Fracción real de positivos")
    ax.set_title("Fiabilidad — test 2020"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{DIR}/eda/calibracion_test.png", dpi=130)

    # --- SHAP ---
    try:
        import shap
        muestra = te[feats_full].sample(min(3000, len(te)), random_state=SEED)
        sv = shap.TreeExplainer(mD).shap_values(muestra)
        plt.figure()
        shap.summary_plot(sv, muestra, show=False, max_display=20)
        plt.tight_layout(); plt.savefig(f"{DIR}/eda/shap_summary.png", dpi=130)
        imp = pd.Series(np.abs(sv).mean(0), index=feats_full).sort_values(
            ascending=False)
        metricas["shap_top15"] = imp.head(15).round(4).to_dict()
        print("\n== SHAP |media| top 15 ==")
        print(imp.head(15).round(4).to_string(), flush=True)
        with open(f"{DIR}/dataset/metricas_v1.json", "w") as f:
            json.dump(metricas, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"SHAP falló (no bloquea): {e}", flush=True)

    print("\nListo: modelos/xgb_v1.ubj + dataset/metricas_v1.json + eda/*.png",
          flush=True)


if __name__ == "__main__":
    main()
