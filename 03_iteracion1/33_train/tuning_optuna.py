#!/usr/bin/env python3
"""
Tuning bayesiano acotado del Modelo B (Optuna, 40 trials).

Objetivo: AUC-PR en val 2019 (el test 2020 no se toca hasta el final).
Presupuesto deliberadamente corto: la literatura y la escalera de features del
propio trabajo muestran que la ganancia está en las features más que en el
tuning (§3.1 PROXIMOS_PASOS).

Salida: dataset/tuning_optuna_v1.json (mejores params + métrica final en test)
        modelos/xgb_v1_tuned.ubj (solo si mejora al modelo base en val)
"""

import json

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from entrenar_modelo import (FEATS_METEO, FEATS_VEG, FEATS_ESTAT, FEATS_HIST,
                             FEATS_CAL, SEED)

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
FULL = FEATS_METEO + FEATS_VEG + FEATS_ESTAT + FEATS_HIST + FEATS_CAL

df = pd.read_parquet(f"{DIR}/dataset/dataset_modelo_v1.parquet")
df["ccaa"] = df["ccaa"].fillna(-1).astype(int)
df["es_festivo"] = df["es_festivo"].astype(float)
tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])


def objetivo(trial):
    params = dict(
        tree_method="hist", n_estimators=3000, random_state=SEED, n_jobs=20,
        eval_metric="aucpr", early_stopping_rounds=100,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )
    m = xgb.XGBClassifier(**params)
    m.fit(tr[FULL], tr["label"], eval_set=[(va[FULL], va["label"])], verbose=False)
    trial.set_user_attr("best_iteration", int(m.best_iteration))
    return average_precision_score(va["label"], m.predict_proba(va[FULL])[:, 1])


optuna.logging.set_verbosity(optuna.logging.WARNING)
estudio = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=SEED))
estudio.optimize(objetivo, n_trials=40, show_progress_bar=False)

print(f"Mejor AUC-PR val: {estudio.best_value:.4f}")
print("Mejores params:", estudio.best_params)

# referencia: el modelo base (params de entrenar_modelo) en val
from entrenar_modelo import PARAMS as PARAMS_BASE
m_base = xgb.XGBClassifier(**PARAMS_BASE)
m_base.fit(tr[FULL], tr["label"], eval_set=[(va[FULL], va["label"])], verbose=False)
auc_base_val = average_precision_score(va["label"],
                                       m_base.predict_proba(va[FULL])[:, 1])
print(f"Base en val: {auc_base_val:.4f}")

res = {"best_val_aucpr": float(estudio.best_value),
       "base_val_aucpr": float(auc_base_val),
       "best_params": estudio.best_params}

if estudio.best_value > auc_base_val:
    params = dict(tree_method="hist", n_estimators=3000, random_state=SEED,
                  n_jobs=20, eval_metric="aucpr", early_stopping_rounds=100,
                  **estudio.best_params)
    m = xgb.XGBClassifier(**params)
    m.fit(tr[FULL], tr["label"], eval_set=[(va[FULL], va["label"])], verbose=False)
    p_te = m.predict_proba(te[FULL])[:, 1]
    res["tuned_test"] = {
        "auc_pr": float(average_precision_score(te["label"], p_te)),
        "auc_roc": float(roc_auc_score(te["label"], p_te)),
        "brier": float(brier_score_loss(te["label"], p_te)),
    }
    m.save_model(f"{DIR}/modelos/xgb_v1_tuned.ubj")
    print(f"Tuned en TEST: {res['tuned_test']}")
else:
    print("El tuning no mejora al base en val → se mantiene xgb_v1.ubj")

with open(f"{DIR}/dataset/tuning_optuna_v1.json", "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print("Guardado dataset/tuning_optuna_v1.json")
