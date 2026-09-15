#!/usr/bin/env python3
"""
Ablaciones del Modelo B: los dos experimentos que sostienen la discusión.

1. FWI absoluto frente a percentil local (la contribución del TFM, §7.1-N5: la
   ablación en GBDT no está publicada). Tres variantes del modelo meteo:
     M-abs   meteo con FWI absoluto, sin percentil/anomalía local
     M-pctl  meteo con percentil/anomalía local, sin FWI absoluto ni ventanas FWI
     M-ambos meteo completo (absoluto + local)

2. Solo calendario (mes, dia_anio, es_festivo, ccaa): cuantifica cuánto del
   rendimiento es estacionalidad pura. Si el calendario solo ya rinde mucho,
   parte del AUC del modelo completo no es "meteo" y conviene decirlo.
   (Precedente: AUC 0.737 con solo variables temporales, Fire 9(6):217.)

Mismo protocolo que entrenar_modelo.py (split congelado, early stopping en val,
métricas en test 2020). Salida: dataset/ablaciones_v1.json.
"""

import os
import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

RAIZ = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../..")
DIR = os.environ.get("TFM_DATOS", f"{RAIZ}/datos")
SEED = 42

METEO_BASE = ["t2m_max", "t2m_min", "rh_min", "viento_max", "precip_dia",
              "vpd_max", "lst", "precip_7d", "precip_15d", "precip_30d",
              "rh_min_med_7d", "t2m_max_med_7d", "viento_max_med_7d",
              "dias_sin_lluvia"]
FWI_ABS = ["fwi", "fwi_med_7d", "fwi_max_7d", "fwi_med_15d", "fwi_med_30d"]
FWI_LOCAL = ["fwi_pctl_local", "fwi_anom_sigma"]
CALENDARIO = ["mes", "dia_anio", "es_festivo", "ccaa"]

PARAMS = dict(tree_method="hist", n_estimators=3000, learning_rate=0.05,
              max_depth=6, min_child_weight=5, subsample=0.9,
              colsample_bytree=0.8, reg_lambda=1.0, eval_metric="aucpr",
              early_stopping_rounds=100, random_state=SEED, n_jobs=20)

EXPERIMENTOS = {
    "M_abs (meteo + FWI absoluto)": METEO_BASE + FWI_ABS,
    "M_pctl (meteo + FWI local)": METEO_BASE + FWI_LOCAL,
    "M_ambos (meteo + ambos)": METEO_BASE + FWI_ABS + FWI_LOCAL,
    "solo_fwi_abs": FWI_ABS,
    "solo_fwi_local": FWI_LOCAL,
    "solo_calendario": CALENDARIO,
    "calendario+fwi_local": CALENDARIO + FWI_LOCAL,
}

# 3. Sesgo de muestreo en autorregresivas de 90d: el buffer de exclusión
# (12,5 km × ±10 d) garantiza que los negativos tengan 0 fuegos a <10 km en los
# últimos 10 días, y los positivos no, así que parte de la señal de
# n_fuegos_*_90d es la regla de muestreo y no física. Se mide cuánto cae el
# modelo completo sin ellas.
FEATS_90D_SESGADAS = ["n_fuegos_1km_90d", "n_fuegos_10km_90d"]


def main():
    df = pd.read_parquet(f"{DIR}/dataset/dataset_modelo_v1.parquet")
    df["ccaa"] = df["ccaa"].fillna(-1).astype(int)
    df["es_festivo"] = df["es_festivo"].astype(float)
    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])

    from entrenar_modelo import (FEATS_METEO, FEATS_VEG, FEATS_ESTAT,
                                 FEATS_HIST, FEATS_CAL)
    full = FEATS_METEO + FEATS_VEG + FEATS_ESTAT + FEATS_HIST + FEATS_CAL
    EXPERIMENTOS["D_completo (referencia)"] = full
    EXPERIMENTOS["D_sin_autorregresivas_90d"] = \
        [f for f in full if f not in FEATS_90D_SESGADAS]

    resultados = {}
    for nombre, feats in EXPERIMENTOS.items():
        m = xgb.XGBClassifier(**PARAMS)
        m.fit(tr[feats], tr["label"],
              eval_set=[(va[feats], va["label"])], verbose=False)
        p = m.predict_proba(te[feats])[:, 1]
        resultados[nombre] = {
            "auc_pr": float(average_precision_score(te["label"], p)),
            "auc_roc": float(roc_auc_score(te["label"], p)),
            "n_features": len(feats),
            "mejor_iter": int(m.best_iteration),
        }
        r = resultados[nombre]
        print(f"{nombre:38s} AUC-PR={r['auc_pr']:.4f}  ROC={r['auc_roc']:.4f} "
              f"({r['n_features']} feats)", flush=True)

    with open(f"{DIR}/dataset/ablaciones_v1.json", "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("\nGuardado dataset/ablaciones_v1.json", flush=True)


if __name__ == "__main__":
    main()
