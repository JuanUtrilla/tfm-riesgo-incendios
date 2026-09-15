#!/usr/bin/env python3
"""
v3: ablación que decide si merece la pena sustituir la autorregresiva EGIF
caducada por su equivalente FIRMS.

La pregunta mal planteada
"¿Rinde más el modelo con `n_fuegos_10km_mismomes_hist` (EGIF) o con las
features FIRMS?". Medido sobre 2019/2020 gana el EGIF, porque en esos años el
registro EGIF está completo. Pero ese no es el escenario de producción.

La pregunta bien planteada
En producción (2026) el EGIF está congelado en 2020: la feature no ve los
incendios de 2021-2026 y se degrada un poco más cada año. Su rendimiento real
está en algún punto entre "como en 2019" (si el pasado lejano bastara) y "como
si no existiera" (si la información reciente fuera lo que aporta). Por tanto:

  cota superior del daño = lo que se pierde al eliminar la feature (variante B)

Y la decisión correcta es comparar esa cota con lo que cuesta el reemplazo:

  A  v2 completo (46 feats, EGIF)              ← lo que se mide hoy, no lo que se opera
  B  v2 sin la autorregresiva EGIF (45)        ← cota inferior de producción a futuro
  C  v2 sin EGIF + FIRMS (49)                  ← el reemplazo no caducable
  D  v2 completo + FIRMS (50)                  ← ¿son complementarias?

Si C ≳ B, el reemplazo recupera lo que el EGIF congelado acabará perdiendo y
nunca caduca: merece la pena para producción.
Si C ≈ A, el reemplazo es gratis y la decisión es obvia.
Si C < B, las features FIRMS no aportan y hay que quedarse con B (retirar la
feature caducada sin más) o asumir A con la limitación documentada.

Se reporta AUC-PR, AUC-ROC y AUC-PR normalizado. Este último es imprescindible
aquí: la prevalencia del dataset varía por año (15,2% en 2018 … 37,0% en 2017,
consecuencia de que las pseudo-ausencias se sortearon uniformemente dentro del
split y no del año), y el suelo del AUC-PR es exactamente la prevalencia, así
que los AUC-PR de años distintos no son comparables entre sí en crudo.

    AUC-PR normalizado = (AP − prevalencia) / (1 − prevalencia)
    0 = azar, 1 = perfecto, comparable entre años

Salida: dataset/ablacion_v3_firms.json   (no sobrescribe nada)
"""

import os
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

RAIZ = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../..")
DIR = os.environ.get("TFM_DATOS", f"{RAIZ}/datos")
SEED = 42
FEAT_EGIF = "n_fuegos_10km_mismomes_hist"
FEATS_FIRMS = ["firms_diasfuego_mismomes_tasa", "firms_diasfuego_tasa",
               "firms_frp_p95_hist", "firms_anios_previos"]

PARAMS = dict(tree_method="hist", eval_metric="aucpr", enable_categorical=True,
              random_state=SEED, n_jobs=20, n_estimators=3000,
              early_stopping_rounds=100,
              learning_rate=0.01704120432383672, max_depth=10,
              min_child_weight=3, subsample=0.8918952588803617,
              colsample_bytree=0.5031060958594304,
              reg_lambda=0.7155371349132547, gamma=0.010555378164328569)


def ap_norm(y, p) -> float:
    prev = float(np.mean(y))
    return (average_precision_score(y, p) - prev) / (1 - prev)


def main() -> None:
    df = pd.read_parquet(f"{DIR}/dataset/dataset_modelo_v1.parquet")
    firms = pd.read_parquet(f"{DIR}/dataset/features_firms_hist_v3.parquet")
    df = df.merge(firms, on="id_muestra", how="left", validate="one_to_one")
    df["ccaa"] = df["ccaa"].fillna(-1).astype(int)
    df["es_festivo"] = df["es_festivo"].astype(float)

    feats_v2 = json.load(open(f"{DIR}/modelos/xgb_v2_prototipo_features.json"))
    sin_egif = [f for f in feats_v2 if f != FEAT_EGIF]

    variantes = {
        "A_v2_completo_egif": feats_v2,
        "B_sin_autorregresiva": sin_egif,
        "C_firms_reemplaza_egif": sin_egif + FEATS_FIRMS,
        "D_egif_mas_firms": feats_v2 + FEATS_FIRMS,
    }

    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
    y_va, y_te = va["label"].values, te["label"].values
    print(f"prevalencia  val {y_va.mean()*100:.2f}%  test {y_te.mean()*100:.2f}%\n",
          flush=True)

    out = {"prevalencia_val": float(y_va.mean()),
           "prevalencia_test": float(y_te.mean()), "variantes": {}}

    for nombre, feats in variantes.items():
        m = xgb.XGBClassifier(**PARAMS)
        m.fit(tr[feats], tr["label"], eval_set=[(va[feats], va["label"])],
              verbose=False)
        p_va = m.predict_proba(va[feats])[:, 1]
        p_te = m.predict_proba(te[feats])[:, 1]
        r = {
            "n_features": len(feats),
            "mejor_iter": int(m.best_iteration),
            "val_auc_pr": float(average_precision_score(y_va, p_va)),
            "val_auc_pr_norm": float(ap_norm(y_va, p_va)),
            "val_auc_roc": float(roc_auc_score(y_va, p_va)),
            "test_auc_pr": float(average_precision_score(y_te, p_te)),
            "test_auc_pr_norm": float(ap_norm(y_te, p_te)),
            "test_auc_roc": float(roc_auc_score(y_te, p_te)),
        }
        out["variantes"][nombre] = r
        print(f"  {nombre:24s} ({r['n_features']:2d}f)  "
              f"VAL AP={r['val_auc_pr']:.4f} (norm {r['val_auc_pr_norm']:.4f}) "
              f"ROC={r['val_auc_roc']:.4f}  |  "
              f"TEST AP={r['test_auc_pr']:.4f} ROC={r['test_auc_roc']:.4f}",
              flush=True)

    v = out["variantes"]
    dano = v["A_v2_completo_egif"]["val_auc_pr"] - v["B_sin_autorregresiva"]["val_auc_pr"]
    recup = v["C_firms_reemplaza_egif"]["val_auc_pr"] - v["B_sin_autorregresiva"]["val_auc_pr"]
    out["lectura"] = {
        "cota_superior_dano_egif_congelado_val": round(dano, 4),
        "recuperado_por_firms_val": round(recup, 4),
        "fraccion_recuperada": round(recup / dano, 3) if dano > 0 else None,
    }
    print(f"\n  Cota superior del daño por EGIF congelado (A−B): {dano:+.4f} AUC-PR")
    print(f"  Recuperado por las features FIRMS   (C−B): {recup:+.4f} AUC-PR")
    if dano > 0:
        print(f"  → FIRMS recupera el {recup/dano*100:.0f}% del margen en juego")

    with open(f"{DIR}/dataset/ablacion_v3_firms.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nGuardado dataset/ablacion_v3_firms.json", flush=True)


if __name__ == "__main__":
    main()
