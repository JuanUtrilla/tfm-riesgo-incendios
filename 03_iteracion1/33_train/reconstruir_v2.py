"""Reconstruye xgb_v2_prototipo: la receta de entrenar_modelo.py (v1) sin las 4
autorregresivas intra-celda contaminadas (MODELO_B_BITACORA §16).

Ejecutado el 02/09/2026: 240/240 árboles idénticos al .ubj servido, best_iteration
139, val AUC-PR 0,8528395, test 2020 AUC-PR 0,8278 (140 árboles) / 0,8292 (240)."""
import json, os
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
# Necesita entrenar_modelo.py (misma carpeta) en el PYTHONPATH: source entorno.sh
import entrenar_modelo as em   # DIR ya apunta a la copia aislada

RAIZ = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../..")
T = os.environ.get("TFM_DATOS", os.environ.get("TFM_FUENTE", f"{RAIZ}/datos"))
QUITAR = {"n_fuegos_1km_hist", "n_fuegos_1km_90d", "n_fuegos_10km_90d", "n_fuegos_10km_365d"}
feats_serv = json.load(open(f"{T}/modelos/xgb_v2_prototipo_features.json"))
feats_v1 = em.FEATS_METEO + em.FEATS_VEG + em.FEATS_ESTAT + em.FEATS_HIST + em.FEATS_CAL
feats = [f for f in feats_v1 if f not in QUITAR]
print("orden igual al json servido:", feats == feats_serv)

df = pd.read_parquet(f"{T}/dataset/dataset_modelo_v1.parquet")
df["ccaa"] = df["ccaa"].fillna(-1).astype(int)
df["es_festivo"] = df["es_festivo"].astype(float)
tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
ytr, yv, yte = tr["label"].values, va["label"].values, te["label"].values

m = em.entrenar_xgb(feats, tr, ytr, va, yv)
p = m.predict_proba(te[feats])[:, 1]
print(f"reconstruido: best_iter {m.best_iteration} · arboles {m.get_booster().num_boosted_rounds()} "
      f"· val aucpr {m.best_score:.7f} · test AUC-PR {average_precision_score(yte,p):.4f} · ROC {roc_auc_score(yte,p):.4f}")

s = xgb.Booster(); s.load_model(f"{T}/modelos/xgb_v2_prototipo.ubj")
ps = s.predict(xgb.DMatrix(te[feats]))
print(f"servido:      best_iter {s.attributes().get('best_iteration')} · arboles {s.num_boosted_rounds()} "
      f"· val aucpr {float(s.attributes().get('best_score')):.7f} · test AUC-PR {average_precision_score(yte,ps):.4f} · ROC {roc_auc_score(yte,ps):.4f}")
dr = m.get_booster().get_dump(dump_format="json"); ds = s.get_dump(dump_format="json")
iguales = sum(a == b for a, b in zip(dr, ds))
print(f"arboles identicos: {iguales}/{min(len(dr),len(ds))} · max |Δpred| test {np.abs(p-ps).max():.2e}")
m.save_model(os.environ.get("TFM_SALIDA", "salida") + "/xgb_v2_reconstruido.ubj")
