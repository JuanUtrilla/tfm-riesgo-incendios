#!/usr/bin/env python3
"""
Modelo B v4: entrenamiento con el EGIF consolidado y test sobre 2022.

No sobrescribe nada. Escribe solo ficheros con sufijo _v4.

Qué cambia respecto a v3
------------------------
v3 y v4 se entrenan con los mismos años (2015-2020) y las mismas 46 features.
Lo único que cambia entre ellos son los datos:

  · La etiqueta. El EGIF que usaba v3 se descargó el 30/06/2026; Civio
    republicó el dataset el 09/07/2026 consolidando provincias enteras. 2020
    pasa de 2.287 incendios a 2.700, y 2021-2022 pasan de ser residuos a estar
    completos (2.897 y 2.520).
  · `n_fuegos_10km_mismomes_hist`, que en v3 estaba congelada en 2020 y aquí
    ve el pasado hasta 2022.
  · El filtro de frontera de 2022 (Navarra y Cantabria faltan ese año; se
    excluyen las celdas a <15 km de esas comunidades para que el muestreador no
    sortee como "día sin fuego" un día en que ardió al otro lado).

Todo lo demás se mantiene idéntico a propósito: cualquier diferencia de
métrica entre v3 y v4 se debe a los datos y no al modelo.

Protocolo
---------
    train 2015-2020 · val 2021 · test 2022

2022 es el año catastrófico de la serie (Losacio, Sierra de la Culebra, Bejís;
242.436 ha) y no se mira hasta el final. Toda decisión (hiperparámetros,
número de árboles, cortes de alerta) se toma en val 2021. El test se usa una
sola vez.

Probar sobre el peor año, nunca visto en entrenamiento, es el argumento más
sólido que tiene el modelo: si aguanta ahí, aguanta.

Las tres comparaciones que produce
----------------------------------
1. Baselines obligatorios sobre el mismo test: FWI a secas y percentil local de
   FWI. Si el XGBoost no supera al FWI puro, el modelo no tiene defensa
   posible. Se reporta salga como salga.

2. v3 sobre el mismo test 2022. v3 nunca vio 2021 ni 2022, así que puede
   puntuarse en las mismas filas. Es la comparación limpia de "¿mejora el EGIF
   consolidado?", y es lo que pide el hito 7. Hay una asimetría que conviene
   declarar: las filas de test las define el muestreo v4, y v3 se evalúa sobre
   ellas; no al revés (el muestreo v3 no tenía 2022).

3. IC95 por bootstrap agrupado por bloque de 100 km, no por fila: dentro de
   un bloque las celdas comparten meteo, vegetación e historial, y remuestrear
   filas como si fueran independientes daría una precisión que no existe.

Salidas (todas nuevas):
  modelos/xgb_v4.ubj · modelos/xgb_v4_metadata.json · dataset/metricas_v4.json

Uso: python entrenar_modelo_v4.py
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)

RAIZ = Path(__file__).resolve().parents[2]
DIR = Path(os.environ.get("TFM_DATOS", RAIZ / "datos"))
REPO_OP = Path(os.environ.get("TFM_COLECTOR", DIR / "colector"))   # colector de AEMET
SEED = 42
N_BOOT = 2000

DATASET = DIR / "dataset" / "dataset_modelo_v4.parquet"
FEATS_JSON = REPO_OP / "modelo" / "xgb_v2_prototipo_features.json"
V3 = DIR / "modelos" / "xgb_v3.ubj"
SAL_MODELO = DIR / "modelos" / "xgb_v4.ubj"
SAL_META = DIR / "modelos" / "xgb_v4_metadata.json"
SAL_METRICAS = DIR / "dataset" / "metricas_v4.json"

# Cobertura operativa de los cortes de v2/v3 (bitácora §15). Se reproduce para
# que el sistema de avisos no cambie de comportamiento al cambiar de modelo:
# se recalcula el umbral que produce esa cobertura y la cobertura se mantiene.
COBERTURA = {"MODERADO": 0.348, "ALTO": 0.220, "EXTREMO": 0.135}

PARAMS_BASE = dict(learning_rate=0.05, max_depth=6, min_child_weight=5,
                   subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0)
PARAMS_TUNED = dict(learning_rate=0.01704120432383672, max_depth=10,
                    min_child_weight=3, subsample=0.8918952588803617,
                    colsample_bytree=0.5031060958594304,
                    reg_lambda=0.7155371349132547, gamma=0.010555378164328569)
COMUNES = dict(tree_method="hist", eval_metric="aucpr", enable_categorical=True,
               random_state=SEED, n_jobs=20)


def ap_norm(y, p):
    """AUC-PR normalizado: (AP − prevalencia)/(1 − prevalencia). Hace falta para
    comparar splits con prevalencias distintas: el suelo del AP es la
    prevalencia, así que los AP crudos no son comparables entre sí."""
    prev = float(np.mean(y))
    return float((average_precision_score(y, p) - prev) / (1 - prev))


def metricas(y, p, y_val=None, p_val=None):
    m = {"prevalencia": round(float(np.mean(y)), 4),
         "auc_roc": round(float(roc_auc_score(y, p)), 4),
         "auc_pr": round(float(average_precision_score(y, p)), 4),
         "ap_norm": round(ap_norm(y, p), 4)}
    # El Brier solo tiene sentido sobre probabilidades. Los baselines son
    # scores crudos (el percentil de FWI va de 0 a 100), así que se omite en
    # vez de calcular un número sin significado.
    if 0.0 <= float(np.min(p)) and float(np.max(p)) <= 1.0:
        m["brier"] = round(float(brier_score_loss(y, p)), 4)
    if y_val is not None:                 # umbral elegido en VAL, nunca en test
        fpr, tpr, u = roc_curve(y_val, p_val)
        umbral = float(u[np.argmax(tpr - fpr)])
        pred = (p >= umbral).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        m["tss"] = round(tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1, 4)
        m["umbral_val"] = round(umbral, 4)
    return m


def boot_ic(y, p, bloques, rng, n=N_BOOT):
    """IC95 de AUC-ROC remuestreando bloques de 100 km."""
    ids = np.unique(bloques)
    idx = {b: np.where(bloques == b)[0] for b in ids}
    v = []
    for _ in range(n):
        sel = np.concatenate([idx[b] for b in rng.choice(ids, len(ids), replace=True)])
        yy = y[sel]
        if 0 < yy.mean() < 1:
            v.append(roc_auc_score(yy, p[sel]))
    return [round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)]


def main():
    FEATS = json.loads(FEATS_JSON.read_text())
    df = pd.read_parquet(DATASET)
    df["ccaa"] = df["ccaa"].fillna(-1).astype(int)
    df["es_festivo"] = df["es_festivo"].astype(float)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    print(f"v4 · {len(FEATS)} features (las mismas de v2/v3)")
    print(f"train 2015-2020 {len(tr):,} · val 2021 {len(va):,} · "
          f"test 2022 {len(te):,}")
    print(f"prevalencias: {tr.label.mean():.1%} / {va.label.mean():.1%} / "
          f"{te.label.mean():.1%}\n")

    # ---- selección de hiperparámetros mirando solo val --------------------
    print("=== selección en val 2021 (el test no se toca) ===")
    cand = {}
    for nombre, P in [("base", PARAMS_BASE), ("tuned", PARAMS_TUNED)]:
        m = xgb.XGBClassifier(**COMUNES, **P, n_estimators=3000,
                              early_stopping_rounds=100)
        m.fit(tr[FEATS], tr.label, eval_set=[(va[FEATS], va.label)], verbose=False)
        p = m.predict_proba(va[FEATS])[:, 1]
        ap = average_precision_score(va.label, p)
        cand[nombre] = (m, ap)
        print(f"  {nombre:<6} AP val {ap:.4f} · AUC val "
              f"{roc_auc_score(va.label, p):.4f} · {m.best_iteration+1} árboles")
    elegido = max(cand, key=lambda k: cand[k][1])
    modelo, _ = cand[elegido]
    n_arboles = int(modelo.best_iteration + 1)
    print(f"  → elegido: {elegido} ({n_arboles} árboles)\n")

    p_va = modelo.predict_proba(va[FEATS])[:, 1]

    # ---- el test, una sola vez -------------------------------------------
    rng = np.random.default_rng(SEED)
    y_te = te.label.values
    bloques = te.bloque_100km.values
    p_te = modelo.predict_proba(te[FEATS])[:, 1]

    print("=== TEST 2022 (el peor año de la serie, nunca visto) ===")
    res = {"v4": metricas(y_te, p_te, va.label.values, p_va)}
    res["v4"]["ic95_auc_roc"] = boot_ic(y_te, p_te, bloques, rng)
    print(f"  v4                 AUC-ROC {res['v4']['auc_roc']:.4f} "
          f"{res['v4']['ic95_auc_roc']} · AUC-PR {res['v4']['auc_pr']:.4f} "
          f"· AP_norm {res['v4']['ap_norm']:.4f} · TSS {res['v4']['tss']:.4f}")

    # ---- baselines obligatorios ------------------------------------------
    for col, nom in [("fwi_pctl_local", "FWI percentil"), ("fwi", "FWI bruto")]:
        b = te[col].fillna(te[col].median()).values
        res[nom] = metricas(y_te, b)
        res[nom]["ic95_auc_roc"] = boot_ic(y_te, b, bloques, rng)
        print(f"  baseline {nom:<9} AUC-ROC {res[nom]['auc_roc']:.4f} "
              f"{res[nom]['ic95_auc_roc']} · AUC-PR {res[nom]['auc_pr']:.4f}")

    # ---- v3 sobre el mismo test ------------------------------------------
    if V3.exists():
        m3 = xgb.XGBClassifier()
        m3.load_model(str(V3))
        p3 = m3.predict_proba(te[FEATS])[:, 1]
        res["v3_en_test_v4"] = metricas(y_te, p3)
        res["v3_en_test_v4"]["ic95_auc_roc"] = boot_ic(y_te, p3, bloques, rng)
        # diferencia pareada: mismo test, mismos bloques
        ids = np.unique(bloques)
        idx = {b: np.where(bloques == b)[0] for b in ids}
        d = []
        for _ in range(N_BOOT):
            sel = np.concatenate([idx[b] for b in rng.choice(ids, len(ids), replace=True)])
            yy = y_te[sel]
            if 0 < yy.mean() < 1:
                d.append(roc_auc_score(yy, p_te[sel]) - roc_auc_score(yy, p3[sel]))
        res["v4_menos_v3"] = {
            "delta_auc_roc": round(float(res["v4"]["auc_roc"]
                                         - res["v3_en_test_v4"]["auc_roc"]), 4),
            "ic95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)]}
        print(f"  v3 (mismo test)    AUC-ROC {res['v3_en_test_v4']['auc_roc']:.4f} "
              f"{res['v3_en_test_v4']['ic95_auc_roc']} · "
              f"AUC-PR {res['v3_en_test_v4']['auc_pr']:.4f}")
        ic = res["v4_menos_v3"]["ic95"]
        veredicto = ("v4 MEJOR" if ic[0] > 0 else
                     "v3 MEJOR" if ic[1] < 0 else "empate (IC cruza 0)")
        print(f"  → v4 − v3 = {res['v4_menos_v3']['delta_auc_roc']:+.4f} "
              f"IC95 {ic}  {veredicto}")

    # ---- cortes de alerta, derivados de VAL ------------------------------
    cortes, detalle = {}, {}
    for nivel, cob in COBERTURA.items():
        u = float(np.quantile(p_va, 1 - cob))
        cortes[nivel] = round(u, 4)
        pred = (p_te >= u).astype(int)
        tp = int(((pred == 1) & (y_te == 1)).sum())
        detalle[nivel] = {
            "corte": round(u, 4), "cobertura_val": cob,
            "precision_test": round(tp / max(pred.sum(), 1), 4),
            "recall_test": round(tp / max(int(y_te.sum()), 1), 4)}
    print("\n=== cortes de alerta (umbral de VAL, cobertura de v2/v3) ===")
    for k, v in detalle.items():
        print(f"  {k:<9} corte {v['corte']:.4f} · precisión {v['precision_test']:.3f} "
              f"· recall {v['recall_test']:.3f}")

    modelo.save_model(str(SAL_MODELO))
    imp = modelo.get_booster().get_score(importance_type="gain")
    tot = sum(imp.values()) or 1
    top = sorted(imp.items(), key=lambda x: -x[1])[:15]
    print("\ntop features (gain %):",
          ", ".join(f"{k} {100*v/tot:.1f}" for k, v in top))

    meta = {
        "version": "v4",
        "descripcion": "EGIF de Civio consolidado (09/07/2026), split "
                       "train 2015-2020 / val 2021 / test 2022",
        "features": FEATS, "n_features": len(FEATS),
        "hiperparametros": elegido,
        "params": PARAMS_TUNED if elegido == "tuned" else PARAMS_BASE,
        "n_estimators": n_arboles,
        "entrenado_con": {"anios": "2015-2020", "n_filas": int(len(tr))},
        "cortes_alerta": cortes, "cortes_alerta_detalle": detalle,
        "metricas_test_2022": res,
        "advertencias_de_uso": [
            "El test 2022 se ha usado UNA vez. Cualquier decisión posterior "
            "tomada mirándolo lo invalida como examen.",
            "Los cortes son los de este modelo; no reutilizar los de v2 ni v3.",
            "v3 se evalúa sobre las filas del muestreo v4: la comparación es "
            "válida en esa dirección, no en la inversa.",
            "2022 excluye Navarra y Cantabria por el filtro de frontera; el "
            "test no cubre la cornisa cantábrica oriental."]}
    SAL_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    SAL_METRICAS.write_text(json.dumps(
        {"protocolo": "train 2015-2020 / val 2021 / test 2022",
         "seleccion": {"criterio": "AP en val 2021", "elegido": elegido,
                       "candidatos": {k: round(float(v[1]), 4) for k, v in cand.items()}},
         "test_2022": res,
         "bootstrap": {"n": N_BOOT, "agrupado_por": "bloque_100km"},
         "importancia_gain_pct": {k: round(100 * v / tot, 2) for k, v in top}},
        indent=2, ensure_ascii=False))
    print(f"\nguardado: {SAL_MODELO}\n          {SAL_META}\n          {SAL_METRICAS}")


if __name__ == "__main__":
    main()
