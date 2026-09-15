#!/usr/bin/env python3
"""
Dos modelos, paso 35: ¿ordena mejor las celdas de cada día otro algoritmo?

No toca producción. Lee el disco externo; escribe salida/dos_35_algoritmos.json
y salida/dos_35_algoritmos.csv. No guarda modelos.

Por qué
-------
El r10 es un XGBoost, y en el trabajo nunca se entrenó otro algoritmo con los
mismos datos. Este paso lo compara con LightGBM, Random Forest y regresión
logística sin cambiar nada más: los datos, las 46 variables y la evaluación
son los del r10. El protocolo quedó escrito antes de ejecutar nada en
docs/TRAZABILIDAD.md («Prerregistro: comparación de algoritmos»).

Diseño
------
Entrenamiento: `dataset_ratio.parquet`, 2015-2022, con el peldaño 1:10 de
`dos_18_ratio.py` (los positivos y los diez primeros negativos vivos del
mismo día). Banco: el `eval_dia` de `dataset_effis.parquet`. La configuración
de cada algoritmo se elige por el AUC medio dentro del día de 2023, entre seis
candidatas; la prueba de 2024 solo se puntúa con la configuración elegida.

La primera fila es XGBoost con la receta del r10 tal cual (441 árboles). Si no
reproduce el 0,8045 documentado en 2024 (±0,005), el script para: la
comparación no valdría.

Ningún algoritmo pondera clases, igual que el r10. Semilla 42 y 16 hilos.

LightGBM no está en el entorno `tfm_fuego`; se instala en un entorno aparte
con acceso a los paquetes del sistema. Si no está, se usa
HistGradientBoostingClassifier de sklearn y la salida lo dice.

Uso:
    python -m venv --system-site-packages venv_lgbm && venv_lgbm/bin/pip install lightgbm
    source entorno.sh
    venv_lgbm/bin/python 05_iteracion2/57_ablaciones/dos_35_algoritmos.py
"""

import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
import config_expansion as ce
from dos_05_modelos import FULL, PARAMS, SEED, metricas_dia
from dos_18_ratio import DATASET, N_ARBOLES, peldano

try:
    import lightgbm as lgb
except ImportError:          # sin LightGBM, el sustituto de sklearn
    lgb = None

K = 10                        # el peldaño del r10
AUC_R10_2024 = 0.8045         # dos_18_ratio.json, donde_dia_effis_r10, 2024
TOLERANCIA = 0.005
HILOS = 16
N_BOOT = 2000
CATEGORICAS = ["ccaa", "mes"]


def xgb_r10(**cambios):
    par = dict(PARAMS)
    par.pop("early_stopping_rounds")
    par.update(n_estimators=N_ARBOLES, n_jobs=HILOS)
    par.update(cambios)
    return xgb.XGBClassifier(**par)


def lgbm(num_leaves, learning_rate):
    if lgb is None:
        return HistGradientBoostingClassifier(
            max_iter=N_ARBOLES, max_leaf_nodes=num_leaves,
            learning_rate=learning_rate, random_state=SEED)
    return lgb.LGBMClassifier(
        n_estimators=N_ARBOLES, num_leaves=num_leaves, learning_rate=learning_rate,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=20, reg_lambda=1.0, random_state=SEED,
        n_jobs=HILOS, verbose=-1)


def bosque(min_samples_leaf, max_features):
    return RandomForestClassifier(
        n_estimators=400, min_samples_leaf=min_samples_leaf,
        max_features=max_features, n_jobs=HILOS, random_state=SEED)


def logistica(C):
    num = [c for c in FULL if c not in CATEGORICAS]
    idx_num = [FULL.index(c) for c in num]
    idx_cat = [FULL.index(c) for c in CATEGORICAS]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("esc", StandardScaler())]), idx_num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), idx_cat)])
    return Pipeline([("prep", prep),
                     ("lr", LogisticRegression(C=C, max_iter=3000))])


REJILLAS = {
    "XGBoost ajustado": [(dict(max_depth=d, learning_rate=lr),
                          lambda d=d, lr=lr: xgb_r10(max_depth=d, learning_rate=lr))
                         for d in (4, 6, 8) for lr in (0.05, 0.1)],
    "LightGBM": [(dict(num_leaves=nl, learning_rate=lr),
                  lambda nl=nl, lr=lr: lgbm(nl, lr))
                 for nl in (31, 63, 127) for lr in (0.05, 0.1)],
    "Random Forest": [(dict(min_samples_leaf=m, max_features=f),
                       lambda m=m, f=f: bosque(m, f))
                      for m in (1, 5, 20) for f in ("sqrt", 0.3)],
    "Regresión logística": [(dict(C=c), lambda c=c: logistica(c))
                            for c in (0.001, 0.01, 0.1, 1, 10, 100)],
}


def auc_dias(ev, score, anio):
    e = ev[ev["anio"] == anio]
    return metricas_dia(e, score[e.index.values], "effis").set_index("fecha")


def ajusta(fabrica, X, y):
    m = fabrica()
    t0 = time.perf_counter()
    m.fit(X, y)
    return m, time.perf_counter() - t0


def puntua(m, X):
    t0 = time.perf_counter()
    s = m.predict_proba(X)[:, 1]
    return s, time.perf_counter() - t0


def resumen(t, ref, rng):
    com = t.index.intersection(ref.index)
    d = (t.loc[com, "auc"] - ref.loc[com, "auc"]).values
    a = t["auc"].values
    ba = a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(1)
    bd = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return {"n_dias": int(len(t)), "n_pos": int(t["n_pos"].sum()),
            "auc_medio": float(a.mean()),
            "ic95": [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))],
            "dif_vs_ref": float(d.mean()),
            "ic95_dif": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
            "dias_gana": float((d > 0).mean())}


def main():
    cols = FULL + ["fecha", "split", "label", "k_neg", "id_pos"]
    df = pd.read_parquet(DATASET, columns=cols)
    tr = peldano(df[df["split"] == "train"], K)
    X, y = tr[FULL].values.astype(np.float32), tr["label"].values
    del df
    ev = pd.read_parquet(f"{ce.DATASET}/dataset_effis.parquet")
    ev = ev[ev["disenio"] == "eval_dia"].reset_index(drop=True)
    Xe = ev[FULL].values.astype(np.float32)
    es23 = (ev["anio"] == 2023).values
    print(f"entrenamiento {X.shape} · {int(y.sum()):,} positivos · banco {ev.shape} · "
          f"LightGBM {'sí' if lgb is not None else 'no (HistGradientBoosting)'}", flush=True)

    rng = np.random.default_rng(SEED)
    R = {"receta": {"dataset": "dataset_ratio.parquet", "peldano": K,
                    "filas_train": int(len(X)), "positivos": int(y.sum()),
                    "variables": len(FULL), "n_arboles_boosting": N_ARBOLES,
                    "hilos": HILOS, "semilla": SEED,
                    "lightgbm": None if lgb is None else lgb.__version__,
                    "xgboost": xgb.__version__},
         "algoritmos": {}}
    dias = []

    # --- referencia: la receta del r10, que tiene que reproducir la cifra ---
    m, t_fit = ajusta(xgb_r10, X, y)
    s, t_pred = puntua(m, Xe)
    ref = {a: auc_dias(ev, s, a) for a in (2023, 2024)}
    auc24 = float(ref[2024]["auc"].mean())
    servido = xgb.XGBClassifier()
    servido.load_model(f"{ce.MODELOS}/donde_dia_effis_r{K}.ubj")
    dmax = float(np.abs(servido.predict_proba(Xe)[:, 1] - s).max())
    print(f"XGBoost receta r10: AUC 2024 {auc24:.4f} (documentado {AUC_R10_2024}) · "
          f"Δ máx con el .ubj servido {dmax:.2e} · {t_fit:.0f} s", flush=True)
    R["reproduccion"] = {"auc_2024": auc24, "documentado": AUC_R10_2024,
                         "dif_max_prediccion_vs_ubj": dmax}
    if abs(auc24 - AUC_R10_2024) > TOLERANCIA:
        json.dump(R, open(config.salida("dos_35_algoritmos.json"), "w"), indent=1)
        raise SystemExit("La receta del r10 no reproduce la cifra documentada: se para.")
    nombre_ref = "XGBoost (receta del r10)"
    R["algoritmos"][nombre_ref] = {
        "rejilla": [], "elegida": {"n_estimators": N_ARBOLES},
        "auc_2023": float(ref[2023]["auc"].mean()),
        "t_entrenamiento_s": t_fit, "t_prediccion_s": t_pred,
        "2024": resumen(ref[2024], ref[2024], rng)}
    for a in (2023, 2024):
        dias.append(ref[a].reset_index().assign(anio=a, algoritmo=nombre_ref))

    # --- rejillas: elección con 2023, prueba con 2024 ---
    for nombre, rejilla in REJILLAS.items():
        filas, mejor = [], None
        for params, fabrica in rejilla:
            m, t_fit = ajusta(fabrica, X, y)
            s23, t_pred = puntua(m, Xe[es23])
            s = np.full(len(ev), np.nan)
            s[es23] = s23
            auc23 = float(auc_dias(ev, s, 2023)["auc"].mean())
            filas.append({"params": params, "auc_2023": auc23,
                          "t_entrenamiento_s": t_fit, "t_prediccion_2023_s": t_pred})
            print(f"  {nombre:20s} {params} · AUC 2023 {auc23:.4f} · {t_fit:.0f} s",
                  flush=True)
            if mejor is None or auc23 > mejor[1]:
                mejor = (params, auc23, m, t_fit)
            del m
        params, auc23, m, t_fit = mejor
        s, t_pred = puntua(m, Xe)
        t = {a: auc_dias(ev, s, a) for a in (2023, 2024)}
        R["algoritmos"][nombre] = {
            "rejilla": filas, "elegida": params, "auc_2023": auc23,
            "t_entrenamiento_s": t_fit, "t_prediccion_s": t_pred,
            "2024": resumen(t[2024], ref[2024], rng)}
        r = R["algoritmos"][nombre]["2024"]
        print(f"{nombre}: elegida {params} · AUC 2024 {r['auc_medio']:.4f} · "
              f"Δ {r['dif_vs_ref']:+.4f} [{r['ic95_dif'][0]:+.4f}, {r['ic95_dif'][1]:+.4f}] · "
              f"gana {r['dias_gana']*100:.0f} %", flush=True)
        for a in (2023, 2024):
            dias.append(t[a].reset_index().assign(anio=a, algoritmo=nombre))
        del m

    json.dump(R, open(config.salida("dos_35_algoritmos.json"), "w"), indent=1,
              ensure_ascii=False)
    pd.concat(dias)[["algoritmo", "anio", "fecha", "n_pos", "auc", "lift"]].to_csv(
        config.salida("dos_35_algoritmos.csv"), index=False)
    print(f"\nguardado {config.salida('dos_35_algoritmos.json')} y .csv")


if __name__ == "__main__":
    main()
