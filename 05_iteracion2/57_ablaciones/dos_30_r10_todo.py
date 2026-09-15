#!/usr/bin/env python3
"""
Dos modelos, paso 30: r10 reentrenado con todos los años, para medir en
2025-2026.

No toca nada de lo que hay. Lee en solo lectura `dataset_ratio.parquet` y
`celdas.parquet` del USB; escribe únicamente en el `salida/` del sandbox.
En particular no escribe en `<externo>/modelos/`, así que
`donde_dia_effis_r10.ubj`, el que sirve la cadena, queda intacto.

Qué pregunta contesta
---------------------
El r10 que se sirve está entrenado con `SPLITS` de `dos_12`:

    train 2015-2022 · val 2023 · test 2024

o sea ocho años: 2023 y 2024 no entran en el entrenamiento, se reservan.
Eso era lo correcto mientras el test fuese 2024. Pero ahora existe una
evaluación posterior e independiente (el replay de 2025 y 2026, con datos que
no están en el cubo), así que se puede usar todo 2015-2024 para entrenar y
medir fuera, en 2025-2026. Son dos años más de datos, un 25 % más, e incluyen
2022 (año récord) ya presente y 2023-2024, que hoy se tiran.

Cómo, para que la comparación sea limpia
----------------------------------------
· Mismos datos de partida: `dataset_ratio.parquet`, que ya trae los diez años
  con su columna `split`. No hay que volver a muestrear ni a extraer features,
  así que el muestreo es idéntico y no introduce ruido de sorteo.
· Mismos hiperparámetros y mismo número de árboles. `dos_18` ya quita el early
  stopping y fija `n_estimators = N_ARBOLES = 441` justamente para que las
  comparaciones no se confundan con el número de árboles. Aquí se hereda esa
  decisión, que además resuelve el problema de no tener val al entrenar con
  todo.
· Se entrenan dos modelos con el mismo código:
      r10_base  train = 2015-2022        (réplica del que se sirve)
      r10_todo  train = 2015-2024        (el candidato)
  La réplica existe para verificar: si `r10_base` no reproduce al que ya está
  en el USB, la comparación no vale y hay que parar.

Uso:
    python dos_30_r10_todo.py
"""

import json

import numpy as np
import pandas as pd
import xgboost as xgb

import config
import config_expansion as ce
from dos_05_modelos import CLIMA, FULL, GEO, HIST, PARAMS
from dos_18_ratio import N_ARBOLES, peldano

SAL = config.salida("dos_30")
K = 10                                   # el peldaño 1:10
DATASET = f"{ce.DATASET}/dataset_ratio.parquet"


def carga():
    cols_cel = ["iy", "ix"] + CLIMA + GEO + HIST + [
        "elev_std", "dist_ferrocarril", "is_natura2000", "clc_arable",
        "clc_urbano", "clc_perm", "egif_1518_10km"] + [f"aspect_{a}" for a in range(1, 9)]
    cel = pd.read_parquet(f"{ce.DATASET}/celdas.parquet")[cols_cel]
    df = pd.read_parquet(DATASET).merge(cel, on=["iy", "ix"], how="left",
                                        validate="m:1")
    df["anio"] = pd.to_datetime(df["fecha"]).dt.year
    return df


def entrena(tr, nombre):
    par = dict(PARAMS)
    par.pop("early_stopping_rounds")
    par["n_estimators"] = N_ARBOLES
    m = xgb.XGBClassifier(**par)
    m.fit(tr[FULL].values, tr["label"].values, verbose=False)
    f = f"{SAL}_{nombre}.ubj"
    m.save_model(f)
    npos = int((tr["label"] == 1).sum())
    print(f"  {nombre:9} n={len(tr):>9,} · positivos {npos:>7,} · "
          f"años {tr.anio.min()}-{tr.anio.max()} → {f}", flush=True)
    return m


def main():
    df = carga()
    print(f"escalera {df.shape} · años {df.anio.min()}-{df.anio.max()}", flush=True)

    base_tr = peldano(df[df["split"] == "train"], K)          # 2015-2022
    todo_tr = peldano(df, K)                                  # 2015-2024
    print("\nentrenando (mismos hiperparámetros, "
          f"{N_ARBOLES} árboles fijos, sin early stopping):")
    m_base = entrena(base_tr, "r10_base")
    m_todo = entrena(todo_tr, "r10_todo")

    # ---- verificación: ¿la réplica reproduce al que se sirve? -------------
    print("\n=== verificación: r10_base contra el donde_dia_effis_r10 del USB ===")
    viejo = xgb.XGBClassifier()
    viejo.load_model(f"{ce.MODELOS}/donde_dia_effis_r10.ubj")
    muestra = df.sample(n=min(200_000, len(df)), random_state=0)
    X = muestra[FULL].values
    p_viejo = viejo.predict_proba(X)[:, 1]
    p_base = m_base.predict_proba(X)[:, 1]
    p_todo = m_todo.predict_proba(X)[:, 1]
    dmax = float(np.abs(p_viejo - p_base).max())
    print(f"  |Δ| máxima  base vs servido : {dmax:.2e}")
    print(f"  Spearman    base vs servido : "
          f"{pd.Series(p_viejo).corr(pd.Series(p_base), method='spearman'):.6f}")
    print(f"  Spearman    todo vs servido : "
          f"{pd.Series(p_viejo).corr(pd.Series(p_todo), method='spearman'):.6f}")
    print("  " + ("OK: la réplica es equivalente, la comparación vale."
                  if dmax < 1e-5 else
                  "AVISO: la réplica NO reproduce al servido. Revisar antes "
                  "de comparar (¿otro dataset, otro nº de árboles?)."))

    json.dump(dict(n_arboles=N_ARBOLES, k=K,
                   n_base=int(len(base_tr)), n_todo=int(len(todo_tr)),
                   pos_base=int((base_tr.label == 1).sum()),
                   pos_todo=int((todo_tr.label == 1).sum()),
                   delta_max_base_vs_servido=dmax),
              open(f"{SAL}_info.json", "w"), indent=1)
    print(f"\n→ {SAL}_r10_base.ubj · {SAL}_r10_todo.ubj · {SAL}_info.json")
    print("\nSIGUIENTE: evaluar los dos en el replay 2025-2026. NO se ha tocado "
          "ni el modelo servido ni la cadena.")


if __name__ == "__main__":
    main()
