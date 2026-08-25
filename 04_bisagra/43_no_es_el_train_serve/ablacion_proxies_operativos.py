#!/usr/bin/env python3
"""
Ablación de los PROXIES OPERATIVOS — cuánto cuesta que producción no tenga
los datos que el modelo vio al entrenar.

NO SOBRESCRIBE NADA. Escribe solo `dataset/ablacion_proxies_operativos.json`.

=============================================================================
LA PREGUNTA
=============================================================================
El modelo se entrena con el cubo IberFire, donde NDVI, LAI, SWI010 y LST
varían día a día y los rayos (WGLC) son reales. En producción (2026) nada de
eso existe:

  · satélite → climatología mensual congelada de la celda (MODELOS_Y_FEATURES §2a)
  · rayos    → 0.0 fijo (WGLC acaba en 2023)

La validación operativa da AUC-ROC 0,64 frente al 0,923 del test 2020. Buena
parte de ese hueco es definicional (prevalencia 1,1% vs 25% de diseño, etiqueta
distinta, unidad estación±25 km vs celda de 1 km). Este script mide la parte
que SÍ es degradación de datos, en el único sitio donde se puede medir: el
entorno de entrenamiento, donde tenemos las dos versiones de cada feature.

=============================================================================
LAS DOS PREGUNTAS QUE HAY QUE SEPARAR (y que casi nadie separa)
=============================================================================
Congelar una feature en inferencia tiene DOS costes distintos:

  (1) COSTE DE INFORMACIÓN — la feature deja de traer la anomalía diaria.
      Inevitable: el dato no existe.
  (2) COSTE DE DESAJUSTE   — el modelo aprendió a leer una distribución y en
      inferencia le llega otra. EVITABLE: basta reentrenar con el proxy.

Producción hoy paga los dos: `xgb_v2` se entrenó con satélite real y en 2026
se le sirve climatología. Por eso cada variante se corre en dos modos:

  MISMATCH  — entrena con el dato real, evalúa con el proxy  → lo que pasa HOY
  COHERENTE — entrena y evalúa con el proxy                  → lo que pasaría
                                                                si se reentrena

Si COHERENTE > MISMATCH, hay puntos de AUC tirados por no reentrenar, y eso es
una recomendación accionable para la memoria, no una limitación a declarar.

=============================================================================
CÓMO SE CONSTRUYE LA CLIMATOLOGÍA (y por qué es un límite superior)
=============================================================================
Producción congela la media mensual DE LA CELDA (2020-24, precalculada del
cubo). Aquí no se puede reproducir exactamente: el dataset es una muestra y la
mediana de filas por (celda, mes) es 1 — la media de la celda sería el propio
valor y la ablación no haría nada. Se aproxima agregando espacialmente:

    (10 km, mes) → (25 km, mes) → (bloque 100 km, mes) → (mes)

bajando de nivel solo si el grupo tiene <3 filas. Estimada con 2015-2019
(train+val), NUNCA con 2020, para que el test siga siendo limpio.

⚠️ Esta aproximación destruye MÁS información que producción: elimina la
anomalía diaria e interanual (como producción) pero además difumina el detalle
espacial dentro de 10-25 km (producción lo conserva, es climatología por
celda). Por tanto **el coste medido es un LÍMITE SUPERIOR** del coste real de
congelar. Se acota por abajo con la variante SIN, que elimina la feature del
todo: congelar no puede costar más que eliminar (en modo coherente).

=============================================================================
PROTOCOLO
=============================================================================
El congelado de v1/v2/v3: train 2015-2018 · val 2019 (early stopping) · test
2020. Mismas 46 features e hiperparámetros que el gemelo de protocolo cuyo
número (0,9234 ROC / 0,8316 PR) figura en xgb_v3_metadata.json — la variante
CONTROL debe reproducirlo, y si no lo hace el resto no vale nada.

IC95 por bootstrap agrupado por BLOQUE de 100 km, no por fila: dentro de un
bloque las celdas comparten meteo, vegetación e historial de fuego, y
remuestrear filas independientes fingiría una precisión que no hay.

Uso:  /home/charredgem/miniconda3/envs/tfm_fuego/bin/python ablacion_proxies_operativos.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

DIR = Path("/home/charredgem/Desktop/Master/TFM_fuego")
SEED = 42
N_BOOT = 1000

RUTA_DATASET = DIR / "dataset" / "dataset_modelo_v1.parquet"
RUTA_FEATS = DIR / "modelos" / "xgb_v2_prototipo_features.json"
SALIDA = DIR / "dataset" / "ablacion_proxies_operativos.json"

# Las que producción sirve como climatología mensual congelada. `ndvi_med_30d`
# entra aquí porque ranking_diario.py le asigna el MISMO valor mensual que a
# `ndvi` (no una media de 30 días real): en producción son la misma columna.
SATELITE = ["ndvi", "ndvi_med_30d", "lai", "swi010", "lst"]
CLIM_ORIGEN = {"ndvi": "ndvi", "ndvi_med_30d": "ndvi",   # comparten proxy
               "lai": "lai", "swi010": "swi010", "lst": "lst"}
RAYOS = ["rayos_dia", "rayos_7d"]

PARAMS = dict(learning_rate=0.01704120432383672, max_depth=10,
              min_child_weight=3, subsample=0.8918952588803617,
              colsample_bytree=0.5031060958594304,
              reg_lambda=0.7155371349132547, gamma=0.010555378164328569)
COMUNES = dict(tree_method="hist", eval_metric="aucpr", enable_categorical=True,
               random_state=SEED, n_jobs=20)


# --------------------------------------------------------------------------- #
# datos
# --------------------------------------------------------------------------- #
def cargar():
    df = pd.read_parquet(RUTA_DATASET)
    df["ccaa"] = df["ccaa"].fillna(-1).astype(int)
    df["es_festivo"] = df["es_festivo"].astype(float)
    return df


def climatologia(df, feats):
    """Media mensual por vecindad espacial, estimada SIN el año de test.

    Devuelve un DataFrame con una columna por feature, alineado con `df`, y el
    reparto de filas por nivel de agregación usado (para poder declararlo)."""
    base = df[df["split"] != "test"]
    out = pd.DataFrame(index=df.index)
    niveles = {}

    claves = [("10km", ["bx10", "by10", "mes"]), ("25km", ["bx25", "by25", "mes"]),
              ("100km", ["bloque_100km", "mes"]), ("mes", ["mes"])]
    for d, pre in [(10, "10"), (25, "25")]:
        for eje in ("ix", "iy"):
            df[f"b{eje[1]}{pre}"] = df[eje] // d
            base = base.assign(**{f"b{eje[1]}{pre}": base[eje] // d})

    for f in feats:
        val = pd.Series(np.nan, index=df.index)
        usado = pd.Series("", index=df.index)
        for nombre, cols in claves:
            falta = val.isna()
            if not falta.any():
                break
            g = base.groupby(cols)[f].agg(["mean", "count"])
            g = g[g["count"] >= 3]["mean"] if nombre != "mes" else g["mean"]
            m = df.loc[falta, cols].merge(g.rename("v"), left_on=cols,
                                          right_index=True, how="left")["v"]
            m.index = df.index[falta]
            val[falta] = m
            usado[falta & val.notna()] = nombre
        out[f] = val
        niveles[f] = usado.value_counts(normalize=True).round(3).to_dict()
    return out, niveles


def aplicar(df, clim, congelar_sat, rayos_cero):
    """Copia de `df` con los proxies operativos aplicados."""
    d = df.copy()
    if congelar_sat:
        for f in SATELITE:
            d[f] = clim[CLIM_ORIGEN[f]].values
    if rayos_cero:
        for f in RAYOS:
            d[f] = 0.0
    return d


# --------------------------------------------------------------------------- #
# entrenamiento y métricas
# --------------------------------------------------------------------------- #
def ajustar(feats, tr, va):
    m = xgb.XGBClassifier(**COMUNES, **PARAMS, n_estimators=3000,
                          early_stopping_rounds=100)
    m.fit(tr[feats], tr["label"], eval_set=[(va[feats], va["label"])],
          verbose=False)
    return m


def evaluar(y, p):
    prev = float(np.mean(y))
    return {"auc_roc": float(roc_auc_score(y, p)),
            "auc_pr": float(average_precision_score(y, p)),
            "ap_norm": float((average_precision_score(y, p) - prev) / (1 - prev)),
            "brier": float(brier_score_loss(y, p))}


def boot_delta(y, p_ref, p_var, bloques, rng):
    """IC95 de AUC-ROC(variante) − AUC-ROC(control), remuestreando BLOQUES."""
    ids = np.unique(bloques)
    idx_por_bloque = {b: np.where(bloques == b)[0] for b in ids}
    d = []
    for _ in range(N_BOOT):
        sel = np.concatenate([idx_por_bloque[b]
                              for b in rng.choice(ids, len(ids), replace=True)])
        yy = y[sel]
        if yy.min() == yy.max():
            continue
        d.append(roc_auc_score(yy, p_var[sel]) - roc_auc_score(yy, p_ref[sel]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --------------------------------------------------------------------------- #
def main():
    feats = json.loads(RUTA_FEATS.read_text())
    df = cargar()
    tr0 = df[df.split == "train"]
    va0 = df[df.split == "val"]
    te0 = df[df.split == "test"]
    print(f"protocolo congelado — train {len(tr0)} · val {len(va0)} · "
          f"test {len(te0)} (prevalencia test {te0.label.mean():.1%})\n")

    clim, niveles = climatologia(df, ["ndvi", "lai", "swi010", "lst"])
    print("climatología: nivel de agregación usado por feature")
    for f, n in niveles.items():
        print(f"   {f:8s} {n}")
    cob = {f: float(clim[f].notna().mean()) for f in clim}
    print(f"cobertura (sin NaN): { {k: round(v,3) for k,v in cob.items()} }\n")

    # correlación entre el valor real y su proxy: cuánta señal queda
    corr = {f: float(pd.Series(df[f]).corr(clim[CLIM_ORIGEN[f]]))
            for f in SATELITE}
    print("corr(valor real, climatología) en todo el dataset:")
    for f, c in corr.items():
        print(f"   {f:14s} r={c:.3f}   (r²={c*c:.3f} de la varianza retenida)")
    print()

    variantes = [
        ("CONTROL", False, False, False, "todo real (debe dar ~0,923 ROC)"),
        ("SAT mismatch", True, False, False, "entrena real, evalúa congelado"),
        ("SAT coherente", True, False, True, "entrena y evalúa congelado"),
        ("SAT eliminado", None, False, True, "las 5 features fuera del modelo"),
        ("RAYOS mismatch", False, True, False, "entrena real, evalúa a 0"),
        ("RAYOS coherente", False, True, True, "entrena y evalúa a 0"),
        ("TOTAL mismatch", True, True, False, "el escenario de producción HOY"),
        ("TOTAL coherente", True, True, True, "producción tras reentrenar"),
    ]

    rng = np.random.default_rng(SEED)
    bloques = te0["bloque_100km"].values
    y_te = te0["label"].values
    res, p_control = {}, None

    for nombre, sat, ray, coherente, nota in variantes:
        if sat is None:                       # variante "eliminado"
            f_use = [f for f in feats if f not in SATELITE]
            tr, va, te = tr0, va0, te0
        else:
            f_use = feats
            if coherente:
                tr = aplicar(tr0, clim.loc[tr0.index], sat, ray)
                va = aplicar(va0, clim.loc[va0.index], sat, ray)
            else:
                tr, va = tr0, va0
            te = aplicar(te0, clim.loc[te0.index], sat, ray)

        m = ajustar(f_use, tr, va)
        p = m.predict_proba(te[f_use])[:, 1]
        met = evaluar(y_te, p)
        met["n_arboles"] = int(m.best_iteration + 1)
        met["nota"] = nota
        if p_control is None:
            p_control = p
            met["delta_auc_roc"] = 0.0
            met["ic95_delta"] = [0.0, 0.0]
        else:
            met["delta_auc_roc"] = met["auc_roc"] - res["CONTROL"]["auc_roc"]
            met["ic95_delta"] = list(boot_delta(y_te, p_control, p, bloques, rng))
        res[nombre] = met
        ic = met["ic95_delta"]
        print(f"{nombre:16s} ROC {met['auc_roc']:.4f}  PR {met['auc_pr']:.4f}  "
              f"Δ{met['delta_auc_roc']:+.4f}  IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}]"
              f"  ({met['n_arboles']} árboles)")

    SALIDA.write_text(json.dumps(
        {"protocolo": {"train": "2015-2018", "val": "2019", "test": "2020",
                       "n_test": int(len(te0)),
                       "prevalencia_test": float(te0.label.mean())},
         "climatologia": {"niveles": niveles, "cobertura": cob,
                          "corr_real_vs_proxy": corr,
                          "estimada_con": "2015-2019 (sin el año de test)",
                          "aviso": "aproximación por vecindad espacial: "
                                   "destruye más información que la "
                                   "climatología por celda de producción, "
                                   "luego el coste medido es un límite superior"},
         "bootstrap": {"n": N_BOOT, "agrupado_por": "bloque_100km"},
         "resultados": res}, indent=2, ensure_ascii=False))
    print(f"\nguardado: {SALIDA}")


if __name__ == "__main__":
    main()
