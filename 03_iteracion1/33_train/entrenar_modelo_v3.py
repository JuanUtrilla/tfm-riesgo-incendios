#!/usr/bin/env python3
"""
Modelo B v3: modelo de producción reajustado con 2015-2020 completo.

No sobrescribe nada. Escribe solo ficheros con sufijo _v3.

Qué cambia respecto a v1/v2 y por qué

1. Reajuste con todos los datos (§10.1)
   v1/v2 se entrenaron solo con train 2015-2018 (53.563 filas). Val 2019 (14.489)
   y test 2020 (8.614) quedaban fuera: 23.103 filas = +43% de datos sin usar, y
   los dos años más recientes. Para el protocolo de evaluación ese sacrificio
   es obligatorio; para el modelo que sale a producción no aporta nada.
   Práctica estándar: evaluar con el protocolo congelado, reportar ese número,
   y desplegar un modelo reajustado con todo.

   Regla que no se rompe: el modelo final (entrenado con 2015-2020) no se
   evalúa sobre 2020; sería fraudulento. Las métricas que ampara son las de la
   fase 1 (protocolo congelado), calculadas con el mismo conjunto de features y
   los mismos hiperparámetros, pero con el modelo que sí respetó el split.

2. Sustitución de la autorregresiva caducada (§10.3.c)
   `n_fuegos_10km_mismomes_hist` (EGIF) es la nº1 en SHAP y en producción está
   congelada en 2020: los incendios de 2021-2026 no existen para el modelo, y la
   degradación crece cada año. Se prueba sustituirla por su equivalente FIRMS
   (extraer_historia_firms_v3.py), que sí se puede actualizar indefinidamente.
   El cambio de fuente exige reentrenar con la misma definición: de lo contrario
   el modelo aplicaría lo aprendido sobre una distribución a otra distinta y
   fallaría en silencio (bitácora §16).
   Se decide con datos, comparando en val, no por preferencia a priori.

3. Selección de n_estimators sin val (§10.1-a)
   Al entrar 2019 en entrenamiento ya no hay conjunto donde parar. Se usa
   validación cruzada temporal (rolling origin): 2015-16→17, 2015-17→18,
   2015-18→19, 2015-19→20. Da la iteración óptima media y una estimación de la
   varianza año a año (material para la memoria).

4. Cortes de alerta recalculados out-of-fold (§10.1-b)
   Los cortes de v2 (0,25 / 0,55 / 0,80) salieron de los cuantiles de las
   predicciones en val 2019. Si 2019 pasa a entrenamiento esas predicciones son
   in-sample, están infladas, y los mismos cortes numéricos harían que el sistema
   avisase de menos. Los cortes de v3 se derivan de las predicciones out-of-fold
   de la CV temporal, fijando la misma cobertura operativa que v2
   (34,8% / 22,0% / 13,5% de días-celda) para no alterar el comportamiento del
   sistema de alertas, y se reportan precisión y recall resultantes.

Salidas (todas nuevas):
  modelos/xgb_v3.ubj                modelo de producción (2015-2020 completo)
  modelos/xgb_v3_metadata.json      features, n_estimators, cortes, procedencia
  dataset/metricas_v3.json          protocolo congelado + CV temporal + cortes
  eda/pr_curves_v3.png, eda/shap_summary_v3.png
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

RAIZ = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../..")
DIR = os.environ.get("TFM_DATOS", f"{RAIZ}/datos")
SEED = 42

RUTA_DATASET = f"{DIR}/dataset/dataset_modelo_v1.parquet"
RUTA_FIRMS_HIST = f"{DIR}/dataset/features_firms_hist_v3.parquet"
RUTA_FEATS_V2 = f"{DIR}/modelos/xgb_v2_prototipo_features.json"

# Cobertura operativa de los cortes de alerta de v2 (bitácora §15), que v3
# reproduce para no alterar el comportamiento del sistema de avisos.
COBERTURA_V2 = {"MODERADO": 0.348, "ALTO": 0.220, "EXTREMO": 0.135}

# Feature que se intenta sustituir por su equivalente FIRMS (§10.3.c)
FEAT_EGIF_CADUCADA = "n_fuegos_10km_mismomes_hist"
FEATS_FIRMS_HIST = ["firms_diasfuego_mismomes_tasa", "firms_diasfuego_tasa",
                    "firms_frp_p95_hist", "firms_anios_previos"]

# Hiperparámetros candidatos: los de v1 y los que encontró Optuna
# (dataset/tuning_optuna_v1.json). Se elige mirando val 2019, nunca test.
PARAMS_BASE = dict(learning_rate=0.05, max_depth=6, min_child_weight=5,
                   subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0)
PARAMS_TUNED = dict(learning_rate=0.01704120432383672, max_depth=10,
                    min_child_weight=3, subsample=0.8918952588803617,
                    colsample_bytree=0.5031060958594304,
                    reg_lambda=0.7155371349132547, gamma=0.010555378164328569)
COMUNES = dict(tree_method="hist", eval_metric="aucpr", enable_categorical=True,
               random_state=SEED, n_jobs=20)


# --------------------------------------------------------------------------- #
# utilidades
# --------------------------------------------------------------------------- #
def cargar() -> pd.DataFrame:
    df = pd.read_parquet(RUTA_DATASET)
    firms = pd.read_parquet(RUTA_FIRMS_HIST)
    n0 = len(df)
    df = df.merge(firms, on="id_muestra", how="left", validate="one_to_one")
    assert len(df) == n0, "el join con FIRMS no es 1:1"
    df["ccaa"] = df["ccaa"].fillna(-1).astype(int)
    df["es_festivo"] = df["es_festivo"].astype(float)
    return df


def ajustar(feats, params, tr, va, n_estimators=3000, early=100):
    """Entrena con early stopping en `va`. Si early es None, nº de árboles fijo."""
    kw = dict(COMUNES, **params, n_estimators=n_estimators)
    if early is not None:
        kw["early_stopping_rounds"] = early
    m = xgb.XGBClassifier(**kw)
    if early is not None:
        m.fit(tr[feats], tr["label"], eval_set=[(va[feats], va["label"])],
              verbose=False)
    else:
        m.fit(tr[feats], tr["label"], verbose=False)
    return m


def ap_norm(y, p) -> float:
    """AUC-PR normalizado: (AP − prevalencia) / (1 − prevalencia).

    Hace falta para comparar años entre sí. La prevalencia del dataset varía
    mucho por año (15,2% en 2018 … 37,0% en 2017) porque las pseudo-ausencias se
    sortearon uniformemente dentro del split y no del año; como el suelo del
    AUC-PR es exactamente la prevalencia, los AP crudos de años distintos no son
    comparables. 0 = azar, 1 = perfecto.
    """
    prev = float(np.mean(y))
    return float((average_precision_score(y, p) - prev) / (1 - prev))


def metricas(y, p, y_val=None, p_val=None) -> dict:
    m = {"auc_pr": float(average_precision_score(y, p)),
         "auc_roc": float(roc_auc_score(y, p)),
         "brier": float(brier_score_loss(y, p))}
    if y_val is not None:                       # TSS con umbral elegido en val
        fpr, tpr, umbrales = roc_curve(y_val, p_val)
        u = float(umbrales[np.argmax(tpr - fpr)])
        pred = (p >= u).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum(); tn = ((pred == 0) & (y == 0)).sum()
        m["tss"] = float(tp / (tp + fn) + tn / (tn + fp) - 1)
        m["umbral_val"] = u
    return m


def linea(nombre, m):
    print(f"  {nombre:34s} AUC-PR={m['auc_pr']:.4f}  ROC={m['auc_roc']:.4f}  "
          f"Brier={m['brier']:.4f}"
          + (f"  TSS={m['tss']:.3f}" if "tss" in m else ""), flush=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    df = cargar()
    feats_v2 = json.load(open(RUTA_FEATS_V2))
    # v3f: se retira la autorregresiva EGIF caducada y entran las de FIRMS
    feats_v3f = [f for f in feats_v2 if f != FEAT_EGIF_CADUCADA] + FEATS_FIRMS_HIST

    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} | total "
          f"{len(df)} (prevalencia test {te['label'].mean()*100:.2f}%)\n",
          flush=True)

    salida = {"n_total": len(df), "n_train_protocolo": len(tr),
              "n_val": len(va), "n_test": len(te),
              "prevalencia_test": float(te["label"].mean())}

    # ---- Fase 1: protocolo congelado, elegir features e hiperparámetros en val
    print("== FASE 1: protocolo congelado (elección en VAL 2019) ==", flush=True)
    candidatos = {
        "v2_egif__params_base":   (feats_v2,  PARAMS_BASE),
        "v2_egif__params_tuned":  (feats_v2,  PARAMS_TUNED),
        "v3_firms__params_base":  (feats_v3f, PARAMS_BASE),
        "v3_firms__params_tuned": (feats_v3f, PARAMS_TUNED),
    }
    resultados, modelos = {}, {}
    for nombre, (feats, params) in candidatos.items():
        m = ajustar(feats, params, tr, va)
        p_va = m.predict_proba(va[feats])[:, 1]
        auc_val = float(average_precision_score(va["label"], p_va))
        resultados[nombre] = {"auc_pr_val": auc_val,
                              "mejor_iter": int(m.best_iteration),
                              "n_features": len(feats)}
        modelos[nombre] = (m, feats, params, p_va)
        print(f"  {nombre:26s} VAL AUC-PR={auc_val:.4f}  "
              f"(iter {m.best_iteration}, {len(feats)} feats)", flush=True)

    ganador = max(resultados, key=lambda k: resultados[k]["auc_pr_val"])
    m_prot, feats, params, p_va = modelos[ganador]
    print(f"\n  → elegido en VAL: {ganador}\n", flush=True)
    salida["fase1_seleccion_en_val"] = resultados
    salida["fase1_ganador"] = ganador
    salida["features"] = feats

    # --- variante ligera -----------------------------------------------------
    # Los hiperparámetros de Optuna (max_depth 10, lr 0,017) ganan +0,0014 de
    # AUC-PR en val sobre los de base y multiplican por ~13 el tamaño del
    # fichero del modelo (11 MB frente a 840 KB). Eso no es gratis: el modelo viaja
    # dentro del repositorio de publicación del colector de AEMET para el job diario de GitHub Actions
    # (bitácora §22), donde cada MB se paga en clonado y en cuota.
    # Se produce también la variante con params base y se documenta el
    # intercambio para que la decisión de despliegue sea explícita.
    hermano = ganador.replace("tuned", "base") if "tuned" in ganador else None
    variantes_finales = {"": (feats, params)}
    if hermano in modelos:
        variantes_finales["_lite"] = (modelos[hermano][1], modelos[hermano][2])
        print(f"  (se producirá además la variante ligera: {hermano})\n", flush=True)

    # El test 2020 se toca una sola vez por variante, dentro de producir():
    # cada variante debe ir acompañada de sus métricas, no de las de otra.

    # ---- Fases 2-4, ejecutadas para cada variante que se quiera desplegar
    for sufijo, (feats, params) in variantes_finales.items():
        etiqueta = "PRINCIPAL" if sufijo == "" else "LIGERA"
        print(f"\n{'='*72}\n=== VARIANTE {etiqueta}  (xgb_v3{sufijo}.ubj, "
              f"{len(feats)} feats)\n{'='*72}", flush=True)
        salida[f"variante{sufijo}"] = producir(sufijo, feats, params, df,
                                               tr, va, te)

    with open(f"{DIR}/dataset/metricas_v3.json", "w") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    tam = {s: os.path.getsize(f"{DIR}/modelos/xgb_v3{s}.ubj") / 1e6
           for s in variantes_finales}
    print(f"\n== Tamaño de los ficheros ==", flush=True)
    for s, mb in tam.items():
        print(f"  xgb_v3{s or ''}.ubj: {mb:.1f} MB", flush=True)


def producir(sufijo, feats, params, df, tr, va, te) -> dict:
    """CV temporal → cortes de alerta → reajuste final, para una variante.

    Reajusta también el modelo gemelo bajo el protocolo congelado (train
    2015-2018), porque las métricas que amparan a esta variante tienen que ser
    las suyas, no las de otra.
    """
    salida = {"features": feats, "n_features": len(feats), "params": params}
    y_te = te["label"].values

    m_prot = ajustar(feats, params, tr, va)
    p_va = m_prot.predict_proba(va[feats])[:, 1]
    p_te = m_prot.predict_proba(te[feats])[:, 1]
    m_test = metricas(y_te, p_te, va["label"].values, p_va)
    print("\n== FASE 1b: TEST 2020 (modelo gemelo del protocolo) ==", flush=True)
    linea(f"v3{sufijo} (protocolo, 2015-2018)", m_test)
    salida["test_2020_protocolo"] = m_test

    # ---- FASE 2: CV temporal rolling-origin ------------------------------- #
    print("\n== FASE 2: CV temporal (rolling origin) ==", flush=True)
    folds = [(2015, 2016, 2017), (2015, 2017, 2018),
             (2015, 2018, 2019), (2015, 2019, 2020)]
    iters, oof = [], []
    salida["fase2_cv_temporal"] = []
    for a0, a1, a_val in folds:
        f_tr = df[(df.anio >= a0) & (df.anio <= a1)]
        f_va = df[df.anio == a_val]
        m = ajustar(feats, params, f_tr, f_va)
        p = m.predict_proba(f_va[feats])[:, 1]
        y = f_va["label"].values
        iters.append(int(m.best_iteration))
        oof.append(pd.DataFrame({"anio": a_val, "label": y, "p": p}))
        salida["fase2_cv_temporal"].append(
            {"train": f"{a0}-{a1}", "val": a_val, "n_train": len(f_tr),
             "n_val": len(f_va), "prevalencia": float(y.mean()),
             "auc_pr": float(average_precision_score(y, p)),
             "auc_pr_norm": ap_norm(y, p),
             "auc_roc": float(roc_auc_score(y, p)),
             "mejor_iter": int(m.best_iteration)})
        f = salida["fase2_cv_temporal"][-1]
        print(f"  train {a0}-{a1} → val {a_val}: AP={f['auc_pr']:.4f} "
              f"(prev {f['prevalencia']*100:4.1f}% → norm {f['auc_pr_norm']:.4f}) "
              f"ROC={f['auc_roc']:.4f}  iter={f['mejor_iter']}", flush=True)

    # Se resume con AUC-PR normalizado y AUC-ROC, no con el AP crudo: los años
    # tienen prevalencias muy distintas y el AP crudo los haría incomparables
    # (2018 parece hundirse a 0,66 solo porque su prevalencia es del 15%).
    norm = [f["auc_pr_norm"] for f in salida["fase2_cv_temporal"]]
    rocs = [f["auc_roc"] for f in salida["fase2_cv_temporal"]]
    print(f"  media AUC-PR normalizado {np.mean(norm):.4f} ± {np.std(norm):.4f}"
          f"   |   media AUC-ROC {np.mean(rocs):.4f} ± {np.std(rocs):.4f}",
          flush=True)
    salida["fase2_resumen"] = {
        "auc_pr_norm_media": float(np.mean(norm)),
        "auc_pr_norm_std": float(np.std(norm)),
        "auc_roc_media": float(np.mean(rocs)),
        "auc_roc_std": float(np.std(rocs)),
        "auc_pr_crudo_media": float(np.mean([f["auc_pr"] for f in salida["fase2_cv_temporal"]])),
        "nota": ("El AP crudo por año NO es comparable entre años: la prevalencia "
                 "va del 15,2% (2018) al 37,0% (2017) porque las pseudo-ausencias "
                 "se sortearon dentro del split, no del año. Usar el normalizado "
                 "y el AUC-ROC."),
        "iters": iters}

    # nº de árboles del modelo final: mediana de los folds, escalada por el
    # aumento de datos respecto al fold más grande (más filas → hace falta algo
    # más de capacidad para el mismo learning rate)
    n_ultimo_fold = len(df[(df.anio >= 2015) & (df.anio <= 2019)])
    escala = len(df) / n_ultimo_fold
    n_final = int(round(np.median(iters) * escala)) + 1
    print(f"\n  iteraciones óptimas por fold: {iters}", flush=True)
    print(f"  → n_estimators final = mediana({int(np.median(iters))}) × "
          f"{escala:.3f} = {n_final}", flush=True)
    salida["n_estimators_final"] = n_final
    salida["escala_datos"] = float(escala)

    # ---- Fase 3: cortes de alerta desde las predicciones out-of-fold
    print("\n== FASE 3: cortes de alerta (out-of-fold, cobertura de v2) ==",
          flush=True)
    oof = pd.concat(oof, ignore_index=True)
    oof.to_parquet(f"{DIR}/dataset/oof_v3{sufijo}.parquet", index=False)

    # Los cortes se derivan solo de los años cuya prevalencia coincide con la de
    # diseño (~26%): 2019 y 2020. Incluir 2017 (37%) y 2018 (15%) desplazaría los
    # cortes por un artefacto del muestreo, no por el comportamiento del modelo.
    # Siguen siendo out-of-fold: en ambos folds ese año quedó fuera del ajuste.
    base = oof[oof.anio.isin([2019, 2020])]
    print(f"  base para los cortes: {len(base)} filas out-of-fold de 2019-2020 "
          f"(prevalencia {base['label'].mean()*100:.2f}%, la de diseño)", flush=True)

    cortes = {}
    for nivel, cob in COBERTURA_V2.items():
        u = float(np.quantile(base["p"].values, 1 - cob))
        pred = (base["p"].values >= u).astype(int)
        cortes[nivel] = {
            "corte": round(u, 4),
            "cobertura": float(pred.mean()),
            "precision": float(precision_score(base["label"], pred)),
            "recall": float(recall_score(base["label"], pred)),
        }
        print(f"  {nivel:9s} ≥{u:.4f}  cobertura={pred.mean()*100:5.1f}%  "
              f"precisión={cortes[nivel]['precision']:.2f}  "
              f"recall={cortes[nivel]['recall']:.2f}", flush=True)
    salida["fase3_cortes_alerta"] = cortes
    salida["fase3_n_oof_total"] = len(oof)
    salida["fase3_n_oof_base_cortes"] = len(base)
    salida["fase3_prevalencia_base"] = float(base["label"].mean())

    # ---- Fase 4: modelo final, reajuste con 2015-2020 completo
    print("\n== FASE 4: reajuste final con 2015-2020 completo ==", flush=True)
    m_final = ajustar(feats, params, df, None, n_estimators=n_final, early=None)
    os.makedirs(f"{DIR}/modelos", exist_ok=True)
    m_final.save_model(f"{DIR}/modelos/xgb_v3{sufijo}.ubj")
    print(f"  entrenado con {len(df)} filas y {n_final} árboles "
          f"(+{(len(df)/len(tr)-1)*100:.0f}% de datos sobre v1/v2)", flush=True)

    meta = {
        "version": f"v3{sufijo}",
        "descripcion": "Modelo de producción reajustado con 2015-2020 completo",
        "features": feats,
        "n_features": len(feats),
        "params": dict(params, n_estimators=n_final, **COMUNES),
        "entrenado_con": {"anios": "2015-2020", "n_filas": len(df),
                          "incremento_sobre_v2_pct": round((len(df)/len(tr)-1)*100, 1)},
        "cortes_alerta": {k: v["corte"] for k, v in cortes.items()},
        "cortes_alerta_detalle": cortes,
        "metricas_que_lo_amparan": {
            "nota": ("El modelo final incluye 2020 en su entrenamiento, así que "
                     "NO puede evaluarse sobre 2020. Estas métricas son las del "
                     "modelo gemelo entrenado bajo el protocolo congelado "
                     "(train 2015-2018) con las mismas features e "
                     "hiperparámetros."),
            "test_2020_protocolo": m_test,
            "cv_temporal": salida["fase2_resumen"],
        },
        "decision_firms": {
            "probado": True,
            "resultado": ("Las features de propensión FIRMS NO recuperan nada "
                          "(ablacion_v3_firms.py: C−B = −0,0001 AUC-PR en val). "
                          "Se descartan. La buena noticia del experimento: "
                          "eliminar por completo la autorregresiva EGIF cuesta "
                          "solo −0,014 AUC-PR (A−B), luego la feature congelada "
                          "en 2020 NO es load-bearing y el problema de caducidad "
                          "está acotado. Detalle en dataset/ablacion_v3_firms.json."),
        },
        "advertencias_de_uso": [
            "Las features deben calcularse EXACTAMENTE con las definiciones del "
            "pipeline de entrenamiento; cambiar una definición sin reentrenar "
            "produce errores silenciosos (bitácora §16).",
            "Los cortes de alerta son los de este modelo; no reutilizar los de v2.",
            ("firms_* se actualizan indefinidamente desde FIRMS NRT: esta es la "
             "razón de ser de v3 frente a la autorregresiva EGIF congelada en 2020."
             if any(f.startswith("firms_") for f in feats) else
             "n_fuegos_10km_mismomes_hist sigue congelada en 2020 (limitación "
             "conocida)."),
        ],
    }
    with open(f"{DIR}/modelos/xgb_v3{sufijo}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # comprobación de cordura: el modelo final debe rankear parecido al del
    # protocolo sobre 2020 (correlación alta) aunque sus valores estén sesgados
    # al alza por ser in-sample. Sirve solo como control de que el reajuste no
    # ha roto nada.
    p_te_final = m_final.predict_proba(te[feats])[:, 1]
    corr = float(np.corrcoef(p_te, p_te_final)[0, 1])
    print(f"  control (NO es una métrica): correlación del ranking en 2020 "
          f"entre modelo del protocolo y modelo final = {corr:.3f}", flush=True)
    salida["control_correlacion_ranking_2020"] = corr
    salida["tam_modelo_mb"] = round(
        os.path.getsize(f"{DIR}/modelos/xgb_v3{sufijo}.ubj") / 1e6, 2)

    # ---- Fase 5: figuras (sobre el modelo del protocolo, el evaluable) ----- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        os.makedirs(f"{DIR}/eda", exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        series = [("FWI absoluto", te["fwi"].values),
                  (f"v3{sufijo} (protocolo)", p_te)]
        for nombre, score in series:
            pr, rc, _ = precision_recall_curve(y_te, score)
            ax.plot(rc, pr,
                    label=f"{nombre} (AP={average_precision_score(y_te, score):.3f})")
        ax.axhline(y_te.mean(), ls="--", c="gray",
                   label=f"prevalencia={y_te.mean():.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"Curvas PR — test 2020 (modelo v3{sufijo} del protocolo)")
        ax.legend(); fig.tight_layout()
        fig.savefig(f"{DIR}/eda/pr_curves_v3{sufijo}.png", dpi=130)

        import shap
        muestra = te[feats].sample(min(3000, len(te)), random_state=SEED)
        sv = shap.TreeExplainer(m_prot).shap_values(muestra)
        plt.figure()
        shap.summary_plot(sv, muestra, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(f"{DIR}/eda/shap_summary_v3{sufijo}.png", dpi=130)
        imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
        salida["shap_top15"] = imp.head(15).round(4).to_dict()
        print(f"\n== SHAP |media| top 15 (v3{sufijo}, modelo del protocolo) ==")
        print(imp.head(15).round(4).to_string(), flush=True)
    except Exception as e:
        print(f"Figuras/SHAP fallaron (no bloquea): {e}", flush=True)

    return salida


if __name__ == "__main__":
    main()
