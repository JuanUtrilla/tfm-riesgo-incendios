#!/usr/bin/env python3
"""
Auditoría train/serve de las 46 features, una por una: ¿le llega al modelo en
producción lo mismo que vio al entrenar, y cuánto cuesta cada desviación?

No modifica nada. Crea solo dataset/auditoria_train_serve.json.

La pregunta y su límite

"Que el modelo en producción sea equivalente al de entrenamiento" es el
objetivo correcto para una parte del problema y es imposible para la otra.
Hay que separarlas o se persigue un fantasma:

  Parte arreglable: el train/serve skew. El modelo aprendió a leer una
  distribución de cada feature y en producción le llega otra: satélite
  congelado, rayos a 0, percentil de FWI cruzando fuentes, viento NaN leído
  como cero. Esto sí se puede cerrar, y es lo que audita este script.

  Parte no arreglable: la pregunta es otra. El 0,923 de test se midió sobre
  celdas de 1 km, etiqueta EGIF ≥1 ha, prevalencia de diseño 25 % con
  negativos sorteados lejos de cualquier fuego. El 0,64 operativo se mide
  sobre estación ±25 km, etiqueta EFFIS ≥30 ha, prevalencia 1,1 % y todas las
  estaciones-día como candidatas. Igualar eso no arregla producción: es
  volver a evaluar en el laboratorio, que es justo lo que se quiere evitar.

Este script sirve para no confundirlas: mide lo que se puede cerrar, lo
ordena por lo que cuesta, y deja fuera lo que no depende de los datos.

Qué hace, en tres pasos

1. Desviación. Para cada feature, compara su distribución en entrenamiento
   (julio-agosto, para no confundir estacionalidad con desajuste) contra la de
   producción 2026. Métrica: PSI (Population Stability Index) sobre los
   deciles del entrenamiento, el estándar en monitorización de modelos.
   Convención habitual: <0,1 estable · 0,1-0,25 moderado · >0,25 grave.

2. Importancia. Ganancia del propio XGBoost. Una feature muy desviada que el
   modelo no mira no importa; una poco desviada de la que depende, sí.

3. Coste. Lo que convierte el diagnóstico en números: se toma el test 2020,
   se remapea por cuantiles cada feature de su distribución de entrenamiento
   a la de producción, y se vuelve a puntuar con el modelo sin reentrenar. La
   caída de AUC es lo que cuesta esa desviación.

   Por qué el remapeo por cuantiles es el simulador correcto: es monótono, así
   que conserva el orden de la feature y por tanto su contenido informativo;
   lo único que rompe son los umbrales absolutos que los árboles aprendieron.
   Y ese es exactamente el mecanismo del train/serve skew.

   Se mide una a una (coste aislado) y acumulando por orden de coste (coste
   conjunto, que no es la suma: los árboles compensan unas features con otras).

Control: la variante sin remapear nada debe reproducir el 0,9234 del gemelo de
protocolo guardado en xgb_v3_metadata.json. Si no lo hace, nada de lo demás
vale.

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python auditoria_train_serve.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

DIR = Path(__file__).parent
REPO_OP = Path("/home/charredgem/Desktop/Master/aemet_horario_verano2026")
SALIDA = DIR / "dataset" / "auditoria_train_serve.json"

# features que en producción no se observan; se listan para poder separarlas
# en el informe de las que sí y aun así se desvían.
CONGELADAS = {"ndvi", "ndvi_med_30d", "lai", "swi010", "lst"}
SINTETICAS = {"rayos_dia", "rayos_7d"}
ESTATICAS = {"elevacion", "pendiente", "rugosidad", "dist_carreteras",
             "dist_rios", "popdens", "clc_bosque", "clc_matorral",
             "clc_agricola", "clc_artificial", "clc_abierto",
             "clc_agric_hetero", "ccaa", "n_fuegos_10km_mismomes_hist"}
CALENDARIO = {"mes", "dia_anio", "es_festivo"}


def psi(tr, pr, n=10):
    """Population Stability Index sobre los deciles del entrenamiento."""
    tr, pr = tr[np.isfinite(tr)], pr[np.isfinite(pr)]
    if len(tr) < 100 or len(pr) < 100:
        return np.nan
    cortes = np.unique(np.percentile(tr, np.linspace(0, 100, n + 1)))
    if len(cortes) < 3:
        return 0.0
    cortes[0], cortes[-1] = -np.inf, np.inf
    a = np.histogram(tr, bins=cortes)[0] / len(tr)
    b = np.histogram(pr, bins=cortes)[0] / len(pr)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def mapa_cuantiles(tr, pr, x):
    """Lleva `x` (escala entrenamiento) a la escala de producción, conservando
    el orden. NaN se mantiene NaN: XGBoost lo trata de forma nativa y el
    remapeo no debe inventar valores."""
    tr_ok, pr_ok = tr[np.isfinite(tr)], pr[np.isfinite(pr)]
    if len(tr_ok) < 100 or len(pr_ok) < 100:
        return x
    q = np.linspace(0, 100, 201)
    ctr, cpr = np.percentile(tr_ok, q), np.percentile(pr_ok, q)
    out = np.asarray(x, dtype=float).copy()
    ok = np.isfinite(out)
    out[ok] = np.interp(out[ok], ctr, cpr)
    return out


def main():
    FEATS = json.loads((REPO_OP / "modelo" /
                        "xgb_v2_prototipo_features.json").read_text())
    ent = pd.read_parquet(DIR / "dataset" / "dataset_modelo_v1.parquet")
    ent["ccaa"] = ent["ccaa"].fillna(-1).astype(int)
    ent["es_festivo"] = ent["es_festivo"].astype(float)

    prod = pd.read_parquet(DIR / "dataset" / "experimento_b_fase2_features.parquet")
    prod = prod[prod["brazo"] == "AEMET obs"]
    print(f"entrenamiento {len(ent):,} filas · producción 2026 {len(prod):,} filas")

    # comparar solo julio-agosto: si no, la estacionalidad se cuela como
    # desajuste y todas las features meteo salen "graves" por serlo.
    ent_v = ent[ent["mes"].isin([7, 8])]
    print(f"comparación estacional: {len(ent_v):,} filas de entrenamiento "
          f"de julio-agosto\n")

    modelo = xgb.XGBClassifier()
    modelo.load_model(str(REPO_OP / "modelo" / "xgb_v2_prototipo.ubj"))
    gain = modelo.get_booster().get_score(importance_type="gain")
    tot = sum(gain.values()) or 1.0

    tr_te = ent[ent["split"] == "test"].reset_index(drop=True)
    y_te = tr_te["label"].values
    base = float(roc_auc_score(y_te, modelo.predict_proba(tr_te[FEATS])[:, 1]))
    print(f"CONTROL: AUC test 2020 sin remapear = {base:.4f} "
          f"(el gemelo de protocolo da 0,9234)\n")

    # ---------------- paso 1 y 2: desviación e importancia
    filas = []
    for f in FEATS:
        if f not in prod.columns:
            continue
        grupo = ("congelada" if f in CONGELADAS else
                 "sintética" if f in SINTETICAS else
                 "estática" if f in ESTATICAS else
                 "calendario" if f in CALENDARIO else "meteo")
        a = ent_v[f].astype(float).values
        b = prod[f].astype(float).values
        filas.append({
            "feature": f, "grupo": grupo,
            "psi": round(psi(a, b), 3),
            "gain_pct": round(100 * gain.get(f, 0.0) / tot, 2),
            "media_ent": round(float(np.nanmean(a)), 3),
            "media_prod": round(float(np.nanmean(b)), 3),
            "nan_ent_pct": round(100 * float(np.mean(~np.isfinite(a))), 1),
            "nan_prod_pct": round(100 * float(np.mean(~np.isfinite(b))), 1)})
    aud = pd.DataFrame(filas).sort_values("psi", ascending=False)

    print("=== DESVIACIÓN POR FEATURE (PSI: <0,1 estable · >0,25 grave) ===")
    print(f"{'feature':<30}{'grupo':<12}{'PSI':>7}{'gain%':>8}"
          f"{'media ent':>11}{'media prod':>12}{'NaN prod':>10}")
    for _, r in aud.iterrows():
        marca = "  <<<" if r.psi > 0.25 and r.gain_pct > 1 else ""
        print(f"{r.feature:<30}{r.grupo:<12}{r.psi:>7.3f}{r.gain_pct:>8.2f}"
              f"{r.media_ent:>11.2f}{r.media_prod:>12.2f}"
              f"{r.nan_prod_pct:>9.1f}%{marca}")

    # ---------------- paso 3: coste de cada desviación
    print("\n=== COSTE DE CADA DESVIACIÓN (remapeo por cuantiles en test 2020) ===")
    candidatas = aud[(aud.psi > 0.10) & (aud.gain_pct > 0.5)].feature.tolist()
    print(f"se simulan {len(candidatas)} features con PSI>0,10 y gain>0,5 %\n")

    costes = {}
    for f in candidatas:
        X = tr_te[FEATS].copy()
        X[f] = mapa_cuantiles(ent_v[f].astype(float).values,
                              prod[f].astype(float).values,
                              tr_te[f].astype(float).values)
        a = float(roc_auc_score(y_te, modelo.predict_proba(X)[:, 1]))
        costes[f] = round(a - base, 4)
        print(f"  {f:<30} AUC {a:.4f}   Δ {a-base:+.4f}")

    orden = sorted(costes, key=lambda k: costes[k])
    print("\n  acumulado, por orden de coste:")
    X = tr_te[FEATS].copy()
    acum = {}
    for f in orden:
        X[f] = mapa_cuantiles(ent_v[f].astype(float).values,
                              prod[f].astype(float).values,
                              tr_te[f].astype(float).values)
        a = float(roc_auc_score(y_te, modelo.predict_proba(X)[:, 1]))
        acum[f] = round(a - base, 4)
        print(f"  +{f:<29} AUC {a:.4f}   Δ acumulada {a-base:+.4f}")

    SALIDA.write_text(json.dumps({
        "control_auc_test2020": round(base, 4),
        "nota_control": "el gemelo de protocolo de xgb_v3_metadata da 0.9234",
        "n_entrenamiento_julago": int(len(ent_v)), "n_produccion": int(len(prod)),
        "features": aud.to_dict("records"),
        "coste_aislado": costes, "coste_acumulado": acum,
        "metodo": {
            "psi": "deciles de la distribución de entrenamiento jul-ago",
            "coste": "remapeo monótono por cuantiles entrenamiento→producción "
                     "sobre el test 2020, modelo sin reentrenar; conserva el "
                     "orden de la feature y rompe solo los umbrales aprendidos",
            "limite": "no cubre la parte definicional del hueco (prevalencia, "
                      "etiqueta, unidad espacial), que no es train/serve skew"}},
        indent=2, ensure_ascii=False))
    print(f"\nguardado: {SALIDA}")


if __name__ == "__main__":
    main()
