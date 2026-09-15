#!/usr/bin/env python3
"""
Modelo nativo de producción: entrenar donde se sirve.

No modifica nada. Crea solo ficheros nuevos con prefijo `nativo_`.

La idea
-------
Hasta ahora se ha intentado que producción se parezca al entrenamiento:
descongelar features, arreglar el percentil del FWI, corregir el viento. Eso
es cerrar el *train/serve skew*, y `auditoria_train_serve.py` acaba de medir
cuánto vale: como mucho 0,04 de AUC, frente a un hueco de 0,28 entre el
0,92 del test y el 0,64 operativo. Se puede cerrar entero y quedarse casi
igual.

El resto del hueco no viene de un desajuste de datos: las dos cifras responden
a preguntas distintas. El 0,92 se midió sobre celdas de 1 km, etiqueta EGIF
≥1 ha, prevalencia de diseño 25 % con negativos sorteados lejos de cualquier
fuego. El sistema opera sobre estaciones-día, todas candidatas, con
prevalencia real.

Así que se le da la vuelta al problema: en vez de forzar producción a
parecerse al laboratorio, se entrena en el laboratorio con las unidades y los
datos de producción.

  unidad     = (estación, día)               ← la que se sirve
  features   = las que produce el pipeline    ← las que se sirven
  fuente     = AEMET                          ← la que se sirve
  prevalencia= la real, sin sortear negativos ← la que se sirve

Con esto el desajuste entrenamiento/inferencia es cero por construcción y la
métrica que salga es directamente la que el sistema va a tener en la calle,
sin traducción de por medio.

Decisiones (todas apoyadas en mediciones previas)
-------------------------------------------------
· Split `train 2015-2020 / val 2021 / test 2022`, el mismo que v4. Permite
  comparar contra el modelo del cubo en igualdad de años y usar 2022 (el peor
  año de la serie) como test nunca visto.
· Sin features de FIRMS ni de rayos. `ablacion_v3_firms.py` midió que la
  propensión FIRMS aporta −0,0001, y `ablacion_proxies_operativos.py` que los
  rayos aportan 0 (y quitarlos mejora +0,0015). Incluirlas solo añadiría
  superficie de desajuste a cambio de nada.
· Satélite y estáticas: las climatologías mensuales congeladas de
  `estaciones_prototipo.parquet`, que son exactamente las que usa producción.
  Aquí no son un proxy degradado, sino la fuente real de las dos partes.
· `fwi_pctl_local` contra `clim_fwi_aemet`, la climatología homogénea
  construida hoy, con el mismo origen que el numerador (el fallo de §10.3.b-bis
  era justo ese).
· Negativos sin sortear. Todas las estaciones-día de la temporada entran. Es lo
  que hace que la métrica sea trasladable, y también lo que la baja: con
  prevalencia real el problema es mucho más difícil que con 25 % de diseño.

Lo que este modelo no arregla
-----------------------------
La resolución espacial. Una estación representa su entorno, no una celda de
1 km, así que este modelo no puede pintar el mapa de 1 km: es un ordenador de
estaciones. El mapa seguiría necesitando interpolación. Se declara y no se
disimula.

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python modelo_nativo_estacion.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

DIR = Path(__file__).parent
REPO_OP = Path("/home/charredgem/Desktop/Master/aemet_horario_verano2026")
import sys
sys.path.insert(0, str(REPO_OP))
from fwi_canadiense import calcular_fwi_serie          # noqa: E402

HIST = DIR / "dataset" / "aemet_historico_2015_2025.parquet"
EGIF = DIR / "egif_civio_2026-08.csv"
CLIM = REPO_OP / "modelo" / "clim_fwi_aemet"
SAL_DATOS = DIR / "dataset" / "nativo_estacion_dataset.parquet"
SAL_MODELO = DIR / "modelos" / "nativo_estacion.ubj"
SAL_JSON = DIR / "dataset" / "nativo_estacion_metricas.json"

RADIO_KM = 25          # mismo radio que la validación operativa
MESES = [5, 6, 7, 8, 9, 10]
SEED = 42

METEO = ["fwi", "fwi_pctl_local", "fwi_anom_sigma", "t2m_max", "t2m_min",
         "rh_min", "viento_max", "precip_dia", "vpd_max", "precip_7d",
         "precip_15d", "precip_30d", "fwi_med_7d", "fwi_max_7d",
         "fwi_med_15d", "fwi_med_30d", "rh_min_med_7d", "t2m_max_med_7d",
         "viento_max_med_7d", "dias_sin_lluvia"]
SATELITE = ["ndvi", "ndvi_med_30d", "lai", "swi010", "lst"]
ESTATICAS = ["elevacion", "pendiente", "rugosidad", "dist_carreteras",
             "dist_rios", "popdens", "clc_bosque", "clc_matorral",
             "clc_agricola", "clc_artificial", "clc_abierto",
             "clc_agric_hetero", "n_fuegos_10km_mismomes_hist"]
CALENDARIO = ["mes", "dia_anio", "es_festivo", "ccaa"]
FEATS = METEO + SATELITE + ESTATICAS + CALENDARIO


# --------------------------------------------------------------------------- #
def construir_dataset():
    import holidays
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    hist = pd.read_parquet(HIST)
    est = pd.read_parquet(REPO_OP / "modelo" / "estaciones_prototipo.parquet")
    con_clim = {p.stem for p in CLIM.glob("*.npz")}
    est = est[est.idema.isin(con_clim)].reset_index(drop=True)
    print(f"estaciones con climatología homogénea: {len(est)}")

    fuegos = pd.read_csv(EGIF, usecols=["fecha", "lat", "lng", "superficie"])
    fuegos["fecha"] = pd.to_datetime(fuegos["fecha"], errors="coerce")
    fuegos = fuegos.dropna(subset=["fecha", "lat", "lng"])
    fuegos = fuegos[fuegos.fecha.dt.year.between(2015, 2022)]
    print(f"incendios EGIF 2015-2022 con coordenadas: {len(fuegos):,}")

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ex, ey = tr.transform(est.lon.values, est.lat.values)
    est_xy = np.column_stack([ex, ey])
    fx, fy = tr.transform(fuegos.lng.values, fuegos.lat.values)
    fuegos = fuegos.assign(x=fx, y=fy)

    festivos = holidays.Spain(years=list(range(2015, 2023)))
    clim = {p.stem: dict(np.load(p)) for p in CLIM.glob("*.npz")}
    est_i = est.set_index("idema")

    filas = []
    for idema, s in hist[hist.idema.isin(set(est.idema))].groupby("idema"):
        e = est_i.loc[idema]
        for anio in range(2015, 2023):
            cal = pd.date_range(f"{anio}-01-01", f"{anio}-12-31", freq="D")
            d = s[s.fecha.dt.year == anio].set_index("fecha").reindex(cal)
            base = d[["tmax", "hr_min", "viento_max"]]
            if base.isna().mean().max() > 0.20:
                continue
            falta = base.isna().any(axis=1)
            grupos = (falta != falta.shift()).cumsum()[falta]
            if len(grupos) and grupos.value_counts().max() > 3:
                continue
            for c in ["tmax", "tmin", "hr_min", "viento_max"]:
                d[c] = d[c].interpolate(limit=3, limit_direction="both")
            d["prec"] = d["prec"].fillna(0.0)
            if d[["tmax", "hr_min", "viento_max"]].isna().any().any():
                continue

            mes = d.index.month.values
            fwi = calcular_fwi_serie(d.tmax.values, d.hr_min.values,
                                     d.viento_max.values * 3.6,
                                     d.prec.values, mes)["fwi"]
            pre = d.prec.values
            rh = d.hr_min.values
            tmx = d.tmax.values
            vto = d.viento_max.values
            # días sin lluvia acumulados de forma vectorial
            secos = np.zeros(len(d), dtype=int)
            for i in range(1, len(d)):
                secos[i] = 0 if pre[i] >= 1.0 else min(secos[i - 1] + 1, 120)

            for i in np.where(np.isin(mes, MESES) & (np.arange(len(d)) >= 30))[0]:
                F = {"fwi": fwi[i], "t2m_max": tmx[i], "t2m_min": d.tmin.values[i],
                     "rh_min": rh[i], "viento_max": vto[i], "precip_dia": pre[i]}
                es_ = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
                F["vpd_max"] = es_ * (1 - F["rh_min"] / 100)
                F.update(precip_7d=np.nansum(pre[i-7:i]),
                         precip_15d=np.nansum(pre[i-15:i]),
                         precip_30d=np.nansum(pre[i-30:i]),
                         fwi_med_7d=np.nanmean(fwi[i-7:i]),
                         fwi_max_7d=np.nanmax(fwi[i-7:i]),
                         fwi_med_15d=np.nanmean(fwi[i-15:i]),
                         fwi_med_30d=np.nanmean(fwi[i-30:i]),
                         rh_min_med_7d=np.nanmean(rh[i-7:i]),
                         t2m_max_med_7d=np.nanmean(tmx[i-7:i]),
                         viento_max_med_7d=np.nanmean(vto[i-7:i]),
                         dias_sin_lluvia=int(secos[i]))
                m = int(mes[i])
                c = clim[idema][f"m{m}"]
                F["fwi_pctl_local"] = (float((c <= fwi[i]).mean() * 100)
                                       if len(c) else np.nan)
                F["fwi_anom_sigma"] = (float((fwi[i] - c.mean()) / c.std())
                                       if len(c) and c.std() > 0 else np.nan)
                for v in ["ndvi", "lai", "swi010", "lst"]:
                    F[v] = e[f"{v}_m{m}"]
                F["ndvi_med_30d"] = e[f"ndvi_m{m}"]
                for v in ESTATICAS[:-1]:
                    F[v] = e[v]
                F["n_fuegos_10km_mismomes_hist"] = e[f"n_mismomes_m{m}"]
                fecha = cal[i]
                F["es_festivo"] = float(fecha.weekday() >= 5
                                        or fecha.date() in festivos)
                F["mes"], F["dia_anio"] = m, fecha.dayofyear
                F["ccaa"] = int(e["ccaa"]) if np.isfinite(e["ccaa"]) else -1
                F.update(idema=idema, fecha=fecha, anio=anio)
                filas.append(F)
        if len(filas) and len(filas) % 200000 < 500:
            print(f"  {len(filas):,} filas...", flush=True)

    df = pd.DataFrame(filas)
    print(f"\nestación-día generadas: {len(df):,}")

    # etiqueta: incendio EGIF a <= RADIO_KM ese mismo día
    df["label"] = 0
    idx_est = {k: i for i, k in enumerate(est.idema)}
    for fecha, g in fuegos.groupby(fuegos.fecha.dt.normalize()):
        sel = df.index[df.fecha == fecha]
        if not len(sel):
            continue
        arbol = cKDTree(np.column_stack([g.x.values, g.y.values]))
        ii = df.loc[sel, "idema"].map(idx_est).values
        d_, _ = arbol.query(est_xy[ii])
        df.loc[sel, "label"] = (d_ <= RADIO_KM * 1000).astype(int)

    df["split"] = np.where(df.anio <= 2020, "train",
                           np.where(df.anio == 2021, "val", "test"))
    return df, est


def main():
    if SAL_DATOS.exists():
        df = pd.read_parquet(SAL_DATOS)
        print(f"dataset ya construido: {len(df):,} filas")
    else:
        df, _ = construir_dataset()
        df.to_parquet(SAL_DATOS)
        print(f"guardado: {SAL_DATOS}")

    print("\n=== dataset nativo de producción ===")
    r = df.groupby("split").agg(n=("label", "size"), pos=("label", "sum"),
                                prev=("label", "mean"))
    r["prev"] = (r["prev"] * 100).round(2)
    print(r.to_string())

    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    params = dict(learning_rate=0.05, max_depth=6, min_child_weight=20,
                  subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                  tree_method="hist", eval_metric="aucpr", n_jobs=20,
                  random_state=SEED, n_estimators=2000,
                  early_stopping_rounds=100,
                  scale_pos_weight=float((tr.label == 0).sum() / max((tr.label == 1).sum(), 1)))
    m = xgb.XGBClassifier(**params)
    m.fit(tr[FEATS], tr.label, eval_set=[(va[FEATS], va.label)], verbose=False)
    SAL_MODELO.parent.mkdir(exist_ok=True)
    m.save_model(str(SAL_MODELO))

    def evalua(g, nombre):
        p = m.predict_proba(g[FEATS])[:, 1]
        base = float(g.label.mean())
        out = {"n": int(len(g)), "positivos": int(g.label.sum()),
               "prevalencia_pct": round(base * 100, 2),
               "auc_roc": round(float(roc_auc_score(g.label, p)), 4),
               "auc_pr": round(float(average_precision_score(g.label, p)), 4),
               "lift_pr": round(float(average_precision_score(g.label, p) / base), 2)}
        k = max(1, int(len(g) * 0.10))
        top = np.argsort(-p)[:k]
        out["lift_decil"] = round(float(g.label.values[top].mean() / base), 2)
        # baseline: el FWI percentil, calculado con la misma fuente
        b = g["fwi_pctl_local"].fillna(g["fwi_pctl_local"].median()).values
        out["auc_roc_fwi_pctl"] = round(float(roc_auc_score(g.label, b)), 4)
        out["auc_roc_fwi"] = round(float(roc_auc_score(g.label, g["fwi"].values)), 4)
        print(f"{nombre:<16} n={out['n']:>8,} pos={out['positivos']:>6,} "
              f"({out['prevalencia_pct']:.2f}%)  AUC {out['auc_roc']:.4f}  "
              f"AP {out['auc_pr']:.4f} (×{out['lift_pr']})  "
              f"decil ×{out['lift_decil']}  | FWIpctl {out['auc_roc_fwi_pctl']:.4f} "
              f"· FWI {out['auc_roc_fwi']:.4f}")
        return out

    print(f"\n=== resultados ({m.best_iteration + 1} árboles) ===")
    res = {"val_2021": evalua(va, "val 2021"), "test_2022": evalua(te, "test 2022")}

    imp = m.get_booster().get_score(importance_type="gain")
    tot = sum(imp.values()) or 1
    top = sorted(imp.items(), key=lambda x: -x[1])[:12]
    print("\ntop features (gain %):",
          ", ".join(f"{k} {100*v/tot:.1f}" for k, v in top))

    SAL_JSON.write_text(json.dumps(
        {"diseno": {"unidad": "estación-día", "radio_km": RADIO_KM,
                    "meses": MESES, "split": "train 2015-2020 / val 2021 / test 2022",
                    "fuente_meteo": "AEMET climatológicos diarios",
                    "clim_fwi": "clim_fwi_aemet (homogénea con el numerador)",
                    "sin_firms_ni_rayos": "medido que aportan ~0 (ablaciones previas)",
                    "negativos": "sin sortear: todas las estaciones-día de la temporada"},
         "n_features": len(FEATS), "features": FEATS,
         "arboles": int(m.best_iteration + 1),
         "resultados": res,
         "importancia_gain_pct": {k: round(100 * v / tot, 2) for k, v in top},
         "limite": "unidad estación, no celda de 1 km: no sustituye al mapa"},
        indent=2, ensure_ascii=False))
    print(f"\nguardado: {SAL_JSON}\n         {SAL_MODELO}")


if __name__ == "__main__":
    main()
