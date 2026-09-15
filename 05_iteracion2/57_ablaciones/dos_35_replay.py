#!/usr/bin/env python3
"""
Dos modelos, paso 35 (ampliación): los algoritmos de dos_35 en las temporadas
2025 y 2026.

No toca producción ni el replay publicado. Lee el archivo de previsiones IFS
(TFM_ARCHIVO) y el disco externo; escribe en salida/ con el prefijo dos_35_:
dos_35_modelos/, dos_35_replay_sandbox/, dos_35_replay_<año>_ifs.csv y
dos_35_replay.json, y añade la clave «temporadas» a dos_35_algoritmos.json.

Por qué
-------
`dos_35_algoritmos.py` compara los algoritmos en la prueba de 2024, con el
banco de 1,000 celdas al azar por día. La elección del r10 se confirmó después
con la reproducción de las temporadas 2025 y 2026 en condiciones de servicio.
Este paso repite esa reproducción con los modelos finales de cada algoritmo,
con el mismo protocolo (ampliación del prerregistro en docs/TRAZABILIDAD.md).

Cómo
----
Tres pasos, porque el replay corre sobre una copia del sandbox con el que se
hizo (sus módulos se llaman igual que los del repo y no pueden convivir en el
mismo proceso):

  entrenar   con el entorno del repo: reentrena el modelo final de cada
             algoritmo con la configuración elegida en dos_35_algoritmos.json,
             comprueba que da el mismo AUC de 2024 y lo guarda.
  replay     con el sandbox: para cada día hace lo que `replay_dia.py`
             (reanálisis hasta D−7, IFS archivado de D−6 a D, FIRMS de 7 días)
             y ejecuta `dos_riesgo_hoy.main`. La matriz de 46 variables que
             ese script pasa al r10 se captura y la puntúan los demás
             algoritmos. Todas las probabilidades se guardan en float16, como
             en el servicio, y el AUC se calcula como en `replay_dia.py`: todas
             las celdas de España, las quemadas ese día frente al resto.
  veredicto  AUC medio por día con fuego, IC95 por remuestreo de días,
             diferencia pareada frente al r10 servido y % de días que lo supera.

Control: el r10 servido tiene que dar el mismo AUC por día que el replay
publicado (replay/replay_<año>_ifs.csv). Si no, el paso veredicto para.

Uso:
    source entorno.sh
    venv_lgbm/bin/python 05_iteracion2/57_ablaciones/dos_35_replay.py entrenar
    venv_lgbm/bin/python 05_iteracion2/57_ablaciones/dos_35_replay.py replay --anio 2025
    venv_lgbm/bin/python 05_iteracion2/57_ablaciones/dos_35_replay.py replay --anio 2026
    venv_lgbm/bin/python 05_iteracion2/57_ablaciones/dos_35_replay.py veredicto
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = os.environ.get("TFM_SALIDA", str(RAIZ / "salida"))
ARCHIVO = os.environ.get("TFM_ARCHIVO", os.path.join(
    os.environ.get("TFM_DATOS", str(RAIZ / "datos")), "archivo"))
MODELOS_DIR = f"{SALIDA}/dos_35_modelos"
SANDBOX = f"{SALIDA}/dos_35_replay_sandbox"
RANGO = {2025: ("2025-05-25", "2025-11-01"), 2026: ("2026-05-25", "2026-09-02")}
FUENTES = {
    2025: dict(re=f"{ARCHIVO}/era5land_diario_2025.nc",
               ifs=f"{ARCHIVO}/ifs_archivo_2025.parquet",
               firms=f"{ARCHIVO}/firms_2025/firms_iberia_2025.parquet"),
    2026: dict(re=f"{ARCHIVO}/era5land_diario_2026.nc",
               ifs=f"{ARCHIVO}/ifs_archivo_2026.parquet",
               firms=f"{ARCHIVO}/firms_2026/firms_iberia_2026.parquet"),
}
CLAVES = {"XGBoost ajustado": "xgb_ajustado", "LightGBM": "lightgbm",
          "Random Forest": "random_forest", "Regresión logística": "logistica"}
N_BOOT, SEED = 2000, 42
HILOS = int(os.environ.get("TFM_HILOS", "8"))


# -------------------------------------------------------------- entrenar ----
def entrenar():
    import joblib
    import config_expansion as ce
    import dos_35_algoritmos as d35
    from dos_05_modelos import FULL
    from dos_18_ratio import DATASET, peldano

    R = json.load(open(f"{SALIDA}/dos_35_algoritmos.json"))
    df = pd.read_parquet(DATASET, columns=FULL + ["fecha", "split", "label", "k_neg", "id_pos"])
    tr = peldano(df[df["split"] == "train"], d35.K)
    X, y = tr[FULL].values.astype(np.float32), tr["label"].values
    del df
    ev = pd.read_parquet(f"{ce.DATASET}/dataset_effis.parquet")
    ev = ev[ev["disenio"] == "eval_dia"].reset_index(drop=True)
    Xe = ev[FULL].values.astype(np.float32)
    fabricas = {
        "XGBoost ajustado": lambda p: d35.xgb_r10(**p),
        "LightGBM": lambda p: d35.lgbm(p["num_leaves"], p["learning_rate"]),
        "Random Forest": lambda p: d35.bosque(p["min_samples_leaf"], p["max_features"]),
        "Regresión logística": lambda p: d35.logistica(p["C"]),
    }
    os.makedirs(MODELOS_DIR, exist_ok=True)
    info = {}
    for nombre, fabrica in fabricas.items():
        m = fabrica(R["algoritmos"][nombre]["elegida"])
        m.fit(X, y)
        auc24 = float(d35.auc_dias(ev, m.predict_proba(Xe)[:, 1], 2024)["auc"].mean())
        doc = R["algoritmos"][nombre]["2024"]["auc_medio"]
        info[nombre] = {"auc_2024": auc24, "dos_35_algoritmos": doc, "dif": auc24 - doc}
        joblib.dump(m, f"{MODELOS_DIR}/{CLAVES[nombre]}.joblib")
        print(f"  {nombre:20s} AUC 2024 {auc24:.4f} (dos_35 {doc:.4f}, Δ {auc24-doc:+.1e})",
              flush=True)
    json.dump(FULL, open(f"{MODELOS_DIR}/variables.json", "w"))
    json.dump(info, open(f"{MODELOS_DIR}/control_2024.json", "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- replay ----
def auc_rangos(pos, neg):
    """AUC por rangos (Mann-Whitney), la misma función del replay publicado."""
    x = np.concatenate([pos, neg])
    r = pd.Series(x).rank().values[:len(pos)]
    return float((r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def replay(anio):
    if not os.path.isdir(SANDBOX):
        shutil.copytree(f"{ARCHIVO}/sandbox_replay", SANDBOX,
                        ignore=shutil.ignore_patterns("__pycache__"))
    os.environ["TFM_FIRMS_DIAS"] = "7"
    os.chdir(SANDBOX)
    # solo los módulos del sandbox: son los que produjeron el replay publicado
    sys.path[:] = [SANDBOX] + [p for p in sys.path if not p.startswith(str(RAIZ))]

    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure
    matplotlib.figure.Figure.savefig = lambda self, *k, **kw: None
    import xarray as xr
    import xgboost as xgb
    import capa_verdad
    import comparar_rankings as cr
    import config
    import dos_riesgo_hoy as dr
    import malla_02b_ifs as ifsmod
    import riesgo_hoy as rh
    capa_verdad.preparar = lambda ds, fecha: {
        "m7": np.zeros(ds["is_spain"].shape, bool), "m1": np.zeros(ds["is_spain"].shape, bool),
        "fx": np.array([]), "fy": np.array([]), "mx": np.array([]), "my": np.array([]),
        "mn": np.array([]), "effis": []}
    assert cr.VENTANA_FIRMS == 7

    full = json.load(open(rh.FEATS_JSON))
    assert full == json.load(open(f"{MODELOS_DIR}/variables.json")), "variables distintas"
    modelos = {c: joblib.load(f"{MODELOS_DIR}/{c}.joblib") for c in CLAVES.values()}
    for c, m in modelos.items():
        if c != "logistica":
            m.set_params(n_jobs=HILOS)

    captura = {}
    original = xgb.XGBClassifier.predict_proba

    def predict_proba(self, X, *k, **kw):
        if getattr(X, "shape", (0, 0))[1] == len(full):
            captura["X"] = X
        return original(self, X, *k, **kw)
    xgb.XGBClassifier.predict_proba = predict_proba

    cubo = xr.open_dataset(config.CUBO, decode_timedelta=False)
    sel = cubo["is_spain"].values.astype(bool).ravel()
    cubo.close()
    F = FUENTES[anio]
    ifs_todo = pd.read_parquet(F["ifs"])
    ifs_todo["fecha"] = pd.to_datetime(ifs_todo["fecha"])
    focos = pd.read_parquet(F["firms"], columns=["latitude", "longitude", "acq_date", "frp"])
    focos["acq_date"] = pd.to_datetime(focos["acq_date"])

    csv = f"{SALIDA}/dos_35_replay_{anio}_ifs.csv"
    hechos = set(pd.read_csv(csv)["fecha"]) if os.path.exists(csv) else set()
    t_ini = time.time()
    for D in pd.date_range(*RANGO[anio], freq="D"):
        fstr = str(D.date())
        if fstr in hechos:
            continue
        t0 = time.time()
        captura.clear()
        corte = D - pd.Timedelta(days=7)
        trunc = config.salida(f"era5land_trunc_ifs_{fstr}.nc")
        try:
            ds = xr.open_dataset(F["re"], decode_timedelta=False)
            f_re = pd.to_datetime(ds["fecha"].values)
            keep = np.where(f_re <= corte)[0]
            if f_re[keep][-1] != corte:
                raise ValueError(f"el reanálisis no llega al {corte.date()}")
            ds.isel({ds["fecha"].dims[0]: keep}).to_netcdf(trunc)
            ds.close()
            rh.REANALISIS = trunc
            ifs = ifs_todo[(ifs_todo["fecha"] > corte) & (ifs_todo["fecha"] <= D + pd.Timedelta(days=1))]
            if ifs["fecha"].min() != corte + pd.Timedelta(days=1):
                raise ValueError("el IFS archivado no empieza en D−6")
            ifs.to_parquet(ifsmod.ruta_cache(fstr), index=False)
            ifsmod.descarga = lambda *k, **kw: ifs.copy()
            fo = focos[(focos["acq_date"] >= D - pd.Timedelta(days=7)) & (focos["acq_date"] < D)]
            os.makedirs(config.salida("_firms7"), exist_ok=True)
            fo.to_csv(config.salida(f"_firms7/{fstr}.csv"), index=False)
            rh.firms_nrt = lambda dsx, D=D: cr.firms_dia(D, dsx)

            class A:
                dias = [0]
                pasada = fstr
            dr.main(A)
        except (AssertionError, ValueError, IndexError, KeyError) as e:
            print(f"  {fstr}: fallo ({type(e).__name__}: {e})", flush=True)
            continue

        mapas = np.load(config.salida(f"mapas_diarios/dos_riesgo_{fstr}.npz"))
        scores = {"r10": mapas["prob_r10"].astype(float).ravel()}
        X = captura["X"]
        for c, m in modelos.items():
            g = np.full(sel.size, np.nan, np.float32)
            g[sel] = m.predict_proba(X)[:, 1]
            scores[c] = g.astype(np.float16).astype(float)
        publicado = f"{ARCHIVO}/replay/{anio}/ifs/{fstr}.npz"
        dif_pub = (float(np.nanmax(np.abs(np.load(publicado)["prob_r10"].astype(float).ravel()
                                          - scores["r10"])))
                   if os.path.exists(publicado) else np.nan)
        V = np.load(f"{ARCHIVO}/replay/verdad_{anio}/{fstr}.npz")
        mask = np.zeros(sel.size, bool)
        mask[V["celda"]] = True
        fila = {"fecha": fstr, "celdas_quemadas": int(mask.sum()),
                "dif_max_r10_vs_publicado": dif_pub}
        for c, pm in scores.items():
            ok = sel & np.isfinite(pm)
            pos = np.flatnonzero(mask & ok)
            fila[f"auc_{c}"] = auc_rangos(pm[pos], pm[~mask & ok]) if len(pos) else np.nan
        fila["t_s"] = round(time.time() - t0, 1)
        prev = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
        pd.concat([prev, pd.DataFrame([fila])], ignore_index=True).to_csv(csv, index=False)
        for f in (trunc, ifsmod.ruta_cache(fstr), config.salida(f"dos_riesgo_{fstr}.json"),
                  config.salida(f"mapas_diarios/dos_riesgo_{fstr}.npz")):
            if os.path.exists(f):
                os.remove(f)
        auc_txt = " · ".join(f"{c} {fila[f'auc_{c}']:.3f}" for c in scores
                             if np.isfinite(fila[f"auc_{c}"]))
        print(f"  {fstr}: {fila['t_s']:.0f} s · Δ r10 {dif_pub:.1e} · {auc_txt}", flush=True)
    print(f"terminado {anio}: {(time.time()-t_ini)/60:.0f} min", flush=True)


# ------------------------------------------------------------- veredicto ----
def ic_media(x, rng):
    n = len(x)
    m = np.array([x[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def veredicto():
    R = {}
    for anio in (2025, 2026):
        df = pd.read_csv(f"{SALIDA}/dos_35_replay_{anio}_ifs.csv")
        pub = pd.read_csv(f"{ARCHIVO}/replay/replay_{anio}_ifs.csv")
        m = df.merge(pub[["fecha", "auc_r10"]], on="fecha", how="outer",
                     suffixes=("", "_pub"), indicator=True)
        solo = m[m["_merge"] != "both"]["fecha"].tolist()
        dif = (m["auc_r10"] - m["auc_r10_pub"]).abs().max()
        con = df[df["celdas_quemadas"] > 0]
        control = {"dias": int(len(df)), "dias_publicado": int(len(pub)),
                   "dias_con_fuego": int(len(con)), "fechas_no_comunes": solo,
                   "dif_max_auc_r10_vs_publicado": float(dif),
                   "dif_max_prediccion_r10": float(df["dif_max_r10_vs_publicado"].max()),
                   "auc_r10_publicado": float(pub[pub["celdas_quemadas"] > 0]["auc_r10"].mean())}
        print(f"{anio}: control {control}")
        if solo or not dif <= 1e-9:
            raise SystemExit(f"{anio}: el r10 servido no reproduce el replay publicado")
        rng = np.random.default_rng(SEED)
        R[anio] = {"control": control, "modelos": {}}
        ref = con["auc_r10"].values
        for c in ["r10"] + list(CLAVES.values()):
            a = con[f"auc_{c}"].values
            d = a - ref
            R[anio]["modelos"][c] = {
                "auc_medio": float(a.mean()), "ic95": ic_media(a, rng),
                "dif_vs_r10": float(d.mean()), "ic95_dif": ic_media(d, rng),
                "dias_gana_r10": float((d > 0).mean())}
            r = R[anio]["modelos"][c]
            print(f"  {c:14s} AUC {r['auc_medio']:.4f} [{r['ic95'][0]:.4f}, {r['ic95'][1]:.4f}] · "
                  f"Δ {r['dif_vs_r10']:+.4f} [{r['ic95_dif'][0]:+.4f}, {r['ic95_dif'][1]:+.4f}] · "
                  f"gana {r['dias_gana_r10']*100:.0f} %")
    json.dump(R, open(f"{SALIDA}/dos_35_replay.json", "w"), indent=1, ensure_ascii=False)
    alg = json.load(open(f"{SALIDA}/dos_35_algoritmos.json"))
    alg["temporadas"] = R
    json.dump(alg, open(f"{SALIDA}/dos_35_algoritmos.json", "w"), indent=1, ensure_ascii=False)
    print(f"guardado {SALIDA}/dos_35_replay.json y la clave «temporadas» de dos_35_algoritmos.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paso", choices=["entrenar", "replay", "veredicto"])
    p.add_argument("--anio", type=int, choices=[2025, 2026])
    a = p.parse_args()
    if a.paso == "replay":
        replay(a.anio)
    else:
        {"entrenar": entrenar, "veredicto": veredicto}[a.paso]()
