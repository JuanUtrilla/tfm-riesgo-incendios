#!/usr/bin/env python3
"""
EXPERIMENTO B, fase 2 — puntuar el modelo con tres fuentes de meteo distintas.

NO SOBRESCRIBE NADA. Escribe solo dataset/experimento_b_fase2_*.{parquet,json}.

=============================================================================
EL DISEÑO
=============================================================================
Mismos días, mismas estaciones, misma etiqueta, mismo modelo (xgb_v2), MISMO
código de construcción de features. Lo único que cambia es de dónde sale la
meteo:

  BRAZO A · ERA5-Land   → la fuente que vio el modelo AL ENTRENAR (dentro del
                          cubo IberFire, aquí cruda a 0,1°)
  BRAZO B · AEMET obs   → serie diaria observada, reconstruida con este mismo
                          script desde la API de climatológicos
  BRAZO C · AEMET fcst  → las probabilidades SELLADAS por commit (prevision_D0),
                          que no se pueden recalcular a posteriori: un forecast
                          emitido el día D-1 ya no se puede volver a pedir

La diferencia A−B es el desplazamiento de FUENTE con el mismo horizonte
(observado contra observado). La diferencia B−C es el coste de PREDECIR en vez
de observar. Juntas descomponen el hueco que VALIDACION.md §4b dejó abierto.

=============================================================================
EL CONTROL QUE HACE CREÍBLE TODO LO DEMÁS
=============================================================================
El brazo B se compara contra los `ranking_<fecha>.csv` que producción commiteó
en su día. Si mi pipeline reproduce esas probabilidades, el montaje es fiel y
las diferencias entre brazos son atribuibles a los datos. Si NO las reproduce,
hay un bug en mi copia y el resto del experimento no vale nada — igual que el
CONTROL validó `ablacion_proxies_operativos.py`.

=============================================================================
FIRMS
=============================================================================
`frp_max_50km_7d` y `n_detec_50km_7d` no dependen de la fuente meteo: se
calculan una vez y se usan idénticas en A y B, para que no contaminen la
comparación. Se replica exactamente lo que hace ranking_diario.firms_frp:
solo VIIRS_NOAA20_NRT, radio 50 km, ventana [D-5, D-1].

Uso:  /home/charredgem/miniconda3/envs/tfm_fuego/bin/python experimento_b_fase2.py
"""

import json
import os
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DIR = Path(__file__).parent
REPO_OP = Path("/home/charredgem/Desktop/Master/aemet_horario_verano2026")
sys.path.insert(0, str(DIR))
sys.path.insert(0, str(REPO_OP))

from experimento_b_era5 import (DIAS_SPINUP, UMBRALES, cargar_produccion,  # noqa: E402
                                construir, serie_estaciones, SALIDA_SERIE)

SAL_FEATS = DIR / "dataset" / "experimento_b_fase2_features.parquet"
SAL_SERIE_AEMET = DIR / "dataset" / "experimento_b_serie_aemet.parquet"
SAL_JSON = DIR / "dataset" / "experimento_b_fase2_resultados.json"

INICIO = pd.Timestamp("2026-04-12")
FIN = pd.Timestamp("2026-08-13")
RADIO_EFFIS, AREA_MIN, MARGEN = 25, 30, 3
N_BOOT = 2000


def env(nombre, alias=()):
    """Lee del entorno o del .env de TFM_fuego (mismo patrón que
    descargar_verdad_operativa.py)."""
    for n in (nombre, *alias):
        if os.environ.get(n):
            return os.environ[n]
    ruta = DIR / ".env"
    if ruta.exists():
        for linea in ruta.read_text().splitlines():
            for n in (nombre, *alias):
                if linea.startswith(f"{n}="):
                    return linea.split("=", 1)[1].strip()
    raise SystemExit(f"falta {nombre} (entorno o .env)")


# --------------------------------------------------------------------------- #
# BRAZO B — serie diaria observada de AEMET, rango explícito
# --------------------------------------------------------------------------- #
def serie_aemet(f0, f1):
    """Climatológicos diarios de todas las estaciones entre f0 y f1.

    Es `ranking_diario.serie_diaria_todas` con el rango como parámetro en vez
    de 'los últimos 80 días': para comparar con ERA5-Land los dos brazos deben
    tener EXACTAMENTE el mismo spinup de FWI, y el de producción se mide desde
    'hoy', que no es el día que estamos evaluando."""
    key = env("AEMET_API_KEY", ("TOKEN_AEMET",))
    filas, ini = [], f0
    while ini <= f1:
        fin = min(ini + pd.Timedelta(days=13), f1)
        url = (f"https://opendata.aemet.es/opendata/api/valores/climatologicos/"
               f"diarios/datos/fechaini/{ini:%Y-%m-%d}T00:00:00UTC/"
               f"fechafin/{fin:%Y-%m-%d}T23:59:59UTC/todasestaciones")
        for intento in range(5):
            try:
                r = requests.get(url, headers={"api_key": key}, timeout=30).json()
                if r.get("estado") == 200:
                    d = requests.get(r["datos"], timeout=120)
                    d.encoding = "latin-1"
                    filas += d.json()
                    break
            except Exception as e:
                print(f"  [{ini:%F}] error {type(e).__name__}", flush=True)
            print(f"  [{ini:%F}] reintento {intento+1}/5 en 70s", flush=True)
            time.sleep(70)
        print(f"  {ini:%F}→{fin:%F}: {len(filas):,} filas acumuladas", flush=True)
        time.sleep(3)
        ini = fin + pd.Timedelta(days=1)

    df = pd.DataFrame(filas)

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")

    velmedia, racha = num(df["velmedia"]), num(df["racha"])
    out = pd.DataFrame({
        "idema": df["indicativo"], "fecha": pd.to_datetime(df["fecha"]),
        "tmax": num(df["tmax"]), "tmin": num(df["tmin"]),
        "hr_min": num(df.get("hrMin", pd.Series(dtype=float))),
        "viento_max": np.minimum(velmedia * 1.5, racha),
        "prec": num(df["prec"].replace("Ip", "0")), "n_horas": 24})
    return (out.sort_values(["idema", "fecha"])
               .drop_duplicates(["idema", "fecha"], keep="first")
               .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# FIRMS — idénticas en los dos brazos
# --------------------------------------------------------------------------- #
def firms_por_dia(est, dias):
    """frp_max / n_detec a <50 km en [D-5, D-1]. Replica ranking_diario.firms_frp
    (solo VIIRS_NOAA20_NRT) pero para muchos días."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    key = env("FIRMS_MAP_KEY")
    d0, d1 = min(dias) - pd.Timedelta(days=5), max(dias)
    trozos, d = [], d0
    while d <= d1:
        n = min(5, (d1 - d).days + 1)
        url = ("https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
               f"{key}/VIIRS_NOAA20_NRT/-10,35,5,44/{n}/{d:%Y-%m-%d}")
        r = requests.get(url, timeout=120)
        if r.ok and r.text.startswith("latitude"):
            trozos.append(pd.read_csv(StringIO(r.text)))
        else:
            print(f"  aviso FIRMS {d:%F}: sin datos", flush=True)
        d += pd.Timedelta(days=n)
    det = pd.concat(trozos, ignore_index=True)
    det["fecha"] = pd.to_datetime(det["acq_date"])
    print(f"FIRMS: {len(det):,} detecciones {det.fecha.min():%F}→{det.fecha.max():%F}")

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ex, ey = tr.transform(est["lon"].values, est["lat"].values)
    est_xy = np.column_stack([ex, ey])
    out = []
    for dia in dias:
        v = det[(det["fecha"] >= dia - pd.Timedelta(days=5)) &
                (det["fecha"] <= dia - pd.Timedelta(days=1))]
        if not len(v):
            out.append(pd.DataFrame({"idema": est["idema"], "fecha": str(dia.date()),
                                     "frp_max_50km_7d": 0.0, "n_detec_50km_7d": 0.0}))
            continue
        fx, fy = tr.transform(v["longitude"].values, v["latitude"].values)
        vec = cKDTree(np.column_stack([fx, fy])).query_ball_point(est_xy, r=50000)
        frp = v["frp"].values
        out.append(pd.DataFrame({
            "idema": est["idema"], "fecha": str(dia.date()),
            "frp_max_50km_7d": [float(frp[k].max()) if k else 0.0 for k in vec],
            "n_detec_50km_7d": [float(len(k)) for k in vec]}))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# evaluación
# --------------------------------------------------------------------------- #
def etiquetas_effis(feats):
    from validar_modelo import perimetros_effis, etiquetar_effis
    polis, ultimo = perimetros_effis(AREA_MIN)
    corte = ultimo - pd.Timedelta(days=MARGEN)
    y = np.zeros(len(feats), dtype=int)
    for dia, g in feats.groupby("fecha"):
        d = pd.Timestamp(dia)
        if d > corte:
            y[g.index] = -1                       # fuera por latencia
            continue
        y[g.index] = etiquetar_effis(g, polis.get(d, []), RADIO_EFFIS)
    return y, corte


def auc_boot(df, col_prob, rng):
    """AUC-ROC agrupando por día, con IC95 por bootstrap de días.

    Descarta las filas con score NaN (hay estaciones sin `clim_fwi/<idema>.npz`,
    y ahí `fwi_pctl_local` no existe) y devuelve el n realmente usado, para que
    no se compare en silencio sobre subconjuntos distintos."""
    from sklearn.metrics import roc_auc_score
    df = df[np.isfinite(df[col_prob].values)]
    dias = sorted(df["fecha"].unique())
    por_dia = {d: g for d, g in df.groupby("fecha")}

    def auc(sel):
        y = np.concatenate([por_dia[d]["y"].values for d in sel])
        p = np.concatenate([por_dia[d][col_prob].values for d in sel])
        return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan

    obs = auc(dias)
    b = [a for a in (auc(rng.choice(dias, len(dias), replace=True))
                     for _ in range(N_BOOT)) if np.isfinite(a)]
    return obs, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(df)


def main():
    import xgboost as xgb
    est = pd.read_parquet(REPO_OP / "modelo" / "estaciones_prototipo.parquet")
    est_i = est.set_index("idema")
    FEATS = json.loads((REPO_OP / "modelo" /
                        "xgb_v2_prototipo_features.json").read_text())
    modelo = xgb.XGBClassifier()
    modelo.load_model(str(REPO_OP / "modelo" / "xgb_v2_prototipo.ubj"))

    prod = cargar_produccion()
    dias = sorted(pd.to_datetime(prod["fecha"].unique()))
    dias = [d for d in dias if INICIO + pd.Timedelta(days=DIAS_SPINUP) <= d <= FIN]
    print(f"días a evaluar: {len(dias)} ({dias[0]:%F} → {dias[-1]:%F})\n")

    # --- series de los dos brazos reconstruibles
    if SALIDA_SERIE.exists():
        s_era5 = pd.read_parquet(SALIDA_SERIE)
    else:
        s_era5 = serie_estaciones(est)
        s_era5.to_parquet(SALIDA_SERIE)
    print(f"serie ERA5-Land : {len(s_era5):,} estación-día")

    if SAL_SERIE_AEMET.exists():
        s_aemet = pd.read_parquet(SAL_SERIE_AEMET)
    else:
        print("descargando serie AEMET observada...")
        s_aemet = serie_aemet(INICIO, FIN)
        s_aemet.to_parquet(SAL_SERIE_AEMET)
    print(f"serie AEMET obs : {len(s_aemet):,} estación-día\n")

    # --- FIRMS, idéntico para los dos
    firms = firms_por_dia(est, dias)

    # --- features + puntuación
    tablas = {}
    for nombre, serie in [("ERA5-Land", s_era5), ("AEMET obs", s_aemet)]:
        f = construir(serie, est_i, dias)
        f = f.drop(columns=["frp_max_50km_7d", "n_detec_50km_7d"], errors="ignore")
        f = f.merge(firms, on=["idema", "fecha"], how="left")
        f[["frp_max_50km_7d", "n_detec_50km_7d"]] = \
            f[["frp_max_50km_7d", "n_detec_50km_7d"]].fillna(0.0)
        f["prob"] = modelo.predict_proba(f[FEATS])[:, 1]
        f["nivel"] = f["prob"].map(lambda p: next(n for u, n in UMBRALES if p >= u))
        f["brazo"] = nombre
        tablas[nombre] = f
        print(f"{nombre}: {len(f):,} estación-día puntuadas")

    # --- CONTROL: ¿reproduce el brazo AEMET obs lo que commiteó producción?
    rk = prod[prod["tipo"] == "ranking (día cerrado)"][["idema", "fecha", "prob"]]
    ctrl = tablas["AEMET obs"].merge(rk, on=["idema", "fecha"],
                                     suffixes=("_mio", "_prod"))
    control = {"n": int(len(ctrl))}
    if len(ctrl):
        control.update(
            r=round(float(ctrl["prob_mio"].corr(ctrl["prob_prod"])), 4),
            mae=round(float((ctrl["prob_mio"] - ctrl["prob_prod"]).abs().mean()), 4),
            sesgo=round(float((ctrl["prob_mio"] - ctrl["prob_prod"]).mean()), 4))
    print(f"\n=== CONTROL (brazo AEMET obs vs ranking_*.csv commiteado) ===\n"
          f"{control}\n")

    # --- etiquetas y AUC por brazo
    largo = pd.concat(tablas.values(), ignore_index=True)
    y, corte = etiquetas_effis(largo)
    largo["y"] = y
    largo = largo[largo["y"] >= 0]
    print(f"etiqueta EFFIS ≥{AREA_MIN} ha, radio {RADIO_EFFIS} km, "
          f"días ≤ {corte:%F} · prevalencia {largo['y'].mean():.2%}\n")

    rng = np.random.default_rng(42)
    res = {}
    print("=== AUC-ROC por brazo (IC95 bootstrap por día) ===")
    for nombre, g in largo.groupby("brazo"):
        g = g.reset_index(drop=True)
        o, lo, hi, n_m = auc_boot(g, "prob", rng)
        # baseline FWI percentil calculado con la MISMA fuente
        of, lof, hif, n_f = auc_boot(g, "fwi_pctl_local", rng)
        res[nombre] = {"n": int(len(g)), "dias": int(g["fecha"].nunique()),
                       "auc_modelo": round(o, 4), "ic95_modelo": [round(lo, 4), round(hi, 4)],
                       "n_modelo": int(n_m),
                       "auc_fwi_pctl": round(of, 4),
                       "ic95_fwi_pctl": [round(lof, 4), round(hif, 4)],
                       "n_fwi_pctl": int(n_f)}
        print(f"  {nombre:<12} modelo {o:.4f} [{lo:.4f}, {hi:.4f}] (n={n_m:,}) · "
              f"FWI pctl {of:.4f} [{lof:.4f}, {hif:.4f}] (n={n_f:,})")

    # brazo C: probabilidades selladas, tal cual las publicó producción
    d0 = prod[prod["tipo"] == "D0 (sellada)"].copy()
    d0 = d0[d0["fecha"].isin(largo["fecha"].unique())].reset_index(drop=True)
    yc, _ = etiquetas_effis(d0)
    d0["y"] = yc
    d0 = d0[d0["y"] >= 0].reset_index(drop=True)
    o, lo, hi, n_c = auc_boot(d0, "prob", rng)
    res["AEMET fcst (D0 sellada)"] = {
        "n": int(len(d0)), "dias": int(d0["fecha"].nunique()),
        "auc_modelo": round(o, 4), "ic95_modelo": [round(lo, 4), round(hi, 4)],
        "nota": "probabilidades selladas por commit; no recalculables"}
    print(f"  {'AEMET fcst':<12} modelo {o:.4f} [{lo:.4f}, {hi:.4f}] (n={n_c:,})")

    largo.to_parquet(SAL_FEATS)
    SAL_JSON.write_text(json.dumps(
        {"dias": [str(d.date()) for d in dias], "control": control,
         "etiqueta": {"fuente": "EFFIS", "area_min_ha": AREA_MIN,
                      "radio_km": RADIO_EFFIS, "corte": str(corte.date())},
         "resultados": res}, indent=2, ensure_ascii=False))
    print(f"\nguardado: {SAL_JSON}")


if __name__ == "__main__":
    main()
