#!/usr/bin/env python3
"""
Verificación meteorológica: ¿qué fuente predice mejor el tiempo?

No modifica nada. Crea solo dataset/verificacion_fuentes_meteo.{parquet,json}.

Por qué esta comparación sí se puede medir
------------------------------------------
Cualquier comparación de aciertos del modelo sobre la temporada 2026 choca con
la muestra: hay 184 incendios en la serie sellada, y con eso el IC95 del AUC
mide ±0,08. No da para decidir nada.

La pregunta "¿esta fuente predice bien el tiempo?" no necesita datos de
incendios: se compara la predicción contra lo que realmente pasó. Ahí la
muestra son ~13.600 estaciones-día, ochenta veces más. Es la única capa del
sistema que hoy se puede medir con precisión.

La motivación es concreta: la red de AEMET tiene huecos (16,3 % de
estaciones-día sin viento; 128 de 858 estaciones sin viento en toda la serie) y
no está distribuida uniformemente (`popdens` media 840 en las estaciones frente
a 100 en las celdas del cubo). Una malla global no tiene ninguno de los dos
problemas. Queda por ver si predice igual o mejor.

Qué se compara
--------------
Verdad-terreno: observación de AEMET (climatológicos diarios consolidados).
Es lo que ocurrió en el punto de la estación.

Candidatos, sobre las mismas estaciones y los mismos días:

  · Previsión municipal de AEMET, la que usa producción hoy. Sus valores
    están guardados en los CSV sellados por commit, así que son la previsión
    real que se emitió y no una reconstrucción.
  · ECMWF IFS (Open-Meteo, archivo de pasadas históricas): malla completa,
    misma casa que produce ERA5. Se piden varias familias más para tener
    contexto: AIFS (el modelo de IA de ECMWF), GFS de NOAA e ICON de DWD.

Métricas: sesgo, MAE, RMSE y correlación por variable. El viento es el foco:
es el eslabón roto del sistema (previsión de AEMET r=0,45 frente a 0,93 de la
temperatura) y entra al FWI por el ISI, que es multiplicativo.

Lo que esta comparación no dice
-------------------------------
Que una fuente prediga mejor la meteorología no implica que el modelo de
incendios rinda mejor con ella: eso hay que medirlo aparte y choca otra vez con
los 184 positivos. Aquí se responde solo la primera mitad, que es la que tiene
respuesta.

Hay una asimetría que declarar: la previsión de AEMET es municipal
(predicción para el municipio de la estación) y el IFS es una celda de malla
interpolada al punto. No es exactamente el mismo objeto.

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python verificar_fuentes_meteo.py
"""

import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DIR = Path(__file__).parent
REPO_OP = Path("/home/charredgem/Desktop/Master/aemet_horario_verano2026")
SAL_PARQUET = DIR / "dataset" / "verificacion_fuentes_meteo.parquet"
SAL_JSON = DIR / "dataset" / "verificacion_fuentes_meteo.json"

INICIO, FIN = "2026-07-14", "2026-08-13"
MODELOS = ["ecmwf_ifs025", "ecmwf_aifs025_single", "gfs_seamless", "icon_seamless"]
LOTE = 60          # estaciones por petición
VARS = ["temperature_2m_max", "relative_humidity_2m_min",
        "wind_speed_10m_max", "precipitation_sum"]
EQUIV = {"temperature_2m_max": "t2m_max", "relative_humidity_2m_min": "rh_min",
         "wind_speed_10m_max": "viento_max", "precipitation_sum": "prec"}


def pedir(lats, lons, modelo, intentos=4):
    url = ("https://historical-forecast-api.open-meteo.com/v1/forecast"
           f"?latitude={','.join(f'{v:.4f}' for v in lats)}"
           f"&longitude={','.join(f'{v:.4f}' for v in lons)}"
           f"&start_date={INICIO}&end_date={FIN}"
           f"&daily={','.join(VARS)}&models={modelo}"
           "&wind_speed_unit=ms&timezone=UTC")
    for i in range(intentos):
        try:
            r = requests.get(url, timeout=180)
            if r.ok:
                d = r.json()
                return d if isinstance(d, list) else [d]
            print(f"    HTTP {r.status_code}: {r.text[:120]}", flush=True)
        except Exception as e:
            print(f"    {type(e).__name__}", flush=True)
        time.sleep(20 * (i + 1))
    return None


def descargar(est):
    filas = []
    for modelo in MODELOS:
        print(f"  {modelo}...", flush=True)
        ok = 0
        for i in range(0, len(est), LOTE):
            lote = est.iloc[i:i + LOTE]
            d = pedir(lote.lat.values, lote.lon.values, modelo)
            if d is None:
                continue
            for idema, resp in zip(lote.idema.values, d):
                da = resp.get("daily")
                if not da:
                    continue
                f = pd.DataFrame(da).rename(columns=EQUIV)
                f["idema"], f["modelo"] = idema, modelo
                f = f.rename(columns={"time": "fecha"})
                filas.append(f)
                ok += 1
            time.sleep(1)
        print(f"    {ok} estaciones", flush=True)
    return pd.concat(filas, ignore_index=True)


def main():
    est = pd.read_parquet(REPO_OP / "modelo" / "estaciones_prototipo.parquet")
    obs = pd.read_parquet(DIR / "dataset" / "experimento_b_serie_aemet.parquet")
    obs["fecha"] = obs.fecha.dt.strftime("%Y-%m-%d")
    obs = obs.rename(columns={"tmax": "t2m_max_obs", "hr_min": "rh_min_obs",
                              "viento_max": "viento_max_obs", "prec": "prec_obs"})
    obs = obs[["idema", "fecha", "t2m_max_obs", "rh_min_obs",
               "viento_max_obs", "prec_obs"]]
    print(f"{len(est)} estaciones · observación {len(obs):,} estación-día\n")

    if SAL_PARQUET.exists():
        malla = pd.read_parquet(SAL_PARQUET)
        print(f"previsiones de malla ya descargadas: {len(malla):,}")
    else:
        malla = descargar(est)
        malla.to_parquet(SAL_PARQUET)
        print(f"\nguardado: {SAL_PARQUET}")

    # previsión municipal de AEMET, tal y como se selló por commit
    filas = []
    for h in ["D0", "D1"]:
        for r in sorted(glob.glob(str(REPO_OP / "rankings" /
                                      f"prevision_{h}_2026-0[78]-*.csv"))):
            d = pd.read_csv(r)
            d["fecha"] = re.search(r"(\d{4}-\d{2}-\d{2})", r).group(1)
            d["modelo"] = f"aemet_municipal_{h}"
            filas.append(d[["idema", "fecha", "modelo", "t2m_max",
                            "rh_min", "viento_max"]])
    aemet = pd.concat(filas, ignore_index=True)

    todo = pd.concat([malla, aemet], ignore_index=True)
    j = todo.merge(obs, on=["idema", "fecha"], how="inner")
    j = j[(j.fecha >= INICIO) & (j.fecha <= FIN)]
    print(f"\nemparejado con observación: {len(j):,} filas\n")

    res = {}
    print(f"{'fuente':<24}{'variable':<14}{'n':>8}{'r':>8}{'sesgo':>9}"
          f"{'MAE':>8}{'RMSE':>8}")
    for modelo, g in j.groupby("modelo"):
        res[modelo] = {}
        for c, u in [("t2m_max", "°C"), ("rh_min", "%"),
                     ("viento_max", "m/s"), ("prec", "mm")]:
            if c not in g or f"{c}_obs" not in g:
                continue
            a, b = pd.to_numeric(g[c], errors="coerce"), g[f"{c}_obs"]
            ok = a.notna() & b.notna()
            if ok.sum() < 200:
                continue
            e = (a - b)[ok]
            res[modelo][c] = {
                "n": int(ok.sum()), "r": round(float(a[ok].corr(b[ok])), 3),
                "sesgo": round(float(e.mean()), 3),
                "mae": round(float(e.abs().mean()), 3),
                "rmse": round(float(np.sqrt((e ** 2).mean())), 3)}
            m = res[modelo][c]
            print(f"{modelo:<24}{c+' ('+u+')':<14}{m['n']:>8,}{m['r']:>8.3f}"
                  f"{m['sesgo']:>9.2f}{m['mae']:>8.2f}{m['rmse']:>8.2f}")
        print()

    SAL_JSON.write_text(json.dumps(
        {"periodo": [INICIO, FIN], "verdad_terreno": "observación AEMET",
         "resultados": res,
         "caveat": "la previsión de AEMET es municipal y la de malla es una "
                   "celda interpolada al punto: no son exactamente el mismo "
                   "objeto"}, indent=2, ensure_ascii=False))
    print(f"guardado: {SAL_JSON}")


if __name__ == "__main__":
    main()
