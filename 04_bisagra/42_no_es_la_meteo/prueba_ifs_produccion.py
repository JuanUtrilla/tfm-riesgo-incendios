#!/usr/bin/env python3
"""
¿Y con la previsión del ECMWF (IFS) en vez del reanálisis? (§6bis)

No sobrescribe nada. Escribe dataset/prueba_ifs_produccion.json.

En §6 se validó el reanálisis ERA5(-Land) contra el cubo: r=0,976. Pero
producción necesita servir D y D+1, y para eso no hay reanálisis, solo
previsión. El IFS operativo es otro modelo (ciclo más moderno que el
congelado de ERA5, y 0,25° ≈ 28 km en vez de 9 km), así que su acuerdo con el
cubo no se puede dar por supuesto a partir de §6.

Esta prueba lo mide con el mismo diseño: mismas 120 estaciones, mismos días
(jun-sep 2024), misma implementación de FWI, cubo cacheado de la corrida
anterior.

Interpretación. La arquitectura real es híbrida: reanálisis para el historial
(hasta D−6) y previsión solo para los últimos días y D+1. Simular eso
exactamente exige una reconstrucción rodante día a día. En su lugar se miden
las dos cotas, que es suficiente para decidir:

    ERA5 puro  → cota superior del híbrido (todo reanálisis)   [§6: −2,0]
    IFS puro   → cota inferior del híbrido (todo previsión)    [esta prueba]

Si el IFS puro ya sale aceptable, el híbrido lo será por construcción, porque
el FWI es un integrador con memoria larga (DC ~52 días) y en el híbrido la
mayor parte del estado viene del reanálisis, que es la rama buena.

Uso: python prueba_ifs_produccion.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
import requests

from fwi_canadiense import calcular_fwi_serie
from prueba_era5_produccion import (CACHE, DIR, EVAL, FIN, SPIN, estaciones,
                                    serie_openmeteo)
from prueba_era5_atribucion import N, cubo_cacheado

VARS = ("temperature_2m_max,relative_humidity_2m_min,"
        "wind_speed_10m_max,precipitation_sum")


def serie_ifs(est):
    """Archivo de pasadas del IFS (previsión), cacheado."""
    ruta = f"{CACHE}/om_ifs_{len(est)}.parquet"
    if os.path.exists(ruta):
        return pd.read_parquet(ruta)
    filas = []
    for k in range(0, len(est), 10):
        lote = est.iloc[k:k + 10]
        r = requests.get(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            params=dict(latitude=",".join(f"{v:.4f}" for v in lote.lat),
                        longitude=",".join(f"{v:.4f}" for v in lote.lon),
                        start_date=SPIN, end_date=FIN, daily=VARS,
                        models="ecmwf_ifs025", timezone="UTC",
                        wind_speed_unit="ms"), timeout=180).json()
        for idema, b in zip(lote["idema"], r if isinstance(r, list) else [r]):
            d = b["daily"]
            filas.append(pd.DataFrame({
                "idema": idema, "fecha": pd.to_datetime(d["time"]),
                "tmax": d["temperature_2m_max"],
                "hr_min": d["relative_humidity_2m_min"],
                "viento_max": d["wind_speed_10m_max"],
                "prec": d["precipitation_sum"]}))
        print(f"    ifs: {min(k+10, len(est))}/{len(est)}", flush=True)
        time.sleep(1.5)
    df = pd.concat(filas, ignore_index=True)
    df.to_parquet(ruta, index=False)
    return df


def main():
    est = estaciones(N)
    fechas = pd.date_range(SPIN, FIN, freq="D")
    m_ev = (fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1]) \
        & (fechas.year == 2024)
    print(f"estaciones {len(est)} | evaluación {m_ev.sum()} días jun-sep 2024\n")

    cubo = cubo_cacheado(est, fechas)
    print("  descargando IFS...", flush=True)
    ifs = serie_ifs(est)
    om = serie_openmeteo(est)

    def matriz(df, c):
        return np.column_stack([
            df[df.idema == i].set_index("fecha").reindex(fechas)[c].values
            for i in est["idema"]])

    C = {"tmax": cubo["t2m_max"], "hr_min": cubo["RH_min"],
         "viento_max": cubo["wind_speed_max"],
         "prec": cubo["total_precipitation_mean"]}
    F = {c: matriz(ifs, c) for c in C}
    E = {c: matriz(om, c) for c in C}

    res = {"n_estaciones": int(len(est)), "variables": {}, "fwi": {}}

    print("\n" + "=" * 72)
    print("VARIABLE A VARIABLE contra el cubo (jun-sep 2024)")
    print("=" * 72)
    print(f"{'variable':<16}{'cubo':>9}{'ERA5':>9}{'IFS':>9}"
          f"{'sesgo IFS':>11}{'corr IFS':>10}")
    for c, u in [("tmax", "°C"), ("hr_min", "%"),
                 ("viento_max", "m/s"), ("prec", "mm")]:
        a, b, f = C[c][m_ev].ravel(), E[c][m_ev].ravel(), F[c][m_ev].ravel()
        ok = np.isfinite(a) & np.isfinite(f)
        oke = np.isfinite(a) & np.isfinite(b)
        cor = np.corrcoef(a[ok], f[ok])[0, 1]
        print(f"{c+' ('+u+')':<16}{a[ok].mean():>9.2f}{b[oke].mean():>9.2f}"
              f"{f[ok].mean():>9.2f}{f[ok].mean()-a[ok].mean():>+11.2f}{cor:>10.3f}")
        res["variables"][c] = dict(cubo=float(a[ok].mean()),
                                   era5=float(b[oke].mean()),
                                   ifs=float(f[ok].mean()),
                                   sesgo_ifs=float(f[ok].mean() - a[ok].mean()),
                                   corr_ifs=float(cor))

    def fwi_m(src):
        out = np.full((len(fechas), len(est)), np.nan)
        for j in range(len(est)):
            out[:, j] = calcular_fwi_serie(
                src["tmax"][:, j], src["hr_min"][:, j],
                src["viento_max"][:, j] * 3.6, src["prec"][:, j],
                fechas.month.values)["fwi"]
        return out

    print("\n" + "=" * 72)
    print("FWI y percentil contra clim_fwi (el denominador de producción)")
    print("=" * 72)
    fw = {"CUBO": fwi_m(C), "ERA5": fwi_m(E), "IFS": fwi_m(F)}
    clim = {i: np.load(f"{DIR}/prototipo/clim_fwi/{i}.npz")
            for i in est["idema"]}
    print(f"{'fuente':<10}{'mediana':>9}{'máx':>9}{'corr':>8}{'sesgo':>9}"
          f"{'pctl med':>10}{'% >=99,9':>11}")
    for nom, M in fw.items():
        a = fw["CUBO"][m_ev].ravel(); b = M[m_ev].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        p = []
        for j, i in enumerate(est["idema"]):
            for k in np.where(m_ev)[0]:
                v = M[k, j]
                if not np.isfinite(v):
                    continue
                c = clim[i][f"m{fechas[k].month}"]
                c = np.sort(c[~np.isnan(c)])
                if len(c):
                    p.append(np.searchsorted(c, v, side="right") / len(c) * 100)
            
        p = np.array(p)
        cor = np.corrcoef(a[ok], b[ok])[0, 1]
        ses = b[ok].mean() - a[ok].mean()
        print(f"{nom:<10}{np.nanmedian(b):>9.1f}{np.nanmax(b):>9.1f}"
              f"{cor:>8.3f}{ses:>+9.2f}{np.median(p):>10.1f}"
              f"{(p >= 99.9).mean()*100:>11.1f}")
        res["fwi"][nom] = dict(mediana=float(np.nanmedian(b)),
                               maximo=float(np.nanmax(b)), corr=float(cor),
                               sesgo=float(ses), pctl_mediana=float(np.median(p)),
                               pctl_pc_ge999=float((p >= 99.9).mean() * 100))

    with open(f"{DIR}/dataset/prueba_ifs_produccion.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("\nGuardado: dataset/prueba_ifs_produccion.json")


if __name__ == "__main__":
    main()
