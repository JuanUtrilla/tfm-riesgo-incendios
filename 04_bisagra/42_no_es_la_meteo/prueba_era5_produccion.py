#!/usr/bin/env python3
"""
¿Puede ERA5 (malla) sustituir a AEMET como entrada meteo de producción?

No sobrescribe nada. Escribe solo dataset/prueba_era5_produccion.json
y un resumen por pantalla.

La pregunta
-----------
`fwi_pctl_local` = percentil del FWI de hoy dentro de la climatología de esa
celda. En entrenamiento numerador y denominador salen los dos del cubo
(ERA5-Land). En producción el numerador pasó a ser un FWI de estación AEMET y
el denominador se quedó en el cubo, y el percentil satura (8,1 % de estaciones
en pctl ≥99,9 un día cualquiera, frente al 1,3 % de entrenamiento).

Hay dos formas de arreglarlo:
  (a) mover el denominador a AEMET  → clim_fwi_aemet (ya construido; pierde
      217 de 683 estaciones por falta de años completos)
  (b) mover el numerador de vuelta a la familia ERA5 → esta prueba

Si (b) funciona, no hay que reentrenar ni reconstruir la climatología: la
entrada vuelve a la distribución de entrenamiento por construcción.

El diseño: aislar la fuente, congelar la implementación
-------------------------------------------------------
Hay tres sabores de FWI en el proyecto y confundirlos invalida la prueba:

  1. `ds["FWI"]` del cubo         : implementación de IberFire sobre ERA5-Land
                                    (la que vio el entrenamiento)
  2. fwi_canadiense sobre el cubo : implementación propia sobre ERA5-Land
                                    (la de `clim_fwi/`, el denominador)
  3. fwi_canadiense sobre AEMET   : implementación propia sobre estación
                                    (el numerador de producción hoy)

El percentil de producción empareja 3 contra 2: misma implementación, distinta
fuente. El sesgo de implementación se cancela y lo único que se mide es la
fuente. Por eso las tres ramas de abajo usan siempre `calcular_fwi_serie`, y
(1) se calcula solo como control.

Tres numeradores, mismas estaciones y mismos días:

  CUBO   fwi_canadiense sobre la meteo del cubo   → la referencia: es, por
                                                    definición, la escala en
                                                    la que vive el denominador
  ERA5   fwi_canadiense sobre Open-Meteo          → el candidato
         (era5_seamless: T/HR de ERA5-Land 9 km,
          viento/precip de ERA5 31 km; ERA5-Land
          no sirve viento ni precipitación)
  AEMET  fwi_canadiense sobre la estación         → lo que hace producción hoy

Métrica que decide: el percentil de cada numerador contra `clim_fwi/<idema>.npz`
(el denominador real de producción). Gana quien reproduzca el comportamiento de
CUBO. Si ERA5 ≈ CUBO y AEMET satura, la arquitectura (b) queda validada.

Periodo: spin-up desde 2023-01-01 (el DC tiene ~52 días de constante de tiempo),
evaluación jun-sep 2024. 2024 es el último año que cubre el cubo (acaba el
2024-12-31), así que es el único solape posible con datos reales de las tres
fuentes.

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python prueba_era5_produccion.py [--n 120]
"""

import argparse
import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr

from fwi_canadiense import calcular_fwi_serie

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
CACHE = ("/tmp/claude-1000/-home-charredgem-Desktop-Master/"
         "232d0846-5a96-422a-9c0b-bf08265e8663/scratchpad")
SPIN = "2023-01-01"
FIN = "2024-12-31"
EVAL = (6, 9)                      # jun-sep: la temporada que importa


# --------------------------------------------------------------------------
def estaciones(n):
    """Estaciones con: celda en el cubo, climatología clim_fwi y datos AEMET."""
    est = pd.read_parquet(f"{DIR}/prototipo/estaciones_prototipo.parquet")
    est = est[[os.path.exists(f"{DIR}/prototipo/clim_fwi/{i}.npz")
               for i in est["idema"]]]
    con = sqlite3.connect(f"{DIR}/aemet_historico.db")
    vivas = pd.read_sql(
        "SELECT idema, COUNT(*) n FROM climatologia_diaria "
        "WHERE fecha BETWEEN ? AND ? AND tmax IS NOT NULL GROUP BY idema",
        con, params=(SPIN, FIN))
    con.close()
    est = est.merge(vivas[vivas["n"] >= 600], on="idema")
    # muestreo espacialmente repartido: rejilla 6x6 sobre lat/lon, hasta n
    est = est.assign(gx=pd.qcut(est.lon, 6, labels=False, duplicates="drop"),
                     gy=pd.qcut(est.lat, 6, labels=False, duplicates="drop"))
    return (est.groupby(["gx", "gy"], group_keys=False)
               .apply(lambda g: g.sample(min(len(g), max(1, n // 30)),
                                         random_state=1))
               .head(n).reset_index(drop=True))


def serie_cubo(est, fechas):
    """Meteo del cubo en las celdas de las estaciones (una lectura vectorizada)."""
    ds = xr.open_dataset(f"{DIR}/iberfire/IberFire.nc", decode_timedelta=False)
    t = ds["time"].values.astype("datetime64[D]")
    m = (t >= np.datetime64(SPIN)) & (t <= np.datetime64(FIN))
    idx = np.where(m)[0]
    iy = xr.DataArray(est["iy"].values, dims="p")
    ix = xr.DataArray(est["ix"].values, dims="p")
    out = {}
    for var in ["t2m_max", "RH_min", "wind_speed_max",
                "total_precipitation_mean", "FWI"]:
        out[var] = ds[var].isel(time=idx, y=iy, x=ix).values   # (tiempo, punto)
        print(f"    cubo: {var} leído", flush=True)
    ds.close()
    assert len(t[m]) == len(fechas), "calendario del cubo != calendario pedido"
    return out


def serie_openmeteo(est):
    """era5_seamless por lotes, cacheado en disco (la API es lenta)."""
    ruta = f"{CACHE}/om_era5_{len(est)}.parquet"
    if os.path.exists(ruta):
        return pd.read_parquet(ruta)
    vs = ("temperature_2m_max,relative_humidity_2m_min,"
          "wind_speed_10m_max,precipitation_sum")
    filas = []
    for k in range(0, len(est), 10):
        lote = est.iloc[k:k + 10]
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=dict(latitude=",".join(f"{v:.4f}" for v in lote.lat),
                        longitude=",".join(f"{v:.4f}" for v in lote.lon),
                        start_date=SPIN, end_date=FIN, daily=vs,
                        models="era5_seamless", timezone="UTC",
                        wind_speed_unit="ms"), timeout=180).json()
        bloques = r if isinstance(r, list) else [r]
        for idema, b in zip(lote["idema"], bloques):
            d = b["daily"]
            filas.append(pd.DataFrame({
                "idema": idema, "fecha": pd.to_datetime(d["time"]),
                "tmax": d["temperature_2m_max"],
                "hr_min": d["relative_humidity_2m_min"],
                "viento_max": d["wind_speed_10m_max"],
                "prec": d["precipitation_sum"]}))
        print(f"    open-meteo: {min(k+10, len(est))}/{len(est)}", flush=True)
        time.sleep(1.5)
    df = pd.concat(filas, ignore_index=True)
    df.to_parquet(ruta, index=False)
    return df


def serie_aemet(est):
    """Misma receta exacta que tiempo_real.serie_diaria_aemet (m/s, mm)."""
    con = sqlite3.connect(f"{DIR}/aemet_historico.db")
    q = ("SELECT idema, fecha, tmax, hrmin, velmedia, racha, prec "
         "FROM climatologia_diaria WHERE fecha BETWEEN ? AND ? "
         f"AND idema IN ({','.join('?' * len(est))})")
    df = pd.read_sql(q, con, params=[SPIN, FIN] + list(est["idema"]))
    con.close()
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", "."),
                                  errors="coerce")
    return pd.DataFrame({
        "idema": df["idema"], "fecha": pd.to_datetime(df["fecha"]),
        "tmax": num(df["tmax"]), "hr_min": num(df["hrmin"]),
        "viento_max": np.minimum(num(df["velmedia"]) * 1.5, num(df["racha"])),
        "prec": num(df["prec"].replace("Ip", "0"))})


def fwi_de(df_est, fechas):
    """Reindexa a calendario continuo y corre calcular_fwi_serie."""
    d = (df_est.set_index("fecha").reindex(fechas)
         if len(df_est) else pd.DataFrame(index=fechas))
    for c in ["tmax", "hr_min", "viento_max", "prec"]:
        if c not in d:
            d[c] = np.nan
    return calcular_fwi_serie(d["tmax"].values, d["hr_min"].values,
                              d["viento_max"].values * 3.6, d["prec"].values,
                              fechas.month.values)["fwi"]


# --------------------------------------------------------------------------
def main(n):
    est = estaciones(n)
    fechas = pd.date_range(SPIN, FIN, freq="D")
    m_eval = (fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1]) \
        & (fechas.year == 2024)
    print(f"estaciones: {len(est)} | días: {len(fechas)} "
          f"| evaluación: {m_eval.sum()} días de jun-sep 2024\n")

    print("  leyendo el cubo...", flush=True)
    cubo = serie_cubo(est, fechas)
    print("  descargando Open-Meteo...", flush=True)
    om = serie_openmeteo(est)
    print("  leyendo AEMET local...", flush=True)
    ae = serie_aemet(est)

    filas = []
    for j, e in est.iterrows():
        i = e["idema"]
        s = {}
        s["CUBO"] = calcular_fwi_serie(
            cubo["t2m_max"][:, j], cubo["RH_min"][:, j],
            cubo["wind_speed_max"][:, j] * 3.6,
            cubo["total_precipitation_mean"][:, j], fechas.month.values)["fwi"]
        s["ERA5"] = fwi_de(om[om.idema == i], fechas)
        s["AEMET"] = fwi_de(ae[ae.idema == i], fechas)
        s["CUBO_NATIVO"] = cubo["FWI"][:, j]          # control, no comparable

        clim = np.load(f"{DIR}/prototipo/clim_fwi/{i}.npz")
        for k in range(len(fechas)):
            if not m_eval[k]:
                continue
            f = {"idema": i, "fecha": fechas[k], "mes": fechas[k].month}
            c = clim[f"m{fechas[k].month}"]
            c = np.sort(c[~np.isnan(c)])
            for nom, v in s.items():
                f[nom] = v[k]
                f[f"pctl_{nom}"] = (np.searchsorted(c, v[k], side="right")
                                    / len(c) * 100) if len(c) and np.isfinite(v[k]) else np.nan
            filas.append(f)
    d = pd.DataFrame(filas)

    # ---- resultados -------------------------------------------------------
    res = {"n_estaciones": int(len(est)), "n_filas": int(len(d)),
           "periodo_eval": "jun-sep 2024", "fuentes": {}}
    print("\n" + "=" * 74)
    print("FWI (jun-sep 2024) — la escala de cada fuente")
    print("=" * 74)
    print(f"{'fuente':<14}{'n':>7}{'mediana':>10}{'p95':>9}{'max':>9}"
          f"{'corr vs CUBO':>14}{'sesgo':>9}")
    ref = d["CUBO"]
    for nom in ["CUBO", "ERA5", "AEMET", "CUBO_NATIVO"]:
        v = d[nom]
        ok = v.notna() & ref.notna()
        cor = v[ok].corr(ref[ok]) if ok.sum() > 100 else np.nan
        ses = (v[ok] - ref[ok]).mean() if ok.sum() > 100 else np.nan
        print(f"{nom:<14}{v.notna().sum():>7}{v.median():>10.1f}"
              f"{v.quantile(.95):>9.1f}{v.max():>9.1f}{cor:>14.3f}{ses:>+9.1f}")
        res["fuentes"][nom] = dict(n=int(v.notna().sum()),
                                   mediana=float(v.median()),
                                   p95=float(v.quantile(.95)),
                                   corr_vs_cubo=float(cor), sesgo=float(ses))

    print("\n" + "=" * 74)
    print("LA MÉTRICA QUE DECIDE: fwi_pctl_local contra clim_fwi (el cubo)")
    print("=" * 74)
    print(f"{'numerador':<14}{'mediana pctl':>14}{'% >=95':>10}"
          f"{'% >=99,9':>11}")
    for nom in ["CUBO", "ERA5", "AEMET"]:
        p = d[f"pctl_{nom}"].dropna()
        print(f"{nom:<14}{p.median():>14.1f}{(p >= 95).mean() * 100:>10.1f}"
              f"{(p >= 99.9).mean() * 100:>11.1f}")
        res["fuentes"][nom].update(pctl_mediana=float(p.median()),
                                   pctl_pc_ge95=float((p >= 95).mean() * 100),
                                   pctl_pc_ge999=float((p >= 99.9).mean() * 100))
    print("\nreferencia de ENTRENAMIENTO (dataset v4, jul-ago): "
          "mediana 56,2 · ≥95 8,3 % · ≥99,9 1,3 %")
    print("CUBO es la referencia interna: es la fuente en la que vive el "
          "denominador.\nGana el numerador que más se le parezca.")

    with open(f"{DIR}/dataset/prueba_era5_produccion.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado: dataset/prueba_era5_produccion.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    main(ap.parse_args().n)
