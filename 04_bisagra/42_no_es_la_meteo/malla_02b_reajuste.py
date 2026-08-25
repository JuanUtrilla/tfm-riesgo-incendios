#!/usr/bin/env python3
"""
Módulo 2b, paso 4: reajustar el mapeo IFS→ERA5-Land con toda la malla.

NO TOCA PRODUCCIÓN. Escribe salida/mapeo_ifs_a_era5land.npz (el que usa
`riesgo_hoy.py`) y salida/reajuste_mapeo.json. El mapeo anterior queda en
salida/mapeo_ifs_a_era5land_120nodos.npz.

=============================================================================
POR QUÉ REAJUSTAR
=============================================================================
El mapeo en uso se ajustó con 120 nodos y jun-jul 2024: unas 7.300 parejas
(nodo, día). El mapeo tiene 501 puntos de cuantil, así que el extremo de la
curva —el percentil 99,9, que es justo el que decide la saturación— se estima
con un puñado de muestras. Ahí es donde más se nota ampliar la muestra.

=============================================================================
LA CUOTA MANDA EL DISEÑO
=============================================================================
Open-Meteo cobra nodos × tramos de 14 días. La temporada jun-sep son 9 tramos:

    5.605 nodos x 9 tramos = 50.445 unidades  ·  límite: 10.000 al día

No cabe en un día. Y el límite es TAMBIÉN por minuto (~600): un lote de 200
nodos cuesta 1.800 unidades y devuelve 429 él solo.

De ahí las dos decisiones:

 1. LOTES DE 50 nodos (450 unidades) con un minuto de espera. Unas 22 min
    consumen la cuota diaria.
 2. TEMPORADA COMPLETA POR NODO, iterando nodos en orden disperso (barajado
    con semilla fija). Cada día de cuota añade ~1.100 nodos con la serie
    ENTERA, que es lo que sirve para ajustar; lo contrario —todos los nodos
    con dos semanas— daría un mapeo atado a un único régimen de tiempo.

Es reanudable: se relanza al día siguiente y sigue por donde iba, hasta
completar los 5.605. Con lo bajado hasta el momento ya reajusta y valida, así
que cada pasada deja un mapeo mejor que el anterior.

=============================================================================
VALIDACIÓN, SIN CIRCULARIDAD
=============================================================================
Igual que en el paso 3: se ajusta con jun-jul y se evalúa con ago-sep. Y se
compara contra el mapeo viejo de 120 nodos SOBRE LOS MISMOS días y nodos, que
es la única forma de saber si ampliar la muestra sirvió de algo.

Uso:
    python malla_02b_reajuste.py                  # consume la cuota del día
    python malla_02b_reajuste.py --presupuesto 4000
    python malla_02b_reajuste.py --solo-ajuste    # sin bajar nada más
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr

import config
from fwi_canadiense import calcular_fwi_serie
from malla_02_descarga import a_diario, abre, pide_mes

INI, FIN = "2024-06-01", "2024-09-30"
TRAMOS = 9                      # ceil(122 días / 14)
LOTE = 50                       # 50 x 9 = 450 unidades, bajo el tope/minuto
PAUSA = 60
L = 6                           # retraso real de ERA5-Land
Q = np.linspace(0, 1, 501)
VARS = ("temperature_2m_max,relative_humidity_2m_min,"
        "wind_speed_10m_max,precipitation_sum")
CACHE = config.salida("ifs_reajuste.parquet")


def orden_nodos():
    """Todos los nodos, barajados: cualquier prefijo cubre el país entero."""
    n = np.load(config.NODOS)
    la, lo = n["nodo_lat"], n["nodo_lon"]
    rng = np.random.default_rng(20240601)
    return rng.permutation(len(la)), la, lo


def descarga(presupuesto):
    orden, la, lo = orden_nodos()
    hechos = set()
    filas = []
    if os.path.exists(CACHE):
        prev = pd.read_parquet(CACHE)
        filas, hechos = [prev], set(prev["nodo"].unique())
        print(f"  ya en caché: {len(hechos):,} nodos", flush=True)

    pend = [int(i) for i in orden if int(i) not in hechos]
    n_lotes = min(presupuesto // (LOTE * TRAMOS), int(np.ceil(len(pend) / LOTE)))
    if not n_lotes:
        print("  nada que bajar (o presupuesto agotado)", flush=True)
        return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
    print(f"  pendientes {len(pend):,} nodos · esta pasada {n_lotes} lotes "
          f"({n_lotes * LOTE} nodos, {n_lotes * LOTE * TRAMOS:,} unidades)",
          flush=True)

    for b in range(n_lotes):
        idx = pend[b * LOTE:(b + 1) * LOTE]
        if not idx:
            break
        espera, corta = PAUSA, False
        for _ in range(5):
            r = requests.get(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params=dict(
                    latitude=",".join(f"{la[i]:.4f}" for i in idx),
                    longitude=",".join(f"{lo[i]:.4f}" for i in idx),
                    start_date=INI, end_date=FIN, daily=VARS,
                    models="ecmwf_ifs025", timezone="UTC",
                    wind_speed_unit="ms"), timeout=240)
            if r.status_code != 429:
                break
            print(f"    429 · esperando {espera}s", flush=True)
            time.sleep(espera)
            espera = min(espera * 2, 900)
        else:
            print("    cuota agotada — se guarda lo bajado y se sigue mañana",
                  flush=True)
            corta = True
        if corta:
            break
        r.raise_for_status()
        j = r.json()
        for i, blo in zip(idx, j if isinstance(j, list) else [j]):
            d = blo["daily"]
            filas.append(pd.DataFrame({
                "nodo": i, "fecha": pd.to_datetime(d["time"]),
                "tmax": d["temperature_2m_max"],
                "hr_min": d["relative_humidity_2m_min"],
                "viento_max": d["wind_speed_10m_max"],
                "prec": d["precipitation_sum"]}))
        pd.concat(filas, ignore_index=True).to_parquet(CACHE, index=False)
        print(f"    lote {b + 1}/{n_lotes} · {len(hechos) + (b + 1) * LOTE:,} "
              f"nodos en total", flush=True)
        if b < n_lotes - 1:
            time.sleep(PAUSA)
    return pd.read_parquet(CACHE)


def era5land(idx, la, lo, fechas):
    rutas = [pide_mes(2024, m, list(range(1, 32))) for m in (6, 7, 8, 9)]
    ps = [a_diario(abre(r)) for r in rutas]
    dim = [d for d in ps[0].dims if d not in ("latitude", "longitude")][0]
    ds = xr.concat(ps, dim=dim).rename({dim: "fecha"}).sortby("fecha").load()
    pts = ds.sel(latitude=xr.DataArray(la[idx], dims="n"),
                 longitude=xr.DataArray(lo[idx], dims="n"), method="nearest")
    f = pd.to_datetime(ds["fecha"].values).normalize()
    pos = {t: i for i, t in enumerate(f)}
    S = {}
    for v in ("tmax", "hr_min", "viento_max", "prec"):
        col = pts[v].values
        M = np.full((len(fechas), len(idx)), np.nan)
        for i, t in enumerate(fechas):
            j = pos.get(t)
            if j is not None:
                M[i] = col[j]
        S[v] = M
    ds.close()
    return S


def main(a):
    _, la, lo = orden_nodos()
    if not a.solo_ajuste:
        df = descarga(a.presupuesto)
    else:
        df = pd.read_parquet(CACHE)
    idx = np.sort(df["nodo"].unique())
    print(f"\n  ajustando con {len(idx):,} nodos de 5.605 "
          f"({len(idx) / 5605 * 100:.0f} %)", flush=True)

    fechas = pd.date_range(INI, FIN, freq="D")
    meses = fechas.month.values
    fit = np.isin(meses, [6, 7])
    tst = np.isin(meses, [8, 9])
    I = {v: df.pivot(index="fecha", columns="nodo", values=v)
         .reindex(fechas)[idx].values
         for v in ("tmax", "hr_min", "viento_max", "prec")}
    print("  ERA5-Land en esos nodos...", flush=True)
    R = era5land(idx, la, lo, fechas)

    nn = len(idx)
    print(f"  FWI ({nn:,} nodos, dos ramas)...", flush=True)
    E = np.full((len(fechas), nn), np.nan)
    H = np.full((len(fechas), nn), np.nan)
    for j in range(nn):
        o = calcular_fwi_serie(R["tmax"][:, j], R["hr_min"][:, j],
                               R["viento_max"][:, j] * 3.6, R["prec"][:, j],
                               meses)
        E[:, j] = o["fwi"]
        for k in range(L, len(fechas)):
            if not np.isfinite(o["ffmc"][k - L - 1]):
                continue
            H[k, j] = calcular_fwi_serie(
                I["tmax"][k - L:k + 1, j], I["hr_min"][k - L:k + 1, j],
                I["viento_max"][k - L:k + 1, j] * 3.6,
                I["prec"][k - L:k + 1, j], meses[k - L:k + 1],
                ffmc0=o["ffmc"][k - L - 1], dmc0=o["dmc"][k - L - 1],
                dc0=o["dc"][k - L - 1])["fwi"][-1]

    x, y = H[fit].ravel(), E[fit].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    xs, ys = np.quantile(x[ok], Q), np.quantile(y[ok], Q)
    print(f"  parejas de ajuste: {ok.sum():,} "
          f"(el mapeo de 120 nodos tenía ~7.300)", flush=True)

    viejo = np.load(config.salida("mapeo_ifs_a_era5land_120nodos.npz"))
    clim = np.load(config.CLIM)
    C = {m: clim[f"m{m}"][:, idx] for m in (8, 9)}
    orden = {m: [np.sort(C[m][:, j][np.isfinite(C[m][:, j])])
                 for j in range(nn)] for m in C}

    def mide(M):
        p = []
        for k in np.where(tst)[0]:
            col = orden[meses[k]]
            for j in range(nn):
                v = M[k, j]
                if np.isfinite(v) and len(col[j]):
                    p.append(np.searchsorted(col[j], v, side="right")
                             / len(col[j]) * 100)
        p = np.array(p)
        return dict(pctl_mediana=float(np.median(p)),
                    ge95=float((p >= 95).mean() * 100),
                    ge999=float((p >= 99.9).mean() * 100), n=int(len(p)))

    Hn = np.where(np.isfinite(H), np.interp(H, xs, ys), np.nan)
    Hv = np.where(np.isfinite(H), np.interp(H, viejo["xs"], viejo["ys"]), np.nan)

    print("\n" + "=" * 74)
    print("FUERA DE MUESTRA — ajuste jun-jul 2024, evaluación ago-sep 2024")
    print("=" * 74)
    print(f"{'numerador':<38}{'pctl med':>10}{'% >=95':>9}{'% >=99,9':>11}")
    res = {"n_nodos": int(nn), "parejas_ajuste": int(ok.sum()), "brazos": {}}
    for nom, M in [("reanálisis (referencia)", E),
                   ("híbrido crudo", H),
                   ("híbrido + mapeo de 120 nodos", Hv),
                   (f"híbrido + mapeo de {nn:,} nodos", Hn)]:
        s = mide(M)
        res["brazos"][nom] = s
        print(f"{nom:<38}{s['pctl_mediana']:>10.1f}{s['ge95']:>9.1f}"
              f"{s['ge999']:>11.2f}")

    np.savez_compressed(config.salida("mapeo_ifs_a_era5land.npz"),
                        xs=xs, ys=ys, lag=L, n_nodos=nn,
                        ajustado_con=f"jun-jul 2024, {nn} nodos")
    with open(config.salida("reajuste_mapeo.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    falta = 5605 - nn
    print(f"\n  mapeo guardado ({nn:,} nodos)."
          + (f" Quedan {falta:,} nodos: relanzar mañana." if falta > 0
             else " Malla completa."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--presupuesto", type=int, default=10000)
    p.add_argument("--solo-ajuste", action="store_true")
    main(p.parse_args())
