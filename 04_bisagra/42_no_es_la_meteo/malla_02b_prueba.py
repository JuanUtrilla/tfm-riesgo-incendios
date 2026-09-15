#!/usr/bin/env python3
"""
Módulo 2b, paso 1: ¿arregla el denominador de ERA5-Land la saturación del IFS?

No toca producción. Escribe salida/prueba_denominador_ifs.json.

La pregunta
El mapa de hoy y mañana no puede salir del reanálisis: ERA5-Land llega con
~6 días de retraso. Tiene que venir del IFS (previsión). Y el IFS ya se midió
en la primera versión del proyecto contra el denominador del cubo:

    saturación de fwi_pctl_local (% en pctl >= 99,9), 120 estaciones
        cubo (referencia)      0,22 %
        ERA5 reanálisis        0,76 %
        IFS previsión          2,44 %   ← inaceptable

Si esos 2,44 % vienen del numerador (el IFS tiene la cola más gorda, máximo
136,7 contra 107,2 del cubo) cambiar el denominador no arregla nada y el 02b
no sirve tal cual. Si vienen del desajuste entre numerador y denominador, la
climatología de ERA5-Land del módulo 4 lo cancela igual que hizo con el
reanálisis, y el 02b sale adelante.

Diseño: 2x2, que es lo que separa la causa
Medir solo "IFS contra clim ERA5-Land" no distingue las dos hipótesis. Con
las cuatro celdas sí:

                        denom: clim ERA5-Land     denom: clim cubo
    num: ERA5-Land            A (la malla)              D
    num: IFS                  B (tiempo real)           C (~2,44 % conocido)

    · C alto y B bajo  → el problema era el desajuste. El 02b sale adelante.
    · B y C los dos altos → el problema es la cola del IFS. Hay que replantear.
    · D alto y A bajo  → confirma que el efecto es del denominador, no del
      numerador, que es la hipótesis de todo el pipeline.

Muestra. 120 nodos, el mismo tamaño que el experimento original para que las
cifras sean del mismo orden de precisión. Se excluyen los 226 nodos costeros
sin dato. Evaluación jun-sep 2024, fuera del periodo de la climatología
(2008-2012).

Spin-up. Los dos numeradores arrancan el 1-jun-2024, el mismo día, para que
la comparación no mezcle el efecto del spin-up con el de la fuente. Es también
lo único que hay en disco de ERA5-Land. Que ese arranque tardío no mueve el
resultado está medido en el módulo 5: −0,30 de FWI a BUI alto, corr 0,9999.

Cuota. Open-Meteo cobra nodos × tramos de 14 días y limita por minuto, no
solo al día: 120 nodos × 9 tramos = 1.080 unidades, pero mandarlas seguidas
devuelve 429. De ahí el freno de PAUSA segundos entre lotes y el reintento
con retroceso. La descarga se cachea por lotes, así que un corte no tira lo
ya bajado.

Uso: python malla_02b_prueba.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr
from pyproj import Transformer

import config
from fwi_canadiense import calcular_fwi_serie
from malla_02_descarga import a_diario, abre, pide_mes

N_NODOS = 120
IFS_INI, IFS_FIN = "2024-06-01", "2024-09-30"
EVAL = (6, 9)
ANIOS_CLIM_CUBO = (2008, 2014)
VARS = ("temperature_2m_max,relative_humidity_2m_min,"
        "wind_speed_10m_max,precipitation_sum")
CACHE = config.salida(f"ifs_nodos_{N_NODOS}.parquet")
PAUSA = 15        # s entre lotes; con 10 s seguidos salta el 429


def muestra_nodos():
    """Nodos con dato, repartidos por el país (no un cluster)."""
    n = np.load(config.NODOS)
    la, lo = n["nodo_lat"], n["nodo_lon"]
    clim = np.load(config.CLIM)
    vivos = np.where(clim["n7"] > 0)[0]          # excluye los costeros a NaN
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(vivos, N_NODOS, replace=False))
    return idx, la[idx], lo[idx]


def serie_ifs(la, lo):
    """IFS diario por lotes de 10 puntos, con freno, reintento y caché parcial."""
    if os.path.exists(CACHE):
        print(f"  IFS: cacheado ({CACHE})", flush=True)
        return pd.read_parquet(CACHE)
    parcial = CACHE + ".parcial"
    filas = []
    hechos = set()
    if os.path.exists(parcial):
        prev = pd.read_parquet(parcial)
        filas.append(prev)
        hechos = set(prev["nodo"].unique())
        print(f"  IFS: reanudando, {len(hechos)} nodos ya bajados", flush=True)

    for k in range(0, len(la), 10):
        if all(k + i in hechos for i in range(min(10, len(la) - k))):
            continue
        sl = slice(k, k + 10)
        espera = PAUSA
        for intento in range(6):
            r = requests.get(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params=dict(latitude=",".join(f"{v:.4f}" for v in la[sl]),
                            longitude=",".join(f"{v:.4f}" for v in lo[sl]),
                            start_date=IFS_INI, end_date=IFS_FIN, daily=VARS,
                            models="ecmwf_ifs025", timezone="UTC",
                            wind_speed_unit="ms"), timeout=180)
            if r.status_code != 429:
                break
            print(f"    429 · esperando {espera}s", flush=True)
            time.sleep(espera)
            espera = min(espera * 2, 300)
        r.raise_for_status()
        j = r.json()
        for i, b in enumerate(j if isinstance(j, list) else [j]):
            d = b["daily"]
            filas.append(pd.DataFrame({
                "nodo": k + i, "fecha": pd.to_datetime(d["time"]),
                "tmax": d["temperature_2m_max"],
                "hr_min": d["relative_humidity_2m_min"],
                "viento_max": d["wind_speed_10m_max"],
                "prec": d["precipitation_sum"]}))
        pd.concat(filas, ignore_index=True).to_parquet(parcial, index=False)
        print(f"    ifs: {min(k + 10, len(la))}/{len(la)}", flush=True)
        time.sleep(PAUSA)
    df = pd.concat(filas, ignore_index=True)
    df.to_parquet(CACHE, index=False)
    os.remove(parcial)
    return df


def serie_era5land(la, lo, fechas):
    """ERA5-Land en los nodos, de los meses ya descargados (jun-sep 2024)."""
    rutas = [pide_mes(2024, m, list(range(1, 32))) for m in (6, 7, 8, 9)]
    ps = [a_diario(abre(r)) for r in rutas]
    dim = [d for d in ps[0].dims if d not in ("latitude", "longitude")][0]
    ds = xr.concat(ps, dim=dim).rename({dim: "fecha"}).sortby("fecha").load()
    pts = ds.sel(latitude=xr.DataArray(la, dims="n"),
                 longitude=xr.DataArray(lo, dims="n"), method="nearest")
    f = pd.to_datetime(ds["fecha"].values).normalize()
    pos = {t: i for i, t in enumerate(f)}
    S = {}
    for v in ("tmax", "hr_min", "viento_max", "prec"):
        col = pts[v].values
        M = np.full((len(fechas), len(la)), np.nan)
        for i, t in enumerate(fechas):
            j = pos.get(t)
            if j is not None:
                M[i] = col[j]
        S[v] = M
    ds.close()
    return S


def fwi_matriz(S, meses):
    M = np.full(S["tmax"].shape, np.nan)
    for j in range(S["tmax"].shape[1]):
        M[:, j] = calcular_fwi_serie(S["tmax"][:, j], S["hr_min"][:, j],
                                     S["viento_max"][:, j] * 3.6,
                                     S["prec"][:, j], meses)["fwi"]
    return M


def clim_cubo(la, lo):
    """Climatología del cubo (2008-2014, por mes) en las celdas de los nodos."""
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    xs, ys = ds["x"].values, ds["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X, Y = tr.transform(lo, la)
    ix = np.rint((X - xs[0]) / (xs[1] - xs[0])).astype(int)
    iy = np.rint((Y - ys[0]) / (ys[1] - ys[0])).astype(int)
    t = ds["time"].values.astype("datetime64[D]")
    anios = t.astype("datetime64[Y]").astype(int) + 1970
    meses = (t.astype("datetime64[M]").astype(int) % 12) + 1
    out = {}
    for m in range(EVAL[0], EVAL[1] + 1):
        idx = np.where((anios >= ANIOS_CLIM_CUBO[0])
                       & (anios <= ANIOS_CLIM_CUBO[1]) & (meses == m))[0]
        # Se lee la rejilla entera por bloques y se indexa después. Indexar
        # 120 puntos sueltos en el isel obliga al NetCDF a traer los chunks
        # completos igualmente, y sale ~10x más lento (medido: >13 min por mes
        # contra ~1 min así).
        trozos = []
        for i in range(0, len(idx), 30):
            blo = ds["FWI"].isel(time=idx[i:i + 30]).values
            trozos.append(blo[:, iy, ix])
            del blo
        out[m] = np.concatenate(trozos, axis=0)
        print(f"    clim cubo mes {m}: {out[m].shape[0]} días", flush=True)
    ds.close()
    return out


def percentiles(num, meses_ev, ev, denom_mes):
    """fwi_pctl_local celda a celda: dónde cae el FWI en su climatología."""
    p = []
    for k in np.where(ev)[0]:
        C = denom_mes[meses_ev[k]]
        for j in range(num.shape[1]):
            v = num[k, j]
            if not np.isfinite(v):
                continue
            c = C[:, j]
            c = c[np.isfinite(c)]
            if len(c):
                p.append(np.searchsorted(np.sort(c), v, side="right")
                         / len(c) * 100)
    return np.array(p)


def main():
    idx, la, lo = muestra_nodos()
    fechas = pd.date_range(IFS_INI, IFS_FIN, freq="D")
    meses = fechas.month.values
    ev = np.asarray((fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1]))
    print(f"{N_NODOS} nodos · {len(fechas)} días · evaluación {ev.sum()} días "
          f"jun-sep 2024\n", flush=True)

    print("  descargando IFS...", flush=True)
    ifs = serie_ifs(la, lo)
    S_ifs = {}
    for v in ("tmax", "hr_min", "viento_max", "prec"):
        M = ifs.pivot(index="fecha", columns="nodo", values=v)
        S_ifs[v] = M.reindex(fechas).values

    print("\n  ERA5-Land en los nodos...", flush=True)
    S_era = serie_era5land(la, lo, fechas)

    print("\n  FWI...", flush=True)
    F = {"IFS": fwi_matriz(S_ifs, meses), "ERA5-Land": fwi_matriz(S_era, meses)}

    print("\n  denominadores...", flush=True)
    cl = np.load(config.CLIM)
    D = {"clim ERA5-Land": {m: cl[f"m{m}"][:, idx] for m in
                            range(EVAL[0], EVAL[1] + 1)},
         "clim cubo": clim_cubo(la, lo)}

    print("\n" + "=" * 74)
    print("SATURACIÓN DE fwi_pctl_local — 2x2 numerador x denominador")
    print("=" * 74)
    print(f"{'numerador':<14}{'denominador':<18}{'pctl med':>10}"
          f"{'% >=95':>9}{'% >=99,9':>11}{'FWI medio':>11}")
    res = {"n_nodos": N_NODOS, "dias_eval": int(ev.sum()), "celdas": {}}
    tabla = {}
    for nn, num in F.items():
        for dn, den in D.items():
            p = percentiles(num, meses, ev, den)
            fila = dict(pctl_mediana=float(np.median(p)),
                        ge95=float((p >= 95).mean() * 100),
                        ge999=float((p >= 99.9).mean() * 100),
                        fwi_medio=float(np.nanmean(num[ev])),
                        n=int(len(p)))
            tabla[(nn, dn)] = fila
            res["celdas"][f"{nn} / {dn}"] = fila
            print(f"{nn:<14}{dn:<18}{fila['pctl_mediana']:>10.1f}"
                  f"{fila['ge95']:>9.1f}{fila['ge999']:>11.2f}"
                  f"{fila['fwi_medio']:>11.2f}")

    b = tabla[("IFS", "clim ERA5-Land")]["ge999"]
    c = tabla[("IFS", "clim cubo")]["ge999"]
    a = tabla[("ERA5-Land", "clim ERA5-Land")]["ge999"]
    d = tabla[("ERA5-Land", "clim cubo")]["ge999"]

    print("\n" + "-" * 74)
    print(f"  efecto del DENOMINADOR con numerador IFS : {c:.2f} % → {b:.2f} %")
    print(f"  efecto del DENOMINADOR con numerador ERA5: {d:.2f} % → {a:.2f} %")
    print(f"  efecto del NUMERADOR con clim ERA5-Land  : {a:.2f} % → {b:.2f} %")
    if b < 1.0:
        ver = ("EL 02b SALE ADELANTE — con el denominador de ERA5-Land el IFS "
               "aterriza por debajo del 1 %")
    elif b < c * 0.5:
        ver = ("MEJORA PARCIAL — el denominador ayuda pero no basta; queda "
               "cola del IFS por tratar")
    else:
        ver = ("HAY QUE REPLANTEAR — el problema es la cola del IFS, no el "
               "desajuste de denominador")
    res["veredicto"] = ver
    print(f"\n  → {ver}")

    with open(config.salida("prueba_denominador_ifs.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado: {config.salida('prueba_denominador_ifs.json')}")


if __name__ == "__main__":
    main()
