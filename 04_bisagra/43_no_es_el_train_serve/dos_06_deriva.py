#!/usr/bin/env python3
"""
Dos modelos — paso 6: deriva de features entre ENTRENAR (cubo) y SERVIR
(ERA5-Land en los nodos). §3.5 del encargo.

NO TOCA PRODUCCIÓN. Lee cubo y `_cds` 2024 (solo lectura); escribe
salida/dos_06_deriva.json.

=============================================================================
POR QUÉ
=============================================================================
La decisión de entrenar con el cubo y servir ERA5-Land directo se apoya en
que el cubo ES ERA5-Land reprocesado. `malla_02_vs_cubo.json` lo midió con
correlaciones y sesgos de MEDIAS. Pero un árbol no ve medias: ve la
distribución entera, y la métrica que el proyecto usa para «¿es la misma
distribución?» es el PSI (`auditoria_train_serve.json`; >0,25 = sospechosa).
Aquí se calcula el PSI feature a feature, con la receta de servicio REAL:

  · extremos diarios (tmax, hr_min, viento_max, prec) de `a_diario`
  · FWI con las 13 UTC (la receta buena) y con el proxy tmax/hr_min (la mala)
  · ventanas (precip_30d, fwi_med_7d, dias_sin_lluvia) sobre cada serie
  · fwi_pctl_local con numerador y denominador de la MISMA fuente en cada
    rama (cubo/cubo, malla-proxy/clim-proxy), y la combinación cruzada
    (FWI 13 UTC / clim-proxy) que es la que habría si se cambia el numerador
    sin rehacer la climatología.

300 nodos al azar, jun-sep 2024 (junio es rodaje del FWI; se puntúa jul-sep).
Los bins del PSI se definen sobre la distribución del CUBO (entrenamiento).
"""

import json

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import config
from fwi_canadiense import calcular_fwi_serie
from malla_02_descarga import a_diario, abre, hr_desde_rocio, pide_mes

N = 300
MESES = (6, 7, 8, 9)
EVAL = (7, 8, 9)


def psi(train, serve, bins=10):
    train, serve = train[np.isfinite(train)], serve[np.isfinite(serve)]
    q = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    q[0], q[-1] = -np.inf, np.inf
    pt = np.histogram(train, q)[0] / len(train)
    ps = np.histogram(serve, q)[0] / len(serve)
    pt, ps = np.clip(pt, 1e-4, None), np.clip(ps, 1e-4, None)
    return float(np.sum((ps - pt) * np.log(ps / pt)))


def ventanas(fwi, prec):
    """precip_30d, fwi_med_7d, dias_sin_lluvia por columna, ventanas [D-w, D-1]."""
    T, n = fwi.shape
    p30 = np.full_like(fwi, np.nan); f7 = np.full_like(fwi, np.nan)
    secos = np.full_like(fwi, np.nan)
    for t in range(30, T):
        p30[t] = np.nansum(prec[t - 30:t], 0)
        f7[t] = np.nanmean(fwi[t - 7:t], 0)
        s = np.zeros(n)
        for j in range(n):
            k = t
            while k >= 0 and s[j] < 120:
                if np.isnan(prec[k, j]) or prec[k, j] >= 1.0:
                    break
                s[j] += 1; k -= 1
        secos[t] = s
    return p30, f7, secos


def main():
    n = np.load(config.NODOS)
    la, lo = n["nodo_lat"], n["nodo_lon"]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(la), N, replace=False))
    LA, LO = xr.DataArray(la[sel], dims="n"), xr.DataArray(lo[sel], dims="n")

    # ---- ERA5-Land en los nodos ----------------------------------------------
    D, H13 = [], []
    for m in MESES:
        ds = abre(pide_mes(2024, m, list(range(1, 32))))
        d = a_diario(ds).sel(latitude=LA, longitude=LO, method="nearest").load()
        D.append(d)
        t = pd.to_datetime(ds["valid_time"].values)
        s = ds.isel(valid_time=np.where(t.hour == 13)[0]) \
              .sel(latitude=LA, longitude=LO, method="nearest").load()
        H13.append(pd.DataFrame({
            "fecha": pd.to_datetime(s["valid_time"].values).normalize(),
            "t": list(s["t2m"].values - 273.15),
            "h": list(hr_desde_rocio(s["t2m"].values, s["d2m"].values)),
            "v": list(np.sqrt(s["u10"].values ** 2 + s["v10"].values ** 2))}))
        ds.close()
        print(f"  2024-{m:02d} leído", flush=True)
    d = xr.concat(D, "fecha").sortby("fecha")
    fechas = pd.to_datetime(d["fecha"].values)
    M = {v: d[v].values for v in ("tmax", "hr_min", "viento_max", "prec")}
    h = pd.concat(H13).set_index("fecha").reindex(fechas)
    T13 = np.array([np.asarray(x) for x in h["t"]])
    H13 = np.array([np.asarray(x) for x in h["h"]])
    V13 = np.array([np.asarray(x) for x in h["v"]])
    meses = fechas.month.values

    def serie(T, HR, V):
        out = np.full(T.shape, np.nan)
        for j in range(T.shape[1]):
            out[:, j] = calcular_fwi_serie(T[:, j], HR[:, j], V[:, j] * 3.6,
                                           M["prec"][:, j], meses)["fwi"]
        return out
    FWI13 = serie(T13, H13, V13)
    FWIpx = serie(M["tmax"], M["hr_min"], M["viento_max"])

    # ---- el cubo en las mismas celdas, mismo periodo + clim 2008-2014 ------
    cu = xr.open_dataset(config.CUBO, decode_timedelta=False)
    xs, ys = cu["x"].values, cu["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X, Y = tr.transform(lo[sel], la[sel])
    ix = np.rint((X - xs[0]) / (xs[1] - xs[0])).astype(int)
    iy = np.rint((Y - ys[0]) / (ys[1] - ys[0])).astype(int)
    tc = pd.to_datetime(cu["time"].values)
    a, b = int(np.where(tc == fechas[0])[0][0]), int(np.where(tc == fechas[-1])[0][0])
    C = {}
    for v, cv in [("fwi", "FWI"), ("tmax", "t2m_max"), ("hr_min", "RH_min"),
                  ("viento_max", "wind_speed_max"), ("prec", "total_precipitation_mean")]:
        C[v] = cu[cv].isel(time=slice(a, b + 1)).values[:, iy, ix]
        print(f"  cubo {v}", flush=True)
    # climatología del cubo por celda y mes (2008-2014), leída por bloques
    kc = np.where((tc.year >= 2008) & (tc.year <= 2014))[0]
    mc = tc.month.values[kc]
    clim_cubo = {m: [None] * N for m in EVAL}
    for (by, bx), idx in pd.DataFrame({"by": iy // 77, "bx": ix // 99}).groupby(["by", "bx"]).groups.items():
        sub = cu["FWI"].isel(time=kc, y=slice(by * 77, by * 77 + 77),
                             x=slice(bx * 99, bx * 99 + 99)).values
        for j in idx:
            s = sub[:, iy[j] - by * 77, ix[j] - bx * 99]
            for m in EVAL:
                c = s[mc == m]; clim_cubo[m][j] = np.sort(c[np.isfinite(c)])
    cu.close()
    print("  clim cubo lista", flush=True)
    _z = np.load(config.CLIM)
    cl = {f"m{m}": _z[f"m{m}"] for m in EVAL}   # npz descomprime en cada acceso: cachear

    def pctl(fwi, clim_de):
        out = np.full(fwi.shape, np.nan)
        for t in range(fwi.shape[0]):
            m = meses[t]
            if m not in EVAL:
                continue
            for j in range(N):
                c = clim_de(m, j)
                if len(c):
                    out[t, j] = np.searchsorted(c, fwi[t, j], side="right") / len(c) * 100
        return out
    def clim_malla(m, j):
        c = cl[f"m{m}"][:, sel[j]]; return c[np.isfinite(c)]   # ya ordenada
    P_cubo = pctl(C["fwi"], lambda m, j: clim_cubo[m][j])
    P_px = pctl(FWIpx, clim_malla)
    P_13_climpx = pctl(FWI13, clim_malla)

    # ventanas
    wc = ventanas(C["fwi"], C["prec"])
    w13 = ventanas(FWI13, M["prec"])
    wpx = ventanas(FWIpx, M["prec"])

    k = np.isin(meses, EVAL)
    R = {}
    def fila(nombre, train, serve):
        t, s = train[k].ravel(), serve[k].ravel()
        o = np.isfinite(t) & np.isfinite(s)
        R[nombre] = {"psi": psi(t[o], s[o]), "media_cubo": float(t[o].mean()),
                     "media_servido": float(s[o].mean()),
                     "p95_cubo": float(np.percentile(t[o], 95)),
                     "p95_servido": float(np.percentile(s[o], 95)),
                     "corr": float(np.corrcoef(t[o], s[o])[0, 1])}
        print(f"  {nombre:34s} PSI {R[nombre]['psi']:6.3f} · media {t[o].mean():7.2f} → "
              f"{s[o].mean():7.2f} · p95 {np.percentile(t[o],95):7.2f} → "
              f"{np.percentile(s[o],95):7.2f} · corr {R[nombre]['corr']:.3f}")
    fila("t2m_max", C["tmax"], M["tmax"])
    fila("rh_min", C["hr_min"], M["hr_min"])
    fila("viento_max", C["viento_max"], M["viento_max"])
    fila("precip_dia", C["prec"], M["prec"])
    fila("precip_dia /2.02", C["prec"], M["prec"] / 2.02)
    fila("fwi (13 UTC)", C["fwi"], FWI13)
    fila("fwi (proxy tmax/hr_min)", C["fwi"], FWIpx)
    fila("fwi_pctl_local (proxy/clim proxy)", P_cubo, P_px)
    fila("fwi_pctl_local (13UTC/clim proxy)", P_cubo, P_13_climpx)
    fila("precip_30d", wc[0], w13[0])
    fila("precip_30d /2.02", wc[0], w13[0] / 2.02)
    fila("fwi_med_7d (13 UTC)", wc[1], w13[1])
    fila("fwi_med_7d (proxy)", wc[1], wpx[1])
    fila("dias_sin_lluvia", wc[2], w13[2])
    R["_nota"] = ("PSI con 10 bins de cuantiles del cubo; jul-sep 2024, 300 nodos; "
                  "junio rodaje del FWI; ventanas [D-w, D-1]")
    json.dump(R, open(config.salida("dos_06_deriva.json"), "w"), indent=1)
    print(f"\nguardado {config.salida('dos_06_deriva.json')}")


if __name__ == "__main__":
    main()
