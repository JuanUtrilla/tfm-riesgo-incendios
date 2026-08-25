#!/usr/bin/env python3
"""
Módulo 3c: prueba de aceptación del mapeo del FWI. Dos comprobaciones.

NO SOBRESCRIBE NADA. Escribe dataset/malla_03c_aceptacion.json.

(1) FUERA DE MUESTRA. En 3b el mapeo se ajustó y se evaluó sobre jun-sep 2024:
    para el sesgo eso es CIRCULAR (un mapeo de cuantiles anula el sesgo sobre
    su propio periodo de ajuste por construcción). Aquí se ajusta con jun-jul
    y se evalúa con ago-sep, que es lo que de verdad haría producción.

(2) LA MÉTRICA QUE DECIDE. Saturación de `fwi_pctl_local` contra `clim_fwi`,
    el denominador real de producción. Es la métrica con la que empezó todo:

        referencia (cubo)          0,2 %
        ERA5 vía Open-Meteo        0,8 %
        IFS híbrido crudo          2,2 %
        AEMET (producción hoy)     3,3 %

    Con el máximo del FWI clavado en 107,2 (= el del cubo), debería aterrizar
    en el nivel de la referencia.

Se mide además la variante de diseño que hace innecesario el mapeo para las
dos features del percentil: usar como denominador la climatología construida
con ERA5-Land en vez de la del cubo. Aquí se aproxima con los propios datos
2024 (muestra corta, es indicativo, no definitivo).

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python malla_03c_aceptacion.py
"""

import json

import numpy as np
import pandas as pd
import xarray as xr

from fwi_canadiense import calcular_fwi_serie
from malla_02_descarga import DATA, DIR, a_diario, abre, pide_mes
from prueba_era5_produccion import EVAL, FIN, SPIN, estaciones
from prueba_era5_atribucion import N, cubo_cacheado

Q = np.linspace(0, 1, 501)


def main():
    est = estaciones(N)
    fechas = pd.date_range(SPIN, FIN, freq="D")
    meses = fechas.month.values
    ev = np.asarray((fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1])
                    & (fechas.year == 2024))
    fit = ev & np.asarray(fechas.month.isin([6, 7]))       # ajuste
    tst = ev & np.asarray(fechas.month.isin([8, 9]))       # evaluación

    rutas = [pide_mes(2024, m, list(range(1, 32))) for m in (6, 7, 8, 9)]
    ps = [a_diario(abre(r)) for r in rutas]
    dim = [d for d in ps[0].dims if d not in ("latitude", "longitude")][0]
    ds = xr.concat(ps, dim=dim).rename({dim: "fecha"}).sortby("fecha").load()

    cubo = cubo_cacheado(est, fechas)
    fmap = {pd.Timestamp(t).normalize(): i for i, t in enumerate(ds.fecha.values)}
    idx = [fmap.get(f) for f in fechas]
    L = {}
    for v in ["tmax", "hr_min", "viento_max", "prec"]:
        M = np.full((len(fechas), len(est)), np.nan)
        for j, (_, e) in enumerate(est.iterrows()):
            col = ds[v].sel(latitude=e.lat, longitude=e.lon,
                            method="nearest").values
            M[:, j] = [np.nan if i is None else col[i] for i in idx]
        L[v] = M
    C = {"tmax": cubo["t2m_max"], "hr_min": cubo["RH_min"],
         "viento_max": cubo["wind_speed_max"],
         "prec": cubo["total_precipitation_mean"]}

    def fwi_m(s, p):
        M = np.full((len(fechas), len(est)), np.nan)
        for j in range(len(est)):
            M[:, j] = calcular_fwi_serie(s["tmax"][:, j], s["hr_min"][:, j],
                                         s["viento_max"][:, j] * 3.6,
                                         p[:, j], meses)["fwi"]
        return M

    ref, mio = fwi_m(C, C["prec"]), fwi_m(L, L["prec"])

    # mapeo ajustado SOLO con junio-julio
    a, b = ref[fit].ravel(), mio[fit].ravel()
    o = np.isfinite(a) & np.isfinite(b)
    xs, ys = np.quantile(b[o], Q), np.quantile(a[o], Q)
    cor = np.where(np.isfinite(mio), np.interp(mio, xs, ys), mio)

    print("=" * 74)
    print("(1) FUERA DE MUESTRA — ajuste jun-jul 2024, evaluación ago-sep 2024")
    print("=" * 74)
    print(f"{'configuración':<34}{'sesgo':>9}{'corr':>8}{'mediana':>10}{'máx':>9}")
    print(f"{'CUBO (referencia)':<34}{0.0:>+9.2f}{1.0:>8.3f}"
          f"{np.nanmedian(ref[tst]):>10.1f}{np.nanmax(ref[tst]):>9.1f}")
    res = {}
    for nom, M in [("ERA5-Land crudo", mio), ("ERA5-Land + FWI mapeado", cor)]:
        u, w = ref[tst].ravel(), M[tst].ravel()
        k = np.isfinite(u) & np.isfinite(w)
        s, r = w[k].mean() - u[k].mean(), np.corrcoef(u[k], w[k])[0, 1]
        print(f"{nom:<34}{s:>+9.2f}{r:>8.3f}"
              f"{np.nanmedian(M[tst]):>10.1f}{np.nanmax(M[tst]):>9.1f}")
        res[nom] = dict(sesgo_fuera_muestra=float(s), corr=float(r))

    # ---- (2) saturación del percentil --------------------------------------
    clim = {i: np.load(f"{DIR}/prototipo/clim_fwi/{i}.npz") for i in est["idema"]}

    def pctl(M, mask, denom=None):
        p = []
        for j, i in enumerate(est["idema"]):
            for k in np.where(mask)[0]:
                v = M[k, j]
                if not np.isfinite(v):
                    continue
                c = (clim[i][f"m{meses[k]}"] if denom is None
                     else denom[np.asarray(meses == meses[k]) & ev, j])
                c = np.sort(c[np.isfinite(c)])
                if len(c):
                    p.append(np.searchsorted(c, v, side="right") / len(c) * 100)
        return np.array(p)

    print("\n" + "=" * 74)
    print("(2) SATURACIÓN DE fwi_pctl_local contra clim_fwi (ago-sep 2024)")
    print("=" * 74)
    print(f"{'numerador':<34}{'pctl med':>11}{'% >=95':>10}{'% >=99,9':>11}")
    for nom, M in [("CUBO (referencia)", ref), ("ERA5-Land crudo", mio),
                   ("ERA5-Land + FWI mapeado", cor)]:
        p = pctl(M, tst)
        print(f"{nom:<34}{np.median(p):>11.1f}{(p>=95).mean()*100:>10.1f}"
              f"{(p>=99.9).mean()*100:>11.1f}")
        res.setdefault(nom, {}).update(
            pctl_mediana=float(np.median(p)),
            pctl_ge95=float((p >= 95).mean() * 100),
            pctl_ge999=float((p >= 99.9).mean() * 100))
    print(f"{'AEMET (producción hoy)':<34}{36.9:>11.1f}{10.0:>10.1f}{3.3:>11.1f}")
    print(f"{'ERA5 vía Open-Meteo (§6)':<34}{38.6:>11.1f}{6.3:>10.1f}{0.8:>11.1f}")

    ok = (abs(res["ERA5-Land + FWI mapeado"]["sesgo_fuera_muestra"]) < 1.5
          and res["ERA5-Land + FWI mapeado"]["pctl_ge999"] < 1.0)
    print("\n" + ("ACEPTADO" if ok else "NO PASA") +
          " — criterio: |sesgo fuera de muestra| < 1,5 y saturación < 1,0 %")
    res["acepta"] = bool(ok)
    with open(f"{DIR}/dataset/malla_03c_aceptacion.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
