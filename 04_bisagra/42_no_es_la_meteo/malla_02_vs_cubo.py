#!/usr/bin/env python3
"""
¿ERA5-Land puro cierra la diferencia de precipitación con el cubo?

No sobrescribe nada. Escribe dataset/malla_02_vs_cubo.json.

Por qué: §6 atribuyó el sesgo frío de −2,0 del FWI a que `era5_seamless` toma
la precipitación de ERA5 (31 km) y llueve el doble que el cubo (1,38 vs 0,69
mm/día, jun-sep 2024). La conclusión fue que ir a ERA5-Land puro desde CDS lo
arreglaría.

La verificación del módulo 2 pone eso en duda: sobre may-ago 2026, ERA5-Land
(CDS) y ERA5 (Open-Meteo) dan la misma precipitación (0,88 vs 0,81 en jun-ago,
r=0,969). Si es así, el factor 2 está entre el cubo y los dos productos de
ERA5, no entre estos dos, y cambiar de producto no arreglaría nada.

Aquel contraste mezclaba años (CDS 2026 contra cubo 2024). Este script lo
resuelve sin ambigüedad: baja ERA5-Land de CDS para el mismo periodo de §6
(jun-sep 2024) y lo compara con el cubo en las mismas 120 estaciones, ya
cacheadas.

Desenlaces:
  · ERA5-Land ≈ 0,69 (el cubo)  → §6 acierta, el arreglo por fuente funciona.
  · ERA5-Land ≈ 1,38 (ERA5)     → §6 se equivoca: el cubo procesa la
    precipitación de otra forma y no hay arreglo por fuente. El camino sería
    entonces el mapeo de cuantiles del módulo 3, que es empírico y ya se midió
    que funciona (saturación 2,2 % → 0,5 %).

Uso: python malla_02_vs_cubo.py
"""

import json

import numpy as np
import pandas as pd
import xarray as xr

from malla_02_descarga import DATA, DIR, a_diario, abre, pide_mes
from prueba_era5_produccion import EVAL, FIN, SPIN, estaciones
from prueba_era5_atribucion import N, cubo_cacheado


def main():
    est = estaciones(N)
    fechas = pd.date_range(SPIN, FIN, freq="D")
    m_ev = np.asarray((fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1])
                      & (fechas.year == 2024))
    print(f"{len(est)} estaciones · {m_ev.sum()} días jun-sep 2024\n")

    rutas = [pide_mes(2024, m, list(range(1, 32))) for m in (6, 7, 8, 9)]
    print("\n  agregando...", flush=True)
    ps = [a_diario(abre(r)) for r in rutas]
    dim = [d for d in ps[0].dims if d not in ("latitude", "longitude")][0]
    ds = xr.concat(ps, dim=dim).rename({dim: "fecha"}).sortby("fecha").load()

    cubo = cubo_cacheado(est, fechas)
    C = {"prec": cubo["total_precipitation_mean"],
         "viento_max": cubo["wind_speed_max"], "tmax": cubo["t2m_max"],
         "hr_min": cubo["RH_min"]}

    L = {v: np.full((len(fechas), len(est)), np.nan) for v in C}
    fmap = {pd.Timestamp(t).normalize(): i
            for i, t in enumerate(ds.fecha.values)}
    idx = [fmap.get(f) for f in fechas]
    for j, (_, e) in enumerate(est.iterrows()):
        p = ds.sel(latitude=e.lat, longitude=e.lon, method="nearest")
        for v in C:
            col = p[v].values
            L[v][:, j] = [np.nan if i is None else col[i] for i in idx]

    print("\n" + "=" * 72)
    print("ERA5-Land puro (CDS) vs CUBO — jun-sep 2024, mismas estaciones")
    print("=" * 72)
    print(f"{'variable':<14}{'cubo':>9}{'ERA5-Land':>12}{'ERA5(§6)':>11}"
          f"{'sesgo':>9}{'corr':>8}")
    ref6 = {"tmax": 27.46, "hr_min": 37.23, "viento_max": 4.12, "prec": 1.38}
    res = {}
    for v in ["tmax", "hr_min", "viento_max", "prec"]:
        a, b = C[v][m_ev].ravel(), L[v][m_ev].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[ok], b[ok])[0, 1]
        print(f"{v:<14}{a[ok].mean():>9.2f}{b[ok].mean():>12.2f}"
              f"{ref6[v]:>11.2f}{b[ok].mean()-a[ok].mean():>+9.2f}{r:>8.3f}")
        res[v] = dict(cubo=float(a[ok].mean()), era5land=float(b[ok].mean()),
                      era5_seamless_ref=ref6[v],
                      sesgo=float(b[ok].mean() - a[ok].mean()), corr=float(r))

    pc, pl, pe = res["prec"]["cubo"], res["prec"]["era5land"], ref6["prec"]
    print("\n" + "-" * 72)
    if abs(pl - pc) < 0.5 * abs(pe - pc):
        print(f"VEREDICTO: ERA5-Land ({pl:.2f}) se acerca al cubo ({pc:.2f}).")
        print("§6 ACIERTA — el arreglo por fuente funciona para la precipitación.")
    else:
        print(f"VEREDICTO: ERA5-Land ({pl:.2f}) sigue lejos del cubo ({pc:.2f}) "
              f"y cerca de ERA5 ({pe:.2f}).")
        print("§6 SE EQUIVOCA en la precipitación: el cubo la procesa de otra")
        print("forma y no hay arreglo por fuente. El camino es el mapeo de")
        print("cuantiles del módulo 3, que es empírico y ya se midió que funciona.")
    print("-" * 72)

    with open(f"{DIR}/dataset/malla_02_vs_cubo.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("\nGuardado: dataset/malla_02_vs_cubo.json")


if __name__ == "__main__":
    main()
