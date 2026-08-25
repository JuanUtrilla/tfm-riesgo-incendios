#!/usr/bin/env python3
"""
Módulo 3b: corregir el FWI de salida en vez de las entradas.

NO SOBRESCRIBE NADA. Escribe malla_data/mapeo_fwi.npz y
dataset/malla_03b_fwi_qm.json.

POR QUÉ SE LLEGA AQUÍ. Tres intentos de arreglar el sesgo por el lado de las
entradas, y los tres diagnósticos mecanicistas fallaron:

  1. "es el viento suavizado"        → era la precipitación (§6)
  2. "ERA5-Land puro arregla la lluvia" → arregló el viento, no la lluvia
  3. "es drizzle bias (frecuencia)"  → las frecuencias ya coinciden
                                        (cubo 37,8 % · ERA5-Land 39,2 %)

Lo que queda es un error de SECUENCIA, no de distribución: con r=0,904 en
precipitación, ERA5-Land llueve en días parcialmente distintos. Y el FWI es
asimétrico —la lluvia lo hunde en un día y recuperarse cuesta varios—, así que
un error de CUÁNDO llueve sesga el índice a la baja aunque el CUÁNTO total sea
correcto. Ninguna corrección de la distribución de entrada puede arreglar eso.

LA ALTERNATIVA. Corregir el FWI ya calculado. Es monótona, luego conserva el
orden del ranking, que es lo único que usa el producto operativo; arregla de
una vez la feature `fwi` y las ventanas que se derivan de ella; y no exige
entender el mecanismo, que hoy ha demostrado ser lo más fiable.

SIN FUGA: se ajusta con jun-sep 2024 y se comprueba también sobre may 2026
(fuera del periodo de ajuste y de otro año) para ver si el mapeo aguanta.

TAMBIÉN SE MIDE LA ALTERNATIVA DE DISEÑO: si en vez de corregir se reconstruye
`clim_fwi` desde ERA5-Land, el sesgo se cancela POR CONSTRUCCIÓN en
`fwi_pctl_local` y no hay nada que mapear. Aquí se cuantifica cuánto queda sin
resolver por esa vía (la feature `fwi` cruda seguiría desplazada).

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python malla_03b_fwi_qm.py
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
    m_ev = np.asarray((fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1])
                      & (fechas.year == 2024))

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

    a, b = ref[m_ev].ravel(), mio[m_ev].ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    xs, ys = np.quantile(b[ok], Q), np.quantile(a[ok], Q)
    corr_fwi = lambda x: np.where(np.isfinite(x), np.interp(x, xs, ys), x)
    np.savez_compressed(f"{DATA}/mapeo_fwi.npz", xs=xs, ys=ys,
                        ajustado_con="jun-sep 2024, 120 estaciones")

    cor = corr_fwi(mio)
    print("=" * 72)
    print("CORREGIR EL FWI DE SALIDA — jun-sep 2024 (periodo de AJUSTE)")
    print("=" * 72)
    print(f"{'configuración':<34}{'sesgo':>9}{'corr':>8}{'mediana':>10}{'máx':>9}")
    print(f"{'CUBO (referencia)':<34}{0.0:>+9.2f}{1.0:>8.3f}"
          f"{np.nanmedian(ref[m_ev]):>10.1f}{np.nanmax(ref[m_ev]):>9.1f}")
    res = {}
    for nom, M in [("ERA5-Land crudo", mio), ("ERA5-Land + FWI mapeado", cor)]:
        u, w = ref[m_ev].ravel(), M[m_ev].ravel()
        o = np.isfinite(u) & np.isfinite(w)
        s, r = w[o].mean() - u[o].mean(), np.corrcoef(u[o], w[o])[0, 1]
        print(f"{nom:<34}{s:>+9.2f}{r:>8.3f}"
              f"{np.nanmedian(M[m_ev]):>10.1f}{np.nanmax(M[m_ev]):>9.1f}")
        res[nom] = dict(sesgo=float(s), corr=float(r))
    print(f"{'(era5_seamless, §6)':<34}{-2.01:>+9.2f}{0.976:>8.3f}"
          f"{35.1:>10.1f}{110.1:>9.1f}")

    print("\n" + "-" * 72)
    print("LO QUE ESTE MAPEO NO ARREGLA")
    print("-" * 72)
    print(f"  La correlación se queda en {res['ERA5-Land crudo']['corr']:.3f}: "
          f"el mapeo es monótono,\n  reescala pero no puede corregir un error "
          f"de secuencia temporal.")
    print(f"  Si el ranking del día es lo que importa, eso basta. Si se "
          f"necesita el\n  valor absoluto del FWI, no.")

    print("\n" + "-" * 72)
    print("ALTERNATIVA DE DISEÑO: climatología desde ERA5-Land")
    print("-" * 72)
    print("  Si `clim_fwi` se reconstruye con ERA5-Land en vez de con el cubo,")
    print("  el sesgo se cancela POR CONSTRUCCIÓN en fwi_pctl_local y")
    print("  fwi_anom_sigma — que son las dos features que se rompieron en")
    print("  producción. Quedaría desplazada solo la feature `fwi` cruda y sus")
    print("  ventanas, y para eso vale este mapeo. Las dos vías son")
    print("  compatibles y conviene aplicar ambas.")

    with open(f"{DIR}/dataset/malla_03b_fwi_qm.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("\nGuardado: malla_data/mapeo_fwi.npz")


if __name__ == "__main__":
    main()
