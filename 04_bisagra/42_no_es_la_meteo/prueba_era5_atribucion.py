#!/usr/bin/env python3
"""
¿De qué variable viene el sesgo frío de ERA5? (−2,0 de FWI, ver §6)

No sobrescribe nada. Escribe dataset/prueba_era5_atribucion.json.

La sospecha: `era5_seamless` toma T y HR de ERA5-Land (9 km) pero viento y
precipitación de ERA5 (31 km), más suavizados. El FWI es muy sensible al
viento vía ISI = 0,208·exp(0,05039·v)·f(FFMC), que es exponencial en el
viento: suavizar el viento hunde el FWI de forma no lineal.

Comprobarlo yendo a CDS costaría días. Aquí se hace en una tarde y sin
descargar nada nuevo: se corre el FWI con entradas híbridas, sustituyendo de
una en una la variable de ERA5 por la del cubo. La variable que al sustituirla
cierra el sesgo es la culpable.

    ERA5 puro                        → sesgo de referencia (−2,0)
    ERA5 pero con el viento del cubo → si el sesgo se va, es el viento
    ERA5 pero con la precip del cubo → ídem
    ERA5 pero con T del cubo         → control (deberían coincidir: ambas
    ERA5 pero con HR del cubo        →          salen de ERA5-Land)

Los dos controles importan: si T y HR no coinciden entre cubo y Open-Meteo,
el problema no está en el viento sino en el regrillado del cubo a 1 km, y la
conclusión cambia por completo.

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python prueba_era5_atribucion.py
"""

import json
import os

import numpy as np
import pandas as pd
import xarray as xr

from fwi_canadiense import calcular_fwi_serie
from prueba_era5_produccion import (CACHE, DIR, EVAL, FIN, SPIN, estaciones,
                                    serie_openmeteo)

N = 120
RUTA_CUBO = f"{CACHE}/cubo_meteo_{N}.npz"


def cubo_cacheado(est, fechas):
    """La lectura del cubo tarda ~15 min: se cachea en disco."""
    if os.path.exists(RUTA_CUBO):
        print("  cubo: desde caché", flush=True)
        return dict(np.load(RUTA_CUBO))
    ds = xr.open_dataset(f"{DIR}/iberfire/IberFire.nc", decode_timedelta=False)
    t = ds["time"].values.astype("datetime64[D]")
    idx = np.where((t >= np.datetime64(SPIN)) & (t <= np.datetime64(FIN)))[0]
    iy = xr.DataArray(est["iy"].values, dims="p")
    ix = xr.DataArray(est["ix"].values, dims="p")
    out = {}
    for var in ["t2m_max", "RH_min", "wind_speed_max",
                "total_precipitation_mean"]:
        out[var] = ds[var].isel(time=idx, y=iy, x=ix).values
        print(f"    cubo: {var}", flush=True)
    ds.close()
    np.savez_compressed(RUTA_CUBO, **out)
    return out


def main():
    est = estaciones(N)
    fechas = pd.date_range(SPIN, FIN, freq="D")
    m_ev = (fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1]) \
        & (fechas.year == 2024)
    print(f"estaciones {len(est)} | evaluación {m_ev.sum()} días jun-sep 2024\n")

    cubo = cubo_cacheado(est, fechas)
    om = serie_openmeteo(est)

    # matrices (tiempo, estación) alineadas para las dos fuentes
    C = {"tmax": cubo["t2m_max"], "hr_min": cubo["RH_min"],
         "viento_max": cubo["wind_speed_max"], "prec": cubo["total_precipitation_mean"]}
    E = {}
    for c in ["tmax", "hr_min", "viento_max", "prec"]:
        E[c] = np.column_stack([
            om[om.idema == i].set_index("fecha").reindex(fechas)[c].values
            for i in est["idema"]])

    res = {"n_estaciones": int(len(est)), "periodo": "jun-sep 2024",
           "variables": {}, "atribucion": {}}

    # ---- 1. comparación variable a variable --------------------------------
    print("\n" + "=" * 70)
    print("VARIABLE A VARIABLE (ERA5/Open-Meteo vs cubo, jun-sep 2024)")
    print("=" * 70)
    print(f"{'variable':<14}{'media cubo':>12}{'media ERA5':>12}"
          f"{'sesgo':>9}{'corr':>8}")
    for c, uni in [("tmax", "°C"), ("hr_min", "%"),
                   ("viento_max", "m/s"), ("prec", "mm")]:
        a, b = C[c][m_ev].ravel(), E[c][m_ev].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        cor = np.corrcoef(a[ok], b[ok])[0, 1]
        print(f"{c+' ('+uni+')':<14}{a[ok].mean():>12.2f}{b[ok].mean():>12.2f}"
              f"{b[ok].mean()-a[ok].mean():>+9.2f}{cor:>8.3f}")
        res["variables"][c] = dict(media_cubo=float(a[ok].mean()),
                                   media_era5=float(b[ok].mean()),
                                   sesgo=float(b[ok].mean() - a[ok].mean()),
                                   corr=float(cor))

    # ---- 2. atribución por sustitución -------------------------------------
    def fwi_matriz(fuente):
        out = np.full((len(fechas), len(est)), np.nan)
        for j in range(len(est)):
            out[:, j] = calcular_fwi_serie(
                fuente["tmax"][:, j], fuente["hr_min"][:, j],
                fuente["viento_max"][:, j] * 3.6, fuente["prec"][:, j],
                fechas.month.values)["fwi"]
        return out

    print("\n" + "=" * 70)
    print("ATRIBUCIÓN: sesgo del FWI al sustituir una variable por la del cubo")
    print("=" * 70)
    fwi_c = fwi_matriz(C)
    escenarios = {"ERA5 puro": E}
    for c in ["viento_max", "prec", "tmax", "hr_min"]:
        v = dict(E); v[c] = C[c]
        escenarios[f"ERA5 + {c} del cubo"] = v

    print(f"{'escenario':<28}{'sesgo FWI':>12}{'corr':>8}{'% recuperado':>14}")
    base = None
    for nom, fuente in escenarios.items():
        f = fwi_matriz(fuente)
        a, b = fwi_c[m_ev].ravel(), f[m_ev].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        ses = b[ok].mean() - a[ok].mean()
        cor = np.corrcoef(a[ok], b[ok])[0, 1]
        if base is None:
            base = ses
            rec = ""
        else:
            rec = f"{(1 - abs(ses) / abs(base)) * 100:>13.0f}%"
        print(f"{nom:<28}{ses:>+12.2f}{cor:>8.3f}{rec:>14}")
        res["atribucion"][nom] = dict(sesgo=float(ses), corr=float(cor))

    with open(f"{DIR}/dataset/prueba_era5_atribucion.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("\nGuardado: dataset/prueba_era5_atribucion.json")


if __name__ == "__main__":
    main()
