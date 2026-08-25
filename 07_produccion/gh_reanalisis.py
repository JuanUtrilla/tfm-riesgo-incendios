#!/usr/bin/env python3
"""
Reanálisis ERA5-Land INCREMENTAL para la cadena en GitHub Actions.

`malla_02_descarga.py` re-agrega cada día todos los meses desde el 1-may a
partir de los horarios en `_cds/` (60 MB por mes). En GitHub no hay disco
persistente y volver a bajar 4 meses de CDS cada día es inviable (cola).
Aquí se baja SOLO el mes en curso (hasta D−6), se agrega a diario con la
misma `a_diario`, y se funde con el `era5land_diario.nc` acumulado que vino
del Release (`gh_estado.py pull`): se sustituyen los días de ese mes y se
conserva el resto. Al cambiar de mes, el mes anterior ya está completo en el
acumulado y no se vuelve a pedir.

El horario del mes en curso NO se conserva en GitHub (se regenera cada día;
la regla de no borrar ERA5 aplica al portátil, donde sigue `malla_02`).
Primer día del mes: CDS aún no tiene datos (D−6 cae en el mes anterior) y
el script sale sin hacer nada: el acumulado ya cubre hasta D−6.

Uso: python gh_reanalisis.py            (credenciales en ~/.cdsapirc)
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

import config
from malla_02_descarga import a_diario, abre, pide_mes

ACUM = config.salida("era5land_diario.nc")


def main():
    hoy = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    fin = hoy - pd.Timedelta(days=6)
    ini_mes = fin.replace(day=1)
    dias = list(range(1, fin.day + 1))
    acum = xr.open_dataset(ACUM).load() if os.path.exists(ACUM) else None
    acum_fin = pd.to_datetime(acum["fecha"].values).max() if acum is not None else None
    print(f"acumulado hasta {acum_fin.date() if acum_fin is not None else '—'} · "
          f"CDS pide {ini_mes:%Y-%m} días 1..{fin.day}", flush=True)
    if acum_fin is not None and acum_fin >= fin:
        print("  nada que hacer: el acumulado ya llega a D−6"); return
    ruta = pide_mes(ini_mes.year, ini_mes.month, dias)
    nuevo = a_diario(abre(ruta))
    dim = [d for d in nuevo.dims if d not in ("latitude", "longitude")][0]
    nuevo = nuevo.rename({dim: "fecha"}).sortby("fecha").load()
    if acum is None:
        out = nuevo
    else:
        fuera = ~pd.to_datetime(acum["fecha"].values).to_period("M").isin(
            [ini_mes.to_period("M")])
        out = xr.concat([acum.isel(fecha=np.where(fuera)[0]), nuevo], dim="fecha") \
                .sortby("fecha")
    out.to_netcdf(ACUM + ".tmp")
    os.replace(ACUM + ".tmp", ACUM)
    f = pd.to_datetime(out["fecha"].values)
    print(f"  era5land_diario.nc: {f.min().date()} → {f.max().date()} ({len(f)} días)")
    # el horario del mes en curso no hace falta más
    import shutil
    shutil.rmtree(ruta + "_x", ignore_errors=True)
    for ext in ("", ".cobertura"):
        if os.path.exists(ruta + ext):
            os.remove(ruta + ext)


if __name__ == "__main__":
    main()
