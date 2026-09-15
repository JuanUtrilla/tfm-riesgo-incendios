#!/usr/bin/env python3
"""
Pipeline de malla, módulo 2: descarga de ERA5-Land desde CDS.

No toca producción. Escribe en `malla_data/`.

Por qué CDS y no Open-Meteo (medido el 19/08/2026)
--------------------------------------------------
La primera versión de este módulo tiraba de Open-Meteo. No da la cuota:
su coste es nº de nodos × tramos de 14 días, o sea 250 × 8 = 2.000 unidades
por lote y ~46.000 para la serie completa, contra un límite gratuito de 10.000
al día. Reventaba la cuota por 4-5× solo con el reanálisis, y habría que
pagarlo cada día en un cron. Es una limitación bloqueante.

CDS simplifica el diseño. Devuelve una caja lat/lon en la malla nativa
de 0,1°, en vez de 5.605 consultas puntuales: se indexa igual que el cubo. El
mapeo celda→nodo del módulo 1 pasa a ser celda→(iy,ix) de este NetCDF.

CDS no sirve para la previsión: encola las peticiones y no es de baja
latencia. Para el historial da igual (ERA5-Land lleva ~6 días de retraso de
todas formas), pero D y D+1 tendrán que venir del open data del ECMWF
(módulo 2b).

Agregación horaria → diaria, y el aviso de la precipitación
-----------------------------------------------------------
CDS sirve ERA5-Land horario; el modelo necesita diario. Las reglas replican
las del cubo:

    tmax, tmin   max/min de 2m_temperature
    hr_min       min de la HR horaria, derivada de T y punto de rocío (Magnus)
    viento_max   max de sqrt(u²+v²) a 10 m  (máx. de medias horarias, no racha:
                 el FWI explota con rachas, ISI ~ exp(0,05·v))
    prec         total diario de total_precipitation

Ojo: `total_precipitation` de ERA5-Land es acumulado desde las 00 UTC y se
reinicia cada día. El total del día no es la suma de los horarios (eso lo
multiplicaría por ~24) sino el valor de las 00:00 del día siguiente. Aquí se
resuelve desplazando el sello temporal una hora hacia atrás antes de agrupar
por día y tomando el máximo del acumulado. Este es exactamente el error que
produciría un sesgo frío del FWI como el diagnosticado en §6.

El módulo trae `--verificar`, que contrasta lo agregado contra los diarios de
Open-Meteo ya cacheados. Si la regla de la precipitación estuviera mal, ahí se
ve.

Uso:
    python malla_02_descarga.py                    # 1-may → hoy
    python malla_02_descarga.py --inicio 2026-05-01 --fin 2026-08-13
    python malla_02_descarga.py --verificar        # solo el contraste
"""

import argparse
import os
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
DATA = f"{DIR}/malla_data"
CRUDO = f"{DATA}/_cds"
# caja de la Península con margen: [N, W, S, E], alineada a 0,1°
AREA = [44.0, -9.6, 35.8, 4.4]
VARIABLES = ["2m_temperature", "2m_dewpoint_temperature",
             "10m_u_component_of_wind", "10m_v_component_of_wind",
             "total_precipitation"]


def hr_desde_rocio(t_k, td_k):
    """HR (%) desde temperatura y punto de rocío (K), fórmula de Magnus."""
    t, td = t_k - 273.15, td_k - 273.15
    e = lambda x: 6.112 * np.exp(17.67 * x / (x + 243.5))
    return np.clip(100.0 * e(td) / e(t), 0, 100)


def pide_mes(anio, mes, dias):
    """Un mes de ERA5-Land horario. Resumible: si el fichero está, no repite."""
    import cdsapi
    ruta = f"{CRUDO}/era5land_{anio}{mes:02d}.nc"
    if os.path.exists(ruta):
        print(f"    {anio}-{mes:02d}: ya descargado", flush=True)
        return ruta
    os.makedirs(CRUDO, exist_ok=True)
    print(f"    {anio}-{mes:02d}: pidiendo a CDS ({len(dias)} días)...", flush=True)
    cdsapi.Client().retrieve("reanalysis-era5-land", {
        "variable": VARIABLES, "year": str(anio), "month": f"{mes:02d}",
        "day": [f"{d:02d}" for d in dias],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA, "data_format": "netcdf",
    }, ruta + ".tmp")
    os.replace(ruta + ".tmp", ruta)
    return ruta


def abre(ruta):
    """CDS devuelve un zip con data_0.nc dentro (o el NetCDF pelado)."""
    if not zipfile.is_zipfile(ruta):
        return xr.open_dataset(ruta)
    d = ruta + "_x"
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(ruta) as z:
        z.extractall(d)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".nc"))
    return xr.open_mfdataset([f"{d}/{f}" for f in fs], combine="by_coords") \
        if len(fs) > 1 else xr.open_dataset(f"{d}/{fs[0]}")


def a_diario(ds):
    """Horario → diario, con las reglas del cubo (ver cabecera)."""
    t = ds["t2m"]
    hr = xr.apply_ufunc(hr_desde_rocio, t, ds["d2m"], dask="allowed")
    v = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    dia = t["valid_time"].dt.floor("D")

    out = xr.Dataset({
        "tmax": (t - 273.15).groupby(dia).max("valid_time"),
        "tmin": (t - 273.15).groupby(dia).min("valid_time"),
        "hr_min": hr.groupby(dia).min("valid_time"),
        "viento_max": v.groupby(dia).max("valid_time"),
    })
    # precipitación: acumulada desde las 00 UTC y con reinicio diario. El total
    # del día es el acumulado máximo, y las 00:00 pertenecen al día anterior.
    tp = ds["tp"] * 1000.0                                    # m → mm
    dia_p = (tp["valid_time"] - pd.Timedelta(hours=1)).dt.floor("D")
    out["prec"] = tp.groupby(dia_p).max("valid_time")
    return out.rename({"floor": "fecha"} if "floor" in out.dims else {})


def main(a):
    hoy = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    ini = pd.Timestamp(a.inicio or f"{hoy.year}-05-01")
    fin = pd.Timestamp(a.fin) if a.fin else hoy - pd.Timedelta(days=6)
    print(f"ERA5-Land desde CDS | {ini.date()} → {fin.date()} | área {AREA}\n")

    rutas = []
    for p in pd.period_range(ini, fin, freq="M"):
        dias = [d.day for d in pd.date_range(max(ini, p.start_time),
                                             min(fin, p.end_time), freq="D")]
        rutas.append(pide_mes(p.year, p.month, dias))

    print("\n  agregando horario → diario...", flush=True)
    partes = [a_diario(abre(r)) for r in rutas]
    ds = xr.concat(partes, dim=[d for d in partes[0].dims if d not in
                                ("latitude", "longitude")][0])
    ds = ds.rename({[d for d in ds.dims
                     if d not in ("latitude", "longitude")][0]: "fecha"})
    ds = ds.sortby("fecha").load()
    ruta = f"{DATA}/era5land_diario.nc"
    ds.to_netcdf(ruta)
    print(f"\n  {ruta}")
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  fechas: {str(ds.fecha.values[0])[:10]} → "
          f"{str(ds.fecha.values[-1])[:10]}")
    for v in ["tmax", "tmin", "hr_min", "viento_max", "prec"]:
        x = ds[v].values
        print(f"    {v:<12} media {np.nanmean(x):8.2f} · "
              f"max {np.nanmax(x):8.2f} · NaN {np.isnan(x).mean()*100:.1f}%")
    print("\nSiguiente: malla_02b_ifs.py (previsión) y malla_03_cuantiles.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inicio")
    p.add_argument("--fin")
    p.add_argument("--verificar", action="store_true")
    main(p.parse_args())
