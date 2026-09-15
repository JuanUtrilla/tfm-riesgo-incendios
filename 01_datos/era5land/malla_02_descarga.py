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
pagarlo cada día en un cron. Es bloqueante.

CDS también simplifica el diseño. Devuelve una caja lat/lon en la malla nativa
de 0,1°, no 5.605 consultas puntuales: se indexa igual que el cubo. El mapeo
celda→nodo del módulo 1 pasa a ser celda→(iy,ix) de este NetCDF.

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
por día y tomando el máximo del acumulado. Sumar los horarios es justo el
error que produciría un sesgo frío del FWI como el diagnosticado en §6.

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

import config

DIR = config.FUENTE
DATA = config.SALIDA
CRUDO = config.CRUDO_PROPIO       # las descargas nuevas caen en este repo
CRUDO_LECTURA = config.CRUDO      # los meses ya bajados en el original
AREA = config.AREA
VARIABLES = config.VARIABLES


def hr_desde_rocio(t_k, td_k):
    """HR (%) desde temperatura y punto de rocío (K), fórmula de Magnus."""
    t, td = t_k - 273.15, td_k - 273.15
    e = lambda x: 6.112 * np.exp(17.67 * x / (x + 243.5))
    return np.clip(100.0 * e(td) / e(t), 0, 100)


def cobertura(ruta):
    """Días que de verdad tiene un fichero de CDS, con sidecar para no abrirlo
    cada vez. Devuelve el último día del mes que contiene, o 0."""
    side = ruta + ".cobertura"
    if os.path.exists(side):
        return int(open(side).read().strip())
    try:
        d = abre(ruta)
        t = d["valid_time"].values if "valid_time" in d.coords else \
            d[list(d.coords)[0]].values
        ult = int(pd.to_datetime(t).max().day)
        d.close()
    except Exception:
        return 0
    try:
        open(side, "w").write(str(ult))
    except OSError:
        pass                                   # el original es de solo lectura
    return ult


def pide_mes(anio, mes, dias):
    """Un mes de ERA5-Land horario, resumible.

    Ojo: el mes en curso crece. La versión anterior salía si el fichero existía,
    y como el mes en curso se guarda con el mismo nombre, el reanálisis se
    quedaba congelado en el día en que se bajó por primera vez, sin error y
    sin aviso. Con el cron, el mapa habría ido envejeciendo en silencio.
    (El fichero de agosto del repo original se llama `era5land_202608.nc` y
    solo tiene hasta el día 13.)

    Ahora se comprueba la cobertura real, no la existencia del nombre, y un
    mes incompleto se guarda con sufijo del último día para no pisar nada.
    """
    import cdsapi
    ultimo = max(dias)
    plano = f"era5land_{anio}{mes:02d}.nc"
    # se reutiliza lo del repo original, pero solo si de verdad llega
    for base in (CRUDO, CRUDO_LECTURA):
        for nombre in (plano, f"era5land_{anio}{mes:02d}_h{ultimo:02d}.nc"):
            r = f"{base}/{nombre}"
            if os.path.exists(r) and cobertura(r) >= ultimo:
                print(f"    {anio}-{mes:02d}: ya descargado (hasta el "
                      f"{cobertura(r)})", flush=True)
                return r
    import calendar
    completo = ultimo == calendar.monthrange(anio, mes)[1]
    ruta = f"{CRUDO}/{plano if completo else f'era5land_{anio}{mes:02d}_h{ultimo:02d}.nc'}"
    os.makedirs(CRUDO, exist_ok=True)
    print(f"    {anio}-{mes:02d}: pidiendo a CDS ({len(dias)} días)...", flush=True)
    cdsapi.Client().retrieve("reanalysis-era5-land", {
        "variable": VARIABLES, "year": str(anio), "month": f"{mes:02d}",
        "day": [f"{d:02d}" for d in dias],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA, "data_format": "netcdf",
    }, ruta + ".tmp")
    os.replace(ruta + ".tmp", ruta)
    open(ruta + ".cobertura", "w").write(str(ultimo))
    retira_parciales(anio, mes, ultimo)
    return ruta


def retira_parciales(anio, mes, ultimo):
    """Los parciales anteriores del mismo mes se mueven al disco externo.

    21/08/2026: cada día el mes en curso se guardaba con un nombre nuevo
    (`_h14`, `_h15`, ...) sin retirar el anterior: ~60 MB de zip + ~60 MB de
    `_x` al día, con 5,7 GB libres en el disco principal. Se llenaba en
    septiembre. No se borra nada (regla del 21/08: lo de ERA5 se mueve, no se
    elimina): van a <externo>/era5land_cds/superseded/. Si el externo no está
    montado se dejan donde están y se avisa."""
    import glob
    import shutil
    viejos = [r for r in glob.glob(f"{CRUDO}/era5land_{anio}{mes:02d}_h??.nc")
              if int(r[-5:-3]) < ultimo]
    if not viejos:
        return
    try:
        import config_expansion as ce
        dest = f"{ce.ERA5_CDS}/superseded"
        os.makedirs(dest, exist_ok=True)
    except SystemExit:
        print(f"    aviso: disco externo no montado; {len(viejos)} parciales "
              f"antiguos se quedan en {CRUDO}", flush=True)
        return
    for r in viejos:
        for extra in ("", ".cobertura"):
            if os.path.exists(r + extra):
                shutil.move(r + extra, f"{dest}/{os.path.basename(r)}{extra}")
        shutil.rmtree(r + "_x", ignore_errors=True)   # extraído del zip: se regenera
        print(f"    parcial antiguo → externo: {os.path.basename(r)}", flush=True)


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
    ruta = config.salida("era5land_diario.nc")
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
