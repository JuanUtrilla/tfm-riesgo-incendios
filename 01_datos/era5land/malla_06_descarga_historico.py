#!/usr/bin/env python3
"""
Pipeline de malla, módulo 6: ERA5-Land horario 2015-2020 para entrenar.

No toca producción. Escribe solo en el disco de expansión.

Aparcado el 21/08/2026 a los 2 minutos de arrancar, sin bajar nada.
Motivo: el cubo IberFire ya es ERA5-Land reprocesado (tmax/hr_min/viento con
corr 0,987-0,998 y sesgo 0 contra los nodos, `malla_02_vs_cubo.json`; el FWI
del cubo se reproduce con las 13 UTC, `verificar_hora_fwi.py`). Entrenar con
el cubo y servir ERA5-Land directo es la alternativa de `PROMPT_DOS_MODELOS.md`
§2, y se adopta: ahorra ~30 h de cola CDS. La única variable que difiere es
la precipitación (cubo = ERA5-Land / 2,02), que se trata al servir; ver
`ANALISIS_DATOS.md` §5. Este script queda por si algún día se reentrena con
ERA5-Land nativo: funciona y es reanudable.

Por qué existe
--------------
`PROMPT_DOS_MODELOS.md` §1 fija que toda la meteo, en entrenamiento y en
producción, sale de ERA5-Land sobre su malla nativa. Pero ERA5-Land horario
solo estaba descargado para jun-sep 2024 y may-ago 2026 (`malla_data/_cds`):
para entrenar sobre los años con EGIF fiable (2015-2020) hacen falta 72 meses
más. Este módulo los baja.

Se lanzó la noche del 21/08/2026 en paralelo con la fase 1 del encargo. Lo
que haya bajado por la mañana es lo que hay: los scripts que consumen este
producto tienen que funcionar con los meses que existan y decir cuáles faltan.

Qué cambia respecto a `malla_04_climatologia.py`
------------------------------------------------
El módulo 4 bajaba cada mes, lo agregaba y borraba el horario, porque el disco
principal iba al 97 %. Esa decisión costó cara: cuando `verificar_hora_fwi.py`
demostró que el FWI del cubo son las 13 UTC y no el proxy tmax/hr_min, rehacer
la climatología exigía bajar los 84 meses otra vez.

Aquí el horario no se borra: se guarda en el disco de expansión (1,5 TB
libres), ~60 MB/mes, ~4,3 GB los 72 meses. Instrucción explícita del 21/08:
"nada de lo que descargues de ERA5 lo elimines, muévelo a expansion".

Y el agregado diario incluye desde el principio las dos recetas:
  · extremos diarios (tmax, tmin, hr_min, viento_max, prec)   ← las features
  · instantáneas de las 13 UTC (t13, hr13, v13)               ← el FWI
así no hay que volver a abrir el horario para calcular el FWI bien.

Orden de descarga
-----------------
Años completos de enero a diciembre, porque el FWI es recursivo y necesita el
spin-up invernal; y del más reciente al más antiguo (2020 → 2015), para que si
la noche no da para los 72, los años completos que queden sean los más
cercanos a la distribución de producción. A ~25 min/mes de cola son ~30 h:
no termina en una noche. Es reanudable: se relanza y sigue por lo que falte.

Respeta la cola de CDS igual que el módulo 4 (tope de 3 en cola, uno menos
que allí porque `cron_malla.sh` también pide un mes a las 05:00; adopción de
trabajos huérfanos; retroceso exponencial ante rechazos).

Salidas (en expansion, ver `config_expansion.py`):
    era5land_cds/era5land_AAAAMM.nc      horario crudo (zip de CDS)
    era5land_cds/diario_AAAAMM.npz       agregado en los 5.605 nodos
    logs/descarga_historico.log

Uso:
    /home/charredgem/miniconda3/envs/tfm_fuego/bin/python malla_06_descarga_historico.py
    ... --anios 2020 2019            # solo esos años
    SIN_DESCARGA=1 ... --estado      # qué hay y qué falta
"""

import argparse
import gc
import os
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr

import config
import config_expansion as ce
from malla_02_descarga import AREA, VARIABLES, a_diario, abre, hr_desde_rocio

ANIOS = list(range(2020, 2014, -1))          # 2020 → 2015
VARS = ["tmax", "tmin", "hr_min", "viento_max", "prec"]
VARS13 = ["t13", "hr13", "v13"]
COLECCION = "reanalysis-era5-land"
HORA_FWI = 13

MAX_COLA = 3
SONDEO = 60
ESPERA_INI = 300
ESPERA_MAX = 3600

DEST = ce.ERA5_CDS
LOG = f"{ce.LOGS}/descarga_historico.log"


def log(msg):
    linea = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(linea, flush=True)
    with open(LOG, "a") as f:
        f.write(linea + "\n")


def peticion(anio, mes):
    return {"variable": VARIABLES, "year": str(anio), "month": f"{mes:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA, "data_format": "netcdf"}


def cliente():
    import ecmwf.datastores as ed
    cfg = dict(l.split(":", 1) for l in open(os.path.expanduser("~/.cdsapirc"))
               if ":" in l)
    return ed.Client(url=cfg["url"].strip(), key=cfg["key"].strip(),
                     progress=False, maximum_tries=10, retry_after=120)


def ruta_horario(a, m):
    return f"{DEST}/era5land_{a}{m:02d}.nc"


def ruta_diario(a, m):
    return f"{DEST}/diario_{a}{m:02d}.npz"


def a_13utc(ds):
    """Instantáneas de las 13 UTC: la receta que reproduce el FWI del cubo
    (`verificar_hora_fwi.py`: sesgo −0,9/+0,9/−0,3 en jul/ago/sep 2024)."""
    t = pd.to_datetime(ds["valid_time"].values)
    k = np.where(t.hour == HORA_FWI)[0]
    s = ds.isel(valid_time=k)
    t2 = s["t2m"] - 273.15
    hr = xr.apply_ufunc(hr_desde_rocio, s["t2m"], s["d2m"], dask="allowed")
    v = np.sqrt(s["u10"] ** 2 + s["v10"] ** 2)
    out = xr.Dataset({"t13": t2, "hr13": hr, "v13": v})
    out = out.assign_coords(valid_time=pd.to_datetime(
        s["valid_time"].values).normalize()).rename({"valid_time": "fecha"})
    return out


def procesa(nc, a, m, la, lo):
    """Horario → (diario + 13 UTC) → nodos → npz. El horario se conserva."""
    raw = abre(nc)
    ds = a_diario(raw)
    dim = [d for d in ds.dims if d not in ("latitude", "longitude")][0]
    ds = ds.rename({dim: "fecha"}).load()
    d13 = a_13utc(raw).load()
    LA = xr.DataArray(la, dims="n")
    LO = xr.DataArray(lo, dims="n")
    p = ds.sel(latitude=LA, longitude=LO, method="nearest")
    p13 = d13.sel(latitude=LA, longitude=LO, method="nearest")
    p13 = p13.reindex(fecha=ds["fecha"].values)
    np.savez_compressed(
        ruta_diario(a, m),
        fecha=ds["fecha"].values.astype("datetime64[D]").astype(int),
        **{v: p[v].values.astype(np.float32) for v in VARS},
        **{v: p13[v].values.astype(np.float32) for v in VARS13})
    raw.close(); ds.close()
    del raw, ds, d13, p, p13
    gc.collect()
    # el zip extraído (_x) se puede regenerar del .nc; se quita por espacio
    shutil.rmtree(nc + "_x", ignore_errors=True)


def adopta(c, pend):
    vivos, sobran = {}, []
    for st in ("successful", "running", "accepted"):
        try:
            jobs = c.get_jobs(limit=100, sortby="-created", status=st) \
                    .json.get("jobs", [])
        except Exception as e:
            log(f"  get_jobs({st}) falló: {type(e).__name__}")
            continue
        for j in jobs:
            if j.get("processID") != COLECCION:
                continue
            try:
                r = c.get_remote(j["jobID"]).request
                k = (int(r["year"]), int(r["month"]))
                # solo los de este script: meses completos (31 días pedidos)
                if len(r.get("day", [])) != 31:
                    continue
            except Exception:
                continue
            if k not in pend:
                continue
            if k in vivos:
                sobran.append(j["jobID"])
            else:
                vivos[k] = j["jobID"]
    for jid in sobran:
        try:
            c.get_remote(jid).delete()
            log(f"  duplicado borrado: {jid}")
        except Exception:
            pass
    return vivos


def descarga(pend, la, lo):
    c = cliente()
    trab = adopta(c, set(pend))
    if trab:
        log("  adoptados de la cola: " +
            ", ".join(f"{a}-{m:02d}" for a, m in sorted(trab)))
    tope, espera, buenos = MAX_COLA, ESPERA_INI, 0
    pend = list(pend)
    while pend:
        for k in list(trab):
            a, m = k
            try:
                r = c.get_remote(trab[k])
                st = r.status
            except Exception as e:
                log(f"  {a}-{m:02d}: sondeo falló ({type(e).__name__})")
                continue
            if st == "successful":
                nc = ruta_horario(a, m)
                log(f"  {a}-{m:02d}: descargando...")
                try:
                    r.download(nc + ".tmp")
                    os.replace(nc + ".tmp", nc)
                    procesa(nc, a, m, la, lo)
                    r.delete()
                except Exception as e:
                    log(f"  !! {a}-{m:02d}: {type(e).__name__} {e}")
                    continue
                trab.pop(k); pend.remove(k)
                buenos += 1
                log(f"  {a}-{m:02d}: LISTO · quedan {len(pend)} meses")
                if buenos % 3 == 0 and tope < MAX_COLA:
                    tope += 1
                    espera = ESPERA_INI
            elif st in ("failed", "rejected"):
                log(f"  {a}-{m:02d}: {st} · reencolar tras {espera//60} min")
                try:
                    c.get_remote(trab[k]).delete()
                except Exception:
                    pass
                trab.pop(k)
                tope = max(1, min(tope - 1, len(trab)))
                buenos = 0
                time.sleep(espera)
                espera = min(espera * 2, ESPERA_MAX)

        for k in pend:
            if len(trab) >= tope:
                break
            if k in trab:
                continue
            a, m = k
            log(f"  {a}-{m:02d}: enviando ({len(trab)+1}/{tope} en cola)...")
            try:
                trab[k] = c.submit(COLECCION, peticion(a, m)).request_id
            except Exception as e:
                log(f"  !! envío rechazado: {type(e).__name__} {e}")
                tope = max(1, len(trab))
                buenos = 0
                log(f"  tope de cola → {tope} · espera {espera//60} min")
                time.sleep(espera)
                espera = min(espera * 2, ESPERA_MAX)
                break
        if pend:
            time.sleep(SONDEO)


def estado(anios):
    hechos = [(a, m) for a in anios for m in range(1, 13)
              if os.path.exists(ruta_diario(a, m))]
    pend = [(a, m) for a in anios for m in range(1, 13)
            if not os.path.exists(ruta_diario(a, m))]
    return hechos, pend


def main(args):
    os.makedirs(DEST, exist_ok=True)
    os.makedirs(ce.LOGS, exist_ok=True)
    anios = args.anios or ANIOS
    n = np.load(config.NODOS)
    la, lo = n["nodo_lat"], n["nodo_lon"]
    hechos, pend = estado(anios)
    log(f"ERA5-Land histórico → {DEST} | años {anios} | "
        f"{len(hechos)} meses hechos · {len(pend)} pendientes")
    if args.estado:
        for a in anios:
            print(f"  {a}: " + " ".join(
                f"{m:02d}" if os.path.exists(ruta_diario(a, m)) else "--"
                for m in range(1, 13)))
        return
    # meses ya horarios pero sin agregar (corte a mitad): se agregan primero
    for a, m in list(pend):
        nc = ruta_horario(a, m)
        if os.path.exists(nc):
            log(f"  {a}-{m:02d}: horario ya en disco, agregando...")
            try:
                procesa(nc, a, m, la, lo)
                pend.remove((a, m))
            except Exception as e:
                log(f"  !! {a}-{m:02d}: {type(e).__name__} {e} · se repide")
                os.remove(nc)
    if pend and not os.environ.get("SIN_DESCARGA"):
        descarga(pend, la, lo)
    log("fin")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anios", nargs="*", type=int)
    p.add_argument("--estado", action="store_true")
    main(p.parse_args())
