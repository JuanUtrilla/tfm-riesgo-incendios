#!/usr/bin/env python3
"""
Pipeline de malla, módulo 4: climatología del FWI por nodo, desde ERA5-Land.

No toca producción. Escribe malla_data/clim_fwi_nodos.npz.

Por qué este módulo es el que importa
-------------------------------------
`fwi_pctl_local` es una división y todo el problema de producción fue que sus
dos términos venían de fuentes distintas (numerador AEMET, denominador cubo).
El invariante que hay que restaurar es que numerador y denominador salgan de
la misma fuente, sea cual sea.

Medido hoy (`malla_03c_aceptacion.py`, ago-sep 2024 fuera de muestra):

    saturación de fwi_pctl_local (% en pctl ≥99,9)
        cubo (referencia)         0,3 %
        ERA5-Land crudo           0,3 %   ← ya resuelto
        AEMET (producción hoy)    3,3 %

    pero ERA5-Land corre frío: sesgo −3,83 y percentil mediano 31,0
    frente a 42,9 de la referencia → infraalertaría.

Ese sesgo no se arregla con mapeos: se probaron dos y fuera de muestra
empeoraban (el del FWI daba +5,78 de sesgo y subía la saturación a 1,4 %).
Se arregla construyendo el denominador con ERA5-Land, y entonces el
desplazamiento se cancela por construcción.

Comprobado con los 58 meses ya en disco (2008-01→2012-10): frente a la
referencia del cubo, la mediana va −3 a −9 (el sesgo frío esperado) pero el
p95 coincide con la referencia: +0,1 / −0,2 / −0,2 en jul/ago/sep. La cola
alta, que es donde viven las alertas, ya cuadra. Lo que aún no ha convergido es
el tamaño muestral: de 4 a 5 años el percentil se mueve 2,4-4,7 puntos de
media, y el suelo de saturación por discretización es 100/(n+1): 0,66 % con 5
años frente a 0,46 % con los 7. De ahí que haya que terminar los 84 meses.

Diseño
------
· 2008-2014, los mismos años que la climatología del cubo (`ANIOS_CLIM` en
  `extraer_features_cubo.py`), elegidos en su día por ser previos al dataset
  2015-2020 y no contaminar el entrenamiento.
· Años completos, no solo temporada: el FWI es recursivo y necesita el
  spin-up desde enero para que converjan DMC y DC.
· Climatología por nodo y por mes, igual que `clim_fwi/<idema>.npz`, pero en
  los 5.605 nodos en vez de en 705 estaciones.
  Ojo: 226 nodos (4,0 %) son costeros y su vecino más próximo cae en mar:
  ERA5-Land no los cubre y quedarán a NaN. No es cobertura del 100 %;
  `malla_05` necesitará un vecino terrestre de respaldo para ellos.

Disco. Quedan ~6,4 GB libres. El horario en crudo de 84 meses son ~5 GB, no
cabe con margen. Por eso cada mes se descarga, se agrega a diario, se extrae en
los nodos y se borra el horario acto seguido: el pico es de un mes (~60 MB).

Resumible: cada mes deja `_clim/diario_AAAAMM.npz`. Si se corta, se relanza el
mismo comando y continúa por el primer mes que falte.

La cola de CDS: por qué la versión anterior se atascó en 58/84
--------------------------------------------------------------
El 20/08 a las 09:35 el script relanzó los 26 meses pendientes y CDS los
rechazó todos en cadena:

    HTTP 400 · "The job has been rejected. Number queued requests for this
    dataset is temporarily limited. Please configure your scripts accordingly"

Consultando la cola con la cuenta (`get_jobs`) se ve la causa exacta: había
5 trabajos propios en estado `accepted` (2012-11 ×2, 2012-12, 2013-01,
2013-06) encolados desde hacía dos horas. Con esos 5 ocupando sitio, CDS
rechaza toda petición nueva sobre el mismo dataset. El `ThreadPoolExecutor`
no lo veía: lanzaba, cobraba el rechazo como excepción, marcaba el mes como
"hecho" y seguía; y el vigilante lo relanzaba cada 5 min, martilleando.

Qué dice ECMWF (no publican el número, es dinámico):
  · "Limits are set on usage of CDS resources... changed from time to time
    according to the current workload" (Climate Data Store documentation).
  · Ante este mismo error, soporte responde: "send the requests sequentially,
    wait for the first one to finish before send the second one"
    (forum.ecmwf.int/t/api-queued-requests/12203).
  · Best practices de los data stores: tope de peticiones en paralelo, y
    pasarse penaliza la prioridad en la cola, no solo rechaza.
  · "Submit small requests over very large and heavy requests to ensure your
    requests are not penalised in the CDS request queue."

De ahí el diseño de abajo:

 1. Nada de hilos. Un solo bucle que vigila la cola real del servidor.
 2. Tope propio de MAX_COLA=4 trabajos en `accepted`+`running` (el rechazo
    llegó con 5). Solo se envía si se está por debajo.
 3. Se adoptan los trabajos que ya están en la cola en vez de repetirlos:
    un trabajo encolado sobrevive a la muerte del proceso y se recupera por
    su jobID. Los 5 de arriba son meses que hacen falta; pedirlos otra vez
    sería tirar dos horas de cola y ocupar más sitio.
 4. Ante un rechazo, retroceso exponencial (5 min → 60 min) y el tope baja;
    se recupera solo tras varios envíos buenos.
 5. Los trabajos ya descargados se borran del servidor para liberar hueco.

Uso: python malla_04_climatologia.py
"""

import gc
import os
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr

from fwi_canadiense import calcular_fwi_serie
from malla_02_descarga import AREA, DATA, DIR, VARIABLES, a_diario, abre

CLIM = f"{DATA}/_clim"
ANIOS = range(2008, 2015)
VARS = ["tmax", "hr_min", "viento_max", "prec"]
COLECCION = "reanalysis-era5-land"

MAX_COLA = 4          # trabajos propios en accepted+running (rechazo con 5)
SONDEO = 60           # s entre sondeos de la cola
ESPERA_INI = 300      # s de retroceso tras el primer rechazo
ESPERA_MAX = 3600     # s de retroceso máximo


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


def ruta_mes(anio, mes):
    return f"{CLIM}/diario_{anio}{mes:02d}.npz"


def adopta(c, pend):
    """Trabajos que ya están en el servidor para meses pendientes.

    Un trabajo encolado sobrevive a la muerte del proceso: repetirlo tira la
    cola ya consumida y encima ocupa otro hueco del tope. Devuelve
    {(anio,mes): jobID} y borra los duplicados que sobren."""
    vivos, sobran = {}, []
    for st in ("successful", "running", "accepted"):
        for j in c.get_jobs(limit=100, sortby="-created", status=st) \
                  .json.get("jobs", []):
            if j.get("processID") != COLECCION:
                continue
            try:
                r = c.get_remote(j["jobID"]).request
                k = (int(r["year"]), int(r["month"]))
            except Exception:
                continue
            if k not in pend:
                continue
            if k in vivos:
                sobran.append(j["jobID"])      # ya había uno para ese mes
            else:
                vivos[k] = j["jobID"]
    for jid in sobran:                       # duplicados: liberan hueco
        try:
            c.get_remote(jid).delete()
            print(f"  duplicado borrado: {jid}", flush=True)
        except Exception:
            pass
    return vivos


def procesa(nc, anio, mes, la, lo):
    """Horario → diario → nodos → npz, y borra el horario acto seguido."""
    ds = a_diario(abre(nc))
    dim = [d for d in ds.dims if d not in ("latitude", "longitude")][0]
    ds = ds.rename({dim: "fecha"}).load()
    pts = ds.sel(latitude=xr.DataArray(la, dims="n"),
                 longitude=xr.DataArray(lo, dims="n"), method="nearest")
    np.savez_compressed(
        ruta_mes(anio, mes),
        fecha=ds["fecha"].values.astype("datetime64[D]").astype(int),
        **{v: pts[v].values.astype(np.float32) for v in VARS})
    ds.close(); del ds, pts; gc.collect()
    os.remove(nc)
    shutil.rmtree(nc + "_x", ignore_errors=True)


def descarga(pend, la, lo):
    """Bucle único que respeta la cola de CDS. Ver la cabecera del módulo."""
    c = cliente()
    os.makedirs(CLIM, exist_ok=True)
    trab = adopta(c, set(pend))
    if trab:
        print("  adoptados de la cola: " +
              ", ".join(f"{a}-{m:02d}" for a, m in sorted(trab)), flush=True)

    tope, espera, buenos = MAX_COLA, ESPERA_INI, 0
    pend = [k for k in pend]
    while pend:
        # --- 1. cosechar lo que ya esté listo -------------------------------
        for k in list(trab):
            a, m = k
            try:
                r = c.get_remote(trab[k])
                st = r.status
            except Exception as e:
                print(f"  {a}-{m:02d}: sondeo falló ({type(e).__name__}), "
                      f"se reintenta", flush=True)
                continue
            if st == "successful":
                nc = f"{CLIM}/_h_{a}{m:02d}.nc"
                print(f"  {a}-{m:02d}: descargando...", flush=True)
                try:
                    r.download(nc)
                    procesa(nc, a, m, la, lo)
                    r.delete()                       # libera hueco en la cola
                except Exception as e:
                    print(f"  !! {a}-{m:02d}: {type(e).__name__} {e}",
                          flush=True)
                    continue
                trab.pop(k); pend.remove(k)
                buenos += 1
                print(f"  {a}-{m:02d}: LISTO · quedan {len(pend)} meses",
                      flush=True)
                if buenos % 3 == 0 and tope < MAX_COLA:
                    tope += 1                        # el tope se recupera solo
                    espera = ESPERA_INI
                    print(f"  tope de cola → {tope}", flush=True)
            elif st in ("failed", "rejected"):
                print(f"  {a}-{m:02d}: {st} · reencolar tras {espera//60} min",
                      flush=True)
                try:
                    c.get_remote(trab[k]).delete()
                except Exception:
                    pass
                trab.pop(k)
                tope = max(1, min(tope - 1, len(trab)))
                buenos = 0
                time.sleep(espera)
                espera = min(espera * 2, ESPERA_MAX)

        # --- 2. rellenar la cola hasta el tope -------------------------------
        for k in pend:
            if len(trab) >= tope:
                break
            if k in trab:
                continue
            a, m = k
            print(f"  {a}-{m:02d}: enviando ({len(trab)+1}/{tope} en cola)...",
                  flush=True)
            try:
                trab[k] = c.submit(COLECCION, peticion(a, m)).request_id
            except Exception as e:
                print(f"  !! envío rechazado: {type(e).__name__} {e}",
                      flush=True)
                tope = max(1, len(trab))
                buenos = 0
                print(f"  tope de cola → {tope} · espera {espera//60} min",
                      flush=True)
                time.sleep(espera)
                espera = min(espera * 2, ESPERA_MAX)
                break

        if pend:
            time.sleep(SONDEO)


def main():
    n = np.load(f"{DATA}/nodos.npz")
    la, lo = n["nodo_lat"], n["nodo_lon"]
    print(f"nodos: {len(la):,} | años {ANIOS.start}-{ANIOS.stop-1} "
          f"| 84 meses\n")

    pend = [(a, m) for a in ANIOS for m in range(1, 13)
            if not os.path.exists(ruta_mes(a, m))]
    print(f"  {84-len(pend)} meses ya en disco · {len(pend)} pendientes "
          f"· tope de cola {MAX_COLA}\n", flush=True)
    if pend and not os.environ.get("SIN_DESCARGA"):
        descarga(pend, la, lo)
    elif pend:
        print(f"  SIN_DESCARGA: se omiten los {len(pend)} meses pendientes; "
              f"la climatologia se monta con lo que hay en disco\n", flush=True)

    rutas = [ruta_mes(a, m) for a in ANIOS for m in range(1, 13)
             if os.path.exists(ruta_mes(a, m))]
    print(f"\n  {len(rutas)} meses en disco. Montando la serie...", flush=True)

    ds = [np.load(r) for r in rutas]
    orden = np.argsort([d["fecha"][0] for d in ds])
    fechas = np.concatenate([ds[i]["fecha"] for i in orden])
    M = {v: np.concatenate([ds[i][v] for i in orden], axis=0) for v in VARS}
    meses = pd.to_datetime(fechas, unit="D").month.values
    print(f"  serie: {M['tmax'].shape[0]:,} días × {M['tmax'].shape[1]:,} nodos")

    print("\n  calculando el FWI por nodo...", flush=True)
    nn = M["tmax"].shape[1]
    fwi = np.full(M["tmax"].shape, np.nan, dtype=np.float32)
    for j in range(nn):
        fwi[:, j] = calcular_fwi_serie(M["tmax"][:, j], M["hr_min"][:, j],
                                       M["viento_max"][:, j] * 3.6,
                                       M["prec"][:, j], meses)["fwi"]
        if (j + 1) % 500 == 0:
            print(f"    {j+1}/{nn}", flush=True)

    print("\n  ordenando la climatología por nodo y mes...", flush=True)
    out = {}
    for m in range(1, 13):
        sel = fwi[meses == m]                       # (dias_mes, nodos)
        sel = np.sort(sel, axis=0)                  # NaN quedan al final
        out[f"m{m}"] = sel.astype(np.float32)
        out[f"n{m}"] = np.isfinite(fwi[meses == m]).sum(0).astype(np.int32)
    np.savez_compressed(f"{DATA}/clim_fwi_nodos.npz",
                        nodo_lat=la, nodo_lon=lo, **out)

    print("\n" + "=" * 66)
    a0 = pd.to_datetime(fechas.min(), unit="D").date()
    a1 = pd.to_datetime(fechas.max(), unit="D").date()
    print(f"CLIMATOLOGÍA POR NODO — ERA5-Land {a0} → {a1} ({len(rutas)} meses)")
    print("=" * 66)
    for m in (6, 7, 8, 9):
        v = out[f"m{m}"]
        fin = v[np.isfinite(v)]
        print(f"  mes {m:2d}: {out[f'n{m}'].min():3d}-{out[f'n{m}'].max():3d} días/nodo · "
              f"mediana {np.nanmedian(fin):6.1f} · p95 {np.nanpercentile(fin,95):6.1f} · "
              f"max {np.nanmax(fin):6.1f}")
    print(f"\nGuardado: malla_data/clim_fwi_nodos.npz")
    print("Siguiente: malla_05_riesgo.py (mapa sin IDW)")


if __name__ == "__main__":
    main()
