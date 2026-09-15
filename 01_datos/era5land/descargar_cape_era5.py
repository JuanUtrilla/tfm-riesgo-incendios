"""Descarga CAPE + precipitación convectiva + K-index diarios de ERA5 single levels.

Proxy de "dry lightning" para el Modelo B (ver ESTADO_ARTE_ML_INCENDIOS.md y DATOS.md
§features_extra/ → rayos). Ojo: ERA5-Land no tiene CAPE; se usa el dataset derivado
de estadísticas diarias de ERA5 single levels (0.25°, CC-BY), misma API cdsapi.

Idempotente: un NetCDF por año y estadística en rayos_data/era5_cape/; si el fichero
existe y pesa >100 KB se salta. Se puede relanzar las veces que haga falta.

Uso: python descargar_cape_era5.py [cape|kindex|cp]
Sin argumento procesa las 3 variables en secuencia; con argumento, solo esa, lo
que permite lanzar 3 procesos en paralelo (el CDS admite varias peticiones por usuario).
"""
import os
import sys
import time
from pathlib import Path

import cdsapi
import requests

DIR_SALIDA = Path(__file__).parent / "rayos_data" / "era5_cape"
# Iberia peninsular + Baleares: N, W, S, E
AREA = [44.5, -10.0, 35.5, 4.5]
ANIOS = range(2010, 2026)

# (clave, nombre_fichero, estadística diaria, variable). Una variable por petición:
# el CDS rechaza por coste las peticiones anuales multivariable en este dataset.
PETICIONES = [
    ("cape", "cape_dailymax", "daily_maximum", "convective_available_potential_energy"),
    ("kindex", "kindex_dailymax", "daily_maximum", "k_index"),
    ("cp", "cp_dailymean", "daily_mean", "convective_precipitation"),
]


def _retrieve_con_reintentos(cliente, dataset, request, destino, max_intentos=50):
    """El CDS a veces rechaza por límite temporal de peticiones en cola por
    usuario ('Number queued requests ... temporarily limited'). En vez de morir,
    esperar y reintentar; la limitación suele levantarse en minutos."""
    for intento in range(1, max_intentos + 1):
        try:
            cliente.retrieve(dataset, request, destino)
            return
        except requests.HTTPError as e:
            texto = str(e)
            transitorio = "temporarily limited" in texto or "rejected" in texto
            if not transitorio or intento == max_intentos:
                raise
            espera = min(120 * intento, 900)
            print(f"  [cola llena] intento {intento}/{max_intentos}, "
                  f"reintento en {espera}s", flush=True)
            time.sleep(espera)


def _limpiar_huerfanos():
    """Borra trabajos accepted/running de este dataset que quedaran encolados en
    el CDS por ejecuciones anteriores matadas (ocupan el cupo de cola de la
    cuenta y hacen que toda petición nueva sea rechazada; lección del
    15/07/2026). Ojo: no lanzar este script varias veces a la vez, la limpieza
    de un proceso borraría los trabajos vivos del otro."""
    try:
        from ecmwf.datastores import Client as DSClient
        key = [l.split("key:")[1].strip()
               for l in open(os.path.expanduser("~/.cdsapirc")) if l.startswith("key:")][0]
        ds = DSClient(url="https://cds.climate.copernicus.eu/api", key=key)
        for j in ds.get_jobs(limit=50).json.get("jobs", []):
            if (j.get("status") in ("accepted", "running")
                    and j.get("processID") == "derived-era5-single-levels-daily-statistics"):
                ds.get_remote(j["jobID"]).delete()
                print(f"[limpieza] borrado trabajo huérfano {j['jobID'][:8]}", flush=True)
    except Exception as e:
        print(f"[limpieza] no se pudo revisar la cola: {type(e).__name__}: {e}", flush=True)


def main():
    filtro = sys.argv[1] if len(sys.argv) > 1 else None
    peticiones = [p for p in PETICIONES if filtro is None or p[0] == filtro]
    if not peticiones:
        raise SystemExit(f"variable desconocida: {filtro} (usar cape|kindex|cp)")
    _limpiar_huerfanos()
    cliente = cdsapi.Client(timeout=600, retry_max=5)
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    for anio in ANIOS:
        for _, nombre, estadistica, variable in peticiones:
            destino = DIR_SALIDA / f"{nombre}_{anio}.nc"
            if destino.exists() and destino.stat().st_size > 100_000:
                print(f"[skip] {destino.name}")
                continue
            print(f"[peticion] {destino.name} ...", flush=True)
            _retrieve_con_reintentos(cliente,
                "derived-era5-single-levels-daily-statistics",
                {
                    "product_type": "reanalysis",
                    "variable": [variable],
                    "year": str(anio),
                    "month": [f"{m:02d}" for m in range(1, 13)],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "daily_statistic": estadistica,
                    "time_zone": "utc+00:00",
                    "frequency": "1_hourly",
                    "area": AREA,
                },
                str(destino),
            )
            print(f"[ok] {destino.name} ({destino.stat().st_size/1e6:.1f} MB)", flush=True)
    print("COMPLETADO CAPE/ERA5")


if __name__ == "__main__":
    main()
