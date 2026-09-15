#!/usr/bin/env python3
"""
Descarga ERA5-Land horario de 2026 para el experimento B.

No sobrescribe nada. Escribe solo en dataset/era5_2026/.

Para qué
--------
La puntuación operativa sobre 2026 da AUC-ROC 0,64 frente al 0,923 del test
2020. El experimento A (`ablacion_proxies_operativos.py`) ya descartó que la
causa sea congelar el satélite o anular los rayos: eso vale −0,0045.

Queda una hipótesis viva: la meteo. En entrenamiento las 20 features
meteorológicas salen de ERA5-Land (dentro del cubo IberFire, reescalado a
1 km); en producción salen de estación AEMET y de forecast municipal. Son
sensores distintos, definiciones distintas y resoluciones distintas.

Este script baja la misma fuente que vio el modelo al entrenar, para los días
de 2026 que ya están puntuados, y así poder evaluar el modelo tres veces sobre
los mismos días y las mismas etiquetas:

    ERA5-Land (como en entrenamiento) → AEMET observada → AEMET forecast

La diferencia entre el primero y los otros dos es la degradación de datos,
aislada de todo lo demás (misma etiqueta, mismo modelo, mismas estáticas).

Decisiones que no son obvias
----------------------------
· `tp` (precipitación) es acumulada desde las 00 UTC del día y se resetea a las
  01:00. Verificado empíricamente el 18/08/2026 en 42,7N 8,2W: el 15-may sube
  de 0,065 a 2,126 mm a lo largo del día, el valor de las 00:00 del 16-may
  sigue siendo 2,126, y a las 01:00 vuelve a 0. Por tanto:
      precipitación del día D = tp a las 00:00 del día D+1
  De ahí que el rango baje un día más del último evaluado.

· El rango empieza 80 días antes del primer día a evaluar: es el `DIAS_SPINUP`
  que usa ranking_diario.py para el arranque del FWI. Menos spinup daría un DC
  (código de sequía, memoria larga) distinto del de producción y la comparación
  dejaría de ser limpia.

· Sobre el mar ERA5-Land es NaN. Las estaciones costeras se resuelven en el
  script de features, no aquí.

Uso:  python descargar_era5_2026.py
Idempotente: si el .nc del mes existe y pesa >1 MB, se salta. Relanzable.
"""

import sys
import time
from pathlib import Path

import cdsapi
import pandas as pd
import requests

DIR = Path(__file__).parent
SALIDA = DIR / "dataset" / "era5_2026"

# 80 días de spinup antes del 2026-07-01 (primer día de la serie retro) y un
# día extra por delante para la precipitación del último día evaluado.
INICIO = pd.Timestamp("2026-04-12")
FIN = pd.Timestamp("2026-08-14")

AREA = [45, -10, 35, 5]          # N, W, S, E; Iberia peninsular
VARIABLES = ["2m_temperature", "2m_dewpoint_temperature",
             "10m_u_component_of_wind", "10m_v_component_of_wind",
             "total_precipitation"]


def retrieve_con_reintentos(cli, req, destino, max_intentos=30):
    """El CDS rechaza peticiones cuando la cola del usuario está llena; no es
    un error definitivo. Mismo patrón que descargar_cape_era5.py."""
    for intento in range(1, max_intentos + 1):
        try:
            cli.retrieve("reanalysis-era5-land", req, str(destino))
            return
        except requests.HTTPError as e:
            transitorio = ("temporarily limited" in str(e)
                           or "rejected" in str(e) or "502" in str(e))
            if not transitorio or intento == max_intentos:
                raise
            espera = min(120 * intento, 900)
            print(f"  [cola] intento {intento}/{max_intentos}, "
                  f"reintento en {espera}s", flush=True)
            time.sleep(espera)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    cli = cdsapi.Client(quiet=True)
    meses = pd.date_range(INICIO, FIN, freq="D").to_period("M").unique()

    for per in meses:
        destino = SALIDA / f"era5land_{per}.nc"
        if destino.exists() and destino.stat().st_size > 1_000_000:
            print(f"{per}: ya está ({destino.stat().st_size/1e6:.0f} MB)",
                  flush=True)
            continue
        dias = pd.date_range(max(INICIO, per.start_time),
                             min(FIN, per.end_time), freq="D")
        req = {"variable": VARIABLES,
               "year": f"{per.year}", "month": f"{per.month:02d}",
               "day": [f"{d.day:02d}" for d in dias],
               "time": [f"{h:02d}:00" for h in range(24)],
               "area": AREA,
               "data_format": "netcdf", "download_format": "unarchived"}
        print(f"{per}: pidiendo {len(dias)} días...", flush=True)
        t0 = time.time()
        retrieve_con_reintentos(cli, req, destino)
        print(f"{per}: {destino.stat().st_size/1e6:.0f} MB en "
              f"{time.time()-t0:.0f}s", flush=True)

    total = sum(f.stat().st_size for f in SALIDA.glob("*.nc"))
    print(f"\nlisto: {len(list(SALIDA.glob('*.nc')))} ficheros, "
          f"{total/1e6:.0f} MB en {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
