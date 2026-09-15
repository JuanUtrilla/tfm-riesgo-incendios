#!/usr/bin/env python3
"""Rellena el hueco del archivo IFS (15-ago -> hoy) fuera del repo.

Baja de historical-forecast-api.open-meteo.com la previsión archivada tal y
como se emitió, misma forma que salida/ifs_malla_<fecha>.parquet:
columnas nodo, fecha, tmax, hr_min, viento_max, prec.

Cuota: 5.605 nodos x 1 unidad por tramo de 14 días, contra 10.000 al día. El
rango 08-ago -> 02-sep son 2 tramos = 11.210 unidades: no cabe en un día. Por
eso es reanudable por nodo y espera ante el 429 hasta que la cuota se renueve
a medianoche UTC.

No pisar la cadena viva: la corrida diaria necesita sus 5.605 unidades. Este
script espera a que la corrida del día haya terminado antes de pedir nada, y
vuelve a esperar cada vez que cruza la medianoche UTC.
"""
import os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{DIR}/ifs_archivo_2026.parquet"
RAIZ = Path(__file__).resolve().parents[2]
NODOS = f'{os.environ.get("TFM_ESTADO", str(RAIZ / "muestras"))}/nodos.npz'
REPO = "JuanUtrilla/tfm-riesgo-incendios"
INI, FIN = "2026-08-08", "2026-09-02"     # D-7 del 15-ago .. D+1 de hoy
VARS = ("temperature_2m_max,relative_humidity_2m_min,"
        "wind_speed_10m_max,precipitation_sum")
LOTE, PAUSA, ESPERA_MAX = 50, 20, 5400


def cadena_del_dia_hecha():
    """True si ya hay una corrida completa y terminada del día UTC en curso."""
    try:
        out = subprocess.run(
            ["gh", "run", "list", "-R", REPO, "--workflow", "mapa_diario.yml",
             "--limit", "20", "--json", "status,conclusion,createdAt"],
            capture_output=True, text=True, timeout=120).stdout
        import json
        hoy = pd.Timestamp.utcnow().tz_localize(None).date()
        for r in json.loads(out):
            c = pd.Timestamp(r["createdAt"]).tz_convert("UTC").tz_localize(None)
            if c.date() == hoy and r["status"] == "completed":
                return True
    except Exception as e:
        print(f"  no se pudo consultar Actions ({type(e).__name__}); espero", flush=True)
    return False


def esperar_turno():
    while not cadena_del_dia_hecha():
        print(f"  {time.strftime('%F %T')} UTC · la cadena del dia aun no ha "
              f"terminado; espero 20 min", flush=True)
        time.sleep(1200)


def main():
    n = np.load(NODOS)
    la, lo = n["nodo_lat"], n["nodo_lon"]
    filas, hechos = [], set()
    if os.path.exists(CACHE):
        prev = pd.read_parquet(CACHE)
        filas, hechos = [prev], set(prev["nodo"].unique())
        print(f"  ya en cache: {len(hechos):,}/{len(la):,} nodos", flush=True)
    pend = [i for i in range(len(la)) if i not in hechos]
    if not pend:
        print("  descarga completa"); return
    print(f"  pendientes {len(pend):,} nodos · ~{len(pend)*2:,} unidades", flush=True)

    dia = None
    for b in range(0, len(pend), LOTE):
        # al arrancar y en cada cambio de día UTC, ceder el paso a la cadena
        hoy = pd.Timestamp.utcnow().tz_localize(None).date()
        if hoy != dia:
            dia = hoy
            esperar_turno()
        idx = pend[b:b+LOTE]
        espera = PAUSA
        for _ in range(12):
            try:
                r = requests.get(
                    "https://historical-forecast-api.open-meteo.com/v1/forecast",
                    params=dict(
                        latitude=",".join(f"{la[i]:.4f}" for i in idx),
                        longitude=",".join(f"{lo[i]:.4f}" for i in idx),
                        start_date=INI, end_date=FIN, daily=VARS,
                        models="ecmwf_ifs025", timezone="UTC",
                        wind_speed_unit="ms"), timeout=240)
            except Exception as e:
                print(f"    red: {type(e).__name__} · {espera}s", flush=True)
                time.sleep(espera); espera = min(espera*2, ESPERA_MAX); continue
            if r.status_code != 429:
                break
            print(f"    429 (cuota) · esperando {espera//60 or 1} min · "
                  f"{time.strftime('%T')}", flush=True)
            time.sleep(espera); espera = min(espera*2, ESPERA_MAX)
        else:
            print("    12 intentos sin exito; guardo y salgo", flush=True); break
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}: {r.text[:150]}", flush=True); break
        j = r.json()
        for i, blo in zip(idx, j if isinstance(j, list) else [j]):
            d = blo["daily"]
            filas.append(pd.DataFrame({
                "nodo": i, "fecha": pd.to_datetime(d["time"]),
                "tmax": d["temperature_2m_max"],
                "hr_min": d["relative_humidity_2m_min"],
                "viento_max": d["wind_speed_10m_max"],
                "prec": d["precipitation_sum"]}))
        pd.concat(filas, ignore_index=True).to_parquet(CACHE, index=False)
        hechos.update(idx)
        print(f"    {len(hechos):,}/{len(la):,} nodos "
              f"({len(hechos)/len(la)*100:.0f} %) · {time.strftime('%T')}", flush=True)
        time.sleep(PAUSA)


if __name__ == "__main__":
    main()
