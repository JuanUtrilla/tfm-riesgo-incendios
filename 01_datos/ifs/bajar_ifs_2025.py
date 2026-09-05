#!/usr/bin/env python3
"""Rellena el verano 2025 del archivo IFS FUERA del repo.

Baja de historical-forecast-api.open-meteo.com la previsión ARCHIVADA tal y
como se emitió, misma forma que salida/ifs_malla_<fecha>.parquet:
columnas nodo, fecha, tmax, hr_min, viento_max, prec.

CUOTA: 5.605 nodos x 1 unidad por tramo de 14 dias, contra 10.000 al dia. El
rango 08-ago -> 02-sep son 2 tramos = 11.210 unidades: NO cabe en un dia. Por
eso es reanudable por nodo y espera ante el 429 hasta que la cuota se renueve
a medianoche UTC.

NO PISAR LA CADENA VIVA: la corrida diaria necesita sus 5.605 unidades. Este
script espera a que la corrida del dia haya terminado antes de pedir nada, y
vuelve a esperar cada vez que cruza la medianoche UTC.
"""
import os, subprocess, sys, time
import numpy as np, pandas as pd, requests

DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{DIR}/ifs_archivo_2025.parquet"
NODOS = "/home/charredgem/Desktop/Master/TFM_fuego_malla/estado/nodos.npz"
REPO = "JuanUtrilla/tfm-fuego-malla"
INI, FIN = "2025-05-25", "2025-11-01"     # D-7 del 1-jun .. D+1 del 31-oct 2025
VARS = ("temperature_2m_max,relative_humidity_2m_min,"
        "wind_speed_10m_max,precipitation_sum")
LOTE, PAUSA, ESPERA_MAX = 50, 20, 5400


def cadena_del_dia_hecha():
    """True si ya hay una corrida COMPLETA y terminada del dia UTC en curso."""
    try:
        out = subprocess.run(
            ["gh", "run", "list", "-R", REPO, "--workflow", "malla_diaria.yml",
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
    # 02/09/2026: la cuota de Open-Meteo es por IP y la cadena corre en los
    # runners de GitHub, asi que no compiten. No se espera.
    return
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
    print(f"  pendientes {len(pend):,} nodos · ~{len(pend)*12:,} unidades", flush=True)

    dia = None
    for b in range(0, len(pend), LOTE):
        # al arrancar y en cada cambio de dia UTC, ceder el paso a la cadena
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
