#!/usr/bin/env python3
"""
Pipeline de malla — módulo 2b: previsión IFS en los nodos.

NO TOCA PRODUCCIÓN. Escribe salida/ifs_malla_<fecha>.parquet.

=============================================================================
QUÉ RAMA CUBRE
=============================================================================
ERA5-Land es reanálisis y llega con ~6 días de retraso, así que el mapa de HOY
y MAÑANA necesita otra fuente para el tramo final:

    ... D-30 ... D-7 │ D-6 ... D, D+1
        ERA5-Land    │      IFS
      (módulo 2)     │   (este módulo)

El IFS se pide a Open-Meteo (`ecmwf_ifs025`), no a CDS: CDS encola y no es de
baja latencia. Y no en GRIB del ECMWF, que exigiría `eccodes`.

=============================================================================
LA RAMA DE PREVISIÓN NO SE SIRVE CRUDA
=============================================================================
Medido en `malla_02b_hibrido.py`: el híbrido crudo satura `fwi_pctl_local` al
2,05 %, contra 0,23 % del reanálisis. No lo salva la memoria del FWI — con
tres días de IFS ya está el 83 % del daño, porque el valor diario lo manda el
FFMC y su constante de tiempo son horas.

La corrección va en `riesgo_hoy.py`, no aquí: este módulo baja la previsión
cruda y la deja tal cual. El mapeo (`mapeo_ifs_a_era5land.npz`) se aplica al
FWI ya calculado, que es donde se midió que funciona — 2,46 % → 0,41 % fuera
de muestra.

=============================================================================
CUOTA DE OPEN-METEO
=============================================================================
Cobra nodos × tramos de 14 días: 5.605 nodos × 1 tramo = 5.605 unidades,
contra 10.000 al día. Cabe, pero el límite es TAMBIÉN por minuto (~600) y
mandarlas seguidas devuelve 429 — comprobado. De ahí los lotes de 200 puntos
(medido: 200 coordenadas por petición responden en 0,4 s) con PAUSA segundos
entre lotes: 29 peticiones, unos 10 minutos.

La caché es por fecha de pasada. Repetir el mismo día no vuelve a pedir nada.

Uso:
    python malla_02b_ifs.py                 # pasada de hoy, D-7 → D+1
    python malla_02b_ifs.py --past-dias 10
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import requests

import config

URL = "https://api.open-meteo.com/v1/forecast"
VARS = ("temperature_2m_max,relative_humidity_2m_min,"
        "wind_speed_10m_max,precipitation_sum")
LOTE = 200            # coordenadas por petición
PAUSA = 20            # s entre lotes; con menos salta el 429
DIAS_PREV = 2         # D y D+1


def ruta_cache(fecha):
    return config.salida(f"ifs_malla_{fecha}.parquet")


def descarga(past_dias=7, fecha=None, forzar=False):
    """IFS diario en los 5.605 nodos. Cacheado por fecha de pasada."""
    fecha = fecha or str(pd.Timestamp.utcnow().tz_localize(None).date())
    ruta = ruta_cache(fecha)
    if os.path.exists(ruta) and not forzar:
        print(f"  IFS: cacheado ({ruta})", flush=True)
        return pd.read_parquet(ruta)

    n = np.load(config.NODOS)
    la, lo = n["nodo_lat"], n["nodo_lon"]
    parcial = ruta + ".parcial"
    filas, hechos = [], set()
    if os.path.exists(parcial):
        prev = pd.read_parquet(parcial)
        filas, hechos = [prev], set(prev["nodo"].unique())
        print(f"  IFS: reanudando, {len(hechos)} nodos ya bajados", flush=True)

    n_lotes = int(np.ceil(len(la) / LOTE))
    # 21/08/2026: Open-Meteo limita también POR HORA (~5.000 unidades): el cron
    # del 21 cayó en el lote 25/29 con 429 tras 6 reintentos de hasta 300 s.
    # Se llevan los envíos de la última hora y, si ya van LOTES_HORA, se
    # espera a que el más antiguo cumpla 61 min. Y el retroceso ante 429 llega
    # a 15 min × 8 intentos: se prefiere tardar a perder el día.
    LOTES_HORA = 23
    enviados = []
    for b in range(n_lotes):
        k = b * LOTE
        sl = slice(k, k + LOTE)
        if all(k + i in hechos for i in range(len(la[sl]))):
            continue
        enviados = [t for t in enviados if time.time() - t < 3660]
        if len(enviados) >= LOTES_HORA:
            dormir = 3660 - (time.time() - enviados[0])
            print(f"    tope horario: {len(enviados)} lotes en la última hora · "
                  f"esperando {dormir/60:.0f} min", flush=True)
            time.sleep(max(dormir, 1))
            enviados = [t for t in enviados if time.time() - t < 3660]
        espera = PAUSA
        for _ in range(8):
            try:
                r = requests.get(URL, params=dict(
                    latitude=",".join(f"{v:.4f}" for v in la[sl]),
                    longitude=",".join(f"{v:.4f}" for v in lo[sl]),
                    past_days=past_dias, forecast_days=DIAS_PREV, daily=VARS,
                    models="ecmwf_ifs025", timezone="UTC",
                    wind_speed_unit="ms"), timeout=180)
            except requests.exceptions.RequestException as e:
                # 21/08/2026: un ReadTimeout desde GitHub tiró la corrida entera
                # (solo se reintentaba el 429). La red también se reintenta.
                print(f"    red: {type(e).__name__} · esperando {espera}s", flush=True)
                time.sleep(espera)
                espera = min(espera * 2, 900)
                continue
            if r.status_code != 429:
                break
            print(f"    429 · esperando {espera}s", flush=True)
            time.sleep(espera)
            espera = min(espera * 2, 900)
        r.raise_for_status()
        enviados.append(time.time())
        j = r.json()
        for i, blo in enumerate(j if isinstance(j, list) else [j]):
            d = blo["daily"]
            filas.append(pd.DataFrame({
                "nodo": k + i, "fecha": pd.to_datetime(d["time"]),
                "tmax": d["temperature_2m_max"],
                "hr_min": d["relative_humidity_2m_min"],
                "viento_max": d["wind_speed_10m_max"],
                "prec": d["precipitation_sum"]}))
        pd.concat(filas, ignore_index=True).to_parquet(parcial, index=False)
        print(f"    lote {b + 1}/{n_lotes} · {min(k + LOTE, len(la))}/{len(la)} "
              f"nodos", flush=True)
        if b < n_lotes - 1:
            time.sleep(PAUSA)

    df = pd.concat(filas, ignore_index=True)
    df.to_parquet(ruta, index=False)
    os.remove(parcial)
    return df


def matrices(df, fechas):
    """DataFrame largo → {variable: (dias, nodos)} alineado a `fechas`."""
    return {v: df.pivot(index="fecha", columns="nodo", values=v)
            .reindex(fechas).values
            for v in ("tmax", "hr_min", "viento_max", "prec")}


def main(a):
    df = descarga(a.past_dias, a.fecha, a.forzar)
    f = pd.to_datetime(df["fecha"])
    print(f"\n  nodos: {df['nodo'].nunique():,}")
    print(f"  fechas: {f.min().date()} → {f.max().date()} "
          f"({f.dt.date.nunique()} días)")
    for v in ("tmax", "hr_min", "viento_max", "prec"):
        x = df[v].values.astype(float)
        print(f"    {v:<12} media {np.nanmean(x):8.2f} · "
              f"max {np.nanmax(x):8.2f} · NaN {np.isnan(x).mean() * 100:.1f}%")
    hoy = a.fecha or str(pd.Timestamp.utcnow().tz_localize(None).date())
    print(f"\nGuardado: {ruta_cache(hoy)}")
    print("Siguiente: riesgo_hoy.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--past-dias", type=int, default=7)
    p.add_argument("--fecha", default=None)
    p.add_argument("--forzar", action="store_true")
    main(p.parse_args())
