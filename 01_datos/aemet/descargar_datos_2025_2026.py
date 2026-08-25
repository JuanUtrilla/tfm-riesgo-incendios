#!/usr/bin/env python3
"""
Descarga de datos 2025-2026 para la validación por estaciones:

1. AEMET climatologías diarias de TODAS las estaciones, 2024-11-01 → hoy
   (el arranque en nov-2024 da ~60 días de spin-up al FWI antes del 1-ene-2025).
   Endpoint `todasestaciones` en trozos de 14 días (~45 peticiones), con la
   caché permanente de tiempo_real._todas_chunk → re-lanzar es gratis.
   Salida: dataset/aemet_diario_2025_2026.parquet

2. Detecciones FIRMS 2025-01-01 → hoy (el parquet histórico termina en ene-2025):
   VIIRS NOAA-20 archivo (SP) con fallback a NRT para lo reciente, trozos de
   10 días. Salida: dataset/firms_2025_2026.parquet

Idempotente y resumible (cachés por trozo). Reintenta con espera ante 429/fallos.
"""

import os
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
load_dotenv(f"{DIR}/.env")

import tiempo_real as trm  # reutiliza _todas_chunk y su caché


def aemet_diario():
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    filas = []
    f0 = pd.Timestamp("2024-11-01")
    n_chunk = 0
    while f0 <= hoy:
        f1 = min(f0 + pd.Timedelta(days=13), hoy)
        edad = 6 * 3600 if f1 >= hoy - pd.Timedelta(days=14) else 10**9
        for intento in range(6):
            datos = trm._todas_chunk(f0.strftime("%Y-%m-%d"),
                                     f1.strftime("%Y-%m-%d"), edad)
            if datos:
                break
            print(f"  [{f0.date()}] vacío/429, reintento {intento+1}/6 en 70s",
                  flush=True)
            time.sleep(70)
        filas += datos or []
        n_chunk += 1
        if n_chunk % 5 == 0:
            print(f"  AEMET: {n_chunk} trozos, {len(filas)} registros", flush=True)
        time.sleep(3)          # cortesía con el rate limit
        f0 = f1 + pd.Timedelta(days=1)

    df = pd.DataFrame(filas)

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")

    velmedia, racha = num(df["velmedia"]), num(df["racha"])
    out = pd.DataFrame({
        "idema": df["indicativo"], "fecha": pd.to_datetime(df["fecha"]),
        "tmax": num(df["tmax"]), "tmin": num(df["tmin"]),
        "hr_min": num(df.get("hrMin", pd.Series(dtype=float))),
        "viento_max": np.minimum(velmedia * 1.5, racha),
        "prec": num(df["prec"].replace("Ip", "0")),
    }).dropna(subset=["fecha"])
    out = out.sort_values(["idema", "fecha"]).drop_duplicates(["idema", "fecha"])
    out.to_parquet(f"{DIR}/dataset/aemet_diario_2025_2026.parquet", index=False)
    print(f"AEMET diario: {len(out)} filas, {out.idema.nunique()} estaciones, "
          f"{out.fecha.min().date()} → {out.fecha.max().date()}", flush=True)


def firms():
    key = os.environ["FIRMS_MAP_KEY"]
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    os.makedirs(f"{DIR}/dataset/_firms_2025", exist_ok=True)
    partes = []
    f0 = pd.Timestamp("2025-01-01")
    while f0 <= hoy:
        ruta = f"{DIR}/dataset/_firms_2025/{f0.strftime('%Y%m%d')}.csv"
        if not os.path.exists(ruta):
            txt = None
            for prod in ["VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"]:
                # el API limita a 5 días por petición (antes eran 10)
                url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
                       f"{prod}/-10,35,5,44/5/{f0.strftime('%Y-%m-%d')}")
                try:
                    r = requests.get(url, timeout=60)
                    if r.ok and r.text.startswith("latitude"):
                        # SP puede devolver solo cabecera si aún no hay archivo
                        if len(r.text.strip().splitlines()) > 1 or prod.endswith("NRT"):
                            txt = r.text
                            break
                except requests.RequestException:
                    pass
                time.sleep(5)
            if txt is None:
                print(f"  FIRMS {f0.date()}: sin datos (se reintentará otro día)",
                      flush=True)
                f0 += pd.Timedelta(days=5)
                continue
            with open(ruta, "w") as fh:
                fh.write(txt)
            time.sleep(3)
        partes.append(pd.read_csv(ruta))
        f0 += pd.Timedelta(days=5)

    df = pd.concat(partes, ignore_index=True)
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    df.to_parquet(f"{DIR}/dataset/firms_2025_2026.parquet", index=False)
    print(f"FIRMS 2025-2026: {len(df)} detecciones, "
          f"{df.acq_date.min()} → {df.acq_date.max()}", flush=True)


if __name__ == "__main__":
    import sys
    if "--solo-firms" not in sys.argv:
        aemet_diario()
    firms()
    print("DESCARGAS 2025-2026 COMPLETAS", flush=True)
