#!/usr/bin/env python3
"""
Histórico diario de AEMET 2015-2025, base para la climatología de FWI propia.

No sobrescribe nada. Escribe solo dataset/aemet_historico_2015_2025.parquet.

Para qué
--------
`modelo/clim_fwi/<idema>.npz` se construyó con la meteo del cubo (ERA5-Land,
2008-2014), ver preparar_prototipo.py:105-108. En producción el numerador del
percentil es un FWI calculado desde estación AEMET. Numerador y referencia son
fuentes distintas, y la feature satura: en entrenamiento julio-agosto el 1,0 %
de las filas cae en el percentil ≥99,9; en producción, el 12,6 %, con la
mediana nacional en el percentil 87 frente a 54 en entrenamiento.

El arreglo es rehacer la climatología con la misma fuente que el numerador.
Este script baja la materia prima.

Por qué 2015-2025 y no 2008-2014 (la ventana del cubo)
------------------------------------------------------
Replicar la ventana original sería lo ideal, porque cambiaría solo la fuente.
No se puede: comprobado el 18/08/2026, en julio de 2008 AEMET solo servía
446 estaciones (hoy 858) y a un tercio de las filas les faltaba `hrMin`,
`velmedia` o `racha`, que son justo los campos que el FWI necesita. Media red
se quedaría sin climatología.

Se usan 2015-2025: red densa, menos huecos, 11 años de muestra. El coste es
que la base climática es más reciente y por tanto algo más cálida que la del
cubo, lo que empuja los percentiles ligeramente hacia abajo. Es un efecto de
segundo orden frente a los 16,5 puntos de percentil que causa el cruce de
fuentes, y queda declarado.

2026 se excluye a propósito: es el año que se está prediciendo.

Detalle
-------
El endpoint `todasestaciones` admite rangos de ~15 días y devuelve todas las
estaciones de una vez, así que 11 años son ~270 peticiones, no 270×858.
Idempotente por tramos: los parquet parciales van a dataset/_hist_aemet/ y si
un tramo ya está, se salta. Se puede relanzar las veces que haga falta.

Uso: python descargar_historico_aemet.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DIR = Path(__file__).parent
PARTES = DIR / "dataset" / "_hist_aemet"
SALIDA = DIR / "dataset" / "aemet_historico_2015_2025.parquet"
INICIO, FIN = pd.Timestamp("2015-01-01"), pd.Timestamp("2025-12-31")
DIAS_TRAMO = 14


def key():
    for linea in (DIR / ".env").read_text().splitlines():
        for n in ("AEMET_API_KEY", "TOKEN_AEMET"):
            if linea.startswith(f"{n}="):
                return linea.split("=", 1)[1].strip()
    raise SystemExit("falta TOKEN_AEMET en .env")


def tramo(k, f0, f1, intentos=6):
    url = (f"https://opendata.aemet.es/opendata/api/valores/climatologicos/"
           f"diarios/datos/fechaini/{f0:%Y-%m-%d}T00:00:00UTC/"
           f"fechafin/{f1:%Y-%m-%d}T23:59:59UTC/todasestaciones")
    for i in range(intentos):
        try:
            r = requests.get(url, headers={"api_key": k}, timeout=30).json()
            if r.get("estado") == 200:
                d = requests.get(r["datos"], timeout=180)
                d.encoding = "latin-1"
                return pd.DataFrame(d.json())
            if r.get("estado") == 404:          # tramo sin datos, no es error
                return pd.DataFrame()
        except Exception as e:
            print(f"    error {type(e).__name__}", flush=True)
        time.sleep(70)
    print(f"    AGOTADO {f0:%F}", flush=True)
    return pd.DataFrame()


def main():
    PARTES.mkdir(parents=True, exist_ok=True)
    k = key()
    f0, n = INICIO, 0
    tramos = []
    while f0 <= FIN:
        f1 = min(f0 + pd.Timedelta(days=DIAS_TRAMO - 1), FIN)
        tramos.append((f0, f1))
        f0 = f1 + pd.Timedelta(days=1)
    print(f"{len(tramos)} tramos de {DIAS_TRAMO} días ({INICIO:%F} → {FIN:%F})")

    for f0, f1 in tramos:
        ruta = PARTES / f"{f0:%Y%m%d}.parquet"
        if ruta.exists():
            continue
        df = tramo(k, f0, f1)
        cols = ["indicativo", "fecha", "tmax", "tmin", "hrMin",
                "velmedia", "racha", "prec"]
        df = df[[c for c in cols if c in df.columns]] if len(df) else df
        df.to_parquet(ruta)
        n += 1
        if n % 10 == 0:
            print(f"  {f0:%F}: {n} tramos nuevos, última {len(df):,} filas",
                  flush=True)
        time.sleep(3)

    partes = sorted(PARTES.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in partes], ignore_index=True)
    df = df[df["indicativo"].notna()]

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")

    vel = num(df.get("velmedia", pd.Series(index=df.index, dtype=float)))
    rac = num(df.get("racha", pd.Series(index=df.index, dtype=float)))
    out = pd.DataFrame({
        "idema": df["indicativo"],
        "fecha": pd.to_datetime(df["fecha"], errors="coerce"),
        "tmax": num(df["tmax"]), "tmin": num(df["tmin"]),
        "hr_min": num(df.get("hrMin", pd.Series(index=df.index, dtype=float))),
        # fmin, no minimum: mismo arreglo que ranking_diario.py:75
        "viento_max": np.fmin(vel * 1.5, rac),
        "prec": num(df["prec"].replace("Ip", "0"))})
    out = (out.dropna(subset=["idema", "fecha"])
              .drop_duplicates(["idema", "fecha"])
              .sort_values(["idema", "fecha"]).reset_index(drop=True))
    out.to_parquet(SALIDA)
    print(f"\n{len(out):,} estación-día · {out.idema.nunique()} estaciones · "
          f"{out.fecha.min():%F} → {out.fecha.max():%F}")
    print(f"NaN: {(out[['tmax','hr_min','viento_max','prec']].isna().mean()*100).round(1).to_dict()}")
    print(f"guardado: {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
