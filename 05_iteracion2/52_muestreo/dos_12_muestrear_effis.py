#!/usr/bin/env python3
"""
Dos modelos — paso 12: tabla maestra con ETIQUETA EFFIS (superficie quemada),
la misma con la que se valida en operación.

NO TOCA PRODUCCIÓN. Lee el cubo; escribe en el disco externo.

=============================================================================
POR QUÉ OTRA ETIQUETA
=============================================================================
`dos_09` midió la temporada 2026: el DÓNDE entrenado con igniciones EGIF se
hunde en los megaincendios, y el entrenado con superficie quemada EFFIS
(`dos_10`) empata con producción. La validación operativa (EFFIS, `puntuar_effis`)
mide superficie quemada; el EGIF llega con 2-4 años de retraso y nunca
servirá para validar. Así que se entrena con la etiqueta que sí se medirá.

Positivo: celda con `is_fire`=1 (EFFIS ≥5 ha) en su PRIMER día (`is_fire` en
t y no en t−1): es la FIREDATE del perímetro, exactamente lo que puntúa
`quemadas()` en la validación. Un perímetro grande son cientos de celdas el
mismo día: se limita a 30 por (día, bloque de 100 km) para que un megaincendio
no sea la mitad del dataset. 2015-2024 (el cubo llega a 2024-12-31).

Negativos, los dos diseños de `dos_02`, 1:3 cada uno:
  cuando   misma celda, otro día de los años del mismo split
  donde    mismo día, otra celda de España al azar
Se sortean 4 por positivo; los que caigan en `is_near_fire`=1 (fuego EFFIS a
<12,5 km en ±10 días) se eliminan en `dos_03` y se recorta a 3.

Split temporal: train 2015-2022 (incluye el año récord a propósito: es donde
se aprende el megaincendio), val 2023, test 2024. La temporada 2026 es la
prueba externa (`dos_09`). `eval_dia`: verano 2024 y 2023, 1.000 celdas al
azar por día + todas las celdas EFFIS de primer día del día (hasta 60).

Salida: <externo>/dataset/maestra_effis.parquet
"""

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import config
import config_expansion as ce

SEED = 7
RATIO = 4
TOPE_POS_DIA_BLOQUE = 30
ANIOS = (2015, 2024)
SPLITS = {"train": (2015, 2022), "val": (2023, 2023), "test": (2024, 2024)}
ANIOS_EVAL, MESES_EVAL = (2023, 2024), (6, 9)
N_AZAR, N_EFFIS = 1000, 60
SALIDA = f"{ce.DATASET}/maestra_effis.parquet"


def split_de(a):
    return next((k for k, (x, y) in SPLITS.items() if x <= a <= y), "fuera")


def main():
    rng = np.random.default_rng(SEED)
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    xs, ys = ds["x"].values, ds["y"].values
    esp = ds["is_spain"].values.astype(bool)
    ccaa = ds["AutonomousCommunities"].values
    iy_esp, ix_esp = np.where(esp)
    t = ds["time"].values.astype("datetime64[D]")
    anio = t.astype("datetime64[Y]").astype(int) + 1970
    k = np.where((anio >= ANIOS[0] - 0) & (anio <= ANIOS[1]))[0]
    k = np.r_[k[0] - 1, k]                      # y el día anterior al primero
    print("leyendo is_fire 2015-2024 por bloques...", flush=True)
    fuego = []
    for by in range(0, esp.shape[0], 77):
        for bx in range(0, esp.shape[1], 99):
            sy, sx = slice(by, by + 77), slice(bx, bx + 99)
            if not esp[sy, sx].any():
                continue
            f = ds["is_fire"].isel(time=k, y=sy, x=sx).values.astype(bool)
            f &= esp[sy, sx][None]
            primer = f[1:] & ~f[:-1]
            tt, yy, xx = np.where(primer)
            fuego.append(pd.DataFrame({"t": k[1:][tt], "iy": yy + by, "ix": xx + bx}))
    fuego = pd.concat(fuego, ignore_index=True)
    fuego["fecha"] = pd.to_datetime(t[fuego["t"].values])
    fuego["x3035"], fuego["y3035"] = xs[fuego["ix"]], ys[fuego["iy"]]
    fuego["bloque_100km"] = (np.floor(fuego["x3035"] / 1e5) * 1000
                             + np.floor(fuego["y3035"] / 1e5)).astype(int)
    print(f"  celdas de primer día 2015-2024: {len(fuego):,}")

    # --- positivos con tope por (día, bloque) -------------------------------
    pos = (fuego.sample(frac=1, random_state=SEED)
                .groupby(["fecha", "bloque_100km"]).head(TOPE_POS_DIA_BLOQUE)
                .reset_index(drop=True))
    pos["label"] = np.int8(1)
    print(f"  positivos tras el tope: {len(pos):,} · "
          f"{pos.groupby(pos.fecha.dt.year).size().to_dict()}")

    # --- negativos cuando: misma celda, otro día del split -----------------
    filas = []
    for r in pos.itertuples():
        a0, a1 = SPLITS[split_de(r.fecha.year)]
        d0 = np.datetime64(f"{a0}-01-01", "D").astype(int)
        d1 = np.datetime64(f"{a1}-12-31", "D").astype(int)
        for d in rng.integers(d0, d1 + 1, RATIO):
            filas.append((pd.Timestamp(np.datetime64(int(d), "D")), r.ix, r.iy))
    cuando = pd.DataFrame(filas, columns=["fecha", "ix", "iy"])
    cuando["label"] = np.int8(0)
    cuando = pd.concat([pos[["fecha", "ix", "iy", "label"]], cuando])
    cuando["disenio"] = "cuando"

    # --- negativos donde: mismo día, otra celda -----------------------------
    filas = []
    for r in pos.itertuples():
        for j in rng.integers(len(iy_esp), size=RATIO):
            filas.append((r.fecha, ix_esp[j], iy_esp[j]))
    donde = pd.DataFrame(filas, columns=["fecha", "ix", "iy"])
    donde["label"] = np.int8(0)
    donde = pd.concat([pos[["fecha", "ix", "iy", "label"]], donde])
    donde["disenio"] = "donde"

    # --- eval_dia 2023 y 2024 ---------------------------------------------
    ev = []
    for a in ANIOS_EVAL:
        for d in pd.date_range(f"{a}-{MESES_EVAL[0]:02d}-01", f"{a}-{MESES_EVAL[1]:02d}-30"):
            for j in rng.choice(len(iy_esp), N_AZAR, replace=False):
                ev.append((d, ix_esp[j], iy_esp[j], "azar", 0))
            fz = fuego[fuego["fecha"] == d]
            if len(fz) > N_EFFIS:
                fz = fz.sample(N_EFFIS, random_state=int(d.dayofyear))
            for r in fz.itertuples():
                ev.append((d, r.ix, r.iy, "effis", 1))
    ev = pd.DataFrame(ev, columns=["fecha", "ix", "iy", "origen", "label"])
    ev["label"] = ev["label"].astype(np.int8)
    ev = ev.sort_values("origen").drop_duplicates(["fecha", "ix", "iy"])  # effis antes que azar
    ev["disenio"] = "eval_dia"
    ev["primer_dia"] = ev["label"]

    df = pd.concat([cuando, donde, ev], ignore_index=True)
    df = df.drop_duplicates(["disenio", "fecha", "ix", "iy"]).reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["anio"] = df["fecha"].dt.year.astype(np.int16)
    df["mes"] = df["fecha"].dt.month.astype(np.int8)
    df["split"] = df["anio"].map(split_de)
    df.loc[df["disenio"] == "eval_dia", "split"] = "eval"
    df["x3035"], df["y3035"] = xs[df["ix"].values], ys[df["iy"].values]
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    df["lon_celda"], df["lat_celda"] = tr.transform(df["x3035"].values, df["y3035"].values)
    df["ccaa"] = ccaa[df["iy"].values, df["ix"].values]
    df["bloque_100km"] = (np.floor(df["x3035"] / 1e5) * 1000
                          + np.floor(df["y3035"] / 1e5)).astype(int)
    df["label_egif"] = np.int8(0)
    df["origen"] = df["origen"].fillna("muestra")
    df["id_muestra"] = np.arange(len(df))
    df.to_parquet(SALIDA, index=False)
    print(f"\nguardado {SALIDA}: {df.shape}")
    print(df.groupby(["disenio", "split", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
