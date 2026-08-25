#!/usr/bin/env python3
"""
Muestreador del dataset del modelo de riesgo (Modelo B).

Produce la tabla maestra (celda IberFire, fecha, label, split) que después se
rellena con features extraídas del datacubo IberFire y de las fuentes propias.

Diseño congelado (PROXIMOS_PASOS.md §⭐ Fase 2 + ESTADO_ARTE_ML_INCENDIOS.md §7.3):
- Positivos: incendios EGIF con coordenadas, asignados a su celda 1 km de
  IberFire (EPSG:3035), deduplicados por (celda, fecha).
- SOLO años 2015-2020: el EGIF de Civio está incompleto a partir de 2021 por
  el lag de consolidación del registro oficial (2021: 888, 2022: 226, 2023: 23
  incendios — 2022 fue en realidad el peor año según GWIS). Incluir esos años
  contaminaría tanto los positivos (faltan) como los negativos (días de fuego
  real sin registrar pasarían por pseudo-ausencias).
- Pseudo-ausencias 1:3: misma celda que un positivo, día aleatorio sin fuego
  dentro de los años del mismo split (así región y susceptibilidad quedan
  emparejadas y el modelo discrimina el CUÁNDO; la fecha uniforme deja que el
  modelo aprenda la estacionalidad — señal deseada según la literatura).
- Buffer de exclusión: un día NO puede ser negativo si hay un incendio EGIF a
  <12,5 km en ±10 días (mismo criterio que `is_near_fire` de IberFire: caja
  25x25 celdas x 10 días).
- Split temporal congelado: train 2015-2018 / val 2019 / test 2020.
- `bloque_100km`: bloque espacial de 100 km para el CV espacial (GroupKFold).

Salida: dataset/muestra_maestra_v1.parquet + resumen por stdout.
Determinista (SEED fija). No descarga nada: todo local.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

SEED = 42
RATIO_NEG = 3            # negativos por positivo
BUFFER_KM = 12.5         # radio de exclusión espacial (= caja 25x25 celdas IberFire)
BUFFER_DIAS = 10         # exclusión temporal alrededor de un incendio
MAX_INTENTOS = 200       # intentos de sorteo de fecha por negativo

RUTA_EGIF = "/home/charredgem/Desktop/Master/TFM_fuego/egif_civio.csv"
RUTA_IBERFIRE = "/home/charredgem/Desktop/Master/TFM_fuego/iberfire/IberFire.nc"
RUTA_SALIDA = "/home/charredgem/Desktop/Master/TFM_fuego/dataset/muestra_maestra_v1.parquet"

SPLITS = {"train": (2015, 2018), "val": (2019, 2019), "test": (2020, 2020)}


def asignar_split(anio: int) -> str:
    for nombre, (a0, a1) in SPLITS.items():
        if a0 <= anio <= a1:
            return nombre
    raise ValueError(f"año fuera de rango: {anio}")


def main() -> None:
    rng = np.random.default_rng(SEED)

    # --- 1. Malla IberFire (solo coords y máscaras estáticas, lectura ligera) ---
    ds = xr.open_dataset(RUTA_IBERFIRE, decode_timedelta=False)
    xs, ys = ds["x"].values, ds["y"].values          # centros de celda EPSG:3035
    es_espana = ds["is_spain"].values.astype(bool)
    ccaa = ds["AutonomousCommunities"].values
    paso_x, paso_y = xs[1] - xs[0], ys[1] - ys[0]  # ojo: y puede ser descendente

    # --- 2. Positivos EGIF 2015-2023 ---
    egif = pd.read_csv(RUTA_EGIF, usecols=["id", "fecha", "lat", "lng",
                                           "superficie", "causa", "idprovincia"])
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna(subset=["fecha", "lat", "lng"])
    egif = egif[(egif["fecha"].dt.year >= 2015) & (egif["fecha"].dt.year <= 2020)]
    n_bruto = len(egif)

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X, Y = tr.transform(egif["lng"].values, egif["lat"].values)
    ix = np.rint((X - xs[0]) / paso_x).astype(int)
    iy = np.rint((Y - ys[0]) / paso_y).astype(int)

    dentro = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
    egif, ix, iy = egif[dentro], ix[dentro], iy[dentro]
    en_espana = es_espana[iy, ix]                    # descarta mar/Portugal/Canarias
    egif, ix, iy = egif[en_espana], ix[en_espana], iy[en_espana]
    print(f"EGIF 2015-2020 con coords: {n_bruto} | dentro de la malla peninsular: {len(egif)}")

    pos = pd.DataFrame({
        "id_egif": egif["id"].values,
        "fecha": egif["fecha"].dt.normalize().values,
        "ix": ix, "iy": iy,
        "superficie": egif["superficie"].values,
        "causa": egif["causa"].values,
        "idprovincia": egif["idprovincia"].values,
    })
    # deduplicar por (celda, fecha): una celda-día solo puede ser un positivo
    pos = (pos.sort_values("superficie", ascending=False)
              .drop_duplicates(subset=["ix", "iy", "fecha"], keep="first")
              .reset_index(drop=True))
    print(f"Positivos únicos (celda, fecha): {len(pos)}")

    # --- 3. Índice espacial de fuegos para el buffer de exclusión ---
    fuego_xy = np.column_stack([xs[pos["ix"]], ys[pos["iy"]]])
    fuego_dia = pos["fecha"].values.astype("datetime64[D]").astype(int)
    arbol = cKDTree(fuego_xy)

    # celdas únicas con >=1 positivo → pool de celdas para negativos
    celdas = pos[["ix", "iy"]].drop_duplicates().reset_index(drop=True)
    celda_xy = np.column_stack([xs[celdas["ix"]], ys[celdas["iy"]]])
    vecinos = arbol.query_ball_point(celda_xy, r=BUFFER_KM * 1000.0)
    # días prohibidos por celda: fechas (ordenadas) de fuegos a <12,5 km
    dias_prohibidos = {i: np.sort(fuego_dia[v]) for i, v in enumerate(vecinos)}
    idx_celda = {(r.ix, r.iy): i for i, r in enumerate(celdas.itertuples())}

    # --- 4. Sorteo de pseudo-ausencias 1:3 ---
    negativos, descartados = [], 0
    for fila in pos.itertuples():
        anio = pd.Timestamp(fila.fecha).year
        split = asignar_split(anio)
        a0, a1 = SPLITS[split]
        d0 = np.datetime64(f"{a0}-01-01", "D").astype(int)
        d1 = np.datetime64(f"{a1}-12-31", "D").astype(int)
        prohibidos = dias_prohibidos[idx_celda[(fila.ix, fila.iy)]]

        conseguidos = 0
        for _ in range(MAX_INTENTOS):
            if conseguidos == RATIO_NEG:
                break
            dia = rng.integers(d0, d1 + 1)
            j = np.searchsorted(prohibidos, dia)
            cerca_antes = j > 0 and dia - prohibidos[j - 1] <= BUFFER_DIAS
            cerca_despues = j < len(prohibidos) and prohibidos[j] - dia <= BUFFER_DIAS
            if cerca_antes or cerca_despues:
                continue
            negativos.append((fila.ix, fila.iy, dia, fila.idprovincia))
            conseguidos += 1
        descartados += RATIO_NEG - conseguidos

    neg = pd.DataFrame(negativos, columns=["ix", "iy", "dia", "idprovincia"])
    neg["fecha"] = pd.to_datetime(neg.pop("dia"), unit="D", origin="unix")
    # un mismo (celda, fecha) sorteado dos veces no aporta: deduplicar
    n_antes = len(neg)
    neg = neg.drop_duplicates(subset=["ix", "iy", "fecha"]).reset_index(drop=True)
    print(f"Negativos sorteados: {n_antes} (dedup → {len(neg)}; "
          f"no conseguidos por buffer: {descartados})")

    # --- 5. Tabla maestra ---
    pos["label"] = 1
    neg["label"] = 0
    df = pd.concat([pos, neg], ignore_index=True)

    df["anio"] = pd.DatetimeIndex(df["fecha"]).year
    df["mes"] = pd.DatetimeIndex(df["fecha"]).month
    df["split"] = df["anio"].map(asignar_split)
    df["x3035"] = xs[df["ix"]]
    df["y3035"] = ys[df["iy"]]
    df["ccaa"] = ccaa[df["iy"], df["ix"]]
    df["bloque_100km"] = (np.floor(df["x3035"] / 100_000).astype(int) * 1000
                          + np.floor(df["y3035"] / 100_000).astype(int))
    inv = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(df["x3035"].values, df["y3035"].values)
    df["lon_celda"], df["lat_celda"] = lon, lat
    df["id_muestra"] = np.arange(len(df))

    df = df[["id_muestra", "id_egif", "fecha", "anio", "mes", "split", "label",
             "ix", "iy", "x3035", "y3035", "lat_celda", "lon_celda",
             "ccaa", "idprovincia", "bloque_100km", "superficie", "causa"]]

    import os
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    df.to_parquet(RUTA_SALIDA, index=False)

    # --- 6. Resumen y comprobaciones ---
    print(f"\nGuardado: {RUTA_SALIDA} ({len(df)} filas)")
    print("\nFilas por split × label:")
    print(df.pivot_table(index="split", columns="label", values="id_muestra",
                         aggfunc="count").rename(columns={0: "neg", 1: "pos"}))
    print("\nRatio neg:pos por split:")
    tabla = df.groupby(["split", "label"]).size().unstack()
    print((tabla[0] / tabla[1]).round(2))
    print("\nPositivos por mes (estacionalidad esperada, pico jul-ago-mar):")
    print(df[df.label == 1].groupby("mes").size().to_string())
    print("\nBloques espaciales de 100 km distintos:", df["bloque_100km"].nunique())
    ds.close()


if __name__ == "__main__":
    main()
