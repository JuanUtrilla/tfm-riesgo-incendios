#!/usr/bin/env python3
"""
Muestreador v4 — tabla maestra con el EGIF actualizado (Civio, release 09/07/2026).

NO SOBRESCRIBE NADA. Lee `egif_civio_2026-08.csv` y escribe
`dataset/muestra_maestra_v4.parquet`. El muestreador v1 y su salida quedan
intactos y siguen reproduciendo los resultados de la memoria actual.

=============================================================================
QUÉ CAMBIA RESPECTO A muestrear_dataset.py (v1)
=============================================================================

1. DOS AÑOS MÁS DE ETIQUETA.
   El CSV de v1 se descargó el 30/06/2026, nueve días antes de que Civio
   republicase el dataset consolidando 2021 para toda España y casi todo 2022.
   El corte en 2020 de v1 era correcto CON AQUELLOS DATOS; con los nuevos es
   una pérdida gratuita. Verificado año × provincia: 2021 pasa de 888 registros
   en 25 provincias a 2.897 en 50, y 2022 de 226 en 7 provincias a 2.520 en 48.

2. SPLIT NUEVO: train 2015-2020 / val 2021 / test 2022.
   v1 testeaba sobre 2020. Testear sobre 2022 —242.436 ha, el peor año de la
   serie, con Losacio y Bejís dentro— es mucho más exigente y mucho más
   defendible: es el escenario en el que a un sistema de riesgo se le pide que
   funcione, y el modelo no lo ha visto nunca.

3. FILTRO DE FRONTERA PARA 2022 (lo único conceptualmente nuevo).
   A 2022 le siguen faltando Navarra y Cantabria. Eso NO produce etiquetas
   erróneas por sí solo —una comunidad sin positivos simplemente no aporta
   celdas al diseño caso-control—, pero sí contamina el BUFFER DE EXCLUSIÓN:
   una celda de La Rioja o de Burgos podría sortear como "día sin fuego" un día
   en que ardió algo a 8 km al otro lado de la frontera, y el buffer no lo vería
   porque ese incendio no está en el CSV. Se excluyen del año 2022 todas las
   celdas a <15 km de Navarra o Cantabria (códigos 15 y 6 de la capa
   `AutonomousCommunities` de IberFire).
   Se excluyen sus positivos TAMBIÉN, no solo sus negativos: perder unos pocos
   positivos es barato; mantener falsos negativos en el año de test, no.
   Las Palmas (prov. 35) también falta en 2022 y no se trata: IberFire no cubre
   Canarias, así que esas celdas nunca entran en el dataset.

Todo lo demás es idéntico a v1 por diseño (mismo SEED, mismo ratio 1:3, mismo
buffer 12,5 km / ±10 días, misma dedup por celda-fecha): los cambios de métrica
entre v3 y v4 tienen que ser atribuibles a los datos nuevos, no al muestreo.

Uso:
    /home/charredgem/miniconda3/envs/tfm_fuego/bin/python muestrear_dataset_v4.py

⚠️ El python3 del sistema no tiene xarray. Usar el del entorno `tfm_fuego`.
"""

import os

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

# --- filtro de frontera 2022 -------------------------------------------------
ANIO_INCOMPLETO = 2022
CCAA_FALTANTES = {6: "Cantabria", 15: "Navarra"}   # códigos de IberFire
FRONTERA_KM = 15.0

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
RUTA_EGIF = f"{DIR}/egif_civio_2026-08.csv"
RUTA_IBERFIRE = f"{DIR}/iberfire/IberFire.nc"
RUTA_SALIDA = f"{DIR}/dataset/muestra_maestra_v4.parquet"

ANIO_MIN, ANIO_MAX = 2015, 2022
SPLITS = {"train": (2015, 2020), "val": (2021, 2021), "test": (2022, 2022)}


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

    # --- 1b. Zona de frontera con las CCAA que faltan en 2022 ---
    iy_f, ix_f = np.nonzero(np.isin(ccaa, list(CCAA_FALTANTES)))
    arbol_frontera = cKDTree(np.column_stack([xs[ix_f], ys[iy_f]]))
    print(f"Zona de frontera: {len(ix_f)} celdas en "
          f"{', '.join(CCAA_FALTANTES.values())}; se excluirá {ANIO_INCOMPLETO} "
          f"a <{FRONTERA_KM:.0f} km de ellas")

    # --- 2. Positivos EGIF ---
    egif = pd.read_csv(RUTA_EGIF, usecols=["id", "fecha", "lat", "lng",
                                           "superficie", "causa", "idprovincia"])
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna(subset=["fecha", "lat", "lng"])
    egif = egif[(egif["fecha"].dt.year >= ANIO_MIN) &
                (egif["fecha"].dt.year <= ANIO_MAX)]
    n_bruto = len(egif)

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X, Y = tr.transform(egif["lng"].values, egif["lat"].values)
    ix = np.rint((X - xs[0]) / paso_x).astype(int)
    iy = np.rint((Y - ys[0]) / paso_y).astype(int)

    dentro = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
    egif, ix, iy = egif[dentro], ix[dentro], iy[dentro]
    en_espana = es_espana[iy, ix]                    # descarta mar/Portugal/Canarias
    egif, ix, iy = egif[en_espana], ix[en_espana], iy[en_espana]
    print(f"EGIF {ANIO_MIN}-{ANIO_MAX} con coords: {n_bruto} | "
          f"dentro de la malla peninsular: {len(egif)}")

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

    # --- 2b. Aplicar el filtro de frontera SOLO al año incompleto ---
    anio_pos = pd.DatetimeIndex(pos["fecha"]).year
    d_frontera, _ = arbol_frontera.query(
        np.column_stack([xs[pos["ix"]], ys[pos["iy"]]]))
    contaminado = (anio_pos == ANIO_INCOMPLETO) & (d_frontera < FRONTERA_KM * 1000)
    print(f"Filtro de frontera: se descartan {int(contaminado.sum())} positivos "
          f"de {ANIO_INCOMPLETO} en la franja de {FRONTERA_KM:.0f} km "
          f"(de {int((anio_pos == ANIO_INCOMPLETO).sum())} de ese año)")
    pos = pos[~contaminado].reset_index(drop=True)

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

    # comprobación: ninguna fila de 2022 puede haber quedado en la franja
    if (df.anio == ANIO_INCOMPLETO).any():
        d, _ = arbol_frontera.query(
            np.column_stack([df.loc[df.anio == ANIO_INCOMPLETO, "x3035"],
                             df.loc[df.anio == ANIO_INCOMPLETO, "y3035"]]))
        assert (d >= FRONTERA_KM * 1000).all(), \
            "quedan filas de 2022 dentro de la franja de frontera"

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
    print("\nPositivos por año:")
    print(df[df.label == 1].groupby("anio").size().to_string())
    print("\nPositivos por mes (estacionalidad esperada, pico jul-ago-mar):")
    print(df[df.label == 1].groupby("mes").size().to_string())
    print("\nBloques espaciales de 100 km distintos:", df["bloque_100km"].nunique())
    ds.close()


if __name__ == "__main__":
    main()
