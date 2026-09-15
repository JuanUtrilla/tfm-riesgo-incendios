#!/usr/bin/env python3
"""
v3: propensión local de fuego a partir de FIRMS (sustituto no caducable de la
autorregresiva del EGIF).

Problema que resuelve
El modelo de producción (xgb_v2_prototipo) conserva una sola feature
autorregresiva, `n_fuegos_10km_mismomes_hist`, que es la nº1 en importancia SHAP
y se calcula sobre el EGIF. El EGIF está incompleto desde 2021 (bitácora §2), así
que en producción esa feature está congelada en 2020: para el modelo, los
incendios de 2021-2026 no han existido, y la degradación crece cada año.

FIRMS (VIIRS, NASA) sí está disponible indefinidamente y en tiempo casi real, y
de hecho ya alimenta dos features vivas del modelo (frp_max_50km_7d,
n_detec_50km_7d). Este script construye el equivalente FIRMS de la propensión
local, para poder reentrenar con la misma definición que se usará en producción.
Es un requisito imprescindible: cambiar la fuente de una feature sin reentrenar
hace que el modelo la interprete con el significado viejo y falle en silencio
(mismo tipo de bug que los NaN y la racha de viento, bitácora §16).

Definiciones (todas estrictamente anteriores al año de la fila → sin fuga ni
artefacto de muestreo; positivos y negativos de la misma celda-año reciben el
mismo valor, luego no pueden discriminar por orden temporal)

  firms_diasfuego_mismomes_tasa
      días distintos con ≥1 detección FIRMS a <10 km, en el mismo mes de años
      anteriores, dividido por el nº de años anteriores disponibles.
      Análogo directo de n_fuegos_10km_mismomes_hist (estacionalidad local).

  firms_diasfuego_tasa
      días distintos con ≥1 detección a <10 km en años anteriores (todo el año),
      dividido por el nº de años anteriores disponibles. Propensión estructural.

  firms_frp_p95_hist
      percentil 95 del FRP (Fire Radiative Power, MW) de las detecciones a <10 km
      en años anteriores. Proxy de la intensidad típica del fuego en la zona,
      no solo de su frecuencia. NaN si no hay detecciones previas.

  firms_anios_previos
      nº de años completos de registro FIRMS disponibles antes del año de la
      fila. Es la "confianza" de las tasas anteriores; se pasa al modelo para que
      pueda descontar el ruido de las estimaciones con poco histórico.

Por qué tasas y no conteos
FIRMS empieza el 1-ene-2015 y el dataset entrena desde 2015. Un conteo acumulado
crecería con el calendario (2015 → 0 años de historia, 2026 → 11), lo que
introduciría una tendencia espuria y, peor, una diferencia sistemática entre
entrenamiento (historia corta) y producción (historia larga). Normalizando por
años disponibles la magnitud es estacionaria en expectativa; la varianza sí
depende del nº de años, y por eso se expone `firms_anios_previos`.
Las filas de 2015 no tienen ningún año previo → NaN (XGBoost lo gestiona de
forma nativa y aquí sí verá NaN al entrenar, que es la condición para que lo
maneje bien, bitácora §16).

Por qué días-fuego y no detecciones
Un solo gran incendio produce miles de detecciones; el conteo crudo es de cola
muy pesada y mide superficie, no recurrencia. Contar días distintos con fuego
se aproxima a un recuento de episodios y es mucho más comparable con la
semántica de "nº de incendios" del EGIF.

Entrada:  dataset/dataset_modelo_v1.parquet  (ix, iy, x3035, y3035, fecha, anio, mes)
          firms_iberia_2015_2024.parquet
Salida:   dataset/features_firms_hist_v3.parquet  (id_muestra + 4 columnas)

No modifica ningún fichero existente.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parents[2]
DIR = os.environ.get("TFM_DATOS", str(RAIZ / "datos"))
RUTA_DATASET = f"{DIR}/dataset/dataset_modelo_v1.parquet"
RUTA_FIRMS = f"{DIR}/firms_iberia_2015_2024.parquet"
RUTA_SALIDA = f"{DIR}/dataset/features_firms_hist_v3.parquet"

R_ZONA = 10_000.0        # mismo radio que la autorregresiva EGIF que sustituye
ANIO_INICIO_FIRMS = 2015  # primer año completo de VIIRS en el parquet


def main() -> None:
    df = pd.read_parquet(RUTA_DATASET,
                         columns=["id_muestra", "ix", "iy", "x3035", "y3035",
                                  "anio", "mes"])
    print(f"Muestras: {len(df)} | celdas únicas: "
          f"{df.groupby(['ix', 'iy'], sort=False).ngroups}", flush=True)

    firms = pd.read_parquet(RUTA_FIRMS,
                            columns=["latitude", "longitude", "acq_date", "frp"])
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    fx, fy = tr.transform(firms["longitude"].values, firms["latitude"].values)
    f_anio = firms["acq_date"].dt.year.values
    f_mes = firms["acq_date"].dt.month.values
    f_dia = firms["acq_date"].values.astype("datetime64[D]").astype(int)
    f_frp = firms["frp"].values.astype(float)
    arbol = cKDTree(np.column_stack([fx, fy]))
    print(f"FIRMS: {len(firms)} detecciones "
          f"{f_anio.min()}-{f_anio.max()}", flush=True)

    anios = np.arange(ANIO_INICIO_FIRMS, f_anio.max() + 1)
    idx_anio = {a: i for i, a in enumerate(anios)}

    out = []
    celdas = df.groupby(["ix", "iy"], sort=False)
    for n, (_, grupo) in enumerate(celdas, 1):
        cx, cy = grupo["x3035"].iloc[0], grupo["y3035"].iloc[0]
        vec = arbol.query_ball_point([cx, cy], r=R_ZONA)

        if vec:
            v_anio, v_mes = f_anio[vec], f_mes[vec]
            v_dia, v_frp = f_dia[vec], f_frp[vec]
            # tabla [año, mes] con nº de días distintos con detección
            tabla = np.zeros((len(anios), 13), dtype=np.int32)
            triples = np.unique(np.column_stack([v_anio, v_mes, v_dia]), axis=0)
            for a, m, _ in triples:
                if a in idx_anio:
                    tabla[idx_anio[a], m] += 1
            # FRP acumulado por año, para el percentil 95 de años previos
            frp_por_anio = [v_frp[v_anio == a] for a in anios]
        else:
            tabla = np.zeros((len(anios), 13), dtype=np.int32)
            frp_por_anio = [np.array([]) for _ in anios]

        for fila in grupo.itertuples():
            n_prev = fila.anio - ANIO_INICIO_FIRMS      # años completos previos
            if n_prev <= 0:
                out.append({"id_muestra": fila.id_muestra,
                            "firms_diasfuego_mismomes_tasa": np.nan,
                            "firms_diasfuego_tasa": np.nan,
                            "firms_frp_p95_hist": np.nan,
                            "firms_anios_previos": 0})
                continue

            prev = tabla[:n_prev]                        # años < anio de la fila
            frp_prev = np.concatenate(frp_por_anio[:n_prev]) \
                if n_prev else np.array([])

            out.append({
                "id_muestra": fila.id_muestra,
                "firms_diasfuego_mismomes_tasa": float(prev[:, fila.mes].sum()) / n_prev,
                "firms_diasfuego_tasa": float(prev.sum()) / n_prev,
                "firms_frp_p95_hist": float(np.percentile(frp_prev, 95))
                                      if len(frp_prev) else np.nan,
                "firms_anios_previos": int(n_prev),
            })

        if n % 2000 == 0:
            print(f"  celda {n}/{celdas.ngroups}", flush=True)

    res = pd.DataFrame(out)
    assert len(res) == len(df), (len(res), len(df))
    assert res["id_muestra"].is_unique
    res.to_parquet(RUTA_SALIDA, index=False)

    print(f"\nGuardado {RUTA_SALIDA}: {len(res)} filas × {len(res.columns)} cols")
    print("\nResumen por año (media de las tasas):")
    chk = res.merge(df[["id_muestra", "anio"]], on="id_muestra")
    print(chk.groupby("anio")[["firms_diasfuego_mismomes_tasa",
                               "firms_diasfuego_tasa", "firms_frp_p95_hist",
                               "firms_anios_previos"]].mean().round(3).to_string())
    print("\nNaN por columna:")
    print(res.isna().sum().to_string())


if __name__ == "__main__":
    main()
