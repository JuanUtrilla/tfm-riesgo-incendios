#!/usr/bin/env python3
"""
Exporta a ~35 MB de npz/csv solo lo que el mapa nacional necesita del cubo
IberFire (29 GB), para que mapa_diario.py pueda correr en GitHub Actions sin
el cubo. Se ejecuta una vez en local (y otra vez solo si cambia el modelo).

Genera en <TFM_COLECTOR>/modelo/malla/ (el colector de AEMET):
- estaticas.npz:     13 capas estáticas 2D (float32) + x/y/is_spain
- mensual_m6..9.npz: clim. mensual 2020-24 de NDVI/LAI/SWI/LST + EGIF mismo-mes
                      (reutiliza malla_mensual de mapa_riesgo_hoy, misma caché)
- estacion_municipio.csv: idema → código INE del municipio más cercano
                      (evita el maestro de municipios en cada run de Actions)

Uso: python3 exportar_malla_gh.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from mapa_riesgo_hoy import DIR, malla_estaticas, malla_mensual

RAIZ = Path(__file__).resolve().parents[2]
REPO = os.environ.get("TFM_COLECTOR", os.path.join(
    os.environ.get("TFM_DATOS", str(RAIZ / "datos")), "colector"))
DESTINO = f"{REPO}/modelo/malla"


def main():
    import os
    os.makedirs(DESTINO, exist_ok=True)
    ds = xr.open_dataset(f"{DIR}/iberfire/IberFire.nc", decode_timedelta=False)

    print("estáticas...", flush=True)
    F = malla_estaticas(ds)
    out = {k: v.astype(np.float32) for k, v in F.items() if k != "ccaa"}
    out["ccaa"] = F["ccaa"].astype(np.int16)
    out["is_spain"] = ds["is_spain"].values.astype(bool)
    out["x"] = ds["x"].values
    out["y"] = ds["y"].values
    np.savez_compressed(f"{DESTINO}/estaticas.npz", **out)
    print(f"  estaticas.npz: {os.path.getsize(f'{DESTINO}/estaticas.npz')/1e6:.1f} MB")

    for mes in [6, 7, 8, 9]:
        M = malla_mensual(ds, mes)          # usa/crea la caché del prototipo
        M = {k: v.astype(np.float32) for k, v in M.items()}
        np.savez_compressed(f"{DESTINO}/mensual_m{mes}.npz", **M)
        print(f"  mensual_m{mes}.npz: "
              f"{os.path.getsize(f'{DESTINO}/mensual_m{mes}.npz')/1e6:.1f} MB", flush=True)

    print("estación → municipio...", flush=True)
    with open(f"{DIR}/prototipo/cache/municipios.json") as f:
        m = pd.DataFrame(json.load(f))
    m["cod"] = m["url"].str.extract(r"-id(\d{5})")
    m = m.dropna(subset=["cod"])
    m["la"] = pd.to_numeric(m["latitud_dec"], errors="coerce")
    m["lo"] = pd.to_numeric(m["longitud_dec"], errors="coerce")
    m = m.dropna(subset=["la", "lo"]).reset_index(drop=True)
    arbol = cKDTree(np.column_stack([m["la"], m["lo"]]))
    est = pd.read_parquet(f"{REPO}/modelo/estaciones_prototipo.parquet")
    _, j = arbol.query(np.column_stack([est["lat"], est["lon"]]))
    pd.DataFrame({"idema": est["idema"], "cod": m["cod"].values[j],
                  "capital": m["capital"].values[j]}) \
        .to_csv(f"{DESTINO}/estacion_municipio.csv", index=False)
    print(f"  estacion_municipio.csv: {len(est)} estaciones")


if __name__ == "__main__":
    main()
