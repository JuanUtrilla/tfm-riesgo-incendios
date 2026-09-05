#!/usr/bin/env python3
"""Verdad EFFIS por día y por incendio, para el replay.

Para cada día de la temporada rasteriza CADA perímetro con FIREDATE ese día
(misma rejilla y regla que `comparar_julio2026.quemadas`: EPSG:3035, celdas de
1 km, all_touched) y guarda:
  replay/verdad_<año>/<fecha>.npz   celda (índice plano), fuego (id), y por
                                     fuego: area_ha, n_celdas
  replay/verdad_<año>.csv            una fila por (fecha, fuego): area_ha,
                                     n_celdas (solo España)
Uso: python replay_verdad.py --anio 2025 --ini 2025-05-25 --fin 2025-11-01
"""
import argparse, os, sys, json
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{AQUI}/sandbox_replay"); os.chdir(f"{AQUI}/sandbox_replay")
import numpy as np, pandas as pd, xarray as xr, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
import config

EFFIS = {2025: f"{AQUI}/effis_ba_2025_ES.geojson",
         2026: f"{AQUI}/effis_ba_2026_season_congelado_2026-09-02.geojson"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anio", type=int, required=True)
    p.add_argument("--ini", required=True); p.add_argument("--fin", required=True)
    a = p.parse_args()
    cubo = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es_esp = cubo["is_spain"].values.astype(bool)
    ny, nx = es_esp.shape
    xs, ys = cubo["x"].values, cubo["y"].values
    px, ay = xs[1] - xs[0], abs(ys[1] - ys[0])
    norte = max(ys[0], ys[-1]) + ay / 2
    tr = from_origin(xs[0] - px / 2, norte, px, ay)
    voltear = ys[1] > ys[0]
    g = gpd.read_file(EFFIS[a.anio])
    col = "FIREDATE" if "FIREDATE" in g.columns else "firedate"
    g["f"] = pd.to_datetime(g[col], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    g = g.set_crs(4326, allow_override=True).to_crs(3035)
    dest = f"{AQUI}/replay/verdad_{a.anio}"; os.makedirs(dest, exist_ok=True)
    filas = []
    for d in pd.date_range(a.ini, a.fin, freq="D"):
        sel = g[g["f"] == d]
        celdas, fuegos, areas, ncel = [], [], [], []
        for k, (_, r) in enumerate(sel.iterrows()):
            if r.geometry is None: continue
            m = rasterize([(r.geometry, 1)], out_shape=(ny, nx), transform=tr,
                          fill=0, all_touched=True).astype(bool)
            if voltear: m = m[::-1]
            m &= es_esp
            idx = np.flatnonzero(m.ravel())
            if not len(idx): continue
            fid = str(r.get("id", k))
            celdas.append(idx); fuegos.append(np.full(len(idx), len(areas)))
            areas.append(float(pd.to_numeric(r.get("AREA_HA"), errors="coerce") or 0)); ncel.append(len(idx))
            filas.append(dict(fecha=str(d.date()), fuego=fid, area_ha=areas[-1], n_celdas=len(idx)))
        np.savez_compressed(f"{dest}/{d.date()}.npz",
                            celda=np.concatenate(celdas) if celdas else np.array([], int),
                            fuego=np.concatenate(fuegos) if fuegos else np.array([], int),
                            area_ha=np.array(areas), n_celdas=np.array(ncel))
    df = pd.DataFrame(filas)
    df.to_csv(f"{AQUI}/replay/verdad_{a.anio}.csv", index=False)
    dias = pd.date_range(a.ini, a.fin, freq="D")
    con = df["fecha"].nunique() if len(df) else 0
    print(f"{a.anio}: {len(dias)} días · {con} con fuego · {len(df)} incendios · "
          f"{(df.area_ha>=100).sum() if len(df) else 0} de ≥100 ha · "
          f"{df.n_celdas.sum() if len(df) else 0} celdas · {df.area_ha.sum() if len(df) else 0:,.0f} ha", flush=True)

if __name__ == "__main__":
    main()
