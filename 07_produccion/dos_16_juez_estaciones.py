#!/usr/bin/env python3
"""
Puntuación por estación: el ranking sellado de producción (AEMET, previsión
real, 687 estaciones) contra los mapas de este repo muestreados en esas mismas
estaciones, día a día, con la verdad de terreno EFFIS a 25 km. Es el único
cara a cara con lo que producción publicó realmente, y corre en GitHub.

Seguimiento diario de la temporada 2026; la validación del trabajo es el
replay de 2025-2026. No toca producción. Lee
`rankings/prevision_D0_<fecha>.csv` del colector de AEMET (TFM_COLECTOR, o la
carpeta que indique RANKINGS_DIR) y los mapas de salida/mapas_diarios/. Escribe salida/veredicto_estaciones.{csv,json}.

Criterio de etiqueta: estación positiva si hay celda quemada EFFIS (FIREDATE
ese día) a ≤25 km, el mismo de la validación del colector de AEMET y de
`comparar_rankings.py`. Métrica: AUC por día sobre las ~687 estaciones y
percentil de la estación en su propio ranking. Producción es previsión
(emitida esa mañana) y los mapas de este repo de ese día también (D0 de la
cadena de las 05:30): es un cara a cara justo, no la cota superior con
reanálisis de `comparar_rankings.py`. Se reescriben los últimos 45 días
(latencia de EFFIS).
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import maximum_filter

import config
import historico
from comparar_julio2026 import auc, quemadas

RANKINGS = os.environ.get(
    "RANKINGS_DIR",
    f"{os.environ.get('TFM_COLECTOR', config.FUENTE + '/colector')}/rankings")
GEOJSON = config.salida("effis_ba_season_ES.geojson")
CSV = config.salida("veredicto_estaciones.csv")
RADIO = 25
VENTANA = 45


def mapas(fecha):
    md = config.salida("mapas_diarios")
    out = {}
    r = f"{md}/riesgo_hoy_{fecha}.npz"
    if os.path.exists(r):
        out["malla"] = np.load(r)["prob"].astype(float)
    d = f"{md}/dos_riesgo_{fecha}.npz"
    if os.path.exists(d):
        z = np.load(d)
        out["unico"], out["pareja"] = z["prob_unico"].astype(float), z["prob_pareja"].astype(float)
        # r10 solo existe en los mapas del 31/08/2026 en adelante; los días
        # anteriores se quedan sin él y el juez los salta sin romperse.
        if "prob_r10" in z.files:
            out["r10"] = z["prob_r10"].astype(float)
    return out


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es = ds["is_spain"].values.astype(bool)
    xs, ys = ds["x"].values, ds["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    hoy = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    filas = []
    for f in sorted(glob.glob(f"{RANKINGS}/prevision_D0_*.csv")):
        fecha = os.path.basename(f)[13:23]
        d = pd.Timestamp(fecha)
        if d < hoy - pd.Timedelta(days=VENTANA) or d > hoy - pd.Timedelta(days=3):
            continue                     # EFFIS aún no ha madurado
        M = mapas(fecha)
        if not M:
            continue
        rk = pd.read_csv(f).dropna(subset=["lat", "lon", "prob"])
        X, Y = tr.transform(rk["lon"].values, rk["lat"].values)
        ix = np.rint((X - xs[0]) / (xs[1] - xs[0])).astype(int)
        iy = np.rint((Y - ys[0]) / (ys[1] - ys[0])).astype(int)
        ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
        ok &= es[np.clip(iy, 0, len(ys) - 1), np.clip(ix, 0, len(xs) - 1)]
        rk, ix, iy = rk[ok], ix[ok], iy[ok]
        quem, ha = quemadas(d, ds, es, ruta=GEOJSON)
        if quem.sum() == 0:
            continue
        cerca = maximum_filter(quem.astype(np.uint8), size=2 * RADIO + 1, mode="constant")
        y = cerca[iy, ix].astype(bool)
        if y.sum() == 0 or y.all():
            continue
        fila = {"fecha": fecha, "n_estaciones": int(len(rk)), "n_pos": int(y.sum()),
                "area_ha": round(ha, 1)}
        scores = {"produccion": rk["prob"].values}
        for nom, A in M.items():
            scores[nom] = A[iy, ix]
        for nom, s in scores.items():
            okv = np.isfinite(s)
            if y[okv].sum() == 0 or (~y[okv]).sum() == 0:
                continue
            fila[f"auc_{nom}"] = round(auc(s[okv & y], s[okv & ~y]), 4)
            r = pd.Series(s[okv]).rank(pct=True).values * 100
            fila[f"pctl_{nom}"] = round(float(np.median(r[y[okv]])), 1)
        filas.append(fila)
        print(f"  {fecha} · {len(rk)} estaciones · {int(y.sum())} positivas · "
              + " · ".join(f"{n} {fila.get(f'auc_{n}', float('nan')):.3f}" for n in scores),
              flush=True)
    ds.close()

    nuevo = pd.DataFrame(filas)
    if os.path.exists(CSV):
        viejo = pd.read_csv(CSV)
        if len(nuevo):
            viejo = viejo[~viejo["fecha"].isin(nuevo["fecha"])]
        nuevo = pd.concat([viejo, nuevo], ignore_index=True)
    if not len(nuevo):
        print("  sin días puntuables todavía"); return
    nuevo = nuevo.sort_values("fecha")
    nuevo.to_csv(CSV, index=False)
    rng = np.random.default_rng(0)
    R = {"dias": int(len(nuevo)), "positivas": int(nuevo.n_pos.sum())}
    print(f"\n== ESTACIONES · {len(nuevo)} días · verdad EFFIS a {RADIO} km ==")
    for nom in ("produccion", "malla", "unico", "r10", "pareja"):
        if f"auc_{nom}" not in nuevo:
            continue
        s = nuevo.dropna(subset=[f"auc_{nom}", "auc_produccion"])
        r = {"n_dias": int(len(s)), "auc": float(s[f"auc_{nom}"].mean()),
             "pctl": float(s[f"pctl_{nom}"].mean())}
        if nom != "produccion" and len(s) >= 3:
            dd = (s[f"auc_{nom}"] - s["auc_produccion"]).values
            b = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
            r["dif_vs_prod"] = float(dd.mean())
            r["ic95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            r["dias_gana"] = float((dd > 0).mean())
        R[nom] = r
        print(f"  {nom:11s} AUC {r['auc']:.3f} · pctl {r['pctl']:.1f}"
              + (f" · Δprod {r['dif_vs_prod']:+.3f} [{r['ic95'][0]:+.3f}, {r['ic95'][1]:+.3f}]"
                 f" · gana {r['dias_gana']*100:.0f}%" if "dif_vs_prod" in r else ""))
    json.dump(R, open(config.salida("veredicto_estaciones.json"), "w"), indent=1)
    historico.anotar("estaciones",
                     {n: r for n, r in R.items() if isinstance(r, dict)},
                     n_positivos=R["positivas"])


if __name__ == "__main__":
    main()
