#!/usr/bin/env python3
"""
Dos modelos, paso 7: el mapa del día con DÓNDE × CUÁNDO, al lado del de
producción. Prototipo, no operativo.

No toca producción. Reutiliza la cadena de `riesgo_hoy.py` (reanálisis +
IFS cacheado + features en nodos → celdas) y escribe salida/dos_07_mapa_<fecha>.*

Por qué así
-----------
`riesgo_hoy.py` ya construye las 46 features por celda para el día D. Lo
único que cambia aquí es qué se hace con ellas:

  prod        xgb_v2_prototipo sobre las 46 features (lo que se publica hoy)
  cuando      modelo `cuando` (solo dinámicas servibles) sobre sus 29 features
  donde       P(susceptibilidad) por celda, precalculada en `dos_05`
              (`dos_05_mapa_donde.npz`, variante con densidad 2008-14)
  riesgo      donde × cuando, el producto, que en `dos_05` empata con la suma
              de logits y gana a producción +0,08 de AUC dentro del día.

El IFS no se descarga: se usa la pasada ya cacheada (`--pasada`, por defecto
la última en salida/) para no pisar la cuota diaria de Open-Meteo que gasta
el cron de las 05:00. Con la pasada del día 20 se sirven D=20 y D+1=21.

La probabilidad del producto no está calibrada (dos prevalencias de diseño
distintas multiplicadas): se publica como percentil del día sobre las celdas
peninsulares, que es lo que usa el ranking operativo. Los cortes de nivel
de producción no se aplican al nuevo mapa: harían falta los suyos.

Uso:
    python dos_07_mapa_hoy.py                       # última pasada IFS cacheada
    python dos_07_mapa_hoy.py --pasada 2026-08-20 --dias 0 1
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from scipy.stats import spearmanr

import config
import config_expansion as ce
import malla_02b_ifs as ifsmod
import riesgo_hoy as rh
from dos_05_modelos import FEATS_CUANDO


def main(a):
    pasadas = sorted(glob.glob(config.salida("ifs_malla_*.parquet")))
    pasada = a.pasada or os.path.basename(pasadas[-1])[10:20]
    base = pd.Timestamp(pasada)
    objetivos = [base + pd.Timedelta(days=h) for h in a.dias]
    print(f"pasada IFS {pasada} · días {[str(o.date()) for o in objetivos]}")
    ifsmod.descarga = lambda *k, **kw: pd.read_parquet(ifsmod.ruta_cache(pasada))
    FULL = json.load(open(rh.FEATS_JSON))

    S, fwi, fechas, idx_nodo, la, lo, nre = rh.series_nodos(objetivos)
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    es_esp = ds["is_spain"].values.astype(bool)
    ys = ds["y"].values
    B = {}
    for k, v in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                 "rugosidad": "roughness_mean", "dist_carreteras": "dist_to_roads_mean",
                 "dist_rios": "dist_to_waterways_mean"}.items():
        B[k] = ds[v].values
    B["popdens"] = ds["popdens_2020"].values
    for k, suf in {"clc_bosque": "forest_proportion", "clc_matorral": "scrub_proportion",
                   "clc_agricola": "agricultural_proportion",
                   "clc_artificial": "artificial_proportion",
                   "clc_abierto": "open_space_proportion",
                   "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
        B[k] = ds[f"CLC_2018_{suf}"].values
    B["ccaa"] = np.nan_to_num(ds["AutonomousCommunities"].values, nan=-1).astype(int)
    B["frp_max_50km_7d"], B["n_detec_50km_7d"] = rh.firms_nrt(ds)
    B["rayos_dia"] = np.zeros((ny, nx)); B["rayos_7d"] = np.zeros((ny, nx))

    prod = xgb.XGBClassifier(); prod.load_model(rh.MODELO)
    cuando = xgb.XGBClassifier(); cuando.load_model(f"{ce.MODELOS}/cuando.ubj")
    md = np.load(config.salida("dos_05_mapa_donde.npz"))
    donde = np.full((ny, nx), np.nan)
    donde[md["iy"], md["ix"]] = md["p_donde_hist"]
    sel = es_esp.ravel()
    import holidays
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = {}
    for obj in objetivos:
        fstr = str(obj.date())
        k = int(np.where(fechas == obj)[0][0])
        print(f"\n=== {fstr} ===", flush=True)
        F, n_malo = rh.meteo_dia(S, fwi, fechas, k, obj.month, la, lo)
        M = rh.a_celdas(F, idx_nodo)
        G = dict(B); G.update(rh.mensual(ds, obj.month))
        G["ndvi_med_30d"] = G["ndvi"]
        fest = holidays.Spain(years=[obj.year])
        G["es_festivo"] = np.full((ny, nx), float(obj.weekday() >= 5 or obj.date() in fest))
        G["mes"] = np.full((ny, nx), obj.month)
        G["dia_anio"] = np.full((ny, nx), obj.dayofyear)
        X = pd.DataFrame({c: (M[c] if c in M else G[c]).ravel()[sel]
                          for c in sorted(set(FULL) | set(FEATS_CUANDO))})
        p_prod = prod.predict_proba(X[FULL].values)[:, 1]
        p_cuando = cuando.predict_proba(X[FEATS_CUANDO].values)[:, 1]
        p_donde = donde.ravel()[sel]
        p_riesgo = p_donde * p_cuando

        def pct(v):
            r = pd.Series(v).rank(pct=True).values * 100
            return r
        mapas = {"prod": p_prod, "cuando": p_cuando, "donde": p_donde, "riesgo": p_riesgo}
        rho = spearmanr(p_prod, p_riesgo, nan_policy="omit").correlation
        rho_dc = spearmanr(p_donde, p_cuando, nan_policy="omit").correlation
        top_prod = set(np.argsort(-p_prod)[:len(p_prod) // 20])
        top_new = set(np.argsort(-p_riesgo)[:len(p_prod) // 20])
        sol = len(top_prod & top_new) / len(top_prod)
        R[fstr] = {"spearman_prod_riesgo": float(rho), "spearman_donde_cuando": float(rho_dc),
                   "solape_top5pct": float(sol), "nodos_respaldo": n_malo,
                   "fwi_medio": float(np.nanmean(M["fwi"][es_esp])),
                   "cuando_mediana": float(np.nanmedian(p_cuando)),
                   "prod_mediana": float(np.nanmedian(p_prod))}
        print(f"  Spearman prod~riesgo {rho:.3f} · donde~cuando {rho_dc:.3f} · "
              f"solape del 5 % superior {sol*100:.0f} %")

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        tit = {"prod": "PRODUCCIÓN (xgb_v2, 46 features) — percentil del día",
               "cuando": "CUÁNDO (solo meteo/dinámicas) — percentil del día",
               "donde": "DÓNDE (susceptibilidad, estático) — percentil",
               "riesgo": "DÓNDE × CUÁNDO — percentil del día"}
        for ax, (nom, v) in zip(axs.ravel(), mapas.items()):
            g = np.full(ny * nx, np.nan); g[sel] = pct(v); g = g.reshape(ny, nx)
            im = ax.imshow(g, origin="lower" if ys[1] > ys[0] else "upper",
                           cmap="YlOrRd", vmin=0, vmax=100)
            ax.set_title(tit[nom], fontsize=11); ax.set_axis_off()
            # p: una fila por celda peninsular, en el orden de is_spain.ravel()
            np.savez_compressed(config.salida(f"dos_07_{nom}_{fstr}.npz"),
                                p=np.asarray(v, np.float32))
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="percentil")
        rama = "reanálisis + IFS mapeado" if k >= nre else "solo reanálisis"
        fig.suptitle(f"{fstr} · {rama} · Spearman producción~nuevo {rho:.2f} · "
                     f"solape del 5 % superior {sol*100:.0f} %", fontsize=13)
        fig.savefig(config.salida(f"dos_07_mapa_{fstr}.png"), dpi=130,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  guardado {config.salida(f'dos_07_mapa_{fstr}.png')}", flush=True)
    ds.close()
    json.dump(R, open(config.salida("dos_07_resumen.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pasada")
    p.add_argument("--dias", type=int, nargs="+", default=[0, 1])
    main(p.parse_args())
