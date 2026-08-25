#!/usr/bin/env python3
"""
Dos modelos — paso 14: cortes de nivel BAJO/MODERADO/ALTO/EXTREMO para los
mapas nuevos, calibrados con la temporada 2026.

NO TOCA PRODUCCIÓN. Lee los mapas diarios de `dos_09` (disco externo) y
EFFIS; escribe salida/dos_14_cortes.json.

=============================================================================
POR QUÉ POR PERCENTIL Y NO POR PROBABILIDAD
=============================================================================
Producción corta la probabilidad en 0,25/0,55/0,80. Esa probabilidad está a
la prevalencia de diseño (25 %), no a la real (3·10⁻⁵), y el proyecto hermano
midió que BAJO tenía más fuego que MODERADO. El producto de dos modelos
además multiplica dos escalas distintas. Un corte por PERCENTIL DEL DÍA es
estable entre modelos y entre días: «EXTREMO = el 2 % de celdas más alto de
hoy» significa lo mismo siempre. El precio: el nivel no sube los días malos
de toda España. Para eso está la ALERTA GLOBAL, que se deja como número
aparte: el FWI medio del día contra su climatología.

Cortes elegidos para reproducir el reparto de producción el 20/08/2026
(EXTREMO 2,3 %, ALTO ~8 %, MODERADO ~70 %): p98 / p90 / p30. Aquí se MIDE, en
los 74 días de 2026, qué fracción de la superficie quemada cae en cada
nivel con cada mapa, para que el nivel tenga significado empírico:
«EXTREMO concentró el X % de lo quemado ocupando el 2 % del territorio».
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

import config
import config_expansion as ce
from comparar_julio2026 import quemadas

CORTES_PCTL = [30, 90, 98]
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]
MAPAS = ["prod", "donde_dia_effis", "donde_effis_c×cuando_egif", "donde_dia"]
GEOJSON = config.salida("effis_ba_season_ES.geojson")


def niveles(v):
    """Nivel por percentil del día (0..3) sobre las celdas finitas."""
    ok = np.isfinite(v)
    r = np.full(v.shape, -1)
    r[ok] = np.searchsorted(np.percentile(v[ok], CORTES_PCTL), v[ok], side="right")
    return r


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es = ds["is_spain"].values.astype(bool)
    acum = {m: np.zeros((4, 2)) for m in MAPAS}     # [nivel] → (celdas, quemadas)
    ha_nivel = {m: np.zeros(4) for m in MAPAS}
    n = 0
    for f in sorted(glob.glob(f"{ce.DATASET}/mapas_2026/*.npz")):
        d = pd.Timestamp(os.path.basename(f)[:10])
        q, ha = quemadas(d, ds, es, ruta=GEOJSON)
        q = q.ravel()[es.ravel()]
        m = np.load(f)
        for nom in MAPAS:
            if nom not in m.files:
                continue
            lv = niveles(m[nom])
            for k in range(4):
                sel = lv == k
                acum[nom][k] += (sel.sum(), (sel & q).sum())
        n += 1
    ds.close()
    R = {"cortes_percentil": CORTES_PCTL, "niveles": NIVELES, "n_dias": n, "mapas": {}}
    print(f"{n} días · cortes p{CORTES_PCTL}\n")
    for nom in MAPAS:
        a = acum[nom]
        if a.sum() == 0:
            continue
        tot_q = a[:, 1].sum()
        fila = {}
        print(f"  {nom}")
        for k, nv in enumerate(NIVELES):
            frac_terr = a[k, 0] / a[:, 0].sum()
            frac_quem = a[k, 1] / tot_q if tot_q else np.nan
            tasa = a[k, 1] / a[k, 0] * 1e5 if a[k, 0] else np.nan
            fila[nv] = {"frac_territorio": float(frac_terr), "frac_quemado": float(frac_quem),
                        "quemadas_por_100k_celdas_dia": float(tasa),
                        "lift": float(frac_quem / frac_terr) if frac_terr else np.nan}
            print(f"    {nv:9s} territorio {frac_terr*100:5.1f} % · quemado {frac_quem*100:5.1f} % "
                  f"· {tasa:6.1f}/100k · lift ×{frac_quem/frac_terr:4.1f}")
        R["mapas"][nom] = fila
    json.dump(R, open(config.salida("dos_14_cortes.json"), "w"), indent=1)
    print(f"\nguardado {config.salida('dos_14_cortes.json')}")


if __name__ == "__main__":
    main()
