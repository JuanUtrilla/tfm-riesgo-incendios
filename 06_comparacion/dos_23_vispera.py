#!/usr/bin/env python3
"""
Dos modelos — paso 23: el mapa de AYER contra el fuego de HOY.

Los pasos 20-22 puntúan el mapa del día D contra el fuego del día D, y ese
mapa lleva dentro el resumen meteorológico del propio día D (tmax, hr_min,
viento_max: la tarde). No es una previsión, es un diagnóstico concurrente.
`meteo_dia` usa slice(0, k+1) — el día D incluido.

Aquí se hace lo único genuinamente anticipado que permiten los datos que ya
están en disco: puntuar los incendios del día D con el mapa del día D−1
(y D−2, D−3), que están construidos SOLO con información existente la noche
anterior. Es una previsión por persistencia: no reentrena nada, asume que el
riesgo de ayer sigue valiendo hoy. Cota INFERIOR de lo que daría un mapa con
IFS, que sí anticipa el tiempo de mañana.

La caída de D+0 a D+1 es la medida honesta de cuánto del acierto venía de
conocer la tarde.

NO TOCA PRODUCCIÓN. Escribe salida/dos_23_vispera.{json,csv}.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

import config
import config_expansion as ce
from dos_21_hectareas import CLASES, verdad_por_dia

TOPK = [0.005, 0.01, 0.02, 0.05]
DESFASES = [0, 1, 2, 3]                 # el mapa de D-desfase contra el fuego de D
MAPAS = ["prod", "prod_sin_firms", "donde_dia_effis", "donde_dia_effis_r10",
         "donde_effis_c×cuando_egif", "donde", "fwi_pctl"]


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es = ds["is_spain"].values.astype(bool)
    _, incendios = verdad_por_dia(ds, es)
    ds.close()
    por_dia = {}
    for inc in incendios:
        por_dia.setdefault(inc["fecha"], []).append(inc)

    ficheros = sorted(glob.glob(f"{ce.DATASET}/mapas_2026/*.npz"))
    fechas = [pd.Timestamp(os.path.basename(f)[:10]) for f in ficheros]

    filas = []
    for nom in MAPAS:
        tops = {k: [] for k in TOPK}
        falta = False
        for ruta in ficheros:
            m = np.load(ruta)
            if nom not in m.files:
                falta = True
                break
            v = m[nom]
            ok = np.isfinite(v)
            n = int(ok.sum())
            orden = np.argsort(-v[ok], kind="stable")
            pos = np.empty(n, int)
            pos[orden] = np.arange(n)
            rango = np.full(len(v), n, int)
            rango[ok] = pos
            for k in TOPK:
                tops[k].append(rango < max(1, int(round(k * n))))
        if falta:
            continue
        for k in TOPK:
            T = np.array(tops[k])
            for dz in DESFASES:
                cap, ha_in, ha_tot = [], 0.0, 0.0
                for i, dia in enumerate(fechas):
                    if i < dz:
                        continue
                    alerta = T[i - dz]          # el mapa de hace `dz` días
                    for inc in por_dia.get(dia, []):
                        dentro = bool(alerta[inc["celdas"]].any())
                        cap.append((inc["clase"], inc["ha"], dentro))
                        ha_tot += inc["ha"]
                        if dentro:
                            ha_in += inc["ha"]
                if not cap:
                    continue
                d = pd.DataFrame(cap, columns=["clase", "ha", "dentro"])
                f = dict(mapa=nom, k=k, desfase=dz,
                         cobertura_ha=ha_in / ha_tot if ha_tot else np.nan,
                         pilla_todo=float(d.dentro.mean()), n=len(d))
                for c, _, _ in CLASES:
                    s = d[d.clase == c]
                    f[f"pilla_{c}"] = float(s.dentro.mean()) if len(s) else np.nan
                filas.append(f)
    df = pd.DataFrame(filas)
    df.to_csv(config.salida("dos_23_vispera.csv"), index=False)
    with open(config.salida("dos_23_vispera.json"), "w") as fh:
        json.dump(df.to_dict("records"), fh, indent=1, ensure_ascii=False)

    for k in TOPK:
        print(f"--- top {k*100:g} % del día · mapa de D−n contra el fuego de D")
        print(f"{'mapa':28s}" + "".join(f"{'D-'+str(d):>22s}" for d in DESFASES))
        print(f"{'':28s}" + "".join(f"{'ha / grandes':>22s}" for _ in DESFASES))
        for nom in MAPAS:
            g = df[(df.mapa == nom) & (df.k == k)].set_index("desfase")
            if not len(g):
                continue
            fila = f"{nom:28s}"
            for dz in DESFASES:
                r = g.loc[dz]
                fila += f"{r.cobertura_ha*100:14.1f}% /{r.pilla_GRANDE*100:4.0f}%"
            print(fila)
        print()


if __name__ == "__main__":
    main()
