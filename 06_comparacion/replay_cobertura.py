#!/usr/bin/env python3
"""Cobertura con tolerancia (a posteriori): se dilata el top del modelo (top-2 %
y top-0,5 %) a R km y se mide qué fracción de las celdas quemadas y de los
incendios del día quedan a ≤R km de alguna celda del top. Se acompaña del % de
España que cubre ese top dilatado (el azar cubriría esa misma fracción de las
celdas quemadas). Condición IFS. Radios 0, 2, 4, 6 km.
Escribe replay/replay_cobertura.{csv,md}."""
import glob, os
import numpy as np, pandas as pd, xarray as xr
from scipy.ndimage import maximum_filter
AQUI = os.path.dirname(os.path.abspath(__file__))
MODELOS = ("prod", "unico", "r10", "pareja", "donde", "cuando")
RADIOS = (0, 2, 4, 6); TOPS = (0.005, 0.02)

def main():
    import sys; sys.path.insert(0, f"{AQUI}/sandbox_replay"); os.chdir(f"{AQUI}/sandbox_replay")
    import config
    es = xr.open_dataset(config.CUBO, decode_timedelta=False)["is_spain"].values.astype(bool)
    ny, nx = es.shape; sel = es.ravel(); N = int(sel.sum())
    filas = []
    for anio in (2025, 2026):
        for f in sorted(glob.glob(f"{AQUI}/replay/{anio}/ifs/*.npz")):
            d = os.path.basename(f)[:-4]
            V = np.load(f"{AQUI}/replay/verdad_{anio}/{d}.npz")
            if not len(V["celda"]): continue
            q, fu, nf = V["celda"], V["fuego"], len(V["area_ha"])
            m = np.load(f)
            fila = dict(fecha=d, anio=anio, celdas_quemadas=int(len(q)), n_inc=nf, area_ha=float(V["area_ha"].sum()))
            for k in MODELOS:
                p = np.nan_to_num(m[f"prob_{k}"].astype(np.float32).ravel(), nan=-1); p[~sel] = -1
                orden = np.argsort(-p)
                for t in TOPS:
                    top = np.zeros(es.size, bool); top[orden[:int(N * t)]] = True; top = top.reshape(ny, nx)
                    for R in RADIOS:
                        cov = (maximum_filter(top, size=2 * R + 1, mode="constant") if R else top).ravel()
                        hit = cov[q]
                        fila[f"celdas_{k}_{t}_{R}"] = float(hit.mean() * 100)
                        fila[f"inc_{k}_{t}_{R}"] = float(np.mean([hit[fu == i].any() for i in range(nf)]) * 100)
                        fila[f"esp_{k}_{t}_{R}"] = float((cov & sel).sum() / N * 100)
            filas.append(fila)
    df = pd.DataFrame(filas); df.to_csv(f"{AQUI}/replay/replay_cobertura.csv", index=False)
    L = ["# Cobertura con tolerancia (a posteriori): top del modelo dilatado a R km, condición IFS\n",
         "celdas = % de celdas quemadas a ≤R km de alguna celda del top · incendios = % de incendios del día con alguna celda a ≤R km del top · España = % del territorio que cubre el top dilatado (lo que acertaría el azar)\n"]
    for anio in (2025, 2026):
        s = df[df["anio"] == anio]
        for t in TOPS:
            L.append(f"\n## {anio} · top {t*100:g} % ({int(N*t):,} celdas) · {len(s)} días con fuego\n")
            L.append("| modelo | " + " | ".join(f"R={R}: celdas / incendios / España" for R in RADIOS) + " |\n|---|" + "---|" * len(RADIOS))
            for k in MODELOS:
                L.append(f"| {k} | " + " | ".join(f"{s[f'celdas_{k}_{t}_{R}'].mean():.0f} / {s[f'inc_{k}_{t}_{R}'].mean():.0f} / {s[f'esp_{k}_{t}_{R}'].mean():.1f} %" for R in RADIOS) + " |")
    open(f"{AQUI}/replay/replay_cobertura.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
