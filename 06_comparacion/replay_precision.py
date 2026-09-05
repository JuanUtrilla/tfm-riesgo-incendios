#!/usr/bin/env python3
"""Precisión de la punta: de las celdas que el modelo ordena más arriba cada
día (top 0,5 / 1 / 2 / 5 %), cuántas ardieron ESE día (EFFIS, exactas) y cuántas
están a ≤10 km de una celda quemada ese día. Días con y sin fuego (en un día sin
fuego la precisión es 0 por definición y cuenta). Condición IFS.
Escribe replay/replay_precision.{csv,md}."""
import glob, os
import numpy as np, pandas as pd, xarray as xr
from scipy.ndimage import maximum_filter
AQUI = os.path.dirname(os.path.abspath(__file__))
MODELOS = ("prod", "unico", "r10", "pareja", "donde", "cuando")
KS = (0.005, 0.01, 0.02, 0.05)

def main():
    import sys; sys.path.insert(0, f"{AQUI}/sandbox_replay"); os.chdir(f"{AQUI}/sandbox_replay")
    import config
    es = xr.open_dataset(config.CUBO, decode_timedelta=False)["is_spain"].values.astype(bool)
    ny, nx = es.shape; sel = es.ravel(); n_esp = int(sel.sum())
    filas = []
    for anio in (2025, 2026):
        for f in sorted(glob.glob(f"{AQUI}/replay/{anio}/ifs/*.npz")):
            d = os.path.basename(f)[:-4]
            V = np.load(f"{AQUI}/replay/verdad_{anio}/{d}.npz")
            quem = np.zeros(es.size, bool); quem[V["celda"]] = True
            cerca = maximum_filter(quem.reshape(ny, nx), size=21, mode="constant").ravel() & sel
            m = np.load(f)
            fila = dict(fecha=d, anio=anio, celdas_quemadas=int(quem.sum()), n_inc=int(len(V["area_ha"])),
                        area_ha=float(V["area_ha"].sum()))
            for k in MODELOS:
                p = m[f"prob_{k}"].astype(np.float32).ravel(); p[~sel] = -1; p = np.nan_to_num(p, nan=-1)
                orden = np.argsort(-p)
                for q in KS:
                    top = orden[:int(n_esp * q)]
                    fila[f"prec_{k}_{q}"] = float(quem[top].mean() * 100)
                    fila[f"prec10km_{k}_{q}"] = float(cerca[top].mean() * 100)
            filas.append(fila)
    df = pd.DataFrame(filas); df.to_csv(f"{AQUI}/replay/replay_precision.csv", index=False)
    L = ["# Precisión de la punta (condición IFS): % de celdas del top-k que ardieron ese día\n"]
    for anio in (2025, 2026):
        s = df[df["anio"] == anio]
        base = s["celdas_quemadas"].sum() / (n_esp * len(s)) * 100
        L.append(f"\n## {anio}: {len(s)} días, {int((s['celdas_quemadas']>0).sum())} con fuego · tasa base {base:.4f} % de celdas-día quemadas\n")
        L.append("| modelo | " + " | ".join(f"top {q*100:g} % exacta" for q in KS) + " | " + " | ".join(f"top {q*100:g} % ≤10 km" for q in KS) + " | lift top 2 % |\n|---|" + "---|" * (2 * len(KS) + 1))
        for k in MODELOS:
            ex = [s[f"prec_{k}_{q}"].mean() for q in KS]; ce = [s[f"prec10km_{k}_{q}"].mean() for q in KS]
            L.append(f"| {k} | " + " | ".join(f"{v:.3f} %" for v in ex) + " | " + " | ".join(f"{v:.1f} %" for v in ce) + f" | ×{ex[2]/base:.1f} |")
        L.append(f"\nDías en que al menos una celda del top 2 % ardió: " + ", ".join(f"{k} {int((s[f'prec_{k}_0.02']>0).sum())}" for k in MODELOS) + f" de {int((s['celdas_quemadas']>0).sum())} días con fuego.")
    open(f"{AQUI}/replay/replay_precision.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
