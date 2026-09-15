#!/usr/bin/env python3
"""Métricas con tolerancia espacial (a posteriori, no prerregistradas): la verdad
del día se dilata a un radio de R km (celda positiva si hay celda quemada EFFIS
ese día a ≤R km, ventana cuadrada de 2R+1 celdas) y se recalcula, por modelo y
día, el AUC, la captura en el top-2 % (fracción de celdas positivas dilatadas
que caen en el top) y la precisión del top-2 % (fracción del top que es
positiva dilatada). Condición IFS. Radios 0 (exacta), 2, 4, 6 km.
Escribe replay/replay_radio.{csv,md}."""
import glob, os
import numpy as np, pandas as pd, xarray as xr
from scipy.ndimage import maximum_filter
AQUI = os.path.dirname(os.path.abspath(__file__))
MODELOS = ("prod", "unico", "r10", "pareja", "donde", "cuando")
RADIOS = (0, 2, 4, 6)

def auc_rangos(p, pos):
    """AUC Mann-Whitney con rangos (rápido para muchos positivos)."""
    r = pd.Series(p).rank().values
    n1 = pos.sum(); n0 = len(p) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def main():
    import sys; sys.path.insert(0, f"{AQUI}/sandbox_replay"); os.chdir(f"{AQUI}/sandbox_replay")
    import config
    es = xr.open_dataset(config.CUBO, decode_timedelta=False)["is_spain"].values.astype(bool)
    ny, nx = es.shape; sel = es.ravel(); n2 = int(sel.sum() * 0.02)
    filas = []
    for anio in (2025, 2026):
        for f in sorted(glob.glob(f"{AQUI}/replay/{anio}/ifs/*.npz")):
            d = os.path.basename(f)[:-4]
            V = np.load(f"{AQUI}/replay/verdad_{anio}/{d}.npz")
            if not len(V["celda"]): continue
            q = np.zeros(es.size, bool); q[V["celda"]] = True; q = q.reshape(ny, nx)
            m = np.load(f)
            P = {k: np.nan_to_num(m[f"prob_{k}"].astype(np.float32).ravel()[sel], nan=-1) for k in MODELOS}
            tops = {k: np.argsort(-P[k])[:n2] for k in MODELOS}
            fila = dict(fecha=d, anio=anio, celdas_quemadas=int(q.sum()), area_ha=float(V["area_ha"].sum()))
            for R in RADIOS:
                pos = (maximum_filter(q, size=2 * R + 1, mode="constant") if R else q).ravel()[sel]
                fila[f"n_pos_{R}"] = int(pos.sum())
                for k in MODELOS:
                    fila[f"auc_{k}_{R}"] = auc_rangos(P[k], pos)
                    fila[f"cap2_{k}_{R}"] = float(pos[tops[k]].sum() / pos.sum() * 100)
                    fila[f"prec2_{k}_{R}"] = float(pos[tops[k]].mean() * 100)
            filas.append(fila)
    df = pd.DataFrame(filas); df.to_csv(f"{AQUI}/replay/replay_radio.csv", index=False)
    L = ["# Tolerancia espacial (a posteriori): verdad dilatada a R km, condición IFS\n",
         "AUC medio por día · captura = % de celdas positivas (dilatadas) en el top-2 % · precisión = % del top-2 % que es positivo (dilatado)\n"]
    for anio in (2025, 2026):
        s = df[df["anio"] == anio]
        L.append(f"\n## {anio}: {len(s)} días con fuego · positivos por día (mediana): " + " · ".join(f"R={R} km: {int(s[f'n_pos_{R}'].median()):,}" for R in RADIOS) + "\n")
        for met, nom in (("auc", "AUC medio"), ("cap2", "captura top-2 % (%)"), ("prec2", "precisión top-2 % (%)")):
            L.append(f"\n### {nom}\n\n| modelo | " + " | ".join(f"R={R} km" for R in RADIOS) + " |\n|---|" + "---|" * len(RADIOS))
            for k in MODELOS:
                vals = [s[f"{met}_{k}_{R}"].mean() for R in RADIOS]
                L.append(f"| {k} | " + " | ".join((f"{v:.3f}" if met == "auc" else f"{v:.1f}") for v in vals) + " |")
    open(f"{AQUI}/replay/replay_radio.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
