#!/usr/bin/env python3
"""Anexo B: distribución del AUC por día de los seis modelos en las dos
temporadas del replay (condición IFS, días con fuego).
Entrada (solo lectura): ~/Desktop/Master/archivo_ifs/replay/replay_{2025,2026}_ifs.csv
(copias en docs/resultados/ si existen). Escribe docs/MEMORIA/figs/anexo_b_boxplot.png."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

M = pathlib.Path.home() / "Desktop/Master"
OUT = pathlib.Path(__file__).resolve().parents[1]
NARANJA, AZUL, BERMELLON, VERDE, GRIS, AZUL_CIELO = (
    "#E69F00", "#0072B2", "#D55E00", "#009E73", "#7F7F7F", "#56B4E9")
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
MODELOS = [("prod", "producción", GRIS), ("unico", "único 1:3", AZUL),
           ("r10", "r10", BERMELLON), ("pareja", "pareja", NARANJA),
           ("donde", "dónde solo", AZUL_CIELO), ("cuando", "cuándo solo", VERDE)]


def leer(anio):
    for ruta in (OUT.parents[1] / "resultados" / f"replay_{anio}_ifs.csv",
                 M / "archivo_ifs/replay" / f"replay_{anio}_ifs.csv"):
        if ruta.exists():
            d = pd.read_csv(ruta)
            return d[d["celdas_quemadas"] > 0]
    raise FileNotFoundError(anio)


fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
for ax, anio in zip(axs, (2025, 2026)):
    d = leer(anio)
    datos = [d[f"auc_{m}"].dropna().values for m, _, _ in MODELOS]
    bp = ax.boxplot(datos, widths=0.55, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", ms=2.5, mec="0.5", alpha=0.6),
                    medianprops=dict(color="k", lw=1.2),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for caja, (_, _, col) in zip(bp["boxes"], MODELOS):
        caja.set(facecolor=col, alpha=0.55, edgecolor=col, lw=0.8)
    for i, v in enumerate(datos, 1):
        ax.text(i, 0.985, f"{np.mean(v):.3f}", ha="center", va="top", fontsize=6.8, color="0.25")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(1, 7), [e for _, e, _ in MODELOS], rotation=30, ha="right")
    ax.set_ylim(0.2, 1.0)
    ax.set_title(f"({'ab'[anio-2025]}) {anio} · {len(d)} días con fuego", loc="left", fontsize=8.5)
axs[0].set_ylabel("AUC dentro del día")
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT / "anexo_b_boxplot.png")
print("escrito", OUT / "anexo_b_boxplot.png")
