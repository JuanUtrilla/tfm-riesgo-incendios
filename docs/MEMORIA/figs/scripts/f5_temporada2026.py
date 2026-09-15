#!/usr/bin/env python3
"""F5 - La temporada 2026 día a día: AUC dentro del día y ventaja acumulada.

Reproduce la lógica de `06_comparacion/dos_19_veredicto.py` (bootstrap de días,
no de celdas; N=2000, SEED=42) sobre `salida/dos_09_temporada2026.csv` (74 días,
ventana FIRMS de 7 días, corrida limpia del 31/08/2026). No reentrena ningún
modelo.
Uso:  python f5_temporada2026.py   [TFM_MALLA=<ruta a TFM_fuego_malla>]
"""
import os
import pathlib

import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from comun import ANCHO, AZUL, BERMELLON, GRIS, NEGRO, MALLA, guarda

SAL = pathlib.Path(os.environ.get("TFM_MALLA", MALLA)) / "salida"
CSV = SAL / "dos_09_temporada2026.csv"
VER = SAL / "dos_19_veredicto.json"
N_BOOT, SEED = 2000, 42
REF = "auc_prod"
CAND = "auc_donde_dia_effis_r10"


def main():
    df = pd.read_csv(CSV, parse_dates=["fecha"]).sort_values("fecha")
    d = (df[CAND] - df[REF]).values
    n = len(d)
    rng = np.random.default_rng(SEED)
    med = np.array([d[:k + 1].mean() for k in range(n)])
    lo = np.empty(n); hi = np.empty(n)
    for k in range(n):
        m = d[:k + 1]
        bb = m[rng.integers(0, k + 1, (N_BOOT, k + 1))].mean(1)
        lo[k], hi[k] = np.percentile(bb, [2.5, 97.5])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(ANCHO, 4.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.25, 1]))
    x = df["fecha"].values
    a1.plot(x, df[REF], color=GRIS, linewidth=0.7, alpha=0.75)
    a1.plot(x, df[CAND], color=BERMELLON, linewidth=0.7, alpha=0.45)
    a1.plot(x, df[REF].expanding().mean(), color=NEGRO, linewidth=1.8,
            label=f"producción (iteración 1) · media final {df[REF].mean():.3f}"
                  .replace(".", ","))
    a1.plot(x, df[CAND].expanding().mean(), color=BERMELLON, linewidth=1.8,
            label=f"único, negativos 1:10 · media final {df[CAND].mean():.3f}"
                  .replace(".", ","))
    a1.axhline(0.5, color=GRIS, linewidth=0.7, linestyle=(0, (4, 3)))
    a1.set_ylabel("AUC dentro del día")
    a1.set_ylim(0.25, 1.02)
    a1.legend(loc="lower right", fontsize=7.4, facecolor="white",
              framealpha=0.9, edgecolor="none")

    a2.fill_between(x, lo, hi, color=BERMELLON, alpha=0.22, linewidth=0)
    a2.plot(x, med, color=BERMELLON, linewidth=1.6)
    a2.axhline(0, color=NEGRO, linewidth=0.8)
    a2.set_ylabel("Δ AUC acumulada\nfrente a producción")
    a2.set_xlabel("día de la temporada 2026 (1-jun a 15-ago)")
    v = json.load(open(VER))["único EFFIS 1:10"]
    a2.annotate(f"Δ {v['dif_final']:+.3f}  IC95 [{v['ic95_final'][0]:+.3f}, "
                f"{v['ic95_final'][1]:+.3f}]".replace(".", ","),
                xy=(x[-1], med[-1]), xytext=(-6, 16), textcoords="offset points",
                ha="right", fontsize=7.6, color=NEGRO)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.align_ylabels([a1, a2])
    fig.subplots_adjust(hspace=0.12)
    guarda(fig, "f5_temporada2026.png")
    print(f"n_dias {n} · Δ final {med[-1]:.4f} [{lo[-1]:.4f}, {hi[-1]:.4f}] · "
          f"gana {(d > 0).mean():.3f}")


if __name__ == "__main__":
    main()
