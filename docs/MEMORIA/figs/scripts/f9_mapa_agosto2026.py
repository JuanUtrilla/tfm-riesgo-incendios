#!/usr/bin/env python3
"""F9 - El mismo día según producción y según el r10, y la resta entre ambos.

Día: 7 de agosto de 2026 (44.672 ha, el segundo mayor de la temporada). Mapas
del replay con la pasada IFS de la víspera (`replay/2026/ifs/` del archivo de previsiones IFS),
pintados en percentil del día como hace `07_produccion/dos_riesgo_hoy.py`; las
celdas EFFIS de primer día (`replay/verdad_2026/`) en negro.
Uso:  python f9_mapa_agosto2026.py   [TFM_ARCHIVO=<ruta al archivo de previsiones IFS>]
"""
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from comun import ANCHO, ARCHIVO, RAIZ, guarda

LIMITES = RAIZ / "muestras" / "limites.npz"
FECHA = "2026-08-07"
# los cortes y colores de dos_riesgo_hoy.py
CORTES = [0, 10, 30, 60, 90, 98, 100]
COLORES = ["#ffffd9", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#bd0026"]


def percentil(p):
    m = ~np.isnan(p)
    r = np.full(p.shape, np.nan)
    v = p[m]
    r[m] = v.argsort().argsort() / (v.size - 1) * 100
    return r


def limites(ax, lw=0.5):
    L = np.load(LIMITES)
    ax.plot(L["prov_x"], L["prov_y"], lw=0.3 * lw, color="0.35", alpha=0.55)
    ax.plot(L["ccaa_x"], L["ccaa_y"], lw=lw, color="0.15", alpha=0.85)


def main():
    d = np.load(ARCHIVO / "replay" / "2026" / "ifs" / f"{FECHA}.npz")
    prod = percentil(d["prob_prod"].astype(float))
    r10 = percentil(d["prob_r10"].astype(float))
    v = np.load(ARCHIVO / "replay" / f"verdad_{FECHA[:4]}" / f"{FECHA}.npz")
    fy, fx = np.unravel_index(v["celda"], prod.shape)

    fig, axs = plt.subplots(1, 3, figsize=(ANCHO, ANCHO * 0.36))
    cmap = ListedColormap(COLORES)
    norm = BoundaryNorm(CORTES, cmap.N)
    for ax, m, t in zip(axs[:2], (prod, r10), ("Producción (iteración 1)", "r10 (iteración 2)")):
        im = ax.imshow(m, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        limites(ax)
        ax.plot(fx, fy, ls="", marker="o", ms=1.6, mfc="none", mec="black", mew=0.5)
        ax.set_title(t, fontsize=7)
        ax.set_axis_off()
    dif = r10 - prod
    imd = axs[2].imshow(dif, origin="upper", cmap="RdBu_r", vmin=-40, vmax=40, interpolation="nearest")
    limites(axs[2])
    axs[2].plot(fx, fy, ls="", marker="o", ms=1.6, mfc="none", mec="black", mew=0.5)
    axs[2].set_title("r10 − producción", fontsize=7)
    axs[2].set_axis_off()
    cb = fig.colorbar(im, ax=axs[:2], orientation="horizontal", fraction=0.05, pad=0.02,
                      boundaries=CORTES, ticks=[10, 30, 60, 90, 98], spacing="uniform")
    cb.set_label("percentil del día", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    cbd = fig.colorbar(imd, ax=axs[2], orientation="horizontal", fraction=0.05, pad=0.02, extend="both")
    cbd.set_label("puntos de percentil", fontsize=6)
    cbd.ax.tick_params(labelsize=5)
    guarda(fig, "f9_mapa_agosto2026")


if __name__ == "__main__":
    main()
