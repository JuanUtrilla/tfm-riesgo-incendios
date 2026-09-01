#!/usr/bin/env python3
"""F2 - Los tres disenios de muestreo sobre la misma rejilla celda x dia.

Esquema, no datos: reproduce la logica de
`05_iteracion2/52_muestreo/dos_12_muestrear_effis.py:97-134`
(negativos `cuando`, negativos `donde`, banco `eval_dia`).
Uso:  python f2_disenios.py
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from comun import ANCHO, AZUL, BERMELLON, GRIS, NEGRO, guarda

NY, NX = 9, 11          # celdas x dias
PY_, PX = 4, 5          # el positivo


def rejilla(ax, marcados, titulo):
    for iy in range(NY):
        for ix in range(NX):
            ax.add_patch(plt.Rectangle((ix, NY - 1 - iy), 0.86, 0.86,
                                       facecolor="#EDEDED", edgecolor="white",
                                       linewidth=0.6))
    for (iy, ix), (color, marca) in marcados.items():
        ax.add_patch(plt.Rectangle((ix, NY - 1 - iy), 0.86, 0.86,
                                   facecolor=color, edgecolor="white",
                                   linewidth=0.6))
        if marca:
            ax.text(ix + 0.43, NY - 1 - iy + 0.43, marca, ha="center",
                    va="center", fontsize=7, color="white", fontweight="bold")
    ax.set_xlim(-0.2, NX); ax.set_ylim(-0.2, NY)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_aspect("equal")
    for lado in ("left", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.set_title(titulo, fontsize=8.2, pad=5)


def main():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(ANCHO, 2.1))

    pos = {(PY_, PX): (BERMELLON, "+")}

    # a) caso-control con celda fija: misma fila, otros dias
    a = dict(pos)
    for ix in rng.choice([i for i in range(NX) if i != PX], 3, replace=False):
        a[(PY_, ix)] = (AZUL, "−")
    rejilla(axes[0], a, "a) caso-control, celda fija")

    # b) disenio donde: misma columna, otras celdas
    b = dict(pos)
    for iy in rng.choice([i for i in range(NY) if i != PY_], 3, replace=False):
        b[(iy, PX)] = (AZUL, "−")
    rejilla(axes[1], b, "b) diseño dónde, día fijo")

    # c) banco de evaluacion: la columna entera
    c = dict(pos)
    c[(PY_ + 1, PX)] = (BERMELLON, "+")
    for iy in range(NY):
        if (iy, PX) not in c:
            c[(iy, PX)] = (GRIS, "?")
    rejilla(axes[2], c, "c) banco de evaluación")

    for ax in axes:
        ax.set_xlabel("día  →", fontsize=7.5, labelpad=1)
    axes[0].set_ylabel("celda  →", fontsize=7.5, labelpad=1)

    leyenda = [Line2D([], [], marker="s", linestyle="", markersize=7,
                      color=BERMELLON, label="positivo (celda quemada, primer día)"),
               Line2D([], [], marker="s", linestyle="", markersize=7,
                      color=AZUL, label="negativo sorteado"),
               Line2D([], [], marker="s", linestyle="", markersize=7,
                      color=GRIS, label="celda del día a ordenar")]
    fig.legend(handles=leyenda, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=7.5)
    fig.subplots_adjust(wspace=0.18)
    guarda(fig, "f2_disenios.png")


if __name__ == "__main__":
    main()
