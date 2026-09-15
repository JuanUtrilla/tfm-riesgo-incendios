#!/usr/bin/env python3
"""F2 - Los tres diseños de muestreo sobre la misma rejilla celda x día.

Esquema sin datos: reproduce la lógica de
`05_iteracion2/52_muestreo/dos_12_muestrear_effis.py:97-134`
(negativos `cuando`, negativos `donde`, banco `eval_dia`).
Rejilla 4 x 4 a propósito (pedido del 09/09/2026: pequeña y legible).
Uso:  python f2_disenios.py
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from comun import ANCHO, AZUL, BERMELLON, GRIS, guarda

NY, NX = 4, 4           # celdas x dias
PY_, PX = 1, 2          # el positivo


def rejilla(ax, marcados, titulo):
    for iy in range(NY):
        for ix in range(NX):
            ax.add_patch(plt.Rectangle((ix, NY - 1 - iy), 0.9, 0.9,
                                       facecolor="#EDEDED", edgecolor="white",
                                       linewidth=1.0))
    for (iy, ix), (color, marca) in marcados.items():
        ax.add_patch(plt.Rectangle((ix, NY - 1 - iy), 0.9, 0.9,
                                   facecolor=color, edgecolor="white",
                                   linewidth=1.0))
        ax.text(ix + 0.45, NY - 1 - iy + 0.45, marca, ha="center",
                va="center", fontsize=10, color="white", fontweight="bold")
    ax.set_xlim(-0.15, NX); ax.set_ylim(-0.15, NY)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_aspect("equal")
    for lado in ("left", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.set_title(titulo, fontsize=8.5, pad=4)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(ANCHO * 0.8, 2.25))
    pos = {(PY_, PX): (BERMELLON, "+")}

    # a) iteración 1: la misma celda, otros días (fila)
    a = dict(pos)
    for ix in (0, 3):
        a[(PY_, ix)] = (AZUL, "−")
    rejilla(axes[0], a, "a) iteración 1\nmisma celda,\notros días")

    # b) iteración 2: otras celdas, el mismo día (columna)
    b = dict(pos)
    for iy in (0, 3):
        b[(iy, PX)] = (AZUL, "−")
    rejilla(axes[1], b, "b) iteración 2\notras celdas,\nel mismo día")

    # c) lo que se pide en operación: ordenar la columna entera
    c = dict(pos)
    for iy in range(NY):
        if (iy, PX) not in c:
            c[(iy, PX)] = (GRIS, "?")
    rejilla(axes[2], c, "c) en operación\ntodas las celdas\nde hoy")

    for ax in axes:
        ax.set_xlabel("día  →", fontsize=8, labelpad=1)
    axes[0].set_ylabel("celda  →", fontsize=8, labelpad=1)

    leyenda = [Line2D([], [], marker="s", linestyle="", markersize=7,
                      color=BERMELLON, label="celda quemada (positivo)"),
               Line2D([], [], marker="s", linestyle="", markersize=7,
                      color=AZUL, label="negativo sorteado"),
               Line2D([], [], marker="s", linestyle="", markersize=7,
                      color=GRIS, label="celda a ordenar")]
    fig.legend(handles=leyenda, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
    fig.subplots_adjust(wspace=0.35, bottom=0.18, top=0.78)
    guarda(fig, "f2_disenios.png")


if __name__ == "__main__":
    main()
