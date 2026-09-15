#!/usr/bin/env python3
"""Figura 17: la malla de servicio, 5,605 nodos de ERA5-Land sobre las celdas de 1 km.

  (a) España entera: cada celda de 1 km toma la meteorología de su nodo de
      ERA5-Land (0.1°, unos 9 km). Las celdas se pintan en damero según la
      paridad del nodo, de modo que cada cuadro es el área que sirve un nodo.
  (b) Detalle alrededor de Madrid con el nodo marcado en el centro de sus
      celdas. No hay interpolación ni huecos: la celda hereda el valor del nodo.

Fuente: muestras/nodos.npz (malla_01_nodos.py) y muestras/limites.npz.
Figura para el README del capítulo 1, no para la memoria.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from comun import ANCHO, AZUL, BERMELLON, REPO, guarda

N = np.load(REPO / "muestras" / "nodos.npz")
L = np.load(REPO / "muestras" / "limites.npz")
TONOS = ListedColormap(["#c6dbef", "#9ecae1"])


def main():
    idx = N["idx_nodo"]
    lat, lon = N["nodo_lat"], N["nodo_lon"]
    paridad = (np.rint(lat * 10) + np.rint(lon * 10)).astype(int) % 2
    m = np.where(idx >= 0, paridad[np.clip(idx, 0, None)], np.nan).astype(float)

    filas, cols = np.nonzero(idx >= 0)
    ids = idx[filas, cols]
    cuenta = np.bincount(ids, minlength=lat.size)
    cy = np.bincount(ids, filas, lat.size) / np.maximum(cuenta, 1)
    cx = np.bincount(ids, cols, lat.size) / np.maximum(cuenta, 1)

    k = list(L["cap_nombre"]).index("Madrid")
    mx, my, r = L["cap_x"][k], L["cap_y"][k], 45

    fig, (a, b) = plt.subplots(1, 2, figsize=(ANCHO, 3.0),
                               gridspec_kw={"width_ratios": [1.45, 1]})
    for ax in (a, b):
        ax.imshow(m, origin="upper", cmap=TONOS, interpolation="nearest")
        ax.plot(L["ccaa_x"], L["ccaa_y"], lw=0.45, color="0.2", alpha=0.85)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    a.add_patch(Rectangle((mx - r, my - r), 2 * r, 2 * r, fill=False,
                          ec=BERMELLON, lw=1.0))
    a.set_title(f"(a) {lat.size:,} nodos; cada cuadro, el área de uno", loc="left")

    dentro = (abs(cx - mx) < r) & (abs(cy - my) < r) & (cuenta > 0)
    b.plot(cx[dentro], cy[dentro], "o", ms=2.2, color=AZUL)
    b.plot(L["prov_x"], L["prov_y"], lw=0.3, color="0.35", alpha=0.6)
    b.plot(mx, my, "*", ms=7, color=BERMELLON)
    b.text(mx + 2, my - 2, "Madrid", fontsize=7.5)
    b.set_xlim(mx - r, mx + r)
    b.set_ylim(my + r, my - r)
    b.set_title("(b) detalle de 90 × 90 km", loc="left")

    guarda(fig, "f17_malla_nodos.png")


if __name__ == "__main__":
    main()
