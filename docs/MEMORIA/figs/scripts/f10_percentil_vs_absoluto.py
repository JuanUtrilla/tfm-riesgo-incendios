#!/usr/bin/env python3
"""F10 - Percentil del día frente a escala absoluta: verano y fuera de temporada.

Dos días del replay de 2025 con el r10 (`archivo_ifs/replay/2025/ifs/`): el
13 de agosto (pleno verano) y el 29 de octubre (fuera de temporada). Columna
izquierda: niveles por percentil del día (p30/p90/p98, el mapa «dónde mirar»
de la cadena). Columna derecha: los
cortes absolutos del cubo, de
`05_iteracion2/56_calibracion/dos_27_escala_absoluta.py` (MODERADO >= 0,0018,
ALTO >= 0,0736, EXTREMO >= 0,4066), y el semáforo del día, que se enciende si el
p98 de la pareja supera 0,076 (umbral de invierno-primavera de dos_26 en
escala cruda; el de verano es 0,064). Es la versión con semáforo; el semáforo
se retiró el 14/09/2026 y la versión sin él, con los cortes calibrados sobre
los mapas servidos, es `07_produccion/escala_y_cifra/fig3_dos_mapas.py`.
Uso:  python f10_percentil_vs_absoluto.py   [TFM_ARCHIVO=<ruta a archivo_ifs>]
"""
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from comun import ANCHO, RAIZ, guarda

ARCHIVO = pathlib.Path(os.environ.get("TFM_ARCHIVO", RAIZ / "archivo_ifs"))
LIMITES = RAIZ / "tfm-riesgo-incendios" / "muestras" / "limites.npz"
DIAS = ["2025-08-13", "2025-10-29"]
CORTES_ABS = [0.0018, 0.0736, 0.4066]          # r10, dos_27_bandas.csv
UMBRAL_SI = 0.076                               # pareja, p98 del dia
COLORES = ["#ffffd9", "#fed976", "#fd8d3c", "#bd0026"]
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]


def nivel_pctl(p):
    m = ~np.isnan(p)
    v = p[m]
    pct = v.argsort().argsort() / (v.size - 1) * 100
    n = np.full(p.shape, np.nan)
    n[m] = np.digitize(pct, [30, 90, 98])
    return n


def nivel_abs(p):
    n = np.full(p.shape, np.nan)
    m = ~np.isnan(p)
    n[m] = np.digitize(p[m], CORTES_ABS)
    return n


def limites(ax, lw=0.5):
    L = np.load(LIMITES)
    ax.plot(L["prov_x"], L["prov_y"], lw=0.3 * lw, color="0.35", alpha=0.55)
    ax.plot(L["ccaa_x"], L["ccaa_y"], lw=lw, color="0.15", alpha=0.85)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(ANCHO, ANCHO * 0.74))
    fig.subplots_adjust(wspace=0.02, hspace=0.2)
    cmap = ListedColormap(COLORES)
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    im_ref = None
    for fila, fecha in zip(axs, DIAS):
        d = np.load(ARCHIVO / "replay" / "2025" / "ifs" / f"{fecha}.npz")
        r10 = d["prob_r10"].astype(float)
        pareja = d["prob_pareja"].astype(float)
        p98 = np.nanpercentile(pareja, 98)
        aviso = p98 >= UMBRAL_SI
        m = ~np.isnan(r10)
        for ax, niv, t in zip(fila, (nivel_pctl(r10), nivel_abs(r10)),
                              ("percentil del día", "escala absoluta")):
            rojo = np.nanmean(niv[m] == 3) * 100
            im = ax.imshow(niv, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
            im_ref = im_ref or im
            limites(ax)
            dia = fecha[8:10].lstrip("0") + {"08": " de agosto", "10": " de octubre"}[fecha[5:7]] + " de " + fecha[:4]
            ax.set_title(f"{dia} · {t}\nEXTREMO: {rojo:.1f} % de las celdas", fontsize=6.5)
            ax.set_axis_off()
        ax_abs = fila[1]
        txt = ("semáforo: AVISO" if aviso else "semáforo: sin aviso → mapa apagado")
        ax_abs.text(0.99, 0.02, f"{txt}\n(p98 pareja {p98:.3f} {'≥' if aviso else '<'} {UMBRAL_SI})",
                    transform=ax_abs.transAxes, fontsize=5.5, va="bottom", ha="right",
                    bbox=dict(fc="white", ec="0.5", lw=0.4, pad=2))
        if not aviso:
            ax_abs.images[0].set_alpha(0.25)
    cb = fig.colorbar(im_ref, ax=axs, orientation="horizontal", fraction=0.035, pad=0.02,
                      boundaries=[0, 1, 2, 3, 4], ticks=[0.5, 1.5, 2.5, 3.5])
    cb.ax.set_xticklabels(NIVELES, fontsize=6)
    guarda(fig, "f10_percentil_vs_absoluto")


if __name__ == "__main__":
    main()
