#!/usr/bin/env python3
"""Figura 3 de la memoria, versión sin semáforo (14/09/2026).

Los mismos dos días del replay de 2025 con el r10 que la figura original
(13 de agosto y 29 de octubre). Izquierda: niveles por percentil del día
(p30/p90/p98). Derecha: escala absoluta (MODERADO >= 0.0018, ALTO >= 0.0736,
EXTREMO; cortes calibrados sobre los mapas servidos, escala_servicio.json) con la
cifra del día: % de España en EXTREMO y su percentil entre los 250 días servidos
del replay 2025-2026. Sin puerta: nada se apaga.

Entradas (solo lectura): archivo_ifs/replay/2025/ifs/<fecha>.npz,
tfm-riesgo-incendios/muestras/limites.npz, calibracion_si/sandbox/salida/dos_27_dias.csv
Salida: ~/Downloads/figs/fig3_dos_mapas.png
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

M = pathlib.Path.home() / "Desktop/Master"
OUT = pathlib.Path.home() / "Downloads/figs/fig3_dos_mapas.png"
DIAS = ["2025-08-13", "2025-10-29"]
import json as _json
_ESC = _json.load(open(pathlib.Path(__file__).resolve().parent / "escala_servicio.json"))
CORTES_ABS = [_ESC["cortes"][k] for k in ("MODERADO", "ALTO", "EXTREMO")]
COLORES = ["#ffffd9", "#fed976", "#fd8d3c", "#bd0026"]
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]
N_CELDAS = 498530
MESES = {"08": "agosto", "10": "octubre"}
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 7})


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
    L = np.load(M / "tfm-riesgo-incendios/muestras/limites.npz")
    ax.plot(L["prov_x"], L["prov_y"], lw=0.3 * lw, color="0.35", alpha=0.55)
    ax.plot(L["ccaa_x"], L["ccaa_y"], lw=lw, color="0.15", alpha=0.85)


def main():
    pct_hist = np.array(list(_ESC["referencia_pct_extremo"]["valores"].values()))
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.3 * 0.74))
    fig.subplots_adjust(wspace=0.12, hspace=0.30)
    cmap = ListedColormap(COLORES)
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    im_ref = None
    for fila, fecha in zip(axs, DIAS):
        r10 = np.load(M / f"archivo_ifs/replay/2025/ifs/{fecha}.npz")["prob_r10"].astype(float)
        m = ~np.isnan(r10)
        dia = f"{int(fecha[8:10])} de {MESES[fecha[5:7]]} de {fecha[:4]}"
        ext = float(np.mean(nivel_abs(r10)[m] == 3) * 100)
        pos = float(np.mean(pct_hist <= ext) * 100)
        subt = ("EXTREMO: 2.0 % de España\n(fijo por construcción)",
                f"EXTREMO: {ext:.1f} % de España\n(más que el {pos:.0f} % de los días de referencia)")
        for ax, niv, t, s in zip(fila, (nivel_pctl(r10), nivel_abs(r10)),
                                 ("percentil del día", "escala absoluta"), subt):
            im = ax.imshow(niv, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
            im_ref = im_ref or im
            limites(ax)
            # el texto va en el título, nunca encima del mapa
            ax.set_title(f"{dia} · {t}\n{s}", fontsize=5.8, linespacing=1.25)
            ax.set_axis_off()
        print(f"{fecha}: EXTREMO absoluto {ext:.2f} % · percentil histórico {pos:.1f}")
    cb = fig.colorbar(im_ref, ax=axs, orientation="horizontal", fraction=0.035, pad=0.02,
                      boundaries=[0, 1, 2, 3, 4], ticks=[0.5, 1.5, 2.5, 3.5])
    cb.ax.set_xticklabels(NIVELES, fontsize=6)
    fig.savefig(OUT)
    plt.close(fig)
    print("escrito", OUT)


if __name__ == "__main__":
    main()
