#!/usr/bin/env python3
"""Figuras nuevas del anexo B (14/09/2026, tarde).

  anexo_b_captura.png   curva de captura de hectáreas frente al % de celdas
                        vigiladas, seis modelos, 2025 y 2026 (captura_lift.csv)
  anexo_c_variantes.png variantes del semáforo: % de días sin EXTREMO frente a
                        % de días grandes perdidos, cubo 2022-2024 y replay
                        (semaforo_variantes.csv)
Decimales con punto. Escribe en docs/MEMORIA/figs/.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = pathlib.Path(__file__).resolve().parent
OUT = AQUI.parents[1] / "docs/MEMORIA/figs"
ANCHO = 6.3
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
COL = {"prod": "#4D4D4D", "unico": "#0072B2", "r10": "#D55E00",
       "pareja": "#009E73", "donde": "#CC79A7", "cuando": "#E69F00"}
NOM = {"prod": "producción", "unico": "único 1:3", "r10": "r10",
       "pareja": "pareja", "donde": "dónde solo", "cuando": "cuándo solo"}


def captura():
    t = pd.read_csv(AQUI / "captura_lift.csv")
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 2.7), sharey=True)
    for ax, anio in zip(axs, (2025, 2026)):
        ax.plot([0, 20], [0, 20], color="0.6", lw=0.9, ls=":", label="azar")
        for m in NOM:
            s = t[(t.temporada == anio) & (t.modelo == m)].sort_values("top_pct")
            x = np.r_[0, s.top_pct.values]
            y = np.r_[0, s.captura_ha_total.values]
            ax.plot(x, y, "o-", color=COL[m], ms=2.8,
                    lw=2.0 if m == "r10" else 1.1, label=NOM[m],
                    zorder=5 if m == "r10" else 3)
        ax.axvline(2, color="0.75", lw=0.8)
        ax.axvline(10, color="0.75", lw=0.8)
        ax.set_xlim(0, 20.5)
        ax.set_ylim(0, 90)
        ax.set_xticks([0, 2, 5, 10, 15, 20])
        ax.set_xlabel("celdas vigiladas cada día (% de España)")
        ax.set_title(f"({'a' if anio == 2025 else 'b'}) temporada {anio}", loc="left")
    axs[0].set_ylabel("hectáreas quemadas dentro (%)")
    axs[1].legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.8,
                  handlelength=1.4)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "anexo_b_captura.png")
    plt.close(fig)
    print("escrito anexo_b_captura.png")


def variantes():
    t = pd.read_csv(AQUI / "semaforo_variantes.csv")
    etiqueta = {"V0": "todo el año", "V1": "fuera de jun.–sep.",
                "V2": "solo dic.–abr.", "V3": "umbral por régimen"}
    marca = {"V0": "o", "V1": "s", "V2": "D", "V3": "^"}
    paneles = [("cubo", "fuera de calibración 2022-2024", "(a) cubo 2022-2024, todo el año"),
               ("replay 2025", "IFS", "(b) replay 2025"),
               ("replay 2026", "IFS", "(c) replay 2026")]
    fig, axs = plt.subplots(1, 3, figsize=(ANCHO, 2.35), sharey=True)
    for ax, (fu, parte, tit) in zip(axs, paneles):
        s = t[(t.fuente == fu) & (t.parte == parte) & (t.regimen == "TOTAL")]
        for v in ("V0", "V1", "V2", "V3"):
            r = s[s.variante == v].iloc[0]
            ax.scatter(r.pct_dias_sin_extremo, r.pct_grandes_perdidos, s=34,
                       marker=marca[v], color="#D55E00" if v == "V0" else
                       ("#009E73" if v == "V2" else "#0072B2" if v == "V1" else "0.45"),
                       label=etiqueta[v], zorder=4)
        ax.set_xlim(-3, 70)
        ax.set_ylim(-3, 60)
        ax.set_title(tit, loc="left", fontsize=8)
        ax.set_xlabel("días sin EXTREMO (%)")
    axs[0].set_ylabel("días grandes sin EXTREMO (%)")
    axs[0].legend(loc="upper left", frameon=False, fontsize=6.8, handletextpad=0.3)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "anexo_c_variantes.png")
    plt.close(fig)
    print("escrito anexo_c_variantes.png")


if __name__ == "__main__":
    captura()
    variantes()
