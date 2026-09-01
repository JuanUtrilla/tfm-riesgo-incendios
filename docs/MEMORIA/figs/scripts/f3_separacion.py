#!/usr/bin/env python3
"""F3 - Que separa un positivo de un negativo en el conjunto de la iteracion 1.

Lee las medianas ya calculadas por `02_eda/eda_dataset.py` y guardadas en
`02_eda/figuras/eda_resumen.log` (53.563 filas: 13.577 positivos y 39.986
negativos del split de train). No recalcula nada.
Uso:  python f3_separacion.py
"""
import re

import numpy as np

import matplotlib.pyplot as plt

from comun import ANCHO, AZUL, BERMELLON, GRIS, NEGRO, REPO, guarda

LOG = REPO / "02_eda" / "figuras" / "eda_resumen.log"
ETIQ = {
    "fwi_pctl_local": "FWI en percentil local",
    "fwi": "FWI absoluto",
    "dias_sin_lluvia": "días sin lluvia",
    "precip_30d": "precipitación 30 d (mm)",
    "rh_min": "humedad relativa mínima (%)",
    "vpd_max": "déficit de presión de vapor (kPa)",
    "t2m_max": "temperatura máxima (°C)",
    "viento_max": "viento máximo (m/s)",
    "ndvi": "NDVI",
    "n_fuegos_10km_365d": "fuegos a 10 km, 365 d",
    "popdens": "densidad de población",
    "dist_carreteras": "distancia a carreteras (km)",
}


def medianas():
    txt = LOG.read_text(encoding="utf-8", errors="replace")
    out = {}
    for m in re.finditer(r"^\s{2}(\w+)\s+pos=\s*([\d.]+)\s+neg=\s*([\d.]+)\s*$",
                         txt, re.M):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def main():
    med = medianas()
    nombres = [n for n in ETIQ if n in med]
    # separacion relativa: |pos-neg| / max(|pos|,|neg|,eps), solo para ordenar
    def sep(n):
        p, q = med[n]
        return abs(p - q) / max(abs(p), abs(q), 1e-9)
    nombres.sort(key=sep)

    fig, ax = plt.subplots(figsize=(ANCHO, 3.1))
    y = np.arange(len(nombres))
    for i, n in enumerate(nombres):
        p, q = med[n]
        # escala relativa al mayor de los dos, para poder pintarlos juntos
        esc = max(abs(p), abs(q), 1e-9)
        ax.plot([q / esc, p / esc], [i, i], color=GRIS, linewidth=1.0, zorder=1)
        ax.scatter(q / esc, i, s=26, color=AZUL, zorder=2)
        ax.scatter(p / esc, i, s=26, color=BERMELLON, zorder=2)
        ax.text(1.04, i, f"{p:,.2f} / {q:,.2f}".replace(",", " "),
                va="center", fontsize=6.8, color=NEGRO)
    ax.set_yticks(y)
    ax.set_yticklabels([ETIQ[n] for n in nombres])
    ax.set_xlim(0, 1.34)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("mediana, reescalada al mayor de las dos clases")
    ax.scatter([], [], s=26, color=BERMELLON, label="positivos (n = 13.577)")
    ax.scatter([], [], s=26, color=AZUL, label="negativos (n = 39.986)")
    ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(0.0, 0.0))
    ax.grid(axis="y", visible=False)
    guarda(fig, "f3_separacion.png")


if __name__ == "__main__":
    main()
