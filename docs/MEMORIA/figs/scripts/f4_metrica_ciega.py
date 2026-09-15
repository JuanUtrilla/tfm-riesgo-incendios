#!/usr/bin/env python3
"""F4 - La misma pareja de modelos bajo dos métricas.

Lee `salida/dos_05_metricas.json` (iteración 2, paso 5). No recalcula nada.
  · AUC caso-control: cada modelo en el banco de su propio diseño
    (`auc_global_test_caso_control`).
  · AUC dentro del día: media por día en el banco `eval_dia`, test 2020,
    verdad EGIF (`2020_egif`).
Uso:  python f4_metrica_ciega.py   [TFM_SALIDA=<carpeta de resultados>]
"""
import json
import os
import pathlib

import numpy as np

import matplotlib.pyplot as plt

from comun import ANCHO, AZUL, BERMELLON, GRIS, NEGRO, SALIDA_REPO, guarda

RUTA = SALIDA_REPO / "dos_05_metricas.json"


def main():
    d = json.load(open(RUTA))
    cc = d["auc_global_test_caso_control"]
    dia = d["2020_egif"]
    valores = {
        "AUC caso-control\n(banco de desarrollo)": (cc["cuando_46_en_test_cuando"],
                                                    cc["donde_46_en_test_donde"]),
        "AUC dentro del día\n(banco operativo)": (dia["cuando_46"]["auc_medio"],
                                                  dia["donde_dia"]["auc_medio"]),
    }
    fig, ax = plt.subplots(figsize=(ANCHO, 2.6))
    x = np.arange(len(valores)); w = 0.32
    a = [v[0] for v in valores.values()]
    b = [v[1] for v in valores.values()]
    ax.bar(x - w / 2, a, w, color=AZUL, label="muestreo caso-control (celda fija)")
    ax.bar(x + w / 2, b, w, color=BERMELLON, label="muestreo del mismo día")
    for xi, (va, vb) in zip(x, zip(a, b)):
        ax.text(xi - w / 2, va + 0.008, f"{va:.3f}".replace(".", ","),
                ha="center", fontsize=7.6)
        ax.text(xi + w / 2, vb + 0.008, f"{vb:.3f}".replace(".", ","),
                ha="center", fontsize=7.6)
        ax.annotate("", xy=(xi + w / 2, min(va, vb) + 0.004),
                    xytext=(xi - w / 2, min(va, vb) + 0.004),
                    arrowprops=dict(arrowstyle="<->", color=GRIS, linewidth=0.8))
        ax.text(xi, min(va, vb) + 0.004, f"Δ {abs(vb - va):.3f}".replace(".", ","),
                ha="center", va="center", fontsize=7.4, color=NEGRO,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.axhline(0.5, color=GRIS, linewidth=0.7, linestyle=(0, (4, 3)))
    ax.text(len(valores) - 0.5, 0.508, "azar", fontsize=7, color=GRIS, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(valores.keys())
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUC-ROC")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))
    ax.grid(axis="x", visible=False)
    guarda(fig, "f4_metrica_ciega.png")


if __name__ == "__main__":
    main()
