#!/usr/bin/env python3
"""Figura 8: por qué el FWI entra normalizado a la climatología de la celda.

En los días de incendio del split de entrenamiento (2015-2018), el FWI
ABSOLUTO mediano va de 3 en Asturias a 41 en Andalucía: un umbral único no
existe. El mismo dato en PERCENTIL LOCAL —frente a la climatología de la propia
celda— se concentra entre 72 y 92 en todas. Es el argumento de
`fwi_pctl_local`, y el mismo que sostiene `fwi_anom_sigma`.

Solo train, para que ninguna decisión de diseño mire a val/test.

Entrada: TFM_fuego/dataset/dataset_modelo_v1.parquet (fijable con TFM_ORIG).
"""
import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from comun import ANCHO, AZUL, BERMELLON, guarda, ORIG

DATOS = pathlib.Path(os.environ.get("TFM_ORIG", ORIG)) / "dataset" / \
    "dataset_modelo_v1.parquet"

# códigos INE de comunidad autónoma, verificados contra el centroide de las
# celdas del propio conjunto (lat/lon medias por código)
CCAA = {1: "Andalucía", 3: "Asturias", 6: "Cantabria", 7: "Castilla y León",
        8: "C.-La Mancha", 11: "Extremadura", 12: "Galicia", 15: "Navarra"}


def main():
    d = pd.read_parquet(DATOS, columns=["split", "label", "ccaa", "fwi",
                                        "fwi_pctl_local"])
    pos = d[(d.split == "train") & (d.label == 1)]
    orden = [c for c in pos.ccaa.value_counts().index if c in CCAA][:8]
    orden = sorted(orden, key=lambda c: pos[pos.ccaa == c].fwi.median())
    etiq = [CCAA[c] for c in orden]

    fig, (a, b) = plt.subplots(1, 2, figsize=(ANCHO, 2.6), sharey=False)
    for ax, col, color, tit in (
            (a, "fwi", BERMELLON, "(a) FWI absoluto"),
            (b, "fwi_pctl_local", AZUL, "(b) FWI en percentil local")):
        datos = [pos.loc[pos.ccaa == c, col].dropna().values for c in orden]
        bp = ax.boxplot(datos, vert=False, showfliers=False, widths=0.6,
                        patch_artist=True, medianprops=dict(color="black", lw=1.2))
        for caja in bp["boxes"]:
            caja.set(facecolor=color, alpha=0.55, edgecolor=color, lw=0.9)
        for pieza in ("whiskers", "caps"):
            for e in bp[pieza]:
                e.set(color=color, lw=0.9)
        ax.set_yticks(range(1, len(orden) + 1), etiq)
        ax.set_title(tit, loc="left")
        ax.grid(axis="y", alpha=0)
    a.set_xlabel("FWI en el día del incendio")
    b.set_xlabel("percentil frente a la climatología de la celda")
    b.set_xlim(0, 100)
    b.set_yticklabels([])
    guarda(fig, "f8_normalizacion.png")


if __name__ == "__main__":
    main()
