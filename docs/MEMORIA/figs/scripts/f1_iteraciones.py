#!/usr/bin/env python3
"""F1 - Las dos iteraciones y la bisagra. Figura estructural, sin datos.

Redibuja como vectorial el diagrama ASCII de `00_marco/README.md:3-32`.
Uso:  python f1_iteraciones.py
"""
import matplotlib.patches as mp
import matplotlib.pyplot as plt

from comun import ANCHO, AZUL, BERMELLON, GRIS, NEGRO, guarda


def caja(ax, x, y, w, h, titulo, lineas, color, relleno):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.0,rounding_size=0.018",
                                   linewidth=1.1, edgecolor=color,
                                   facecolor=relleno, zorder=2))
    ax.text(x + w / 2, y + h - 0.045, titulo, ha="center", va="top", zorder=3,
            fontsize=8.2, fontweight="bold", color=color)
    if lineas:
        ax.text(x + w / 2, y + (h - 0.075) / 2, "\n".join(lineas), ha="center",
                va="center", fontsize=7.0, color=NEGRO, linespacing=1.5,
                zorder=3)


def flecha(ax, xy0, xy1, texto=None, color=GRIS, dx=0.012):
    ax.annotate("", xy=xy1, xytext=xy0, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.0,
                                shrinkA=0, shrinkB=0))
    if texto:
        ax.text((xy0[0] + xy1[0]) / 2 + dx, (xy0[1] + xy1[1]) / 2, texto,
                fontsize=6.8, color=color, ha="left", va="center", zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(ANCHO, 4.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

    caja(ax, 0.14, 0.855, 0.72, 0.145, "Fuentes y tratamiento comunes",
         ["cubo IberFire · 498.530 celdas de 1 km × día · 46 variables"],
         NEGRO, "#F2F2F2")

    caja(ax, 0.005, 0.505, 0.46, 0.275, "Iteración 1 · grupo de control",
         ["etiqueta EGIF (ignición, punto)",
          "caso-control con celda fija, 1:3",
          "«¿es hoy peligroso en esta celda?»",
          "AUC 0,89 en test 2022"],
         AZUL, "#EAF2F8")

    caja(ax, 0.535, 0.505, 0.46, 0.275, "Iteración 2 · rediseño",
         ["etiqueta EFFIS (superficie quemada)",
          "negativos del MISMO dia",
          "«¿cuál de las 498.530 celdas arde hoy?»",
          "ratio de negativos ablacionado"],
         BERMELLON, "#FDF0E7")

    caja(ax, 0.005, 0.145, 0.46, 0.275, "Bisagra · el diagnóstico",
         ["en operación cae a AUC 0,56-0,64",
          "¿la meteorología? NO",
          "¿el desajuste train/serve? NO",
          "es la ESPECIFICACIÓN"],
         NEGRO, "#FFFFFF")

    caja(ax, 0.19, 0.0, 0.62, 0.135, "Comparación sobre los mismos días",
         ["temporada 2026 · 74 días · misma fuente y mismo banco"],
         NEGRO, "#F2F2F2")

    flecha(ax, (0.235, 0.855), (0.235, 0.780))
    flecha(ax, (0.765, 0.855), (0.765, 0.780))
    flecha(ax, (0.235, 0.505), (0.235, 0.420), "se pone\nen producción")
    flecha(ax, (0.235, 0.145), (0.235, 0.135))
    flecha(ax, (0.765, 0.505), (0.765, 0.135))
    ax.annotate("", xy=(0.535, 0.29), xytext=(0.465, 0.29), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=BERMELLON, linewidth=1.2))
    ax.text(0.50, 0.315, "obliga a", fontsize=7, color=BERMELLON, ha="center")

    guarda(fig, "f1_iteraciones.png")


if __name__ == "__main__":
    main()
