#!/usr/bin/env python3
"""Figura 7: la prevalencia real del fuego y la persistencia de la etiqueta.

Los dos números del análisis exploratorio del cubo de los que depende el
diseño posterior:

  (a) celdas-día con `is_fire` por cada 100.000, año a año, frente al 25 % de
      positivos que tenía por construcción el conjunto de la iteración 1.
      Hay cinco órdenes de magnitud de diferencia; el eje es logarítmico
      porque si no la barra del diseño aplasta a las demás.
  (b) P(arde en t+k | arde en t): 0,51 a un día. `is_fire` marca los días en
      que la celda está ardiendo, no las igniciones, y por eso la evaluación
      usa `primer_dia`.

Fuente: 02_eda/ANALISIS_CUBO.md, tabla de prevalencia (`dos_00`) y sección de
persistencia. Medido sobre las 498.530 celdas peninsulares × día.
"""
import matplotlib.pyplot as plt
import numpy as np

from comun import ANCHO, AZUL, BERMELLON, GRIS, NARANJA, guarda

# ANALISIS_CUBO.md: año -> (anual, verano jun-sep), por 100.000 celdas-día
PREV = {2015: (1.56, 4.24), 2017: (4.01, 5.30), 2018: (0.40, 0.74),
        2019: (1.71, 1.89), 2020: (3.08, 6.40), 2021: (3.46, 6.69),
        2022: (14.16, 37.96), 2023: (3.80, 0.50), 2024: (1.82, 2.40)}
DISENIO_IT1 = 25_000          # 25 % de positivos = 25.000 por 100.000
PERSIS = {1: 0.51, 3: 0.23, 7: 0.04}


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(ANCHO, 2.5),
                               gridspec_kw={"width_ratios": [2.05, 1]})

    an = np.array([PREV[y][0] for y in PREV])
    ve = np.array([PREV[y][1] for y in PREV])
    x = np.arange(len(PREV))
    a.bar(x - 0.2, an, 0.4, color=AZUL, label="año completo")
    a.bar(x + 0.2, ve, 0.4, color=BERMELLON, label="verano (jun–sep)")
    a.axhline(DISENIO_IT1, color=GRIS, ls="--", lw=1)
    a.text(len(PREV) - 0.5, DISENIO_IT1 / 2.2,
           "diseño de la iteración 1: 25 % de positivos",
           ha="right", va="top", fontsize=7.5, color=GRIS)
    a.set_yscale("log")
    a.set_ylim(0.3, 2e5)
    a.set_xticks(x, [str(y) for y in PREV], rotation=45)
    a.set_ylabel("celdas-día con fuego\npor 100.000")
    a.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2,
             columnspacing=1.0, handlelength=1.2)
    a.set_title("(a) prevalencia real frente a la del conjunto", loc="left")

    k = list(PERSIS)
    b.plot(k, [PERSIS[i] for i in k], "o-", color=NARANJA, lw=1.6, ms=5)
    for i in k:
        b.annotate(f"{PERSIS[i]:.2f}".replace(".", ","), (i, PERSIS[i]),
                   textcoords="offset points", xytext=(4, 5), fontsize=7.5)
    b.set_xticks(k)
    b.set_ylim(0, 0.62)
    b.set_xlim(0.2, 7.8)
    b.set_xlabel("días después ($k$)")
    b.set_ylabel("P(arde en $t+k$ | arde en $t$)")
    b.set_title("(b) la etiqueta persiste", loc="left")

    guarda(fig, "f7_prevalencia.png")


if __name__ == "__main__":
    main()
