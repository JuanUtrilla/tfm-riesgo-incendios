#!/usr/bin/env python3
"""Figura 18: la escalera de baselines de la iteración 1.

AUC-PR en el conjunto de prueba de 2020 de cada peldaño, del FWI sin modelo al XGBoost con las
50 variables. La línea discontinua es el azar, que en un AUC-PR coincide con
la prevalencia del conjunto de prueba (26.28 %). Cada peldaño gana al
anterior, y el salto grande llega al añadir las variables no meteorológicas.

Fuente: 03_iteracion1/MODELO_B_BITACORA.md, tabla «Escalera de baselines»,
producida por 03_iteracion1/33_train/entrenar_modelo.py (metricas_v1.json).
Figura del README del capítulo 3 y del anexo B.2.
"""
import matplotlib.pyplot as plt

from comun import ANCHO, AZUL, GRIS, guarda

PELDANOS = [
    ("A. FWI absoluto (sin modelo)", 0.488),
    ("A2. FWI percentil local (sin modelo)", 0.481),
    ("B. XGBoost solo con FWI", 0.491),
    ("C. XGBoost meteorología + FWI", 0.716),
    ("D. XGBoost completo (50 variables)", 0.843),
]
AZAR = 0.2628


def main():
    fig, ax = plt.subplots(figsize=(ANCHO, 2.3))
    nombres = [p[0] for p in PELDANOS][::-1]
    valores = [p[1] for p in PELDANOS][::-1]
    y = range(len(PELDANOS))
    colores = [AZUL if n.startswith("D.") else GRIS for n in nombres]
    ax.barh(y, valores, height=0.62, color=colores)
    for yi, v in zip(y, valores):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(AZAR, color="#333333", ls="--", lw=0.9)
    ax.text(AZAR + 0.008, len(PELDANOS) - 0.45, "azar (prevalencia 26.3 %)",
            fontsize=7.5, color="#333333", va="bottom")
    ax.set_yticks(list(y), nombres)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(PELDANOS) - 0.1)
    ax.set_xlabel("AUC-PR en el conjunto de prueba de 2020")
    ax.grid(axis="y", visible=False)
    guarda(fig, "f18_escalera_it1.png")


if __name__ == "__main__":
    main()
