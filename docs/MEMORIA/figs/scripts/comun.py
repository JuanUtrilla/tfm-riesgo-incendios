"""Ajustes compartidos por los scripts de figuras de la sección de modelado.

Todas las figuras salen a ../ en PNG de 300 dpi, con ancho pensado para el
ancho de página de la memoria (16 cm), sin título dentro de la figura (el
título va en el pie) y con la paleta Okabe-Ito, segura para daltonismo.

Rutas: RAIZ es la raíz de este repositorio (REPO, un alias). Los datos
externos se indican con TFM_DATOS, el archivo de previsiones IFS con
TFM_ARCHIVO y la carpeta de resultados de los scripts con TFM_SALIDA.
"""
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = pathlib.Path(__file__).resolve().parents[4]
REPO = RAIZ
DATOS = pathlib.Path(os.environ.get("TFM_DATOS", RAIZ / "datos"))
ARCHIVO = pathlib.Path(os.environ.get("TFM_ARCHIVO", DATOS / "archivo"))
SALIDA_REPO = pathlib.Path(os.environ.get("TFM_SALIDA", RAIZ / "salida"))
SALIDA = pathlib.Path(__file__).resolve().parents[1]

# Okabe-Ito
NEGRO = "#000000"
NARANJA = "#E69F00"
AZUL_CIELO = "#56B4E9"
VERDE = "#009E73"
AMARILLO = "#F0E442"
AZUL = "#0072B2"
BERMELLON = "#D55E00"
ROSA = "#CC79A7"
GRIS = "#7F7F7F"

ANCHO = 6.3          # pulgadas ~= 16 cm

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


def guarda(fig, nombre):
    ruta = SALIDA / nombre
    fig.savefig(ruta)
    plt.close(fig)
    print(f"escrito {ruta}")
