#!/usr/bin/env bash
# Pone en PYTHONPATH todas las carpetas con código del repo.
#
# POR QUÉ HACE FALTA
# El árbol está ordenado por capítulos de la memoria, no por paquetes de
# Python, y los scripts se importan entre sí con nombres planos
# (`import config`, `from dos_05_modelos import FEATS_CUANDO`) exactamente
# como en los repos originales. Eso es deliberado: tocar los imports sería
# reescribir 124 ficheros y perder la garantía de que este código es el que
# produjo los números de la memoria. El precio es una línea antes de ejecutar:
#
#     source entorno.sh
#     python 05_iteracion2/57_ablaciones/dos_18_ratio.py
#
# Dónde están los datos (ninguna ruta personal vive ya en el código):
#   TFM_DATOS   fuentes crudas (cubo IberFire, EGIF, FIRMS, rayos). Sin ella
#               se usa ./datos, y si tampoco está, el recorte de ./muestras.
#   TFM_USB     disco con los datasets de entrenamiento (dos_*). Opcional.
#   TFM_SALIDA  dónde escribir. Por defecto ./salida.
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRS=$(find "$RAIZ" -name '*.py' -not -path '*/.git/*' -printf '%h\n' | sort -u | tr '\n' ':')
export PYTHONPATH="${DIRS}${PYTHONPATH}"
export TFM_DATOS="${TFM_DATOS:-$RAIZ/datos}"
echo "PYTHONPATH: $(echo "$DIRS" | tr ':' '\n' | grep -c .) carpetas · datos en $TFM_DATOS"
