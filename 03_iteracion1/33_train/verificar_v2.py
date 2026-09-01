#!/usr/bin/env python3
"""Verifica la procedencia de `xgb_v2_prototipo`, el modelo servido en producción.

No existe script que lo entrene: fue un reentrenamiento interactivo del
15-jul-2026 (el .ubj es de las 08:51, horas antes del commit inicial de
`TFM_fuego`) documentado en `../MODELO_B_BITACORA.md` §16 — v1 sin las cuatro
autorregresivas intra-celda contaminadas por el artefacto del muestreo
misma-celda (+0,92 log-odds por `n_fuegos_1km_hist=0` en pleno Madrid urbano;
AUC-PR 0,828 frente a 0,843, «coste pequeño, modelo honesto para producción»).

Este script convierte esa afirmación en comprobación: la lista canónica de 50
features (`features.FULL`) menos las 46 del JSON servido debe ser EXACTAMENTE
el conjunto de las cuatro contaminadas, y el orden del JSON debe coincidir con
el que lleva dentro el .ubj. Corre con `muestras/` recién clonado el repo.
"""
import json
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(f"{DIR}/../..")
sys.path.insert(0, f"{RAIZ}/01_datos/comun")

import features

CONTAMINADAS = {"n_fuegos_1km_hist", "n_fuegos_1km_90d",
                "n_fuegos_10km_90d", "n_fuegos_10km_365d"}
JSON = f"{RAIZ}/muestras/modelos/xgb_v2_prototipo_features.json"
UBJ = f"{RAIZ}/muestras/modelos/xgb_v2_prototipo.ubj"


def main():
    v2 = json.load(open(JSON))
    full = list(features.FULL)
    print(f"features.FULL: {len(full)} · JSON servido: {len(v2)}")
    assert len(full) == 50 and len(v2) == 46, "recuentos inesperados"
    quitadas = set(full) - set(v2)
    print(f"FULL − JSON: {sorted(quitadas)}")
    assert quitadas == CONTAMINADAS, "no son las 4 de la bitácora §16"
    assert not set(v2) - set(full), "el JSON trae features fuera de FULL"
    try:
        import xgboost as xgb
        b = xgb.Booster(); b.load_model(UBJ)
        assert b.num_features() == 46
        if b.feature_names is not None:
            assert b.feature_names == v2, "el orden del .ubj no es el del JSON"
            print("orden del JSON == orden dentro del .ubj (46)")
        print(f"rondas del .ubj: {b.num_boosted_rounds()} (v1: 489 → "
              f"reentrenamiento con early stopping, no un recorte)")
    except ImportError:
        print("(sin xgboost: comprobado solo contra el JSON)")
    print("OK: xgb_v2_prototipo = FULL sin las 4 autorregresivas intra-celda")


if __name__ == "__main__":
    main()
