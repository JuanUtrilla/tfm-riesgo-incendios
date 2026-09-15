#!/usr/bin/env python3
"""
Las features del modelo, y su orden.

=============================================================================
POR QUÉ ESTÁN COPIADAS Y NO IMPORTADAS
=============================================================================
El original vive en `03_iteracion1/33_train/entrenar_modelo.py`. Importarlo
desde aquí ataría la cadena de servicio a todo el entrenamiento (xgboost,
sklearn, los datasets), cuando lo único que necesita son cinco listas.

El precio es que pueden DIVERGIR: si alguien reentrena con otras features y
este fichero no se actualiza, el modelo recibiría las columnas en otro orden
y predeciría basura sin dar ningún error. XGBoost no comprueba nombres.

Por eso `verifica()` contrasta esta copia con `entrenar_modelo.py` y lo llaman los
módulos que predicen, antes de construir la matriz. Es barato y convierte un
fallo silencioso en una excepción.

Ejecutado como script, hace esa comprobación y ya está.
"""

import importlib.util
import sys

import config

FEATS_METEO = ["fwi", "fwi_pctl_local", "fwi_anom_sigma", "t2m_max", "t2m_min",
               "rh_min", "viento_max", "precip_dia", "vpd_max", "lst",
               "precip_7d", "precip_15d", "precip_30d", "fwi_med_7d",
               "fwi_max_7d", "fwi_med_15d", "fwi_med_30d", "rh_min_med_7d",
               "t2m_max_med_7d", "viento_max_med_7d", "dias_sin_lluvia"]
FEATS_VEG = ["ndvi", "ndvi_med_30d", "lai", "swi010"]
FEATS_ESTAT = ["elevacion", "pendiente", "rugosidad", "dist_carreteras",
               "dist_rios", "popdens", "clc_bosque", "clc_matorral",
               "clc_agricola", "clc_artificial", "clc_abierto",
               "clc_agric_hetero"]
FEATS_HIST = ["n_fuegos_1km_90d", "n_fuegos_10km_90d", "n_fuegos_10km_365d",
              "n_fuegos_1km_hist", "n_fuegos_10km_mismomes_hist",
              "frp_max_50km_7d", "n_detec_50km_7d", "rayos_dia", "rayos_7d"]
FEATS_CAL = ["mes", "dia_anio", "es_festivo", "ccaa"]

FULL = FEATS_METEO + FEATS_VEG + FEATS_ESTAT + FEATS_HIST + FEATS_CAL

# las que recalcula la malla desde ERA5-Land. `lst` NO: es satélite (MODIS),
# no meteo, y ERA5-Land no lo da — sale del cubo igual que en producción.
METEO_MALLA = [f for f in FEATS_METEO if f != "lst"]


def verifica(estricto=True):
    """¿Sigue coincidiendo esta copia con `entrenar_modelo.py`?"""
    ruta = f"{config.FUENTE}/entrenar_modelo.py"
    spec = importlib.util.spec_from_file_location("_entrenar_modelo", ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_entrenar_modelo"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # entrenar_modelo.py importa pesado
        if estricto:
            raise
        print(f"  aviso: no se pudo verificar features ({type(e).__name__})")
        return None

    dif = []
    for nom in ("FEATS_METEO", "FEATS_VEG", "FEATS_ESTAT", "FEATS_HIST",
                "FEATS_CAL"):
        aqui, alli = globals()[nom], getattr(mod, nom)
        if aqui != alli:
            dif.append(f"{nom}: aquí {aqui}\n           entrenamiento {alli}")
    if dif:
        msg = ("Las features han DIVERGIDO de las de entrenar_modelo.py:\n  "
               + "\n  ".join(dif))
        if estricto:
            raise SystemExit(msg)
        print(msg)
    return not dif


if __name__ == "__main__":
    ok = verifica(estricto=False)
    print(f"{len(FULL)} features · coinciden con entrenar_modelo.py: {ok}")
