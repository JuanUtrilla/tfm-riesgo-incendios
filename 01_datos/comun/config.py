#!/usr/bin/env python3
"""
Rutas y constantes compartidas del pipeline de malla.

=============================================================================
POR QUÉ ESTE REPO ESTÁ SEPARADO
=============================================================================
El repo original (`TFM_fuego`) tiene una serie de previsiones SELLADAS POR
COMMIT: cada mapa operativo publicado se corresponde con un estado exacto del
código. Tocar `mapa_riesgo_hoy.py` para meterle la malla partiría esa serie y
la haría irreproducible hacia atrás.

Así que aquí NO se modifica nada de allí. Este repo:

  · LEE del repo original, en solo lectura: el cubo IberFire, los modelos
    entrenados, FIRMS, EGIF, rayos WGLC y los productos ya calculados del
    pipeline de malla (`malla_data/`).
  · ESCRIBE únicamente en su propio `salida/`.

Cuando la malla sustituya a producción, lo que se moverá será este repo
entero, no un parche sobre el otro.

=============================================================================
LOS DATOS NO SE COPIAN
=============================================================================
`iberfire/IberFire.nc` son 29 GB y `rayos_data/wglc` otros 2,4 GB. Duplicarlos
no cabe en el disco (quedan ~6,7 GB libres) y además tendrían que mantenerse
sincronizados. Se apuntan con `FUENTE`.

⚠️ `malla_data/` del repo original lo escriben los módulos 2 y 4. Si hay otra
sesión descargando meses de CDS, ESTE repo solo lee de ahí. No escribas en
`FUENTE` desde aquí.
"""

import os

# ADAPTADO al repo definitivo (25/08/2026): la única diferencia con el
# `config.py` de `TFM_fuego_malla` es de RUTAS. Ni una constante del pipeline
# cambia. Aquí el módulo vive en `01_datos/comun/`, así que la raíz del repo
# son dos niveles arriba, y no queda ninguna ruta personal en el código: el
# disco de datos se declara con la variable de entorno TFM_DATOS.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SALIDA = os.environ.get("TFM_SALIDA", f"{BASE}/salida")
FUENTE = os.environ.get("TFM_DATOS", os.environ.get("TFM_FUENTE", f"{BASE}/datos"))
MUESTRAS = f"{BASE}/muestras"

# =============================================================================
# SIN EL REPO ORIGINAL: GITHUB ACTIONS (21/08/2026)
# =============================================================================
# El portátil se apaga a diario, así que la cadena corre también en GitHub
# Actions. Allí no está `TFM_fuego` ni el cubo de 29 GB: lo que la cadena
# diaria necesita de verdad (capas estáticas 2D, climatología, nodos, modelos,
# cachés mensuales) se exporta con `gh_exportar_estado.py` a `estado/`, que
# se sirve como asset de un Release del repo. Cada ruta de abajo prefiere el
# original si existe y cae a `estado/` si no. Nada cambia en local.
ESTADO = os.environ.get("TFM_ESTADO", MUESTRAS)
SIN_ORIGINAL = not os.path.isdir(FUENTE)


def _o_estado(original, en_estado):
    return original if os.path.exists(original) else f"{ESTADO}/{en_estado}"


# --- entradas, solo lectura ---------------------------------------------
CUBO = _o_estado(f"{FUENTE}/iberfire/IberFire.nc", "cubo_estaticas.nc")
# sin el cubo de 29 GB se cae a las capas estáticas del recorte (16 MB)
MODELOS = _o_estado(f"{FUENTE}/modelos", "modelos")
FIRMS = f"{FUENTE}/firms_iberia_2015_2024.parquet"
EGIF = f"{FUENTE}/egif_civio.csv"
WGLC = f"{FUENTE}/rayos_data/wglc/wglc_timeseries_30m_daily.nc"

# productos del pipeline de malla que ya están calculados en el repo original
DATA = f"{FUENTE}/malla_data"
CRUDO = f"{DATA}/_cds"
# Se resuelven con entrada() al final del módulo (ver más abajo): salida/
# propio si existe, si no el malla_data/ del original.

# --- malla y descarga ----------------------------------------------------
PASO = 0.1                       # resolución nativa de ERA5-Land, en grados
AREA = [44.0, -9.6, 35.8, 4.4]   # [N, W, S, E], alineada a 0,1°
VARIABLES = ["2m_temperature", "2m_dewpoint_temperature",
             "10m_u_component_of_wind", "10m_v_component_of_wind",
             "total_precipitation"]

os.makedirs(SALIDA, exist_ok=True)


def entrada(nombre):
    """Ruta de lectura de un producto del pipeline.

    Prefiere el `salida/` de ESTE repo; si no está, cae al `malla_data/` del
    original. Así el repo funciona desde el primer día —reutilizando lo ya
    calculado allí— y se va volviendo autónomo según regenera sus piezas, sin
    escribir nunca en el repo original.
    """
    propio = f"{SALIDA}/{nombre}"
    if os.path.exists(propio):
        return propio
    alli = f"{DATA}/{nombre}"
    if os.path.exists(alli):
        return alli
    # último recurso: el recorte de julio de 2026 que viaja en el repo, para
    # que el pipeline se pueda ejecutar sin descargar los 44 GB de fuentes.
    return f"{ESTADO}/{nombre}"


def salida(nombre):
    """Ruta de escritura. SIEMPRE en este repo."""
    return f"{SALIDA}/{nombre}"


NODOS = entrada("nodos.npz")
CLIM = entrada("clim_fwi_nodos.npz")
MAPEO_FWI = entrada("mapeo_fwi.npz")
CRUDO_PROPIO = f"{SALIDA}/_cds"      # las descargas nuevas caen aquí
