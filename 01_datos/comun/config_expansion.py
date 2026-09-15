#!/usr/bin/env python3
"""
Rutas del disco de EXPANSIÓN para el trabajo de los dos modelos.

=============================================================================
POR QUÉ UN SEGUNDO CONFIG
=============================================================================
El disco principal va al 97 % (5,8 GB libres el 21/08/2026), así que se usa
un volumen de expansión sin borrar nada. Todo lo NUEVO y
pesado del encargo (ERA5-Land horario 2015-2020, datasets remuestreados,
modelos, logs largos) vive aquí; en el repo solo quedan código, notas y
resultados pequeños (json/csv/md).

Los volúmenes son exFAT/vfat: sin enlaces simbólicos ni permisos POSIX. Se
usan por ruta absoluta y nada más.

=============================================================================
DÓNDE SE BUSCA
=============================================================================
La raíz del volumen de expansión se declara con `TFM_USB`. Por defecto es
`<TFM_DATOS>/expansion`. La copia al volumen se hizo con rsync y se verificó
con md5 (208 ficheros, 1,55 GB).

Si no existe ninguno, se falla ALTO al importar, no a mitad de un paso.
"""

import os
from pathlib import Path

_RAIZ_REPO = Path(__file__).resolve().parents[2]
_DATOS = os.environ.get("TFM_DATOS", str(_RAIZ_REPO / "datos"))
CANDIDATOS = [os.environ.get("TFM_USB"),
              os.path.join(_DATOS, "expansion")]
EXPANSION = None if os.environ.get("TFM_SIN_EXTERNO") else \
    next((c for c in CANDIDATOS if c and os.path.isdir(c)), None)
if EXPANSION is None:
    # GitHub Actions (21/08/2026): sin disco externo, los modelos y lo poco que
    # la cadena diaria necesita vienen de `estado/` (ver config.py). Los
    # scripts de ENTRENAMIENTO (dos_00..dos_13) no se ejecutan allí.
    import config as _c
    RAIZ = _c.ESTADO
    if not os.path.isdir(RAIZ):
        raise SystemExit("No se encuentra el volumen de expansión ni estado/ "
                         "(exporta TFM_USB=/ruta)")
else:
    RAIZ = EXPANSION

ERA5_CDS = f"{RAIZ}/era5land_cds"     # horario 2015-2020 + diario en nodos
DATASET = f"{RAIZ}/dataset"           # muestras remuestreadas (parquet)
MODELOS = f"{RAIZ}/modelos"           # modelos dónde/cuándo/mixto
LOGS = f"{RAIZ}/logs"

if EXPANSION is not None:
    for d in (ERA5_CDS, DATASET, MODELOS, LOGS):
        os.makedirs(d, exist_ok=True)
