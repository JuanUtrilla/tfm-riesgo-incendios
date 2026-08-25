#!/usr/bin/env python3
"""
Rutas del disco de EXPANSIÓN para el trabajo de los dos modelos.

=============================================================================
POR QUÉ UN SEGUNDO CONFIG
=============================================================================
El disco principal va al 97 % (5,8 GB libres el 21/08/2026). `PROMPT_DOS_MODELOS.md`
§2.1 manda usar el volumen de expansión sin borrar nada. Todo lo NUEVO y
pesado del encargo (ERA5-Land horario 2015-2020, datasets remuestreados,
modelos, logs largos) vive aquí; en el repo solo quedan código, notas y
resultados pequeños (json/csv/md).

Los volúmenes son exFAT/vfat: sin enlaces simbólicos ni permisos POSIX. Se
usan por ruta absoluta y nada más.

=============================================================================
21/08/2026 (mañana): TODO SE COPIÓ AL USB NUEVO, QUE PASA A SER EL PRINCIPAL
=============================================================================
Orden de búsqueda: primero el USB (`3D97-F226`, vfat, 58 GB), después el
Expansion como respaldo. La copia se hizo con rsync y se verificó con md5
(208 ficheros, 1,55 GB; ver `MOVIDO_A_EXPANSION.md`). El Expansion conserva
una copia idéntica a esa fecha, pero lo que se escriba a partir de ahora va
SOLO al USB: no se sincronizan. Con `TFM_USB=/otra/ruta` se fuerza otro sitio.

Si no hay ninguno montado, se falla ALTO al importar, no a mitad de un paso.
"""

import os

CANDIDATOS = [os.environ.get("TFM_USB"),
              "/run/media/charredgem/3D97-F226",      # USB nuevo (principal)
              "/run/media/charredgem/Expansion"]      # Expansion (respaldo)
EXPANSION = None if os.environ.get("TFM_SIN_EXTERNO") else \
    next((c for c in CANDIDATOS if c and os.path.ismount(c)), None)
if EXPANSION is None:
    # GitHub Actions (21/08/2026): sin disco externo, los modelos y lo poco que
    # la cadena diaria necesita vienen de `estado/` (ver config.py). Los
    # scripts de ENTRENAMIENTO (dos_00..dos_13) no se ejecutan allí.
    import config as _c
    RAIZ = _c.ESTADO
    if not os.path.isdir(RAIZ):
        raise SystemExit("Ni el USB 3D97-F226 ni el Expansion están montados, y "
                         "no hay estado/ (o exporta TFM_USB=/ruta)")
else:
    RAIZ = f"{EXPANSION}/TFM_fuego_expansion"

ERA5_CDS = f"{RAIZ}/era5land_cds"     # horario 2015-2020 + diario en nodos
DATASET = f"{RAIZ}/dataset"           # muestras remuestreadas (parquet)
MODELOS = f"{RAIZ}/modelos"           # modelos dónde/cuándo/mixto
LOGS = f"{RAIZ}/logs"

if EXPANSION is not None:
    for d in (ERA5_CDS, DATASET, MODELOS, LOGS):
        os.makedirs(d, exist_ok=True)
