#!/usr/bin/env python3
"""
Exporta a `estado/` todo lo que la cadena diaria necesita de los datos externos
(TFM_DATOS) y del disco de expansión, para que corra en GitHub Actions sin ellos.

Se ejecuta en local (con el cubo y el USB) cada vez que cambie algo de esto:
un modelo nuevo, otra climatología, otra capa. Después: `gh_estado.py push --base`.

Qué exporta y por qué:
  cubo_estaticas.nc     las 16 variables 2D del cubo que leen riesgo_hoy /
                        dos_riesgo_hoy / quemadas / firms_dia (+ x, y, is_spain).
                        Comprimido: unos 15 MB frente a los 29 GB del cubo.
  nodos.npz, clim_fwi_nodos.npz, mapeo_ifs_a_era5land.npz   la malla y su clim
  malla_mensual_m{4..11}.npz   vegetación/LST/EGIF mismo-mes (caché de producción)
  modelos/              xgb_v2 (+features), donde_dia_effis, cuando, mapa dónde
  municipios.json       geocodificador de MITECO
  vendor/               parser de partes del MITECO y geocodificador de validar_miteco
                        (copias literales, con su origen en cabecera)
  julio2026_effis.json, estaciones_prototipo.parquet  bases de los jueces en
                        vivo que siguieron la temporada 2026 (dos_15, dos_16)
"""

import glob
import os
import shutil

import numpy as np
import xarray as xr

import config
import config_expansion as ce

E = config.ESTADO
os.makedirs(f"{E}/modelos", exist_ok=True)
os.makedirs(f"{E}/vendor", exist_ok=True)

# 1. cubo → NetCDF pequeño
VARS = ["is_spain", "AutonomousCommunities", "elevation_mean", "slope_mean",
        "roughness_mean", "dist_to_roads_mean", "dist_to_waterways_mean",
        "popdens_2020"] + [f"CLC_2018_{s}" for s in (
            "forest_proportion", "scrub_proportion", "agricultural_proportion",
            "artificial_proportion", "open_space_proportion",
            "heterogeneous_agriculture_proportion")]
ruta = f"{E}/cubo_estaticas.nc"
if not os.path.exists(ruta):
    ds = xr.open_dataset(f"{config.FUENTE}/iberfire/IberFire.nc", decode_timedelta=False)
    sub = ds[VARS].load()
    sub.attrs = {"origen": "IberFire.nc (Ercibengoa 2025), solo capas 2D",
                 "nota": "exportado por gh_exportar_estado.py para la cadena diaria"}
    enc = {v: {"zlib": True, "complevel": 6} for v in VARS}
    sub.to_netcdf(ruta, encoding=enc)
    ds.close()
    print(f"  cubo_estaticas.nc: {os.path.getsize(ruta)/1e6:.1f} MB")

# 2. malla, climatología, mapeo, cachés mensuales
for n in ("nodos.npz", "clim_fwi_nodos.npz", "mapeo_ifs_a_era5land.npz"):
    shutil.copy(config.entrada(n), f"{E}/{n}")
for m in range(4, 12):
    for origen in (config.salida(f"malla_mensual_m{m}.npz"),
                   f"{config.FUENTE}/prototipo/cache/malla_mensual_m{m}.npz"):
        if os.path.exists(origen):
            shutil.copy(origen, f"{E}/malla_mensual_m{m}.npz")
            break
    else:
        print(f"  aviso: sin caché mensual m{m} (se precomputaría del cubo; en GH no se puede)")

# 3. modelos
for f in ("xgb_v2_prototipo.ubj", "xgb_v2_prototipo_features.json"):
    shutil.copy(f"{config.FUENTE}/modelos/{f}", f"{E}/modelos/{f}")
for f in ("donde_dia_effis.ubj", "cuando.ubj", "donde_effis_c.ubj"):
    shutil.copy(f"{ce.MODELOS}/{f}", f"{E}/modelos/{f}")
shutil.copy(config.salida("dos_13_mapa_donde_effis_c.npz"), f"{E}/dos_13_mapa_donde_effis_c.npz")

# 4. jueces en vivo (seguimiento de la temporada 2026)
shutil.copy(f"{config.FUENTE}/prototipo/cache/municipios.json", f"{E}/municipios.json")
shutil.copy(f"{config.FUENTE}/prototipo/estaciones_prototipo.parquet",
            f"{E}/estaciones_prototipo.parquet")
for f in ("julio2026_effis.json",):
    if os.path.exists(config.salida(f)):
        shutil.copy(config.salida(f), f"{E}/{f}")
# vendor: copias literales con cabecera de procedencia
for origen, dest in ((f"{config.BASE}/01_datos/miteco/validar_miteco.py", "validar_miteco.py"),
                     (f"{config.BASE}/01_datos/miteco/parseo_y_chuncking.py",
                      "parseo_y_chuncking.py")):
    txt = open(origen).read()
    if dest == "validar_miteco.py":
        # en GitHub el maestro de municipios vive en estado/
        txt = txt.replace('MUNICIPIOS = DIR / "prototipo" / "cache" / "municipios.json"',
                          'MUNICIPIOS = DIR / "prototipo" / "cache" / "municipios.json"\n'
                          'if not MUNICIPIOS.exists():\n'
                          '    MUNICIPIOS = Path(__file__).resolve().parents[1] / "municipios.json"')
    cab = (f"# COPIA LITERAL de {os.path.relpath(origen, config.BASE)}\n# (exportada por "
           f"gh_exportar_estado.py para la cadena diaria; no editar aquí: editar el original y reexportar)\n")
    open(f"{E}/vendor/{dest}", "w").write(cab + txt)
open(f"{E}/vendor/__init__.py", "w").write("")

tot = sum(os.path.getsize(f) for f in glob.glob(f"{E}/**/*", recursive=True) if os.path.isfile(f))
print(f"estado/: {tot/1e6:.0f} MB")
for f in sorted(glob.glob(f"{E}/**/*", recursive=True)):
    if os.path.isfile(f):
        print(f"  {os.path.relpath(f, E):45s} {os.path.getsize(f)/1e6:6.1f} MB")
