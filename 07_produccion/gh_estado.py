#!/usr/bin/env python3
"""
Estado de la cadena diaria en un Release de GitHub: bajar y subir.

Por qué un Release y no git
---------------------------
El portátil se apaga a diario; la cadena corre en GitHub Actions. Allí no hay
disco persistente: lo que cada corrida necesita de la anterior (el reanálisis
diario acumulado desde el 1-may, los mapas de los últimos 45 días para
contrastarlos cuando llegue EFFIS, los CSV de veredictos, el geojson de EFFIS,
los partes de MITECO) tiene que vivir fuera. Commitearlo inflaría el repo
unos 50 MB/día; un Release admite assets grandes y se reescriben con --clobber.
Mismo patrón que el colector del proyecto hermano (`mapa_diario.yml`).

Dos assets en el release `estado`:
  estado_base.tar.gz    lo que no cambia: capas estáticas 2D del cubo, clim
                        FWI por nodo, nodos, mapeo IFS, cachés mensuales,
                        modelos, municipios, ficheros de la comparación de
                        julio. Lo genera `gh_exportar_estado.py` en local.
  estado_diario.tar.gz  lo que cambia cada día. Lo sube la corrida.

Uso:
    python gh_estado.py pull            # baja y desempaqueta los dos
    python gh_estado.py push            # empaqueta y sube el diario
    python gh_estado.py push --base     # (local) sube también el base
"""

import argparse
import os
import subprocess
import sys
import json
import tarfile
import time

import pandas as pd

import config

RELEASE = "estado"
BASE_DIR = config.ESTADO                    # estado/ (inmutable)
SAL = config.SALIDA
DIARIO = [                                  # rutas relativas a salida/
    "era5land_diario.nc", "effis_ba_season_ES.geojson",
    "puntuacion_effis.csv", "veredicto_acumulado.json",
    "veredicto_miteco.csv", "veredicto_miteco.json", "miteco",
    "miteco_incidentes.csv",
    "mapas_diarios", "ifs_historico.parquet", "rankings_justo.csv",
    "veredicto_estaciones.csv", "veredicto_estaciones.json",
    "historico_veredictos.csv",
]


def gh(*args, intentos=3):
    for i in range(intentos):
        r = subprocess.run(["gh", *args], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        print(f"  gh {' '.join(args[:3])} falló ({i+1}/{intentos}): "
              f"{r.stderr.strip()[:200]}", flush=True)
        time.sleep(20)
    return None


def asegurar_release():
    if gh("release", "view", RELEASE, intentos=1) is None:
        gh("release", "create", RELEASE, "--title", "estado de la cadena diaria",
           "--notes", "Assets reescritos por la cadena diaria; no editar a mano.")


def pull():
    os.makedirs(SAL, exist_ok=True)
    for nombre, dest in (("estado_base.tar.gz", config.BASE),
                         ("estado_diario.tar.gz", SAL)):
        if gh("release", "download", RELEASE, "--pattern", nombre,
              "--dir", "/tmp", "--clobber") is None:
            print(f"  {nombre}: no disponible", flush=True)
            continue
        with tarfile.open(f"/tmp/{nombre}") as t:
            t.extractall(dest)
        print(f"  {nombre} → {dest}", flush=True)
    os.makedirs(f"{SAL}/mapas_diarios", exist_ok=True)


def diario_remoto_mas_nuevo():
    """Fecha del asset diario del Release, o None si no se puede leer.

    31/08/2026: `push --base` sube también el diario. Lanzado desde un portátil
    con `salida/` congelada, eso pisó el estado acumulado que la cadena llevaba
    hasta ese día: se perdieron los mapas diarios del 27-ago al 1-sep, y con
    ellos la posibilidad de contrastarlos con EFFIS cuando llegaran sus
    perímetros. Los veredictos ya calculados se recuperaron de `publicado/`
    en git; los mapas no estaban en ningún sitio más. De ahí esta guarda.
    """
    r = gh("release", "view", RELEASE, "--json", "assets", intentos=1)
    if not r:
        return None
    try:
        for a in json.loads(r).get("assets", []):
            if a["name"] == "estado_diario.tar.gz":
                return pd.Timestamp(a["updatedAt"]).tz_convert("UTC").tz_localize(None)
    except Exception:
        return None
    return None


def diario_local_mas_nuevo():
    t = [os.path.getmtime(f"{SAL}/{r}") for r in DIARIO if os.path.exists(f"{SAL}/{r}")]
    return pd.Timestamp(max(t), unit="s") if t else None


def push(base=False, forzar=False):
    asegurar_release()
    rem, loc = diario_remoto_mas_nuevo(), diario_local_mas_nuevo()
    viejo_local = rem is not None and loc is not None and rem > loc
    if viejo_local and not forzar:
        aviso = (f"el diario del Release es más nuevo que tu salida/ "
                 f"({rem:%F %H:%M} > {loc:%F %H:%M} UTC): subirlo pisaría el "
                 f"estado acumulado de la cadena, y los mapas diarios NO están "
                 f"en ningún otro sitio")
        # Abortar solo en el camino manual (`--base`), que es el que provocó la
        # pérdida. El `push` a secas lo ejecuta la cadena con `if: always()`
        # para salvar el progreso aunque un paso haya fallado: convertirlo en
        # error rompería runs que hoy terminan bien. Ahí basta con avisar.
        if base:
            raise SystemExit(f"ABORTADO: {aviso}.\nHaz `gh_estado.py pull` "
                             f"primero, o --forzar si de verdad quieres pisarlo.")
        print(f"  ⚠️  {aviso}", flush=True)
    if base:
        with tarfile.open("/tmp/estado_base.tar.gz", "w:gz") as t:
            t.add(BASE_DIR, arcname="estado")
        print(f"  base: {os.path.getsize('/tmp/estado_base.tar.gz')/1e6:.0f} MB", flush=True)
        gh("release", "upload", RELEASE, "/tmp/estado_base.tar.gz", "--clobber")
    with tarfile.open("/tmp/estado_diario.tar.gz", "w:gz") as t:
        for r in DIARIO:
            if os.path.exists(f"{SAL}/{r}"):
                t.add(f"{SAL}/{r}", arcname=r)
    print(f"  diario: {os.path.getsize('/tmp/estado_diario.tar.gz')/1e6:.0f} MB", flush=True)
    gh("release", "upload", RELEASE, "/tmp/estado_diario.tar.gz", "--clobber")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("accion", choices=["pull", "push"])
    p.add_argument("--base", action="store_true")
    p.add_argument("--forzar", action="store_true",
                   help="sube el diario aunque el del Release sea más nuevo")
    a = p.parse_args()
    pull() if a.accion == "pull" else push(a.base, a.forzar)
