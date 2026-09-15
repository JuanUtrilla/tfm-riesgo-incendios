#!/usr/bin/env python3
"""
Publica los mapas nacionales de hoy y D+1 en el repositorio de publicación del colector de AEMET para
verlos desde el móvil (el README los embebe con nombre fijo).

Flujo (se lanza tras mapa_riesgo_hoy.py en el cron diario):
1. Toma eda/mapa_riesgo_rt_<hoy>.png y _<mañana>.png.
2. Los convierte a JPG (~200 KB) en <repo>/mapas/mapa_D0.jpg y mapa_D1.jpg
   (con nombre fijo el README siempre muestra el último; el histórico queda en git).
3. git add/commit/push (solo si hay cambios).

Uso: python3 publicar_mapas_gh.py
"""

import os
import subprocess

import pandas as pd
from PIL import Image

RAIZ = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../..")
DIR = os.environ.get("TFM_DATOS", f"{RAIZ}/datos")
# repositorio de publicación del colector de AEMET (clon local con permiso de push)
REPO = os.environ.get("TFM_COLECTOR", f"{DIR}/colector")


def main():
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    os.makedirs(f"{REPO}/mapas", exist_ok=True)
    publicados = []
    for h, nombre in [(0, "mapa_D0.jpg"), (1, "mapa_D1.jpg")]:
        fecha = (hoy + pd.Timedelta(days=h)).date()
        png = f"{DIR}/eda/mapa_riesgo_rt_{fecha}.png"
        if not os.path.exists(png):
            print(f"no existe {png} — omitido")
            continue
        img = Image.open(png).convert("RGB")
        img.save(f"{REPO}/mapas/{nombre}", quality=82, optimize=True)
        publicados.append((nombre, str(fecha)))
    if not publicados:
        raise SystemExit("nada que publicar")

    fechas = ", ".join(f"{n}={f}" for n, f in publicados)
    subprocess.run(["git", "add", "mapas/"], cwd=REPO, check=True)
    r = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO)
    if r.returncode == 0:
        print("sin cambios — no se commitea")
        return
    subprocess.run(["git", "commit", "-m", f"Mapas de riesgo {fechas}"],
                   cwd=REPO, check=True)
    # el bot de GitHub Actions pushea cada 6 h, así que se integra el remoto antes del push
    subprocess.run(["git", "pull", "--rebase"], cwd=REPO, check=True)
    subprocess.run(["git", "push"], cwd=REPO, check=True)
    print(f"publicado: {fechas}")


if __name__ == "__main__":
    main()
