#!/usr/bin/env python3
"""
Mapa de FOCOS ACTIVOS — evidencia satelital pura, sin modelo (job ligero de
GitHub Actions, 3×/día por la tarde).

Complementa al mapa de riesgo: el modelo predice a día vista y su JPG se
congela a las ~06:45; si por la tarde se declara un incendio, esta capa lo
enseña el mismo día. VIIRS (NOAA-20 + NOAA-21) detecta a 375 m con ~4
pasadas/día sobre España (~13:30 y ~01:30 solares por satélite) y el dato
NRT se publica ~3 h tras la pasada → un fuego de las 14:00 aparece ~17:00.

Límites (en el pie del mapa): no ve fuegos pequeños o bajo nubes, puede dar
falsos positivos (industria, solar). Evidencia, no servicio de emergencias.

Salida: mapas/focos_activos.jpg (README) — hoy en rojo, ayer en naranja.
Uso: FIRMS_MAP_KEY=... python3 focos_activos.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import firms_api

RAIZ = Path(__file__).parent
MALLA = RAIZ / "modelo" / "malla"

# misma capa base (límites CCAA/provincias + capitales) que el mapa de riesgo
from mapa_diario import capa_base


def detecciones():
    """VIIRS NOAA-20 + NOAA-21, hoy y ayer.

    Con un satélite basta para publicar (se avisa del que falta), pero si
    caen los dos se aborta: un mapa que dice "0 focos hoy" porque la API no
    respondió es peor que no actualizar el de la pasada anterior."""
    key = os.environ["FIRMS_MAP_KEY"]
    trozos, caidos = [], []
    for sat in ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]:
        try:
            d = firms_api.descargar(key, sat, 2)
        except firms_api.FirmsCaido as e:
            caidos.append(str(e))
            continue
        d["sat"] = sat
        trozos.append(d)
    if not trozos:
        raise firms_api.FirmsCaido("ningún satélite disponible: "
                                   + " | ".join(caidos))
    if caidos:
        print(f"aviso: mapa con un solo satélite — {caidos[0]}", flush=True)
    if not sum(len(d) for d in trozos):
        return pd.DataFrame()
    det = pd.concat(trozos, ignore_index=True)
    det["cuando"] = pd.to_datetime(
        det["acq_date"] + " " + det["acq_time"].astype(int).astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M")
    return det


def main():
    det = detecciones()
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    ahora = pd.Timestamp.utcnow().tz_localize(None)

    estat = np.load(MALLA / "estaticas.npz")
    elev = estat["elevacion"].copy()
    es_esp = estat["is_spain"].astype(bool)
    xs, ys = estat["x"], estat["y"]
    ny, nx = es_esp.shape
    elev[~es_esp] = np.nan

    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(elev, origin="lower" if ys[1] > ys[0] else "upper",
              cmap="Greys", vmin=-500, vmax=3500, alpha=0.65)

    n_hoy = n_ayer = 0
    ultima = None
    if len(det):
        fx, fy = tr.transform(det["longitude"].values, det["latitude"].values)
        dix = (fx - xs[0]) / (xs[1] - xs[0])
        diy = (fy - ys[0]) / (ys[1] - ys[0])
        dentro = (dix >= 0) & (dix < nx) & (diy >= 0) & (diy < ny)
        det, dix, diy = det[dentro], dix[dentro], diy[dentro]
        es_hoy = (pd.to_datetime(det["acq_date"]) == hoy).values
        n_hoy, n_ayer = int(es_hoy.sum()), int((~es_hoy).sum())
        if len(det):
            ultima = det["cuando"].max()
        ax.scatter(dix[~es_hoy], diy[~es_hoy], s=22, marker="o",
                   facecolors="none", edgecolors="darkorange", linewidths=1.2,
                   label=f"focos de AYER (n={n_ayer})")
        ax.scatter(dix[es_hoy], diy[es_hoy], s=55, marker="x", c="red",
                   linewidths=2, label=f"focos de HOY (n={n_hoy})")

    capa_base(ax, nx, ny)
    ax.set_xlim(0, nx)
    ax.set_ylim((ny, 0) if ys[1] < ys[0] else (0, ny))
    ax.legend(loc="lower right", fontsize=9)
    txt_ult = (f" · última detección {ultima:%H:%M} UTC" if ultima is not None
               else "")
    ax.set_title(f"Focos térmicos activos — evidencia satelital VIIRS 375 m "
                 f"(NOAA-20+21, sin modelo)\n"
                 f"actualizado {ahora:%Y-%m-%d %H:%M} UTC{txt_ult} · "
                 f"~4 pasadas/día, el dato se publica ~3 h tras la pasada",
                 fontsize=11)
    ax.text(0.01, 0.01,
            "No detecta fuegos pequeños o bajo nubes; puede haber falsos "
            "positivos (industria). Evidencia, no emergencias: ante un "
            "incendio, 112.", transform=ax.transAxes, fontsize=8,
            color="dimgray")
    ax.set_axis_off()
    fig.tight_layout()
    tmp = str(RAIZ / "mapas" / "focos_activos.png")
    fig.savefig(tmp, dpi=150)
    from PIL import Image
    Image.open(tmp).convert("RGB").save(RAIZ / "mapas" / "focos_activos.jpg",
                                        quality=82, optimize=True)
    os.remove(tmp)
    print(f"focos_activos.jpg: {n_hoy} focos hoy · {n_ayer} ayer")


if __name__ == "__main__":
    main()
