#!/usr/bin/env python3
"""
Descarga las verdades-terreno para la evaluación operativa del modelo.

Trae dos fuentes; ninguna toca el repo del colector (se guardan en dataset/ de
este proyecto):

1. EFFIS: perímetros de área quemada cartografiados (Copernicus EMS).
   WFS abierto, sin clave. Es la mejor etiqueta disponible hoy: geometría real
   del incendio (ni el centro de un municipio ni un píxel térmico), fecha
   estimada de inicio y superficie en hectáreas, que permite fijar un umbral de
   completitud defendible en vez de asumir que la fuente lo ve todo.
   Ojo con la latencia: cartografiar un perímetro lleva días; los últimos 2-3
   días están sistemáticamente incompletos y no deben validarse.

2. FIRMS: detecciones VIIRS NOAA-20/21 del periodo de evaluación, para poder
   reconstruir eventos (clústeres) en vez de contar píxeles. La copia local
   llega solo hasta 2026-07-15, antes de la ventana de previsiones selladas.

Uso:
    python3 descargar_verdad_operativa.py                    # ambas
    python3 descargar_verdad_operativa.py --solo effis
    python3 descargar_verdad_operativa.py --desde 2026-07-15 --hasta 2026-08-11

Salidas: dataset/effis_ba_2026.geojson · dataset/firms_ventana_2026.parquet
"""

import argparse
import json
import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DIR = Path("/home/charredgem/Desktop/Master/TFM_fuego")
WFS = "https://maps.effis.emergency.copernicus.eu/effis"
BBOX = "-10,35,5,44"
SATS = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
CAMPOS = ["id", "FIREDATE", "LASTUPDATE", "COUNTRY", "PROVINCE", "COMMUNE",
          "AREA_HA", "CLASS"]


def effis(capa: str = "season", pais: str = "ES") -> None:
    """Perímetros de la temporada en curso, filtrados a España en local
    (el servidor no admite filtros CQL de forma fiable)."""
    print(f"EFFIS: descargando capa '{capa}' (puede tardar varios minutos)…",
          flush=True)
    r = requests.get(WFS, timeout=900, params={
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typename": f"ms:modis.ba.poly.{capa}",
        "outputformat": "application/json; subtype=geojson",
        "srsname": "EPSG:4326"})
    r.raise_for_status()
    geo = r.json()
    feats = [f for f in geo["features"]
             if (f["properties"].get("COUNTRY") or "").upper().startswith(pais)]
    salida = DIR / "dataset" / f"effis_ba_{capa}_2026.geojson"
    salida.write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                      encoding="utf-8")

    fechas = sorted((f["properties"].get("FIREDATE") or "")[:10] for f in feats)
    areas = sorted(float(f["properties"].get("AREA_HA") or 0) for f in feats)
    print(f"  {len(feats)} perímetros de {pais} · {fechas[0]} → {fechas[-1]}")
    for u in (0, 10, 30, 100, 500):
        print(f"    ≥{u:4d} ha: {sum(1 for a in areas if a >= u):5d}")
    print(f"  guardado {salida.name} ({salida.stat().st_size/1e6:.1f} MB)",
          flush=True)


def firms(desde: str, hasta: str) -> None:
    """Detecciones VIIRS del periodo. La API sirve como mucho 5 días por
    petición ('Invalid day range. Expects [1..5]')."""
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        for linea in (DIR / ".env").read_text().splitlines():
            if linea.startswith("FIRMS_MAP_KEY="):
                key = linea.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("falta FIRMS_MAP_KEY (variable de entorno o .env)")

    d0, d1 = pd.Timestamp(desde), pd.Timestamp(hasta)
    trozos = []
    for sat in SATS:
        d = d0
        while d <= d1:
            n = min(5, (d1 - d).days + 1)
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
                   f"{sat}/{BBOX}/{n}/{d:%Y-%m-%d}")
            r = requests.get(url, timeout=180)
            if r.ok and r.text.startswith("latitude"):
                t = pd.read_csv(StringIO(r.text))
                t["satellite_src"] = sat
                trozos.append(t)
                print(f"  {sat} {d:%F} +{n}d → {len(t)} detecciones", flush=True)
            else:
                print(f"  aviso: {sat} {d:%F} sin datos", file=sys.stderr)
            d += pd.Timedelta(days=n)
    if not trozos:
        sys.exit("FIRMS no devolvió detecciones")
    det = pd.concat(trozos, ignore_index=True)
    det["acq_date"] = pd.to_datetime(det["acq_date"])
    det = det.drop_duplicates(subset=["latitude", "longitude", "acq_date",
                                      "acq_time"])
    salida = DIR / "dataset" / "firms_ventana_2026.parquet"
    det.to_parquet(salida, index=False)
    print(f"FIRMS: {len(det)} detecciones únicas "
          f"{det.acq_date.min():%F} → {det.acq_date.max():%F} → {salida.name}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", choices=["effis", "firms"])
    ap.add_argument("--desde", default="2026-07-10")
    ap.add_argument("--hasta", default=str(pd.Timestamp.today().date()))
    a = ap.parse_args()
    if a.solo != "firms":
        effis()
    if a.solo != "effis":
        firms(a.desde, a.hasta)


if __name__ == "__main__":
    main()
