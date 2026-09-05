#!/usr/bin/env python3
"""Perímetros EFFIS del año 2025, solo España, para evaluar sobre 2025.

Misma consulta probada de descargar_verdad_operativa.py / puntuar_effis.py:
WFS abierto sin clave, capa anual ms:modis.ba.poly.2025, filtro de país en
local (el servidor no admite CQL de forma fiable). Una sola petición.
"""
import json
import os

import pandas as pd
import requests

DIR = os.path.dirname(os.path.abspath(__file__))
WFS = "https://maps.effis.emergency.copernicus.eu/effis"
CAMPOS = ["id", "FIREDATE", "LASTUPDATE", "COUNTRY", "PROVINCE", "COMMUNE",
          "AREA_HA", "CLASS"]
SALIDA = f"{DIR}/effis_ba_2025_ES.geojson"


def main():
    print("EFFIS: capa modis.ba.poly.2025 (puede tardar varios minutos)…",
          flush=True)
    r = requests.get(WFS, timeout=900, params={
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typename": "ms:modis.ba.poly.2025",
        "outputformat": "application/json; subtype=geojson",
        "srsname": "EPSG:4326"})
    r.raise_for_status()
    geo = json.loads(r.content.decode("utf-8"))
    fs = [f for f in geo.get("features", [])
          if (f["properties"].get("COUNTRY") or "").upper().startswith("ES")]
    if len(fs) < 100:
        raise SystemExit(f"respuesta sospechosa: {len(fs)} incendios; no guardo")
    for f in fs:
        f["properties"] = {k: f["properties"].get(k) for k in CAMPOS}
    with open(SALIDA, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": fs}, fh,
                  ensure_ascii=False)
    p = pd.to_datetime([f["properties"]["FIREDATE"] for f in fs],
                       format="ISO8601")
    areas = sorted(float(f["properties"].get("AREA_HA") or 0) for f in fs)
    print(f"  {len(fs)} perímetros ES · {p.min():%F} → {p.max():%F}")
    for u in (0, 10, 30, 100, 500):
        print(f"    ≥{u:4d} ha: {sum(1 for a in areas if a >= u):5d}")
    print(f"  guardado {os.path.basename(SALIDA)} "
          f"({os.path.getsize(SALIDA)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
