#!/usr/bin/env python3
"""
Descarga los perímetros de área quemada de EFFIS (Copernicus EMS) para España.

Por qué: la etiqueta de validación por defecto son detecciones VIIRS, que
mezclan incendio forestal con quema agrícola, industria y falsos positivos.
EFFIS publica SUPERFICIE CARTOGRAFIADA a partir de MODIS/Sentinel-2: es una
etiqueta mucho más cercana a lo que el modelo aprendió (EGIF).

Servicio: WFS abierto, sin registro ni API key.
  https://maps.effis.emergency.copernicus.eu/effis
Capas útiles (ver GetCapabilities): modis.ba.poly.season (temporada en curso),
modis.ba.poly.<año> (2016…2025), effis.nrt.ba.poly (NRT).

⚠️ LATENCIA: cartografiar un perímetro lleva días. Los 2-3 últimos días están
sistemáticamente incompletos — no valides sobre ellos (ver VALIDACION.md).

Licencia: datos EFFIS/Copernicus EMS, reutilizables citando la fuente
(© European Union, Copernicus Emergency Management Service — EFFIS).

Salida: data/effis_ba_<capa>.geojson (solo España) + resumen por consola.
Uso: python3 descargar_effis.py [--capa season] [--pais ES]
"""

import argparse
import json
from pathlib import Path

import requests

RAIZ = Path(__file__).parent
WFS = "https://maps.effis.emergency.copernicus.eu/effis"
CAMPOS = ["id", "FIREDATE", "LASTUPDATE", "COUNTRY", "PROVINCE", "COMMUNE",
          "AREA_HA", "CLASS"]


def descargar(capa):
    """GeoJSON crudo de la capa (toda Europa; el filtro por país es local:
    el servidor no admite filtros CQL de forma fiable)."""
    r = requests.get(WFS, timeout=600, params={
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typename": f"ms:modis.ba.poly.{capa}" if capa != "nrt"
                    else "ms:effis.nrt.ba.poly",
        "outputformat": "application/json; subtype=geojson",
        "srsname": "EPSG:4326"})
    r.raise_for_status()
    return json.loads(r.content.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capa", default="season",
                    help="season | nrt | 2016..2025")
    ap.add_argument("--pais", default="ES")
    args = ap.parse_args()

    geo = descargar(args.capa)
    feats = [f for f in geo["features"]
             if f["properties"].get("COUNTRY") == args.pais]
    for f in feats:                     # solo los campos que usamos
        f["properties"] = {k: f["properties"].get(k) for k in CAMPOS}

    destino = RAIZ / "data" / f"effis_ba_{args.capa}_{args.pais}.geojson"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(
        {"type": "FeatureCollection", "features": feats}, ensure_ascii=False))

    import pandas as pd
    p = pd.DataFrame([f["properties"] for f in feats])
    p["FIREDATE"] = pd.to_datetime(p["FIREDATE"], format="ISO8601")
    p["AREA_HA"] = p["AREA_HA"].astype(float)
    print(f"{destino}: {len(feats)} incendios · "
          f"{destino.stat().st_size/1e6:.1f} MB")
    print(f"rango {p['FIREDATE'].min():%F} → {p['FIREDATE'].max():%F} · "
          f"{p['AREA_HA'].sum():,.0f} ha · mediana {p['AREA_HA'].median():.0f} ha")
    print("último perímetro cartografiado:", p["LASTUPDATE"].max())


if __name__ == "__main__":
    main()
