#!/usr/bin/env python3
"""
Genera la capa base administrativa congelada de los mapas: `modelo/malla/
limites.npz` (~pocos cientos de KB) con los límites de comunidades autónomas
y provincias YA proyectados a píxeles de la malla 1 km EPSG:3035, más las
capitales de provincia.

Se ejecuta A MANO y en local (necesita shapely + red); GitHub Actions solo lee
el .npz, igual que con `estaticas.npz` / `mensual_m*.npz`. Regenerar solo si
cambia la malla.

Fuente de los polígonos: Natural Earth 10 m admin-1 (dominio público), campo
`region` = comunidad autónoma, `name` = provincia.

Contenido del .npz:
  ccaa_x/ccaa_y, prov_x/prov_y : polilíneas en coordenadas de píxel, con NaN
                                 como separador entre tramos (plot directo)
  cap_x/cap_y/cap_nombre       : capitales de provincia

Uso: python3 exportar_limites.py
"""

import json
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union

RAIZ = Path(__file__).parent
MALLA = RAIZ / "modelo" / "malla"
NE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
      "geojson/ne_10m_admin_1_states_provinces.geojson")

# Tolerancia de simplificación en metros. La malla es de 1 km: por debajo de
# ~800 m el detalle no se ve y solo engorda el fichero.
TOL = 800

# Capitales de provincia (50) + ciudades autónomas. Se validan al generar:
# cada punto debe caer dentro del polígono de SU provincia (ver comprobar()).
CAPITALES = [
    ("A Coruña", 43.362, -8.412), ("Albacete", 38.995, -1.858),
    ("Alicante", 38.345, -0.481), ("Almería", 36.840, -2.468),
    ("Ávila", 40.657, -4.700), ("Badajoz", 38.879, -6.970),
    ("Barcelona", 41.387, 2.170), ("Bilbao", 43.263, -2.935),
    ("Burgos", 42.344, -3.697), ("Cáceres", 39.476, -6.372),
    ("Cádiz", 36.530, -6.293), ("Castellón", 39.986, -0.037),
    ("Ciudad Real", 38.986, -3.927), ("Córdoba", 37.888, -4.779),
    ("Cuenca", 40.070, -2.137), ("Girona", 41.984, 2.825),
    ("Granada", 37.177, -3.598), ("Guadalajara", 40.633, -3.167),
    ("Huelva", 37.261, -6.944), ("Huesca", 42.140, -0.409),
    ("Jaén", 37.766, -3.791), ("Las Palmas", 28.124, -15.430),
    ("León", 42.599, -5.567), ("Lleida", 41.617, 0.620),
    ("Logroño", 42.466, -2.450), ("Lugo", 43.012, -7.556),
    ("Madrid", 40.417, -3.704), ("Málaga", 36.721, -4.421),
    ("Murcia", 37.984, -1.128), ("Ourense", 42.336, -7.864),
    ("Oviedo", 43.362, -5.845), ("Palencia", 42.009, -4.529),
    ("Palma", 39.570, 2.650), ("Pamplona", 42.813, -1.646),
    ("Pontevedra", 42.431, -8.645), ("Salamanca", 40.965, -5.664),
    ("San Sebastián", 43.318, -1.981), ("Santa Cruz de Tenerife", 28.468, -16.254),
    ("Santander", 43.463, -3.805), ("Segovia", 40.948, -4.118),
    ("Sevilla", 37.389, -5.984), ("Soria", 41.764, -2.465),
    ("Tarragona", 41.119, 1.245), ("Teruel", 40.344, -1.107),
    ("Toledo", 39.863, -4.028), ("Valencia", 39.470, -0.377),
    ("Valladolid", 41.652, -4.724), ("Vitoria", 42.847, -2.673),
    ("Zamora", 41.503, -5.745), ("Zaragoza", 41.649, -0.889),
    ("Ceuta", 35.889, -5.317), ("Melilla", 35.292, -2.938),
]

# Provincia (campo `name` de Natural Earth) que debe contener cada capital.
# Sirve de test: si una coordenada estuviera mal, el punto caería fuera.
PROVINCIA_DE = {
    "A Coruña": "La Coruña", "Albacete": "Albacete", "Alicante": "Alicante",
    "Almería": "Almería", "Ávila": "Ávila", "Badajoz": "Badajoz",
    "Barcelona": "Barcelona", "Bilbao": "Bizkaia", "Burgos": "Burgos",
    "Cáceres": "Cáceres", "Cádiz": "Cádiz", "Castellón": "Castellón",
    "Ciudad Real": "Ciudad Real", "Córdoba": "Córdoba", "Cuenca": "Cuenca",
    "Girona": "Gerona", "Granada": "Granada", "Guadalajara": "Guadalajara",
    "Huelva": "Huelva", "Huesca": "Huesca", "Jaén": "Jaén",
    "Las Palmas": "Las Palmas", "León": "León", "Lleida": "Lérida",
    "Logroño": "La Rioja", "Lugo": "Lugo", "Madrid": "Madrid",
    "Málaga": "Málaga", "Murcia": "Murcia", "Ourense": "Orense",
    "Oviedo": "Asturias", "Palencia": "Palencia", "Palma": "Baleares",
    "Pamplona": "Navarra", "Pontevedra": "Pontevedra",
    "Salamanca": "Salamanca", "San Sebastián": "Gipuzkoa",
    "Santa Cruz de Tenerife": "Santa Cruz de Tenerife",
    "Santander": "Cantabria", "Segovia": "Segovia", "Sevilla": "Sevilla",
    "Soria": "Soria", "Tarragona": "Tarragona", "Teruel": "Teruel",
    "Toledo": "Toledo", "Valencia": "Valencia", "Valladolid": "Valladolid",
    "Vitoria": "Álava", "Zamora": "Zamora", "Zaragoza": "Zaragoza",
    "Ceuta": "Ceuta", "Melilla": "Melilla",
}


def provincias_esp():
    """Features admin-1 de España desde Natural Earth."""
    r = requests.get(NE, timeout=180)
    r.raise_for_status()
    geo = json.loads(r.content.decode("utf-8"))
    esp = [f for f in geo["features"]
           if f["properties"].get("adm0_a3") == "ESP"]
    if len(esp) < 50:
        raise RuntimeError(f"Natural Earth devolvió {len(esp)} provincias")
    return esp


def comprobar(esp):
    """Cada capital debe caer dentro del polígono de su provincia."""
    from shapely.geometry import Point
    poli = {f["properties"]["name"]: shape(f["geometry"]) for f in esp}
    malas = []
    for nom, la, lo in CAPITALES:
        prov = PROVINCIA_DE[nom]
        g = poli.get(prov)
        # buffer de 0,02° (~2 km): las capitales costeras (Cádiz, Santander)
        # quedan a veces justo fuera del polígono por la resolución de la costa
        if g is None or not g.buffer(0.02).contains(Point(lo, la)):
            malas.append(f"{nom} → {prov}")
    if malas:
        raise RuntimeError("capitales fuera de su provincia: " + ", ".join(malas))
    print(f"OK: {len(CAPITALES)} capitales dentro de su provincia")


def a_pixeles(geom, tr, xs, ys):
    """Contorno de una geometría → lista de polilíneas en píxeles de malla."""
    borde = geom.simplify(TOL).boundary
    partes = getattr(borde, "geoms", [borde])
    salida = []
    for ln in partes:
        cx, cy = np.asarray(ln.coords).T
        ix = (cx - xs[0]) / (xs[1] - xs[0])
        iy = (cy - ys[0]) / (ys[1] - ys[0])
        salida.append((ix, iy))
    return salida


def concatenar(tramos, nx, ny):
    """Polilíneas → un par de arrays con NaN de separador, sin lo que cae
    fuera de la malla (Canarias, Marruecos, mar abierto)."""
    px, py = [], []
    margen = 20
    for ix, iy in tramos:
        dentro = ((ix > -margen) & (ix < nx + margen) &
                  (iy > -margen) & (iy < ny + margen))
        if not dentro.any():
            continue
        px.append(np.where(dentro, ix, np.nan)); px.append([np.nan])
        py.append(np.where(dentro, iy, np.nan)); py.append([np.nan])
    if not px:
        return np.array([], np.float32), np.array([], np.float32)
    return (np.concatenate(px).astype(np.float32),
            np.concatenate(py).astype(np.float32))


def main():
    estat = np.load(MALLA / "estaticas.npz")
    xs, ys = estat["x"], estat["y"]
    ny, nx = estat["is_spain"].shape

    esp = provincias_esp()
    comprobar(esp)

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    from shapely.ops import transform as sh_transform
    proyectar = lambda g: sh_transform(lambda a, b: tr.transform(a, b), g)

    geos = {f["properties"]["name"]: proyectar(shape(f["geometry"]))
            for f in esp}
    regiones = {}
    for f in esp:
        regiones.setdefault(f["properties"]["region"], []).append(
            geos[f["properties"]["name"]])

    prov = [t for g in geos.values() for t in a_pixeles(g, tr, xs, ys)]
    ccaa = [t for gs in regiones.values()
            for t in a_pixeles(unary_union(gs), tr, xs, ys)]
    prov_x, prov_y = concatenar(prov, nx, ny)
    ccaa_x, ccaa_y = concatenar(ccaa, nx, ny)

    cx, cy = tr.transform([lo for _, _, lo in CAPITALES],
                          [la for _, la, _ in CAPITALES])
    cap_x = ((np.asarray(cx) - xs[0]) / (xs[1] - xs[0])).astype(np.float32)
    cap_y = ((np.asarray(cy) - ys[0]) / (ys[1] - ys[0])).astype(np.float32)

    destino = MALLA / "limites.npz"
    np.savez_compressed(destino,
                        prov_x=prov_x, prov_y=prov_y,
                        ccaa_x=ccaa_x, ccaa_y=ccaa_y,
                        cap_x=cap_x, cap_y=cap_y,
                        cap_nombre=np.array([n for n, _, _ in CAPITALES]))
    print(f"{destino}: {destino.stat().st_size/1024:.0f} KB · "
          f"{len(prov_x)} vértices provincias, {len(ccaa_x)} CCAA, "
          f"{len(cap_x)} capitales")


if __name__ == "__main__":
    main()
