"""
Descarga y procesa las features "humanas/estructurales" del Modelo B que faltaban:
  - Carreteras (Natural Earth) → distancia a carretera por punto EGIF.
  - Población (WorldPop España 2020, 100m) → densidad de población por punto EGIF.
  - Uso del suelo (ESA WorldCover 2021, 10m, 12 teselas para la Península) → clase por punto EGIF.

Idempotente: si un fichero ya existe y es válido (se puede leer con rasterio/geopandas
sin error), no se vuelve a descargar. Pensado para lanzar con buena conexión y dejar
correr: en un entorno con ancho de banda limitado, las 12 teselas WorldCover (~1GB)
+ WorldPop (380MB) pueden tardar horas; con conexión normal son 5-10 minutos.

Uso:
    python descargar_features_extra.py
"""
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

BASE = Path("/home/charredgem/Desktop/Master/TFM_fuego")
RAW = BASE / "features_extra" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
(RAW / "worldcover").mkdir(exist_ok=True)
OUT = BASE / "features_extra"

BBOX_ESP = (-9.6, 35.9, 3.6, 43.9)  # lon_min, lat_min, lon_max, lat_max (península)


def curl(url: str, dest: Path):
    print(f"  descargando {dest.name} ...")
    subprocess.run(["curl", "-s", "-o", str(dest), url], check=True)


def tif_valido(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            src.read(1, window=((0, 10), (0, 10)))
            src.read(1, window=((src.height - 10, src.height), (0, 10)))
        return True
    except Exception:
        return False


# ── 1. Carreteras (Natural Earth) ────────────────────────────────────────
roads_gpkg = RAW / "roads_spain.gpkg"
if not roads_gpkg.exists():
    print("Carreteras (Natural Earth)...")
    for ext in ("shp", "shx", "dbf", "prj"):
        f = RAW / f"ne_10m_roads.{ext}"
        if not f.exists():
            curl(f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/10m_cultural/ne_10m_roads.{ext}", f)
    roads = gpd.read_file(RAW / "ne_10m_roads.shp")
    esp = roads[roads.geometry.intersects(box(*BBOX_ESP))].copy()
    esp.to_file(roads_gpkg, driver="GPKG")
    print(f"  {len(esp)} carreteras en bbox España")

# ── 2. Población (WorldPop España 2020) ──────────────────────────────────
pop_tif = RAW / "esp_ppp_2020.tif"
if not tif_valido(pop_tif):
    print("Población (WorldPop)...")
    curl("https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/ESP/esp_ppp_2020.tif", pop_tif)
    if not tif_valido(pop_tif):
        raise RuntimeError("esp_ppp_2020.tif incompleto tras la descarga — reintenta con mejor conexión")

# ── 3. Uso del suelo (ESA WorldCover, 12 teselas para la Península) ─────
TILES = [f"{lat}{lon}" for lat in ("N36", "N39", "N42") for lon in ("W009", "W006", "W003", "E000")]
for tile in TILES:
    f = RAW / "worldcover" / f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    if not tif_valido(f):
        print(f"WorldCover {tile}...")
        curl(f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{f.name}", f)
        if not tif_valido(f):
            print(f"  !! {tile} incompleta, reintentar más tarde")

# ── 4. Extracción sobre los puntos EGIF ──────────────────────────────────
print("Extrayendo features sobre EGIF...")
egif = pd.read_csv(BASE / "egif_civio.csv").dropna(subset=["lat", "lng"])
coords = list(zip(egif["lng"], egif["lat"]))

# distancia a carretera (ya calculado antes, se reusa si existe)
dist_csv = OUT / "egif_dist_carretera.csv"
if not dist_csv.exists():
    roads = gpd.read_file(roads_gpkg).to_crs("EPSG:25830")
    gdf = gpd.GeoDataFrame(egif, geometry=gpd.points_from_xy(egif["lng"], egif["lat"]), crs="EPSG:4326").to_crs("EPSG:25830")
    joined = gpd.sjoin_nearest(gdf, roads[["geometry"]], distance_col="dist_carretera_m")
    out = joined[["id", "dist_carretera_m"]].drop_duplicates("id")
    out["dist_carretera_km"] = (out["dist_carretera_m"] / 1000).round(2)
    out[["id", "dist_carretera_km"]].to_csv(dist_csv, index=False)

# población
if tif_valido(pop_tif):
    with rasterio.open(pop_tif) as src:
        pop = np.array([v[0] for v in src.sample(coords)])
        pop = np.where(pop < 0, np.nan, pop)
    pd.DataFrame({"id": egif["id"], "pop_density_2020": pop}).to_csv(OUT / "egif_poblacion.csv", index=False)
    print(f"  población: {np.isfinite(pop).sum()}/{len(pop)} puntos con dato")
else:
    print("  población: SALTADO (tif incompleto)")

# uso del suelo: se muestrea tesela a tesela (sin fusionar en memoria: cada
# tesela son 36000x36000 px, fusionar las 12 a la vez puede agotar la RAM)
tiles_validas = [RAW / "worldcover" / f"ESA_WorldCover_10m_2021_v200_{t}_Map.tif" for t in TILES]
tiles_validas = [t for t in tiles_validas if tif_valido(t)]
print(f"  uso del suelo: {len(tiles_validas)}/12 teselas válidas")
if tiles_validas:
    landcover = pd.Series([None] * len(egif), index=egif.index, dtype="object")
    for t in tiles_validas:
        with rasterio.open(t) as src:
            l, b, r, top = src.bounds
            dentro = egif[(egif["lng"] >= l) & (egif["lng"] < r) & (egif["lat"] >= b) & (egif["lat"] < top)]
            if dentro.empty:
                continue
            vals = [v[0] for v in src.sample(list(zip(dentro["lng"], dentro["lat"])))]
            landcover.loc[dentro.index] = vals
    pd.DataFrame({"id": egif["id"], "landcover_worldcover": landcover.values}).to_csv(OUT / "egif_landcover.csv", index=False)
    n_ok = landcover.notna().sum()
    print(f"  uso del suelo: {n_ok}/{len(landcover)} puntos con dato (resto = fuera de teselas descargadas)")

print("\nOK — revisa features_extra/*.csv. Borra features_extra/raw/ si ya no lo necesitas (varios GB).")
