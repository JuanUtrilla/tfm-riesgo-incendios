"""
Descarga el censo ganadero armonizado de Europa (Zenodo 11058509) y filtra
a España para los tres cortes disponibles (2000, 2010, 2020).

Fuente: "Harmonized ruminant livestock dataset for Europe" (Nature Sci. Data, 2024).
https://zenodo.org/records/11058509 — GeoPackage, sin registro, CC BY 4.0.

Uso:
    conda activate tfm_fuego
    python ganaderia_descarga.py

Requiere:
    - pip install geopandas pyogrio requests tqdm
"""

import logging
from pathlib import Path

import geopandas as gpd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RECORD_ID = "11058509"
FILES = {
    2000: "livestock2000.zip",
    2010: "livestock2010.zip",
    2020: "livestock2020.zip",
}
BASE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files/{{fname}}/content"

RAW_DIR = Path("ganaderia_zenodo/raw")
OUT_DIR = Path("ganaderia_zenodo")
PAIS = "esp"  # co_code de España en el dataset


def descargar(fname: str, destino: Path) -> None:
    if destino.exists():
        log.info("Ya existe %s, se omite descarga.", destino)
        return
    url = BASE_URL.format(fname=fname)
    log.info("Descargando %s ...", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(destino.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=fname) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(destino)


def extraer_gpkg(zip_path: Path, year: int) -> Path:
    gpkg_path = zip_path.with_suffix(".gpkg")
    if gpkg_path.exists():
        return gpkg_path
    import zipfile

    log.info("Extrayendo %s ...", zip_path.name)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(zip_path.parent)
    return gpkg_path


def filtrar_espana(gpkg_path: Path, year: int) -> Path:
    out_path = OUT_DIR / f"ganaderia_espana_{year}.gpkg"
    if out_path.exists():
        log.info("Ya existe %s, se omite filtrado.", out_path)
        return out_path

    layer = f"europe_livestock{year}"
    log.info("Filtrando España en %s (layer=%s) ...", gpkg_path.name, layer)
    gdf = gpd.read_file(gpkg_path, layer=layer, where=f"co_code = '{PAIS}'")
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(out_path, driver="GPKG")
    log.info("Guardado %s (%d unidades administrativas).", out_path, len(gdf))
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resultados = []
    for year, fname in FILES.items():
        zip_path = RAW_DIR / fname
        descargar(fname, zip_path)
        gpkg_path = extraer_gpkg(zip_path, year)
        out_path = filtrar_espana(gpkg_path, year)
        resultados.append(out_path)

    log.info("Listo. Cortes de España generados: %s", [str(p) for p in resultados])


if __name__ == "__main__":
    main()
