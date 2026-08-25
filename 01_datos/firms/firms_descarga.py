"""
Descarga detecciones de incendios de NASA FIRMS (VIIRS_SNPP_SP)
para la Península Ibérica (España + Portugal), 2015–2024.

Uso:
    conda activate tfm_fuego
    python firms_descarga.py

Requiere:
    - FIRMS_MAP_KEY en el archivo .env
    - pip install requests pandas python-dotenv tqdm
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
MAP_KEY   = os.environ.get("FIRMS_MAP_KEY", "")
BASE_URL  = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Bounding box Península Ibérica completa (España + Portugal)
# lon_min, lat_min, lon_max, lat_max
BBOX      = "-9.5,35.9,4.5,43.8"

SENSOR    = "VIIRS_SNPP_SP"   # 375m resolución, histórico desde 2012

FECHA_INI = date(2015, 1, 1)
FECHA_FIN = date(2024, 12, 31)
CHUNK_DAYS = 5                # máximo que admite la API por petición

OUT_DIR   = Path("firms_data")
OUT_FINAL = Path("firms_iberia_2015_2024.parquet")

PAUSA     = 0.5               # segundos entre peticiones
MAX_REINTENTOS = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("firms_descarga.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def chunks_fechas(inicio: date, fin: date, dias: int):
    cur = inicio
    while cur <= fin:
        yield cur
        cur += timedelta(days=dias)


def nombre_chunk(d: date) -> str:
    return d.strftime("%Y%m%d")


def ya_descargado(d: date) -> bool:
    return (OUT_DIR / f"{nombre_chunk(d)}.csv").exists()


def descargar_chunk(d: date) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{MAP_KEY}/{SENSOR}/{BBOX}/5/{d}"
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException as e:
            log.warning("Error de red (intento %d): %s", intento, e)
            time.sleep(10 * intento)
            continue

        if r.status_code == 429:
            log.warning("Rate limit. Esperando 60s...")
            time.sleep(60)
            continue

        if r.status_code == 400:
            log.error("MAP_KEY inválida o parámetros incorrectos: %s", r.text[:200])
            return None

        if r.status_code != 200:
            log.warning("HTTP %d (intento %d)", r.status_code, intento)
            time.sleep(5 * intento)
            continue

        # La respuesta puede ser CSV vacío (solo cabecera) o con datos
        lines = r.text.strip().splitlines()
        if len(lines) <= 1:
            # Sin detecciones en este periodo
            return pd.DataFrame()

        try:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            return df
        except Exception as e:
            log.warning("Error parseando CSV (intento %d): %s", intento, e)
            time.sleep(5)

    return None

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not MAP_KEY:
        raise SystemExit(
            "ERROR: FIRMS_MAP_KEY no encontrada en .env\n"
            "Regístrate en https://firms.modaps.eosdis.nasa.gov/api/map_key/\n"
            "y añade FIRMS_MAP_KEY=tu_key al archivo .env"
        )

    OUT_DIR.mkdir(exist_ok=True)

    fechas = list(chunks_fechas(FECHA_INI, FECHA_FIN, CHUNK_DAYS))
    total  = len(fechas)
    log.info("Total de chunks a descargar: %d (%d días c/u)", total, CHUNK_DAYS)
    log.info("Sensor: %s | BBox: %s", SENSOR, BBOX)

    for i, d in enumerate(fechas, 1):
        if ya_descargado(d):
            log.info("[%d/%d] %s — ya descargado, omitido", i, total, d)
            continue

        log.info("[%d/%d] Descargando desde %s...", i, total, d)
        df = descargar_chunk(d)

        if df is None:
            log.error("Chunk %s fallido tras %d reintentos", d, MAX_REINTENTOS)
            continue

        path_chunk = OUT_DIR / f"{nombre_chunk(d)}.csv"
        df.to_csv(path_chunk, index=False)
        log.info("  → %d detecciones guardadas en %s", len(df), path_chunk)

        time.sleep(PAUSA)

    # ── Consolidar todos los chunks en un único Parquet ────────────────────────
    log.info("Consolidando chunks en %s...", OUT_FINAL)
    csvs = sorted(OUT_DIR.glob("*.csv"))
    if not csvs:
        log.error("No hay archivos CSV para consolidar.")
        return

    dfs = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        log.error("Todos los CSVs estaban vacíos.")
        return

    df_total = pd.concat(dfs, ignore_index=True)

    # Filtro de confianza: solo detecciones con confidence >= 'nominal'
    # VIIRS: confidence puede ser 'low', 'nominal', 'high'
    if "confidence" in df_total.columns:
        antes = len(df_total)
        df_total = df_total[df_total["confidence"].isin(["nominal", "high"])]
        log.info("Filtro confianza: %d → %d detecciones", antes, len(df_total))

    # Convertir fecha a datetime
    df_total["acq_date"] = pd.to_datetime(df_total["acq_date"])

    df_total.to_parquet(OUT_FINAL, index=False)
    log.info("Consolidación completa: %d detecciones totales en %s", len(df_total), OUT_FINAL)

    # Estadísticas rápidas
    print("\n── Resumen ──────────────────────────────────────")
    print(f"Total detecciones:  {len(df_total):,}")
    print(f"Rango fechas:       {df_total['acq_date'].min().date()} → {df_total['acq_date'].max().date()}")
    print(f"Columnas:           {list(df_total.columns)}")
    print(f"Archivo final:      {OUT_FINAL} ({OUT_FINAL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
