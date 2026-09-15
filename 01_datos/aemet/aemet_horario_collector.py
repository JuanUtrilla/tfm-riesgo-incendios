"""
Recolector de observaciones horarias de AEMET OpenData.
Descarga las últimas ~24h de todas las estaciones y las inserta en SQLite.
Diseñado para ejecutarse cada 6h desde GitHub Actions.

Variables de entorno requeridas:
    AEMET_API_KEY: clave de la API de AEMET OpenData
"""

import os
import sqlite3
import logging
import requests
import time
from datetime import datetime, timezone
from pathlib import Path

API_KEY  = os.environ["AEMET_API_KEY"]
BASE_URL = "https://opendata.aemet.es/opendata/api"
DB_PATH  = Path(__file__).parent / "data" / "aemet_horario.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ── Base de datos ───────────────────────────────────────────────────────────────

def init_db(con: sqlite3.Connection):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS observacion_horaria (
            idema   TEXT,
            fint    TEXT,          -- fecha/hora ISO de la observación
            ta      REAL,          -- temperatura aire (°C)
            tamax   REAL,          -- temperatura máxima
            tamin   REAL,          -- temperatura mínima
            hr      REAL,          -- humedad relativa (%)
            vv      REAL,          -- velocidad viento (m/s)
            dv      REAL,          -- dirección viento (°)
            vmax    REAL,          -- racha máxima (m/s)
            prec    REAL,          -- precipitación (mm)
            pres    REAL,          -- presión (hPa)
            vis     REAL,          -- visibilidad (m)
            nombre  TEXT,
            ubi     TEXT,
            PRIMARY KEY (idema, fint)
        );

        CREATE TABLE IF NOT EXISTS ejecuciones (
            ts      TEXT PRIMARY KEY,   -- timestamp UTC de la ejecución
            nuevos  INTEGER,            -- registros nuevos insertados
            total   INTEGER             -- total de registros en BD tras inserción
        );

        CREATE INDEX IF NOT EXISTS idx_obs_idema ON observacion_horaria(idema);
        CREATE INDEX IF NOT EXISTS idx_obs_fint  ON observacion_horaria(fint);
    """)
    con.commit()


def insertar_observaciones(con: sqlite3.Connection, registros: list[dict]) -> int:
    def f(r, k):
        v = r.get(k)
        if v is None or v == "":
            return None
        try:
            return float(str(v).replace(",", "."))
        except (ValueError, TypeError):
            return None

    filas = [(
        r.get("idema"), r.get("fint"),
        f(r,"ta"), f(r,"tamax"), f(r,"tamin"),
        f(r,"hr"),
        f(r,"vv"), f(r,"dv"), f(r,"vmax"),
        f(r,"prec"), f(r,"pres"), f(r,"vis"),
        r.get("nombre"), r.get("ubi"),
    ) for r in registros if r.get("idema") and r.get("fint")]

    cur = con.executemany(
        "INSERT OR IGNORE INTO observacion_horaria VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        filas,
    )
    con.commit()
    return cur.rowcount


# ── Cliente AEMET ────────────────────────────────────────────────────────────────

def aemet_get(url: str, reintentos: int = 4) -> dict | None:
    headers = {"api_key": API_KEY}
    for intento in range(1, reintentos + 1):
        try:
            r = requests.get(url, headers=headers, timeout=20)
        except requests.RequestException as e:
            log.warning("Red (intento %d): %s", intento, e)
            time.sleep(10 * intento)
            continue

        if r.status_code == 429:
            espera = 65 * intento
            log.warning("Rate limit 429 — esperando %ds...", espera)
            time.sleep(espera)
            continue

        if r.status_code != 200:
            log.error("HTTP %d en %s", r.status_code, url)
            return None

        return r.json()

    log.error("Agotados reintentos para %s", url)
    return None


def fetch_datos(url: str, reintentos: int = 3) -> list | None:
    for intento in range(1, reintentos + 1):
        try:
            r = requests.get(url, timeout=30)
            r.encoding = "latin-1"
            datos = r.json()
            if isinstance(datos, list):
                return datos
            log.warning("Respuesta inesperada (no es lista): %s", str(datos)[:200])
            return None
        except Exception as e:
            log.warning("Error datos (intento %d): %s", intento, e)
            time.sleep(5 * intento)
    return None


def descargar_todas_estaciones() -> list[dict]:
    url = f"{BASE_URL}/observacion/convencional/todas"
    meta = aemet_get(url)
    if not meta or meta.get("estado") != 200:
        log.error("No se pudo obtener observaciones: %s", meta)
        return []
    time.sleep(1.5)
    return fetch_datos(meta["datos"]) or []


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    init_db(con)

    ts_inicio = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.info("Ejecución %s — descargando observaciones...", ts_inicio)

    registros = descargar_todas_estaciones()
    if not registros:
        log.warning("Sin datos en esta ejecución.")
        con.close()
        return

    log.info("Recibidos %d registros de la API.", len(registros))
    nuevos = insertar_observaciones(con, registros)
    total = con.execute("SELECT COUNT(*) FROM observacion_horaria").fetchone()[0]

    con.execute("INSERT OR REPLACE INTO ejecuciones VALUES (?,?,?)", (ts_inicio, nuevos, total))
    con.commit()

    log.info("Nuevos: %d | Total acumulado: %d registros", nuevos, total)
    con.close()


if __name__ == "__main__":
    main()
