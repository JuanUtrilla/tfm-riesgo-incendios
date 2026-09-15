"""
Descarga datos climatológicos diarios de todas las estaciones AEMET.
Guarda en SQLite local. Resumible en cualquier momento.

Orden de descarga:
  1. Estaciones peninsulares (idema numérico, 0-9), las que tienen más datos
  2. Baleares (prefijo B)
  3. Canarias (prefijo C)

Uso:
    conda activate tfm_fuego
    python aemet_descarga_historico.py

Para detener: Ctrl+C  →  el script para limpiamente tras el chunk actual.
Para reanudar: volver a ejecutar el mismo comando.
"""

import os
import signal
import sys
import time
import sqlite3
import logging
import requests
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ───────────────────────────────────────────────────────────────
API_KEY    = os.environ["TOKEN_AEMET_2"]
BASE_URL   = "https://opendata.aemet.es/opendata/api"
DB_PATH    = "aemet_historico.db"
IDEMA_DESDE = "2775X"  # key2: segundo tercio (key1 cubre hasta 2766E, key3 desde 6172X)

FECHA_INI  = date(2015, 1, 1)
FECHA_FIN  = date(2024, 12, 31)
CHUNK_DAYS = 150              # 5 meses por petición (límite API AEMET: 6 meses)

PAUSA_MIN  = 1.5              # segundos mínimos entre peticiones (~40 req/min, bajo el límite de 50)
PAUSA_MAX  = 3.0              # techo reducido; no hace falta frenar tanto
BACKOFF_BASE = 30             # espera mucho menor al primer 429
MAX_REINTENTOS = 5

# ── Señal de parada limpia ──────────────────────────────────────────────────────
_parar = False

def _handler(sig, frame):
    global _parar
    print("\n\n⏸  Ctrl+C recibido — parando tras el chunk actual...")
    _parar = True

signal.signal(signal.SIGINT, _handler)
signal.signal(signal.SIGTERM, _handler)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [KEY2] %(message)s",
    handlers=[
        logging.FileHandler("aemet_descarga_key2.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Base de datos ───────────────────────────────────────────────────────────────
def init_db(con):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS estaciones (
            idema     TEXT PRIMARY KEY,
            nombre    TEXT,
            provincia TEXT,
            latitud   TEXT,
            longitud  TEXT,
            altitud   TEXT
        );

        CREATE TABLE IF NOT EXISTS climatologia_diaria (
            idema    TEXT,
            fecha    TEXT,
            tmed     REAL, tmax REAL, tmin REAL,
            hr       REAL, hrmax REAL, hrmin REAL,
            velmedia REAL, racha REAL, dir REAL,
            prec     REAL, sol  REAL,
            presmax  REAL, presmin REAL,
            PRIMARY KEY (idema, fecha)
        );

        CREATE TABLE IF NOT EXISTS progreso (
            idema     TEXT,
            fecha_ini TEXT,
            fecha_fin TEXT,
            estado    TEXT,   -- 'ok', 'sin_datos', 'error'
            PRIMARY KEY (idema, fecha_ini, fecha_fin)
        );

        CREATE INDEX IF NOT EXISTS idx_progreso_idema ON progreso(idema);
    """)
    con.commit()


def ya_descargado(con, idema, fi, ff):
    return con.execute(
        "SELECT 1 FROM progreso WHERE idema=? AND fecha_ini=? AND fecha_fin=? AND estado IN ('ok','sin_datos')",
        (idema, fi, ff),
    ).fetchone() is not None


def todos_sin_datos(con, idema):
    """True si todos los chunks de esta estación ya se intentaron y dieron sin_datos."""
    rows = con.execute(
        "SELECT estado FROM progreso WHERE idema=?", (idema,)
    ).fetchall()
    if not rows:
        return False
    return all(r[0] == "sin_datos" for r in rows)


def guardar_progreso(con, idema, fi, ff, estado):
    con.execute("INSERT OR REPLACE INTO progreso VALUES (?,?,?,?)", (idema, fi, ff, estado))
    con.commit()


def insertar_registros(con, registros):
    def n(r, campo):
        v = r.get(campo, "").replace(",", ".")
        try:
            return float(v) if v and v not in ("Ip", "Acum") else None
        except ValueError:
            return None

    filas = [(
        r.get("indicativo"), r.get("fecha", "")[:10],
        n(r,"tmed"), n(r,"tmax"), n(r,"tmin"),
        n(r,"hrMedia"), n(r,"hrMax"), n(r,"hrMin"),
        n(r,"velmedia"), n(r,"racha"), n(r,"dir"),
        n(r,"prec"), n(r,"sol"),
        n(r,"presMax"), n(r,"presMin"),
    ) for r in registros]

    con.executemany(
        "INSERT OR IGNORE INTO climatologia_diaria VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        filas,
    )
    con.commit()


# ── Cliente AEMET con pausa adaptativa ─────────────────────────────────────────
_pausa_actual = PAUSA_MIN

def _dormir(extra=0):
    time.sleep(_pausa_actual + extra)

def aemet_get(url):
    global _pausa_actual
    headers = {"api_key": API_KEY}
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            r = requests.get(url, headers=headers, timeout=20)
        except requests.RequestException as e:
            log.warning("Error de red (intento %d): %s", intento, e)
            time.sleep(10 * intento)
            continue

        if r.status_code == 429:
            espera = BACKOFF_BASE * intento
            _pausa_actual = min(_pausa_actual * 1.5, PAUSA_MAX)
            log.warning("Rate limit 429. Pausa actual: %.1fs. Esperando %ds...", _pausa_actual, espera)
            time.sleep(espera)
            continue

        if r.status_code != 200:
            log.error("HTTP %d en %s", r.status_code, url)
            return None

        # Sin 429 en esta petición, la pausa se reduce poco a poco
        _pausa_actual = max(_pausa_actual * 0.97, PAUSA_MIN)
        return r.json()

    log.error("Agotados reintentos para %s", url)
    return None


def fetch_datos(url_datos):
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            r = requests.get(url_datos, timeout=30)
            r.encoding = "latin-1"
            return r.json()
        except Exception as e:
            log.warning("Error fetching datos (intento %d): %s", intento, e)
            time.sleep(5 * intento)
    return None


def get_inventario():
    url = f"{BASE_URL}/valores/climatologicos/inventarioestaciones/todasestaciones"
    meta = aemet_get(url)
    if not meta or meta.get("estado") != 200:
        raise RuntimeError(f"No se pudo obtener inventario: {meta}")
    _dormir()
    return fetch_datos(meta["datos"]) or []


def get_climatologia_diaria(idema, fi, ff):
    url = (f"{BASE_URL}/valores/climatologicos/diarios/datos"
           f"/fechaini/{fi}/fechafin/{ff}/estacion/{idema}")
    meta = aemet_get(url)
    _dormir()
    if meta is None:
        return None
    estado = meta.get("estado")
    if estado == 404:
        desc = meta.get("descripcion", "")
        if "rango" in desc.lower() or "superior" in desc.lower():
            # Rango de fechas demasiado grande: error de configuración, no estación sin datos
            log.error("Rango inválido para %s %s-%s: %s", idema, fi, ff, desc)
            return None
        # Estación sin datos para ese período
        return []
    if estado != 200:
        log.warning("Estado %s para %s %s-%s: %s", estado, idema, fi, ff, meta.get("descripcion"))
        return None
    return fetch_datos(meta["datos"])


def chunks_fechas(inicio, fin, dias):
    cur = inicio
    while cur <= fin:
        sig = min(cur + timedelta(days=dias - 1), fin)
        yield cur, sig
        cur = sig + timedelta(days=1)


def fmt(d):
    return d.strftime("%Y-%m-%dT00:00:00UTC")


def ordenar_estaciones(estaciones):
    def prioridad(e):
        p = e["indicativo"][0]
        if p.isdigit(): return (0, e["indicativo"])
        if p == "B":    return (1, e["indicativo"])
        return              (2, e["indicativo"])
    ordenadas = sorted(estaciones, key=prioridad)
    # Key2: primero su tramo asignado (peninsulares entre IDEMA_DESDE y 6172X);
    # al agotarlo, sigue con el resto de estaciones para ayudar donde haga falta
    # (ya_descargado()/todos_sin_datos() saltan rápido lo que otra key ya hizo).
    tramo = [e for e in ordenadas if e["indicativo"][0].isdigit()
             and IDEMA_DESDE <= e["indicativo"] < "6172X"]
    tramo_ids = {e["indicativo"] for e in tramo}
    resto = [e for e in ordenadas if e["indicativo"] not in tramo_ids]
    return tramo + resto


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    global _parar

    con = sqlite3.connect(DB_PATH)
    init_db(con)

    log.info("Obteniendo inventario de estaciones AEMET...")
    estaciones_raw = get_inventario()
    estaciones = ordenar_estaciones(estaciones_raw)
    log.info("Estaciones: %d (peninsulares primero)", len(estaciones))

    # Guardar metadatos de estaciones
    con.executemany(
        "INSERT OR IGNORE INTO estaciones VALUES (?,?,?,?,?,?)",
        [(e["indicativo"], e.get("nombre"), e.get("provincia"),
          e.get("latitud"), e.get("longitud"), e.get("altitud"))
         for e in estaciones],
    )
    con.commit()

    n_chunks_total = len(estaciones) * len(list(chunks_fechas(FECHA_INI, FECHA_FIN, CHUNK_DAYS)))
    n_ya_hechos = con.execute("SELECT COUNT(*) FROM progreso WHERE estado IN ('ok','sin_datos')").fetchone()[0]

    log.info("Chunks totales: %d | Ya procesados: %d | Pendientes: %d",
             n_chunks_total, n_ya_hechos, n_chunks_total - n_ya_hechos)

    procesados = 0
    omitidos   = 0
    con_datos  = 0
    t_inicio   = time.time()

    for i, estacion in enumerate(estaciones):
        if _parar:
            break

        idema = estacion["indicativo"]
        grupo = "Península" if idema[0].isdigit() else ("Baleares" if idema[0]=="B" else "Canarias")

        # Si todos los chunks de esta estación ya se procesaron como sin_datos, saltar
        if todos_sin_datos(con, idema):
            omitidos += 3  # 3 chunks por estación aprox
            continue

        for fi_date, ff_date in chunks_fechas(FECHA_INI, FECHA_FIN, CHUNK_DAYS):
            if _parar:
                break

            fi_str = fmt(fi_date)
            ff_str = fmt(ff_date)

            if ya_descargado(con, idema, fi_str, ff_str):
                omitidos += 1
                continue

            log.info("[%s | %s] %s → %s  (pausa=%.1fs)",
                     idema, grupo, fi_date, ff_date, _pausa_actual)

            registros = get_climatologia_diaria(idema, fi_str, ff_str)

            if registros is None:
                guardar_progreso(con, idema, fi_str, ff_str, "error")
            elif len(registros) == 0:
                guardar_progreso(con, idema, fi_str, ff_str, "sin_datos")
            else:
                insertar_registros(con, registros)
                guardar_progreso(con, idema, fi_str, ff_str, "ok")
                con_datos += 1
                log.info("  → %d registros guardados", len(registros))

            procesados += 1

            # Progreso y ETA cada 50 chunks
            if procesados % 50 == 0:
                hechos_total = n_ya_hechos + procesados
                pct = hechos_total / n_chunks_total * 100
                elapsed = time.time() - t_inicio
                vel = procesados / elapsed if elapsed > 0 else 0
                pendientes = n_chunks_total - hechos_total
                eta_h = (pendientes / vel / 3600) if vel > 0 else 0
                log.info("━━ PROGRESO: %.1f%% (%d/%d chunks) | Con datos: %d | ETA: %.1fh",
                         pct, hechos_total, n_chunks_total, con_datos, eta_h)

    # Resumen final
    registros_total = con.execute("SELECT COUNT(*) FROM climatologia_diaria").fetchone()[0]
    estaciones_con_datos = con.execute(
        "SELECT COUNT(DISTINCT idema) FROM climatologia_diaria"
    ).fetchone()[0]

    if _parar:
        log.info("⏸  DESCARGA PAUSADA. Para reanudar: python aemet_descarga_historico.py")
    else:
        log.info("✅ DESCARGA COMPLETADA.")

    log.info("Chunks procesados esta sesión: %d | Omitidos (ya tenía): %d", procesados, omitidos)
    log.info("Registros en BD: %d | Estaciones con datos: %d", registros_total, estaciones_con_datos)
    con.close()


if __name__ == "__main__":
    main()
