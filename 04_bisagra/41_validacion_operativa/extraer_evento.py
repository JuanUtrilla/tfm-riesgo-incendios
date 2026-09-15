#!/usr/bin/env python3
"""
Extracción evento a evento (geolocalizada y temporal) para comprobar la
hipótesis del modelo de riesgo antes de escalar a los 21k puntos EGIF.

Para cada incendio conocido descarga, en el punto exacto y una ventana temporal
alrededor de la fecha:
  - ERA5-Land (CDS)  : temperatura, punto de rocío, viento (u,v), precipitación, radiación  [horario]
  - FWI      (EWDS)  : Fire Weather Index                                                    [diario]

Agrega ERA5-Land a diario, calcula derivadas (HR, racha, días secos consecutivos),
lo cruza con el FWI y genera por evento:
  - eventos_validacion/<evento>.csv   (serie diaria)
  - eventos_validacion/<evento>.png   (gráfico con la fecha del incendio marcada)

Puerta de decisión: si el FWI y la sequedad suben antes del fuego respecto al
baseline de la ventana, la hipótesis se sostiene y se escala a EGIF completo.

Requisitos: cdsapi + ~/.cdsapirc con la key CDS/ECMWF, xarray, netCDF4, pandas, matplotlib.
Las descargas se cachean en eventos_validacion/_raw/ (relanzar es seguro y resumible).
"""
import os
import zipfile
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import cdsapi
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "eventos_validacion")
RAW = os.path.join(OUT, "_raw")
os.makedirs(RAW, exist_ok=True)

VENTANA_ANTES = 30   # días antes del incendio
VENTANA_DESPUES = 15  # días después

URL_CDS = "https://cds.climate.copernicus.eu/api"    # ERA5-Land
URL_EWDS = "https://ewds.climate.copernicus.eu/api"  # FWI

# Incendios candidatos. Coordenadas aproximadas; se refinan con EGIF Civio al escalar.
EVENTOS = [
    dict(nombre="galicia_ponteareas_2017", lat=42.17, lon=-8.50, fecha=date(2017, 10, 15),
         desc="Oleada de Galicia (Ponteareas, Rías Baixas)"),
    dict(nombre="moguer_donana_2017",      lat=37.22, lon=-6.83, fecha=date(2017, 6, 24),
         desc="Las Peñuelas / Moguer (entorno Doñana, Huelva)"),
    dict(nombre="sotalvo_avila_2021",      lat=40.60, lon=-4.95, fecha=date(2021, 8, 14),
         desc="Sotalvo (Ávila) — ya trabajado con Sentinel-2/FIRMS"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(OUT, "extraccion.log")),
              logging.StreamHandler()],
)
log = logging.getLogger("evento")


def _key():
    with open(os.path.expanduser("~/.cdsapirc")) as f:
        for ln in f:
            if ln.strip().startswith("key:"):
                return ln.split("key:", 1)[1].strip()
    raise RuntimeError("No encuentro 'key:' en ~/.cdsapirc")


KEY = _key()
cli_cds = cdsapi.Client(url=URL_CDS, key=KEY)
cli_ewds = cdsapi.Client(url=URL_EWDS, key=KEY)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def meses_en_ventana(ini, fin):
    """Lista de (año, mes) tocados por la ventana [ini, fin]."""
    out, y, m = [], ini.year, ini.month
    while (y, m) <= (fin.year, fin.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def descargar(cliente, dataset, request, destino):
    """Descarga con caché: si el fichero ya existe, no repite."""
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        log.info("  cache hit: %s", os.path.basename(destino))
        return destino
    log.info("  descargando %s ...", os.path.basename(destino))
    cliente.retrieve(dataset, request, destino)
    return destino


def abrir_nc(path):
    """Abre un .nc; si CDS lo devolvió como .zip, extrae y combina los .nc internos."""
    if zipfile.is_zipfile(path):
        carpeta = path + "_x"
        os.makedirs(carpeta, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(carpeta)
            ncs = [n for n in z.namelist() if n.endswith(".nc")]
        rutas = [os.path.join(carpeta, n) for n in ncs]
        if len(rutas) == 1:
            return xr.open_dataset(rutas[0])
        return xr.merge([xr.open_dataset(r) for r in rutas], compat="override")
    return xr.open_dataset(path)


def _norm(ds):
    """Normaliza nombres de coordenadas a lat/lon/time."""
    ren = {}
    for c in ds.coords:
        cl = c.lower()
        if cl in ("latitude", "lat"):
            ren[c] = "lat"
        elif cl in ("longitude", "lon"):
            ren[c] = "lon"
        elif cl in ("valid_time", "time", "forecast_reference_time"):
            ren[c] = "time"
    return ds.rename(ren)


def celda_mas_cercana(ds, lat, lon):
    ds = _norm(ds)
    return ds.sel(lat=lat, lon=lon, method="nearest")


# ---------------------------------------------------------------------------
# ERA5-Land: descarga horaria de la ventana + agregación diaria en el punto
# ---------------------------------------------------------------------------
ERA5_VARS = [
    "2m_temperature", "2m_dewpoint_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "total_precipitation", "surface_solar_radiation_downwards",
]


def era5_diario(ev, ini, fin):
    lat, lon = ev["lat"], ev["lon"]
    caja = [lat + 0.15, lon - 0.15, lat - 0.15, lon + 0.15]  # N,W,S,E (varias celdas 0.1º)
    frames = []
    for (y, m) in meses_en_ventana(ini, fin):
        dst = os.path.join(RAW, f"era5_{ev['nombre']}_{y}{m:02d}.nc")
        req = {
            "variable": ERA5_VARS,
            "year": str(y), "month": f"{m:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": caja,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        descargar(cli_cds, "reanalysis-era5-land", req, dst)
        ds = celda_mas_cercana(abrir_nc(dst), lat, lon)
        df = ds.to_dataframe().reset_index()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    # --- Desacumulación de tp/ssrd antes de recortar la ventana ---
    # En ERA5-Land tp y ssrd acumulan desde las 00 UTC del día de pronóstico: el
    # valor de las 00:00 contiene el día anterior completo y el de las 01:00 solo
    # la primera hora del día en curso. Incremento horario = diff, salvo a las
    # 01:00 (reinicio de la acumulación), donde el incremento es el propio valor.
    # Cada incremento pertenece al día de (time - 1h).
    acumuladas = {"tp": "prec_mm_h", "ssrd": "rad_J_h"}
    for var, col in acumuladas.items():
        inc = df[var].diff()
        inc[df["time"].dt.hour == 1] = df.loc[df["time"].dt.hour == 1, var]
        df[col] = inc.clip(lower=0)  # ruido numérico puede dar diffs < 0
    df["dia_acum"] = (df["time"] - pd.Timedelta(hours=1)).dt.date
    diario_acum = df.groupby("dia_acum").agg(
        prec_mm=("prec_mm_h", "sum"),
        rad_MJ=("rad_J_h", "sum"),
    )
    diario_acum["prec_mm"] *= 1000.0
    diario_acum["rad_MJ"] /= 1e6
    diario_acum.index = pd.to_datetime(diario_acum.index)

    df = df[(df["time"].dt.date >= ini) & (df["time"].dt.date <= fin)].copy()

    # Derivadas horarias
    df["tC"] = df["t2m"] - 273.15
    df["tdC"] = df["d2m"] - 273.15
    # HR (Magnus) a partir de T y punto de rocío
    a, b = 17.625, 243.04
    df["hr"] = 100 * (np.exp(a * df["tdC"] / (b + df["tdC"])) /
                      np.exp(a * df["tC"] / (b + df["tC"])))
    df["hr"] = df["hr"].clip(0, 100)
    df["viento"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    df["dia"] = df["time"].dt.date

    # Agregación diaria
    g = df.groupby("dia")
    diario = pd.DataFrame({
        "tmax": g["tC"].max(),
        "tmin": g["tC"].min(),
        "tmed": g["tC"].mean(),
        "hr_min": g["hr"].min(),
        "hr_med": g["hr"].mean(),
        "viento_med": g["viento"].mean(),
        "viento_max": g["viento"].max(),
    })
    diario.index = pd.to_datetime(diario.index)
    diario = diario.join(diario_acum)  # prec_mm y rad_MJ desacumulados arriba
    diario = diario[(diario.index.date >= ini) & (diario.index.date <= fin)]
    diario = diario.sort_index()

    # Días secos consecutivos (prec < 1 mm)
    seco = (diario["prec_mm"] < 1.0).astype(int)
    run = np.zeros(len(seco), dtype=int)
    acc = 0
    for i, s in enumerate(seco.values):
        acc = acc + 1 if s else 0
        run[i] = acc
    diario["dias_secos_consec"] = run
    return diario


# ---------------------------------------------------------------------------
# FWI (EWDS): descarga diaria de la ventana en el punto
# ---------------------------------------------------------------------------
def fwi_diario(ev, ini, fin):
    lat, lon = ev["lat"], ev["lon"]
    caja = [lat + 0.3, lon - 0.3, lat - 0.3, lon + 0.3]  # rejilla FWI ~0.25º
    frames = []
    for (y, m) in meses_en_ventana(ini, fin):
        dst = os.path.join(RAW, f"fwi_{ev['nombre']}_{y}{m:02d}.nc")
        req = {
            "product_type": "reanalysis",
            "variable": ["fire_weather_index"],
            "dataset_type": "consolidated_dataset",
            "system_version": ["4_1"],
            "year": [str(y)], "month": [f"{m:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "grid": ["0.25/0.25"],
            "area": caja,
            "data_format": "netcdf",
        }
        descargar(cli_ewds, "cems-fire-historical-v1", req, dst)
        ds = celda_mas_cercana(abrir_nc(dst), lat, lon)
        var = "fwinx" if "fwinx" in ds.data_vars else list(ds.data_vars)[0]
        s = ds[var].to_dataframe()[var]
        frames.append(s)
    s = pd.concat(frames)
    s.index = pd.to_datetime(pd.Series(s.index).dt.date.values)
    s = s.groupby(level=0).mean().sort_index()
    s = s[(s.index.date >= ini) & (s.index.date <= fin)]
    return s.rename("fwi")


# ---------------------------------------------------------------------------
# Gráfico + veredicto
# ---------------------------------------------------------------------------
def grafico(ev, df):
    f = pd.Timestamp(ev["fecha"])
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f"{ev['desc']}\nincendio: {ev['fecha']}  ({ev['lat']:.2f}, {ev['lon']:.2f})",
                 fontsize=12)

    axs[0].plot(df.index, df["fwi"], color="#c1121f", lw=2)
    axs[0].axhspan(0, 5, color="#2a9d8f", alpha=.06)
    axs[0].axhspan(21, 100, color="#e76f51", alpha=.06)
    axs[0].set_ylabel("FWI")

    axs[1].plot(df.index, df["tmax"], color="#e07a1f", lw=1.6, label="Tmax")
    axs[1].plot(df.index, df["tmed"], color="#f4a261", lw=1, label="Tmed")
    axs[1].set_ylabel("Temp (°C)"); axs[1].legend(fontsize=8, loc="upper left")

    axs[2].plot(df.index, df["hr_min"], color="#1d7874", lw=1.6, label="HR mín")
    axs[2].set_ylabel("HR mín (%)")
    ax2b = axs[2].twinx()
    ax2b.plot(df.index, df["dias_secos_consec"], color="#8d99ae", lw=1, ls="--", label="días secos")
    ax2b.set_ylabel("días secos consec.")

    axs[3].bar(df.index, df["prec_mm"], color="#457b9d", width=.8)
    axs[3].set_ylabel("Precip (mm)")

    for ax in axs:
        ax.axvline(f, color="black", ls=":", lw=1.5)
        ax.grid(alpha=.2)
    axs[0].annotate("incendio", (f, axs[0].get_ylim()[1]), fontsize=9,
                    ha="center", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(OUT, ev["nombre"] + ".png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def veredicto(ev, df):
    f = pd.Timestamp(ev["fecha"])
    if f not in df.index:
        f = df.index[df.index.get_indexer([f], method="nearest")[0]]
    prev = df[df.index < f]              # baseline anterior al incendio
    fwi_dia = df.loc[f, "fwi"]
    pct = (prev["fwi"] < fwi_dia).mean() * 100 if len(prev) else float("nan")
    log.info("VEREDICTO %s:", ev["nombre"])
    log.info("  FWI día incendio = %.1f (percentil %.0f del baseline previo)", fwi_dia, pct)
    log.info("  HR mín día = %.0f%% | baseline HR mín medio = %.0f%%",
             df.loc[f, "hr_min"], prev["hr_min"].mean() if len(prev) else float("nan"))
    log.info("  Tmax día = %.1f°C | días secos consec. = %d",
             df.loc[f, "tmax"], int(df.loc[f, "dias_secos_consec"]))


# ---------------------------------------------------------------------------
def procesar(ev):
    ini = ev["fecha"] - timedelta(days=VENTANA_ANTES)
    fin = ev["fecha"] + timedelta(days=VENTANA_DESPUES)
    log.info("=== %s  (%s a %s) ===", ev["nombre"], ini, fin)
    era = era5_diario(ev, ini, fin)
    fwi = fwi_diario(ev, ini, fin)
    df = era.join(fwi, how="outer").sort_index()
    csv = os.path.join(OUT, ev["nombre"] + ".csv")
    df.to_csv(csv)
    log.info("  guardado %s (%d días)", os.path.basename(csv), len(df))
    grafico(ev, df)
    veredicto(ev, df)
    return df


if __name__ == "__main__":
    for ev in EVENTOS:
        try:
            procesar(ev)
        except Exception as e:
            log.error("FALLO en %s: %s: %s", ev["nombre"], type(e).__name__, str(e)[:400])
    log.info("HECHO. Resultados en %s", OUT)
