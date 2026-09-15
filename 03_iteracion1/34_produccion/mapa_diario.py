#!/usr/bin/env python3
"""
Mapa nacional de riesgo de hoy (D0) y mañana (D1), job de GitHub Actions (TFM).

Port autocontenido de `mapa_riesgo_hoy.py`: no necesita el cubo
IberFire (29 GB). Las capas del cubo que el modelo usa por celda están
congeladas en `modelo/malla/` (ver `01_datos/cubo/exportar_malla_gh.py`):
estáticas (elevación, CLC, población...), climatología mensual 2020-24 de
vegetación/LST y densidad EGIF mismo-mes. Lo único que cambia cada día es la
meteo, y esa sale de las estaciones AEMET:

1. Serie diaria de todas las estaciones (ranking_diario.serie_diaria_todas:
   API climatologías + colector horario del repo, spin-up FWI 80 días).
2. Para D0/D1: predicción municipal AEMET por estación (1 fetch por municipio,
   throttled ~40 req/min, reintentos con espera si 429) y el FWI se propaga
   desde lo observado con la meteo prevista (diseño MODELO_B_BITACORA §19).
3. Features por estación y prob por estación (se archivan en
   `rankings/prevision_D{0,1}_<fecha>.csv`: el commit sella la predicción).
4. Las 19 features meteo se interpolan a la malla de 1 km (IDW k=8 en
   EPSG:3035, corrección de altitud −6,5 °C/km en temperaturas; vpd_max se
   recalcula) + capas congeladas + FIRMS NRT + calendario; con eso XGBoost
   produce `mapas/mapa_D0.jpg` y `mapas/mapa_D1.jpg` y la tabla top-20 del README.

Limitación (para la memoria): la resolución efectiva de la meteo es la densidad
de la red (~30-50 km); el 1 km lo aportan estáticas y vegetación.

Uso: python3 mapa_diario.py   (necesita AEMET_API_KEY; FIRMS_MAP_KEY opcional)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import firms_api
from fwi_canadiense import calcular_fwi_serie
from ranking_diario import UMBRALES, serie_diaria_todas

_DATOS = Path(os.environ.get("TFM_DATOS", Path(__file__).resolve().parents[2] / "datos"))
RAIZ = Path(os.environ.get("TFM_COLECTOR", _DATOS / "colector"))   # raíz del colector de AEMET
MALLA = RAIZ / "modelo" / "malla"
GAMMA = 0.0065                      # lapse rate °C/m para corrección de altitud
CORTES = [-1, 0.25, 0.55, 0.80, 2]
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]

# features meteo que se interpolan de estaciones a malla (vpd_max no: se
# recalcula tras corregir; lst no: climatología mensual de la malla)
FEATS_IDW = ["fwi", "fwi_pctl_local", "fwi_anom_sigma", "t2m_max", "t2m_min",
             "rh_min", "viento_max", "precip_dia", "precip_7d", "precip_15d",
             "precip_30d", "fwi_med_7d", "fwi_max_7d", "fwi_med_15d",
             "fwi_med_30d", "rh_min_med_7d", "t2m_max_med_7d",
             "viento_max_med_7d", "dias_sin_lluvia"]
FEATS_TEMP = {"t2m_max", "t2m_min", "t2m_max_med_7d"}   # corrección de altitud

# ------------------------------------------------- capa base administrativa

def capa_base(ax, nx, ny, capitales=True):
    """Dibuja límites de CCAA y provincias (solo la línea, sin topónimos de
    región) y las capitales de provincia, desde `modelo/malla/limites.npz`
    (congelado, ver exportar_limites.py). Las líneas van encima del raster
    porque un imshow opaco las taparía, pero con trazo fino y gris para que
    lean como fondo. Si el .npz no está, el mapa sale igual que antes."""
    import matplotlib.patheffects as pe
    ruta = MALLA / "limites.npz"
    if not ruta.exists():
        print("aviso: falta modelo/malla/limites.npz, mapa sin capa base",
              flush=True)
        return
    lim = np.load(ruta, allow_pickle=False)
    ax.plot(lim["prov_x"], lim["prov_y"], lw=0.35, color="0.35", alpha=0.55,
            zorder=2.5, solid_capstyle="round")
    ax.plot(lim["ccaa_x"], lim["ccaa_y"], lw=0.9, color="0.15", alpha=0.8,
            zorder=2.6, solid_capstyle="round")
    if not capitales:
        return
    cx, cy, noms = lim["cap_x"], lim["cap_y"], lim["cap_nombre"]
    dentro = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
    ax.plot(cx[dentro], cy[dentro], marker="o", ms=2.6, mfc="white",
            mec="black", mew=0.7, ls="", zorder=4,
            label="capitales de provincia")
    for x, y, nom in zip(cx[dentro], cy[dentro], noms[dentro]):
        ax.annotate(nom, (x, y), xytext=(3, 3), textcoords="offset points",
                    fontsize=6, color="black", zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.6,
                                                foreground="white")])


# ---------------------------------------------------------------- FIRMS NRT

def firms_detecciones():
    """Detecciones FIRMS NRT [D-5, D-1] sobre Iberia (excluye hoy). Vacío solo
    si no hay key (uso local); si la API falla, `descargar` levanta FirmsCaido
    en vez de devolver 0 detecciones; con las features de fuego a cero el
    mapa saldría igual pero mal, y se perdería la verificación del día
    (`verificar_prevision` no distingue 'sin focos' de 'sin datos')."""
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        return pd.DataFrame()
    det = firms_api.descargar(key, "VIIRS_NOAA20_NRT", 5)
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    return det[pd.to_datetime(det["acq_date"]) < hoy]


# ------------------------------------------------- forecast municipal AEMET

def forecast_municipio(cod):
    """Predicción diaria AEMET del municipio, como filas futuras (tmax, tmin,
    hr_min, viento_max m/s, prec). Precipitación prevista: la predicción da
    probabilidad y no cantidad, así que 0 si prob<60% y 2 mm si ≥60%
    (conservador hacia el riesgo; documentado). None si la API no responde."""
    key = os.environ["AEMET_API_KEY"]
    d = None
    for intento in range(3):
        time.sleep(1.2)             # ~40 req/min (límite AEMET ~50/min)
        try:
            r = requests.get(f"https://opendata.aemet.es/opendata/api/"
                             f"prediccion/especifica/municipio/diaria/{cod}",
                             headers={"api_key": key}, timeout=15).json()
        except Exception:
            time.sleep(10)
            continue
        if r.get("estado") == 200:
            try:
                d = requests.get(r["datos"], timeout=15).json()
            except Exception:
                d = None
            break
        if r.get("estado") == 429:
            # desde runners de GitHub los 429 son mucho más frecuentes que en
            # local (IPs de datacenter): espera corta y presupuesto global en
            # evaluar_estaciones; lo que no dé tiempo cae a Open-Meteo
            time.sleep(30)
            continue
        return None
    if not d:
        return None
    filas = []
    for dia in d[0]["prediccion"]["dia"]:
        t, h = dia.get("temperatura", {}), dia.get("humedadRelativa", {})
        vel = [x.get("velocidad") for x in dia.get("viento", []) if x.get("velocidad")]
        pp = [p.get("value") for p in dia.get("probPrecipitacion", [])
              if p.get("value") is not None]
        if t.get("maxima") is None:
            continue
        filas.append({
            "fecha": pd.Timestamp(dia["fecha"][:10]),
            "tmax": float(t["maxima"]), "tmin": float(t.get("minima", np.nan)),
            "hr_min": float(h.get("minima", np.nan)),
            "viento_max": max(vel) / 3.6 if vel else np.nan,   # km/h → m/s
            "prec": 2.0 if (pp and max(pp) >= 60) else 0.0, "n_horas": 24})
    return pd.DataFrame(filas)


def forecast_openmeteo(pend):
    """Fallback en bloque cuando el presupuesto AEMET se agota (los runners de
    GitHub sufren 429 masivos de AEMET; medido el 22/07: ~5× más lento que en
    local). Open-Meteo: gratuito, sin key, ~100 estaciones por petición.
    pend: DataFrame con idema/lat/lon. Devuelve {idema: DataFrame de filas}."""
    out = {}
    for i0 in range(0, len(pend), 100):
        ch = pend.iloc[i0:i0 + 100]
        datos = None
        for intento in range(4):
            try:
                r = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": ",".join(f"{v:.4f}" for v in ch["lat"]),
                        "longitude": ",".join(f"{v:.4f}" for v in ch["lon"]),
                        "daily": "temperature_2m_max,temperature_2m_min,"
                                 "precipitation_sum",
                        "hourly": "relative_humidity_2m,wind_speed_10m",
                        "forecast_days": 3, "timezone": "UTC",
                        "wind_speed_unit": "ms"},
                    timeout=60)
                d = r.json() if r.ok else None
                # un lote de 100 ubicaciones cuenta como ~100 llamadas del
                # límite por minuto (~600/min), de ahí el backoff largo si rechaza
                if d is not None and not (isinstance(d, dict) and d.get("error")):
                    datos = d
                    break
            except Exception:
                pass
            time.sleep(45)
        if datos is None:
            print(f"open-meteo: chunk {i0} sin respuesta", flush=True)
            continue
        time.sleep(10)                  # respiro entre lotes (límite/minuto)
        datos = datos if isinstance(datos, list) else [datos]
        for idema, d in zip(ch["idema"], datos):
            fechas = pd.to_datetime(d["daily"]["time"])
            rh = np.array(d["hourly"]["relative_humidity_2m"],
                          dtype=float).reshape(-1, 24)
            vv = np.array(d["hourly"]["wind_speed_10m"],
                          dtype=float).reshape(-1, 24)
            out[idema] = pd.DataFrame({
                "fecha": fechas,
                "tmax": np.array(d["daily"]["temperature_2m_max"], dtype=float),
                "tmin": np.array(d["daily"]["temperature_2m_min"], dtype=float),
                "prec": np.array(d["daily"]["precipitation_sum"], dtype=float),
                "hr_min": np.nanmin(rh, axis=1),
                "viento_max": np.nanmax(vv, axis=1),      # ya en m/s
                "n_horas": 24})
    return out


# --------------------------------------------- features por estación y fecha

def features_estacion(s, i, fwi_s, e, fecha, festivos):
    """Vector de features en el índice i de la serie s (misma definición que
    ranking_diario/tiempo_real). Las de FIRMS se añaden fuera (van en bloque)."""
    mes = fecha.month
    F = {"fwi": fwi_s[i], "t2m_max": s.loc[i, "tmax"],
         "t2m_min": s.loc[i, "tmin"], "rh_min": s.loc[i, "hr_min"],
         "viento_max": s.loc[i, "viento_max"], "precip_dia": s.loc[i, "prec"]}
    es_ = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
    F["vpd_max"] = es_ * (1 - F["rh_min"] / 100)
    pre, rh, tmx, vto = (s[c].values for c in
                         ["prec", "hr_min", "tmax", "viento_max"])
    F.update(precip_7d=np.nansum(pre[i-7:i]), precip_15d=np.nansum(pre[i-15:i]),
             precip_30d=np.nansum(pre[i-30:i]),
             fwi_med_7d=np.nanmean(fwi_s[i-7:i]), fwi_max_7d=np.nanmax(fwi_s[i-7:i]),
             fwi_med_15d=np.nanmean(fwi_s[i-15:i]), fwi_med_30d=np.nanmean(fwi_s[i-30:i]),
             rh_min_med_7d=np.nanmean(rh[i-7:i]), t2m_max_med_7d=np.nanmean(tmx[i-7:i]),
             viento_max_med_7d=np.nanmean(vto[i-7:i]))
    secos = 0
    for k in range(i, -1, -1):
        if np.isnan(pre[k]) or pre[k] >= 1.0 or secos >= 120:
            break
        secos += 1
    F["dias_sin_lluvia"] = secos
    clim = np.load(RAIZ / "modelo" / "clim_fwi" / f"{e.name}.npz")[f"m{mes}"]
    F["fwi_pctl_local"] = (float((clim <= F["fwi"]).mean() * 100)
                           if len(clim) else np.nan)
    F["fwi_anom_sigma"] = (float((F["fwi"] - clim.mean()) / clim.std())
                           if len(clim) and clim.std() > 0 else np.nan)
    for c in ["ndvi", "lai", "swi010", "lst"]:
        F[c] = e[f"{c}_m{mes}"]
    F["ndvi_med_30d"] = e[f"ndvi_m{mes}"]
    for c in ["elevacion", "pendiente", "rugosidad", "dist_carreteras",
              "dist_rios", "popdens", "clc_bosque", "clc_matorral",
              "clc_agricola", "clc_artificial", "clc_abierto",
              "clc_agric_hetero"]:
        F[c] = e[c]
    F["n_fuegos_10km_mismomes_hist"] = e[f"n_mismomes_m{mes}"]
    F["rayos_dia"], F["rayos_7d"] = 0.0, 0.0
    F["es_festivo"] = float(fecha.weekday() >= 5 or fecha.date() in festivos)
    F["mes"], F["dia_anio"] = mes, fecha.dayofyear
    F["ccaa"] = int(e["ccaa"]) if np.isfinite(e["ccaa"]) else -1
    return F


def evaluar_estaciones(objetivos, est):
    """Features por estación para cada fecha objetivo (con forecast si la
    fecha va más allá de lo observado). Forecast en dos niveles: AEMET
    municipal (primario, 1 fetch/municipio) hasta agotar un presupuesto de
    tiempo (PRESUPUESTO_AEMET_S, def. 35 min; en runners de GitHub los 429
    de AEMET son masivos y sin tope el job muere por timeout, medido el 22/07),
    y Open-Meteo en bloque para las estaciones restantes.
    Devuelve {objetivo: DataFrame} con features + idema/nombre/lat/lon."""
    import hashlib

    import holidays
    festivos = holidays.Spain(years=sorted({o.year for o in objetivos}))
    muni = pd.read_csv(MALLA / "estacion_municipio.csv", dtype={"cod": str}) \
             .set_index("idema")
    todas = serie_diaria_todas()
    fin = max(objetivos)
    presupuesto = float(os.environ.get("PRESUPUESTO_AEMET_S", 2100))

    # --- fase 1: serie reindexada y último día completo por estación
    series = {}
    for idema, s in todas.groupby("idema"):
        if idema not in est.index or s["tmax"].notna().sum() < 45:
            continue
        cal = pd.date_range(s["fecha"].min(), fin, freq="D")
        s = s.set_index("fecha").reindex(cal).rename_axis("fecha").reset_index()
        comp = s.index[(s["tmax"].notna()) & (s["n_horas"].fillna(24) >= 18)]
        if not len(comp):
            continue
        series[idema] = (s, s.loc[int(comp[-1]), "fecha"])

    # --- fase 2: forecasts. Orden por hash del idema: estable entre días pero
    # geográficamente mezclado (los idema van por provincias; si el
    # presupuesto corta, la cobertura AEMET sigue siendo nacional)
    necesitan = sorted(
        (i for i, (s, ult) in series.items() if ult < fin and i in muni.index),
        key=lambda i: hashlib.md5(i.encode()).hexdigest())
    cache_aemet, via_om = {}, []
    t0 = time.monotonic()
    for idema in necesitan:
        cod = muni.loc[idema, "cod"]
        if cod in cache_aemet:
            if cache_aemet[cod] is None:
                via_om.append(idema)
            continue
        if time.monotonic() - t0 > presupuesto:
            via_om.append(idema)
            continue
        cache_aemet[cod] = forecast_municipio(cod)
        if cache_aemet[cod] is None or cache_aemet[cod].empty:
            cache_aemet[cod] = None
            via_om.append(idema)
        if len(cache_aemet) % 100 == 0:
            print(f"  forecast AEMET: {len(cache_aemet)} municipios "
                  f"({time.monotonic()-t0:.0f}s)", flush=True)
    om = (forecast_openmeteo(est.loc[via_om].reset_index()
                             .rename(columns={"index": "idema"}))
          if via_om else {})
    print(f"forecast: {sum(v is not None for v in cache_aemet.values())} "
          f"municipios AEMET en {time.monotonic()-t0:.0f}s (presupuesto "
          f"{presupuesto:.0f}s) · {len(via_om)} estaciones vía Open-Meteo "
          f"({len(om)} resueltas)", flush=True)

    # --- fase 3: aplicar forecast y calcular features
    filas = {o: [] for o in objetivos}
    n_est = n_sin_fc = 0
    for idema, (s, ult) in series.items():
        e = est.loc[idema]
        fuente = "obs"
        if ult < fin:                            # hace falta forecast
            cod = muni.loc[idema, "cod"] if idema in muni.index else None
            fc, fuente = cache_aemet.get(cod), "aemet"
            if fc is None or fc.empty:
                fc, fuente = om.get(idema), "open-meteo"
            if fc is None or fc.empty:
                n_sin_fc += 1
                continue
            fc = fc[fc["fecha"] > ult].set_index("fecha")
            s = s.set_index("fecha")
            inter = s.index.intersection(fc.index)   # puede ser vacía si el
            for c in ["tmax", "tmin", "hr_min", "viento_max", "prec", "n_horas"]:
                s.loc[inter, c] = fc[c].reindex(inter)   # forecast cae fuera
            s = s.reset_index()
        fwi_s = calcular_fwi_serie(s["tmax"].values, s["hr_min"].values,
                                   s["viento_max"].values * 3.6,
                                   s["prec"].values,
                                   s["fecha"].dt.month.values)["fwi"]
        n_est += 1
        for obj in objetivos:
            m = s.index[s["fecha"] == obj]
            if not len(m):
                continue
            i = int(m[0])
            if not (np.isfinite(s.loc[i, "tmax"]) and np.isfinite(s.loc[i, "hr_min"])):
                continue
            F = features_estacion(s, i, fwi_s, e, obj, festivos)
            F.update(idema=idema, nombre=e["nombre"], lat=e["lat"], lon=e["lon"],
                     x3035=e["x3035"], y3035=e["y3035"],
                     fuente_forecast=fuente,
                     dias_forecast=max(0, int((obj - ult).days)))
            filas[obj].append(F)
    print(f"estaciones con serie válida: {n_est} · sin ninguna predicción: "
          f"{n_sin_fc}", flush=True)
    return {o: pd.DataFrame(f) for o, f in filas.items()}


# --------------------------------------------------------------- malla 1 km

def malla_firms(det, xs, ys, ny, nx):
    """frp_max_50km_7d y n_detec_50km_7d en malla desde las detecciones."""
    from pyproj import Transformer
    from scipy.ndimage import maximum_filter, uniform_filter
    gfrp, gn = np.zeros((ny, nx)), np.zeros((ny, nx))
    if len(det):
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        fx, fy = tr.transform(det["longitude"].values, det["latitude"].values)
        fix = np.rint((fx - xs[0]) / (xs[1] - xs[0])).astype(int)
        fiy = np.rint((fy - ys[0]) / (ys[1] - ys[0])).astype(int)
        ok = (fix >= 0) & (fix < nx) & (fiy >= 0) & (fiy < ny)
        np.maximum.at(gfrp, (fiy[ok], fix[ok]), det["frp"].values[ok])
        np.add.at(gn, (fiy[ok], fix[ok]), 1)
    k = 101
    return (maximum_filter(gfrp, size=k, mode="constant"),
            uniform_filter(gn, size=k, mode="constant") * k * k)


def frp_estaciones(det, rk):
    """FRP máx / nº detecciones a <50 km por estación (para el CSV/tabla)."""
    if not len(det):
        return 0.0, 0.0
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    fx, fy = tr.transform(det["longitude"].values, det["latitude"].values)
    arbol = cKDTree(np.column_stack([fx, fy]))
    vec = arbol.query_ball_point(np.column_stack([rk["x3035"], rk["y3035"]]),
                                 r=50000)
    frp = det["frp"].values
    return ([float(frp[v].max()) if v else 0.0 for v in vec],
            [float(len(v)) for v in vec])


def generar_mapa(obj, rk, estat, det, modelo, feats, salida,
                 det_overlay=None, etiqueta_focos="ayer"):
    """Interpola la meteo de rk a la malla, predice y guarda el JPG.
    det alimenta las features FIRMS de la malla (ventana D-5…D-1);
    det_overlay (si se da) es lo que se dibuja encima; permite pintar los
    focos del propio día evaluado en mapas retrospectivos sin contaminar
    las features (mostrar no es alimentar)."""
    from scipy.spatial import cKDTree
    es_esp = estat["is_spain"]
    xs, ys = estat["x"], estat["y"]
    ny, nx = es_esp.shape
    gx, gy = np.meshgrid(xs, ys)
    cel = np.column_stack([gx[es_esp], gy[es_esp]])
    arbol = cKDTree(np.column_stack([rk["x3035"], rk["y3035"]]))
    dist, j = arbol.query(cel, k=min(8, len(rk)), workers=-1)
    w = 1.0 / np.maximum(dist, 1.0) ** 2

    def interpolar(vals):
        v = vals[j]
        m = np.isfinite(v)
        ww = np.where(m, w, 0.0)
        campo = np.full((ny, nx), np.nan)
        with np.errstate(invalid="ignore"):
            campo[es_esp] = (ww * np.where(m, v, 0)).sum(1) / ww.sum(1)
        return campo

    F = {k: v for k, v in estat.items() if k not in ("is_spain", "x", "y")}
    z_est = rk["elevacion"].values
    for c in FEATS_IDW:
        if c in FEATS_TEMP:  # se baja al nivel del mar, se interpola y se devuelve a la altitud
            F[c] = interpolar(rk[c].values + GAMMA * z_est) - GAMMA * F["elevacion"]
        else:
            F[c] = interpolar(rk[c].values)
    F["fwi"] = np.clip(F["fwi"], 0, None)
    F["fwi_pctl_local"] = np.clip(F["fwi_pctl_local"], 0, 100)
    F["rh_min"] = np.clip(F["rh_min"], 0, 100)
    for c in ["precip_dia", "precip_7d", "precip_15d", "precip_30d",
              "viento_max", "dias_sin_lluvia", "fwi_med_7d", "fwi_max_7d",
              "fwi_med_15d", "fwi_med_30d", "viento_max_med_7d"]:
        F[c] = np.clip(F[c], 0, None)
    es_ = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
    F["vpd_max"] = es_ * (1 - F["rh_min"] / 100)

    mensual = dict(np.load(MALLA / f"mensual_m{obj.month}.npz"))
    for var, col in [("ndvi", "ndvi"), ("lai", "lai"),
                     ("swi010", "swi010"), ("lst", "lst")]:
        F[col] = mensual[var]
    F["ndvi_med_30d"] = mensual["ndvi"]     # mismo proxy que el prototipo
    F["n_fuegos_10km_mismomes_hist"] = mensual["n_fuegos_10km_mismomes_hist"]
    F["frp_max_50km_7d"], F["n_detec_50km_7d"] = malla_firms(det, xs, ys, ny, nx)
    F["rayos_dia"] = np.zeros((ny, nx)); F["rayos_7d"] = np.zeros((ny, nx))
    import holidays
    fest = holidays.Spain(years=[obj.year])
    F["es_festivo"] = np.full((ny, nx), float(obj.weekday() >= 5
                                              or obj.date() in fest))
    F["mes"] = np.full((ny, nx), obj.month)
    F["dia_anio"] = np.full((ny, nx), obj.dayofyear)

    X = pd.DataFrame({c: np.asarray(F[c], dtype=np.float32).ravel()
                      for c in feats})
    prob = modelo.predict_proba(X)[:, 1].reshape(ny, nx)
    prob[~es_esp] = np.nan

    p = prob[es_esp]
    cuenta = pd.cut(pd.Series(p), CORTES, labels=NIVELES).value_counts()
    pct = (cuenta / len(p) * 100).round(1)
    resumen = " · ".join(f"{n}: {pct[n]}%" for n in NIVELES)
    print(f"celdas por nivel — {resumen}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(prob, origin="lower" if ys[1] > ys[0] else "upper",
                   cmap="YlOrRd", vmin=0, vmax=1)
    cb = fig.colorbar(im, label="probabilidad del modelo (prevalencia de "
                                "diseño 25%; cortes de validación 2019)")
    cb.set_ticks([0, 0.25, 0.55, 0.80, 1])
    cb.set_ticklabels(["0 · BAJO", "0,25 · MODERADO", "0,55 · ALTO",
                       "0,80 · EXTREMO", "1"])
    cb.ax.hlines([0.25, 0.55, 0.80], 0, 1, colors="black", lw=0.8)
    six = np.rint((rk["x3035"].values - xs[0]) / (xs[1] - xs[0]))
    siy = np.rint((rk["y3035"].values - ys[0]) / (ys[1] - ys[0]))
    ax.scatter(six, siy, s=2, c="gray", alpha=0.35,
               label=f"estaciones AEMET (n={len(rk)}) — de ellas sale la "
                     f"meteo interpolada")
    capa_base(ax, nx, ny)
    # capa visual de verdad-terreno (nunca feature del día evaluado): focos
    # térmicos FIRMS de los últimos 5 días, el más reciente destacado
    det = det if det_overlay is None else det_overlay
    if len(det):
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        dx, dy = tr.transform(det["longitude"].values, det["latitude"].values)
        dix = np.rint((dx - xs[0]) / (xs[1] - xs[0]))
        diy = np.rint((dy - ys[0]) / (ys[1] - ys[0]))
        # solo focos dentro de la malla (el bbox FIRMS incluye Marruecos/mar,
        # que expandirían los ejes y encogerían el mapa)
        dentro = (dix >= 0) & (dix < nx) & (diy >= 0) & (diy < ny)
        det, dix, diy = det[dentro], dix[dentro], diy[dentro]
        reciente = (pd.to_datetime(det["acq_date"])
                    == pd.to_datetime(det["acq_date"]).max()).values
        ax.scatter(dix[~reciente], diy[~reciente], s=14, marker="o",
                   facecolors="none", edgecolors="royalblue", linewidths=0.8,
                   label=f"focos satelitales días previos D-5…D-2 "
                         f"(n={int((~reciente).sum())})")
        ax.set_xlim(0, nx); ax.set_ylim((ny, 0) if ys[1] < ys[0] else (0, ny))
        ax.scatter(dix[reciente], diy[reciente], s=30, marker="x",
                   c="blue", linewidths=1.4,
                   label=f"fuegos REALES detectados {etiqueta_focos} por "
                         f"satélite (FIRMS, n={int(reciente.sum())})")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9,
              title="el color = predicción · el azul = lo que pasó de verdad",
              title_fontsize=8.5)
    d_fc = int(rk["dias_forecast"].max())
    etiqueta = ("observado" if d_fc == 0 else
                "PREVISTO — forecast municipal AEMET + FWI propagado")
    gen = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    ax.set_title(f"Riesgo de incendio — {obj.date()} ({etiqueta})\n"
                 f"meteo de {len(rk)} estaciones AEMET interpolada (IDW + "
                 f"corr. altitud) · estáticas 1 km IberFire\n"
                 f"celdas: {resumen} · generado {gen} (cron GitHub Actions)",
                 fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    tmp = str(salida) + ".png"
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    from PIL import Image
    Image.open(tmp).convert("RGB").save(salida, quality=82, optimize=True)
    os.remove(tmp)
    print(f"guardado {salida}", flush=True)
    return resumen


# -------------------------------------------- verificación predicción frente a real

def verificar_prevision(det, fecha, ruta_csv):
    """Contrasta una previsión sellada (CSV commiteado antes de que ocurriera)
    con los focos FIRMS del día `fecha`. Cada foco se empareja con su estación
    más cercana (≤35 km, el mismo criterio que en la validación del TFM). La
    métrica principal es el lift = (% de focos en zonas previstas ALTO/EXTREMO)
    / (% de estaciones en ALTO/EXTREMO), es decir, cuánto mejor que el azar
    señaló el modelo dónde ardería. Es seguimiento en operación; la validación
    del trabajo es el replay de 2025-2026.
    Devuelve dict de métricas o None si no hay previsión/datos."""
    if not ruta_csv.exists() or not len(det):
        return None
    focos = det[pd.to_datetime(det["acq_date"]) == fecha]
    prev = pd.read_csv(ruta_csv)
    base_alto = float(np.isin(prev["nivel"], ["ALTO", "EXTREMO"]).mean() * 100)
    res = {"fecha_focos": str(fecha.date()), "n_focos": int(len(focos)),
           "n_verificables": 0, "prob_mediana_en_focos": np.nan,
           "pctl_mediana_en_focos": np.nan, "pct_focos_en_altoextremo": np.nan,
           "pct_estaciones_altoextremo": round(base_alto, 1), "lift": np.nan}
    if not len(focos):
        return res
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ex, ey = tr.transform(prev["lon"].values, prev["lat"].values)
    fx, fy = tr.transform(focos["longitude"].values, focos["latitude"].values)
    dist, j = cKDTree(np.column_stack([ex, ey])).query(
        np.column_stack([fx, fy]))
    ok = dist <= 35000
    if not ok.any():
        return res
    p = prev["prob"].values[j[ok]]
    base = prev["prob"].values
    pctl = np.array([(base <= v).mean() * 100 for v in p])
    alto = float(np.isin(prev["nivel"].values[j[ok]],
                         ["ALTO", "EXTREMO"]).mean() * 100)
    res.update(n_verificables=int(ok.sum()),
               prob_mediana_en_focos=round(float(np.median(p)), 3),
               pctl_mediana_en_focos=round(float(np.median(pctl)), 1),
               pct_focos_en_altoextremo=round(alto, 1),
               lift=round(alto / base_alto, 2) if base_alto > 0 else np.nan)
    return res


def tabla_verificacion_readme():
    """Inyecta las últimas verificaciones entre los marcadores del README."""
    ruta_v = RAIZ / "rankings" / "verificacion_diaria.csv"
    ruta = RAIZ / "README.md"
    txt = ruta.read_text()
    ini, fin = "<!-- VERIFICACION:INICIO -->", "<!-- VERIFICACION:FIN -->"
    if ini not in txt or fin not in txt or not ruta_v.exists():
        return
    v = pd.read_csv(ruta_v).drop_duplicates(
        ["fecha_focos", "prevision"], keep="last").tail(14)
    filas = ["", "| Día | Previsión | Focos | Pctl mediano del riesgo "
                 "previsto donde ardió | % focos en ALTO+EXTREMO | % territorio "
                 "en ALTO+EXTREMO | Lift |", "|---|---|---|---|---|---|---|"]
    for _, r in v.iterrows():
        filas.append(
            f"| {r['fecha_focos']} | {r['prevision']} | {int(r['n_focos'])} "
            f"| {r['pctl_mediana_en_focos']} | {r['pct_focos_en_altoextremo']}% "
            f"| {r['pct_estaciones_altoextremo']}% | ×{r['lift']} |")
    filas.append("")
    txt = txt.split(ini)[0] + ini + "\n".join(filas) + fin + txt.split(fin)[1]
    ruta.write_text(txt)
    print("README: tabla de verificación actualizada")


# ------------------------------------------------------------------- README

def tabla_readme(rk, obj):
    """Inyecta el top-20 de estaciones (D0) entre los marcadores del README."""
    ruta = RAIZ / "README.md"
    txt = ruta.read_text()
    ini, fin = "<!-- RANKING:INICIO -->", "<!-- RANKING:FIN -->"
    if ini not in txt or fin not in txt:
        print("README sin marcadores de ranking — tabla omitida")
        return
    top = rk.sort_values("prob", ascending=False).head(20)
    filas = ["", f"**Top-20 estaciones por riesgo — {obj.date()}** "
                 f"(actualizado {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')})",
             "", "| # | Estación | Prob | Nivel | FWI | Tmax | HRmin | Días sin lluvia | Focos <50 km (5 d) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for pos, (_, r) in enumerate(top.iterrows(), 1):
        filas.append(f"| {pos} | {r['nombre']} | {r['prob']:.3f} | {r['nivel']} "
                     f"| {r['fwi']:.1f} | {r['t2m_max']:.1f}°C | {r['rh_min']:.0f}% "
                     f"| {int(r['dias_sin_lluvia'])} | {int(r['n_detec_50km_7d'])} |")
    filas.append("")
    bloque = txt.split(ini)[0] + ini + "\n".join(filas) + fin + txt.split(fin)[1]
    ruta.write_text(bloque)
    print("README: tabla top-20 actualizada")


def main():
    import xgboost as xgb
    with open(RAIZ / "modelo" / "xgb_v2_prototipo_features.json") as fh:
        feats = json.load(fh)
    modelo = xgb.XGBClassifier()
    modelo.load_model(str(RAIZ / "modelo" / "xgb_v2_prototipo.ubj"))
    est = pd.read_parquet(RAIZ / "modelo" / "estaciones_prototipo.parquet") \
            .set_index("idema")

    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    objetivos = [hoy, hoy + pd.Timedelta(days=1)]
    rks = evaluar_estaciones(objetivos, est)
    det = firms_detecciones()
    estat = dict(np.load(MALLA / "estaticas.npz"))
    estat["is_spain"] = estat["is_spain"].astype(bool)

    (RAIZ / "mapas").mkdir(exist_ok=True)
    for h, obj in enumerate(objetivos):
        rk = rks[obj]
        # dias_forecast > 5: estación con la serie rancia (la predicción
        # municipal cubre ~7 días; propagar más es rellenar sobre huecos NaN)
        rk = rk[np.isfinite(rk["fwi"]) & np.isfinite(rk["t2m_max"])
                & np.isfinite(rk["rh_min"]) & np.isfinite(rk["elevacion"])
                & (rk["dias_forecast"] <= 5)].reset_index(drop=True)
        print(f"\n=== D{h} {obj.date()}: {len(rk)} estaciones válidas ===",
              flush=True)
        if len(rk) < 100:
            raise RuntimeError(f"solo {len(rk)} estaciones para D{h} — abortando")
        rk["frp_max_50km_7d"], rk["n_detec_50km_7d"] = frp_estaciones(det, rk)
        rk["prob"] = modelo.predict_proba(rk[feats])[:, 1].round(4)
        rk["nivel"] = rk["prob"].map(lambda p: next(n for u, n in UMBRALES if p >= u))

        cols = ["idema", "nombre", "lat", "lon", "prob", "nivel",
                "fuente_forecast", "dias_forecast",
                "fwi", "fwi_pctl_local", "fwi_anom_sigma", "t2m_max", "rh_min",
                "viento_max", "dias_sin_lluvia", "precip_30d", "n_detec_50km_7d"]
        salida_csv = RAIZ / "rankings" / f"prevision_D{h}_{obj.date()}.csv"
        rk.sort_values("prob", ascending=False)[cols].round(3) \
          .to_csv(salida_csv, index=False)
        print(f"{salida_csv.name}: {len(rk)} estaciones · "
              f"{rk['nivel'].value_counts().to_dict()}")

        salida_jpg = RAIZ / "mapas" / f"mapa_D{h}.jpg"
        generar_mapa(obj, rk, estat, det, modelo, feats, salida_jpg)
        # archivo diario: el nombre fijo se sobreescribe (README), la copia
        # fechada conserva el histórico navegable sin bucear en git
        import shutil
        (RAIZ / "mapas" / "historico").mkdir(exist_ok=True)
        shutil.copy(salida_jpg, RAIZ / "mapas" / "historico"
                    / f"{obj.date()}_D{h}.jpg")
        if h == 0:
            tabla_readme(rk, obj)

    # verificación de ayer: lo que el modelo dijo (CSV sellado por commit)
    # frente a los focos FIRMS que realmente hubo. D0 = dicho ayer por la
    # mañana; D1 = dicho anteayer para ayer (forecast puro).
    ayer = hoy - pd.Timedelta(days=1)
    vers = []
    for tipo in ["D0", "D1"]:
        v = verificar_prevision(
            det, ayer, RAIZ / "rankings" / f"prevision_{tipo}_{ayer.date()}.csv")
        if v is not None:
            vers.append({"prevision": tipo, **v})
    if vers:
        ruta_v = RAIZ / "rankings" / "verificacion_diaria.csv"
        pd.DataFrame(vers).to_csv(ruta_v, mode="a",
                                  header=not ruta_v.exists(), index=False)
        for v in vers:
            print(f"verificación {ayer.date()} ({v['prevision']}): "
                  f"{v['n_focos']} focos · pctl mediano "
                  f"{v['pctl_mediana_en_focos']} · lift {v['lift']}", flush=True)
        tabla_verificacion_readme()


if __name__ == "__main__":
    main()
