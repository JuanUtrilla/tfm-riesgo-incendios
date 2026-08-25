#!/usr/bin/env python3
"""
Motor de predicción en TIEMPO REAL del prototipo: estación AEMET → riesgo de hoy.

Flujo (diseño en MODELO_B_BITACORA §14):
1. Serie diaria desde el 1-may: API AEMET climatologías diarias (lag 3-4 días,
   caché 6 h) + colector horario agregado a diario para el hueco final y hoy.
2. FWI propio (fwi_canadiense) sobre esa serie → fwi del día + ventanas.
3. Percentil/anomalía vs climatología FWI-propio 2008-14 de la celda (npz
   precomputado — mismo algoritmo, el sesgo de implementación se cancela).
4. Vegetación: climatología mensual 2020-24 de la celda. Estáticas: precomputadas.
   Autorregresivas: congeladas a 2020 (las de 90 d → NaN, no observables).
   FIRMS NRT para frp/n_detec 50 km (caché diaria; fallback NaN). Rayos → NaN.
5. Predicción con xgb_v1_tuned + SHAP local.

Uso CLI de prueba: python3 tiempo_real.py <idema>   (p. ej. 3195 Madrid Retiro)
"""

import json
import os
import sqlite3
import time as _t
from io import StringIO

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from entrenar_modelo import (FEATS_METEO, FEATS_VEG, FEATS_ESTAT, FEATS_HIST,
                             FEATS_CAL)
from fwi_canadiense import calcular_fwi_serie

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
DB_COLECTOR = "/home/charredgem/Desktop/Master/aemet_horario_verano2026/data/aemet_horario_verano2026.db"
DIR_CACHE = f"{DIR}/prototipo/cache"
# El prototipo usa xgb_v2_prototipo: el modelo SIN las autorregresivas
# intra-celda (n_fuegos_1km_hist/90d, 10km_90d/365d), que en el dataset llevan
# un artefacto del muestreo misma-celda (el positivo se suma al historial de
# los negativos posteriores → el modelo aprende orden temporal, no física).
# Coste medido: AUC-PR 0,843→0,828. Detalle: MODELO_B_BITACORA §16.
with open(f"/home/charredgem/Desktop/Master/TFM_fuego/modelos/"
          f"xgb_v2_prototipo_features.json") as _f:
    FULL = json.load(_f)
INICIO_SERIE = "2026-05-01"          # ~75 días de spin-up del FWI
load_dotenv(f"{DIR}/.env")

_EST = None
_MODELO = None


def _estaciones():
    global _EST
    if _EST is None:
        _EST = pd.read_parquet(f"{DIR}/prototipo/estaciones_prototipo.parquet") \
                 .set_index("idema")
    return _EST


def _modelo():
    global _MODELO
    if _MODELO is None:
        import xgboost as xgb
        _MODELO = xgb.XGBClassifier()
        _MODELO.load_model(f"{DIR}/modelos/xgb_v2_prototipo.ubj")
    return _MODELO


def _cache_json(ruta, max_edad_s, generador):
    if os.path.exists(ruta) and _t.time() - os.path.getmtime(ruta) < max_edad_s:
        with open(ruta) as f:
            return json.load(f)
    datos = generador()
    if datos is not None:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, "w") as f:
            json.dump(datos, f)
    return datos


def serie_diaria_aemet(idema):
    """API climatologías diarias 1-may→hoy (caché 6 h). Devuelve DataFrame."""
    def descargar():
        key = os.environ["TOKEN_AEMET"]
        hoy = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        url = (f"https://opendata.aemet.es/opendata/api/valores/climatologicos/"
               f"diarios/datos/fechaini/{INICIO_SERIE}T00:00:00UTC/"
               f"fechafin/{hoy}T23:59:59UTC/estacion/{idema}")
        r = requests.get(url, headers={"api_key": key}, timeout=15).json()
        if r.get("estado") != 200:
            return None
        return requests.get(r["datos"], timeout=15).json()

    datos = _cache_json(f"{DIR_CACHE}/diario_{idema}.json", 6 * 3600, descargar)
    if not datos:
        return pd.DataFrame()
    df = pd.DataFrame(datos)

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")

    # viento_max del modelo = wind_speed_max del cubo (máx. de MEDIAS horarias),
    # NO la racha (el FWI explota con rachas: ISI ~ exp(0.05·v)). Aproximación
    # para días de la API diaria (sin horarios): 1.5×velmedia, acotado por racha.
    velmedia = num(df.get("velmedia", pd.Series(dtype=float)))
    racha = num(df.get("racha", pd.Series(dtype=float)))
    out = pd.DataFrame({
        "fecha": pd.to_datetime(df["fecha"]),
        "tmax": num(df["tmax"]), "tmin": num(df["tmin"]),
        "hr_min": num(df.get("hrMin", pd.Series(dtype=float))),
        "viento_max": np.minimum(velmedia * 1.5, racha),               # m/s
        "prec": num(df["prec"].replace("Ip", "0")),                   # mm
    })
    return out


def serie_diaria_colector(idema, desde):
    """Agrega el colector horario a diario para los días ≥ desde (incluye hoy)."""
    con = sqlite3.connect(DB_COLECTOR)
    df = pd.read_sql(
        "SELECT fint, ta, tamax, tamin, hr, vv, vmax, prec FROM observacion_horaria "
        "WHERE idema = ? AND fint >= ?", con, params=(idema, str(desde)))
    if df.empty:
        return pd.DataFrame()
    df["fecha"] = pd.to_datetime(df["fint"], utc=True).dt.tz_localize(None).dt.normalize()
    g = df.groupby("fecha")
    return pd.DataFrame({
        "tmax": g["tamax"].max().combine_first(g["ta"].max()),
        "tmin": g["tamin"].min().combine_first(g["ta"].min()),
        "hr_min": g["hr"].min(),
        "viento_max": g["vv"].max(),   # máx. de medias horarias = wind_speed_max
        "prec": g["prec"].sum(min_count=1),
        "n_horas": g["ta"].count(),
    }).reset_index()


def serie_diaria(idema):
    """Serie diaria fusionada: climatologías (autoritativa) + colector (hueco)."""
    a = serie_diaria_aemet(idema)
    ultimo = a["fecha"].max() if len(a) else pd.Timestamp(INICIO_SERIE)
    c = serie_diaria_colector(idema, (ultimo + pd.Timedelta(days=1)).date())
    a["n_horas"] = 24                       # días de la API = completos
    df = pd.concat([a, c], ignore_index=True).sort_values("fecha")
    df = df.drop_duplicates("fecha", keep="first").reset_index(drop=True)
    # reindexar a calendario continuo (huecos → NaN, el FWI los salta)
    cal = pd.date_range(INICIO_SERIE, df["fecha"].max(), freq="D")
    return df.set_index("fecha").reindex(cal).rename_axis("fecha").reset_index()


def firms_nrt_df():
    """Detecciones FIRMS NRT [D-5, D-1] sobre Iberia → DataFrame (caché 6 h).
    Excluye el día en curso (circularidad). Vacío si no hay key o falla."""
    def descargar():
        key = os.environ.get("FIRMS_MAP_KEY")
        if not key:
            return None
        # el API NRT limita a 5 días → ventana [D-5, D-1] en vez de [D-7, D-1]
        # (aprox. documentada; el FRP relevante es el más reciente)
        # VIIRS_NOAA20_NRT: Suomi-NPP ya no publica NRT en 2026 (devolvía 0)
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
               f"VIIRS_NOAA20_NRT/-10,35,5,44/5")
        r = requests.get(url, timeout=30)
        return r.text if r.ok and r.text.startswith("latitude") else None

    txt = _cache_json(f"{DIR_CACHE}/firms_nrt.json", 6 * 3600, lambda: descargar())
    if not txt:
        return pd.DataFrame()
    df = pd.read_csv(StringIO(txt))
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    df["d"] = pd.to_datetime(df["acq_date"])
    return df[df["d"] < hoy]


def firms_nrt_frp(lat, lon):
    """FRP máx y nº detecciones a <50 km en los últimos 7 días (FIRMS NRT)."""
    df = firms_nrt_df()
    if df.empty:
        # fallback 0 = "sin detección": las features frp/n_detec NUNCA fueron
        # NaN en entrenamiento (0 era el valor sin fuego) — un NaN aquí manda
        # al árbol por ramas no aprendidas y corrompe la predicción
        return 0.0, 0.0
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X0, Y0 = tr.transform(lon, lat)
    fx, fy = tr.transform(df["longitude"].values, df["latitude"].values)
    m = np.hypot(fx - X0, fy - Y0) <= 50000
    return (float(df.loc[m, "frp"].max()) if m.any() else 0.0, int(m.sum()))


_MUNIS = None


def _municipio_cercano(lat, lon):
    """Código INE del municipio más cercano (maestro AEMET cacheado)."""
    global _MUNIS
    if _MUNIS is None:
        def descargar():
            key = os.environ["TOKEN_AEMET"]
            r = requests.get("https://opendata.aemet.es/opendata/api/maestro/"
                             "municipios", headers={"api_key": key}, timeout=30)
            d = r.json()
            if isinstance(d, dict) and d.get("datos"):
                d = requests.get(d["datos"], timeout=60).json()
            return d
        d = _cache_json(f"{DIR_CACHE}/municipios.json", 30 * 86400, descargar)
        from scipy.spatial import cKDTree
        m = pd.DataFrame(d)
        m["cod"] = m["url"].str.extract(r"-id(\d{5})")
        m = m.dropna(subset=["cod"])
        m["la"] = pd.to_numeric(m["latitud_dec"], errors="coerce")
        m["lo"] = pd.to_numeric(m["longitud_dec"], errors="coerce")
        m = m.dropna(subset=["la", "lo"]).reset_index(drop=True)
        _MUNIS = (m, cKDTree(np.column_stack([m["la"], m["lo"]])))
    m, arbol = _MUNIS
    _, j = arbol.query([lat, lon])
    return m.iloc[int(j)]["cod"], m.iloc[int(j)]["capital"]


def _forecast_municipio(idema, e):
    """Predicción AEMET del municipio de la estación → filas diarias futuras.
    Precipitación prevista: la predicción da PROBABILIDAD, no cantidad →
    prec=0 si prob<60%, 2 mm si ≥60% (conservador hacia el riesgo; documentado)."""
    cod, capital = _municipio_cercano(e["lat"], e["lon"])

    def descargar():
        # solo llega aquí sin caché → ritmo ~45 req/min (límite AEMET ~50/min,
        # y cada forecast son 2 peticiones) + reintento con espera si 429
        key = os.environ["TOKEN_AEMET"]
        for intento in range(4):
            _t.sleep(1.4)
            r = requests.get(f"https://opendata.aemet.es/opendata/api/prediccion/"
                             f"especifica/municipio/diaria/{cod}",
                             headers={"api_key": key}, timeout=15).json()
            if r.get("estado") == 200:
                return requests.get(r["datos"], timeout=15).json()
            if r.get("estado") == 429:
                _t.sleep(60)
                continue
            return None
        return None

    d = _cache_json(f"{DIR_CACHE}/forecast_{cod}.json", 6 * 3600, descargar)
    if not d:
        raise ValueError(f"sin predicción para el municipio {cod}")
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
            "prec": 2.0 if (pp and max(pp) >= 60) else 0.0,
            "n_horas": 24, "es_forecast": True})
    return pd.DataFrame(filas), capital


def construir_features(idema, horizonte=0):
    """Vector de features para el último día completo (horizonte=0) o para
    D+1/D+2 usando la predicción municipal AEMET (horizonte=1|2): se propaga
    el FWI desde lo observado con la meteo prevista."""
    est = _estaciones()
    if idema not in est.index:
        raise KeyError(f"estación {idema} no disponible (¿fuera de la península?)")
    e = est.loc[idema]

    df = serie_diaria(idema)
    if df.empty or df["tmax"].notna().sum() < 45:
        raise ValueError(f"serie diaria insuficiente para {idema}")
    # último día COMPLETO (≥18 h de observación): evaluar "hoy" con solo las
    # horas de madrugada daba Tmax nocturnas y riesgo falso a la baja (bug
    # detectado 15/07 — mediana nacional real 32,7°C vs 26°C evaluados).
    completos = df.index[(df["tmax"].notna()) & (df["n_horas"].fillna(24) >= 18)]
    i = int(completos[-1])
    if horizonte > 0:
        df = df.iloc[:i + 1].copy()
        fc, _capital = _forecast_municipio(idema, e)
        nuevos = fc[fc["fecha"] > df["fecha"].max()].head(horizonte)
        if len(nuevos) < horizonte:
            raise ValueError("la predicción municipal no cubre ese horizonte")
        df = pd.concat([df, nuevos], ignore_index=True)
        i = len(df) - 1
    fecha = df.loc[i, "fecha"]
    mes, anio = fecha.month, fecha.year

    fwi_out = calcular_fwi_serie(df["tmax"].values, df["hr_min"].values,
                                 df["viento_max"].values * 3.6,
                                 df["prec"].values, df["fecha"].dt.month.values)
    fwi_s = fwi_out["fwi"]

    F = {}
    F["fwi"] = fwi_s[i]
    F["t2m_max"], F["t2m_min"] = df.loc[i, "tmax"], df.loc[i, "tmin"]
    F["rh_min"] = df.loc[i, "hr_min"]
    F["viento_max"] = df.loc[i, "viento_max"]
    F["precip_dia"] = df.loc[i, "prec"]
    es = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
    F["vpd_max"] = es * (1 - F["rh_min"] / 100)

    pre, rh, tmx, vto = (df[c].values for c in ["prec", "hr_min", "tmax", "viento_max"])
    F["precip_7d"] = np.nansum(pre[i-7:i]); F["precip_15d"] = np.nansum(pre[i-15:i])
    F["precip_30d"] = np.nansum(pre[i-30:i])
    F["fwi_med_7d"] = np.nanmean(fwi_s[i-7:i]); F["fwi_max_7d"] = np.nanmax(fwi_s[i-7:i])
    F["fwi_med_15d"] = np.nanmean(fwi_s[i-15:i]); F["fwi_med_30d"] = np.nanmean(fwi_s[i-30:i])
    F["rh_min_med_7d"] = np.nanmean(rh[i-7:i])
    F["t2m_max_med_7d"] = np.nanmean(tmx[i-7:i])
    F["viento_max_med_7d"] = np.nanmean(vto[i-7:i])
    secos = 0
    for k in range(i, -1, -1):
        if np.isnan(pre[k]) or pre[k] >= 1.0 or secos >= 120:
            break
        secos += 1
    F["dias_sin_lluvia"] = secos

    clim = np.load(f"{DIR}/prototipo/clim_fwi/{idema}.npz")[f"m{mes}"]
    F["fwi_pctl_local"] = float((clim <= F["fwi"]).mean() * 100) if len(clim) else np.nan
    F["fwi_anom_sigma"] = (float((F["fwi"] - clim.mean()) / clim.std())
                           if len(clim) and clim.std() > 0 else np.nan)

    for c in ["ndvi", "lai", "swi010", "lst"]:
        F[c] = e[f"{c}_m{mes}"]
    F["ndvi_med_30d"] = e[f"ndvi_m{mes}"]

    for c in ["elevacion", "pendiente", "rugosidad", "dist_carreteras",
              "dist_rios", "popdens", "clc_bosque", "clc_matorral", "clc_agricola",
              "clc_artificial", "clc_abierto", "clc_agric_hetero"]:
        F[c] = e[c]

    # REGLA (aprendida en la prueba del 15/07): una feature que NUNCA fue NaN
    # en entrenamiento no puede ser NaN en inferencia — el árbol la manda por
    # una rama arbitraria. Para las no observables se usa su MODA en train (0).
    F["n_fuegos_1km_90d"] = 0.0             # EGIF no disponible en tiempo real
    F["n_fuegos_10km_90d"] = 0.0            # (moda en train; caveat documentado)
    F["n_fuegos_10km_365d"] = e["n_fuegos_10km_365d"]      # congelada (2020)
    F["n_fuegos_1km_hist"] = e["n_fuegos_1km_hist"]
    F["n_fuegos_10km_mismomes_hist"] = e[f"n_mismomes_m{mes}"]

    F["frp_max_50km_7d"], F["n_detec_50km_7d"] = firms_nrt_frp(e["lat"], e["lon"])
    F["rayos_dia"], F["rayos_7d"] = 0.0, 0.0   # WGLC llega a 2023; moda = 0

    import holidays
    festivos = holidays.Spain(years=[anio])
    F["es_festivo"] = float(fecha.weekday() >= 5 or fecha.date() in festivos)
    F["mes"], F["dia_anio"] = mes, fecha.dayofyear
    F["ccaa"] = int(e["ccaa"]) if np.isfinite(e["ccaa"]) else -1

    return F, fecha, df, fwi_out


def predecir(idema, horizonte=0):
    """Predicción completa: prob, nivel, SHAP top, condiciones y series.
    horizonte: 0 = último día completo; 1|2 = mañana/pasado con forecast."""
    F, fecha, df, fwi_out = construir_features(idema, horizonte)
    X = pd.DataFrame([{k: F[k] for k in FULL}])
    prob = float(_modelo().predict_proba(X)[0, 1])
    # cortes fijados con cuantiles de val 2019 (ver app): 0.25/0.55/0.80
    nivel = ("EXTREMO" if prob >= 0.80 else "ALTO" if prob >= 0.55
             else "MODERADO" if prob >= 0.25 else "BAJO")
    res = {"idema": idema, "fecha": str(fecha.date()), "prob": prob,
           "nivel": nivel, "features": F, "horizonte": horizonte}
    try:
        import shap
        sv = shap.TreeExplainer(_modelo()).shap_values(X)[0]
        top = pd.Series(sv, index=FULL).sort_values(key=abs, ascending=False).head(8)
        res["shap_top"] = [(k, float(v), float(X[k].iloc[0])) for k, v in top.items()]
    except Exception:
        res["shap_top"] = []
    return res, df, fwi_out


# ----------------------------------------------------------------------------
# Evaluación MASIVA: ranking nacional de riesgo (todas las estaciones)
# Usa el endpoint `todasestaciones` (serie diaria de las ~819 estaciones en
# ~6 peticiones de 14 días) + colector horario en bloque → FWI y features por
# estación → predicción batch. Caché diaria del ranking.
# ----------------------------------------------------------------------------

def _todas_chunk(f0, f1, max_edad_s):
    def descargar():
        key = os.environ["TOKEN_AEMET"]
        url = (f"https://opendata.aemet.es/opendata/api/valores/climatologicos/"
               f"diarios/datos/fechaini/{f0}T00:00:00UTC/"
               f"fechafin/{f1}T23:59:59UTC/todasestaciones")
        r = requests.get(url, headers={"api_key": key}, timeout=30).json()
        if r.get("estado") != 200:
            return None
        return requests.get(r["datos"], timeout=120).json()
    return _cache_json(f"{DIR_CACHE}/todas_{f0}.json", max_edad_s, descargar) or []


def serie_diaria_todas():
    """Serie diaria de TODAS las estaciones desde el 1-may (API + colector)."""
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    filas = []
    f0 = pd.Timestamp(INICIO_SERIE)
    while f0 <= hoy:
        f1 = min(f0 + pd.Timedelta(days=13), hoy)
        # chunks antiguos: caché permanente; el último: 6 h
        edad = 6 * 3600 if f1 >= hoy - pd.Timedelta(days=14) else 10**9
        filas += _todas_chunk(f0.strftime("%Y-%m-%d"), f1.strftime("%Y-%m-%d"), edad)
        f0 = f1 + pd.Timedelta(days=1)
    df = pd.DataFrame(filas)

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")

    velmedia, racha = num(df["velmedia"]), num(df["racha"])
    api = pd.DataFrame({
        "idema": df["indicativo"], "fecha": pd.to_datetime(df["fecha"]),
        "tmax": num(df["tmax"]), "tmin": num(df["tmin"]),
        "hr_min": num(df.get("hrMin", pd.Series(dtype=float))),
        "viento_max": np.minimum(velmedia * 1.5, racha),
        "prec": num(df["prec"].replace("Ip", "0")), "n_horas": 24})

    ult_api = api["fecha"].max()
    con = sqlite3.connect(DB_COLECTOR)
    c = pd.read_sql("SELECT idema, fint, ta, tamax, tamin, hr, vv, prec "
                    "FROM observacion_horaria WHERE fint >= ?", con,
                    params=(str((ult_api + pd.Timedelta(days=1)).date()),))
    if len(c):
        c["fecha"] = pd.to_datetime(c["fint"], utc=True).dt.tz_localize(None).dt.normalize()
        g = c.groupby(["idema", "fecha"])
        col = pd.DataFrame({
            "tmax": g["tamax"].max().combine_first(g["ta"].max()),
            "tmin": g["tamin"].min().combine_first(g["ta"].min()),
            "hr_min": g["hr"].min(), "viento_max": g["vv"].max(),
            "prec": g["prec"].sum(min_count=1),
            "n_horas": g["ta"].count()}).reset_index()
        api = pd.concat([api, col], ignore_index=True)
    return api.sort_values(["idema", "fecha"]).drop_duplicates(
        ["idema", "fecha"], keep="first")


def evaluar_todas(forzar=False, fecha_objetivo=None):
    """Ranking nacional: 1 fila/estación con prob, nivel y features clave.

    fecha_objetivo (AAAA-MM-DD o Timestamp, opcional): en vez del último día
    completo observado, evalúa esa fecha. Si es futura respecto a lo observado
    de una estación, se añaden filas de la predicción municipal AEMET y el FWI
    se propaga (mismo mecanismo que predecir(horizonte>0)). Estaciones sin
    predicción disponible se omiten. Primera ejecución del día: lenta (~1
    petición/estación, ritmo limitado por la API)."""
    hoy = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    if fecha_objetivo is not None:
        fecha_objetivo = pd.Timestamp(fecha_objetivo).normalize()
        ruta = f"{DIR_CACHE}/ranking_{hoy}_obj{fecha_objetivo.date()}.parquet"
    else:
        ruta = f"{DIR_CACHE}/ranking_{hoy}.parquet"
    if os.path.exists(ruta) and not forzar:
        return pd.read_parquet(ruta)

    est = _estaciones()
    todas = serie_diaria_todas()
    frp_txt_ok = True
    import holidays as _hol
    filas = []
    n_sin_forecast = 0
    for idema, s in todas.groupby("idema"):
        if idema not in est.index or s["tmax"].notna().sum() < 45:
            continue
        e = est.loc[idema]
        cal = pd.date_range(INICIO_SERIE, s["fecha"].max(), freq="D")
        s = s.set_index("fecha").reindex(cal).rename_axis("fecha").reset_index()
        comp = s.index[(s["tmax"].notna()) & (s["n_horas"].fillna(24) >= 18)]
        if not len(comp):
            continue
        i = int(comp[-1])
        fecha = s.loc[i, "fecha"]
        dias_forecast = 0
        if fecha_objetivo is not None and fecha < fecha_objetivo:
            try:
                fc, _capital = _forecast_municipio(idema, e)
            except Exception:
                n_sin_forecast += 1
                continue
            nuevos = fc[(fc["fecha"] > fecha) & (fc["fecha"] <= fecha_objetivo)]
            if nuevos.empty or nuevos["fecha"].max() != fecha_objetivo:
                n_sin_forecast += 1
                continue
            dias_forecast = int((fecha_objetivo - fecha).days)
            s = pd.concat([s.iloc[:i + 1], nuevos], ignore_index=True)
            cal = pd.date_range(INICIO_SERIE, fecha_objetivo, freq="D")
            s = s.set_index("fecha").reindex(cal).rename_axis("fecha").reset_index()
            i = len(s) - 1
            fecha = s.loc[i, "fecha"]
            if (len(filas) + n_sin_forecast) % 100 == 0:
                print(f"  forecast: {len(filas)} estaciones procesadas "
                      f"({n_sin_forecast} sin predicción)", flush=True)
        elif fecha_objetivo is not None and fecha > fecha_objetivo:
            # la estación ya tiene observado un día posterior al objetivo
            m = s.index[s["fecha"] == fecha_objetivo]
            if not len(m) or m[0] not in set(comp):
                continue
            i = int(m[0])
            fecha = fecha_objetivo
        fwi_s = calcular_fwi_serie(s["tmax"].values, s["hr_min"].values,
                                   s["viento_max"].values * 3.6, s["prec"].values,
                                   s["fecha"].dt.month.values)["fwi"]
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
        clim = np.load(f"{DIR}/prototipo/clim_fwi/{idema}.npz")[f"m{mes}"]
        F["fwi_pctl_local"] = float((clim <= F["fwi"]).mean() * 100) if len(clim) else np.nan
        F["fwi_anom_sigma"] = (float((F["fwi"] - clim.mean()) / clim.std())
                               if len(clim) and clim.std() > 0 else np.nan)
        for c in ["ndvi", "lai", "swi010", "lst"]:
            F[c] = e[f"{c}_m{mes}"]
        F["ndvi_med_30d"] = e[f"ndvi_m{mes}"]
        for c in ["elevacion", "pendiente", "rugosidad", "dist_carreteras",
                  "dist_rios", "popdens", "clc_bosque", "clc_matorral",
                  "clc_agricola", "clc_artificial", "clc_abierto", "clc_agric_hetero"]:
            F[c] = e[c]
        F["n_fuegos_10km_mismomes_hist"] = e[f"n_mismomes_m{mes}"]
        if frp_txt_ok:
            try:
                F["frp_max_50km_7d"], F["n_detec_50km_7d"] = firms_nrt_frp(e["lat"], e["lon"])
            except Exception:
                frp_txt_ok = False
        if not frp_txt_ok:
            F["frp_max_50km_7d"], F["n_detec_50km_7d"] = 0.0, 0.0
        F["rayos_dia"], F["rayos_7d"] = 0.0, 0.0
        fest = _hol.Spain(years=[fecha.year])
        F["es_festivo"] = float(fecha.weekday() >= 5 or fecha.date() in fest)
        F["mes"], F["dia_anio"] = mes, fecha.dayofyear
        F["ccaa"] = int(e["ccaa"]) if np.isfinite(e["ccaa"]) else -1
        F.update(idema=idema, nombre=e["nombre"], lat=e["lat"], lon=e["lon"],
                 fecha=str(fecha.date()), dias_forecast=dias_forecast)
        filas.append(F)

    if fecha_objetivo is not None:
        print(f"Objetivo {fecha_objetivo.date()}: {len(filas)} estaciones "
              f"({n_sin_forecast} omitidas sin predicción municipal)", flush=True)
    df = pd.DataFrame(filas)
    df["prob"] = _modelo().predict_proba(df[FULL])[:, 1]
    df["nivel"] = pd.cut(df["prob"], [-1, 0.25, 0.55, 0.80, 2],
                         labels=["BAJO", "MODERADO", "ALTO", "EXTREMO"])
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df.to_parquet(ruta, index=False)
    return df


if __name__ == "__main__":
    import sys
    r, df, _ = predecir(sys.argv[1] if len(sys.argv) > 1 else "3195")
    est = _estaciones().loc[r["idema"]]
    print(f"\n=== {est['nombre']} ({r['idema']}) — {r['fecha']} ===")
    print(f"RIESGO: {r['nivel']}  (prob={r['prob']:.3f}, prevalencia de diseño 25%)")
    F = r["features"]
    print(f"FWI={F['fwi']:.1f} (pctl local {F['fwi_pctl_local']:.0f}, "
          f"anomalía {F['fwi_anom_sigma']:+.1f}σ) · Tmax={F['t2m_max']:.1f}°C · "
          f"HRmin={F['rh_min']:.0f}% · días sin lluvia={F['dias_sin_lluvia']}")
    print("\nTop contribuciones SHAP:")
    for k, v, val in r["shap_top"]:
        print(f"  {'+' if v > 0 else '−'} {k:30s} {v:+.3f} (valor={val:.2f})")
