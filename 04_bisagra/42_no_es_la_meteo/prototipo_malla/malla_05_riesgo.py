#!/usr/bin/env python3
"""
Pipeline de malla, módulo 5: mapa de riesgo sin IDW, retrospectivo.

No toca producción. Escribe eda/malla_mapa_riesgo_<fecha>.png y
dataset/malla_05_riesgo_<fecha>.json.

Qué cierra este módulo
----------------------
Producción (`mapa_riesgo_hoy.py`) calcula las features meteo por estación
AEMET y las interpola a la malla de 1 km con IDW k=8. Ese paso nunca se
validó. Aquí se sustituye por lo que preparan los módulos 1-4: la meteo se
calcula en los 5.605 nodos de la malla nativa de ERA5-Land y cada celda toma
el valor de su nodo. No hay interpolación, no hay huecos y no depende de que
una estación concreta reporte.

Y, sobre todo, `fwi_pctl_local` pasa a tener numerador y denominador de la
misma fuente: FWI de ERA5-Land sobre climatología de ERA5-Land
(`clim_fwi_nodos.npz`, módulo 4). Ese era el invariante roto que originó todo
el trabajo: en producción el numerador es AEMET y el denominador el cubo, y
la división satura al 3,3 %.

El módulo pinta los dos mapas (referencia del cubo y malla ERA5-Land) con
todo lo no-meteo compartido, de modo que la única diferencia entre ellos es
la fuente de la meteo. Sirve de demo visual y de prueba de aceptación.

Por qué retrospectivo y por qué esta fecha
------------------------------------------
ERA5-Land es reanálisis: llega con ~6 días de retraso y no da D+1. El mapa de
hoy/mañana necesita la previsión del módulo 2b, que aún no existe. El
retrospectivo no necesita 2b y permite comparar contra el cubo, que es la
referencia.

Por defecto 2024-09-17: 97 detecciones FIRMS dentro de España y FRP máximo de
710 (el mayor de jun-sep 2024), con el cubo cubriendo 2024 y ERA5-Land ya en
disco para jun-sep. Es verdad-terreno independiente y año fuera del rango de
entrenamiento (2015-2018).

Dos limitaciones, ambas medidas antes de aceptarlas
---------------------------------------------------
1. Ventana corta. Solo hay ERA5-Land desde el 1-jun-2024, así que
   `dias_sin_lluvia` (ventana de 120 d) se trunca a los días disponibles.
   Medido sobre el cubo para esta fecha: afecta al 0,30 % de las celdas, con
   10 días de diferencia media donde afecta. Despreciable.

2. Spin-up del FWI. Arrancar la recursión el 1-jun deja el DC bajo. Medido
   con la misma meteo del cubo, arranque 1-ene contra 1-jun al 17-09-2024:
   DC 1.023 contra 782, pero el FWI solo cae −0,30 (correlación 0,9999). A
   BUI alto el FWI ya no depende del DC. Aceptable.

`lst` sale del cubo en los dos mapas: es satélite (MODIS), no meteo, y
ERA5-Land no lo da. Igual que en producción. Lo mismo vegetación, estáticas,
autorregresivas EGIF, FIRMS y rayos: idénticas en ambos mapas por diseño.

Nodos costeros
--------------
226 nodos (4,0 %) tienen su vecino ERA5-Land más próximo en mar y quedan a
NaN, como avisaba el módulo 4. Aquí se resuelven con el respaldo que ese
módulo dejaba pendiente: cada nodo sin dato hereda el nodo terrestre más
cercano, y lo hereda entero (meteo y climatología del mismo nodo) para no
volver a mezclar numerador y denominador de sitios distintos.

Uso:
    python malla_05_riesgo.py                 # 2024-09-17
    python malla_05_riesgo.py 2024-08-11
    python malla_05_riesgo.py --permitir-descarga   # si falta algún mes
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.spatial import cKDTree

from entrenar_modelo import (FEATS_CAL, FEATS_ESTAT, FEATS_HIST, FEATS_METEO,
                             FEATS_VEG)
from fwi_canadiense import calcular_fwi_serie
from malla_02_descarga import CRUDO, DATA, DIR, a_diario, abre, pide_mes

FULL = FEATS_METEO + FEATS_VEG + FEATS_ESTAT + FEATS_HIST + FEATS_CAL
ANIOS_CLIM = (2008, 2014)          # climatología del cubo (referencia)
DIAS_ATRAS = 120                   # ventana máxima que piden las features
MODELO = f"{DIR}/modelos/xgb_v1.ubj"

# las que este módulo recalcula desde ERA5-Land; el resto se comparte
METEO_MALLA = [f for f in FEATS_METEO if f != "lst"]


def suma_caja(campo, radio_celdas):
    k = 2 * radio_celdas + 1
    return uniform_filter(campo.astype(float), size=k, mode="constant") * k * k


def ventanas(fwi, tmx, tmn, hrm, vto, pre):
    """Las 20 features meteo del día final, desde series (dias, puntos).

    Réplica exacta de las ventanas de `mapa_riesgo_dia.py`: el día D es el
    último de cada serie y las medias móviles excluyen D (`[-8:-1]` = los 7
    días previos), igual que en el extractor de entrenamiento.
    """
    F = {}
    F["fwi"] = fwi[-1]
    F["t2m_max"] = tmx[-1]
    F["t2m_min"] = tmn[-1]
    F["rh_min"] = hrm[-1]
    F["viento_max"] = vto[-1]
    F["precip_dia"] = pre[-1]
    es = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
    F["vpd_max"] = es * (1 - F["rh_min"] / 100)

    F["precip_7d"] = np.nansum(pre[-8:-1], axis=0)
    F["precip_15d"] = np.nansum(pre[-16:-1], axis=0)
    F["precip_30d"] = np.nansum(pre[-31:-1], axis=0)
    F["fwi_med_7d"] = np.nanmean(fwi[-8:-1], axis=0)
    F["fwi_max_7d"] = np.nanmax(fwi[-8:-1], axis=0)
    F["fwi_med_15d"] = np.nanmean(fwi[-16:-1], axis=0)
    F["fwi_med_30d"] = np.nanmean(fwi[-31:-1], axis=0)
    F["rh_min_med_7d"] = np.nanmean(hrm[-8:-1], axis=0)
    F["t2m_max_med_7d"] = np.nanmean(tmx[-8:-1], axis=0)
    F["viento_max_med_7d"] = np.nanmean(vto[-8:-1], axis=0)

    lluvia = pre >= 1.0
    secos = np.zeros(pre.shape[1:], dtype=float)
    activo = np.ones(pre.shape[1:], dtype=bool)
    for k in range(pre.shape[0] - 1, -1, -1):
        activo &= ~lluvia[k] & ~np.isnan(pre[k])
        secos += activo
    F["dias_sin_lluvia"] = np.minimum(secos, DIAS_ATRAS)
    return F


# ---------------------------------------------------------------- meteo malla
def meteo_malla(d, mes, permitir_descarga, mapear=False):
    """Las 20 features meteo en los nodos, desde ERA5-Land + clim del módulo 4."""
    n = np.load(f"{DATA}/nodos.npz")
    la, lo, idx_nodo = n["nodo_lat"], n["nodo_lon"], n["idx_nodo"]
    ini = pd.Timestamp(str(d)) - pd.Timedelta(days=DIAS_ATRAS)

    meses = pd.period_range(ini, pd.Timestamp(str(d)), freq="M")
    faltan = [(p.year, p.month) for p in meses
              if not os.path.exists(f"{CRUDO}/era5land_{p.year}{p.month:02d}.nc")]
    if faltan and not permitir_descarga:
        # No es fatal: la única feature que mira tan atrás es `dias_sin_lluvia`
        # y el truncado está medido (0,30 % de celdas). Se avisa y se sigue.
        print("  aviso: sin ERA5-Land para "
              + ", ".join(f"{x}-{y:02d}" for x, y in faltan)
              + " · la ventana se trunca (--permitir-descarga para bajarlos)",
              flush=True)
    rutas, extraidos = [], []
    for p in meses:
        r = f"{CRUDO}/era5land_{p.year}{p.month:02d}.nc"
        if os.path.exists(r) or permitir_descarga:
            rutas.append(pide_mes(p.year, p.month, list(range(1, 32))))
            extraidos.append(rutas[-1] + "_x")

    print(f"  ERA5-Land: {len(rutas)} meses ({str(ini)[:10]} → {str(d)})",
          flush=True)
    partes = [a_diario(abre(r)) for r in rutas]
    dim = [x for x in partes[0].dims if x not in ("latitude", "longitude")][0]
    ds = xr.concat(partes, dim=dim).rename({dim: "fecha"}).sortby("fecha").load()

    pts = ds.sel(latitude=xr.DataArray(la, dims="n"),
                 longitude=xr.DataArray(lo, dims="n"), method="nearest")
    fechas = pd.to_datetime(ds["fecha"].values).normalize()
    hasta = fechas <= pd.Timestamp(str(d))
    fechas = fechas[hasta]
    if fechas[-1] != pd.Timestamp(str(d)):
        raise SystemExit(f"ERA5-Land no llega al {d} (último {fechas[-1].date()})")
    S = {v: pts[v].values[hasta] for v in
         ["tmax", "tmin", "hr_min", "viento_max", "prec"]}
    ds.close()
    for e in extraidos:
        shutil.rmtree(e, ignore_errors=True)          # el disco va justo

    print(f"  FWI en {len(la):,} nodos sobre {len(fechas)} días...", flush=True)
    m_ser = fechas.month.values
    fwi = np.full(S["tmax"].shape, np.nan, dtype=np.float32)
    for j in range(len(la)):
        fwi[:, j] = calcular_fwi_serie(S["tmax"][:, j], S["hr_min"][:, j],
                                       S["viento_max"][:, j] * 3.6,
                                       S["prec"][:, j], m_ser)["fwi"]

    F = ventanas(fwi, S["tmax"], S["tmin"], S["hr_min"], S["viento_max"],
                 S["prec"])
    dias = len(fechas)

    # --- variante híbrida (--mapear-fwi) -----------------------------------
    # El modelo se entrenó con el FWI del cubo y el de ERA5-Land corre más
    # alto: alimentarlo crudo es desajuste train/serve. El mapeo de cuantiles
    # del módulo 3b lo lleva a la escala del cubo. Se aplica solo a las
    # features de nivel; el percentil y la anomalía se calculan abajo con el
    # FWI sin mapear contra la climatología sin mapear, que es donde el sesgo
    # se cancela por construcción y donde mapear rompería el invariante.
    # El mapeo es monótono pero no lineal: hay que mapear la serie y luego
    # promediar, no al revés.
    if mapear:
        mp = np.load(f"{DATA}/mapeo_fwi.npz")
        fwi_m = np.where(np.isfinite(fwi),
                         np.interp(fwi, mp["xs"], mp["ys"]), np.nan)
        Fm = ventanas(fwi_m, S["tmax"], S["tmin"], S["hr_min"],
                      S["viento_max"], S["prec"])
        for v in ("fwi", "fwi_med_7d", "fwi_max_7d", "fwi_med_15d",
                  "fwi_med_30d"):
            F[v] = Fm[v]
        F["_fwi_sin_mapear"] = fwi[-1]

    # --- percentil y anomalía contra la climatología ERA5-Land (módulo 4) ---
    clim = np.load(f"{DATA}/clim_fwi_nodos.npz")
    C, nC = clim[f"m{mes}"], clim[f"n{mes}"].astype(float)
    with np.errstate(invalid="ignore"):
        crudo = F.pop("_fwi_sin_mapear", F["fwi"])
        cnt = np.nansum(C <= crudo[None, :], axis=0)
        F["fwi_pctl_local"] = np.where(nC > 0, cnt / np.maximum(nC, 1) * 100,
                                       np.nan)
        cmed, cstd = np.nanmean(C, axis=0), np.nanstd(C, axis=0)
        F["fwi_anom_sigma"] = np.where((nC > 0) & (cstd > 0),
                                       (crudo - cmed) / cstd, np.nan)

    # --- respaldo terrestre para los nodos sin dato (costeros) --------------
    malo = ~np.isfinite(crudo) | (nC == 0)
    n_malo = int(malo.sum())
    if n_malo:
        bueno = np.where(~malo)[0]
        arbol = cKDTree(np.column_stack([la[bueno], lo[bueno]]))
        _, k = arbol.query(np.column_stack([la[malo], lo[malo]]))
        origen = bueno[k]
        for v in F:                       # el nodo se hereda entero
            F[v][malo] = F[v][origen]
        print(f"  nodos sin dato con respaldo terrestre: {n_malo} "
              f"({n_malo/len(la)*100:.1f} %)", flush=True)
    return F, idx_nodo, n_malo, dias


def a_celdas(F, idx_nodo):
    """Cada celda toma el valor de su nodo. Sin IDW, sin pesos, sin vecinos."""
    seg = np.clip(idx_nodo, 0, None)
    fuera = idx_nodo < 0
    out = {}
    for v, x in F.items():
        c = np.asarray(x, dtype=float)[seg]
        c[fuera] = np.nan
        out[v] = c
    return out


# ---------------------------------------------------------------- meteo cubo
def meteo_cubo(ds, t, mes, ny, nx):
    """La referencia: las mismas features desde el cubo IberFire."""
    def slab(var, t0, t1):
        return ds[var].isel(time=slice(t0, t1)).values

    F = ventanas(slab("FWI", t - DIAS_ATRAS, t + 1),
                 slab("t2m_max", t - DIAS_ATRAS, t + 1),
                 slab("t2m_min", t - DIAS_ATRAS, t + 1),
                 slab("RH_min", t - DIAS_ATRAS, t + 1),
                 slab("wind_speed_max", t - DIAS_ATRAS, t + 1),
                 slab("total_precipitation_mean", t - DIAS_ATRAS, t + 1))

    # climatología del cubo, por trozos para no cargar 210 mapas de golpe
    tiempos = ds["time"].values.astype("datetime64[D]")
    anios_t = tiempos.astype("datetime64[Y]").astype(int) + 1970
    meses_t = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    idx = np.where((anios_t >= ANIOS_CLIM[0]) & (anios_t <= ANIOS_CLIM[1])
                   & (meses_t == mes))[0]
    cnt = np.zeros((ny, nx)); n = np.zeros((ny, nx))
    s1 = np.zeros((ny, nx)); s2 = np.zeros((ny, nx))
    for i in range(0, len(idx), 30):
        blo = ds["FWI"].isel(time=idx[i:i + 30]).values
        fin = np.isfinite(blo)
        cnt += np.nansum(blo <= F["fwi"][None, :], axis=0)
        n += fin.sum(0)
        s1 += np.where(fin, blo, 0).sum(0)
        s2 += np.where(fin, blo, 0).__pow__(2).sum(0)
        del blo
    with np.errstate(divide="ignore", invalid="ignore"):
        F["fwi_pctl_local"] = np.where(n > 0, cnt / np.maximum(n, 1) * 100, np.nan)
        cmed = s1 / np.maximum(n, 1)
        cstd = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - cmed ** 2, 0))
        F["fwi_anom_sigma"] = np.where((n > 0) & (cstd > 0),
                                       (F["fwi"] - cmed) / cstd, np.nan)
    return F


# ------------------------------------------------------- resto del mapa
def compartidas(ds, d, t, anio, mes, dia_anio, ny, nx, xs, ys):
    """Todo lo no meteo: vegetación, estáticas, EGIF, FIRMS y rayos.

    Es la variable de control del experimento: idéntica en los dos
    mapas, así la única diferencia entre ellos es la fuente meteo.
    Devuelve (B, firms, fd, tr) porque el FIRMS del propio día y el
    transformador se reutilizan luego para la métrica y la figura."""
    B = {}
    B["lst"] = ds["LST"].isel(time=t).values
    B["ndvi"] = ds["NDVI"].isel(time=t).values
    B["ndvi_med_30d"] = np.nanmean(
        ds["NDVI"].isel(time=slice(t - 30, t)).values, axis=0)
    B["lai"] = ds["LAI"].isel(time=t).values
    B["swi010"] = ds["SWI_010"].isel(time=t).values
    B["es_festivo"] = ds["is_holiday"].isel(time=t).values.astype(float)
    for k, v in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                 "rugosidad": "roughness_mean",
                 "dist_carreteras": "dist_to_roads_mean",
                 "dist_rios": "dist_to_waterways_mean"}.items():
        B[k] = ds[v].values
    B["popdens"] = ds[f"popdens_{min(anio, 2020)}"].values
    for k, suf in {"clc_bosque": "forest_proportion",
                   "clc_matorral": "scrub_proportion",
                   "clc_agricola": "agricultural_proportion",
                   "clc_artificial": "artificial_proportion",
                   "clc_abierto": "open_space_proportion",
                   "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
        B[k] = ds[f"CLC_2018_{suf}"].values
    B["ccaa"] = np.nan_to_num(ds["AutonomousCommunities"].values, nan=-1).astype(int)
    B["mes"] = np.full((ny, nx), mes)
    B["dia_anio"] = np.full((ny, nx), dia_anio)

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    egif = pd.read_csv(f"{DIR}/egif_civio.csv",
                       usecols=["fecha", "lat", "lng"]).dropna()
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna()
    egif = egif[egif["fecha"].dt.year >= 2008]
    ex, ey = tr.transform(egif["lng"].values, egif["lat"].values)
    eix = np.rint((ex - xs[0]) / (xs[1] - xs[0])).astype(int)
    eiy = np.rint((ey - ys[0]) / (ys[1] - ys[0])).astype(int)
    ed = egif["fecha"].values.astype("datetime64[D]")
    ok = (eix >= 0) & (eix < nx) & (eiy >= 0) & (eiy < ny) & (ed < d)

    def malla_fuegos(m):
        g = np.zeros((ny, nx))
        np.add.at(g, (eiy[ok & m], eix[ok & m]), 1)
        return g

    g90 = malla_fuegos(ed >= d - 90)
    g365 = malla_fuegos(ed >= d - 365)
    ghist = malla_fuegos(np.ones(len(ed), bool))
    gmm = malla_fuegos((egif["fecha"].dt.month.values == mes)
                       & (egif["fecha"].dt.year.values < anio))
    B["n_fuegos_1km_90d"] = suma_caja(g90, 1)
    B["n_fuegos_10km_90d"] = suma_caja(g90, 10)
    B["n_fuegos_10km_365d"] = suma_caja(g365, 10)
    B["n_fuegos_1km_hist"] = suma_caja(ghist, 1)
    B["n_fuegos_10km_mismomes_hist"] = suma_caja(gmm, 10)

    firms = pd.read_parquet(f"{DIR}/firms_iberia_2015_2024.parquet",
                            columns=["latitude", "longitude", "acq_date", "frp"])
    fd = firms["acq_date"].values.astype("datetime64[D]")
    m7 = (fd >= d - 7) & (fd < d)
    fx, fy = tr.transform(firms.loc[m7, "longitude"].values,
                          firms.loc[m7, "latitude"].values)
    fix = np.rint((fx - xs[0]) / (xs[1] - xs[0])).astype(int)
    fiy = np.rint((fy - ys[0]) / (ys[1] - ys[0])).astype(int)
    okf = (fix >= 0) & (fix < nx) & (fiy >= 0) & (fiy < ny)
    gfrp = np.zeros((ny, nx)); gn = np.zeros((ny, nx))
    np.maximum.at(gfrp, (fiy[okf], fix[okf]), firms.loc[m7, "frp"].values[okf])
    np.add.at(gn, (fiy[okf], fix[okf]), 1)
    B["frp_max_50km_7d"] = maximum_filter(gfrp, size=101, mode="constant")
    B["n_detec_50km_7d"] = suma_caja(gn, 50)

    wglc = xr.open_dataset(f"{DIR}/rayos_data/wglc/wglc_timeseries_30m_daily.nc")
    wd = wglc["time"].values.astype("datetime64[D]")
    tw = int((d - wd[0]).astype(int))
    lon2d, lat2d = np.meshgrid(xs, ys)
    tri = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lons, lats = tri.transform(lon2d.ravel(), lat2d.ravel())
    li = np.clip(np.searchsorted(wglc["lat"].values, lats), 0, 359)
    lj = np.clip(np.searchsorted(wglc["lon"].values, lons), 0, 719)
    if 0 <= tw < wglc.sizes["time"]:
        B["rayos_dia"] = wglc["density"].isel(time=tw).values[li, lj].reshape(ny, nx)
        r7 = wglc["density"].isel(time=slice(max(0, tw - 7), tw)).sum("time").values
        B["rayos_7d"] = r7[li, lj].reshape(ny, nx)
    else:
        B["rayos_dia"] = np.full((ny, nx), np.nan)
        B["rayos_7d"] = np.full((ny, nx), np.nan)
    wglc.close()
    return B, firms, fd, tr


def main(a):
    d = np.datetime64(a.fecha, "D")
    anio, mes = int(str(d)[:4]), int(str(d)[5:7])
    dia_anio = int((d - np.datetime64(f"{anio}-01-01", "D")).astype(int)) + 1
    print(f"Mapa de malla para {a.fecha} · modelo {os.path.basename(MODELO)}\n",
          flush=True)

    ds = xr.open_dataset(f"{DIR}/iberfire/IberFire.nc", decode_timedelta=False)
    tiempos = ds["time"].values.astype("datetime64[D]")
    t = int((d - tiempos[0]).astype(int))
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    es_esp = ds["is_spain"].values.astype(bool)
    xs, ys = ds["x"].values, ds["y"].values

    # ---------------- meteo: las dos versiones --------------------------
    print("[1/5] meteo desde ERA5-Land en los nodos", flush=True)
    Fn, idx_nodo, n_malo, dias_ventana = meteo_malla(
        d, mes, a.permitir_descarga, a.mapear_fwi)
    M = a_celdas(Fn, idx_nodo)
    print("\n[2/5] meteo desde el cubo (referencia)", flush=True)
    R = meteo_cubo(ds, t, mes, ny, nx)

    # ---------------- lo compartido -------------------------------------
    print("\n[3/5] vegetación, estáticas, autorregresivas, FIRMS y rayos",
          flush=True)
    B, firms, fd, tr = compartidas(ds, d, t, anio, mes, dia_anio,
                                   ny, nx, xs, ys)

    # ---------------- predicción ----------------------------------------
    print("\n[4/5] predicción con los dos juegos de meteo", flush=True)
    modelo = xgb.XGBClassifier()
    modelo.load_model(MODELO)
    sel = es_esp.ravel()

    def predice(MET):
        X = pd.DataFrame({k: (MET[k] if k in MET else B[k]).ravel()[sel]
                          for k in FULL})
        p = np.full(ny * nx, np.nan)
        p[sel] = modelo.predict_proba(X)[:, 1]
        return p.reshape(ny, nx)

    prob_r = predice(R)
    prob_m = predice(M)

    # ---------------- métricas y figura ---------------------------------
    print("\n[5/5] métricas y figura", flush=True)
    res = {"fecha": a.fecha, "modelo": os.path.basename(MODELO),
           "nodos_con_respaldo": n_malo, "ventana_secos_dias": dias_ventana}

    def satura(p):
        v = p[es_esp]; v = v[np.isfinite(v)]
        return dict(mediana=float(np.median(v)),
                    ge95=float((v >= 95).mean() * 100),
                    ge999=float((v >= 99.9).mean() * 100))

    print("\n" + "=" * 72)
    print(f"fwi_pctl_local — la métrica que decide ({a.fecha})")
    print("=" * 72)
    print(f"{'denominador':<40}{'mediana':>10}{'% >=95':>10}{'% >=99,9':>11}")
    for nom, p in [("CUBO / clim cubo (referencia)", R["fwi_pctl_local"]),
                   ("ERA5-Land / clim ERA5-Land (malla)", M["fwi_pctl_local"])]:
        s = satura(p)
        print(f"{nom:<40}{s['mediana']:>10.1f}{s['ge95']:>10.1f}"
              f"{s['ge999']:>11.2f}")
        res[nom] = s
    print("\n  (producción con AEMET marcaba 3,30 % de saturación, pero sobre")
    print("   120 estaciones y todo ago-sep 2024: NO es comparable celda a")
    print("   celda con un solo día. Lo comparable es cubo contra malla.)")

    print("\n" + "=" * 72)
    print("FWI y probabilidad")
    print("=" * 72)
    for nom, f_, p_ in [("cubo", R["fwi"], prob_r), ("malla", M["fwi"], prob_m)]:
        fv, pv = f_[es_esp], p_[es_esp]
        print(f"  {nom:<6} FWI medio {np.nanmean(fv):6.2f} · "
              f"p95 {np.nanpercentile(fv, 95):6.2f} | "
              f"prob media {np.nanmean(pv):.4f} · "
              f"p99 {np.nanpercentile(pv, 99):.4f}")
        res[f"fwi_{nom}"] = float(np.nanmean(fv))
        res[f"prob_media_{nom}"] = float(np.nanmean(pv))
    kk = np.isfinite(prob_r) & np.isfinite(prob_m) & es_esp
    res["corr_prob"] = float(np.corrcoef(prob_r[kk], prob_m[kk])[0, 1])
    print(f"\n  correlación entre los dos mapas de probabilidad: "
          f"{res['corr_prob']:.4f}")

    # discriminación contra FIRMS del propio día
    mD = fd == d
    fx2, fy2 = tr.transform(firms.loc[mD, "longitude"].values,
                            firms.loc[mD, "latitude"].values)
    fix2 = np.rint((fx2 - xs[0]) / (xs[1] - xs[0])).astype(int)
    fiy2 = np.rint((fy2 - ys[0]) / (ys[1] - ys[0])).astype(int)
    okd = (fix2 >= 0) & (fix2 < nx) & (fiy2 >= 0) & (fiy2 < ny)
    det = np.zeros((ny, nx), bool)
    det[fiy2[okd], fix2[okd]] = True
    det &= es_esp
    print(f"\n  detecciones FIRMS del día dentro de España: {int(det.sum())} celdas")
    print(f"{'mapa':<10}{'riesgo mediano en fuego':>26}"
          f"{'en España':>12}{'percentil':>11}")
    for nom, p_ in [("cubo", prob_r), ("malla", prob_m)]:
        pd_ = p_[det]; pd_ = pd_[np.isfinite(pd_)]
        pa = p_[es_esp]; pa = pa[np.isfinite(pa)]
        pc = float((pa < np.median(pd_)).mean() * 100)
        print(f"{nom:<10}{np.median(pd_):>26.4f}{np.median(pa):>12.4f}"
              f"{pc:>11.1f}")
        res[f"pctl_riesgo_fuego_{nom}"] = pc

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmax = float(np.nanpercentile(np.concatenate(
        [prob_r[es_esp], prob_m[es_esp]]), 99.5))
    orig = "lower" if ys[1] > ys[0] else "upper"
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, p_, ti in [(axes[0], prob_r, "referencia — meteo del cubo"),
                       (axes[1], prob_m,
                        "malla — ERA5-Land en nodos, sin IDW")]:
        im = ax.imshow(p_, origin=orig, cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.scatter(fix2[okd], fiy2[okd], s=12, marker="o", facecolors="none",
                   edgecolors="blue", linewidths=0.7)
        ax.set_title(ti)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, label="probabilidad del modelo",
                 fraction=0.03)
    fig.suptitle(f"Riesgo de incendio — {a.fecha}   ·   círculos azules = "
                 f"detecciones FIRMS del día (n={int(det.sum())})")
    salida = f"{DIR}/eda/malla_mapa_riesgo_{a.fecha}.png"
    fig.savefig(salida, dpi=140, bbox_inches="tight")

    with open(f"{DIR}/dataset/malla_05_riesgo_{a.fecha}.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    ds.close()
    print(f"\nGuardado: eda/malla_mapa_riesgo_{a.fecha}.png")
    print(f"          dataset/malla_05_riesgo_{a.fecha}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fecha", nargs="?", default="2024-09-17")
    p.add_argument("--permitir-descarga", action="store_true")
    p.add_argument("--mapear-fwi", action="store_true",
                   help="lleva las features de nivel del FWI a la escala del "
                        "cubo con malla_data/mapeo_fwi.npz (módulo 3b)")
    main(p.parse_args())
