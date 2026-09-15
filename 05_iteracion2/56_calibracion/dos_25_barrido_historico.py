#!/usr/bin/env python3
"""
Dos modelos, paso 25: barrido histórico 2015-2024 sobre el cubo, para poder
calibrar el «si» (aviso de día) con días de cero fuego.

No toca producción. Lee IberFire.nc, celdas.parquet, los modelos y FIRMS
2015-2024; escribe salida/dos_25_*.{npz,csv}.

Por qué este script y no el replay
----------------------------------
El «si» (¿hoy hay aviso?) necesita un umbral absoluto, y un umbral absoluto
solo se puede estimar con días en los que no arde nada. `replay_temporada.py`
solo cubre 2025-2026 de mayo a noviembre, donde el 90 % de los días tienen
fuego: ahí el umbral no se mide, se extrapola.

El replay existe para reconstruir lo que la cadena habría publicado (descarga
ERA5-Land, IFS archivado, FWI, FIRMS). Para calibrar sobre el histórico nada
de eso hace falta: el cubo IberFire ya trae la meteorología diaria a 1 km de
2007-2024 y la etiqueta. Se puntúa directamente sobre él, que es lo que ya
hacen `dos_05` y `dos_13` con el banco de evaluación, pero sobre todos los
días y no solo sobre las muestras.

Coste medido: el NetCDF está troceado (521, 77, 99), con el eje temporal
largo, así que un bloque espacial con toda su serie se lee en 0,3 s y un día
suelto de todo el mapa cuesta 2,8 s. Por eso el bucle va por bloque (144) y
no por día.

Qué se puntúa
-------------
  prod    xgb_v2_prototipo (46 features, etiqueta EGIF)
  unico   donde_dia_effis (46 features, negativos del mismo día 1:3)
  r10     donde_dia_effis_r10 (ídem, 1:10)
  cuando  cuando (29 dinámicas, etiqueta EGIF)
  pareja  donde_effis_c (mapa estático de dos_13) × cuando

Features: réplica exacta de `extraer_features_cubo.procesar_bloque`, pero
vectorizada sobre (días × celdas) en vez de fila a fila. Ventanas 7/15/30 que
excluyen D; `dias_sin_lluvia` cuenta D incluido y corta en NaN; climatología
local del FWI 2008-2014 por celda y mes. `rayos_*` = 0, que es lo que sirve
producción (`riesgo_hoy.py` L304). FIRMS de `firms_iberia_2015_2024.parquet`.

Qué se guarda (y por qué no los mapas)
--------------------------------------
3.653 días × 498.530 celdas × 5 modelos no cabe en disco y no hace falta. Lo
que la calibración necesita es la distribución del día, así que se acumula un
histograma por día y modelo (4.096 bins logarítmicos). De ahí salen exactos,
a resolución de bin, cualquier cuantil y el número de celdas sobre cualquier
umbral candidato, sin fijar el umbral ahora.

  · histograma sobre una submuestra sistemática de 1 de cada 5 celdas
    (99.706), que estima el p98 del día con error despreciable (el p98 son
    ~2.000 celdas de la submuestra);
  · la puntuación exacta en cada celda que arde ese día (todas, no la
    submuestra), para el percentil del fuego;
  · la verdad del día: celdas quemadas, celdas de primer día, y el desglose
    noroeste/resto (los dos regímenes, ver dos_26).

Uso:
    python dos_25_barrido_historico.py --prep      # solo FIRMS y EGIF mensual
    python dos_25_barrido_historico.py             # el barrido (reanudable)
    python dos_25_barrido_historico.py --bloques 0 3   # un rango, para probar
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer
from scipy.ndimage import maximum_filter, uniform_filter

import config
import config_expansion as ce
from dos_05_modelos import FEATS_CUANDO

INI, FIN = "2015-01-01", "2024-12-31"
CHUNK_Y, CHUNK_X = 77, 99          # troceado interno del NetCDF
ANIOS_CLIM = (2008, 2014)
UMBRAL_LLUVIA_MM = 1.0
TOPE_DIAS_SECOS = 120
PASO_SUB = 5                       # 1 de cada 5 celdas para el histograma
CHUNK_D = 512                      # días por tramo de predicción
NBINS = 4096
BINS = np.concatenate([[0.0], np.logspace(-8, 0, NBINS - 1)])
VARS = ["FWI", "t2m_max", "t2m_min", "RH_min", "wind_speed_max",
        "total_precipitation_mean", "LST", "NDVI", "LAI", "SWI_010",
        "is_holiday", "is_fire"]
MODELOS = ("prod", "unico", "r10", "cuando", "pareja")
SAL = config.salida("dos_25")
NO_PROV = {"15", "27", "32", "36", "33", "39", "24", "49"}   # Galicia, AST, CAN, LE, ZA


# --------------------------------------------------------------------------
# preparación: rejilla, celdas, FIRMS y EGIF mensual
# --------------------------------------------------------------------------
def suma_caja(campo, radio):
    k = 2 * radio + 1
    return uniform_filter(campo.astype(float), size=k, mode="constant") * k * k


def celdas_y_region(ds):
    """Celdas de España con estáticas, submuestra y etiqueta de región.

    Solo se procesan las celdas `keep` = submuestra ∪ las que arden algún día
    de 2015-2024. El resto no aporta: el histograma del día sale de la
    submuestra (insesgada) y el percentil del fuego, de las que arden. Sin
    esto el bloque son 3.400 celdas en vez de 800 y la memoria se dispara.
    """
    cel = pd.read_parquet(f"{ce.DATASET}/celdas.parquet")
    cel = cel.sort_values(["iy", "ix"]).reset_index(drop=True)
    mun = pd.DataFrame(json.load(open(f"{config.ESTADO}/municipios.json")))
    mun["lat"] = pd.to_numeric(mun.latitud_dec, errors="coerce")
    mun["lon"] = pd.to_numeric(mun.longitud_dec, errors="coerce")
    mun = mun.dropna(subset=["lat", "lon"])
    from scipy.spatial import cKDTree
    _, j = cKDTree(np.c_[mun.lat, mun.lon]).query(np.c_[cel.lat, cel.lon])
    cel["prov"] = mun.id_old.str[:2].values[j]
    cel["es_no"] = cel.prov.isin(NO_PROV)
    cel["es_sub"] = (np.arange(len(cel)) % PASO_SUB) == 0
    z = np.load(f"{ce.DATASET}/cubo_etiquetas.npz")
    an = list(z["anios"])
    arde = z["n_fuego_anio"][an.index(2015):an.index(2024) + 1].sum(0) > 0
    cel["arde"] = arde[cel.iy.values, cel.ix.values]
    cel = cel[cel.es_sub | cel.arde].reset_index(drop=True)
    return cel


def egif_mensual(ds):
    """n_fuegos_10km_mismomes_hist para los 12 meses (la caché solo trae 4-11)."""
    f = f"{SAL}_egif_mes.npz"
    if os.path.exists(f):
        return np.load(f)["g"]
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    xs, ys = ds["x"].values, ds["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    eg = pd.read_csv(config.EGIF, usecols=["fecha", "lat", "lng"]).dropna()
    eg["fecha"] = pd.to_datetime(eg["fecha"], errors="coerce")
    eg = eg.dropna()
    eg = eg[eg["fecha"].dt.year >= 2008]
    ex, ey = tr.transform(eg["lng"].values, eg["lat"].values)
    eix = np.rint((ex - xs[0]) / (xs[1] - xs[0])).astype(int)
    eiy = np.rint((ey - ys[0]) / (ys[1] - ys[0])).astype(int)
    mes = eg["fecha"].dt.month.values
    out = np.zeros((12, ny, nx), np.float32)
    for m in range(1, 13):
        ok = (eix >= 0) & (eix < nx) & (eiy >= 0) & (eiy < ny) & (mes == m)
        g = np.zeros((ny, nx))
        np.add.at(g, (eiy[ok], eix[ok]), 1)
        out[m - 1] = suma_caja(g, 10)
        print(f"  EGIF m{m}: {int(ok.sum()):,} incendios", flush=True)
    np.savez_compressed(f, g=out)
    return out


def firms_diario(ds, cel, dias):
    """frp_max_50km_7d y n_detec_50km_7d en las celdas, día a día [D-7, D-1]."""
    f = f"{SAL}_firms_keep.npy"     # solo celdas `keep`: 3,4 GB en vez de 14,6
    if os.path.exists(f):
        return np.load(f, mmap_mode="r")
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    xs, ys = ds["x"].values, ds["y"].values
    fp = f"{config.FUENTE}/firms_iberia_2015_2024.parquet"
    d = pd.read_parquet(fp)
    col = {c.lower(): c for c in d.columns}
    la, lo = col.get("latitude", "latitude"), col.get("longitude", "longitude")
    fe = col.get("acq_date", "acq_date")
    frp = col.get("frp", "frp")
    d["fecha"] = pd.to_datetime(d[fe]).dt.normalize()
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    px, py = tr.transform(d[lo].values, d[la].values)
    d["ix"] = np.rint((px - xs[0]) / (xs[1] - xs[0])).astype(int)
    d["iy"] = np.rint((py - ys[0]) / (ys[1] - ys[0])).astype(int)
    d = d[(d.ix >= 0) & (d.ix < nx) & (d.iy >= 0) & (d.iy < ny)]
    print(f"  FIRMS: {len(d):,} focos {d.fecha.min().date()}→{d.fecha.max().date()}",
          flush=True)
    por_dia = {k: v for k, v in d.groupby("fecha")}
    iy, ix = cel.iy.values, cel.ix.values
    out = np.zeros((len(dias), len(cel), 2), np.float32)
    gf = np.zeros((8, ny, nx), np.float32)     # buffer circular de 8 días
    gn = np.zeros((8, ny, nx), np.float32)
    for k, dia in enumerate(dias):
        j = k % 8
        gf[j] = 0.0
        gn[j] = 0.0
        v = por_dia.get(dia)
        if v is not None:
            np.maximum.at(gf[j], (v.iy.values, v.ix.values), v[frp].values)
            np.add.at(gn[j], (v.iy.values, v.ix.values), 1.0)
        prev = [(k - i) % 8 for i in range(1, 8) if k - i >= 0]   # D-7..D-1
        if prev:
            a = maximum_filter(gf[prev].max(0), size=101, mode="constant")
            b = suma_caja(gn[prev].sum(0), 50)
            out[k, :, 0] = a[iy, ix]
            out[k, :, 1] = b[iy, ix]
        if k % 365 == 0:
            print(f"    FIRMS {dia.date()}", flush=True)
    np.save(f, out)
    return out


# --------------------------------------------------------------------------
# features del bloque, vectorizadas sobre (dias, celdas)
# --------------------------------------------------------------------------
def ventana_media(a, w):
    """Media de los w días anteriores a cada t (excluye t). Ignora los NaN."""
    n = a.shape[0]
    v = np.nan_to_num(a, nan=0.0)
    ok = (~np.isnan(a)).astype(np.float32)
    cs = np.concatenate([np.zeros((1,) + a.shape[1:], np.float32), np.cumsum(v, 0)])
    ck = np.concatenate([np.zeros((1,) + a.shape[1:], np.float32), np.cumsum(ok, 0)])
    lo = np.clip(np.arange(n) - w, 0, n)
    hi = np.arange(n)
    s = cs[hi] - cs[lo]
    c = ck[hi] - ck[lo]
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def ventana_suma(a, w):
    n = a.shape[0]
    v = np.nan_to_num(a, nan=0.0)
    cs = np.concatenate([np.zeros((1,) + a.shape[1:], np.float32), np.cumsum(v, 0)])
    return cs[np.arange(n)] - cs[np.clip(np.arange(n) - w, 0, n)]


def ventana_max(a, w):
    n = a.shape[0]
    v = np.where(np.isnan(a), -np.inf, a)
    out = np.full(a.shape, -np.inf, np.float32)
    for k in range(1, w + 1):                  # w=7: siete desplazamientos
        out[k:] = np.maximum(out[k:], v[:-k])
    return np.where(np.isinf(out), np.nan, out)


def dias_secos(pre):
    """Días consecutivos sin lluvia contando D; corta en NaN. Tope 120."""
    n = pre.shape[0]
    seco = (pre < UMBRAL_LLUVIA_MM) & ~np.isnan(pre)
    out = np.zeros(pre.shape, np.float32)
    acc = np.zeros(pre.shape[1], np.float32)
    for t in range(n):
        acc = np.where(seco[t], np.minimum(acc + 1, TOPE_DIAS_SECOS), 0.0)
        out[t] = acc
    return out


def pctl_local(fwi, meses_t, clim_idx):
    """Percentil y anomalía del FWI contra la climatología 2008-14 de la celda
    y el mes. Réplica de `procesar_bloque`, por bloques de días."""
    n, nc = fwi.shape
    pct = np.full((n, nc), np.nan, np.float32)
    ano = np.full((n, nc), np.nan, np.float32)
    for m in range(1, 13):
        idx = clim_idx[m]
        if not len(idx):
            continue
        c = fwi[idx]
        csort = np.sort(np.where(np.isnan(c), np.inf, c), axis=0)
        ncl = (~np.isnan(c)).sum(0).astype(np.float32)
        cmed = np.nanmean(c, 0)
        cstd = np.nanstd(c, 0)
        dias = np.where(meses_t == m)[0]
        for i in range(0, len(dias), 64):
            sl = dias[i:i + 64]
            q = fwi[sl]                                    # (k, nc)
            cnt = (csort[:, None, :] <= q[None, :, :]).sum(0).astype(np.float32)
            pct[sl] = np.where(ncl > 0, cnt / np.maximum(ncl, 1) * 100, np.nan)
        ano[dias] = np.where(cstd > 0, (fwi[dias] - cmed) / cstd, np.nan)
    return pct, ano


def vpd_kpa(t_c, rh):
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    return es * (1.0 - rh / 100.0)


def guarda(fh, hist, hechos, quem):
    """Checkpoint atómico: .tmp y rename, para no dejarlo a medias."""
    q = np.concatenate(quem) if quem else np.zeros((0, 4 + len(MODELOS)), np.float32)
    np.savez_compressed(fh + ".tmp.npz", hist=hist,
                        hechos=np.array(sorted(hechos)), quem=q)
    os.replace(fh + ".tmp.npz", fh)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--bloques", nargs=2, type=int, default=None)
    a = ap.parse_args()

    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    tiempos = ds["time"].values.astype("datetime64[D]")
    t0 = tiempos[0].astype(int)
    dias = pd.date_range(INI, FIN, freq="D")
    t_of = dias.values.astype("datetime64[D]").astype(int) - t0
    anios = tiempos.astype("datetime64[Y]").astype(int) + 1970
    meses = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    m_clim = (anios >= ANIOS_CLIM[0]) & (anios <= ANIOS_CLIM[1])
    clim_idx = {m: np.where(m_clim & (meses == m))[0] for m in range(1, 13)}
    meses_t = meses[t_of]
    print(f"cubo {tiempos[0]}→{tiempos[-1]} · barrido {dias[0].date()}→"
          f"{dias[-1].date()} ({len(dias)} días)", flush=True)

    cel = celdas_y_region(ds)
    print(f"celdas {len(cel):,} · submuestra {int(cel.es_sub.sum()):,} "
          f"· NO {int(cel.es_no.sum()):,}", flush=True)
    cel[["iy", "ix", "lat", "lon", "prov", "es_no", "es_sub"]].to_parquet(
        f"{SAL}_celdas.parquet", index=False)
    eg_mes = egif_mensual(ds)
    firms = firms_diario(ds, cel, dias)
    if a.prep:
        print("prep terminada.")
        return

    def carga(fich):
        """Los modelos EFFIS viven en la expansión; producción, en los datos externos."""
        for d in (ce.MODELOS, config.MODELOS, f"{config.ESTADO}/modelos"):
            r = f"{d}/{fich}.ubj"
            if os.path.exists(r):
                m = xgb.XGBClassifier(); m.load_model(r)
                print(f"  modelo {fich}: {d}", flush=True)
                return m
        raise SystemExit(f"falta el modelo {fich}.ubj")

    FULL = json.load(open(f"{config.MODELOS}/xgb_v2_prototipo_features.json"))
    mods = {"prod": (carga("xgb_v2_prototipo"), FULL),
            "unico": (carga("donde_dia_effis"), FULL),
            "r10": (carga("donde_dia_effis_r10"), FULL),
            "cuando": (carga("cuando"), FEATS_CUANDO)}
    z = np.load(config.salida("dos_13_mapa_donde_effis_c.npz"))
    mapa_de = np.zeros((ds.sizes["y"], ds.sizes["x"]), np.float32)
    mapa_de[z["iy"], z["ix"]] = z["p"]
    p_de = mapa_de[cel.iy.values, cel.ix.values]

    nb_y = int(np.ceil(ds.sizes["y"] / CHUNK_Y))
    nb_x = int(np.ceil(ds.sizes["x"] / CHUNK_X))
    cel["_by"], cel["_bx"] = cel.iy // CHUNK_Y, cel.ix // CHUNK_X
    cel["_pos"] = np.arange(len(cel))
    bloques = sorted(cel.groupby(["_by", "_bx"]).groups)
    r0, r1 = a.bloques if a.bloques else (0, len(bloques))
    print(f"{len(bloques)} bloques con celdas; procesando [{r0}, {r1})", flush=True)

    hist = np.zeros((len(dias), len(MODELOS), NBINS), np.int32)
    quem = []                       # (dia, iy, ix, es_no, 5 scores)
    fh = f"{SAL}_hist.npz"
    if os.path.exists(fh):
        d0 = np.load(fh)
        hist = d0["hist"]
        hechos = set(map(tuple, d0["hechos"]))
        # `list(array_2d)` lo parte en filas de 1-D y luego np.concatenate
        # falla contra los bloques nuevos, que son 2-D: va como un solo elemento.
        quem = [d0["quem"]] if len(d0["quem"]) else []
        print(f"  reanudando: {len(hechos)} bloques ya hechos", flush=True)
    else:
        hechos = set()

    t_ini = time.time()
    for nb in range(r0, r1):
        by, bx = bloques[nb]
        if (by, bx) in hechos:
            continue
        sub = cel[(cel._by == by) & (cel._bx == bx)]
        sy = slice(by * CHUNK_Y, min((by + 1) * CHUNK_Y, ds.sizes["y"]))
        sx = slice(bx * CHUNK_X, min((bx + 1) * CHUNK_X, ds.sizes["x"]))
        t_b = time.time()
        # una variable cada vez y recorte inmediato a las celdas del bloque:
        # el diccionario entero eran 2,3 GB antes del recorte.
        yl = sub.iy.values - sy.start
        xl = sub.ix.values - sx.start
        loc = yl * (sx.stop - sx.start) + xl
        A = {}
        for v in VARS:
            A[v] = ds[v].isel(y=sy, x=sx).values.reshape(len(tiempos), -1)[:, loc]
        t_lee = time.time() - t_b

        fwi = A["FWI"]
        pct, ano = pctl_local(fwi, meses, clim_idx)
        F = {}
        F["fwi"] = fwi[t_of]
        F["fwi_pctl_local"] = pct[t_of]
        F["fwi_anom_sigma"] = ano[t_of]
        F["t2m_max"] = A["t2m_max"][t_of]
        F["t2m_min"] = A["t2m_min"][t_of]
        F["rh_min"] = A["RH_min"][t_of]
        F["viento_max"] = A["wind_speed_max"][t_of]
        F["precip_dia"] = A["total_precipitation_mean"][t_of]
        F["vpd_max"] = vpd_kpa(F["t2m_max"], F["rh_min"])
        F["lst"] = A["LST"][t_of]
        F["ndvi"] = A["NDVI"][t_of]
        F["lai"] = A["LAI"][t_of]
        F["swi010"] = A["SWI_010"][t_of]
        F["es_festivo"] = A["is_holiday"][t_of]
        pre = A["total_precipitation_mean"]
        for w in (7, 15, 30):
            F[f"precip_{w}d"] = ventana_suma(pre, w)[t_of]
        for w in (7, 15, 30):
            F[f"fwi_med_{w}d"] = ventana_media(fwi, w)[t_of]
        F["fwi_max_7d"] = ventana_max(fwi, 7)[t_of]
        F["rh_min_med_7d"] = ventana_media(A["RH_min"], 7)[t_of]
        F["t2m_max_med_7d"] = ventana_media(A["t2m_max"], 7)[t_of]
        F["viento_max_med_7d"] = ventana_media(A["wind_speed_max"], 7)[t_of]
        F["ndvi_med_30d"] = ventana_media(A["NDVI"], 30)[t_of]
        F["dias_sin_lluvia"] = dias_secos(pre)[t_of]
        nd, nc = F["fwi"].shape
        for c in ("elevacion", "pendiente", "rugosidad", "dist_carreteras",
                  "dist_rios", "popdens", "clc_bosque", "clc_matorral",
                  "clc_agricola", "clc_artificial", "clc_abierto",
                  "clc_agric_hetero", "ccaa"):
            F[c] = np.tile(sub[c].values.astype(np.float32), (nd, 1))
        # Ojo: `eg_mes[meses_t - 1]` materializaba (3.653, 920, 1188) = 16 GB.
        # Con indexado conjunto por difusión queda en (3.653, nc).
        F["n_fuegos_10km_mismomes_hist"] = eg_mes[(meses_t - 1)[:, None],
                                                  sub.iy.values[None, :],
                                                  sub.ix.values[None, :]]
        ff = np.asarray(firms[:, sub._pos.values, :])
        F["frp_max_50km_7d"] = ff[:, :, 0]
        F["n_detec_50km_7d"] = ff[:, :, 1]
        F["rayos_dia"] = np.zeros((nd, nc), np.float32)
        F["rayos_7d"] = np.zeros((nd, nc), np.float32)
        F["mes"] = np.tile(meses_t.astype(np.float32)[:, None], (1, nc))
        F["dia_anio"] = np.tile(dias.dayofyear.values.astype(np.float32)[:, None],
                                (1, nc))
        t_fea = time.time() - t_b - t_lee

        # Predicción por tramos de días: apilar las 46 columnas de golpe eran
        # 3.653 × nc × 46 × 4 B y, con el diccionario `X` duplicando `F`,
        # el proceso se iba de memoria y lo mataba el núcleo.
        S = {n: np.empty((nd, nc), np.float32) for n in MODELOS}
        for i in range(0, nd, CHUNK_D):
            sl = slice(i, min(i + CHUNK_D, nd))
            Xf = np.column_stack([F[c][sl].astype(np.float32).ravel() for c in FULL])
            k = sl.stop - sl.start
            for nom in ("prod", "unico", "r10"):
                S[nom][sl] = mods[nom][0].predict_proba(Xf)[:, 1].reshape(k, nc)
            del Xf
            Xc = np.column_stack([F[c][sl].astype(np.float32).ravel()
                                  for c in FEATS_CUANDO])
            S["cuando"][sl] = mods["cuando"][0].predict_proba(Xc)[:, 1].reshape(k, nc)
            del Xc
        S["pareja"] = S["cuando"] * p_de[sub._pos.values][None, :]
        t_pre = time.time() - t_b - t_lee - t_fea

        es_sub = sub["es_sub"].values
        ns = int(es_sub.sum())
        if ns:
            dia_idx = np.repeat(np.arange(nd), ns)
            for k, nom in enumerate(MODELOS):
                b = np.clip(np.searchsorted(BINS, S[nom][:, es_sub].ravel(),
                                            "right") - 1, 0, NBINS - 1)
                hist[:, k, :] += np.bincount(dia_idx * NBINS + b,
                                             minlength=nd * NBINS
                                             ).reshape(nd, NBINS).astype(np.int32)

        arde = A["is_fire"][t_of] > 0
        ti, ci = np.where(arde)
        if len(ti):
            quem.append(np.column_stack(
                [ti, sub.iy.values[ci], sub.ix.values[ci],
                 sub.es_no.values[ci].astype(int)] +
                [S[n][ti, ci] for n in MODELOS]).astype(np.float32))
        hechos.add((by, bx))
        el = time.time() - t_ini
        print(f"  bloque {nb+1}/{len(bloques)} ({by},{bx}) {nc} celdas · "
              f"lee {t_lee:.1f}s feat {t_fea:.1f}s pred {t_pre:.1f}s hist "
              f"{time.time()-t_b-t_lee-t_fea-t_pre:.1f}s · "
              f"quemadas {int(arde.sum()):,} · total {el/60:.1f} min", flush=True)
        if (nb - r0) % 20 == 19 or nb == r1 - 1:
            guarda(fh, hist, hechos, quem)
            print(f"    checkpoint guardado ({len(hechos)} bloques)", flush=True)

    guarda(fh, hist, hechos, quem)
    print(f"listo en {(time.time()-t_ini)/60:.1f} min → {fh}")


if __name__ == "__main__":
    main()
