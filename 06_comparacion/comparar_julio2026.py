#!/usr/bin/env python3
"""
Comparación de producción (AEMET + IDW) contra la malla en julio de 2026,
con las áreas quemadas de EFFIS como referencia.

No toca producción. Escribe salida/julio2026_*.png / .json.

Por qué esta comparación y no la del 20-ago
-------------------------------------------
La comparación del 20-ago fue de un día y 35 celdas con detección FIRMS. Con
eso no se decide nada: mide acuerdo, no acierto. Y FIRMS de [D-5, D-1] es
feature de los dos mapas, así que las detecciones del día están
correlacionadas con una entrada que ambos comparten.

Aquí se mejoran tres cosas:

 1. Varios días. El colector horario de AEMET cubre 2026-07-01 a 07-27, con
    864 estaciones. Eso
    permite reconstruir el camino de producción sin llamar a su API, que es
    lo que hizo tardar 2h30 la corrida del 20-ago.

 2. Verdad-terreno de superficie. `effis_ba_season_ES.geojson`: 1.871
    polígonos de área quemada con fecha. Superficie real, no detecciones
    puntuales de satélite. Y no es entrada de ningún mapa.

 3. La misma temporada que se quiere servir.

Qué se compara y qué se deja fijo
---------------------------------
Las dos ramas comparten todo lo no meteorológico: estáticas y CLC del cubo,
vegetación y LST de la climatología mensual, EGIF, FIRMS, rayos, calendario y
el mismo modelo (`xgb_v2_prototipo`). Solo cambia de dónde sale la meteo:

    producción : 705 estaciones AEMET -> features por estación -> IDW k=8 con
                 corrección de altitud (GAMMA) -> celdas
    malla      : 5.605 nodos ERA5-Land -> cada celda toma su nodo

Julio de 2026 está entero dentro del reanálisis (era5land_diario llega al
13-ago), así que la malla va sin rama de previsión. Es deliberado: aquí se
mide el IDW contra los nodos, que es una pregunta distinta de la del módulo 2b.

Dos decisiones que hay que conocer para leer los números
--------------------------------------------------------
- Spin-up igualado. El colector empieza el 1-jul y no hay mayo-junio de AEMET
  sin volver a su API. Si la malla arrancase en mayo y AEMET en julio, la
  ventaja sería del método, no del dato. Así que ambas ramas arrancan el FWI
  el 1-jul con el mismo estado inicial, y solo se evalúa desde el 15-jul
  (>=14 días de spin-up). Un DC bajo deprime las dos por igual y la métrica
  es el orden dentro de cada día, donde un desplazamiento común se cancela.

- Viento. Producción, cuando tira de la API diaria, aproxima `viento_max` por
  min(1,5 x velmedia, racha). Aquí se usa el máximo horario real del colector,
  que es la definición del cubo y la que usa `serie_diaria_colector`. Es un
  dato mejor que el de producción: si acaso, la favorece.

Uso: python comparar_julio2026.py [--desde 2026-07-15] [--hasta 2026-07-27]
"""

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer
from scipy.spatial import cKDTree

import os

import config
from fwi_canadiense import calcular_fwi_serie
from malla_05_riesgo import ventanas
from riesgo_hoy import firms_nrt, mensual

COLECTOR = os.environ.get("TFM_COLECTOR", f"{config.FUENTE}/colector")
DB = f"{COLECTOR}/data/aemet_horario.db"
EFFIS = (f"{COLECTOR}/data/"
         "effis_ba_season_ES.geojson")
INI = "2026-07-01"
GAMMA = 0.0065
FEATS_IDW = ["fwi", "fwi_pctl_local", "fwi_anom_sigma", "t2m_max", "t2m_min",
             "rh_min", "viento_max", "precip_dia", "precip_7d", "precip_15d",
             "precip_30d", "fwi_med_7d", "fwi_max_7d", "fwi_med_15d",
             "fwi_med_30d", "rh_min_med_7d", "t2m_max_med_7d",
             "viento_max_med_7d", "dias_sin_lluvia"]
FEATS_TEMP = {"t2m_max", "t2m_min", "t2m_max_med_7d"}


def estaciones():
    """La misma tabla que usa producción (`tiempo_real._estaciones`).

    Se lee el parquet directamente en vez de importar `tiempo_real`: ese
    módulo arrastra todas las dependencias del entrenamiento, y de él solo hace falta
    esta tabla."""
    return pd.read_parquet(
        f"{config.FUENTE}/prototipo/estaciones_prototipo.parquet") \
        .set_index("idema")


# ------------------------------------------------------------- rama AEMET
def series_estaciones(fechas):
    """Diario por estación desde el colector horario, receta de producción."""
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT idema, fint, ta, tamax, tamin, hr, vv, prec "
                     "FROM observacion_horaria", con)
    con.close()
    df["fecha"] = pd.to_datetime(df["fint"], utc=True).dt.tz_localize(None) \
                    .dt.normalize()
    g = df.groupby(["idema", "fecha"])
    d = pd.DataFrame({
        "tmax": g["tamax"].max().combine_first(g["ta"].max()),
        "tmin": g["tamin"].min().combine_first(g["ta"].min()),
        "hr_min": g["hr"].min(),
        "viento_max": g["vv"].max(),
        "prec": g["prec"].sum(min_count=1),
        "n_horas": g["ta"].count()}).reset_index()
    d = d[d["n_horas"] >= 18]              # misma regla que producción
    S = {}
    for v in ("tmax", "tmin", "hr_min", "viento_max", "prec"):
        S[v] = d.pivot(index="fecha", columns="idema", values=v).reindex(fechas)
    return S


def features_aemet(S, fwi, est, ids, fechas, k, mes):
    """Las 21 features meteo por estación para el día `k`, como producción."""
    sl = slice(0, k + 1)
    F = ventanas(fwi[sl], S["tmax"].values[sl], S["tmin"].values[sl],
                 S["hr_min"].values[sl], S["viento_max"].values[sl],
                 S["prec"].values[sl])
    pct = np.full(len(ids), np.nan)
    ano = np.full(len(ids), np.nan)
    for j, i in enumerate(ids):
        try:
            c = np.load(f"{config.FUENTE}/prototipo/clim_fwi/{i}.npz")[f"m{mes}"]
        except Exception:
            continue
        c = c[np.isfinite(c)]
        if len(c) and np.isfinite(F["fwi"][j]):
            pct[j] = (c <= F["fwi"][j]).mean() * 100
            if c.std() > 0:
                ano[j] = (F["fwi"][j] - c.mean()) / c.std()
    F["fwi_pctl_local"], F["fwi_anom_sigma"] = pct, ano
    return F


def idw(est, ids, ok, ds, es_esp, k=8):
    """Pesos IDW de estaciones a celdas peninsulares, como `mapa_riesgo_hoy`."""
    xs, ys = ds["x"].values, ds["y"].values
    gx, gy = np.meshgrid(xs, ys)
    cel = np.column_stack([gx[es_esp], gy[es_esp]])
    e = est.loc[ids[ok]]
    arbol = cKDTree(np.column_stack([e["x3035"], e["y3035"]]))
    dist, j = arbol.query(cel, k=min(k, len(e)), workers=-1)
    w = 1.0 / np.maximum(dist, 1.0) ** 2
    ny, nx = es_esp.shape

    def interpolar(vals):
        v = vals[ok][j]
        m = np.isfinite(v)
        ww = np.where(m, w, 0.0)
        campo = np.full((ny, nx), np.nan)
        with np.errstate(invalid="ignore"):
            campo[es_esp] = (ww * np.where(m, v, 0)).sum(1) / ww.sum(1)
        return campo

    return interpolar


# -------------------------------------------------------------- EFFIS
def quemadas(dia, ds, es_esp, ruta=EFFIS):
    """Celdas con área quemada de EFFIS cuya FIREDATE es ese día."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    g = gpd.read_file(ruta)
    col = "FIREDATE" if "FIREDATE" in g.columns else "firedate"
    g["f"] = pd.to_datetime(g[col], errors="coerce", utc=True) \
               .dt.tz_localize(None).dt.normalize()
    sel = g[g["f"] == dia]
    ny, nx = es_esp.shape
    if not len(sel):
        return np.zeros((ny, nx), bool), 0.0
    sel = sel.set_crs(4326, allow_override=True).to_crs(3035)
    xs, ys = ds["x"].values, ds["y"].values
    px, ay = xs[1] - xs[0], abs(ys[1] - ys[0])
    # from_origin quiere la esquina noroeste y un paso vertical positivo. En
    # este cubo `y` va de norte a sur (paso -1000), así que el norte es ys[0];
    # pasarle el paso negativo devolvía una máscara vacía sin avisar.
    norte = max(ys[0], ys[-1]) + ay / 2
    tr = from_origin(xs[0] - px / 2, norte, px, ay)
    m = rasterize(((geom, 1) for geom in sel.geometry if geom is not None),
                  out_shape=(ny, nx), transform=tr, fill=0,
                  all_touched=True).astype(bool)
    if ys[1] > ys[0]:                    # si la fila 0 fuese el sur, voltear
        m = m[::-1]
    ha = pd.to_numeric(sel["AREA_HA"], errors="coerce").sum()
    return m & es_esp, float(ha)


def auc(pos, neg):
    """AUC-ROC por rangos (Mann-Whitney), sin sklearn."""
    x = np.concatenate([pos, neg])
    r = pd.Series(x).rank().values[:len(pos)]
    return float((r.sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main(a):
    fechas = pd.date_range(INI, "2026-07-27", freq="D")
    obj = pd.date_range(a.desde, a.hasta, freq="D")
    est = estaciones()

    print("  agregando el colector horario...", flush=True)
    S = series_estaciones(fechas)
    ids = np.array([i for i in S["tmax"].columns if i in est.index])
    for v in S:
        S[v] = S[v][ids]
    print(f"  {len(ids)} estaciones con estáticas de producción", flush=True)

    meses = fechas.month.values
    fwi = np.full((len(fechas), len(ids)), np.nan)
    for j in range(len(ids)):
        fwi[:, j] = calcular_fwi_serie(
            S["tmax"].values[:, j], S["hr_min"].values[:, j],
            S["viento_max"].values[:, j] * 3.6, S["prec"].values[:, j],
            meses)["fwi"]

    # --- la rama malla, con el mismo arranque del FWI (1-jul) --------------
    print("  ERA5-Land en los nodos (arranque igualado el 1-jul)...", flush=True)
    n = np.load(config.NODOS)
    la, lo, idx_nodo = n["nodo_lat"], n["nodo_lon"], n["idx_nodo"]
    dsr = xr.open_dataset(config.entrada("era5land_diario.nc"))
    pts = dsr.sel(latitude=xr.DataArray(la, dims="n"),
                  longitude=xr.DataArray(lo, dims="n"), method="nearest")
    fr = pd.to_datetime(dsr["fecha"].values).normalize()
    dentro = np.isin(fr, fechas)
    N = {v: pts[v].values[dentro] for v in
         ("tmax", "tmin", "hr_min", "viento_max", "prec")}
    dsr.close()
    fwi_n = np.full(N["tmax"].shape, np.nan)
    for j in range(len(la)):
        fwi_n[:, j] = calcular_fwi_serie(
            N["tmax"][:, j], N["hr_min"][:, j], N["viento_max"][:, j] * 3.6,
            N["prec"][:, j], meses)["fwi"]

    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    es_esp = ds["is_spain"].values.astype(bool)
    B = {}
    for k_, v in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                  "rugosidad": "roughness_mean",
                  "dist_carreteras": "dist_to_roads_mean",
                  "dist_rios": "dist_to_waterways_mean"}.items():
        B[k_] = ds[v].values
    B["popdens"] = ds["popdens_2020"].values
    for k_, suf in {"clc_bosque": "forest_proportion",
                    "clc_matorral": "scrub_proportion",
                    "clc_agricola": "agricultural_proportion",
                    "clc_artificial": "artificial_proportion",
                    "clc_abierto": "open_space_proportion",
                    "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
        B[k_] = ds[f"CLC_2018_{suf}"].values
    B["ccaa"] = np.nan_to_num(ds["AutonomousCommunities"].values,
                              nan=-1).astype(int)
    B["frp_max_50km_7d"], B["n_detec_50km_7d"] = firms_nrt(ds)
    B["rayos_dia"] = np.zeros((ny, nx))
    B["rayos_7d"] = np.zeros((ny, nx))
    B.update(mensual(ds, 7))
    B["ndvi_med_30d"] = B["ndvi"]

    FULL = json.load(open(f"{config.MODELOS}/xgb_v2_prototipo_features.json"))
    modelo = xgb.XGBClassifier()
    modelo.load_model(f"{config.MODELOS}/xgb_v2_prototipo.ubj")
    sel = es_esp.ravel()
    import holidays
    fest = holidays.Spain(years=[2026])
    clim = np.load(config.CLIM)
    C7, n7 = clim["m7"], clim["n7"].astype(float)

    filas, pool = [], {"produccion": [], "malla": []}
    for dia in obj:
        k = int(np.where(fechas == dia)[0][0])
        quem, ha = quemadas(dia, ds, es_esp)
        if quem.sum() == 0:
            print(f"  {dia.date()}: sin área quemada de EFFIS · se salta",
                  flush=True)
            continue

        # --- producción ---
        FA = features_aemet(S, fwi, est, ids, fechas, k, 7)
        ok = np.isfinite(FA["fwi"]) & np.isfinite(FA["t2m_max"]) \
            & np.isfinite(FA["rh_min"])
        # Misma regla que producción: por debajo de 100 estaciones no se sirve
        # mapa. El 27-jul el colector se cortó a las 09:00 y ningún día llega
        # a las 18 h que exige la receta, así que queda entero a NaN.
        if ok.sum() < 100:
            print(f"  {dia.date()}: solo {int(ok.sum())} estaciones válidas "
                  f"· se salta (producción tampoco serviría)", flush=True)
            continue
        interp = idw(est, ids, ok, ds, es_esp)
        # corrección de altitud igual que producción: la temperatura se lleva
        # a nivel del mar en la estación, se interpola, y se devuelve a la
        # altitud de la celda.
        z_est = est.loc[ids, "elevacion"].values
        P = dict(B)
        for c in FEATS_IDW:
            if c in FEATS_TEMP:
                P[c] = interp(FA[c] + GAMMA * z_est) - GAMMA * B["elevacion"]
            else:
                P[c] = interp(FA[c])
        P["fwi"] = np.clip(P["fwi"], 0, None)
        P["fwi_pctl_local"] = np.clip(P["fwi_pctl_local"], 0, 100)
        P["rh_min"] = np.clip(P["rh_min"], 0, 100)
        es = 0.6108 * np.exp(17.27 * P["t2m_max"] / (P["t2m_max"] + 237.3))
        P["vpd_max"] = es * (1 - P["rh_min"] / 100)

        # --- malla ---
        FN = ventanas(fwi_n[:k + 1], N["tmax"][:k + 1], N["tmin"][:k + 1],
                      N["hr_min"][:k + 1], N["viento_max"][:k + 1],
                      N["prec"][:k + 1])
        with np.errstate(invalid="ignore"):
            cnt = np.nansum(C7 <= FN["fwi"][None, :], axis=0)
            FN["fwi_pctl_local"] = np.where(n7 > 0, cnt / np.maximum(n7, 1) * 100,
                                            np.nan)
            cm, cs = np.nanmean(C7, axis=0), np.nanstd(C7, axis=0)
            FN["fwi_anom_sigma"] = np.where((n7 > 0) & (cs > 0),
                                            (FN["fwi"] - cm) / cs, np.nan)
        malo = ~np.isfinite(FN["fwi"]) | (n7 == 0)
        if malo.any():
            bu = np.where(~malo)[0]
            ar = cKDTree(np.column_stack([la[bu], lo[bu]]))
            _, kk = ar.query(np.column_stack([la[malo], lo[malo]]))
            for v in FN:
                FN[v][malo] = FN[v][bu[kk]]
        seg = np.clip(idx_nodo, 0, None)
        M = dict(B)
        for v, x in FN.items():
            c = np.asarray(x, float)[seg]
            c[idx_nodo < 0] = np.nan
            M[v] = c

        cal = {"es_festivo": np.full((ny, nx), float(dia.weekday() >= 5
                                                     or dia.date() in fest)),
               "mes": np.full((ny, nx), 7),
               "dia_anio": np.full((ny, nx), dia.dayofyear)}
        fila = {"fecha": str(dia.date()), "celdas_quemadas": int(quem.sum()),
                "area_ha": ha, "estaciones": int(ok.sum())}
        for nom, F in (("produccion", P), ("malla", M)):
            F.update(cal)
            X = pd.DataFrame({c: F[c].ravel()[sel] for c in FULL})
            pr = np.full(ny * nx, np.nan)
            pr[sel] = modelo.predict_proba(X)[:, 1]
            pr = pr.reshape(ny, nx)
            todas = pr[es_esp]
            fin = np.isfinite(todas)
            q = pr[quem & es_esp]
            q = q[np.isfinite(q)]
            orden = np.sort(todas[fin])
            pct = np.searchsorted(orden, q, side="right") / len(orden) * 100
            pool[nom].append(pct)
            noq = pr[es_esp & ~quem]
            noq = noq[np.isfinite(noq)]
            fila[f"pctl_{nom}"] = float(np.median(pct))
            fila[f"auc_{nom}"] = auc(q, noq)
        filas.append(fila)
        print(f"  {dia.date()}  quemadas {fila['celdas_quemadas']:4d} celdas "
              f"({ha:8.0f} ha) · pctl prod {fila['pctl_produccion']:5.1f} "
              f"malla {fila['pctl_malla']:5.1f} · AUC "
              f"{fila['auc_produccion']:.3f}/{fila['auc_malla']:.3f}",
              flush=True)
    ds.close()

    A = {k_: np.concatenate(v) for k_, v in pool.items()}
    print("\n" + "=" * 72)
    print("JULIO 2026 — área quemada EFFIS como verdad-terreno")
    print("=" * 72)
    print(f"  días evaluados: {len(filas)} · celdas quemadas: "
          f"{len(A['produccion']):,}")
    print(f"\n{'mapa':<14}{'pctl mediano':>14}{'pctl medio':>13}"
          f"{'AUC medio':>12}")
    res = {"dias": filas, "resumen": {}}
    for nom in ("produccion", "malla"):
        am = float(np.mean([f[f"auc_{nom}"] for f in filas]))
        r = dict(pctl_mediano=float(np.median(A[nom])),
                 pctl_medio=float(A[nom].mean()), auc_medio=am)
        res["resumen"][nom] = r
        print(f"{nom:<14}{r['pctl_mediano']:>14.1f}{r['pctl_medio']:>13.1f}"
              f"{am:>12.3f}")

    # Bootstrap por días, no por celdas. Las celdas de un mismo incendio no
    # son independientes: remuestrearlas da un intervalo demasiado estrecho
    # (con estos datos, [+4,2, +5,3] en vez de [-2,1, +10,4]). La unidad de
    # muestreo real es el día, y son 17.
    rng = np.random.default_rng(0)
    dif = np.array([f["pctl_malla"] - f["pctl_produccion"] for f in filas])
    dauc = np.array([f["auc_malla"] - f["auc_produccion"] for f in filas])
    w = np.array([f["celdas_quemadas"] for f in filas], float)

    def boot(v, peso=None):
        o = []
        for _ in range(20000):
            i = rng.integers(0, len(v), len(v))
            o.append(np.average(v[i], weights=peso[i]) if peso is not None
                     else v[i].mean())
        return [float(x) for x in np.percentile(o, [2.5, 97.5])]

    ic, ica, icw = boot(dif), boot(dauc), boot(dif, w)
    print(f"\n  diferencia malla − producción, remuestreando DÍAS:")
    print(f"    percentil            {dif.mean():+5.1f} pts · IC95 "
          f"[{ic[0]:+.1f}, {ic[1]:+.1f}]")
    print(f"    AUC                  {dauc.mean():+5.3f}     · IC95 "
          f"[{ica[0]:+.3f}, {ica[1]:+.3f}]")
    print(f"    percentil ponderado  {np.average(dif, weights=w):+5.1f} pts · "
          f"IC95 [{icw[0]:+.1f}, {icw[1]:+.1f}]  (por superficie quemada)")
    print(f"\n  la malla gana en {int((dif > 0).sum())} de {len(dif)} días; "
          f"en los 5 de más superficie, {int((dif[np.argsort(-w)[:5]] > 0).sum())}/5")
    ver = ("la malla gana" if ic[0] > 0 else
           ("la malla pierde" if ic[1] < 0 else
            "mejor de media, pero con 17 días no se distingue del ruido"))
    res.update(dif_pctl_media=float(dif.mean()), ic95_pctl=ic,
               dif_auc_media=float(dauc.mean()), ic95_auc=ica,
               dif_pctl_ponderada=float(np.average(dif, weights=w)),
               ic95_ponderada=icw, dias_ganados=int((dif > 0).sum()),
               veredicto=ver)
    print(f"  → {ver}")
    with open(config.salida("julio2026_effis.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado: {config.salida('julio2026_effis.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--desde", default="2026-07-15")
    p.add_argument("--hasta", default="2026-07-27")
    main(p.parse_args())
