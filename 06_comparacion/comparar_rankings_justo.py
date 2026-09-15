#!/usr/bin/env python3
"""
El cara a cara justo: previsión contra previsión, 18 días.

No toca producción. Escribe salida/ifs_historico.parquet (descarga) y
salida/rankings_justo.{csv,json} (evaluación).

Qué corrige
-----------
`comparar_rankings.py` comparó los rankings sellados del prototipo (que son
previsión) contra la malla alimentada con reanálisis. La malla tenía
retrovisor y aun así la diferencia no era significativa (+0,025 de AUC,
IC95 [-0,023, +0,073]). Ese número es una cota superior, no un empate.

Aquí la malla se reconstruye como lo que se serviría en operación: reanálisis hasta
D-7 y previsión IFS los últimos 7 días, con el mapeo de cuantiles a la escala
de ERA5-Land. El mismo híbrido que monta `riesgo_hoy.py`, pero para fechas
pasadas, con el IFS archivado tal y como se emitió entonces.

La cuota y la noche
-------------------
Para los días objetivo 22-jul a 14-ago con L=6 hacen falta días de IFS del
16-jul al 14-ago: 30 días = 3 tramos de 14. Coste 5.605 nodos x 3 = 16.815
unidades, contra 10.000 diarias. No cabe en un día.

Por eso el descargador está pensado para dejarlo toda la noche: lotes de 50
nodos (150 unidades, bajo el tope por minuto), y ante un 429 espera con
retroceso hasta 90 minutos en vez de rendirse. La cuota se renueva a
medianoche UTC y el proceso la coge solo. Es reanudable: si muere, se relanza
y sigue por donde iba.

El volumen en disco es irrelevante (5.605 nodos x 30 días son menos de 1 MB).
Lo que escasea es cuota, no espacio.

Uso:
    python comparar_rankings_justo.py --descargar    # dejarlo toda la noche
    python comparar_rankings_justo.py --evaluar
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr
import xgboost as xgb
from dotenv import load_dotenv
from pyproj import Transformer
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

import config
from comparar_julio2026 import auc, quemadas
from comparar_rankings import RANKINGS, RADIO_KM, firms_dia
from fwi_canadiense import calcular_fwi_serie
from malla_05_riesgo import ventanas
from puntuar_effis import GEOJSON
from riesgo_hoy import mensual

load_dotenv(f"{config.FUENTE}/.env")
IFS_INI, IFS_FIN = "2026-07-16", "2026-08-14"
L = 6
LOTE = 50                 # 50 nodos x 3 tramos = 150 unidades
PAUSA = 20
ESPERA_MAX = 5400         # 90 min: aguanta hasta que la cuota se renueve
CACHE = config.salida("ifs_historico.parquet")
VARS = ("temperature_2m_max,relative_humidity_2m_min,"
        "wind_speed_10m_max,precipitation_sum")


def descargar():
    n = np.load(config.NODOS)
    la, lo = n["nodo_lat"], n["nodo_lon"]
    filas, hechos = [], set()
    if os.path.exists(CACHE):
        prev = pd.read_parquet(CACHE)
        filas, hechos = [prev], set(prev["nodo"].unique())
        print(f"  ya en caché: {len(hechos):,} de {len(la):,} nodos", flush=True)
    pend = [i for i in range(len(la)) if i not in hechos]
    if not pend:
        print("  descarga completa", flush=True)
        return
    print(f"  pendientes {len(pend):,} nodos · "
          f"{len(pend) * 3:,} unidades · lotes de {LOTE}", flush=True)

    for b in range(0, len(pend), LOTE):
        idx = pend[b:b + LOTE]
        espera = PAUSA
        for intento in range(12):
            try:
                r = requests.get(
                    "https://historical-forecast-api.open-meteo.com/v1/forecast",
                    params=dict(
                        latitude=",".join(f"{la[i]:.4f}" for i in idx),
                        longitude=",".join(f"{lo[i]:.4f}" for i in idx),
                        start_date=IFS_INI, end_date=IFS_FIN, daily=VARS,
                        models="ecmwf_ifs025", timezone="UTC",
                        wind_speed_unit="ms"), timeout=240)
            except Exception as e:
                print(f"    red: {type(e).__name__} · reintento en {espera}s",
                      flush=True)
                time.sleep(espera)
                espera = min(espera * 2, ESPERA_MAX)
                continue
            if r.status_code != 429:
                break
            print(f"    429 (cuota) · esperando {espera // 60 or 1} min · "
                  f"{time.strftime('%H:%M:%S')}", flush=True)
            time.sleep(espera)
            espera = min(espera * 2, ESPERA_MAX)
        else:
            print("    no se pudo tras 12 intentos; se guarda y se sale",
                  flush=True)
            break
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}: {r.text[:120]}", flush=True)
            break
        j = r.json()
        for i, blo in zip(idx, j if isinstance(j, list) else [j]):
            d = blo["daily"]
            filas.append(pd.DataFrame({
                "nodo": i, "fecha": pd.to_datetime(d["time"]),
                "tmax": d["temperature_2m_max"],
                "hr_min": d["relative_humidity_2m_min"],
                "viento_max": d["wind_speed_10m_max"],
                "prec": d["precipitation_sum"]}))
        pd.concat(filas, ignore_index=True).to_parquet(CACHE, index=False)
        hechos.update(idx)
        print(f"    {len(hechos):,}/{len(la):,} nodos "
              f"({len(hechos) / len(la) * 100:.0f} %) · "
              f"{time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(PAUSA)


def evaluar():
    ifs = pd.read_parquet(CACHE)
    nodos_ok = np.sort(ifs["nodo"].unique())
    n = np.load(config.NODOS)
    la, lo, idx_nodo = n["nodo_lat"], n["nodo_lon"], n["idx_nodo"]
    if len(nodos_ok) < len(la):
        print(f"  ⚠️ solo {len(nodos_ok):,}/{len(la):,} nodos bajados: los que "
              f"falten se quedan sin rama de previsión (usan reanálisis)",
              flush=True)

    dsr = xr.open_dataset(config.entrada("era5land_diario.nc"))
    pts = dsr.sel(latitude=xr.DataArray(la, dims="n"),
                  longitude=xr.DataArray(lo, dims="n"), method="nearest")
    fechas = pd.to_datetime(dsr["fecha"].values).normalize()
    R = {v: pts[v].values for v in ("tmax", "tmin", "hr_min", "viento_max",
                                    "prec")}
    dsr.close()
    meses = fechas.month.values

    I = {v: np.full((len(fechas), len(la)), np.nan) for v in
         ("tmax", "hr_min", "viento_max", "prec")}
    piv = {v: ifs.pivot(index="fecha", columns="nodo", values=v)
           .reindex(fechas) for v in I}
    for v in I:
        I[v][:, nodos_ok] = piv[v][nodos_ok].values

    print(f"  FWI del reanálisis con estado ({len(la):,} nodos)...", flush=True)
    est = {k: np.full((len(fechas), len(la)), np.nan)
           for k in ("ffmc", "dmc", "dc", "fwi")}
    for j in range(len(la)):
        o = calcular_fwi_serie(R["tmax"][:, j], R["hr_min"][:, j],
                               R["viento_max"][:, j] * 3.6, R["prec"][:, j],
                               meses)
        for k in est:
            est[k][:, j] = o[k]

    mp = np.load(config.salida("mapeo_ifs_a_era5land.npz"))
    dias = sorted(pd.Timestamp(os.path.basename(p)[13:23])
                  for p in glob.glob(RANKINGS % "*"))
    dias = [d for d in dias if d in set(fechas)]

    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    es_esp = ds["is_spain"].values.astype(bool)
    xs, ys = ds["x"].values, ds["y"].values
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
    B["rayos_dia"] = np.zeros((ny, nx))
    B["rayos_7d"] = np.zeros((ny, nx))
    mens = {m: mensual(ds, m) for m in sorted({d.month for d in dias})}
    FULL = json.load(open(f"{config.MODELOS}/xgb_v2_prototipo_features.json"))
    modelo = xgb.XGBClassifier()
    modelo.load_model(f"{config.MODELOS}/xgb_v2_prototipo.ubj")
    # 21/08/2026: los modelos de etiqueta EFFIS sobre el mismo híbrido, para
    # que el cara a cara previsión-contra-previsión incluya a los candidatos
    import config_expansion as ce
    from dos_05_modelos import FEATS_CUANDO
    m_unico = xgb.XGBClassifier(); m_unico.load_model(f"{ce.MODELOS}/donde_dia_effis.ubj")
    m_cuando = xgb.XGBClassifier(); m_cuando.load_model(f"{ce.MODELOS}/cuando.ubj")
    md = np.load(config.entrada("dos_13_mapa_donde_effis_c.npz"))
    g_donde = np.full((ny, nx), np.nan); g_donde[md["iy"], md["ix"]] = md["p"]
    sel = es_esp.ravel()
    p_donde = g_donde.ravel()[sel]
    clim = np.load(config.CLIM)
    import holidays
    fest = holidays.Spain(years=[2026])
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    filas = []
    for dia in dias:
        k = int(np.where(fechas == dia)[0][0])
        if k - L - 1 < 0:
            continue
        quem, ha = quemadas(dia, ds, es_esp, ruta=GEOJSON)
        if quem.sum() == 0:
            continue
        cerca = maximum_filter(quem.astype(np.uint8), size=2 * RADIO_KM + 1,
                               mode="constant").astype(bool)

        # serie híbrida: reanálisis hasta k-L-1, IFS después
        H = {v: R[v][:k + 1].copy() for v in R}
        for v in ("tmax", "hr_min", "viento_max", "prec"):
            tramo = I[v][k - L:k + 1]
            H[v][k - L:k + 1] = np.where(np.isfinite(tramo), tramo,
                                         H[v][k - L:k + 1])
        fwi = est["fwi"][:k + 1].copy()
        for j in range(len(la)):
            if not np.isfinite(est["ffmc"][k - L - 1, j]):
                continue
            p = calcular_fwi_serie(
                H["tmax"][k - L:k + 1, j], H["hr_min"][k - L:k + 1, j],
                H["viento_max"][k - L:k + 1, j] * 3.6,
                H["prec"][k - L:k + 1, j], meses[k - L:k + 1],
                ffmc0=est["ffmc"][k - L - 1, j], dmc0=est["dmc"][k - L - 1, j],
                dc0=est["dc"][k - L - 1, j])["fwi"]
            fwi[k - L:k + 1, j] = np.where(
                np.isfinite(p), np.interp(p, mp["xs"], mp["ys"]), np.nan)

        F = ventanas(fwi, H["tmax"], H["tmin"], H["hr_min"],
                     H["viento_max"], H["prec"])
        C, nC = clim[f"m{dia.month}"], clim[f"n{dia.month}"].astype(float)
        with np.errstate(invalid="ignore"):
            cnt = np.nansum(C <= F["fwi"][None, :], axis=0)
            F["fwi_pctl_local"] = np.where(nC > 0, cnt / np.maximum(nC, 1) * 100,
                                           np.nan)
            cm, cs = np.nanmean(C, axis=0), np.nanstd(C, axis=0)
            F["fwi_anom_sigma"] = np.where((nC > 0) & (cs > 0),
                                           (F["fwi"] - cm) / cs, np.nan)
        malo = ~np.isfinite(F["fwi"]) | (nC == 0)
        if malo.any():
            bu = np.where(~malo)[0]
            ar = cKDTree(np.column_stack([la[bu], lo[bu]]))
            _, kk = ar.query(np.column_stack([la[malo], lo[malo]]))
            for v in F:
                F[v][malo] = F[v][bu[kk]]

        seg = np.clip(idx_nodo, 0, None)
        M = dict(B)
        M.update(mens[dia.month])
        M["ndvi_med_30d"] = M["ndvi"]
        M["frp_max_50km_7d"], M["n_detec_50km_7d"] = firms_dia(dia, ds)
        for v, x in F.items():
            c = np.asarray(x, float)[seg]
            c[idx_nodo < 0] = np.nan
            M[v] = c
        M["es_festivo"] = np.full((ny, nx), float(dia.weekday() >= 5
                                                  or dia.date() in fest))
        M["mes"] = np.full((ny, nx), dia.month)
        M["dia_anio"] = np.full((ny, nx), dia.dayofyear)

        X = pd.DataFrame({c: M[c].ravel()[sel] for c in sorted(set(FULL) | set(FEATS_CUANDO))})
        pr = np.full(ny * nx, np.nan)
        pr[sel] = modelo.predict_proba(X[FULL].values)[:, 1]
        pr = pr.reshape(ny, nx)
        mapas = {"malla_hibrida": pr}
        pu = np.full(ny * nx, np.nan); pu[sel] = m_unico.predict_proba(X[FULL].values)[:, 1]
        mapas["unico"] = pu.reshape(ny, nx)
        pq = np.full(ny * nx, np.nan); pq[sel] = p_donde * m_cuando.predict_proba(X[FEATS_CUANDO].values)[:, 1]
        mapas["pareja"] = pq.reshape(ny, nx)

        rk = pd.read_csv(RANKINGS % f"{dia:%Y-%m-%d}")
        ex, ey = tr.transform(rk["lon"].values, rk["lat"].values)
        ix = np.clip(np.rint((ex - xs[0]) / (xs[1] - xs[0])).astype(int), 0, nx - 1)
        iy = np.clip(np.rint((ey - ys[0]) / (ys[1] - ys[0])).astype(int), 0, ny - 1)
        y, pm, pp = cerca[iy, ix], pr[iy, ix], rk["prob"].values
        ok = np.isfinite(pm) & np.isfinite(pp)
        if y[ok].sum() < 3 or (~y[ok]).sum() < 3:
            continue
        fila = {"fecha": f"{dia:%F}", "positivas": int(y[ok].sum()),
                "estaciones": int(ok.sum()), "area_ha": round(ha, 1),
                "celdas_quemadas": int(quem.sum()),
                "auc_ranking": round(auc(pp[ok][y[ok]], pp[ok][~y[ok]]), 4),
                "auc_malla_hibrida": round(auc(pm[ok][y[ok]], pm[ok][~y[ok]]), 4)}
        for nom, g_ in mapas.items():
            # por estación (contra el ranking sellado) y por celda (contra EFFIS)
            pe = g_[iy, ix]
            fila[f"auc_{nom}"] = round(auc(pe[ok][y[ok]], pe[ok][~y[ok]]), 4)
            v = g_[es_esp]; okv = np.isfinite(v); qv = quem[es_esp]
            fila[f"auc_cel_{nom}"] = round(auc(v[okv & qv], v[okv & ~qv]), 4)
        filas.append(fila)
        print(f"  {dia:%F}  positivas {fila['positivas']:3d} · AUC ranking "
              f"{fila['auc_ranking']:.3f} · malla {fila['auc_malla_hibrida']:.3f} · "
              f"único {fila['auc_unico']:.3f} · pareja {fila['auc_pareja']:.3f} · "
              f"celda: malla {fila['auc_cel_malla_hibrida']:.3f} único {fila['auc_cel_unico']:.3f}",
              flush=True)
    ds.close()

    d = pd.DataFrame(filas)
    d.to_csv(config.salida("rankings_justo.csv"), index=False)
    dif = (d["auc_malla_hibrida"] - d["auc_ranking"]).values
    rng = np.random.default_rng(0)
    bs = np.array([dif[rng.integers(0, len(dif), len(dif))].mean()
                   for _ in range(20000)])
    lo_, hi = np.percentile(bs, [2.5, 97.5])
    print("\n" + "=" * 70)
    print(f"CARA A CARA JUSTO — previsión contra previsión, {len(d)} días")
    print("=" * 70)
    print(f"  ranking sellado (AEMET + previsión municipal): "
          f"{d['auc_ranking'].mean():.3f}")
    print(f"  malla híbrida (reanálisis + IFS mapeado)     : "
          f"{d['auc_malla_hibrida'].mean():.3f}")
    print(f"\n  diferencia {dif.mean():+.3f} · IC95 [{lo_:+.3f}, {hi:+.3f}] · "
          f"gana {int((dif > 0).sum())}/{len(dif)} días")
    ver = ("la malla gana" if lo_ > 0 else
           ("la malla pierde" if hi < 0 else
            f"con {len(d)} días no se distingue del ruido"))
    print(f"  → {ver}")
    extra = {}
    for nom in ("unico", "pareja"):
        dd = (d[f"auc_{nom}"] - d["auc_ranking"]).values
        b = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(20000)])
        extra[nom] = {"auc_estacion": float(d[f"auc_{nom}"].mean()),
                      "dif_vs_ranking": float(dd.mean()),
                      "ic95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                      "gana": int((dd > 0).sum())}
        dc = (d[f"auc_cel_{nom}"] - d["auc_cel_malla_hibrida"]).values
        b = np.array([dc[rng.integers(0, len(dc), len(dc))].mean() for _ in range(20000)])
        extra[nom].update({"auc_celda": float(d[f"auc_cel_{nom}"].mean()),
                           "auc_celda_malla": float(d["auc_cel_malla_hibrida"].mean()),
                           "dif_celda_vs_malla": float(dc.mean()),
                           "ic95_celda": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]})
        print(f"  {nom:7s}: por estación {extra[nom]['auc_estacion']:.3f} (Δ ranking "
              f"{extra[nom]['dif_vs_ranking']:+.3f} [{extra[nom]['ic95'][0]:+.3f}, {extra[nom]['ic95'][1]:+.3f}], "
              f"gana {extra[nom]['gana']}/{len(d)}) · por celda {extra[nom]['auc_celda']:.3f} "
              f"vs malla {extra[nom]['auc_celda_malla']:.3f} (Δ {dc.mean():+.3f} "
              f"[{extra[nom]['ic95_celda'][0]:+.3f}, {extra[nom]['ic95_celda'][1]:+.3f}])")
    json.dump({"dias": filas, "auc_ranking": float(d["auc_ranking"].mean()),
               "candidatos_effis": extra,
               "auc_malla_hibrida": float(d["auc_malla_hibrida"].mean()),
               "dif": float(dif.mean()), "ic95": [float(lo_), float(hi)],
               "veredicto": ver},
              open(config.salida("rankings_justo.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--descargar", action="store_true")
    p.add_argument("--evaluar", action="store_true")
    a = p.parse_args()
    if a.descargar:
        descargar()
    if a.evaluar or not a.descargar:
        evaluar()
