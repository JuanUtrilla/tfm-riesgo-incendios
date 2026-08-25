#!/usr/bin/env python3
"""
Dos modelos — paso 9: la temporada 2026, día a día, contra el área quemada
de EFFIS. Producción vs dónde×cuándo vs único-dónde, MISMAS features.

NO TOCA PRODUCCIÓN. Lee salida/era5land_diario.nc (reanálisis may→ago 2026),
el cubo, EFFIS y la caché FIRMS; escribe salida/dos_09_temporada2026.{json,csv,png}.

=============================================================================
DISEÑO
=============================================================================
Lo de `dos_05` es retrospectivo (2019-2022). Esto es la temporada en curso.
Para que la comparación sea DEL MODELO y no de la meteo, los tres se puntúan
sobre el mismo vector de 46 features por celda, construido como en
`riesgo_hoy.py` pero solo con reanálisis (el IFS archivado no está; con
reanálisis los tres tienen el mismo retrovisor, así que el cara a cara es
justo entre ellos aunque sea una cota superior de lo que daría la operación).

  prod          xgb_v2_prototipo (46 features)                  ← lo publicado
  donde×cuando  P_donde_hist(celda) × P_cuando(29 dinámicas)    ← recomendado
  donde_dia     modelo único, 46 features, muestreo dónde       ← el alternativo
  donde         el mapa estático solo (cota: sin meteo)
  fwi_pctl      percentil del FWI (línea base sin modelo)

Juez: celdas con perímetro EFFIS cuya FIREDATE es ese día (función `quemadas`
de `comparar_julio2026`, que ya usa el cron). Métricas por día: AUC de celdas
quemadas contra TODAS las demás de España (es la métrica de
`puntuar_effis.py`), percentil mediano del fuego y lift del 10 % superior.
IC95 por bootstrap de DÍAS. Como referencia se añaden, los 4 días que existen,
los mapas SELLADOS de producción (AEMET+IDW, previsión): eso sí es lo que se
publicó, con su meteo y todo.

FIRMS [D−5, D−1]: caché `salida/_firms/` (22-jul → 11-ago) y API para el
resto; si el archivo NRT no llega a junio, esos días van a cero PARA LOS
TRES modelos, y se deja contado en el JSON.
"""

import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

import config
import config_expansion as ce
import riesgo_hoy as rh
from comparar_julio2026 import auc, quemadas
from comparar_rankings import firms_dia
from dos_05_modelos import FEATS_CUANDO
from fwi_canadiense import calcular_fwi_serie

INI, FIN = "2026-06-01", None          # FIN = último día de reanálisis
GEOJSON = config.salida("effis_ba_season_ES.geojson")


def reanalisis():
    n = np.load(config.NODOS)
    la, lo, idx = n["nodo_lat"], n["nodo_lon"], n["idx_nodo"]
    ds = xr.open_dataset(rh.REANALISIS)
    pts = ds.sel(latitude=xr.DataArray(la, dims="n"),
                 longitude=xr.DataArray(lo, dims="n"), method="nearest")
    fechas = pd.to_datetime(ds["fecha"].values).normalize()
    S = {v: pts[v].values for v in ("tmax", "tmin", "hr_min", "viento_max", "prec")}
    ds.close()
    meses = fechas.month.values
    fwi = np.full(S["tmax"].shape, np.nan)
    for j in range(len(la)):
        fwi[:, j] = calcular_fwi_serie(S["tmax"][:, j], S["hr_min"][:, j],
                                       S["viento_max"][:, j] * 3.6, S["prec"][:, j],
                                       meses)["fwi"]
    return S, fwi, fechas, idx, la, lo


def main():
    FULL = json.load(open(rh.FEATS_JSON))
    S, fwi, fechas, idx_nodo, la, lo = reanalisis()
    fin = pd.Timestamp(FIN) if FIN else fechas[-1]
    dias = pd.date_range(INI, fin, freq="D")
    print(f"reanálisis {fechas[0].date()} → {fechas[-1].date()} · "
          f"evaluando {dias[0].date()} → {dias[-1].date()}", flush=True)

    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    es_esp = ds["is_spain"].values.astype(bool)
    sel = es_esp.ravel()
    B = {}
    for k, v in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                 "rugosidad": "roughness_mean", "dist_carreteras": "dist_to_roads_mean",
                 "dist_rios": "dist_to_waterways_mean"}.items():
        B[k] = ds[v].values
    B["popdens"] = ds["popdens_2020"].values
    for k, suf in {"clc_bosque": "forest_proportion", "clc_matorral": "scrub_proportion",
                   "clc_agricola": "agricultural_proportion",
                   "clc_artificial": "artificial_proportion",
                   "clc_abierto": "open_space_proportion",
                   "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
        B[k] = ds[f"CLC_2018_{suf}"].values
    B["ccaa"] = np.nan_to_num(ds["AutonomousCommunities"].values, nan=-1).astype(int)
    B["rayos_dia"] = np.zeros((ny, nx)); B["rayos_7d"] = np.zeros((ny, nx))

    prod = xgb.XGBClassifier(); prod.load_model(rh.MODELO)
    cuando = xgb.XGBClassifier(); cuando.load_model(f"{ce.MODELOS}/cuando.ubj")
    ddia = xgb.XGBClassifier(); ddia.load_model(f"{ce.MODELOS}/donde_dia.ubj")
    md = np.load(config.salida("dos_05_mapa_donde.npz"))
    donde = np.full((ny, nx), np.nan); donde[md["iy"], md["ix"]] = md["p_donde_hist"]
    p_donde = donde.ravel()[sel]
    # modelos entrenados con etiqueta EFFIS (dos_13), si existen
    effis_models = {}
    for nom in ("cuando_effis", "donde_dia_effis"):
        f = f"{ce.MODELOS}/{nom}.ubj"
        if os.path.exists(f):
            m = xgb.XGBClassifier(); m.load_model(f); effis_models[nom] = m
    # escalera de negativos por positivo (dos_18), los que estén entrenados
    import glob as _glob
    rmods = {}
    for f in sorted(_glob.glob(f"{ce.MODELOS}/donde_dia_effis_r*.ubj"),
                    key=lambda x: int(x.split("_r")[-1][:-4])):
        m = xgb.XGBClassifier(); m.load_model(f)
        rmods[os.path.basename(f)[:-4]] = m
    if rmods:
        print("  escalera dos_18:", ", ".join(rmods), flush=True)
    f13 = config.salida("dos_13_mapa_donde_effis_c.npz")
    p_donde_effis_c = None
    if os.path.exists(f13):
        me = np.load(f13); g = np.full((ny, nx), np.nan); g[me["iy"], me["ix"]] = me["p"]
        p_donde_effis_c = g.ravel()[sel]
    # variante con etiqueta EFFIS (dos_10), si existe
    extra = {}
    rf = config.salida("dos_10_mapa_donde_effis.npz")
    if os.path.exists(rf):
        me = np.load(rf)
        for k in ("donde_effis", "donde_effis_mix"):
            g = np.full((ny, nx), np.nan); g[me["iy"], me["ix"]] = me[k]
            extra[k] = g.ravel()[sel]
    import holidays
    fest = holidays.Spain(years=[2026])

    filas, firms_vacio = [], []
    for d in dias:
        quem, ha = quemadas(d, ds, es_esp, ruta=GEOJSON)
        nq = int(quem.sum())
        if nq == 0:
            continue
        k = int(np.where(fechas == d)[0][0])
        F, _ = rh.meteo_dia(S, fwi, fechas, k, d.month, la, lo)
        M = rh.a_celdas(F, idx_nodo)
        G = dict(B); G.update(rh.mensual(ds, d.month))
        G["ndvi_med_30d"] = G["ndvi"]
        G["frp_max_50km_7d"], G["n_detec_50km_7d"] = firms_dia(d, ds)
        if G["n_detec_50km_7d"].max() == 0:
            firms_vacio.append(str(d.date()))
        G["es_festivo"] = np.full((ny, nx), float(d.weekday() >= 5 or d.date() in fest))
        G["mes"] = np.full((ny, nx), d.month); G["dia_anio"] = np.full((ny, nx), d.dayofyear)
        X = pd.DataFrame({c: (M[c] if c in M else G[c]).ravel()[sel]
                          for c in sorted(set(FULL) | set(FEATS_CUANDO))})
        mapas = {"prod": prod.predict_proba(X[FULL].values)[:, 1]}
        pc = cuando.predict_proba(X[FEATS_CUANDO].values)[:, 1]
        mapas["donde×cuando"] = p_donde * pc
        mapas["donde_dia"] = ddia.predict_proba(X[FULL].values)[:, 1]
        mapas["donde"] = p_donde
        # ablación: ¿cuánto de la ventaja es FIRMS [D-5,D-1] encendido alrededor
        # de fuegos que siguen ardiendo? Se ponen a cero las dos features.
        X0 = X.copy(); X0["frp_max_50km_7d"] = 0.0; X0["n_detec_50km_7d"] = 0
        mapas["prod_sin_firms"] = prod.predict_proba(X0[FULL].values)[:, 1]
        mapas["donde_dia_sin_firms"] = ddia.predict_proba(X0[FULL].values)[:, 1]
        mapas["d×c_sin_firms"] = p_donde * cuando.predict_proba(X0[FEATS_CUANDO].values)[:, 1]
        mapas["fwi_pctl"] = X["fwi_pctl_local"].values
        for k, v in extra.items():
            mapas[k] = v
            mapas[f"{k}×cuando"] = v * pc
        if "cuando_effis" in effis_models:
            pce = effis_models["cuando_effis"].predict_proba(X[FEATS_CUANDO].values)[:, 1]
            mapas["cuando_effis"] = pce
            if p_donde_effis_c is not None:
                mapas["donde_effis_c"] = p_donde_effis_c
                mapas["donde_effis_c×cuando_effis"] = p_donde_effis_c * pce
                mapas["donde_effis_c×cuando_egif"] = p_donde_effis_c * pc
                X0e = X.copy(); X0e["frp_max_50km_7d"] = 0.0; X0e["n_detec_50km_7d"] = 0
                mapas["donde_effis_c×cuando_effis_sin_firms"] = p_donde_effis_c * \
                    effis_models["cuando_effis"].predict_proba(X0e[FEATS_CUANDO].values)[:, 1]
        if "donde_dia_effis" in effis_models:
            mapas["donde_dia_effis"] = effis_models["donde_dia_effis"].predict_proba(X[FULL].values)[:, 1]
            mapas["donde_dia_effis_sin_firms"] = effis_models["donde_dia_effis"].predict_proba(X0[FULL].values)[:, 1]
        for nom, m in rmods.items():
            mapas[nom] = m.predict_proba(X[FULL].values)[:, 1]
        sp = f"{config.FUENTE}/prototipo/cache/malla_prob_{d.date()}.npz"
        if os.path.exists(sp):
            mapas["prod_sellado"] = np.load(sp)["prob"].astype(float).ravel()[sel]
        os.makedirs(f"{ce.DATASET}/mapas_2026", exist_ok=True)
        np.savez_compressed(f"{ce.DATASET}/mapas_2026/{d.date()}.npz",
                            **{k: v.astype(np.float32) for k, v in mapas.items()})
        q = quem.ravel()[sel]
        fila = {"fecha": str(d.date()), "celdas_quemadas": nq, "area_ha": round(ha, 1)}
        for nom, A in mapas.items():
            ok = np.isfinite(A)
            pos, neg = A[ok & q], A[ok & ~q]
            if not len(pos):
                continue
            orden = np.sort(A[ok])
            pct = np.searchsorted(orden, pos, side="right") / len(orden) * 100
            umbral = np.quantile(A[ok], 0.9)
            fila[f"auc_{nom}"] = round(auc(pos, neg), 4)
            fila[f"pctl_{nom}"] = round(float(np.median(pct)), 2)
            fila[f"lift_{nom}"] = round(float((pos >= umbral).mean() / 0.1), 3)
        filas.append(fila)
        fila["pctl_donde_top"] = round(float(np.mean(p_donde[q & np.isfinite(p_donde)] >= np.quantile(p_donde[np.isfinite(p_donde)], .9)))*100, 1)
        print(f"  {d.date()} quem {nq:5d} ({ha:7.0f} ha) · AUC prod {fila['auc_prod']:.3f} "
              f"d×c {fila['auc_donde×cuando']:.3f} donde_dia {fila['auc_donde_dia']:.3f} "
              f"donde {fila['auc_donde']:.3f} fwi_pctl {fila['auc_fwi_pctl']:.3f}", flush=True)
    ds.close()

    df = pd.DataFrame(filas)
    df.to_csv(config.salida("dos_09_temporada2026.csv"), index=False)
    rng = np.random.default_rng(0)
    R = {"n_dias": len(df), "celdas_quemadas": int(df.celdas_quemadas.sum()),
         "area_ha": float(df.area_ha.sum()), "dias_firms_vacio": firms_vacio,
         "periodo": [df.fecha.min(), df.fecha.max()]}
    for nom in [c[4:] for c in df.columns if c.startswith("auc_") and c != "auc_prod_sellado"]:
        a, p, l = df[f"auc_{nom}"].values, df[f"pctl_{nom}"].values, df[f"lift_{nom}"].values
        w = df.area_ha.values
        d_auc = a - df["auc_prod"].values
        idx = rng.integers(0, len(df), (2000, len(df)))
        b = d_auc[idx].mean(1)
        R[nom] = {"auc_medio": float(a.mean()), "auc_mediana": float(np.median(a)),
                  "pctl_medio": float(p.mean()), "pctl_ponderado_ha": float((p * w).sum() / w.sum()),
                  "lift_medio": float(l.mean()),
                  "dif_auc_vs_prod": float(d_auc.mean()),
                  "ic95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                  "dias_gana": float((d_auc > 0).mean())}
    if "auc_prod_sellado" in df:
        s = df.dropna(subset=["auc_prod_sellado"])
        R["prod_sellado"] = {"n_dias": len(s), "fechas": s.fecha.tolist(),
                             "auc": s["auc_prod_sellado"].tolist(),
                             "auc_prod_mismas_features": s["auc_prod"].tolist(),
                             "auc_donde×cuando": s["auc_donde×cuando"].tolist()}
    json.dump(R, open(config.salida("dos_09_temporada2026.json"), "w"), indent=1)
    print("\n== temporada 2026 · AUC dentro del día (celdas quemadas vs resto) ==")
    for nom in [c[4:] for c in df.columns if c.startswith("auc_") and c != "auc_prod_sellado"]:
        r = R[nom]
        print(f"  {nom:14s} AUC {r['auc_medio']:.3f} · pctl {r['pctl_medio']:.1f} "
              f"(pond. ha {r['pctl_ponderado_ha']:.1f}) · lift {r['lift_medio']:.2f} · "
              f"Δprod {r['dif_auc_vs_prod']:+.3f} [{r['ic95'][0]:+.3f}, {r['ic95'][1]:+.3f}] "
              f"· gana {r['dias_gana']*100:.0f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    x = pd.to_datetime(df.fecha)
    for nom, c in [("prod", "gray"), ("donde×cuando", "crimson"), ("donde_dia", "darkorange")]:
        ax.plot(x, df[f"auc_{nom}"], "o-", ms=4, lw=1.2, color=c, label=nom)
    if "auc_prod_sellado" in df:
        ax.plot(x, df["auc_prod_sellado"], "k*", ms=11, label="producción sellada (AEMET, previsión)")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_ylabel("AUC dentro del día (EFFIS)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Temporada 2026 · {len(df)} días con fuego EFFIS · mismas features de reanálisis")
    fig.tight_layout(); fig.savefig(config.salida("dos_09_temporada2026.png"), dpi=130)


if __name__ == "__main__":
    main()
