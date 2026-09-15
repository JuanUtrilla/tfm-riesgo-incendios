#!/usr/bin/env python3
"""
Mapa de hoy y mañana con los modelos de etiqueta EFFIS sobre la malla
ERA5-Land. Candidato a producción; corre en paralelo con `riesgo_hoy.py`.

No toca producción. Escribe salida/dos_riesgo_<fecha>.{npz,png,json}.

Qué sirve
---------
Misma cadena que `riesgo_hoy.py` (reanálisis ERA5-Land hasta D−7, IFS D−6→D+1
con el FWI mapeado a la escala de ERA5-Land, 46 features por celda) y dos
mapas nuevos en vez del modelo de producción:

  unico    `donde_dia_effis`: un XGBoost con las 46 features, muestreo del
           mismo día (negativos en otras celdas) y etiqueta EFFIS (`is_fire`
           del cubo = celda quemada ≥5 ha). Train y servicio comparten malla
           (ERA5-Land → 1 km), features y target. Medido durante el
           seguimiento de la temporada 2026: AUC 0,773 contra 0,737 de
           producción; 0,744 sin FIRMS.
  pareja   `donde_effis_c` (susceptibilidad por celda, etiqueta EFFIS) ×
           `cuando` (solo dinámicas, etiqueta EGIF: el cuándo con EFFIS salió
           peor). 0,754 en 2026. Interpretable:
           el mapa estático va aparte.
  r10      `donde_dia_effis_r10`: el mismo modelo que `unico` cambiando solo
           cuántos negativos del mismo día ve en entrenamiento, 10 por
           positivo en vez de 3 (`dos_18_ratio.py`, escalera anidada: los
           negativos del 1:3 son un subconjunto de los del 1:10, así que la
           diferencia es cuántos y no cuáles). Añadido el 31/08/2026 como
           candidato adicional: en AUC medio empata con el 1:3 (0,759 frente
           a 0,752) pero mete más fuego grande en la punta del día (34 % de
           los incendios de ≥500 ha en el top-2 % contra 29 %) y sube el
           percentil ponderado por hectáreas (81,9 frente a 79,8). No
           sustituye al 1:3: este sigue sirviéndose y su serie diaria no se
           rompe.

Los dos se publican como percentil del día sobre las 498.530 celdas y con
niveles por percentil (p30/p90/p98, `dos_14_cortes.py`): la probabilidad
cruda de estos modelos no está calibrada a la prevalencia real y el
producto multiplica dos escalas. La "alerta global" del día (FWI medio
contra climatología) se guarda como número aparte en el JSON.

Guarda también, si existe, el mapa de la malla con el modelo de producción
(`riesgo_hoy_<fecha>.npz`) para pintar los tres juntos.

Uso:
    python dos_riesgo_hoy.py                 # hoy y mañana (IFS de hoy)
    python dos_riesgo_hoy.py --pasada 2026-08-20 --dias 0 1
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from scipy.stats import spearmanr

import capa_base
import capa_verdad
import config
import config_expansion as ce
import malla_02b_ifs as ifsmod
import riesgo_hoy as rh
from dos_05_modelos import FEATS_CUANDO

CORTES_PCTL = [30, 90, 98]
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]
# Escala discreta en vez del degradado continuo. Los cortes operativos
# (p30/p90/p98, `dos_14_cortes.py`) son bordes duros del color, así que ALTO y
# EXTREMO se ven como regiones y no como «un rojo algo más oscuro». BAJO y
# MODERADO se parten en dos tonos cada uno para no perder la textura del
# mapa, que es lo que se compara entre paneles.
CORTES_MAPA = [0, 10, 30, 60, 90, 98, 100]
COLORES_MAPA = ["#ffffd9", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#bd0026"]
MODELO_UNICO = f"{ce.MODELOS}/donde_dia_effis.ubj"
MODELO_R10 = f"{ce.MODELOS}/donde_dia_effis_r10.ubj"
MODELO_CUANDO = f"{ce.MODELOS}/cuando.ubj"
MAPA_DONDE = config.entrada("dos_13_mapa_donde_effis_c.npz")


def percentil_dia(v):
    p = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    p[ok] = pd.Series(v[ok]).rank(pct=True).values * 100
    return p


def main(a):
    if a.pasada:
        base = pd.Timestamp(a.pasada)
        ifsmod.descarga = lambda *k, **kw: pd.read_parquet(ifsmod.ruta_cache(a.pasada))
    else:
        base = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    objetivos = [base + pd.Timedelta(days=h) for h in a.dias]
    FULL = json.load(open(rh.FEATS_JSON))
    print(f"Mapas EFFIS · {[str(o.date()) for o in objetivos]}", flush=True)

    S, fwi, fechas, idx_nodo, la, lo, nre = rh.series_nodos(objetivos)
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    es_esp = ds["is_spain"].values.astype(bool)
    ys = ds["y"].values
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
    B["frp_max_50km_7d"], B["n_detec_50km_7d"] = rh.firms_nrt(ds)
    B["rayos_dia"] = np.zeros((ny, nx)); B["rayos_7d"] = np.zeros((ny, nx))

    unico = xgb.XGBClassifier(); unico.load_model(MODELO_UNICO)
    r10 = xgb.XGBClassifier(); r10.load_model(MODELO_R10)
    cuando = xgb.XGBClassifier(); cuando.load_model(MODELO_CUANDO)
    md = np.load(MAPA_DONDE)
    donde = np.full((ny, nx), np.nan); donde[md["iy"], md["ix"]] = md["p"]
    p_donde = donde.ravel()[sel]
    clim = np.load(config.CLIM)
    import holidays
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for obj in objetivos:
        fstr = str(obj.date())
        k = int(np.where(fechas == obj)[0][0])
        print(f"\n=== {fstr} ===", flush=True)
        F, n_malo = rh.meteo_dia(S, fwi, fechas, k, obj.month, la, lo)
        M = rh.a_celdas(F, idx_nodo)
        G = dict(B); G.update(rh.mensual(ds, obj.month))
        G["ndvi_med_30d"] = G["ndvi"]
        fest = holidays.Spain(years=[obj.year])
        G["es_festivo"] = np.full((ny, nx), float(obj.weekday() >= 5 or obj.date() in fest))
        G["mes"] = np.full((ny, nx), obj.month)
        G["dia_anio"] = np.full((ny, nx), obj.dayofyear)
        X = pd.DataFrame({c: (M[c] if c in M else G[c]).ravel()[sel]
                          for c in sorted(set(FULL) | set(FEATS_CUANDO))})
        p_unico = unico.predict_proba(X[FULL].values)[:, 1]
        p_r10 = r10.predict_proba(X[FULL].values)[:, 1]
        p_cuando = cuando.predict_proba(X[FEATS_CUANDO].values)[:, 1]
        p_pareja = p_donde * p_cuando

        out, res = {}, {"fecha": fstr, "rama": "reanálisis + IFS mapeado" if k >= nre
                        else "solo reanálisis", "nodos_respaldo": int(n_malo)}
        for nom, p in (("unico", p_unico), ("r10", p_r10),
                       ("pareja", p_pareja),
                       ("donde", p_donde), ("cuando", p_cuando)):
            g = np.full(ny * nx, np.nan, np.float32); g[sel] = p
            out[f"prob_{nom}"] = g.reshape(ny, nx)
            pc = percentil_dia(p)
            g = np.full(ny * nx, np.nan, np.float32); g[sel] = pc
            out[f"pctl_{nom}"] = g.reshape(ny, nx)
            if nom in ("unico", "r10", "pareja"):
                lv = np.searchsorted(CORTES_PCTL, pc, side="right")
                res[f"niveles_{nom}"] = {NIVELES[i]: float((lv == i).mean() * 100)
                                        for i in range(4)}
        # alerta global: FWI medio del día contra la climatología del mes
        fw = M["fwi"][es_esp]; fw = fw[np.isfinite(fw)]
        C = clim[f"m{obj.month}"]; C = C[np.isfinite(C)]
        res["fwi_medio"] = float(fw.mean())
        res["fwi_medio_pctl_clim"] = float((C <= fw.mean()).mean() * 100)
        res["spearman_unico_pareja"] = float(spearmanr(p_unico, p_pareja).correlation)
        res["spearman_unico_r10"] = float(spearmanr(p_unico, p_r10).correlation)
        rp = config.salida(f"riesgo_hoy_{fstr}.npz")
        if os.path.exists(rp):
            pr = np.load(rp)["prob"].astype(float).ravel()[sel]
            res["spearman_unico_prod_malla"] = float(spearmanr(p_unico, pr, nan_policy="omit").correlation)
            g = np.full(ny * nx, np.nan, np.float32); g[sel] = percentil_dia(pr)
            out["pctl_prod_malla"] = g.reshape(ny, nx)
        # los mapas de cada día van a salida/mapas_diarios/ (lo que persiste
        # entre corridas en GitHub): probabilidades en float16, ~3 MB/día.
        # El percentil se recalcula de la probabilidad cuando hace falta.
        md_dir = config.salida("mapas_diarios")
        os.makedirs(md_dir, exist_ok=True)
        np.savez_compressed(f"{md_dir}/dos_riesgo_{fstr}.npz",
                            **{k: v.astype(np.float16) for k, v in out.items()
                               if k.startswith("prob_")})
        if os.path.exists(rp):
            np.savez_compressed(f"{md_dir}/riesgo_hoy_{fstr}.npz",
                                prob=np.load(rp)["prob"].astype(np.float16))
        json.dump(res, open(config.salida(f"dos_riesgo_{fstr}.json"), "w"), indent=1)
        print(f"  FWI medio {res['fwi_medio']:.1f} (pctl clim {res['fwi_medio_pctl_clim']:.0f}) · "
              f"unico~pareja {res['spearman_unico_pareja']:.2f} · "
              f"unico~r10 {res['spearman_unico_r10']:.2f}"
              + (f" · unico~prod {res['spearman_unico_prod_malla']:.2f}"
                 if "spearman_unico_prod_malla" in res else ""), flush=True)

        # Paneles de diferencia contra producción. Cuatro manchas rojas casi
        # iguales no se comparan a ojo, porque el ojo no resta mapas. Lo que
        # hay que ver es dónde discrepa el candidato, y eso es una resta de
        # percentiles con paleta divergente centrada en cero.
        if "pctl_prod_malla" in out:
            for nom in ("unico", "r10", "pareja"):
                if f"pctl_{nom}" in out:
                    out[f"dif_{nom}"] = out[f"pctl_{nom}"] - out["pctl_prod_malla"]
        paneles = [("pctl_prod_malla", "PRODUCCIÓN (xgb_v2 sobre la malla)"),
                   ("pctl_unico", "ÚNICO 1:3 · etiqueta EFFIS · mismo día"),
                   ("pctl_r10", "ÚNICO 1:10 · mismo modelo, más negativos"),
                   ("pctl_pareja", "DÓNDE × CUÁNDO"),
                   ("pctl_donde", "DÓNDE · susceptibilidad EFFIS (estático)")]
        paneles = [p for p in paneles if p[0] in out]
        difs = [("dif_unico", "ÚNICO 1:3 − PRODUCCIÓN"),
                ("dif_r10", "ÚNICO 1:10 − PRODUCCIÓN"),
                ("dif_pareja", "PAREJA − PRODUCCIÓN")]
        difs = [p for p in difs if p[0] in out]
        from matplotlib.colors import BoundaryNorm, ListedColormap
        cmap_niv = ListedColormap(COLORES_MAPA)
        norm_niv = BoundaryNorm(CORTES_MAPA, cmap_niv.N)
        capas = capa_verdad.preparar(ds, fstr)
        # con el r10 son 5 paneles de nivel + 3 restas = 8 = 2x4 exacto
        nc = 4 if difs else 2
        fig, axs = plt.subplots(2, nc, figsize=(8 * nc, 12))
        rejilla = list(axs.ravel())
        # fila de arriba los tres mapas, fila de abajo el estático y las restas
        ejes_niv = rejilla[:len(paneles)]
        ejes_dif = rejilla[len(paneles):len(paneles) + len(difs)]
        for ax, (key, tit) in zip(ejes_niv, paneles):
            # interpolation="nearest" y dpi 165 (el panel mide 1.188 px, los
            # mismos que la malla): con el "antialiased" por defecto y dpi 130
            # el render remuestrea y funde las celdas EXTREMO sueltas con sus
            # vecinas. Medido el 23/08 sobre el mapa del 22: 8.816 píxeles en
            # color EXTREMO con el ajuste viejo contra 22.620 con este, y el
            # 74 % de las manchas EXTREMO son de 1-2 celdas, o sea que lo que
            # se borraba era la mayoría de los avisos distintos. El PNG pesa
            # menos, porque los colores planos comprimen mejor.
            im = ax.imshow(out[key], origin="lower" if ys[1] > ys[0] else "upper",
                           cmap=cmap_niv, norm=norm_niv, interpolation="nearest")
            capa_base.dibujar(ax, out[key].shape[1], out[key].shape[0],
                              etiquetas=12, lw=0.9)
            capa_verdad.dibujar(ax, capas, lw=0.9, numerar=6)
            ax.set_title(tit, fontsize=11); ax.set_axis_off()
        for ax, (key, tit) in zip(ejes_dif, difs):
            imd = ax.imshow(out[key], origin="lower" if ys[1] > ys[0] else "upper",
                            cmap="RdBu_r", vmin=-40, vmax=40,
                            interpolation="nearest")
            capa_base.dibujar(ax, out[key].shape[1], out[key].shape[0],
                              etiquetas=0, lw=0.9)
            capa_verdad.dibujar(ax, capas, lw=0.9, numerar=6)
            ax.set_title(tit, fontsize=11); ax.set_axis_off()
        for ax in rejilla[len(paneles) + len(difs):]:
            ax.set_axis_off()
        # la barra de niveles cuelga solo de la fila de arriba: si se le pasan
        # también los ejes de abajo, matplotlib le roba sitio a la unión de
        # ambos y la barra se planta encima del panel de diferencia
        cb = fig.colorbar(im, ax=ejes_niv[:nc], shrink=0.85,
                          boundaries=CORTES_MAPA,
                          ticks=CORTES_MAPA, spacing="uniform")
        cb.set_label("percentil del día")
        capa_base.niveles_en_barra(cb, cortes=CORTES_PCTL, nombres=NIVELES)
        if ejes_dif:
            cbd = fig.colorbar(imd, ax=ejes_dif, orientation="horizontal",
                               shrink=0.75, pad=0.03, extend="both")
            cbd.set_label("puntos de percentil de diferencia · azul: el "
                          "candidato baja el riesgo · rojo: lo sube", fontsize=9)
        capa_base.leyenda(fig, capa_verdad.handles(capas))
        pie = capa_verdad.resumen(capas, claves=True)
        if pie:
            fig.text(0.5, -0.005, pie, ha="center", va="top", fontsize=7.5,
                     color="0.25", linespacing=1.5)
        capa_base.alerta_global(fig, res.get("fwi_medio_pctl_clim"),
                                res.get("fwi_medio"))
        nv = res["niveles_unico"]
        fig.suptitle(f"{fstr} · {res['rama']} · FWI medio {res['fwi_medio']:.1f} "
                     f"(pctl clim {res['fwi_medio_pctl_clim']:.0f}) · único: "
                     + " · ".join(f"{k} {v:.1f}%" for k, v in nv.items()), fontsize=12)
        fig.savefig(config.salida(f"dos_riesgo_{fstr}.png"), dpi=165, bbox_inches="tight")
        plt.close(fig)
        print(f"  guardado {config.salida(f'dos_riesgo_{fstr}.png')}", flush=True)
    ds.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pasada")
    p.add_argument("--dias", type=int, nargs="+", default=[0, 1])
    main(p.parse_args())
