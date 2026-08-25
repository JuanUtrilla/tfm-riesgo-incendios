#!/usr/bin/env python3
"""
Mapa nacional de riesgo para HOY (D) y MAÑANA (D+1) — sin IDW.

NO TOCA PRODUCCIÓN. Escribe salida/riesgo_hoy_<fecha>.png y .npz.

=============================================================================
QUÉ SUSTITUYE
=============================================================================
`mapa_riesgo_hoy.py` calcula las 21 features meteo por estación AEMET y las
interpola a la malla de 1 km con IDW k=8, con corrección de altitud. Ese paso
nunca se validó, depende de que ~690 estaciones reporten (16 % sin viento) y
hace que `fwi_pctl_local` divida numerador AEMET entre denominador del cubo:
satura al 3,30 %.

Aquí la meteo se calcula en los 5.605 nodos de la malla nativa de ERA5-Land y
cada celda de 1 km toma el valor de SU nodo. Sin interpolación y sin huecos.

TODO LO DEMÁS SE MANTIENE IGUAL que en producción, a propósito: vegetación y
LST de la climatología mensual 2020-24 del cubo, estáticas y CLC del cubo,
EGIF mismo-mes, FIRMS NRT en malla, rayos=0, y el MISMO modelo
(`xgb_v2_prototipo`, 46 features). Lo único que cambia es de dónde sale la
meteo, que es lo que se quiere medir.

=============================================================================
LAS DOS RAMAS Y LA CORRECCIÓN
=============================================================================
    ... 1-may ... D-7 │ D-6 ... D, D+1
        ERA5-Land     │      IFS
     (era5land_diario)│  (malla_02b_ifs)

El FWI es recursivo: se corre sobre el reanálisis guardando (FFMC, DMC, DC) y
se REANUDA desde el estado del último día de reanálisis para avanzar la rama
de previsión. No se recalcula desde cero ni se parte la recursión.

La rama de previsión NO se sirve cruda. Medido en `malla_02b_hibrido.py`: el
híbrido crudo satura al 2,05 % contra 0,23 % del reanálisis, y no lo salva la
memoria del FWI (con 3 días de IFS ya está el 83 % del daño; el valor diario
lo manda el FFMC, cuya constante de tiempo son horas). Se corrige con el mapeo
de cuantiles a la escala de ERA5-Land, que es el denominador real:
2,46 % → 0,41 % fuera de muestra, contra 0,31 % de la referencia.

El mapeo es monótono, así que NO altera el orden del ranking dentro de un día,
que es lo único que usa el producto operativo.

=============================================================================
LÍMITES CONOCIDOS
=============================================================================
· El mapeo actual se ajustó con 120 nodos y solo con 2024. Reajustarlo con los
  5.605 y más de una temporada está pendiente.
· Solo se mapea el FWI. Las otras variables del IFS (tmax, hr_min, viento,
  precipitación) entran crudas en el modelo. Módulo 3b midió que corregir la
  SALIDA gana a corregir las entradas, pero para estas cuatro no está medido.
· `dias_sin_lluvia` mira 120 días y la serie disponible es más corta; el
  truncado afecta al 0,30 % de celdas (medido en el módulo 5).
· 226 nodos costeros (4,0 %) no tienen dato ERA5-Land y heredan el nodo
  terrestre más cercano, entero.

Uso:
    python riesgo_hoy.py                # HOY y MAÑANA
    python riesgo_hoy.py --dias 0
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from dotenv import load_dotenv

import config
import malla_02b_ifs as ifsmod

# Las claves viven en el .env del repo original. Sin esto, FIRMS_MAP_KEY no
# está en el entorno y las features de FIRMS salen a CERO, mientras que
# producción sí las carga (vía `tiempo_real`): la comparación quedaría
# trucada justo en las celdas con fuego activo, que es donde se decide.
load_dotenv(f"{config.FUENTE}/.env")
from fwi_canadiense import calcular_fwi_serie
from malla_05_riesgo import a_celdas, suma_caja, ventanas

MODELO = f"{config.MODELOS}/xgb_v2_prototipo.ubj"
FEATS_JSON = f"{config.MODELOS}/xgb_v2_prototipo_features.json"
REANALISIS = config.entrada("era5land_diario.nc")
MAPEO = config.entrada("mapeo_ifs_a_era5land.npz")
CORTES = [-1, 0.25, 0.55, 0.80, 2]
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]


def series_nodos(objetivos):
    """Serie diaria por nodo: reanálisis + previsión, y el FWI ya corregido.

    Devuelve (S, fwi, fechas, idx_nodo, n_respaldo) con S en (dias, nodos)."""
    n = np.load(config.NODOS)
    la, lo, idx_nodo = n["nodo_lat"], n["nodo_lon"], n["idx_nodo"]

    ds = xr.open_dataset(REANALISIS)
    pts = ds.sel(latitude=xr.DataArray(la, dims="n"),
                 longitude=xr.DataArray(lo, dims="n"), method="nearest")
    f_re = pd.to_datetime(ds["fecha"].values).normalize()
    R = {v: pts[v].values for v in ("tmax", "tmin", "hr_min", "viento_max",
                                    "prec")}
    ds.close()
    ultimo_re = f_re[-1]
    print(f"  reanálisis: {f_re[0].date()} → {ultimo_re.date()} "
          f"({len(f_re)} días)", flush=True)

    ifs = ifsmod.descarga()
    f_ifs_all = pd.to_datetime(ifs["fecha"]).dt.normalize()
    fin = max(objetivos)
    nuevas = sorted(set(f_ifs_all[(f_ifs_all > ultimo_re)
                                  & (f_ifs_all <= fin)]))
    if not nuevas:
        raise SystemExit("el IFS no aporta días más allá del reanálisis")
    if fin not in nuevas:
        raise SystemExit(f"el IFS no llega al {fin.date()}")
    print(f"  previsión : {nuevas[0].date()} → {nuevas[-1].date()} "
          f"({len(nuevas)} días)", flush=True)
    I = ifsmod.matrices(ifs, pd.DatetimeIndex(nuevas))

    fechas = pd.DatetimeIndex(list(f_re) + list(nuevas))
    meses = fechas.month.values
    nn = len(la)
    S = {}
    for v in ("tmax", "hr_min", "viento_max", "prec"):
        S[v] = np.vstack([R[v], I[v]])
    # el IFS no da tmin; para D y D+1 se arrastra el último observado. t2m_min
    # pesa poco en el modelo y no entra en el FWI, pero queda anotado.
    S["tmin"] = np.vstack([R["tmin"],
                           np.repeat(R["tmin"][-1:], len(nuevas), axis=0)])

    # --- FWI: reanálisis con estado, y la previsión reanudando desde él -----
    print(f"  FWI en {nn:,} nodos...", flush=True)
    nre = len(f_re)
    fwi = np.full((len(fechas), nn), np.nan)
    mp = np.load(MAPEO)
    for j in range(nn):
        o = calcular_fwi_serie(R["tmax"][:, j], R["hr_min"][:, j],
                               R["viento_max"][:, j] * 3.6, R["prec"][:, j],
                               meses[:nre])
        fwi[:nre, j] = o["fwi"]
        if not np.isfinite(o["ffmc"][-1]):
            continue
        p = calcular_fwi_serie(
            S["tmax"][nre:, j], S["hr_min"][nre:, j],
            S["viento_max"][nre:, j] * 3.6, S["prec"][nre:, j], meses[nre:],
            ffmc0=o["ffmc"][-1], dmc0=o["dmc"][-1], dc0=o["dc"][-1])["fwi"]
        # la rama de previsión, a la escala de ERA5-Land (ver cabecera)
        fwi[nre:, j] = np.where(np.isfinite(p), np.interp(p, mp["xs"], mp["ys"]),
                                np.nan)
    return S, fwi, fechas, idx_nodo, la, lo, nre


def meteo_dia(S, fwi, fechas, k, mes, la, lo):
    """Las 21 features meteo en los nodos para el día `k` de la serie."""
    sl = slice(0, k + 1)
    F = ventanas(fwi[sl], S["tmax"][sl], S["tmin"][sl], S["hr_min"][sl],
                 S["viento_max"][sl], S["prec"][sl])
    clim = np.load(config.CLIM)
    C, nC = clim[f"m{mes}"], clim[f"n{mes}"].astype(float)
    with np.errstate(invalid="ignore"):
        cnt = np.nansum(C <= F["fwi"][None, :], axis=0)
        F["fwi_pctl_local"] = np.where(nC > 0, cnt / np.maximum(nC, 1) * 100,
                                       np.nan)
        cmed, cstd = np.nanmean(C, axis=0), np.nanstd(C, axis=0)
        F["fwi_anom_sigma"] = np.where((nC > 0) & (cstd > 0),
                                       (F["fwi"] - cmed) / cstd, np.nan)
    malo = ~np.isfinite(F["fwi"]) | (nC == 0)
    n_malo = int(malo.sum())
    if n_malo:
        bueno = np.where(~malo)[0]
        arbol = cKDTree(np.column_stack([la[bueno], lo[bueno]]))
        _, kk = arbol.query(np.column_stack([la[malo], lo[malo]]))
        origen = bueno[kk]
        for v in F:
            F[v][malo] = F[v][origen]
    return F, n_malo


def mensual(ds, mes):
    """Vegetación/LST 2020-24 + EGIF mismo-mes. Reutiliza la caché de producción."""
    ya = f"{config.FUENTE}/prototipo/cache/malla_mensual_m{mes}.npz"
    propia = config.salida(f"malla_mensual_m{mes}.npz")
    for r in (propia, ya, f"{config.ESTADO}/malla_mensual_m{mes}.npz"):
        if os.path.exists(r):
            print(f"  mensual m{mes}: caché ({os.path.basename(r)})", flush=True)
            return dict(np.load(r))
    print(f"  precomputando mensual m{mes}...", flush=True)
    t = ds["time"].values.astype("datetime64[D]")
    anios = t.astype("datetime64[Y]").astype(int) + 1970
    meses = (t.astype("datetime64[M]").astype(int) % 12) + 1
    idx = np.where((anios >= 2020) & (anios <= 2024) & (meses == mes))[0]
    out = {}
    for var, col in [("NDVI", "ndvi"), ("LAI", "lai"),
                     ("SWI_010", "swi010"), ("LST", "lst")]:
        out[col] = np.nanmean(ds[var].isel(time=idx).values, axis=0)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    xs, ys = ds["x"].values, ds["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    eg = pd.read_csv(config.EGIF, usecols=["fecha", "lat", "lng"]).dropna()
    eg["fecha"] = pd.to_datetime(eg["fecha"], errors="coerce")
    eg = eg.dropna()
    eg = eg[(eg["fecha"].dt.year >= 2008) & (eg["fecha"].dt.month == mes)]
    ex, ey = tr.transform(eg["lng"].values, eg["lat"].values)
    eix = np.rint((ex - xs[0]) / (xs[1] - xs[0])).astype(int)
    eiy = np.rint((ey - ys[0]) / (ys[1] - ys[0])).astype(int)
    ok = (eix >= 0) & (eix < nx) & (eiy >= 0) & (eiy < ny)
    g = np.zeros((ny, nx))
    np.add.at(g, (eiy[ok], eix[ok]), 1)
    out["n_fuegos_10km_mismomes_hist"] = suma_caja(g, 10)
    np.savez_compressed(propia, **out)
    return out


def firms_nrt(ds):
    """FIRMS NRT [D-5, D-1] en malla. Sin key → ceros, como en producción."""
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    gfrp, gn = np.zeros((ny, nx)), np.zeros((ny, nx))
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        print("  FIRMS NRT: sin FIRMS_MAP_KEY → ceros. OJO: producción SÍ "
              "las carga; comparar así la penaliza", flush=True)
        return maximum_filter(gfrp, size=101, mode="constant"), suma_caja(gn, 50)
    import requests
    from io import StringIO
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
           f"VIIRS_NOAA20_NRT/-10,35,5,44/5")
    # 21/08/2026: la API de FIRMS estuvo intermitente para los runners de
    # GitHub (ConnectTimeout). Una sola llamada de 30 s y a ceros degradaba el
    # mapa en silencio: ahora 4 intentos con espera creciente (≈6 min en total).
    import time
    df = pd.DataFrame()
    for intento, espera in enumerate((0, 60, 120, 180), 1):
        time.sleep(espera)
        try:
            r = requests.get(url, timeout=90)
            if r.ok and r.text.startswith("latitude"):
                df = pd.read_csv(StringIO(r.text))
                break
            print(f"  FIRMS NRT intento {intento}: HTTP {r.status_code}", flush=True)
        except Exception as e:
            print(f"  FIRMS NRT intento {intento}: {type(e).__name__}", flush=True)
    else:
        print("  ⚠️ FIRMS NRT no disponible tras 4 intentos → ceros (el mapa "
              "de hoy va SIN fuego activo)", flush=True)
    if len(df):
        hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
        df = df[pd.to_datetime(df["acq_date"]) < hoy]
    if len(df):
        xs, ys = ds["x"].values, ds["y"].values
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        fx, fy = tr.transform(df["longitude"].values, df["latitude"].values)
        fix = np.rint((fx - xs[0]) / (xs[1] - xs[0])).astype(int)
        fiy = np.rint((fy - ys[0]) / (ys[1] - ys[0])).astype(int)
        ok = (fix >= 0) & (fix < nx) & (fiy >= 0) & (fiy < ny)
        np.maximum.at(gfrp, (fiy[ok], fix[ok]), df["frp"].values[ok])
        np.add.at(gn, (fiy[ok], fix[ok]), 1)
        print(f"  FIRMS NRT: {len(df)} detecciones", flush=True)
        # la caché cruda la reutiliza `capa_verdad` para pintar los focos sin
        # volver a llamar a la API (que ya falla sola bastante)
        os.makedirs(config.salida("_firms"), exist_ok=True)
        df.to_csv(config.salida(f"_firms/nrt_{hoy:%Y-%m-%d}.csv"), index=False)
    return maximum_filter(gfrp, size=101, mode="constant"), suma_caja(gn, 50)


def main(a):
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    objetivos = [hoy + pd.Timedelta(days=h) for h in a.dias]
    FULL = json.load(open(FEATS_JSON))
    print(f"Riesgo sin IDW · modelo {os.path.basename(MODELO)} "
          f"({len(FULL)} features)\n", flush=True)

    S, fwi, fechas, idx_nodo, la, lo, nre = series_nodos(objetivos)

    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    es_esp = ds["is_spain"].values.astype(bool)
    ys = ds["y"].values
    print("\n  estáticas del cubo...", flush=True)
    B = {}
    for k, v in {"elevacion": "elevation_mean", "pendiente": "slope_mean",
                 "rugosidad": "roughness_mean",
                 "dist_carreteras": "dist_to_roads_mean",
                 "dist_rios": "dist_to_waterways_mean"}.items():
        B[k] = ds[v].values
    B["popdens"] = ds["popdens_2020"].values
    for k, suf in {"clc_bosque": "forest_proportion",
                   "clc_matorral": "scrub_proportion",
                   "clc_agricola": "agricultural_proportion",
                   "clc_artificial": "artificial_proportion",
                   "clc_abierto": "open_space_proportion",
                   "clc_agric_hetero": "heterogeneous_agriculture_proportion"}.items():
        B[k] = ds[f"CLC_2018_{suf}"].values
    B["ccaa"] = np.nan_to_num(ds["AutonomousCommunities"].values,
                              nan=-1).astype(int)
    B["frp_max_50km_7d"], B["n_detec_50km_7d"] = firms_nrt(ds)
    B["rayos_dia"] = np.zeros((ny, nx))
    B["rayos_7d"] = np.zeros((ny, nx))

    modelo = xgb.XGBClassifier()
    modelo.load_model(MODELO)
    sel = es_esp.ravel()
    import holidays

    for obj in objetivos:
        fstr = str(obj.date())
        k = int(np.where(fechas == obj)[0][0])
        print(f"\n=== {fstr} ===", flush=True)
        F, n_malo = meteo_dia(S, fwi, fechas, k, obj.month, la, lo)
        M = a_celdas(F, idx_nodo)
        G = dict(B)
        G.update(mensual(ds, obj.month))
        G["ndvi_med_30d"] = G["ndvi"]          # mismo proxy que producción
        fest = holidays.Spain(years=[obj.year])
        G["es_festivo"] = np.full((ny, nx), float(obj.weekday() >= 5
                                                  or obj.date() in fest))
        G["mes"] = np.full((ny, nx), obj.month)
        G["dia_anio"] = np.full((ny, nx), obj.dayofyear)

        X = pd.DataFrame({c: (M[c] if c in M else G[c]).ravel()[sel]
                          for c in FULL})
        prob = np.full(ny * nx, np.nan)
        prob[sel] = modelo.predict_proba(X)[:, 1]
        prob = prob.reshape(ny, nx)

        p = prob[es_esp]
        cuenta = pd.cut(pd.Series(p), CORTES, labels=NIVELES).value_counts()
        pct = (cuenta / len(p) * 100).round(1)
        resumen = " · ".join(f"{n}: {pct[n]}%" for n in NIVELES)
        pl = M["fwi_pctl_local"][es_esp]
        pl = pl[np.isfinite(pl)]
        print(f"  nodos con respaldo: {n_malo} · FWI medio "
              f"{np.nanmean(M['fwi'][es_esp]):.1f}")
        print(f"  fwi_pctl_local: mediana {np.median(pl):.1f} · "
              f"% >=99,9 {(pl >= 99.9).mean() * 100:.2f}")
        print(f"  {resumen}", flush=True)

        np.savez_compressed(config.salida(f"riesgo_hoy_{fstr}.npz"),
                            prob=prob.astype(np.float32))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 9))
        im = ax.imshow(prob, origin="lower" if ys[1] > ys[0] else "upper",
                       cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
        import capa_base
        import capa_verdad
        capa_base.dibujar(ax, prob.shape[1], prob.shape[0], etiquetas=25, lw=1.2)
        capas = capa_verdad.preparar(ds, fstr)
        capa_verdad.dibujar(ax, capas, lw=1.2, rotular=6)
        capa_base.leyenda(fig, capa_verdad.handles(capas))
        fig.colorbar(im, label="probabilidad del modelo "
                                "(prevalencia de diseño 25%)")
        rama = "reanálisis + IFS mapeado" if k >= nre else "solo reanálisis"
        ax.set_title(f"Riesgo de incendio — {fstr}  (sin IDW)\n"
                     f"meteo en 5.605 nodos ERA5-Land · {rama} · "
                     f"estáticas 1 km del cubo\n{resumen}", fontsize=11)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(config.salida(f"riesgo_hoy_{fstr}.png"), dpi=150)
        plt.close(fig)
        print(f"  guardado {config.salida(f'riesgo_hoy_{fstr}.png')}",
              flush=True)
    ds.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dias", type=int, nargs="+", default=[0, 1])
    main(p.parse_args())
