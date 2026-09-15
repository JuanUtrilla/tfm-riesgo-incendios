#!/usr/bin/env python3
"""
Experimento B: ¿se va el AUC en la reconstrucción meteorológica?

No sobrescribe nada. Escribe solo dataset/experimento_b_*.{parquet,json}.

La pregunta, y por qué es la última que queda
Test 2020: AUC-ROC 0,923. Comprobación operativa 2026: 0,64. El experimento A
(`ablacion_proxies_operativos.py`) midió que congelar el satélite y anular los
rayos cuesta 0,0045, así que la causa no está ahí. Del hueco quedan dos causas
posibles:

  (a) Definicional: prevalencia 25 % de diseño frente a 1,1 % real, negativos
      con buffer, etiqueta EGIF ≥1 ha frente a EFFIS ≥30 ha, unidad celda de
      1 km frente a estación ±25 km. No se puede "arreglar": son dos
      preguntas distintas.
  (b) Meteorológica: en entrenamiento las 20 features meteo salen de
      ERA5-Land (reescalado a 1 km dentro del cubo IberFire); en producción
      salen de estación AEMET (serie observada) o de forecast municipal.

Este script ataca (b) sirviéndole al modelo, para los mismos días y estaciones
de 2026, la meteo de ERA5-Land: la misma fuente que vio al entrenar.

Lo que se puede responder sin ninguna clave (fase 1)
Los CSV sellados de producción guardan 7 columnas meteo ya calculadas:
`fwi`, `fwi_pctl_local`, `fwi_anom_sigma`, `t2m_max`, `rh_min`, `viento_max`,
`dias_sin_lluvia`, `precip_30d`. Comparar esas mismas features calculadas
desde ERA5-Land contra las de producción, estación a estación y día a día,
cuantifica el desplazamiento de fuente feature por feature sin tocar el
modelo. Es el pendiente nº1 de VALIDACION.md §7.

Lo que exige claves (fase 2)
Puntuar con el modelo pide las 46 features. 44 salen de aquí o del parquet de
estaciones congelado; faltan las dos de FIRMS. `n_detec_50km_7d` está en el
CSV de producción y se reutiliza; `frp_max_50km_7d` no está guardada y exige
FIRMS_MAP_KEY. Sin clave se pone a 0 y queda declarado: por el experimento A
se sabe que las features FIRMS pesan ~0, así que el sesgo es pequeño, pero es
un sesgo y no se esconde.

Caveats que hay que declarar en la memoria
· ERA5-Land es 0,1° (~9 km) y aquí se toma la celda de tierra más cercana a la
  estación. El cubo usa ERA5-Land reescalado a 1 km. No es lo mismo: esta
  comparación mide "ERA5-Land crudo vs AEMET", no "el cubo vs AEMET".
· No se corrige por altitud. La orografía de ERA5-Land es suave y en montaña
  la temperatura tendrá sesgo frío/cálido frente a la estación. Corregirlo
  metería una diferencia más y dejaría de medirse la fuente tal cual es.
· `viento_max` en producción es min(velmedia×1,5, racha) de AEMET; aquí es el
  máximo horario del viento a 10 m. La definición también cambia entre fuentes,
  y eso forma parte de lo que se está midiendo, no es un error.

Uso:  /home/charredgem/miniconda3/envs/tfm_fuego/bin/python experimento_b_era5.py
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DIR = Path(__file__).parent
REPO_OP = Path("/home/charredgem/Desktop/Master/aemet_horario_verano2026")
ERA5 = DIR / "dataset" / "era5_2026"
SALIDA_SERIE = DIR / "dataset" / "experimento_b_serie_era5.parquet"
SALIDA_FEATS = DIR / "dataset" / "experimento_b_features.parquet"
SALIDA_JSON = DIR / "dataset" / "experimento_b_resultados.json"

sys.path.insert(0, str(REPO_OP))
from fwi_canadiense import calcular_fwi_serie          # noqa: E402

DIAS_SPINUP = 80          # el mismo que ranking_diario.py
UMBRALES = [(0.80, "EXTREMO"), (0.55, "ALTO"), (0.25, "MODERADO"), (-1, "BAJO")]


# --------------------------------------------------------------------------- #
# 1. ERA5-Land horario → serie diaria por estación
# --------------------------------------------------------------------------- #
def celdas_de_tierra(ds):
    """KDTree de las celdas con dato (ERA5-Land es NaN sobre el mar). Se usa el
    primer instante: la máscara de tierra no cambia con el tiempo."""
    from scipy.spatial import cKDTree
    m = ds["t2m"].isel(valid_time=0).notnull().values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    pts = np.column_stack([LA[m], LO[m]])
    return cKDTree(pts), pts


def serie_estaciones(est):
    """Serie diaria por estación desde los NetCDF mensuales de ERA5-Land."""
    ficheros = sorted(ERA5.glob("era5land_*.nc"))
    if not ficheros:
        raise SystemExit(f"no hay ficheros en {ERA5}: ejecuta antes "
                         f"descargar_era5_2026.py")
    trozos = []
    arbol = pts = None
    for f in ficheros:
        ds = xr.open_dataset(f)
        if arbol is None:
            arbol, pts = celdas_de_tierra(ds)
            _, idx = arbol.query(est[["lat", "lon"]].values)
            slat = xr.DataArray(pts[idx, 0], dims="est",
                                coords={"est": est["idema"].values})
            slon = xr.DataArray(pts[idx, 1], dims="est",
                                coords={"est": est["idema"].values})
            d_km = np.round(np.hypot(
                (pts[idx, 0] - est["lat"].values) * 111,
                (pts[idx, 1] - est["lon"].values) * 111
                * np.cos(np.radians(est["lat"].values))), 1)
            print(f"celda de tierra más cercana: mediana {np.median(d_km):.1f} km, "
                  f"p95 {np.percentile(d_km, 95):.1f} km, máx {d_km.max():.1f} km")
        sub = ds[["t2m", "d2m", "u10", "v10", "tp"]].sel(
            latitude=slat, longitude=slon, method="nearest")
        trozos.append(sub.to_dataframe().reset_index())
        ds.close()
        print(f"  {f.name}: {len(trozos[-1]):,} filas horarias", flush=True)

    h = pd.concat(trozos, ignore_index=True)
    h = h.rename(columns={"est": "idema", "valid_time": "ts"})
    h["fecha"] = h["ts"].dt.normalize()

    # humedad relativa desde T y punto de rocío (Magnus, sobre agua)
    def es(tc):
        return 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    tc = h["t2m"] - 273.15
    tdc = h["d2m"] - 273.15
    h["hr"] = (es(tdc) / es(tc) * 100).clip(0, 100)
    h["ta"] = tc
    h["vv"] = np.hypot(h["u10"], h["v10"])

    g = h.groupby(["idema", "fecha"])
    diaria = pd.DataFrame({
        "tmax": g["ta"].max(), "tmin": g["ta"].min(),
        "hr_min": g["hr"].min(), "viento_max": g["vv"].max(),
        "n_horas": g["ta"].count()}).reset_index()

    # precipitación del día D = tp a las 00:00 del día D+1 (acumulada 00→24).
    # Verificado empíricamente, ver descargar_era5_2026.py.
    p = h[h["ts"].dt.hour == 0][["idema", "fecha", "tp"]].copy()
    p["fecha"] = p["fecha"] - pd.Timedelta(days=1)
    p["prec"] = (p["tp"] * 1000).clip(lower=0)
    diaria = diaria.merge(p[["idema", "fecha", "prec"]],
                          on=["idema", "fecha"], how="left")
    return diaria.sort_values(["idema", "fecha"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. serie diaria → las 46 features, para cada día objetivo
# --------------------------------------------------------------------------- #
def features_dia(s, i, e, idema, fecha, festivos, clim_dir):
    """Copia fiel del bloque de ranking_diario.main(), parametrizada por día."""
    mes = fecha.month
    fwi_s = calcular_fwi_serie(s["tmax"].values, s["hr_min"].values,
                               s["viento_max"].values * 3.6,
                               s["prec"].values,
                               s["fecha"].dt.month.values)["fwi"]
    F = {"fwi": fwi_s[i], "t2m_max": s.loc[i, "tmax"],
         "t2m_min": s.loc[i, "tmin"], "rh_min": s.loc[i, "hr_min"],
         "viento_max": s.loc[i, "viento_max"], "precip_dia": s.loc[i, "prec"]}
    es_ = 0.6108 * np.exp(17.27 * F["t2m_max"] / (F["t2m_max"] + 237.3))
    F["vpd_max"] = es_ * (1 - F["rh_min"] / 100)
    pre, rh, tmx, vto = (s[c].values for c in
                         ["prec", "hr_min", "tmax", "viento_max"])
    F.update(precip_7d=np.nansum(pre[i-7:i]),
             precip_15d=np.nansum(pre[i-15:i]),
             precip_30d=np.nansum(pre[i-30:i]),
             fwi_med_7d=np.nanmean(fwi_s[i-7:i]),
             fwi_max_7d=np.nanmax(fwi_s[i-7:i]),
             fwi_med_15d=np.nanmean(fwi_s[i-15:i]),
             fwi_med_30d=np.nanmean(fwi_s[i-30:i]),
             rh_min_med_7d=np.nanmean(rh[i-7:i]),
             t2m_max_med_7d=np.nanmean(tmx[i-7:i]),
             viento_max_med_7d=np.nanmean(vto[i-7:i]))
    secos = 0
    for k in range(i, -1, -1):
        if np.isnan(pre[k]) or pre[k] >= 1.0 or secos >= 120:
            break
        secos += 1
    F["dias_sin_lluvia"] = secos
    ruta_clim = clim_dir / f"{idema}.npz"
    clim = np.load(ruta_clim)[f"m{mes}"] if ruta_clim.exists() else np.array([])
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
    F.update(idema=idema, nombre=e["nombre"], lat=e["lat"], lon=e["lon"],
             fecha=str(fecha.date()))
    return F


def construir(diaria, est, dias_objetivo):
    import holidays
    festivos = holidays.Spain(years=[2026])
    clim_dir = REPO_OP / "modelo" / "clim_fwi"
    filas = []
    for idema, s in diaria.groupby("idema"):
        if idema not in est.index:
            continue
        e = est.loc[idema]
        cal = pd.date_range(s["fecha"].min(), s["fecha"].max(), freq="D")
        s = (s.set_index("fecha").reindex(cal).rename_axis("fecha")
              .reset_index())
        pos = {f: i for i, f in enumerate(s["fecha"])}
        for dia in dias_objetivo:
            i = pos.get(dia)
            if i is None or i < DIAS_SPINUP or pd.isna(s.loc[i, "tmax"]):
                continue
            # Ventana rodante de exactamente DIAS_SPINUP días antes del día
            # objetivo, no la serie entera. Producción arranca el FWI 80 días
            # antes de "hoy" (ranking_diario.serie_diaria_todas), y el DC del
            # FWI tiene memoria de meses: alimentarlo con 124 días en vez de 80
            # da un DC distinto y el brazo deja de ser comparable con lo que
            # producción publicó. Medido: sin esto, la reconstrucción del brazo
            # AEMET se desvía −0,097 de AUC respecto a los ranking_*.csv
            # commiteados, casi 3× el efecto que el experimento quiere medir.
            sl = (s.iloc[i - DIAS_SPINUP:i + 1]
                   .reset_index(drop=True))
            filas.append(features_dia(sl, DIAS_SPINUP, e, idema, dia,
                                      festivos, clim_dir))
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# 3. comparación con producción y evaluación
# --------------------------------------------------------------------------- #
COMPARABLES = ["fwi", "fwi_pctl_local", "fwi_anom_sigma", "t2m_max", "rh_min",
               "viento_max", "dias_sin_lluvia", "precip_30d"]


def cargar_produccion():
    """Los CSV sellados/cerrados de producción, con su tipo y día."""
    filas = []
    for ruta in sorted((REPO_OP / "rankings").glob("*.csv")):
        m = re.match(r"(prevision_D0|prevision_D1|ranking)_"
                     r"(\d{4}-\d{2}-\d{2})\.csv$", ruta.name)
        if not m:
            continue
        df = pd.read_csv(ruta)
        if "prob" not in df or "idema" not in df:
            continue
        df["tipo"] = {"prevision_D0": "D0 (sellada)",
                      "prevision_D1": "D1 (sellada)",
                      "ranking": "ranking (día cerrado)"}[m.group(1)]
        df["fecha"] = m.group(2)
        filas.append(df)
    return pd.concat(filas, ignore_index=True)


def comparar_features(era5, prod):
    """Correlación y sesgo de cada feature meteo: ERA5-Land vs producción."""
    out = {}
    for tipo, p in prod.groupby("tipo"):
        j = era5.merge(p, on=["idema", "fecha"], suffixes=("_era5", "_prod"))
        if not len(j):
            continue
        d = {}
        for c in COMPARABLES:
            a, b = j.get(f"{c}_era5"), j.get(f"{c}_prod")
            if a is None or b is None:
                continue
            ok = a.notna() & b.notna()
            if ok.sum() < 50:
                continue
            d[c] = {"n": int(ok.sum()),
                    "r": round(float(a[ok].corr(b[ok])), 3),
                    "sesgo_era5_menos_prod": round(float((a[ok] - b[ok]).mean()), 2),
                    "mae": round(float((a[ok] - b[ok]).abs().mean()), 2),
                    "media_era5": round(float(a[ok].mean()), 2),
                    "media_prod": round(float(b[ok].mean()), 2)}
        out[tipo] = {"n_estacion_dia": int(len(j)), "features": d}
    return out


def main():
    est = pd.read_parquet(REPO_OP / "modelo" / "estaciones_prototipo.parquet")
    print(f"estaciones: {len(est)}")

    if SALIDA_SERIE.exists():
        diaria = pd.read_parquet(SALIDA_SERIE)
        print(f"serie ERA5 ya calculada: {len(diaria):,} estación-día")
    else:
        diaria = serie_estaciones(est)
        diaria.to_parquet(SALIDA_SERIE)
        print(f"serie ERA5: {len(diaria):,} estación-día → {SALIDA_SERIE}")

    prod = cargar_produccion()
    dias = sorted(pd.to_datetime(prod["fecha"].unique()))
    print(f"producción: {len(prod):,} filas · {len(dias)} días "
          f"({dias[0]:%F} → {dias[-1]:%F})")

    est_i = est.set_index("idema")
    feats = construir(diaria, est_i, dias)
    feats.to_parquet(SALIDA_FEATS)
    print(f"features ERA5: {len(feats):,} estación-día → {SALIDA_FEATS}\n")

    comp = comparar_features(feats, prod)
    print("=== FASE 1: ERA5-Land vs producción, feature a feature ===")
    for tipo, d in comp.items():
        print(f"\n{tipo}  (n={d['n_estacion_dia']:,})")
        print(f"  {'feature':<18}{'r':>7}{'sesgo':>9}{'MAE':>8}"
              f"{'media ERA5':>12}{'media prod':>12}")
        for c, m in d["features"].items():
            print(f"  {c:<18}{m['r']:>7.3f}{m['sesgo_era5_menos_prod']:>9.2f}"
                  f"{m['mae']:>8.2f}{m['media_era5']:>12.2f}"
                  f"{m['media_prod']:>12.2f}")

    SALIDA_JSON.write_text(json.dumps(
        {"fase1_comparacion_features": comp,
         "n_serie_era5": int(len(diaria)),
         "n_features_era5": int(len(feats)),
         "caveats": [
             "ERA5-Land 0,1° celda de tierra más cercana; el cubo usa "
             "ERA5-Land reescalado a 1 km. Mide ERA5-Land crudo vs AEMET.",
             "Sin corrección de altitud.",
             "viento_max: máximo horario a 10 m aquí; min(velmedia×1,5, racha) "
             "en producción. La definición también cambia."]},
        indent=2, ensure_ascii=False))
    print(f"\nguardado: {SALIDA_JSON}")


if __name__ == "__main__":
    main()
