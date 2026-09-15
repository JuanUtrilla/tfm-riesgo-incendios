#!/usr/bin/env python3
"""
La malla contra los rankings sellados del prototipo por estación, 24 días.

No toca producción. Escribe salida/rankings_effis.csv y .json.

Por qué existía esto y no se estaba usando
------------------------------------------
El colector de AEMET lleva guardando su salida
operativa diaria: `rankings/prevision_D0_<fecha>.csv`, 687 estaciones con su
probabilidad, del 22-jul al 18-ago de 2026. Son 24 días de sistema real,
sellados en su momento, que ya estaban en disco.

Con el reanálisis llegando al 14-ago, se solapan 24 días. Eso dobla largamente
los 17 días de julio que ya había, sin esperar a que pase la temporada.

Aviso: la comparación favorece a la malla; el resultado es una cota superior
---------------------------------------------------------------------------
El ranking de cada día es una previsión (`dias_forecast=1`, meteo prevista de
AEMET). La malla que se compara aquí usa reanálisis de ese mismo día, o sea
con la meteo ya observada. Tiene ventaja de retrovisor.

Y la ventaja no es pequeña: medido en `malla_02b_hibrido.py`, servir con
previsión en vez de reanálisis dispara la saturación de 0,23 % a 2,05 %. La
rama de previsión pesa.

Así que lo de aquí se lee como cota superior de la malla, no como empate ni
como victoria. Hacerlo justo exige reconstruir el híbrido (reanálisis hasta
D-7 + IFS archivado) para los 24 días, y eso son ~22.400 unidades de
Open-Meteo contra 10.000 diarias: dos o tres días de cuota. Queda anotado.

FIRMS sí se reconstruye bien: la API admite ventanas de 5 días terminadas en
una fecha pasada, así que cada día lleva su [D-5, D-1] real. Sin eso la malla
iría ciega a los incendios en curso y perdería por un motivo falso, que es
exactamente el error que se coló en la comparación del 20-ago.

Etiqueta: estación positiva si hay superficie quemada de EFFIS a menos de
25 km ese día, el mismo criterio que usa la validación del colector de AEMET.

Uso: python comparar_rankings.py
"""

import glob
import io
import json
import os
import time

from io import StringIO

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from dotenv import load_dotenv
from pyproj import Transformer
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

import config
from comparar_julio2026 import auc, quemadas
from fwi_canadiense import calcular_fwi_serie
from malla_05_riesgo import suma_caja, ventanas
from puntuar_effis import GEOJSON
from riesgo_hoy import mensual

load_dotenv(f"{config.FUENTE}/.env")
COLECTOR = os.environ.get("TFM_COLECTOR", f"{config.FUENTE}/colector")
RANKINGS = f"{COLECTOR}/rankings/prevision_D0_%s.csv"
RADIO_KM = 25
# días de la ventana FIRMS [D-n, D-1]. 7 = la del entrenamiento.
VENTANA_FIRMS = int(os.environ.get("TFM_FIRMS_DIAS", 7))


def firms_dia(dia, ds):
    """FIRMS [D-5, D-1] de una fecha pasada, cacheado. Ventana de 5 días.

    31/08/2026, fuga de futuro corregida. La API de FIRMS interpreta
    `/{rango}/{fecha}` como `rango` días hacia adelante desde `fecha`,
    inclusive; no hacia atrás. Pasar D-1 devolvía [D-1, D+3]: el mapa del día D
    llevaba dentro los focos del propio incendio y de los tres días siguientes.
    Se veía en que `prod` no perdía acierto al alejarse del día del fuego
    (63 % a 62 % de incendios grandes de D-0 a D-2) mientras que `prod_sin_firms`
    sí (31 % a 18 %), y en que los 33 incendios de >=500 ha tenían un foco FIRMS
    a menos de 5 km en su ventana «pasada» (control aleatorio: 1,2 %).
    Para obtener [D-5, D-1] hay que pedir como inicio D-5.

    31/08/2026, desajuste train/serve. El modelo se entrenó con
    `[D-7, D-1]` (`extraer_features_historia.py:117`, y de ahí el nombre
    `frp_max_50km_7d`), pero toda la tubería de servicio pedía 5 días. Como
    `n_detec_50km_7d` es un conteo, en producción llegaba ~29 % más bajo de lo
    que el modelo aprendió. `VENTANA_FIRMS` fija la ventana; el valor por
    defecto es 7 para que coincida con el entrenamiento. Se cachea en
    `_firms{n}/` para poder comparar las dos.

    Producción nunca estuvo afectada: `riesgo_hoy.py` y el `firms_api.py` del
    scripts del colector llaman sin fecha de inicio (`.../-10,35,5,44/5`), que son los
    5 últimos días hasta hoy. La fuga solo se materializó aquí porque la caché
    de junio y julio se descargó en agosto, cuando el futuro ya existía.
    """
    n = VENTANA_FIRMS
    cache = config.salida(f"_firms{n}/{dia:%Y-%m-%d}.csv")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not os.path.exists(cache):
        import requests
        # la API topa a 5 días por petición ("Invalid day range. Expects [1..5]"),
        # así que una ventana de 7 se arma con dos llamadas consecutivas.
        trozos, ini = [], dia - pd.Timedelta(days=n)
        while ini < dia:
            k = min(5, (dia - ini).days)
            u = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                 f"{os.environ['FIRMS_MAP_KEY']}/VIIRS_NOAA20_NRT/"
                 f"-10,35,5,44/{k}/{ini:%Y-%m-%d}")
            r = requests.get(u, timeout=90)
            if r.text.startswith("latitude"):
                trozos.append(pd.read_csv(StringIO(r.text)))
            ini += pd.Timedelta(days=k)
            time.sleep(2)
        if trozos:
            d = pd.concat(trozos, ignore_index=True).drop_duplicates()
            d = d[pd.to_datetime(d["acq_date"]) < dia]      # cinturón: nunca el día D
            d.to_csv(cache, index=False)
        else:
            open(cache, "w").write("latitude,longitude,acq_date,frp\n")
    df = pd.read_csv(cache)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    gf, gn = np.zeros((ny, nx)), np.zeros((ny, nx))
    if len(df):
        xs, ys = ds["x"].values, ds["y"].values
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        fx, fy = tr.transform(df["longitude"].values, df["latitude"].values)
        ix = np.rint((fx - xs[0]) / (xs[1] - xs[0])).astype(int)
        iy = np.rint((fy - ys[0]) / (ys[1] - ys[0])).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.maximum.at(gf, (iy[ok], ix[ok]), df["frp"].values[ok])
        np.add.at(gn, (iy[ok], ix[ok]), 1)
    return maximum_filter(gf, size=101, mode="constant"), suma_caja(gn, 50)


def main():
    dias = sorted(pd.Timestamp(os.path.basename(p)[13:23])
                  for p in glob.glob(RANKINGS % "*"))
    n = np.load(config.NODOS)
    la, lo, idx_nodo = n["nodo_lat"], n["nodo_lon"], n["idx_nodo"]
    dsr = xr.open_dataset(config.entrada("era5land_diario.nc"))
    pts = dsr.sel(latitude=xr.DataArray(la, dims="n"),
                  longitude=xr.DataArray(lo, dims="n"), method="nearest")
    fechas = pd.to_datetime(dsr["fecha"].values).normalize()
    R = {v: pts[v].values for v in ("tmax", "tmin", "hr_min", "viento_max",
                                    "prec")}
    dsr.close()
    dias = [d for d in dias if d in set(fechas)]
    print(f"  {len(dias)} días con ranking sellado y reanálisis: "
          f"{dias[0]:%F} → {dias[-1]:%F}", flush=True)

    print(f"  FWI en {len(la):,} nodos...", flush=True)
    meses = fechas.month.values
    fwi = np.full(R["tmax"].shape, np.nan)
    for j in range(len(la)):
        fwi[:, j] = calcular_fwi_serie(R["tmax"][:, j], R["hr_min"][:, j],
                                       R["viento_max"][:, j] * 3.6,
                                       R["prec"][:, j], meses)["fwi"]

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
    sel = es_esp.ravel()
    clim = np.load(config.CLIM)
    import holidays
    fest = holidays.Spain(years=[2026])
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    rad = int(RADIO_KM)

    filas = []
    for dia in dias:
        rk = pd.read_csv(RANKINGS % f"{dia:%Y-%m-%d}")
        quem, ha = quemadas(dia, ds, es_esp, ruta=GEOJSON)
        if quem.sum() == 0:
            continue
        cerca = maximum_filter(quem.astype(np.uint8), size=2 * rad + 1,
                               mode="constant").astype(bool)

        k = int(np.where(fechas == dia)[0][0])
        F = ventanas(fwi[:k + 1], R["tmax"][:k + 1], R["tmin"][:k + 1],
                     R["hr_min"][:k + 1], R["viento_max"][:k + 1],
                     R["prec"][:k + 1])
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

        X = pd.DataFrame({c: M[c].ravel()[sel] for c in FULL})
        pr = np.full(ny * nx, np.nan)
        pr[sel] = modelo.predict_proba(X)[:, 1]
        pr = pr.reshape(ny, nx)

        ex, ey = tr.transform(rk["lon"].values, rk["lat"].values)
        ix = np.clip(np.rint((ex - xs[0]) / (xs[1] - xs[0])).astype(int), 0, nx - 1)
        iy = np.clip(np.rint((ey - ys[0]) / (ys[1] - ys[0])).astype(int), 0, ny - 1)
        y = cerca[iy, ix]
        pm = pr[iy, ix]
        pp = rk["prob"].values
        ok = np.isfinite(pm) & np.isfinite(pp)
        if y[ok].sum() < 3 or (~y[ok]).sum() < 3:
            print(f"  {dia:%F}: solo {int(y[ok].sum())} estaciones con fuego "
                  f"a <{RADIO_KM} km · se salta", flush=True)
            continue
        fila = {"fecha": f"{dia:%F}", "estaciones": int(ok.sum()),
                "positivas": int(y[ok].sum()), "area_ha": round(ha, 1),
                "auc_ranking": round(auc(pp[ok][y[ok]], pp[ok][~y[ok]]), 4),
                "auc_malla": round(auc(pm[ok][y[ok]], pm[ok][~y[ok]]), 4)}
        filas.append(fila)
        print(f"  {dia:%F}  positivas {fila['positivas']:3d}/{fila['estaciones']}"
              f" ({ha:7.0f} ha) · AUC ranking {fila['auc_ranking']:.3f} · "
              f"malla {fila['auc_malla']:.3f}", flush=True)
    ds.close()

    d = pd.DataFrame(filas)
    d.to_csv(config.salida("rankings_effis.csv"), index=False)
    dif = (d["auc_malla"] - d["auc_ranking"]).values
    rng = np.random.default_rng(0)
    bs = np.array([dif[rng.integers(0, len(dif), len(dif))].mean()
                   for _ in range(20000)])
    lo_, hi = np.percentile(bs, [2.5, 97.5])
    print("\n" + "=" * 70)
    print(f"AUC POR ESTACIÓN (fuego EFFIS a <{RADIO_KM} km) — {len(d)} días")
    print("=" * 70)
    print(f"  ranking sellado (AEMET, previsión) : {d['auc_ranking'].mean():.3f}")
    print(f"  malla (reanálisis, CON retrovisor) : {d['auc_malla'].mean():.3f}")
    print(f"\n  diferencia {dif.mean():+.3f} · IC95 [{lo_:+.3f}, {hi:+.3f}] · "
          f"la malla gana {int((dif > 0).sum())}/{len(dif)} días")
    print("\n  ⚠️ COTA SUPERIOR: el ranking prevé, la malla observa. No es un")
    print("     cara a cara. Ver la cabecera del módulo.")
    json.dump({"dias": filas, "auc_ranking": float(d["auc_ranking"].mean()),
               "auc_malla": float(d["auc_malla"].mean()),
               "dif": float(dif.mean()), "ic95": [float(lo_), float(hi)],
               "aviso": "cota superior: ranking=previsión, malla=reanálisis"},
              open(config.salida("rankings_effis.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
