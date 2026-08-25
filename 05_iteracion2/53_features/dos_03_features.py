#!/usr/bin/env python3
"""
Dos modelos — paso 3: features de la tabla maestra (cubo + historia).

NO TOCA PRODUCCIÓN. Lee cubo, EGIF, FIRMS y WGLC; escribe en expansión.

=============================================================================
POR QUÉ SE REUTILIZA EL EXTRACTOR ORIGINAL Y NO SE REESCRIBE
=============================================================================
Las features tienen que ser EXACTAMENTE las del modelo en producción para que
la comparación sea del diseño de muestreo y no de la receta de features. Por
eso `procesar_bloque` y `extraer_estaticas` se importan de
`extraer_features_cubo.py` del repo original (solo lectura, con importlib: no
se copia el código, así no puede divergir). La historia (EGIF/FIRMS/rayos) sí
está copiada de `extraer_features_historia.py` porque allí vive dentro de
`main()` y no es importable; la lógica es la misma línea a línea.

Dos diferencias deliberadas respecto al original:
  · PARALELO. 530k filas en vez de 78k: los 144 bloques del cubo van a 8
    procesos y las celdas únicas de la historia a 12. El original era serial.
  · `popdens` del año de la fila NO existe para 2022 (el cubo llega a 2020):
    se usa 2020, que es exactamente lo que sirve producción (congelado).

Salidas (expansión): features_cubo_dos.parquet, features_historia_dos.parquet,
y la tabla final dataset_dos.parquet (maestra + features + vpd_max + dia_anio),
con los negativos de `cuando`/`donde` que caen en `is_near_fire`=1 ELIMINADOS,
igual que hacía `ensamblar_dataset.py`. En `eval_dia` no se elimina nada: en
operación nadie filtra celdas, y `label_effis` se toma del propio cubo.
"""

import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

import config
import config_expansion as ce

# sufijo de la tabla maestra: `dos` (EGIF, dos_02) o `effis` (dos_12).
# Se pasa como primer argumento; por defecto `dos`.
SUF = sys.argv[1] if len(sys.argv) > 1 else "dos"
MAESTRA = f"{ce.DATASET}/maestra_{SUF}.parquet"
PARTES = f"{ce.DATASET}/_partes_{SUF}"
F_CUBO = f"{ce.DATASET}/features_cubo_{SUF}.parquet"
F_HIST = f"{ce.DATASET}/features_historia_{SUF}.parquet"
FINAL = f"{ce.DATASET}/dataset_{SUF}.parquet"
R_CELDA, R_ZONA, R_FIRMS = 1500.0, 10_000.0, 50_000.0


def carga_original():
    ruta = f"{config.FUENTE}/extraer_features_cubo.py"
    spec = importlib.util.spec_from_file_location("_efc", ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_efc"] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- cubo ----
def _bloque(args):
    key, df_b, t_idx, clim_idx_mes = args
    mod = carga_original()
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    try:
        return key, mod.procesar_bloque(ds, df_b, t_idx, clim_idx_mes)
    finally:
        ds.close()


def features_cubo(df):
    mod = carga_original()
    os.makedirs(PARTES, exist_ok=True)
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    tiempos = ds["time"].values.astype("datetime64[D]")
    t0 = tiempos[0].astype(int)
    anios = tiempos.astype("datetime64[Y]").astype(int) + 1970
    meses = (tiempos.astype("datetime64[M]").astype(int) % 12) + 1
    m_clim = (anios >= mod.ANIOS_CLIM[0]) & (anios <= mod.ANIOS_CLIM[1])
    clim_idx_mes = {m: np.where(m_clim & (meses == m))[0] for m in range(1, 13)}

    # estáticas: popdens del año, con 2020 como tope (el cubo acaba ahí)
    ruta_est = f"{PARTES}/estaticas.parquet"
    if not os.path.exists(ruta_est):
        d2 = df.copy()
        d2["anio"] = d2["anio"].clip(upper=2020)
        mod.extraer_estaticas(ds, d2).to_parquet(ruta_est, index=False)
        print("  estáticas listas", flush=True)
    ds.close()

    df = df.copy()
    df["_by"], df["_bx"] = df["iy"] // mod.CHUNK_Y, df["ix"] // mod.CHUNK_X
    t_of = df["fecha"].values.astype("datetime64[D]").astype(int) - t0
    tareas = []
    for (by, bx), g in df.groupby(["_by", "_bx"]):
        ruta = f"{PARTES}/bloque_{by:02d}_{bx:02d}.parquet"
        if os.path.exists(ruta):
            continue
        t_idx = dict(zip(g["id_muestra"].values,
                         t_of[g.index.values]))
        tareas.append(((by, bx), g, t_idx, clim_idx_mes))
    print(f"  bloques pendientes: {len(tareas)}", flush=True)
    t_ini = time.time()
    with ProcessPoolExecutor(8) as ex:
        for n, (key, parte) in enumerate(ex.map(_bloque, tareas), 1):
            by, bx = key
            parte.to_parquet(f"{PARTES}/bloque_{by:02d}_{bx:02d}.parquet",
                             index=False)
            if n % 10 == 0:
                print(f"  {n}/{len(tareas)} bloques · {time.time()-t_ini:.0f}s",
                      flush=True)
    partes = [pd.read_parquet(f"{PARTES}/{f}") for f in sorted(os.listdir(PARTES))
              if f.startswith("bloque_")]
    din = pd.concat(partes, ignore_index=True)
    est = pd.read_parquet(ruta_est)
    out = din.merge(est, on="id_muestra", validate="1:1")
    assert len(out) == len(df), (len(out), len(df))
    out.to_parquet(F_CUBO, index=False)
    print(f"  cubo: {out.shape}", flush=True)
    return out


# -------------------------------------------------------------- historia ----
G = {}


def _hist_chunk(celdas):
    ex, ey, e_dia, e_mes = G["ex"], G["ey"], G["e_dia"], G["e_mes"]
    f_dia, f_frp = G["f_dia"], G["f_frp"]
    wglc, w0 = G["wglc"], G["w0"]
    out = []
    for (cx, cy, lat_c, lon_c), filas in celdas:
        vz = G["arbol_egif"].query_ball_point([cx, cy], r=R_ZONA)
        vz_d, vz_m = e_dia[vz], e_mes[vz]
        dist_z = np.hypot(ex[vz] - cx, ey[vz] - cy)
        vc_d = vz_d[dist_z <= R_CELDA]
        vf = G["arbol_firms"].query_ball_point([cx, cy], r=R_FIRMS)
        vf_d, vf_frp = f_dia[vf], f_frp[vf]
        serie = wglc.sel(lat=lat_c, lon=lon_c, method="nearest").values
        for idm, d, mes, anio in filas:
            atras_90 = (vz_d >= d - 90) & (vz_d < d)
            atras_365 = (vz_d >= d - 365) & (vz_d < d)
            a0 = np.datetime64(f"{anio}-01-01", "D").astype(int)
            mismomes = (vz_m == mes) & (vz_d < a0)
            m_f = (vf_d >= d - 7) & (vf_d < d)
            frp7 = vf_frp[m_f]
            t_w = d - w0
            rd = float(serie[t_w]) if 0 <= t_w < len(serie) else np.nan
            r7 = serie[max(0, t_w - 7):t_w] if t_w > 0 else np.array([])
            out.append((idm,
                        int(((vc_d >= d - 90) & (vc_d < d)).sum()),
                        int(atras_90.sum()), int(atras_365.sum()),
                        int((vc_d < d).sum()), int(mismomes.sum()),
                        float(frp7.max()) if len(frp7) else 0.0, int(m_f.sum()),
                        rd, float(np.nansum(r7)) if len(r7) else np.nan))
    return out


def features_historia(df):
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    egif = pd.read_csv(config.EGIF, usecols=["fecha", "lat", "lng"])
    egif["fecha"] = pd.to_datetime(egif["fecha"], errors="coerce")
    egif = egif.dropna()
    egif = egif[(egif["fecha"].dt.year >= 2008) & (egif["fecha"].dt.year <= 2020)]
    G["ex"], G["ey"] = tr.transform(egif["lng"].values, egif["lat"].values)
    G["e_dia"] = egif["fecha"].values.astype("datetime64[D]").astype(int)
    G["e_mes"] = egif["fecha"].dt.month.values
    G["arbol_egif"] = cKDTree(np.column_stack([G["ex"], G["ey"]]))
    firms = pd.read_parquet(config.FIRMS, columns=["latitude", "longitude",
                                                   "acq_date", "frp"])
    fx, fy = tr.transform(firms["longitude"].values, firms["latitude"].values)
    G["f_dia"] = firms["acq_date"].values.astype("datetime64[D]").astype(int)
    G["f_frp"] = firms["frp"].values.astype(float)
    G["arbol_firms"] = cKDTree(np.column_stack([fx, fy]))
    w = xr.open_dataset(config.WGLC)
    G["wglc"] = w["density"].sel(lat=slice(35, 45), lon=slice(-10, 5)).load()
    G["w0"] = w["time"].values.astype("datetime64[D]").astype(int)[0]
    print(f"  EGIF {len(egif):,} · FIRMS {len(firms):,} · WGLC "
          f"{dict(G['wglc'].sizes)}", flush=True)

    d_m = df["fecha"].values.astype("datetime64[D]").astype(int)
    grupos = []
    for (ix_, iy_), g in df.groupby(["ix", "iy"], sort=False):
        r0 = g.iloc[0]
        grupos.append(((r0.x3035, r0.y3035, r0.lat_celda, r0.lon_celda),
                       list(zip(g["id_muestra"].values, d_m[g.index.values],
                                g["mes"].values, g["anio"].values))))
    print(f"  celdas únicas: {len(grupos):,}", flush=True)
    n_ch = 12 * 8
    chunks = [grupos[i::n_ch] for i in range(n_ch)]
    t_ini = time.time()
    out = []
    with ProcessPoolExecutor(12) as ex:          # fork: G compartido
        for n, r in enumerate(ex.map(_hist_chunk, chunks), 1):
            out.extend(r)
            if n % 12 == 0:
                print(f"  {n}/{n_ch} trozos · {time.time()-t_ini:.0f}s", flush=True)
    cols = ["id_muestra", "n_fuegos_1km_90d", "n_fuegos_10km_90d",
            "n_fuegos_10km_365d", "n_fuegos_1km_hist",
            "n_fuegos_10km_mismomes_hist", "frp_max_50km_7d", "n_detec_50km_7d",
            "rayos_dia", "rayos_7d"]
    res = pd.DataFrame(out, columns=cols)
    assert len(res) == len(df)
    res.to_parquet(F_HIST, index=False)
    print(f"  historia: {res.shape}", flush=True)
    return res


def vpd_kpa(t_c, rh):
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    return es * (1.0 - rh / 100.0)


def main():
    df = pd.read_parquet(MAESTRA)
    print(f"maestra: {df.shape}")
    cubo = pd.read_parquet(F_CUBO) if os.path.exists(F_CUBO) else features_cubo(df)
    hist = pd.read_parquet(F_HIST) if os.path.exists(F_HIST) else features_historia(df)
    out = df.merge(cubo, on="id_muestra", validate="1:1") \
            .merge(hist, on="id_muestra", validate="1:1")
    assert len(out) == len(df)
    out["vpd_max"] = vpd_kpa(out["t2m_max"], out["rh_min"])
    out["dia_anio"] = pd.DatetimeIndex(out["fecha"]).dayofyear
    # label_effis desde el cubo para TODAS las filas (en eval ya venía)
    out["label_effis"] = out["is_fire_dia"].fillna(0).astype(np.int8)
    amb = (out["disenio"] != "eval_dia") & (out["label"] == 0) & \
          (out["is_near_fire_dia"] == 1)
    print(f"negativos ambiguos (is_near_fire) eliminados: {amb.sum():,} "
          f"({amb.mean()*100:.2f} %)")
    print(out[amb].groupby("disenio").size().to_string())
    out = out[~amb].reset_index(drop=True)
    if SUF == "effis":
        # se sortearon 4 negativos por positivo para absorber el filtro; se
        # recorta a 3 por positivo (al azar) en cada diseño y split
        partes = []
        for (dis, sp), g in out.groupby(["disenio", "split"]):
            if dis == "eval_dia":
                partes.append(g); continue
            npos = (g["label"] == 1).sum()
            neg = g[g["label"] == 0]
            partes.append(pd.concat([g[g["label"] == 1],
                                     neg.sample(min(len(neg), 3 * npos), random_state=0)]))
        out = pd.concat(partes).reset_index(drop=True)
    out.to_parquet(FINAL, index=False)
    print(f"\nguardado {FINAL}: {out.shape}")
    print(out.groupby(["disenio", "split", "label"]).size().unstack(fill_value=0)
          .to_string())


if __name__ == "__main__":
    main()
