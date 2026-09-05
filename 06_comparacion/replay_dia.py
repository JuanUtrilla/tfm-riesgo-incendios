#!/usr/bin/env python3
"""Replay de UN día en condiciones de servicio, con los datos de archivo.

Reproduce lo que la cadena diaria habría publicado la mañana del día D:
  producción (riesgo_hoy.py, xgb_v2_prototipo) + candidatos (dos_riesgo_hoy.py:
  único 1:3, único 1:10, pareja, dónde, cuándo), y lo puntúa contra EFFIS.

Dos condiciones:
  ifs         reanálisis ERA5-Land hasta D−7, IFS archivado de D−6 a D  (servicio)
  reanalisis  reanálisis hasta el propio D                                (cota superior)

NO toca ningún .py de los repos: corre sobre una copia en sandbox_replay/
(salida propia) y parchea desde fuera: reanálisis truncado, `descarga` del IFS
sustituida por el archivo, `firms_nrt` por `comparar_rankings.firms_dia`
(ventana 7, cacheada desde los parquets de focos) y la fecha «hoy» de
riesgo_hoy.main. Las entradas vienen de archivo_ifs/ (ver PENDIENTE.md del
repo definitivo, «Replay de las temporadas 2025 y 2026»).

Uso:  python replay_dia.py --fecha 2025-08-13 --condicion ifs
Salida: replay/<temporada>/<condicion>/<fecha>.npz (prob_* de los 6 mapas),
        replay/replay_<temporada>_<condicion>.csv (una fila por día: AUC y
        percentil por modelo, celdas y ha quemadas, tiempos).
"""
import argparse, json, os, shutil, sys, time

AQUI = os.path.dirname(os.path.abspath(__file__))
SANDBOX = f"{AQUI}/sandbox_replay"
os.environ["TFM_FIRMS_DIAS"] = "7"          # la ventana del entrenamiento (oficial)
os.chdir(SANDBOX)
sys.path.insert(0, SANDBOX)

import numpy as np
import pandas as pd
import xarray as xr

FUENTES = {
    2025: dict(re=f"{AQUI}/era5land_diario_2025.nc",
               ifs=f"{AQUI}/ifs_archivo_2025.parquet",
               firms=f"{AQUI}/firms_2025/firms_iberia_2025.parquet",
               effis=f"{AQUI}/effis_ba_2025_ES.geojson"),
    2026: dict(re=f"{AQUI}/era5land_diario_2026.nc",
               ifs=f"{AQUI}/ifs_archivo_2026.parquet",
               firms=f"{AQUI}/firms_2026/firms_iberia_2026.parquet",
               effis=f"{AQUI}/effis_ba_2026_season_congelado_2026-09-02.geojson"),
}
MODELOS = ("prod", "unico", "r10", "pareja", "donde", "cuando")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fecha", required=True)
    p.add_argument("--condicion", choices=["ifs", "reanalisis"], default="ifs")
    p.add_argument("--sin-png", action="store_true", help="no dibujar (savefig anulado)")
    a = p.parse_args()
    D = pd.Timestamp(a.fecha)
    F = FUENTES[D.year]
    t0 = time.time(); T = {}

    import config
    import riesgo_hoy as rh
    import malla_02b_ifs as ifsmod
    import comparar_rankings as cr
    import dos_riesgo_hoy as dr
    from comparar_julio2026 import auc, quemadas
    assert cr.VENTANA_FIRMS == 7
    if a.sin_png:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.figure
        matplotlib.figure.Figure.savefig = lambda self, *k, **kw: None
        import capa_verdad
        capa_verdad.preparar = lambda ds, fecha: {"m7": np.zeros(ds["is_spain"].shape, bool),
            "m1": np.zeros(ds["is_spain"].shape, bool), "fx": np.array([]), "fy": np.array([]),
            "mx": np.array([]), "my": np.array([]), "mn": np.array([]), "effis": []}

    # --- 1. reanálisis truncado ------------------------------------------
    corte = D - pd.Timedelta(days=7) if a.condicion == "ifs" else D
    ds = xr.open_dataset(F["re"], decode_timedelta=False)
    f_re = pd.to_datetime(ds["fecha"].values)
    keep = np.where(f_re <= corte)[0]
    assert f_re[keep][-1] == corte, f"el reanálisis no llega al {corte.date()}"
    trunc = config.salida(f"era5land_trunc_{a.condicion}_{D.date()}.nc")
    ds.isel({ds["fecha"].dims[0]: keep}).to_netcdf(trunc); ds.close()
    rh.REANALISIS = trunc

    # --- 2. IFS del archivo: de corte+1 a D+1 (D+1 solo para la condición
    # reanálisis, donde series_nodos exige que el IFS aporte algo) ---------
    ifs = pd.read_parquet(F["ifs"])
    ifs["fecha"] = pd.to_datetime(ifs["fecha"])
    ifs = ifs[(ifs["fecha"] > corte) & (ifs["fecha"] <= D + pd.Timedelta(days=1))]
    assert ifs["fecha"].min() == corte + pd.Timedelta(days=1)
    ifs.to_parquet(ifsmod.ruta_cache(str(D.date())), index=False)
    ifsmod.descarga = lambda *k, **kw: ifs.copy()

    # --- 3. FIRMS [D−7, D−1] desde el parquet de focos → caché de firms_dia -
    fo = pd.read_parquet(F["firms"], columns=["latitude", "longitude", "acq_date", "frp"])
    fo["acq_date"] = pd.to_datetime(fo["acq_date"])
    fo = fo[(fo["acq_date"] >= D - pd.Timedelta(days=7)) & (fo["acq_date"] < D)]
    os.makedirs(config.salida("_firms7"), exist_ok=True)
    fo.to_csv(config.salida(f"_firms7/{D:%Y-%m-%d}.csv"), index=False)
    rh.firms_nrt = lambda dsx: cr.firms_dia(D, dsx)
    T["preparar"] = time.time() - t0

    # --- 4. producción (riesgo_hoy.main con «hoy» = D) --------------------
    class _TS(pd.Timestamp):
        @classmethod
        def utcnow(cls): return pd.Timestamp(D, tz="UTC")
    class _PD:
        Timestamp = _TS
        def __getattr__(self, k): return getattr(pd, k)
    rh.pd = _PD()
    dias = [0] if a.condicion == "ifs" else [0, 1]
    class A: pass
    A.dias = dias; A.pasada = str(D.date())
    t1 = time.time(); rh.main(A); T["prod"] = time.time() - t1
    # --- 5. candidatos ----------------------------------------------------
    t1 = time.time(); dr.main(A); T["candidatos"] = time.time() - t1

    # --- 6. recoger mapas y puntuar contra EFFIS --------------------------
    fstr = str(D.date())
    m = np.load(config.salida(f"mapas_diarios/dos_riesgo_{fstr}.npz"))
    out = {f"prob_{k}": m[f"prob_{k}"] for k in MODELOS if k != "prod"}
    out["prob_prod"] = np.load(config.salida(f"riesgo_hoy_{fstr}.npz"))["prob"].astype(np.float16)
    dest = f"{AQUI}/replay/{D.year}/{a.condicion}"
    os.makedirs(dest, exist_ok=True)
    np.savez_compressed(f"{dest}/{fstr}.npz", **out)
    shutil.copy2(config.salida(f"dos_riesgo_{fstr}.json"), f"{dest}/{fstr}.json")
    for png in (f"dos_riesgo_{fstr}.png", f"riesgo_hoy_{fstr}.png"):
        if os.path.exists(config.salida(png)):
            shutil.move(config.salida(png), f"{dest}/{png}")

    cubo = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es_esp = cubo["is_spain"].values.astype(bool)
    V = np.load(f"{AQUI}/replay/verdad_{D.year}/{fstr}.npz")
    celda, fuego, area, ncel = V["celda"], V["fuego"], V["area_ha"], V["n_celdas"]
    mask = np.zeros(es_esp.size, bool); mask[celda] = True
    fila = dict(fecha=fstr, condicion=a.condicion, celdas_quemadas=int(mask.sum()),
                n_incendios=int(len(area)), n_incendios_100ha=int((area >= 100).sum()),
                area_ha=float(area.sum()))
    sel = es_esp.ravel()
    # peso por celda = ha del incendio / celdas del incendio (ponderación por hectáreas)
    peso = np.zeros(es_esp.size); 
    if len(celda): peso[celda] = area[fuego] / ncel[fuego]
    for k in MODELOS:
        pm = out[f"prob_{k}"].astype(float).ravel()
        ok = sel & np.isfinite(pm)
        pos_idx = np.flatnonzero(mask & ok)
        neg = pm[~mask & ok]
        if len(pos_idx):
            pos = pm[pos_idx]
            fila[f"auc_{k}"] = auc(pos, neg)
            pc = np.array([(neg < v).mean() * 100 for v in pos])
            fila[f"pctl_{k}"] = float(np.median(pc))
            w = peso[pos_idx]; fila[f"pctl_ha_{k}"] = float((pc * w).sum() / w.sum()) if w.sum() > 0 else np.nan
            # captura en el 2 % más alto del mapa del día
            n2 = int(ok.sum() * 0.02)
            umbral = np.partition(pm[ok], -n2)[-n2]
            top = pm >= umbral
            fila[f"top2_celdas_{k}"] = float(top[pos_idx].mean() * 100)
            fila[f"top2_ha_{k}"] = float((w * top[pos_idx]).sum() / w.sum() * 100) if w.sum() > 0 else np.nan
            grandes = np.flatnonzero(area >= 100)
            if len(grandes):
                acierta = [top[celda[fuego == f]].any() for f in grandes]
                fila[f"top2_inc100_{k}"] = float(np.mean(acierta) * 100)
            else:
                fila[f"top2_inc100_{k}"] = np.nan
        else:
            for m in ("auc", "pctl", "pctl_ha", "top2_celdas", "top2_ha", "top2_inc100"):
                fila[f"{m}_{k}"] = np.nan
        # nivel absoluto (para la pregunta del «si»): nº de celdas por encima de
        # varios umbrales fijos de probabilidad, y estadísticos del mapa
        fila[f"p_med_{k}"] = float(np.nanmedian(pm[ok])); fila[f"p_p98_{k}"] = float(np.nanpercentile(pm[ok], 98))
        fila[f"p_max_{k}"] = float(np.nanmax(pm[ok])); fila[f"p_sum_{k}"] = float(np.nansum(pm[ok]))
    T["puntuar"] = time.time() - t0 - sum(T.values())
    T["total"] = time.time() - t0
    fila.update({f"t_{k}": round(v, 1) for k, v in T.items()})
    csv = f"{AQUI}/replay/replay_{D.year}_{a.condicion}.csv"
    prev = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    df = pd.concat([prev[prev["fecha"] != fstr] if len(prev) else prev,
                    pd.DataFrame([fila])], ignore_index=True).sort_values("fecha")
    df.to_csv(csv, index=False)

    # --- 7. limpiar el sandbox del día (los mapas ya están en replay/) ----
    for f in (trunc, ifsmod.ruta_cache(fstr), config.salida(f"riesgo_hoy_{fstr}.npz"),
              config.salida(f"dos_riesgo_{fstr}.json"),
              config.salida(f"mapas_diarios/dos_riesgo_{fstr}.npz"),
              config.salida(f"mapas_diarios/riesgo_hoy_{fstr}.npz")):
        if os.path.exists(f): os.remove(f)
    print("\n=== replay", fstr, a.condicion, "===")
    print(f"  quemadas {fila['celdas_quemadas']} celdas · {fila['area_ha']:,.0f} ha · {fila['n_incendios']} incendios")
    for k in MODELOS:
        print(f"  {k:8s} AUC {fila[f'auc_{k}']:.3f} · pctl {fila[f'pctl_{k}']:.1f} · top2 ha {fila[f'top2_ha_{k}']:.0f} %")
    print("  tiempos:", {k: round(v) for k, v in T.items()}, flush=True)


if __name__ == "__main__":
    main()
