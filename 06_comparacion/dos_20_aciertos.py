#!/usr/bin/env python3
"""
Dos modelos — paso 20: ¿de verdad arde donde el mapa dice que va a arder?

dos_14 midió el reparto por NIVEL (BAJO/MODERADO/ALTO/EXTREMO). Esto es la
misma pregunta pero mirada como la mira un jefe de guardia: si hoy solo puedo
vigilar el 1 % del territorio, ¿cuánto de lo que arde cae dentro?

Para cada uno de los 74 días de la temporada 2026 (mapas de dos_09, verdad
EFFIS por FIREDATE) y para cada mapa candidato se calcula, en varios cortes
top-k del día:

  cobertura  = celdas quemadas dentro del top-k / celdas quemadas del día
  precisión  = celdas quemadas dentro del top-k / celdas del top-k
  lift       = precisión / prevalencia del día
  dias_pilla = % de días con fuego en que el top-k atrapa al menos una celda

Agregado POOLED (suma de celdas sobre los 74 días), que es lo que se puede
enseñar: «el 1 % más alto concentró el X % de lo quemado». IC por bootstrap
de días (los días son la unidad independiente, no las celdas).

NO TOCA PRODUCCIÓN. Solo lee. Escribe salida/dos_20_aciertos.{json,csv}.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

import config
import config_expansion as ce

TOPK = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
MAPAS = ["prod", "donde_dia_effis", "donde_dia_effis_r10",
         "donde_effis_c×cuando_egif", "donde_dia", "fwi_pctl",
         "prod_sin_firms", "donde_dia_effis_sin_firms", "d×c_sin_firms"]
GEOJSON = config.salida("effis_ba_season_ES.geojson")
N_BOOT = 2000
SEMILLA = 20260831


def verdad_por_dia(ds, es):
    """{fecha -> máscara plana de celdas quemadas ese día}, leyendo EFFIS una vez."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    g = gpd.read_file(GEOJSON)
    col = "FIREDATE" if "FIREDATE" in g.columns else "firedate"
    g["f"] = (pd.to_datetime(g[col], errors="coerce", utc=True)
                .dt.tz_localize(None).dt.normalize())
    g = g.set_crs(4326, allow_override=True).to_crs(3035)
    ny, nx = es.shape
    xs, ys = ds["x"].values, ds["y"].values
    px, ay = xs[1] - xs[0], abs(ys[1] - ys[0])
    norte = max(ys[0], ys[-1]) + ay / 2
    tr = from_origin(xs[0] - px / 2, norte, px, ay)
    plano = es.ravel()
    out = {}
    for f, sel in g.groupby("f"):
        m = rasterize(((geom, 1) for geom in sel.geometry if geom is not None),
                      out_shape=(ny, nx), transform=tr, fill=0,
                      all_touched=True).astype(bool)
        if ys[1] > ys[0]:
            m = m[::-1]
        out[f] = (m & es).ravel()[plano]
    return out


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es = ds["is_spain"].values.astype(bool)
    verdad = verdad_por_dia(ds, es)
    ds.close()

    ficheros = sorted(glob.glob(f"{ce.DATASET}/mapas_2026/*.npz"))
    # acumuladores por (mapa, k): filas = un día con fuego
    filas = []
    n_dias = n_fuego = 0
    for ruta in ficheros:
        dia = pd.Timestamp(os.path.basename(ruta)[:10])
        n_dias += 1
        q = verdad.get(dia)
        if q is None or not q.any():
            continue
        n_fuego += 1
        m = np.load(ruta)
        nq = int(q.sum())
        for nom in MAPAS:
            if nom not in m.files:
                continue
            v = m[nom]
            ok = np.isfinite(v)
            n = int(ok.sum())
            orden = np.argsort(-v[ok], kind="stable")   # de mayor a menor riesgo
            qo = q[ok][orden]
            acum = np.cumsum(qo)
            for k in TOPK:
                nk = max(1, int(round(k * n)))
                dentro = int(acum[nk - 1])
                filas.append(dict(fecha=dia.date().isoformat(), mapa=nom, k=k,
                                  n_celdas=n, n_top=nk, n_quemadas=nq,
                                  dentro=dentro))
    df = pd.DataFrame(filas)
    df.to_csv(config.salida("dos_20_aciertos.csv"), index=False)

    rng = np.random.default_rng(SEMILLA)
    dias = sorted(df["fecha"].unique())
    idx = {d: i for i, d in enumerate(dias)}
    boot = rng.integers(0, len(dias), size=(N_BOOT, len(dias)))

    R = {"n_dias": n_dias, "n_dias_con_fuego": n_fuego, "topk": TOPK, "mapas": {}}
    for nom, g in df.groupby("mapa"):
        R["mapas"][nom] = {}
        for k, gk in g.groupby("k"):
            gk = gk.set_index("fecha").reindex(dias)
            dentro = gk["dentro"].values.astype(float)
            quem = gk["n_quemadas"].values.astype(float)
            ntop = gk["n_top"].values.astype(float)
            ncel = gk["n_celdas"].values.astype(float)
            cob = dentro.sum() / quem.sum()
            prec = dentro.sum() / ntop.sum()
            prev = quem.sum() / ncel.sum()
            b = np.array([dentro[i].sum() / max(quem[i].sum(), 1) for i in boot])
            R["mapas"][nom][f"top_{k*100:g}%"] = dict(
                cobertura=float(cob),
                ic95_cobertura=[float(np.percentile(b, 2.5)),
                                float(np.percentile(b, 97.5))],
                precision=float(prec),
                lift=float(prec / prev),
                celdas_top_dia=float(ntop.mean()),
                dias_pilla=float((dentro > 0).mean()),
                quemadas_por_1000_celdas_top=float(1000 * prec))
    R["prevalencia"] = float(df.groupby("fecha")["n_quemadas"].first().sum()
                             / df.groupby("fecha")["n_celdas"].first().sum())
    with open(config.salida("dos_20_aciertos.json"), "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False)

    print(f"{n_dias} días · {n_fuego} con fuego EFFIS · "
          f"prevalencia {R['prevalencia']*100:.4f} % de celdas-día\n")
    for k in TOPK:
        key = f"top_{k*100:g}%"
        print(f"--- top {k*100:g} % del día "
              f"(~{R['mapas'][MAPAS[0]][key]['celdas_top_dia']:.0f} celdas)")
        print(f"{'mapa':28s} {'cobertura':>10s} {'IC95':>18s} "
              f"{'lift':>6s} {'días pilla':>11s}")
        for nom in MAPAS:
            if nom not in R["mapas"]:
                continue
            d = R["mapas"][nom][key]
            print(f"{nom:28s} {d['cobertura']*100:9.1f}% "
                  f"[{d['ic95_cobertura'][0]*100:5.1f},{d['ic95_cobertura'][1]*100:5.1f}] "
                  f"{d['lift']:6.1f} {d['dias_pilla']*100:10.0f}%")
        print()
    figura(R)


def figura(R):
    """Curva cobertura vs. top-k: cuánto de lo quemado atrapas vigilando k %."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ver = ["prod", "prod_sin_firms", "donde_dia_effis", "donde_dia_effis_r10",
           "donde_effis_c×cuando_egif", "fwi_pctl"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = [k * 100 for k in TOPK]
    for nom in ver:
        if nom not in R["mapas"]:
            continue
        ys = [R["mapas"][nom][f"top_{k*100:g}%"]["cobertura"] * 100 for k in TOPK]
        ax.plot(xs, ys, marker="o", ms=4, label=nom)
    ax.plot(xs, xs, "k--", lw=1, label="azar")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}%" for x in xs])
    ax.set_xlabel("% del territorio vigilado (celdas de mayor riesgo del día)")
    ax.set_ylabel("% de la superficie quemada que cae dentro")
    ax.set_title(f"¿Arde donde el mapa dice? — {R['n_dias_con_fuego']} días, "
                 f"temporada 2026 (EFFIS)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.salida("dos_20_aciertos.png"), dpi=130)


if __name__ == "__main__":
    main()
