#!/usr/bin/env python3
"""
Dos modelos — paso 21: lo mismo que dos_20, pero ponderando por HECTÁREAS y
separando fuegos grandes de pequeños.

dos_20 contaba celdas quemadas: un conato de 1 ha y el incendio de 43.772 ha
pesaban lo mismo por celda. Pero en la temporada 2026 de EFFIS los 35 fuegos
de ≥500 ha son el 92 % de la superficie quemada y los 320 conatos de <30 ha
el 0,9 %. Ordenar bien los conatos no sirve de nada operativamente; fallar un
fuego grande se paga entero. Así que aquí:

  · cada incendio se rasteriza POR SEPARADO y sus hectáreas se reparten entre
    sus celdas (ha_celda = AREA_HA / nº de celdas del incendio), de modo que
    la suma sobre celdas devuelve la superficie real;
  · «cobertura_ha» = hectáreas dentro del top-k / hectáreas del día;
  · las tres clases (PEQUEÑO <30 ha, MEDIO 30-500, GRANDE ≥500) se miden por
    separado, en celdas y en incendios capturados;
  · «incendios pillados» = % de incendios de la clase con AL MENOS una celda
    dentro del top-k, que es la pregunta de guardia de verdad: ¿lo tenía yo
    en el radar ese día?

IC95 por bootstrap de DÍAS (unidad independiente). NO TOCA PRODUCCIÓN.
Escribe salida/dos_21_hectareas.{json,csv,png}.
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
MAPAS = ["prod", "prod_sin_firms", "donde_dia_effis", "donde_dia_effis_r10",
         "donde_effis_c×cuando_egif", "donde_dia", "fwi_pctl"]
CLASES = [("PEQUEÑO", 0, 30), ("MEDIO", 30, 500), ("GRANDE", 500, np.inf)]
GEOJSON = config.salida("effis_ba_season_ES.geojson")
N_BOOT = 2000
SEMILLA = 20260831


def clase(ha):
    for nom, lo, hi in CLASES:
        if lo <= ha < hi:
            return nom
    return CLASES[0][0]


def verdad_por_dia(ds, es):
    """{fecha -> (ha_celda, {clase -> máscara})} rasterizando incendio a incendio."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    g = gpd.read_file(GEOJSON)
    col = "FIREDATE" if "FIREDATE" in g.columns else "firedate"
    g["f"] = (pd.to_datetime(g[col], errors="coerce", utc=True)
                .dt.tz_localize(None).dt.normalize())
    g["ha"] = pd.to_numeric(g["AREA_HA"], errors="coerce").fillna(0.0)
    g = g.set_crs(4326, allow_override=True).to_crs(3035)
    ny, nx = es.shape
    xs, ys = ds["x"].values, ds["y"].values
    px, ay = xs[1] - xs[0], abs(ys[1] - ys[0])
    tr = from_origin(xs[0] - px / 2, max(ys[0], ys[-1]) + ay / 2, px, ay)
    voltear = ys[1] > ys[0]
    plano = es.ravel()
    n = int(plano.sum())
    out, incendios = {}, []
    for f, sel in g.groupby("f"):
        ha_celda = np.zeros(n)
        masc = {c[0]: np.zeros(n, bool) for c in CLASES}
        for _, fila in sel.iterrows():
            if fila.geometry is None:
                continue
            m = rasterize([(fila.geometry, 1)], out_shape=(ny, nx), transform=tr,
                          fill=0, all_touched=True).astype(bool)
            if voltear:
                m = m[::-1]
            celdas = (m & es).ravel()[plano]
            k = int(celdas.sum())
            if k == 0:                      # polígono más pequeño que la celda
                continue
            cl = clase(fila.ha)
            ha_celda[celdas] += fila.ha / k
            masc[cl] |= celdas
            incendios.append(dict(fecha=f, ha=fila.ha, clase=cl,
                                  celdas=np.flatnonzero(celdas)))
        if ha_celda.sum() > 0 or any(v.any() for v in masc.values()):
            out[f] = (ha_celda, masc)
    return out, incendios


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es = ds["is_spain"].values.astype(bool)
    verdad, incendios = verdad_por_dia(ds, es)
    ds.close()
    por_dia = {}
    for inc in incendios:
        por_dia.setdefault(inc["fecha"], []).append(inc)

    filas, cap = [], []
    for ruta in sorted(glob.glob(f"{ce.DATASET}/mapas_2026/*.npz")):
        dia = pd.Timestamp(os.path.basename(ruta)[:10])
        if dia not in verdad:
            continue
        ha_celda, masc = verdad[dia]
        m = np.load(ruta)
        for nom in MAPAS:
            if nom not in m.files:
                continue
            v = m[nom]
            ok = np.isfinite(v)
            n = int(ok.sum())
            orden = np.argsort(-v[ok], kind="stable")
            pos = np.empty(n, int)                 # rango de cada celda (0 = más alto)
            pos[orden] = np.arange(n)
            rango = np.full(len(v), n, int)
            rango[ok] = pos
            for k in TOPK:
                nk = max(1, int(round(k * n)))
                dentro = rango < nk
                fila = dict(fecha=dia.date().isoformat(), mapa=nom, k=k,
                            n_celdas=n, n_top=nk,
                            ha_dia=float(ha_celda.sum()),
                            ha_dentro=float(ha_celda[dentro[ok]].sum()))
                for c, _, _ in CLASES:
                    q = masc[c]
                    fila[f"celdas_{c}"] = int(q.sum())
                    fila[f"dentro_{c}"] = int((q & dentro[ok]).sum())
                filas.append(fila)
            for inc in por_dia.get(dia, []):
                r = rango[ok][inc["celdas"]]
                for k in TOPK:
                    nk = max(1, int(round(k * n)))
                    cap.append(dict(fecha=dia.date().isoformat(), mapa=nom, k=k,
                                    clase=inc["clase"], ha=inc["ha"],
                                    pillado=int((r < nk).any()),
                                    frac=float((r < nk).mean())))
    df = pd.DataFrame(filas)
    dc = pd.DataFrame(cap)
    df.to_csv(config.salida("dos_21_hectareas.csv"), index=False)
    dc.to_csv(config.salida("dos_21_incendios.csv"), index=False)

    rng = np.random.default_rng(SEMILLA)
    dias = sorted(df["fecha"].unique())
    boot = rng.integers(0, len(dias), size=(N_BOOT, len(dias)))

    R = {"n_dias": len(dias), "topk": TOPK,
         "clases": {c: dict(n_incendios=int((dc[(dc.mapa == MAPAS[0]) &
                                               (dc.k == TOPK[0])].clase == c).sum()),
                            ha=float(dc[(dc.mapa == MAPAS[0]) & (dc.k == TOPK[0]) &
                                        (dc.clase == c)].ha.sum()))
                    for c, _, _ in CLASES},
         "mapas": {}}
    for nom, g in df.groupby("mapa"):
        R["mapas"][nom] = {}
        for k, gk in g.groupby("k"):
            gk = gk.set_index("fecha").reindex(dias)
            dd, hd = gk["ha_dentro"].values, gk["ha_dia"].values
            b = np.array([dd[i].sum() / max(hd[i].sum(), 1e-9) for i in boot])
            d = dict(cobertura_ha=float(dd.sum() / hd.sum()),
                     ic95_ha=[float(np.percentile(b, 2.5)),
                              float(np.percentile(b, 97.5))])
            for c, _, _ in CLASES:
                cel = gk[f"celdas_{c}"].values.sum()
                d[f"cobertura_celdas_{c}"] = float(gk[f"dentro_{c}"].values.sum()
                                                   / cel) if cel else float("nan")
                sub = dc[(dc.mapa == nom) & (dc.k == k) & (dc.clase == c)]
                d[f"incendios_pillados_{c}"] = float(sub.pillado.mean()) if len(sub) else float("nan")
                d[f"n_incendios_{c}"] = int(len(sub))
            R["mapas"][nom][f"top_{k*100:g}%"] = d
    with open(config.salida("dos_21_hectareas.json"), "w") as f:
        json.dump(R, f, indent=1, ensure_ascii=False)

    print(f"{len(dias)} días · " + " · ".join(
        f"{c}: {R['clases'][c]['n_incendios']} incendios / "
        f"{R['clases'][c]['ha']:,.0f} ha" for c, _, _ in CLASES) + "\n")
    for k in TOPK:
        key = f"top_{k*100:g}%"
        print(f"--- top {k*100:g} % del día")
        print(f"{'mapa':28s} {'cob.HA':>8s} {'IC95':>14s} "
              + "".join(f"{'pilla ' + c:>16s}" for c, _, _ in CLASES))
        for nom in MAPAS:
            if nom not in R["mapas"]:
                continue
            d = R["mapas"][nom][key]
            print(f"{nom:28s} {d['cobertura_ha']*100:7.1f}% "
                  f"[{d['ic95_ha'][0]*100:5.1f},{d['ic95_ha'][1]*100:5.1f}]"
                  + "".join(f"{d[f'incendios_pillados_{c}']*100:15.0f}%"
                            for c, _, _ in CLASES))
        print()
    figura(R)


def figura(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ver = ["prod", "prod_sin_firms", "donde_dia_effis", "donde_dia_effis_r10",
           "donde_effis_c×cuando_egif", "fwi_pctl"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xs = [k * 100 for k in TOPK]
    for ax, (campo, tit) in zip(axes, [
            ("cobertura_ha", "Hectáreas quemadas dentro del top-k"),
            ("incendios_pillados_GRANDE",
             f"Incendios ≥500 ha con alguna celda dentro "
             f"(n={R['clases']['GRANDE']['n_incendios']})")]):
        for nom in ver:
            if nom not in R["mapas"]:
                continue
            ys = [R["mapas"][nom][f"top_{k*100:g}%"][campo] * 100 for k in TOPK]
            ax.plot(xs, ys, marker="o", ms=4, label=nom)
        ax.plot(xs, xs, "k--", lw=1, label="azar")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}%" for x in xs])
        ax.set_xlabel("% del territorio vigilado")
        ax.set_ylabel("%")
        ax.set_title(tit, fontsize=10)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Ponderado por hectáreas — temporada 2026 (EFFIS)")
    fig.tight_layout()
    fig.savefig(config.salida("dos_21_hectareas.png"), dpi=130)


if __name__ == "__main__":
    main()
