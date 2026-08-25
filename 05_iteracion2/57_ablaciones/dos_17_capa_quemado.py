#!/usr/bin/env python3
"""
Dos modelos — paso 17: ¿mejora el mapa una capa de "YA QUEMADO"? Evaluado
sobre julio-agosto de 2026 contra EFFIS, con los mapas diarios ya calculados.

NO TOCA PRODUCCIÓN. Lee mapas_2026 (disco externo) y el geojson de EFFIS;
escribe salida/dos_17_capa_quemado.json.

=============================================================================
QUÉ SE PRUEBA
=============================================================================
`ESTUDIO_COMBUSTIBLE_CONSUMIDO.md` midió que los modelos siguen dando
percentil ~72 a una celda dos semanas después de quemarse. Aquí se aplica una
capa "ya quemado" como POSTPROCESO del mapa (las celdas marcadas bajan al
fondo del ranking) y se mide si el mapa mejora o empeora. Variantes:

  capa       qué celdas se apagan
  effis_N    con perímetro EFFIS en los últimos N días (N = 7, 30, 60)
  effis80_N  ídem, pero solo las celdas quemadas en >80 % de su superficie
             (rasterizado a 100 m): las de borde siguen con combustible
  firms_N    con detección VIIRS en la propia celda en los últimos N días
             (es lo que habría en tiempo real sin esperar a EFFIS)

Dos verdades-terreno, porque la respuesta depende de qué se quiera acertar:
  todo       celdas con FIREDATE ese día (lo que juzga `puntuar_effis`)
  nuevos     solo perímetros que NO tocan (≤1 km) ningún perímetro de los
             30 días anteriores: igniciones nuevas, no crecimiento ni
             reactivación de un complejo ya en marcha.

Métricas por día: AUC dentro del día y lift del decil; diferencia pareada
con el mapa sin capa, IC95 por bootstrap de días. Mapas: producción en malla,
único EFFIS y pareja.
"""

import glob
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation

import config
import config_expansion as ce
from comparar_julio2026 import auc

GEO = config.salida("effis_ba_season_ES.geojson")
MODELOS = ["prod", "donde_dia_effis", "donde_effis_c×cuando_egif"]
NS = (7, 30, 60)


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es = ds["is_spain"].values.astype(bool)
    xs, ys = ds["x"].values, ds["y"].values
    ny, nx = es.shape
    g = gpd.read_file(GEO)
    g["fd"] = pd.to_datetime(g.FIREDATE, utc=True).dt.tz_localize(None).dt.normalize()
    g = g.set_crs(4326, allow_override=True).to_crs(3035)
    g = g[g.fd >= pd.Timestamp("2026-05-01")]
    files = sorted(glob.glob(f"{ce.DATASET}/mapas_2026/*.npz"))
    dias = [pd.Timestamp(os.path.basename(f)[:10]) for f in files]

    # --- máscaras diarias de EFFIS a 1 km (tocada) y >80 % quemada --------
    px = xs[1] - xs[0]
    tr1 = from_origin(xs[0] - px / 2, max(ys[0], ys[-1]) + px / 2, px, px)
    tr100 = from_origin(xs[0] - px / 2, max(ys[0], ys[-1]) + px / 2, 100, 100)
    Q, Q80 = {}, {}
    for d, gg in g.groupby("fd"):
        geoms = [x for x in gg.geometry if x is not None]
        m = rasterize(((x, 1) for x in geoms), out_shape=(ny, nx), transform=tr1,
                      fill=0, all_touched=True).astype(bool)
        f = rasterize(((x, 1) for x in geoms), out_shape=(ny * 10, nx * 10),
                      transform=tr100, fill=0, all_touched=False).astype(np.uint8)
        frac = f.reshape(ny, 10, nx, 10).mean((1, 3))
        if ys[1] > ys[0]:
            m, frac = m[::-1], frac[::-1]
        Q[d] = m & es
        Q80[d] = (frac > 0.8) & es
    print(f"  días con perímetro EFFIS: {len(Q)}", flush=True)

    # --- FIRMS por celda y día (caché _firms: [D-5, D-1] por fecha) --------
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    firms_dia = {}
    for f in glob.glob(config.salida("_firms/*.csv")):
        df = pd.read_csv(f)
        if not len(df):
            continue
        fx, fy = tr.transform(df["longitude"].values, df["latitude"].values)
        ix = np.rint((fx - xs[0]) / px).astype(int)
        iy = np.rint((fy - ys[0]) / (ys[1] - ys[0])).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        for d_, a, b in zip(pd.to_datetime(df["acq_date"].values[ok]), iy[ok], ix[ok]):
            firms_dia.setdefault(pd.Timestamp(d_).normalize(), np.zeros((ny, nx), bool))[a, b] = True
    print(f"  días con FIRMS en caché: {len(firms_dia)}", flush=True)

    def acumula(D, d, N):
        m = np.zeros((ny, nx), bool)
        for k in range(1, N + 1):
            dk = d - pd.Timedelta(days=k)
            if dk in D:
                m |= D[dk]
        return m

    capas = {f"effis_{N}": (lambda d, N=N: acumula(Q, d, N)) for N in NS}
    capas.update({f"effis80_{N}": (lambda d, N=N: acumula(Q80, d, N)) for N in NS})
    capas.update({f"firms_{N}": (lambda d, N=N: acumula(firms_dia, d, N)) for N in NS})

    filas = []
    for d, f in zip(dias, files):
        if d not in Q or Q[d].sum() == 0:
            continue
        q_todo = Q[d]
        # nuevos: perímetro que no toca (≤1 km) nada de los 30 días anteriores
        prev = binary_dilation(acumula(Q, d, 30), iterations=1)
        q_nuevo = q_todo & ~prev
        M = np.load(f)
        sel = es.ravel()                      # los mapas son vectores de España
        mascaras = {k: fn(d).ravel()[sel] for k, fn in capas.items()}
        q_todo, q_nuevo = q_todo.ravel()[sel], q_nuevo.ravel()[sel]
        for nom in MODELOS:
            v = M[nom].astype(float)
            ok = np.isfinite(v)
            umb = np.quantile(v[ok], 0.9)
            for verdad, q in (("todo", q_todo), ("nuevos", q_nuevo)):
                if q.sum() == 0:
                    continue
                fila = {"fecha": str(d.date()), "modelo": nom, "verdad": verdad,
                        "n_pos": int(q.sum()), "auc_base": auc(v[ok & q], v[ok & ~q]),
                        "lift_base": float((v[ok & q] >= umb).mean() / 0.1)}
                for k, m in mascaras.items():
                    v2 = v.copy()
                    v2[m] = np.nanmin(v) - 1
                    umb2 = np.quantile(v2[ok], 0.9)
                    fila[f"auc_{k}"] = auc(v2[ok & q], v2[ok & ~q])
                    fila[f"lift_{k}"] = float((v2[ok & q] >= umb2).mean() / 0.1)
                    fila[f"pos_apagados_{k}"] = float((m & q).sum() / q.sum())
                    fila[f"frac_mapa_{k}"] = float(m.mean())
                filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(config.salida("dos_17_capa_quemado.csv"), index=False)

    rng = np.random.default_rng(0)
    R = {}
    for verdad in ("todo", "nuevos"):
        print(f"\n== verdad: {verdad} ==")
        for nom in MODELOS:
            s = df[(df.modelo == nom) & (df.verdad == verdad)]
            R[f"{verdad}_{nom}"] = {"n_dias": int(len(s)), "n_pos": int(s.n_pos.sum()),
                                    "auc_base": float(s.auc_base.mean()),
                                    "lift_base": float(s.lift_base.mean()), "capas": {}}
            print(f"  {nom:26s} base AUC {s.auc_base.mean():.3f} lift {s.lift_base.mean():.2f} "
                  f"({len(s)} días, {int(s.n_pos.sum())} celdas)")
            for k in capas:
                dd = (s[f"auc_{k}"] - s.auc_base).values
                b = dd[rng.integers(0, len(dd), (3000, len(dd)))].mean(1)
                R[f"{verdad}_{nom}"]["capas"][k] = {
                    "dif_auc": float(dd.mean()),
                    "ic95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                    "dif_lift": float((s[f"lift_{k}"] - s.lift_base).mean()),
                    "pos_apagados": float(s[f"pos_apagados_{k}"].mean()),
                    "frac_mapa": float(s[f"frac_mapa_{k}"].mean())}
                print(f"      {k:10s} ΔAUC {dd.mean():+.3f} [{np.percentile(b,2.5):+.3f}, "
                      f"{np.percentile(b,97.5):+.3f}] · Δlift {(s[f'lift_{k}']-s.lift_base).mean():+.2f}"
                      f" · positivos apagados {s[f'pos_apagados_{k}'].mean()*100:4.1f} %"
                      f" · mapa apagado {s[f'frac_mapa_{k}'].mean()*100:.2f} %")
    json.dump(R, open(config.salida("dos_17_capa_quemado.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
