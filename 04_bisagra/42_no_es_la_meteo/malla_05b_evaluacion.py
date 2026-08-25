#!/usr/bin/env python3
"""
Módulo 5b: ¿discrimina la malla tan bien como el cubo? Varios días.

NO TOCA PRODUCCIÓN. Escribe dataset/malla_05b_evaluacion.json.

=============================================================================
POR QUÉ HACE FALTA ESTO
=============================================================================
El módulo 5 pinta un día y da una cifra de discriminación, pero ese día tenía
39 celdas con detección FIRMS dentro de España. Con 39 celdas, la diferencia
entre los dos mapas no se distingue del ruido, así que ese número NO decide
nada. Aquí se repite sobre varios días y se agrupan las celdas.

MÉTRICA. Para cada celda con detección FIRMS de ese día se calcula su
PERCENTIL dentro de la distribución de riesgo de España ESE día. Un mapa
perfecto pondría todos los fuegos en el percentil 100; uno inútil, en el 50.
Se agrupan todas las celdas de todos los días y se compara la mediana.

Es mejor que "el percentil de la mediana" del módulo 5 porque no colapsa el
día a un único número antes de comparar, y porque normaliza por día: los días
de peligro general alto no pesan más que los tranquilos.

DÍAS. Los de más detecciones en España dentro del rango con ERA5-Land en
disco (jun-sep 2024), evitando días consecutivos: 16, 17 y 18 de septiembre
son el MISMO episodio y contarlos como tres muestras infla la confianza.

SPIN-UP. La recursión del FWI arranca el 1-jun (§ módulo 5). Cuanto más
temprano el día evaluado, menos spin-up. Por eso el más temprano aquí es de
agosto (62 días) y el script mide e informa del sesgo real por día.

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python malla_05b_evaluacion.py
"""

import json

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

import config
from features import FULL
from malla_05_riesgo import (MODELO, a_celdas, compartidas, meteo_cubo,
                             meteo_malla)

ARMS = ("cubo", "malla", "malla_qm")
DIAS = ["2024-09-17", "2024-09-05", "2024-08-22", "2024-08-14",
        "2024-08-11", "2024-08-06"]


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    tiempos = ds["time"].values.astype("datetime64[D]")
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    es_esp = ds["is_spain"].values.astype(bool)
    xs, ys = ds["x"].values, ds["y"].values
    modelo = xgb.XGBClassifier()
    modelo.load_model(MODELO)
    sel = es_esp.ravel()

    filas, pool = [], {k: [] for k in ARMS}
    for fecha in DIAS:
        d = np.datetime64(fecha, "D")
        anio, mes = int(fecha[:4]), int(fecha[5:7])
        dia_anio = int((d - np.datetime64(f"{anio}-01-01", "D")).astype(int)) + 1
        t = int((d - tiempos[0]).astype(int))
        print(f"\n=== {fecha} ===", flush=True)

        Fn, idx_nodo, n_malo, _ = meteo_malla(d, mes, False)
        M = a_celdas(Fn, idx_nodo)
        Fq, _, _, _ = meteo_malla(d, mes, False, mapear=True)
        Q = a_celdas(Fq, idx_nodo)
        R = meteo_cubo(ds, t, mes, ny, nx)
        B, firms, fd, tr = compartidas(ds, d, t, anio, mes, dia_anio,
                                       ny, nx, xs, ys)

        def predice(MET):
            X = pd.DataFrame({k: (MET[k] if k in MET else B[k]).ravel()[sel]
                              for k in FULL})
            p = np.full(ny * nx, np.nan)
            p[sel] = modelo.predict_proba(X)[:, 1]
            return p.reshape(ny, nx)

        P = {"cubo": predice(R), "malla": predice(M),
             "malla_qm": predice(Q)}

        # celdas con detección FIRMS del propio día, dentro de España
        mD = fd == d
        fx, fy = tr.transform(firms.loc[mD, "longitude"].values,
                              firms.loc[mD, "latitude"].values)
        ix = np.rint((fx - xs[0]) / (xs[1] - xs[0])).astype(int)
        iy = np.rint((fy - ys[0]) / (ys[1] - ys[0])).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        det = np.zeros((ny, nx), bool)
        det[iy[ok], ix[ok]] = True
        det &= es_esp

        fila = {"fecha": fecha, "celdas_fuego": int(det.sum()),
                "nodos_respaldo": n_malo}
        for nom in ARMS:
            p = P[nom]
            todas = p[es_esp]; todas = todas[np.isfinite(todas)]
            fuego = p[det]; fuego = fuego[np.isfinite(fuego)]
            # percentil de CADA celda de fuego dentro del país ese día
            orden = np.sort(todas)
            pct = np.searchsorted(orden, fuego, side="right") / len(orden) * 100
            pool[nom].append(pct)
            fila[f"pctl_mediano_{nom}"] = float(np.median(pct))
            sat = {"cubo": R, "malla": M, "malla_qm": Q}[nom]["fwi_pctl_local"]
            v = sat[es_esp]; v = v[np.isfinite(v)]
            fila[f"sat999_{nom}"] = float((v >= 99.9).mean() * 100)
        fila["corr_prob"] = float(np.corrcoef(
            P["cubo"][es_esp & np.isfinite(P["cubo"]) & np.isfinite(P["malla"])],
            P["malla"][es_esp & np.isfinite(P["cubo"]) & np.isfinite(P["malla"])]
        )[0, 1])
        filas.append(fila)
        print(f"  fuego={fila['celdas_fuego']:3d} · pctl cubo "
              f"{fila['pctl_mediano_cubo']:5.1f} · malla "
              f"{fila['pctl_mediano_malla']:5.1f} · malla_qm "
              f"{fila['pctl_mediano_malla_qm']:5.1f}", flush=True)
        del P, M, R, Q, B, Fn, Fq

    ds.close()
    A = {k: np.concatenate(v) for k, v in pool.items()}

    print("\n" + "=" * 78)
    print("DISCRIMINACIÓN — percentil de riesgo de las celdas con fuego")
    print("=" * 78)
    print(f"{'fecha':<13}{'fuego':>7}{'cubo':>9}{'malla':>9}{'malla_qm':>10}"
          f"{'sat cubo':>10}{'sat malla':>11}{'sat qm':>9}")
    for f in filas:
        print(f"{f['fecha']:<13}{f['celdas_fuego']:>7}"
              f"{f['pctl_mediano_cubo']:>9.1f}{f['pctl_mediano_malla']:>9.1f}"
              f"{f['pctl_mediano_malla_qm']:>10.1f}"
              f"{f['sat999_cubo']:>10.2f}{f['sat999_malla']:>11.2f}"
              f"{f['sat999_malla_qm']:>9.2f}")
    print("-" * 78)
    print(f"{'AGRUPADO':<13}{len(A['cubo']):>7}"
          f"{np.median(A['cubo']):>9.1f}{np.median(A['malla']):>9.1f}"
          f"{np.median(A['malla_qm']):>10.1f}")

    # BOOTSTRAP POR DÍAS, no por celdas. Las celdas con fuego de un mismo día
    # pertenecen a los mismos incendios y no son independientes: remuestrearlas
    # da un intervalo demasiado estrecho. La unidad de muestreo real es el día.
    rng = np.random.default_rng(0)
    comp = {}
    print("\n  contra la referencia del cubo, remuestreando DÍAS:")
    for nom in ("malla", "malla_qm"):
        dif = np.array([f[f"pctl_mediano_{nom}"] - f["pctl_mediano_cubo"]
                        for f in filas])
        bs = np.array([dif[rng.integers(0, len(dif), len(dif))].mean()
                       for _ in range(20000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        if lo > 0:
            v = "gana"
        elif hi < 0:
            v = "pierde"
        elif abs(dif.mean()) < 3.0:
            v = "empate práctico"
        else:
            v = "indistinguible del ruido"
        print(f"    {nom:<10} media {dif.mean():+6.1f} pts · "
              f"IC95 [{lo:+.1f}, {hi:+.1f}] · gana {int((dif>0).sum())}/"
              f"{len(dif)} días → {v}")
        comp[nom] = dict(dif_media=float(dif.mean()),
                         ic95=[float(lo), float(hi)],
                         dias_ganados=int((dif > 0).sum()), veredicto=v)

    res = {"dias": filas, "n_celdas_fuego": int(len(A["cubo"])),
           "pctl_mediano": {k: float(np.median(A[k])) for k in ARMS},
           "vs_cubo": comp}
    with open(config.salida("evaluacion.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado: {config.salida('evaluacion.json')}")


if __name__ == "__main__":
    main()
