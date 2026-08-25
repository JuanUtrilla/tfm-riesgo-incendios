#!/usr/bin/env python3
"""
La configuración REAL de producción: reanálisis para el historial + IFS para
los últimos días y D+1.  (§6ter)

NO SOBRESCRIBE NADA. Escribe dataset/prueba_hibrido_produccion.json.

Las dos cotas ya están medidas contra el cubo (jun-sep 2024, 120 estaciones):

    ERA5 puro (todo reanálisis)  corr 0,976 · sesgo −2,01 · pctl ≥99,9 = 0,8 %
    IFS puro  (todo previsión)   corr 0,935 · sesgo +0,41 · pctl ≥99,9 = 2,4 %
    (referencia CUBO                                        pctl ≥99,9 = 0,2 %)

El IFS puro NO vale: satura casi tanto como AEMET (3,3 %). Pero producción
nunca correría con IFS puro — el reanálisis llega hasta D−6 y la previsión solo
cubre la cola. Como el FWI es recursivo con memoria larga (DC ~52 días, DMC
~12 d, FFMC ~1 d), el estado en D lo fija casi todo el reanálisis y la
previsión solo mueve el FFMC/ISI.

Esta prueba simula el híbrido EXACTAMENTE, no por aproximación:

  1. Se corre el FWI con ERA5 sobre toda la serie guardando el ESTADO
     (ffmc, dmc, dc) de cada día.
  2. Para cada día de evaluación D se retoma el estado del día D−L y se
     avanzan L días con la meteo del IFS.

Se barre L = 0 (ERA5 puro), 2, 4, 6, 8 días para ver cuánta cola de previsión
aguanta el esquema antes de degradarse. L=6 es el retraso real de ERA5-Land
medido el 19/08/2026; los demás sirven de sensibilidad.

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python prueba_hibrido_produccion.py
"""

import json

import numpy as np
import pandas as pd

from fwi_canadiense import _bui, _dc, _dmc, _ffmc, _fwi, _isi, calcular_fwi_serie
from prueba_era5_produccion import DIR, EVAL, FIN, SPIN, estaciones, serie_openmeteo
from prueba_era5_atribucion import N, cubo_cacheado
from prueba_ifs_produccion import serie_ifs

LAGS = [0, 2, 4, 6, 8]


def paso(t, h, w, p, mes, est):
    """Un día del FWI a partir de un estado (ffmc, dmc, dc)."""
    ffmc, dmc, dc = est
    if np.isnan(t) or np.isnan(h):
        return est, np.nan
    w = 0.0 if np.isnan(w) else w
    p = 0.0 if np.isnan(p) else p
    h = min(h, 100.0)
    ffmc = _ffmc(t, h, w, p, ffmc)
    dmc = _dmc(t, h, p, dmc, mes)
    dc = _dc(t, p, dc, mes)
    return (ffmc, dmc, dc), _fwi(_isi(w, ffmc), _bui(dmc, dc))


def main():
    est = estaciones(N)
    fechas = pd.date_range(SPIN, FIN, freq="D")
    meses = fechas.month.values
    m_ev = np.where((fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1])
                    & (fechas.year == 2024))[0]
    print(f"estaciones {len(est)} | evaluación {len(m_ev)} días jun-sep 2024\n")

    cubo = cubo_cacheado(est, fechas)
    om, ifs = serie_openmeteo(est), serie_ifs(est)

    def matriz(df, c):
        return np.column_stack([
            df[df.idema == i].set_index("fecha").reindex(fechas)[c].values
            for i in est["idema"]])

    C = {"tmax": cubo["t2m_max"], "hr_min": cubo["RH_min"],
         "viento_max": cubo["wind_speed_max"],
         "prec": cubo["total_precipitation_mean"]}
    E = {c: matriz(om, c) for c in C}
    F = {c: matriz(ifs, c) for c in C}

    # referencia: el cubo
    ref = np.full((len(fechas), len(est)), np.nan)
    for j in range(len(est)):
        ref[:, j] = calcular_fwi_serie(C["tmax"][:, j], C["hr_min"][:, j],
                                       C["viento_max"][:, j] * 3.6,
                                       C["prec"][:, j], meses)["fwi"]

    clim = {i: np.load(f"{DIR}/prototipo/clim_fwi/{i}.npz") for i in est["idema"]}
    res, filas = {"lags": {}}, []

    for L in LAGS:
        M = np.full((len(fechas), len(est)), np.nan)
        for j in range(len(est)):
            s = calcular_fwi_serie(E["tmax"][:, j], E["hr_min"][:, j],
                                   E["viento_max"][:, j] * 3.6,
                                   E["prec"][:, j], meses)
            if L == 0:
                M[:, j] = s["fwi"]
                continue
            for k in m_ev:                       # retomar en k−L y avanzar L días
                e = (s["ffmc"][k - L], s["dmc"][k - L], s["dc"][k - L])
                if not np.all(np.isfinite(e)):
                    continue
                v = np.nan
                for q in range(k - L + 1, k + 1):
                    e, v = paso(F["tmax"][q, j], F["hr_min"][q, j],
                                F["viento_max"][q, j] * 3.6, F["prec"][q, j],
                                meses[q], e)
                M[k, j] = v

        a, b = ref[m_ev].ravel(), M[m_ev].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        p = []
        for j, i in enumerate(est["idema"]):
            for k in m_ev:
                v = M[k, j]
                if np.isfinite(v):
                    c = np.sort(clim[i][f"m{meses[k]}"][
                        ~np.isnan(clim[i][f"m{meses[k]}"])])
                    if len(c):
                        p.append(np.searchsorted(c, v, side="right") / len(c) * 100)
        p = np.array(p)
        r = dict(corr=float(np.corrcoef(a[ok], b[ok])[0, 1]),
                 sesgo=float(b[ok].mean() - a[ok].mean()),
                 maximo=float(np.nanmax(b)), pctl_mediana=float(np.median(p)),
                 pctl_pc_ge999=float((p >= 99.9).mean() * 100))
        res["lags"][f"L={L}"] = r
        filas.append((L, r))
        print(f"  L={L} hecho", flush=True)

    print("\n" + "=" * 74)
    print("HÍBRIDO: ERA5 hasta D−L, luego L días de IFS (jun-sep 2024)")
    print("=" * 74)
    print(f"{'config':<26}{'corr':>8}{'sesgo':>9}{'máx':>9}"
          f"{'pctl med':>10}{'% >=99,9':>11}")
    print(f"{'CUBO (referencia)':<26}{1.0:>8.3f}{0.0:>+9.2f}"
          f"{np.nanmax(ref[m_ev]):>9.1f}{46.5:>10.1f}{0.2:>11.1f}")
    for L, r in filas:
        nom = "ERA5 puro (L=0)" if L == 0 else f"ERA5 + {L}d de IFS"
        print(f"{nom:<26}{r['corr']:>8.3f}{r['sesgo']:>+9.2f}"
              f"{r['maximo']:>9.1f}{r['pctl_mediana']:>10.1f}"
              f"{r['pctl_pc_ge999']:>11.1f}")
    print(f"{'IFS puro (medido antes)':<26}{0.935:>8.3f}{0.41:>+9.2f}"
          f"{136.7:>9.1f}{42.9:>10.1f}{2.4:>11.1f}")

    with open(f"{DIR}/dataset/prueba_hibrido_produccion.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("\nGuardado: dataset/prueba_hibrido_produccion.json")


if __name__ == "__main__":
    main()
