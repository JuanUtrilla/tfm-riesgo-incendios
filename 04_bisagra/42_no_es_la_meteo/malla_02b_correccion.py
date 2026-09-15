#!/usr/bin/env python3
"""
Módulo 2b, paso 3: corregir la rama de previsión. ¿Queda producción viable?

No toca producción. Escribe salida/prueba_correccion.json.

De dónde viene
    paso 1  IFS puro    / clim ERA5-Land → 2,18 %   (referencia: 0,23 %)
    paso 2  híbrido L=6 / clim ERA5-Land → 2,05 %

El paso 2 refutó el supuesto de la primera versión del proyecto de que la memoria larga del FWI
absorbería la rama de previsión: con solo 3 días de IFS ya se llega a 1,91 %.
La memoria está en DC y DMC, pero el valor diario lo manda el FFMC, cuya
constante de tiempo son horas. La rama de previsión domina.

La corrección, y por qué esta
El problema es el de siempre en este proyecto: numerador y denominador de
distribuciones distintas. El denominador es la climatología de ERA5-Land; el
numerador, en la rama de previsión, es IFS. No coinciden y la división satura.

Ya no se puede arreglar por la vía del módulo 4 (reconstruir el denominador)
porque no existe una climatología de IFS de siete años, ni la va a haber: es
un modelo operativo que cambia de ciclo. Así que se arregla por el otro lado,
llevando el numerador a la escala del denominador con un mapeo de cuantiles,
que es monótono y por tanto conserva el ranking.

Es la misma herramienta del módulo 3b, pero con el par correcto: allí se mapeó
ERA5-Land → cubo, y el módulo 5 midió que eso empeora (−3,2 pts de percentil)
porque el cubo ya no es el denominador de nada. Aquí se mapea híbrido →
ERA5-Land, que sí es el denominador real.

Sin circularidad
Un mapeo de cuantiles anula por construcción la diferencia de distribución
sobre su propio periodo de ajuste. Evaluarlo ahí no mide nada; es el error
que el módulo 3c detectó y corrigió en su día.

Aquí se ajusta con junio-julio y se evalúa con agosto-septiembre, que es lo
que haría producción: mapeo fijo, días nuevos. Los tres brazos se evalúan
sobre los mismos días de ago-sep para que las cifras sean comparables.

Uso: python malla_02b_correccion.py
"""

import json

import numpy as np
import pandas as pd

import config
from fwi_canadiense import calcular_fwi_serie
from malla_02b_prueba import IFS_FIN, IFS_INI, muestra_nodos, serie_era5land, serie_ifs

L = 6                              # el retraso real de ERA5-Land
Q = np.linspace(0, 1, 501)


def main():
    idx, la, lo = muestra_nodos()
    fechas = pd.date_range(IFS_INI, IFS_FIN, freq="D")
    meses = fechas.month.values
    fit = np.asarray(np.isin(meses, [6, 7]))        # ajuste
    tst = np.asarray(np.isin(meses, [8, 9]))        # evaluación

    ifs = serie_ifs(la, lo)
    S_ifs = {v: ifs.pivot(index="fecha", columns="nodo", values=v)
             .reindex(fechas).values
             for v in ("tmax", "hr_min", "viento_max", "prec")}
    print("\n  ERA5-Land en los nodos...", flush=True)
    S_era = serie_era5land(la, lo, fechas)

    nd, nn = len(fechas), len(la)
    print(f"\n  FWI del reanálisis con estado ({nn} nodos)...", flush=True)
    est = {k: np.full((nd, nn), np.nan) for k in ("ffmc", "dmc", "dc", "fwi")}
    for j in range(nn):
        o = calcular_fwi_serie(S_era["tmax"][:, j], S_era["hr_min"][:, j],
                               S_era["viento_max"][:, j] * 3.6,
                               S_era["prec"][:, j], meses)
        for k in est:
            est[k][:, j] = o[k]

    print("  híbrido L=6...", flush=True)
    H = np.full((nd, nn), np.nan)
    for j in range(nn):
        for k in range(L, nd):
            if not np.isfinite(est["ffmc"][k - L - 1, j]):
                continue
            o = calcular_fwi_serie(
                S_ifs["tmax"][k - L:k + 1, j], S_ifs["hr_min"][k - L:k + 1, j],
                S_ifs["viento_max"][k - L:k + 1, j] * 3.6,
                S_ifs["prec"][k - L:k + 1, j], meses[k - L:k + 1],
                ffmc0=est["ffmc"][k - L - 1, j], dmc0=est["dmc"][k - L - 1, j],
                dc0=est["dc"][k - L - 1, j])
            H[k, j] = o["fwi"][-1]

    # --- mapeo de cuantiles híbrido → ERA5-Land, ajustado con jun-jul -------
    a, b = est["fwi"][fit].ravel(), H[fit].ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    xs, ys = np.quantile(b[ok], Q), np.quantile(a[ok], Q)
    Hc = np.where(np.isfinite(H), np.interp(H, xs, ys), np.nan)
    np.savez_compressed(config.salida("mapeo_ifs_a_era5land.npz"),
                        xs=xs, ys=ys, ajustado_con="jun-jul 2024, 120 nodos",
                        lag=L)

    clim = np.load(config.CLIM)
    C = {m: clim[f"m{m}"][:, idx] for m in (8, 9)}
    orden = {m: [np.sort(C[m][:, j][np.isfinite(C[m][:, j])])
                 for j in range(nn)] for m in C}

    def mide(M):
        p, v = [], []
        for k in np.where(tst)[0]:
            col = orden[meses[k]]
            for j in range(nn):
                x = M[k, j]
                if np.isfinite(x) and len(col[j]):
                    p.append(np.searchsorted(col[j], x, side="right")
                             / len(col[j]) * 100)
                    v.append(x)
        p, v = np.array(p), np.array(v)
        return dict(pctl_mediana=float(np.median(p)),
                    ge95=float((p >= 95).mean() * 100),
                    ge999=float((p >= 99.9).mean() * 100),
                    fwi_medio=float(v.mean()), n=int(len(p)))

    print("\n" + "=" * 72)
    print("FUERA DE MUESTRA — ajuste jun-jul 2024, evaluación ago-sep 2024")
    print("=" * 72)
    print(f"{'numerador':<34}{'pctl med':>10}{'% >=95':>9}{'% >=99,9':>11}"
          f"{'FWI medio':>11}")
    res = {"n_nodos": nn, "lag": L, "brazos": {}}
    for nom, M in [("reanálisis ERA5-Land (referencia)", est["fwi"]),
                   ("híbrido L=6 crudo", H),
                   ("híbrido L=6 + mapeo a ERA5-Land", Hc)]:
        s = mide(M)
        res["brazos"][nom] = s
        print(f"{nom:<34}{s['pctl_mediana']:>10.1f}{s['ge95']:>9.1f}"
              f"{s['ge999']:>11.2f}{s['fwi_medio']:>11.2f}")

    ref = res["brazos"]["reanálisis ERA5-Land (referencia)"]["ge999"]
    cru = res["brazos"]["híbrido L=6 crudo"]["ge999"]
    cor = res["brazos"]["híbrido L=6 + mapeo a ERA5-Land"]["ge999"]
    print("\n" + "-" * 72)
    print(f"  saturación: referencia {ref:.2f} % · crudo {cru:.2f} % · "
          f"corregido {cor:.2f} %")
    print(f"  producción hoy con AEMET: 3,30 %")
    if cor < 1.0:
        ver = ("PRODUCCIÓN VIABLE — el mapeo deja la rama de previsión por "
               "debajo del 1 % fuera de muestra")
    elif cor < cru * 0.6:
        ver = ("MEJORA REAL PERO INSUFICIENTE — el mapeo baja la saturación "
               "sin llegar al nivel de la referencia")
    else:
        ver = ("EL MAPEO NO ARREGLA LA PREVISIÓN — hay que buscar otra vía "
               "para servir D y D+1")
    res["veredicto"] = ver
    print(f"\n  → {ver}")

    with open(config.salida("prueba_correccion.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado: {config.salida('prueba_correccion.json')}")


if __name__ == "__main__":
    main()
