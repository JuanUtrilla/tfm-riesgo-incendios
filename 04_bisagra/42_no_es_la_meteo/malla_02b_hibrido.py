#!/usr/bin/env python3
"""
Módulo 2b, paso 2: el híbrido real. ¿Sirve para producción?

NO TOCA PRODUCCIÓN. Escribe salida/prueba_hibrido.json.

=============================================================================
POR QUÉ HACE FALTA ESTE PASO
=============================================================================
El paso 1 (`malla_02b_prueba.py`) midió el IFS PURO contra la climatología de
ERA5-Land y salió 2,18 % de saturación, contra 0,23 % del reanálisis. El
argumento del repo original era: "si el IFS puro sale aceptable, el híbrido lo
será por construcción". Pero el IFS puro NO sale aceptable, así que ese
argumento no cierra y hay que medir el híbrido de verdad.

Y el híbrido no es IFS puro. Producción sirve así:

    ... D-30 ... D-7 │ D-6 ... D, D+1
        ERA5-Land    │      IFS
      (reanálisis)   │  (previsión)

El FWI es un integrador con memoria larga (DC ~52 días). En el híbrido casi
todo el estado viene del reanálisis y el IFS solo empuja los últimos días. La
pregunta es cuánto de los 2,18 % sobrevive a eso.

=============================================================================
CÓMO SE MIDE SIN REHACERLO TODO CADA DÍA
=============================================================================
Reconstruir el híbrido día a día parecía caro y por eso el repo original midió
las dos cotas en vez del híbrido. No hace falta: el estado del FWI en D−L−1 es
el MISMO para todos los días objetivo, porque esa rama es siempre reanálisis.

Así que se corre el FWI sobre ERA5-Land UNA vez guardando (FFMC, DMC, DC) de
cada día, y para cada día objetivo D se reanuda desde el estado de D−L−1 y se
avanzan solo los L+1 días de previsión. De O(días²) a O(días·L).

Se miden varios L porque el retraso real de ERA5-Land varía (~5-6 días) y
conviene saber si el resultado es sensible a eso.

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python malla_02b_hibrido.py
"""

import json

import numpy as np
import pandas as pd

import config
from fwi_canadiense import calcular_fwi_serie
from malla_02b_prueba import (EVAL, IFS_FIN, IFS_INI, muestra_nodos,
                              serie_era5land, serie_ifs)

LAGS = (0, 2, 4, 6, 8)        # 0 = reanálisis puro; 6 = el retraso real


def main():
    idx, la, lo = muestra_nodos()
    fechas = pd.date_range(IFS_INI, IFS_FIN, freq="D")
    meses = fechas.month.values
    ev = np.asarray((fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1]))

    ifs = serie_ifs(la, lo)
    S_ifs = {v: ifs.pivot(index="fecha", columns="nodo", values=v)
             .reindex(fechas).values
             for v in ("tmax", "hr_min", "viento_max", "prec")}
    print("\n  ERA5-Land en los nodos...", flush=True)
    S_era = serie_era5land(la, lo, fechas)

    nd, nn = len(fechas), len(la)
    print(f"\n  FWI del reanálisis, guardando estado ({nn} nodos)...",
          flush=True)
    est = {k: np.full((nd, nn), np.nan) for k in ("ffmc", "dmc", "dc", "fwi")}
    for j in range(nn):
        o = calcular_fwi_serie(S_era["tmax"][:, j], S_era["hr_min"][:, j],
                               S_era["viento_max"][:, j] * 3.6,
                               S_era["prec"][:, j], meses)
        for k in est:
            est[k][:, j] = o[k]

    clim = np.load(config.CLIM)
    C = {m: clim[f"m{m}"][:, idx] for m in range(EVAL[0], EVAL[1] + 1)}
    orden = {m: [np.sort(C[m][:, j][np.isfinite(C[m][:, j])])
                 for j in range(nn)] for m in C}

    def satura(fwi_dia):
        """fwi_pctl_local de cada (día, nodo) contra la clim de ERA5-Land."""
        p = []
        for k in np.where(ev)[0]:
            col = orden[meses[k]]
            for j in range(nn):
                v = fwi_dia[k, j]
                if np.isfinite(v) and len(col[j]):
                    p.append(np.searchsorted(col[j], v, side="right")
                             / len(col[j]) * 100)
        p = np.array(p)
        return dict(pctl_mediana=float(np.median(p)),
                    ge95=float((p >= 95).mean() * 100),
                    ge999=float((p >= 99.9).mean() * 100), n=int(len(p)))

    print("\n" + "=" * 70)
    print("HÍBRIDO: reanálisis hasta D−L−1, previsión IFS los últimos L+1 días")
    print("=" * 70)
    print(f"{'L':>3}{'rama IFS':>12}{'pctl med':>11}{'% >=95':>9}"
          f"{'% >=99,9':>11}{'FWI medio':>11}")
    res = {"n_nodos": nn, "lags": {}}
    for L in LAGS:
        if L == 0:
            H = est["fwi"]
        else:
            H = np.full((nd, nn), np.nan)
            for j in range(nn):
                for k in np.where(ev)[0]:
                    i0 = k - L                      # primer día de previsión
                    if i0 - 1 < 0 or not np.isfinite(est["ffmc"][i0 - 1, j]):
                        continue
                    o = calcular_fwi_serie(
                        S_ifs["tmax"][i0:k + 1, j],
                        S_ifs["hr_min"][i0:k + 1, j],
                        S_ifs["viento_max"][i0:k + 1, j] * 3.6,
                        S_ifs["prec"][i0:k + 1, j], meses[i0:k + 1],
                        ffmc0=est["ffmc"][i0 - 1, j],
                        dmc0=est["dmc"][i0 - 1, j],
                        dc0=est["dc"][i0 - 1, j])
                    H[k, j] = o["fwi"][-1]
        s = satura(H)
        s["fwi_medio"] = float(np.nanmean(H[ev]))
        res["lags"][f"L={L}"] = s
        eti = "reanálisis puro" if L == 0 else f"{L + 1} días"
        print(f"{L:>3}{eti:>12}{s['pctl_mediana']:>11.1f}{s['ge95']:>9.1f}"
              f"{s['ge999']:>11.2f}{s['fwi_medio']:>11.2f}")

    print(f"\n{'IFS puro (paso 1)':>15}{'':>0}{46.0:>11.1f}{10.2:>9.1f}"
          f"{2.18:>11.2f}{36.80:>11.2f}")

    real = res["lags"]["L=6"]["ge999"]
    base = res["lags"]["L=0"]["ge999"]
    print("\n" + "-" * 70)
    print(f"  reanálisis puro {base:.2f} % → híbrido real (L=6) {real:.2f} % "
          f"→ IFS puro 2,18 %")
    if real < 1.0:
        ver = ("PRODUCCIÓN VIABLE — el híbrido se queda por debajo del 1 %; "
               "la memoria del FWI absorbe la cola del IFS")
    elif real < 1.5:
        ver = ("VIABLE CON RESERVA — mejor que hoy (3,3 % con AEMET) pero "
               "lejos de la referencia; conviene tratar la cola del IFS")
    else:
        ver = ("NO VIABLE TAL CUAL — la rama de previsión domina; hay que "
               "corregir el IFS antes de servir con él")
    res["veredicto"] = ver
    print(f"\n  → {ver}")

    with open(config.salida("prueba_hibrido.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado: {config.salida('prueba_hibrido.json')}")


if __name__ == "__main__":
    main()
