#!/usr/bin/env python3
"""
Dos modelos, paso 29: calibración por celda estacional.

No toca producción ni ningún repo. Lee salida/dos_25_*; escribe
salida/dos_29_*.{csv,json}.

Por qué hace falta
------------------
`dos_28` ajusta una sola curva isotónica puntuación→probabilidad para todo el
año. Medido el 05/09/2026 sobre el top 0,02 % de cada mes (r10):

    mes         puntuación   curva global   real del mes
    febrero        1,0000        2,68 %         3,74 %
    octubre        0,9900        0,88 %         3,16 %
    diciembre      1,0000        2,68 %         0,28 %     (9,6x de más)
    julio          0,9440        0,24 %        10,36 %     (43x de menos)

La puntuación no es comparable entre estaciones: las peores celdas de julio
puntúan 0,944 (menos que las de diciembre, que puntúan 1,0000) y sin embargo
arden 37 veces más. Servir la curva global diría 2,7 % en diciembre (alarma
donde no la hay) y 0,24 % en julio (falsa calma en temporada alta, que es el
error que más pesa: jun-sep concentra el 66 % de las celdas quemadas).

Qué hace
--------
Ajusta la isotónica por estrato y compara tres opciones:
    global   una curva            (lo de dos_28)
    regimen  tres curvas          (verano / invierno-primavera / transición)
    mes      doce curvas
La estratificación se elige en validación (2022-2023) y se reporta en test
2024. Partición idéntica a dos_26 y dos_28; el test no se toca para elegir.

Criterio: error de calibración en la punta, que es lo que se publica. Para
cada mes se toma el top 0,02 % de celdas-día de ese mes, se compara la
probabilidad predicha media con la observada, y se resume con la mediana del
|log10(predicha/observada)|. 0 = perfecto, 1 = fallo de un factor 10.

Uso:
    python dos_29_calibra_estacional.py
"""

import json

import numpy as np
import pandas as pd

import config
from dos_25_barrido_historico import BINS, FIN, INI, MODELOS, NBINS
from dos_26_calibra_si import REGIMEN
from dos_27_escala_absoluta import BORDE

SAL = config.salida("dos_29")
CALIB = (2015, 2021)
VAL = (2022, 2023)
TEST = (2024, 2024)
# Punta de evaluación. El 0,02 % (el del entregable, §9) deja ~600 celdas-día
# por mes-año: la mayoría de meses de 2024 salen con cero quemadas y el error
# no es calculable. Al 0,2 % son ~6.000 y ya se puede medir. El entregable
# sigue reportándose al 0,02 % sobre el clima de los diez años.
FRAC_PUNTA = 0.002


def isotonica(n, y):
    from sklearn.isotonic import IsotonicRegression
    ok = n > 0
    if ok.sum() < 2 or y[ok].sum() == 0:
        return None
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(BORDE[ok], y[ok] / n[ok], sample_weight=n[ok])
    return ir


def cuentas(hist, k, q, sel):
    """Denominador (celdas-día por bin) y numerador (quemadas) en `sel`."""
    n = hist[sel, k, :].sum(0).astype(float)
    qq = q[sel[q.ti.values]]
    idx = np.clip(np.searchsorted(BINS, qq.s.values, "right") - 1, 0, NBINS - 1)
    return n, np.bincount(idx, minlength=NBINS).astype(float)


def error_punta(n, y, ir):
    """Predicha vs observada en el top FRAC_PUNTA. None si no hay datos."""
    tot = n.sum()
    if tot == 0 or ir is None:
        return None
    cn = np.cumsum(n[::-1])[::-1]
    cy = np.cumsum(y[::-1])[::-1]
    j = int(np.argmax(cn <= tot * FRAC_PUNTA))
    if cn[j] == 0:
        return None
    obs = cy[j] / cn[j]
    pred = float((n[j:] * ir.predict(BORDE[j:])).sum() / max(n[j:].sum(), 1))
    return pred, float(obs), int(cn[j]), int(cy[j])


def main():
    z = np.load(config.salida("dos_25_hist.npz"), allow_pickle=True)
    hist = z["hist"]
    quem = z["quem"]
    cel = pd.read_parquet(config.salida("dos_25_celdas.parquet"))
    dias = pd.date_range(INI, FIN, freq="D")
    anio, mes = dias.year.values, dias.month.values
    reg = np.array([REGIMEN[m] for m in mes])

    sub = cel[cel.es_sub][["iy", "ix"]].copy()
    sub["_ok"] = True
    base = pd.DataFrame(quem[:, :3], columns=["ti", "iy", "ix"]).astype(int)
    base = base.merge(sub, on=["iy", "ix"], how="left")
    base = base[base._ok.fillna(False)].copy()

    ESTRATOS = {"global": np.zeros(len(dias), int),
                "regimen": pd.factorize(reg)[0],
                "mes": mes}
    NOM = {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo",
           6: "junio", 7: "julio", 8: "agosto", 9: "septiembre",
           10: "octubre", 11: "noviembre", 12: "diciembre"}

    filas, guardar = [], {}
    for nom in MODELOS:
        k = MODELOS.index(nom)
        q = base.copy()
        q["s"] = quem[:, 4 + k][base.index.values]
        for est, etiq in ESTRATOS.items():
            curvas = {}
            for e in np.unique(etiq):
                sel = (etiq == e) & (anio >= CALIB[0]) & (anio <= CALIB[1])
                curvas[int(e)] = isotonica(*cuentas(hist, k, q, sel))
            if nom == "r10":
                guardar[est] = {int(e): (None if c is None else
                                         dict(x=[float(v) for v in c.X_thresholds_],
                                              y=[float(v) for v in c.y_thresholds_]))
                                for e, c in curvas.items()}
            for parte, (a, b) in (("val", VAL), ("test", TEST)):
                for m in range(1, 13):
                    sel = (mes == m) & (anio >= a) & (anio <= b)
                    if not sel.any():
                        continue
                    ir = curvas.get(int(etiq[sel][0]))
                    r = error_punta(*cuentas(hist, k, q, sel), ir)
                    if r is None:
                        continue
                    pred, obs, nn, yy = r
                    filas.append(dict(
                        modelo=nom, estrato=est, parte=parte, mes=NOM[m],
                        celdas_dia=nn, quemadas=yy,
                        pred_pct=round(100 * pred, 4),
                        obs_pct=round(100 * obs, 4),
                        log10_err=(np.nan if obs <= 0 or pred <= 0
                                   else round(abs(np.log10(pred / obs)), 3))))

    F = pd.DataFrame(filas)
    F.to_csv(f"{SAL}_punta.csv", index=False)
    json.dump(guardar, open(f"{SAL}_curvas_r10.json", "w"), indent=1)

    print("=== ELECCIÓN DEL ESTRATO (en VALIDACIÓN 2022-2023) ===")
    print("mediana de |log10(predicha/observada)| en la punta · menor es mejor")
    piv = (F[F.parte == "val"].groupby(["modelo", "estrato"]).log10_err
           .median().unstack().round(3))
    print(piv.to_string())
    mejor = piv.loc["r10"].idxmin()
    print(f"\n→ mejor estratificación para r10 en validación: **{mejor}**")

    print(f"\n=== TEST 2024 · r10 · estrato «{mejor}» contra «global» ===")
    t = F[(F.modelo == "r10") & (F.parte == "test")]
    a = t[t.estrato == "global"].set_index("mes")
    b = t[t.estrato == mejor].set_index("mes")
    print(f"{'mes':11} {'quem':>5} {'observada':>10} {'global':>9} {'err':>6} "
          f"{mejor:>9} {'err':>6}")
    for m in [NOM[i] for i in range(1, 13)]:
        if m not in a.index or m not in b.index:
            continue
        print(f"{m:11} {int(b.loc[m, 'quemadas']):5d} "
              f"{b.loc[m, 'obs_pct']:9.2f}% {a.loc[m, 'pred_pct']:8.2f}% "
              f"{a.loc[m, 'log10_err']:6.2f} {b.loc[m, 'pred_pct']:8.2f}% "
              f"{b.loc[m, 'log10_err']:6.2f}")
    print(f"\nmediana del error · global {a.log10_err.median():.3f} · "
          f"{mejor} {b.log10_err.median():.3f}")
    print(f"\n→ {SAL}_punta.csv · {SAL}_curvas_r10.json")


if __name__ == "__main__":
    main()
