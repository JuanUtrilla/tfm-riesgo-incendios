#!/usr/bin/env python3
"""
Dos modelos — paso 31: calibración por celda con VENTANA MÓVIL.

NO TOCA PRODUCCIÓN NI NINGÚN REPO. Lee salida/dos_25_*; escribe
salida/dos_31_*.{csv,json}.

=============================================================================
QUÉ PROBLEMA ATACA
=============================================================================
`dos_29` calibra por mes con un bloque FIJO de años (2015-2021) y deja el error
típico de la punta en 1,8x. Pero **julio infrapredice por 7x** (0,88 % contra
6,22 % observado), y julio es el mes que concentra más superficie quemada.

La causa está identificada (`LIMITACIONES.md` §9): la ventana 2015-2021 **no
contiene ningún año extremo**, y el periodo de evaluación sí. 2022 aporta por sí
solo más celdas quemadas en dos años (6.500) que los siete de calibración juntos
(5.695). Un bloque fijo que empieza en 2015 arrastra para siempre una década
menos incendiaria que el presente.

La hipótesis de este paso: **calibrar con los K años ANTERIORES a cada año
evaluado**, en vez de con un bloque fijo, sigue la deriva de la tasa base y
corrige la infrapredicción.

=============================================================================
CÓMO
=============================================================================
Para cada año Y de evaluación (2022, 2023, 2024) y cada ventana K:

    curva = isotónica ajustada con los años [Y-K, Y-1], estratificada por mes
    se evalúa en Y

Esto es honesto por construcción: la curva que puntúa el año Y solo ha visto
años anteriores a Y. No hay fuga temporal, y a diferencia de `dos_29` tampoco
hay un bloque fijo que envejece.

K = 3, 5, 7 y «todo lo anterior». K=7 evaluando 2022 equivale al bloque fijo de
`dos_29`, así que la comparación es directa.

Criterio, el mismo que `dos_29`: mediana de |log10(predicha/observada)| en el
top 0,2 % de cada mes. Se reporta aparte el error SOLO en jun-sep, que es lo
que de verdad importa (el 66 % de las celdas quemadas).

Uso:
    python dos_31_ventana_movil.py
"""

import json

import numpy as np
import pandas as pd

import config
from dos_25_barrido_historico import BINS, FIN, INI, MODELOS, NBINS
from dos_27_escala_absoluta import BORDE
from dos_29_calibra_estacional import FRAC_PUNTA, cuentas, isotonica

SAL = config.salida("dos_31")
VENTANAS = (3, 5, 7, 99)                 # 99 = todo lo anterior
EVAL = (2022, 2023, 2024)
NOM = {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
       7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre",
       11: "noviembre", 12: "diciembre"}


def main():
    z = np.load(config.salida("dos_25_hist.npz"), allow_pickle=True)
    hist, quem = z["hist"], z["quem"]
    cel = pd.read_parquet(config.salida("dos_25_celdas.parquet"))
    dias = pd.date_range(INI, FIN, freq="D")
    anio, mes = dias.year.values, dias.month.values

    sub = cel[cel.es_sub][["iy", "ix"]].copy()
    sub["_ok"] = True
    base = pd.DataFrame(quem[:, :3], columns=["ti", "iy", "ix"]).astype(int)
    base = base.merge(sub, on=["iy", "ix"], how="left")
    base = base[base._ok.fillna(False)].copy()

    filas = []
    for nom in MODELOS:
        k = MODELOS.index(nom)
        q = base.copy()
        q["s"] = quem[:, 4 + k][base.index.values]
        for K in VENTANAS:
            for Y in EVAL:
                ini = 2015 if K == 99 else max(2015, Y - K)
                for m in range(1, 13):
                    ca = (mes == m) & (anio >= ini) & (anio <= Y - 1)
                    ev = (mes == m) & (anio == Y)
                    if not ca.any() or not ev.any():
                        continue
                    ir = isotonica(*cuentas(hist, k, q, ca))
                    if ir is None:
                        continue
                    n, y = cuentas(hist, k, q, ev)
                    tot = n.sum()
                    if tot == 0:
                        continue
                    cn = np.cumsum(n[::-1])[::-1]
                    cy = np.cumsum(y[::-1])[::-1]
                    j = int(np.argmax(cn <= tot * FRAC_PUNTA))
                    if cn[j] == 0 or cy[j] == 0:
                        continue
                    obs = cy[j] / cn[j]
                    pred = float((n[j:] * ir.predict(BORDE[j:])).sum()
                                 / max(n[j:].sum(), 1))
                    if pred <= 0:
                        continue
                    filas.append(dict(
                        modelo=nom, ventana=("todo" if K == 99 else f"{K}a"),
                        anio=Y, mes=NOM[m], mes_n=m, quemadas=int(cy[j]),
                        anios_calib=f"{ini}-{Y-1}",
                        pred_pct=round(100 * pred, 4),
                        obs_pct=round(100 * obs, 4),
                        log10_err=round(abs(np.log10(pred / obs)), 3)))

    F = pd.DataFrame(filas)
    F.to_csv(f"{SAL}_ventanas.csv", index=False)

    print("=== ERROR DE LA PUNTA POR VENTANA (mediana |log10|, menor es mejor) ===")
    print("evaluando 2022-2024, curva ajustada solo con años ANTERIORES\n")
    tot = F.groupby(["modelo", "ventana"]).log10_err.median().unstack()
    ver = (F[F.mes_n.between(6, 9)].groupby(["modelo", "ventana"])
           .log10_err.median().unstack())
    orden = ["3a", "5a", "7a", "todo"]
    print("TODO EL AÑO");  print(tot[orden].round(3).to_string())
    print("\nSOLO jun-sep (66 % de las celdas quemadas)")
    print(ver[orden].round(3).to_string())

    print("\n=== r10, JULIO: el caso que motivaba este paso ===")
    j = F[(F.modelo == "r10") & (F.mes == "julio")]
    print(f"{'ventana':8} {'año':>5} {'calib':>10} {'quem':>5} {'obs':>8} "
          f"{'pred':>8} {'error':>7}")
    for v in orden:
        for _, r in j[j.ventana == v].iterrows():
            print(f"{v:8} {r.anio:5d} {r.anios_calib:>10} {r.quemadas:5d} "
                  f"{r.obs_pct:7.2f}% {r.pred_pct:7.2f}% {10**r.log10_err:6.1f}x")

    mej = ver.loc["r10"].idxmin()
    print(f"\n→ mejor ventana para r10 en jun-sep: **{mej}** "
          f"({ver.loc['r10', mej]:.3f} contra {ver.loc['r10','todo']:.3f} "
          f"de usar todo lo anterior)")
    json.dump(dict(mejor_ventana_verano=str(mej),
                   mediana_verano={c: float(v) for c, v in ver.loc["r10"].items()},
                   mediana_anual={c: float(v) for c, v in tot.loc["r10"].items()}),
              open(f"{SAL}_resumen.json", "w"), indent=1)
    print(f"\n→ {SAL}_ventanas.csv · {SAL}_resumen.json")


if __name__ == "__main__":
    main()
