#!/usr/bin/env python3
"""
Dos modelos — paso 32: la probabilidad por celda como PRODUCTO DE DOS CAPAS.

NO TOCA PRODUCCIÓN NI NINGÚN REPO. Lee salida/dos_25_* y dos_26_*; escribe
salida/dos_32_*.{csv,json}.

=============================================================================
DE DÓNDE VIENE
=============================================================================
`dos_31` refutó la ventana móvil y dejó el diagnóstico correcto: la frecuencia
observada en la punta de julio va del 0,00 % (2023) al 12,76 % (2022), mediana
0,53 %, y esa variación **no está en la puntuación de la celda** sino en la
intensidad del día. Ninguna curva puntuación→probabilidad puede seguirla.

La descomposición que sí puede es un producto de dos factores:

    P(arde la celda el día d) = P(d es día grande) x P(celda | día grande)
                                \_______________/   \____________________/
                                  la capa «si»          la capa «dónde»
                                  (dos_26, por día)     (por celda, aquí)

El segundo factor se calibra usando **solo los días grandes**, que es la
población en la que la pregunta «¿cuál arde?» tiene sentido. El primero ya está
calibrado en `dos_26`.

=============================================================================
CÓMO SE EVALÚA, PARA QUE SEA COMPARABLE
=============================================================================
Mismo criterio que `dos_29` y `dos_31`: la punta es el **top 0,2 % de las
celdas-día de cada mes ORDENADAS POR PUNTUACIÓN** —la selección no cambia, solo
cambia el número que se les asigna— y se compara la probabilidad media predicha
con la frecuencia observada, resumiendo con |log10(pred/obs)|.

Se comparan tres opciones sobre la misma punta:
    incondicional  la curva de dos_29 (por mes, todos los días)
    solo_dónde     P(celda | día grande), sin multiplicar. Debe sobreestimar.
    producto       P(día grande) x P(celda | día grande)

Ajuste con 2015-2021, evaluación en 2022-2024. Sin fuga temporal.

Uso:
    python dos_32_producto_dos_capas.py
"""

import json

import numpy as np
import pandas as pd

import config
from dos_25_barrido_historico import BINS, FIN, INI, MODELOS, NBINS
from dos_27_escala_absoluta import BORDE
from dos_29_calibra_estacional import FRAC_PUNTA, isotonica

SAL = config.salida("dos_32")
MODELO = "r10"
CALIB = (2015, 2021)
EVAL = (2022, 2024)
UMBRAL_DIA = 5                    # celdas de primer día que definen «día grande»
NOM = {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
       7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre",
       11: "noviembre", 12: "diciembre"}


def beta_p(s, w):
    s = np.clip(np.asarray(s, float), 1e-9, 1 - 1e-9)
    X = np.column_stack([np.log(s), -np.log(1 - s)])
    return 1 / (1 + np.exp(-(X @ np.array(w[:2]) + w[2])))


def main():
    z = np.load(config.salida("dos_25_hist.npz"), allow_pickle=True)
    hist, quem = z["hist"], z["quem"]
    k = MODELOS.index(MODELO)
    cel = pd.read_parquet(config.salida("dos_25_celdas.parquet"))
    dias = pd.date_range(INI, FIN, freq="D")
    anio, mes = dias.year.values, dias.month.values

    # --- verdad diaria y probabilidad del «si» (dos_26) --------------------
    v = pd.read_csv(f"{config.salida('dos_26')}_verdad.csv", parse_dates=["fecha"])
    d26 = pd.read_csv(f"{config.salida('dos_26')}_dias.csv", parse_dates=["fecha"])
    J = json.load(open(f"{config.salida('dos_26')}_calibracion.json"))
    grande = (v.primer_dia.values >= UMBRAL_DIA)
    p_si = beta_p(d26[f"p980_{MODELO}"].values, J[f"{MODELO}|TODO"]["beta"])
    print(f"días grandes: {grande.sum():,} de {len(grande):,} "
          f"({100*grande.mean():.1f} %)", flush=True)

    # --- quemadas dentro de la submuestra ---------------------------------
    sub = cel[cel.es_sub][["iy", "ix"]].copy()
    sub["_ok"] = True
    q = pd.DataFrame(quem[:, :3], columns=["ti", "iy", "ix"]).astype(int)
    q = q.merge(sub, on=["iy", "ix"], how="left")
    q = q[q._ok.fillna(False)].copy()
    q["s"] = quem[:, 4 + k][q.index.values]

    def cuentas(sel):
        n = hist[sel, k, :].sum(0).astype(float)
        qq = q[sel[q.ti.values]]
        i = np.clip(np.searchsorted(BINS, qq.s.values, "right") - 1, 0, NBINS - 1)
        return n, np.bincount(i, minlength=NBINS).astype(float)

    ca = (anio >= CALIB[0]) & (anio <= CALIB[1])
    ev = (anio >= EVAL[0]) & (anio <= EVAL[1])

    filas = []
    for m in range(1, 13):
        # --- segundo factor: P(celda | día grande), ajustado por mes -------
        cg = ca & (mes == m) & grande
        if not cg.any():
            continue
        ir_g = isotonica(*cuentas(cg))
        # --- referencia incondicional: la curva de dos_29 ------------------
        ir_u = isotonica(*cuentas(ca & (mes == m)))
        if ir_g is None or ir_u is None:
            continue

        em = ev & (mes == m)
        n, y = cuentas(em)
        tot = n.sum()
        if tot == 0:
            continue
        cn = np.cumsum(n[::-1])[::-1]
        cy = np.cumsum(y[::-1])[::-1]
        j = int(np.argmax(cn <= tot * FRAC_PUNTA))
        if cn[j] == 0 or cy[j] == 0:
            continue
        obs = cy[j] / cn[j]

        # predicciones sobre la MISMA punta (bins >= j), pesadas por celdas-día
        g_bin = ir_g.predict(BORDE[j:])
        u_bin = ir_u.predict(BORDE[j:])
        H = hist[em, k, j:].astype(float)          # (días del mes, bins)
        peso = H.sum()
        p_dia = p_si[em]                            # (días del mes,)
        pred_prod = float((H * g_bin[None, :] * p_dia[:, None]).sum() / peso)
        pred_solo = float((H * g_bin[None, :]).sum() / peso)
        pred_unc = float((H * u_bin[None, :]).sum() / peso)

        filas.append(dict(mes=NOM[m], mes_n=m, quemadas=int(cy[j]),
                          obs_pct=round(100 * obs, 4),
                          incondicional_pct=round(100 * pred_unc, 4),
                          solo_donde_pct=round(100 * pred_solo, 4),
                          producto_pct=round(100 * pred_prod, 4),
                          err_incond=round(abs(np.log10(pred_unc / obs)), 3),
                          err_solo=round(abs(np.log10(pred_solo / obs)), 3),
                          err_prod=round(abs(np.log10(pred_prod / obs)), 3)))

    F = pd.DataFrame(filas).sort_values("mes_n")
    F.to_csv(f"{SAL}_punta.csv", index=False)

    print(f"\n=== PUNTA (top 0,2 % del mes), {MODELO}, evaluado 2022-2024 ===")
    print(f"{'mes':11} {'quem':>5} {'OBSERVADA':>10} {'incond.':>9} "
          f"{'sólo dónde':>11} {'PRODUCTO':>9}")
    for _, r in F.iterrows():
        print(f"{r.mes:11} {r.quemadas:5d} {r.obs_pct:9.2f}% "
              f"{r.incondicional_pct:8.2f}% {r.solo_donde_pct:10.2f}% "
              f"{r.producto_pct:8.2f}%")

    ver = F[F.mes_n.between(6, 9)]
    print(f"\n{'':22}{'todo el año':>13}{'jun-sep':>11}")
    for c, lab in (("err_incond", "incondicional"), ("err_solo", "sólo dónde"),
                   ("err_prod", "PRODUCTO")):
        print(f"  mediana |log10|  {lab:14} {F[c].median():8.3f} "
              f"{ver[c].median():10.3f}")
    print(f"\n  factor de error típico (jun-sep): "
          f"incondicional {10**ver.err_incond.median():.1f}x · "
          f"producto {10**ver.err_prod.median():.1f}x")

    json.dump(dict(modelo=MODELO, umbral_dia=UMBRAL_DIA,
                   mediana_anual={c: float(F[c].median())
                                  for c in ("err_incond", "err_solo", "err_prod")},
                   mediana_verano={c: float(ver[c].median())
                                   for c in ("err_incond", "err_solo", "err_prod")}),
              open(f"{SAL}_resumen.json", "w"), indent=1)
    print(f"\n→ {SAL}_punta.csv · {SAL}_resumen.json")


if __name__ == "__main__":
    main()
