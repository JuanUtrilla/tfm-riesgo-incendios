#!/usr/bin/env python3
"""
Dos modelos, paso 28: calibrar la probabilidad por celda.

No toca producción ni ningún repo. Lee salida/dos_25_* y dos_27_cortes.json;
escribe salida/dos_28_*.{csv,json,md}.

Qué pregunta contesta
---------------------
`dos_26` calibró el «si» (¿hoy es día grande?). Esto calibra el «dónde»: qué
probabilidad tiene esta celda de arder hoy.

Hace falta porque la puntuación del modelo no es una probabilidad. Los modelos
se entrenaron con submuestreo de negativos (1:3 el único, 1:10 el r10), así que
su salida vive en la prevalencia del entrenamiento y no en la real. Una celda
que marca 0,8 no es un 80 %: es «0,8 en una población donde 1 de cada 4 ardía».
Sin este paso, el mapa solo admite lectura ordinal, y por eso se lee como «van
a arder».

Por qué no la corrección analítica
----------------------------------
El atajo de Elkan / King-Zeng / Dal Pozzolo (p = βp_s/(βp_s − p_s + 1)) no vale
para árboles: arXiv 2412.16209 concluye que la prevalencia estimada depende del
número de predictores y del ratio, y que los árboles pueden estar sesgados hacia
la minoritaria. Aquí no hace falta ningún atajo porque se tiene el censo. El
histograma de `dos_25` da el denominador exacto (celdas-día por bin de
puntuación) y `quem` el numerador (celdas que ardieron, con su puntuación).
P(arde | s) se mide directamente.

Cómo
----
· Numerador y denominador sobre la misma población: `es_sub`, la submuestra
  1-de-5 que forma el histograma. `quem` trae también celdas de fuera (las que
  arden alguna vez y entraron en `keep`); esas se descartan aquí, porque su
  denominador no está en el histograma.
· Partición de `dos_26`, sin tocar: calibración 2015-2021 · validación
  2022-2023 · test 2024 intacto.
· Calibración beta (Kull et al. 2017) sobre los conteos por bin, con peso =
  número de celdas-día. Misma familia que `dos_26`, por coherencia. La curva
  empírica por bin se guarda aparte como comprobación de fiabilidad.

Uso:
    python dos_28_calibra_celda.py
"""

import json

import numpy as np
import pandas as pd

import config
from dos_25_barrido_historico import BINS, FIN, INI, MODELOS, NBINS
from dos_27_escala_absoluta import BORDE, NIVELES

SAL = config.salida("dos_28")
CALIB = (2015, 2021)
VAL = (2022, 2023)
TEST = (2024, 2024)


def beta_ajusta(n, y):
    """Calibración beta sobre conteos por bin.

    n = celdas-día por bin, y = cuántas ardieron. Se ajusta una logística
    sobre [log s, −log(1−s)] con peso = conteo, que es equivalente a hacerlo
    sobre las 365 millones de filas sin materializarlas.
    """
    from sklearn.linear_model import LogisticRegression
    ok = n > 0
    s = np.clip(BORDE[ok], 1e-9, 1 - 1e-9)
    X = np.column_stack([np.log(s), -np.log(1 - s)])
    # cada bin aporta una fila de positivos y otra de negativos, con su peso
    XX = np.vstack([X, X])
    yy = np.r_[np.ones(ok.sum()), np.zeros(ok.sum())]
    w = np.r_[y[ok], n[ok] - y[ok]].astype(float)
    m = (w > 0)
    lr = LogisticRegression(C=1e6, max_iter=5000)
    lr.fit(XX[m], yy[m], sample_weight=w[m])
    return lr


def aplica(lr, s):
    s = np.clip(np.asarray(s, float), 1e-9, 1 - 1e-9)
    X = np.column_stack([np.log(s), -np.log(1 - s)])
    return lr.predict_proba(X)[:, 1]


def iso_ajusta(n, y):
    """Calibración isotónica sobre los conteos por bin. Es la que se usa.

    Por qué aquí sí y en `dos_26` no. Allí se rechazó por tres razones que no
    se cumplen en este paso: había 822 positivos a nivel de día (aquí 12.897
    celdas quemadas sobre 365 millones de celdas-día, agrupadas en 4.096 bins
    con denominador exacto), la escalera importaba porque de la curva se
    extraía un corte (aquí se publica la curva entera), y con un dato escaso
    convenía una familia paramétrica.

    Y sobre todo: la beta no vale para esta forma. Ajustada sin restricción
    da coeficiente negativo en −log(1−s) en los cinco modelos (p. ej. r10:
    [2,928, −1,737]), y una beta solo es monótona creciente con ambos ≥ 0. La
    curva se desplomaba al acercarse a s=1: la celda de puntuación máxima
    salía con probabilidad 0,000 %. La isotónica es monótona por construcción.
    """
    from sklearn.isotonic import IsotonicRegression
    ok = n > 0
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(BORDE[ok], (y[ok] / n[ok]), sample_weight=n[ok])
    return ir


def main():
    z = np.load(config.salida("dos_25_hist.npz"), allow_pickle=True)
    hist = z["hist"]
    quem = z["quem"]
    cel = pd.read_parquet(config.salida("dos_25_celdas.parquet"))
    cortes = json.load(open(config.salida("dos_27_cortes.json")))
    dias = pd.date_range(INI, FIN, freq="D")
    anio = dias.year.values

    # quemadas restringidas a `es_sub` (mismo universo que el histograma)
    sub = cel[cel.es_sub][["iy", "ix"]].copy()
    sub["_ok"] = True
    q = pd.DataFrame(quem[:, :3], columns=["ti", "iy", "ix"]).astype(int)
    for k, nom in enumerate(MODELOS):
        q[nom] = quem[:, 4 + k]
    q = q.merge(sub, on=["iy", "ix"], how="left")
    q = q[q._ok.fillna(False)].copy()
    q["anio"] = anio[q.ti.values]
    print(f"celdas-día totales {int(hist[:, 0].sum()):,} · "
          f"quemadas en la submuestra {len(q):,}", flush=True)

    partes = {"calib": CALIB, "val": VAL, "test": TEST}
    res, rel, tab, niv, ajustes = {}, [], [], [], {}
    for k, nom in enumerate(MODELOS):
        cnt = {}
        for p, (a, b) in partes.items():
            md = (anio >= a) & (anio <= b)
            n = hist[md, k, :].sum(0).astype(float)
            qq = q[(q.anio >= a) & (q.anio <= b)]
            idx = np.clip(np.searchsorted(BINS, qq[nom].values, "right") - 1,
                          0, NBINS - 1)
            y = np.bincount(idx, minlength=NBINS).astype(float)
            cnt[p] = (n, y)

        ir = iso_ajusta(*cnt["calib"])
        ajustes[nom] = ir
        p_bin = ir.predict(BORDE)
        lr = beta_ajusta(*cnt["calib"])          # solo para dejar constancia
        res[nom] = dict(
            isotonica=dict(x=[float(v) for v in ir.X_thresholds_],
                           y=[float(v) for v in ir.y_thresholds_]),
            beta_descartada=dict(coef=[float(x) for x in lr.coef_[0]],
                                 intercept=float(lr.intercept_[0])))

        for p, (n, y) in cnt.items():
            base = y.sum() / n.sum()
            # Brier y BSS agregados por bin (equivalentes a los de celda-día)
            br = float((n * p_bin ** 2 - 2 * y * p_bin + y).sum() / n.sum())
            br0 = float((n * base ** 2 - 2 * y * base + y).sum() / n.sum())
            tab.append(dict(modelo=nom, parte=p, celdas_dia=int(n.sum()),
                            quemadas=int(y.sum()),
                            tasa_base_pct=round(100 * base, 5),
                            brier=br, bss_vs_base=round(1 - br / br0, 4)))

        # fiabilidad en test: deciles de población sobre las celdas-día
        n, y = cnt["test"]
        cs = np.cumsum(n)
        for i, (lo, hi) in enumerate(zip(np.r_[0, cs[-1] * np.arange(1, 10) / 10],
                                         np.r_[cs[-1] * np.arange(1, 10) / 10,
                                               cs[-1]])):
            m = (cs > lo) & (cs <= hi + 1e-9)
            if not m.any() or n[m].sum() == 0:
                continue
            rel.append(dict(modelo=nom, decil=i + 1, celdas_dia=int(n[m].sum()),
                            p_medio_pct=round(100 * float((n[m] * p_bin[m]).sum()
                                                          / n[m].sum()), 5),
                            obs_pct=round(100 * float(y[m].sum() / n[m].sum()), 5)))

        # ---- el entregable: qué probabilidad hay detrás de cada color -----
        c = cortes[nom]["cortes"]
        nt, yt = cnt["test"]
        for i, nombre in enumerate(NIVELES):
            lo = ([0.0] + c)[i]
            hi = (c + [1.0])[i]
            m = (BORDE > lo) & (BORDE <= hi) if i else (BORDE <= hi)
            if nt[m].sum() == 0:
                continue
            niv.append(dict(
                modelo=nom, nivel=nombre, corte_inf=round(lo, 4),
                celdas_dia=int(nt[m].sum()), quemadas=int(yt[m].sum()),
                p_modelo_pct=round(100 * float((nt[m] * p_bin[m]).sum()
                                               / nt[m].sum()), 4),
                p_observada_pct=round(100 * float(yt[m].sum() / nt[m].sum()), 4)))

    T, R, N = pd.DataFrame(tab), pd.DataFrame(rel), pd.DataFrame(niv)
    T.to_csv(f"{SAL}_metricas.csv", index=False)
    R.to_csv(f"{SAL}_fiabilidad.csv", index=False)
    N.to_csv(f"{SAL}_niveles.csv", index=False)
    json.dump(res, open(f"{SAL}_beta.json", "w"), indent=1)

    print("\n=== CALIBRACIÓN POR CELDA ===")
    print(T.to_string(index=False))
    print("\n=== FIABILIDAD EN TEST 2024, r10 (deciles de población) ===")
    print(R[R.modelo == "r10"].to_string(index=False))
    print("\n=== LO QUE DICE CADA COLOR, EN PROBABILIDAD (test 2024) ===")
    print(N.to_string(index=False))

    print("\n=== TECHO: la celda más roja que llega a existir ===")
    for k, nom in enumerate(MODELOS):
        n = hist[:, k, :].sum(0)
        top = int(np.where(n > 0)[0].max())
        p = float(ajustes[nom].predict([BORDE[top]])[0])
        print(f"  {nom:8} puntuación máxima observada {BORDE[top]:.4f} → "
              f"probabilidad calibrada {100 * p:.3f} %")
    print(f"\n→ {SAL}_metricas.csv · {SAL}_fiabilidad.csv · "
          f"{SAL}_niveles.csv · {SAL}_beta.json")


if __name__ == "__main__":
    main()
