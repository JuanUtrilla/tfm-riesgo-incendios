#!/usr/bin/env python3
"""Compara qué señal diaria separa mejor los días con incendio grande (14/09/2026).

Solo lectura. Señales, evaluadas en 2022-2024 (años que ninguna calibración vio):
  calendario      frecuencia de día grande por día del año (±15 d), 2015-2021
  p98             p98 de la puntuación del r10 en España (la señal que usaba el semáforo)
  % EXTREMO       % de celdas en EXTREMO con la escala absoluta
  termómetro mes  percentil del % EXTREMO dentro de su mes (distribución 2015-2021)
Métrica: AUC por época con IC95 por bootstrap de días (1,000, semilla 42), y % de
días grandes perdidos si se apagan tantos días como apagaba el semáforo V0.
Día grande: al menos 5 celdas nuevas quemadas.

Entradas: dos_27_dias.csv y dos_26_dias.csv de la carpeta de resultados (TFM_SALIDA)
Salida:   comparar_senales.txt (junto a este script)
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

S = pathlib.Path(os.environ.get(
    "TFM_SALIDA", pathlib.Path(__file__).resolve().parents[2] / "salida"))
AQUI = pathlib.Path(__file__).resolve().parent


def main():
    d = pd.read_csv(S / "dos_27_dias.csv").merge(
        pd.read_csv(S / "dos_26_dias.csv")[["fecha", "p980_r10", "doy", "anio"]], on="fecha")
    d["grande"] = (d.primer_dia >= 5).astype(int)
    d["pct_ext"] = 100 * d.ext_r10 / 498530
    cal, ho = d[d.anio <= 2021], d[d.anio >= 2022].copy()
    f = cal.groupby("doy").grande.mean().reindex(range(1, 367)).fillna(0)
    clim = pd.Series(np.convolve(np.r_[f.values[-15:], f.values, f.values[:15]],
                                 np.ones(31) / 31, "valid"), index=range(1, 367))
    ho["clim"] = ho.doy.map(clim)
    ho["termo"] = [(cal.loc[cal.mes == m, "pct_ext"] <= v).mean() * 100
                   for m, v in zip(ho.mes, ho.pct_ext)]
    rng = np.random.default_rng(42)

    def auc_ic(y, x):
        a, bs = roc_auc_score(y, x), []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                bs.append(roc_auc_score(y[i], x[i]))
        return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    senales = [("calendario", "clim"), ("p98 (semáforo)", "p980_r10"),
               ("% EXTREMO", "pct_ext"), ("termómetro mes", "termo")]
    print("AUC para separar días grandes, 2022-2024 (IC95 bootstrap de días)")
    for reg in ["invierno-primavera", "transicion", "verano", "TODO"]:
        g = ho if reg == "TODO" else ho[ho.regimen == reg]
        y = g.grande.values
        out = [f"{reg:19s} n={len(g)} grandes={y.sum()}"]
        for nom, col in senales:
            a, lo, hi = auc_ic(y, g[col].values)
            out.append(f"{nom} {a:.3f} [{lo:.3f},{hi:.3f}]")
        print(" | ".join(out))
    print("\nApagando el mismo % de días que el semáforo actual (2022-2024), % de días grandes perdidos")
    for reg, off in [("invierno-primavera", 39.9), ("transicion", 70.7), ("verano", 59.8)]:
        g = ho[ho.regimen == reg]
        y = g.grande.values.astype(bool)
        res = []
        for nom, col in senales:
            x = g[col].values + rng.normal(0, 1e-9, len(g))
            u = np.percentile(x, off)
            res.append(f"{nom} {100 * ((x <= u) & y).sum() / y.sum():.1f}")
        print(f"{reg:19s} apagando {off} %: " + " · ".join(res))


if __name__ == "__main__":
    with open(AQUI / "comparar_senales.txt", "w") as fh:
        sys.stdout = fh
        main()
