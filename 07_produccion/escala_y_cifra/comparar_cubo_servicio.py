#!/usr/bin/env python3
"""¿Se trasladan los cortes del cubo a los mapas servidos? (14/09/2026)

Solo lectura. Compara, mes a mes, el % de España en EXTREMO con los cortes de la
escala absoluta calculados en el cubo (dos_27: EXTREMO >= 0.4066):

  cubo       dos_27_dias.csv (columna ext_r10), 3,653 días de 2015-2024
  servicio   mapas del replay 2025-2026 (prob_r10), con previsión IFS y con
             reanálisis, el mismo camino de cálculo que la cadena diaria

Resultado del 14/09/2026: con los mismos cortes, los mapas servidos marcan entre
2 y 9 veces más EXTREMO que el cubo en el mismo mes. Por eso la cadena usa los
cortes de calibrar_servicio.py.

Entradas: ~/Desktop/Master/archivo_ifs/replay/<año>/<condición>/<fecha>.npz y
~/Desktop/Master/calibracion_si/sandbox/salida/dos_27_dias.csv (rutas del equipo
donde se ejecutó). Salida: comparar_cubo_servicio.txt junto a este script.
"""
import glob
import pathlib

import numpy as np
import pandas as pd

M = pathlib.Path.home() / "Desktop/Master"
AQUI = pathlib.Path(__file__).resolve().parent
CORTE_EXT_CUBO = 0.40661572679295743
N_CELDAS = 498530


def main():
    filas = []
    for anio in (2025, 2026):
        for cond in ("ifs", "reanalisis"):
            for f in sorted(glob.glob(str(M / f"archivo_ifs/replay/{anio}/{cond}/*.npz"))):
                p = np.load(f)["prob_r10"].astype(float)
                m = np.isfinite(p)
                fecha = pathlib.Path(f).stem
                filas.append(dict(anio=anio, condicion=cond, mes=int(fecha[5:7]),
                                  pct_ext=100 * np.mean(p[m] >= CORTE_EXT_CUBO)))
    servicio = (pd.DataFrame(filas).groupby(["anio", "condicion", "mes"])
                .agg(dias=("pct_ext", "size"), mediana_servicio=("pct_ext", "median"))
                .reset_index())
    h = pd.read_csv(M / "calibracion_si/sandbox/salida/dos_27_dias.csv")
    h["pct_ext"] = 100 * h["ext_r10"] / N_CELDAS
    cubo = h.groupby("mes").agg(mediana_cubo=("pct_ext", "median"),
                                p90_cubo=("pct_ext", lambda x: x.quantile(0.9)))
    t = servicio.merge(cubo, left_on="mes", right_index=True)
    t["veces"] = t["mediana_servicio"] / t["mediana_cubo"]
    texto = ["% de España en EXTREMO con los cortes del cubo: mediana diaria por mes",
             t.round(3).to_string(index=False),
             f"\nMáximo del cubo en 2015-2024: {h['pct_ext'].max():.2f} %"]
    (AQUI / "comparar_cubo_servicio.txt").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
