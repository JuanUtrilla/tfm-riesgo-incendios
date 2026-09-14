#!/usr/bin/env python3
"""Variantes del semáforo del r10, evaluadas fuera de los años de calibración.

Solo lectura. Entradas:
  calibracion_si/sandbox/salida/dos_26_dias.csv        p98 diario del r10 2015-2024
  calibracion_si/sandbox/salida/dos_26_calibracion.json betas y umbrales (ajustados
                                                         en 2015-2021, dos_26)
  archivo_ifs/replay/replay_<año>_ifs.csv               p98 diario del r10 en el replay

Variantes:
  V0 actual           beta y umbral globales todo el año (mapa_r10_semaforo.py)
  V1 sin puerta jun-sep  V0 fuera de verano; en verano siempre se permite EXTREMO
  V2 puerta solo dic-abr  V0 solo en invierno-primavera; resto siempre EXTREMO
  V3 umbral por régimen   beta y umbral propios de cada régimen (dos_26)
  SIN                 sin semáforo (referencia)

Día grande: al menos 5 celdas nuevas quemadas (el objetivo de dos_26).
"""
import json
import pathlib

import numpy as np
import pandas as pd

M = pathlib.Path.home() / "Desktop/Master"
OUT = pathlib.Path(__file__).resolve().parent
REG = {**{m: "invierno-primavera" for m in (12, 1, 2, 3, 4)},
       **{m: "transicion" for m in (5, 10, 11)},
       **{m: "verano" for m in (6, 7, 8, 9)}}
CAL = json.load(open(M / "calibracion_si/sandbox/salida/dos_26_calibracion.json"))


def prob(s, beta):
    s = np.clip(np.asarray(s, float), 1e-9, 1 - 1e-9)
    a, b, c = beta
    return 1 / (1 + np.exp(-(a * np.log(s) + b * (-np.log(1 - s)) + c)))


def aviso(df, variante):
    s, reg = df["p98"].values, df["regimen"].values
    g = CAL["r10|TODO"]
    v0 = prob(s, g["beta"]) >= g["umbral_prob"]
    if variante == "V0":
        return v0
    if variante == "V1":
        return np.where(reg == "verano", True, v0)
    if variante == "V2":
        return np.where(reg == "invierno-primavera", v0, True)
    if variante == "V3":
        out = np.zeros(len(df), bool)
        for r in ("verano", "invierno-primavera", "transicion"):
            c = CAL[f"r10|{r}"]
            m = reg == r
            out[m] = prob(s[m], c["beta"]) >= c["umbral_prob"]
        return out
    if variante == "SIN":
        return np.ones(len(df), bool)
    raise ValueError(variante)


def resume(df, av, extra_ha=False):
    apag = ~av
    g = df["grande"].values.astype(bool)
    fila = dict(dias=len(df),
                pct_dias_sin_extremo=100 * apag.mean(),
                dias_grandes=int(g.sum()),
                pct_grandes_perdidos=100 * (apag & g).sum() / max(g.sum(), 1),
                pct_celdas_en_dias_apagados=100 * df["celdas"].values[apag].sum()
                / max(df["celdas"].sum(), 1))
    if extra_ha:
        fila["pct_ha_en_dias_apagados"] = (100 * df["ha"].values[apag].sum()
                                           / max(df["ha"].sum(), 1))
    return fila


def cubo():
    d = pd.read_csv(M / "calibracion_si/sandbox/salida/dos_26_dias.csv")
    d = d.assign(p98=d["p980_r10"], regimen=d["mes"].map(REG),
                 grande=d["primer_dia"] >= 5, celdas=d["primer_dia"])
    assert (d["regimen"] == d["regimen"]).all()
    filas = []
    for parte, msk in (("calibración 2015-2021", d.anio.between(2015, 2021)),
                       ("fuera de calibración 2022-2024", d.anio.between(2022, 2024)),
                       ("todo 2015-2024", d.anio.between(2015, 2024))):
        dd = d[msk]
        for var in ("SIN", "V0", "V1", "V2", "V3"):
            for reg in ("invierno-primavera", "transicion", "verano", "TOTAL"):
                sub = dd if reg == "TOTAL" else dd[dd.regimen == reg]
                av = aviso(sub, var)
                filas.append(dict(fuente="cubo", parte=parte, variante=var,
                                  regimen=reg, **resume(sub, av)))
    return filas


def replay():
    filas = []
    for anio in (2025, 2026):
        r = pd.read_csv(M / f"archivo_ifs/replay/replay_{anio}_ifs.csv")
        r["fecha"] = pd.to_datetime(r["fecha"])
        r = r.assign(p98=r["p_p98_r10"], regimen=r["fecha"].dt.month.map(REG),
                     grande=r["celdas_quemadas"] >= 5, celdas=r["celdas_quemadas"],
                     ha=r["area_ha"]).dropna(subset=["p98"])
        for var in ("SIN", "V0", "V1", "V2", "V3"):
            for reg in ("transicion", "verano", "TOTAL"):
                sub = r if reg == "TOTAL" else r[r.regimen == reg]
                if not len(sub):
                    continue
                av = aviso(sub, var)
                filas.append(dict(fuente=f"replay {anio}", parte="IFS", variante=var,
                                  regimen=reg, **resume(sub, av, extra_ha=True)))
    return filas


def main():
    t = pd.DataFrame(cubo() + replay())
    t.to_csv(OUT / "semaforo_variantes.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(t.round(1).to_string(index=False))
    # comprobación: V0 en todo 2015-2024, invierno = 44.0 / 6.2
    c = t[(t.fuente == "cubo") & (t.parte == "todo 2015-2024") & (t.variante == "V0")
          & (t.regimen == "invierno-primavera")].iloc[0]
    print(f"\ncontrol V0 invierno 2015-2024: {c.pct_dias_sin_extremo:.1f} % días, "
          f"{c.pct_grandes_perdidos:.1f} % grandes (esperado 44.0 / 6.2)")


if __name__ == "__main__":
    main()
