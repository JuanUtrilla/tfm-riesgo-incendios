#!/usr/bin/env python3
"""Curva de captura y lift de los seis modelos del replay (condición IFS).

Solo lectura. Entradas:
  archivo_ifs/replay/<año>/ifs/<fecha>.npz   puntuación de cada modelo (920 x 1188)
  archivo_ifs/replay/verdad_<año>/<fecha>.npz  celdas quemadas: celda (índice plano),
                                             fuego, area_ha y n_celdas por incendio
  archivo_ifs/replay/replay_veredicto.json    para comprobar la reproducción

Definiciones, las de replay_dia.py:
  peso de una celda quemada = ha de su incendio / celdas del incendio
  top x % = celdas cuyo percentil dentro del día (sobre las 498,530 de España) es
            >= 100 - x
Salidas por modelo, temporada y x:
  captura_ha_dia     media por día de la fracción de ha dentro del top (como el replay)
  captura_ha_total   ha dentro del top / ha totales de la temporada
  captura_celdas     celdas quemadas dentro del top / celdas quemadas
  inc100             incendios >= 100 ha con alguna celda en el top / incendios >= 100 ha
  lift_ha            captura_ha_total / x
Y para el r10 con escala absoluta: nivel >= EXTREMO y >= ALTO, con y sin semáforo
(V0 y V1 de semaforo_variantes.py), con la fracción media de territorio marcada.
"""
import glob
import json
import pathlib

import numpy as np
import pandas as pd

M = pathlib.Path.home() / "Desktop/Master/archivo_ifs/replay"
OUT = pathlib.Path(__file__).resolve().parent
MODELOS = ["prod", "unico", "r10", "pareja", "donde", "cuando"]
XS = [0.5, 1, 2, 5, 10, 20]
CORTE_ALTO, CORTE_EXT = 0.07355870903046814, 0.40661572679295743
BETA = (1.9364816488092353, 0.005305402168090773, 0.2418997277088091)
UMBRAL = 0.15463618712368413

try:
    from scipy.stats import rankdata
except ImportError:                                   # pragma: no cover
    rankdata = None


def percentil(v):
    if rankdata is not None:
        return (rankdata(v, method="average") - 1) / (len(v) - 1) * 100
    return v.argsort().argsort() / (len(v) - 1) * 100


def p_si(s):
    s = float(np.clip(s, 1e-9, 1 - 1e-9))
    a, b, c = BETA
    return 1 / (1 + np.exp(-(a * np.log(s) + b * (-np.log(1 - s)) + c)))


def temporada(anio):
    acc = {(m, x): dict(ha_top=0.0, ha=0.0, cel_top=0, cel=0, inc_top=0, inc=0, dias=[])
           for m in MODELOS for x in XS}
    abs_acc = {k: dict(ha_top=0.0, ha=0.0, terr=[], dias=0)
               for k in ("EXT sin semáforo", "EXT V0", "EXT V1",
                         "ALTO+EXT sin semáforo", "ALTO+EXT V0", "ALTO+EXT V1")}
    n_dias = 0
    for f in sorted(glob.glob(str(M / f"{anio}/ifs/*.npz"))):
        fecha = pathlib.Path(f).stem
        fv = M / f"verdad_{anio}/{fecha}.npz"
        z = np.load(f)
        ok = np.isfinite(z["prob_r10"].astype(float).ravel())
        idx_ok = np.flatnonzero(ok)
        pos_en_ok = np.full(ok.size, -1)
        pos_en_ok[idx_ok] = np.arange(idx_ok.size)
        mes = int(fecha[5:7])
        hay_verdad = fv.exists()
        if hay_verdad:
            V = np.load(fv)
            celda, fuego, area, ncel = V["celda"], V["fuego"], V["area_ha"], V["n_celdas"]
            keep = ok[celda]
            celda, fuego = celda[keep], fuego[keep]
            peso = area[fuego] / ncel[fuego]
            grandes = np.flatnonzero(area >= 100)
        else:
            celda = np.array([], int)
        # --- escala absoluta + semáforo, r10 (todos los días, haya fuego o no)
        r10 = z["prob_r10"].astype(float).ravel()[ok]
        av0 = p_si(np.percentile(r10, 98)) >= UMBRAL
        av1 = True if mes in (6, 7, 8, 9) else av0
        for nivel, corte in (("EXT", CORTE_EXT), ("ALTO+EXT", CORTE_ALTO)):
            base = r10 >= corte
            for sem, a in (("sin semáforo", True), ("V0", av0), ("V1", av1)):
                if nivel == "EXT":
                    marca = base if a else np.zeros_like(base)
                else:
                    marca = base          # sin aviso EXTREMO pasa a ALTO: sigue marcado
                k = f"{nivel} {sem}"
                abs_acc[k]["terr"].append(marca.mean() * 100)
                abs_acc[k]["dias"] += 1
                if len(celda):
                    dentro = marca[pos_en_ok[celda]]
                    abs_acc[k]["ha_top"] += float((peso * dentro).sum())
                    abs_acc[k]["ha"] += float(peso.sum())
        n_dias += 1
        if not len(celda):
            continue
        for m in MODELOS:
            pm = z[f"prob_{m}"].astype(float).ravel()[ok]
            pc = percentil(pm)
            pc_pos = pc[pos_en_ok[celda]]
            for x in XS:
                top = pc_pos >= 100 - x
                a = acc[(m, x)]
                a["ha_top"] += float((peso * top).sum())
                a["ha"] += float(peso.sum())
                a["cel_top"] += int(top.sum())
                a["cel"] += int(top.size)
                a["dias"].append(float((peso * top).sum() / peso.sum() * 100))
                for g in grandes:
                    a["inc"] += 1
                    a["inc_top"] += int(top[fuego == g].any())
    filas = []
    for (m, x), a in acc.items():
        cap = 100 * a["ha_top"] / a["ha"]
        filas.append(dict(temporada=anio, modelo=m, top_pct=x, dias=len(a["dias"]),
                          captura_ha_dia=np.mean(a["dias"]), captura_ha_total=cap,
                          captura_celdas=100 * a["cel_top"] / a["cel"],
                          inc100=100 * a["inc_top"] / max(a["inc"], 1),
                          lift_ha=cap / x))
    abs_filas = []
    for k, a in abs_acc.items():
        terr = float(np.mean(a["terr"]))
        cap = 100 * a["ha_top"] / a["ha"] if a["ha"] else np.nan
        abs_filas.append(dict(temporada=anio, regla=k, dias=a["dias"],
                              territorio_medio_pct=terr, captura_ha_total=cap,
                              lift_ha=cap / terr if terr else np.nan))
    return filas, abs_filas, n_dias


def main():
    todas, todas_abs = [], []
    for anio in (2025, 2026):
        f, fa, n = temporada(anio)
        print(f"{anio}: {n} días")
        todas += f
        todas_abs += fa
    t = pd.DataFrame(todas)
    ta = pd.DataFrame(todas_abs)
    t.to_csv(OUT / "captura_lift.csv", index=False)
    ta.to_csv(OUT / "captura_absoluta_semaforo.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(2).to_string(index=False))
    print(ta.round(2).to_string(index=False))
    ver = json.load(open(M / "replay_veredicto.json"))
    for anio in (2025, 2026):
        for m in MODELOS:
            ref = ver[f"{anio}_ifs"]["modelos"][m]
            mio = t[(t.temporada == anio) & (t.modelo == m) & (t.top_pct == 2)].iloc[0]
            print(f"control {anio} {m:7s} top2 ha/día {mio.captura_ha_dia:5.1f} vs {ref['top2_ha']:5.1f}"
                  f" · inc100 {mio.inc100:5.1f} vs {ref['top2_inc100']:5.1f}")


if __name__ == "__main__":
    main()
