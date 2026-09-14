#!/usr/bin/env python3
"""Escala absoluta y cifra del día calibradas sobre los mapas SERVIDOS (opción 1).

Solo lectura. Entradas: archivo_ifs/replay/<año>/ifs/<fecha>.npz (prob_r10, float16,
el mismo camino de servicio que la cadena diaria), verdad_<año>/<fecha>.npz y
replay_<año>_ifs.csv (celdas quemadas y ha por día).

1. Cortes: la puntuación a partir de la cual una celda-día cae en el 70 %, 10 % y
   2 % más alto de la historia servida (mismas proporciones que dos_27 en el cubo:
   BAJO 30 %, MODERADO 60 %, ALTO 8 %, EXTREMO 2 %). Tres juegos: 2025, 2026 y
   2025+2026 (el que se publica).
2. Significado de cada nivel: frecuencia de quema y factor sobre la media, y
   fracción de hectáreas. Con los cortes de 2025 medidos en 2026 (fuera de
   muestra) y con los de 2025+2026 en las dos temporadas.
3. Cifra del día: % de España en EXTREMO. AUC para separar días grandes (al
   menos 5 celdas quemadas) por temporada, frente al calendario (frecuencia de
   día grande por día del año en el cubo 2015-2024, dos_26_dias.csv).
4. Referencia para publicar: la serie diaria del % EXTREMO con los cortes
   2025+2026, que la cadena compara con el valor de cada día.

Los float16 se cuentan exactos con bincount sobre su representación de 16 bits.
Salidas junto al script: calibracion_servicio.json, calibracion_servicio_dias.csv
"""
import glob
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

M = pathlib.Path.home() / "Desktop/Master"
R = M / "archivo_ifs/replay"
AQUI = pathlib.Path(__file__).resolve().parent
CODIGOS = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]
CUBO = [0.0017807815593375862, 0.07355870903046814, 0.40661572679295743]


def cargar():
    dias = []
    for anio in (2025, 2026):
        tabla = pd.read_csv(R / f"replay_{anio}_ifs.csv").set_index("fecha")
        for f in sorted(glob.glob(str(R / f"{anio}/ifs/*.npz"))):
            fecha = pathlib.Path(f).stem
            p = np.load(f)["prob_r10"]
            ok = np.isfinite(p)
            cod = p[ok].view(np.uint16)
            h = np.bincount(cod, minlength=65536).astype(np.int64)
            hq = np.zeros(65536, np.float64)
            hq_ha = np.zeros(65536, np.float64)
            fv = R / f"verdad_{anio}/{fecha}.npz"
            if fv.exists():
                V = np.load(fv)
                celda, fuego = V["celda"], V["fuego"]
                okc = ok.ravel()[celda]
                celda, fuego = celda[okc], fuego[okc]
                if len(celda):
                    c = p.ravel()[celda].view(np.uint16)
                    np.add.at(hq, c, 1)
                    np.add.at(hq_ha, c, V["area_ha"][fuego] / V["n_celdas"][fuego])
            fila = tabla.loc[fecha] if fecha in tabla.index else None
            dias.append(dict(anio=anio, fecha=fecha, h=h, hq=hq, hq_ha=hq_ha,
                             celdas=int(fila["celdas_quemadas"]) if fila is not None else 0,
                             ha=float(fila["area_ha"]) if fila is not None else 0.0))
    return dias


def cortes_de(h):
    orden = np.argsort(CODIGOS)
    v, c = CODIGOS[orden], h[orden]
    m = c > 0
    v, c = v[m], c[m]
    acum = np.cumsum(c) / c.sum()
    return [float(v[np.searchsorted(acum, q, side="left")]) for q in (0.30, 0.90, 0.98)]


def niveles_de(cortes):
    return np.digitize(CODIGOS, cortes)          # nivel de cada código float16


def significado(dias, cortes):
    niv = niveles_de(cortes)
    tot = sum(d["h"] for d in dias)
    q = sum(d["hq"] for d in dias)
    qha = sum(d["hq_ha"] for d in dias)
    base = q.sum() / tot.sum()
    out = []
    for i, n in enumerate(NIVELES):
        sel = niv == i
        celdas, quem = tot[sel].sum(), q[sel].sum()
        tasa = quem / celdas if celdas else np.nan
        out.append(dict(nivel=n, pct_celdas=100 * celdas / tot.sum(), quemadas=int(quem),
                        tasa_pct=100 * tasa, factor=tasa / base,
                        uno_de_cada=(1 / tasa) if tasa else None,
                        pct_ha=100 * qha[sel].sum() / qha.sum()))
    return out


def pct_extremo(d, cortes):
    niv = niveles_de(cortes)
    return 100 * d["h"][niv == 3].sum() / d["h"].sum()


def main():
    dias = cargar()
    h25 = sum(d["h"] for d in dias if d["anio"] == 2025)
    h26 = sum(d["h"] for d in dias if d["anio"] == 2026)
    juegos = {"2025": cortes_de(h25), "2026": cortes_de(h26), "2025+2026": cortes_de(h25 + h26)}
    res = {"cortes": juegos, "cortes_cubo": CUBO, "significado": {}}
    d25 = [d for d in dias if d["anio"] == 2025]
    d26 = [d for d in dias if d["anio"] == 2026]
    for nom, conj, cor in (("cortes 2025 en 2025 (dentro)", d25, juegos["2025"]),
                           ("cortes 2025 en 2026 (fuera)", d26, juegos["2025"]),
                           ("cortes 2025+2026 en 2025", d25, juegos["2025+2026"]),
                           ("cortes 2025+2026 en 2026", d26, juegos["2025+2026"]),
                           ("cortes 2025+2026 en las dos", dias, juegos["2025+2026"]),
                           ("cortes del cubo en las dos", dias, CUBO)):
        res["significado"][nom] = significado(conj, cor)

    clim = pd.read_csv(M / "calibracion_si/sandbox/salida/dos_26_dias.csv")
    clim = clim.assign(g=(clim.primer_dia >= 5).astype(float)).groupby("doy").g.mean()
    clim = pd.Series(np.convolve(np.r_[clim.values[-15:], clim.values, clim.values[:15]],
                                 np.ones(31) / 31, "valid"), index=range(1, 367))
    filas = []
    for d in dias:
        f = pd.Timestamp(d["fecha"])
        filas.append(dict(fecha=d["fecha"], anio=d["anio"], mes=f.month,
                          pct_ext_servicio=pct_extremo(d, juegos["2025+2026"]),
                          pct_ext_cortes2025=pct_extremo(d, juegos["2025"]),
                          pct_ext_cubo=pct_extremo(d, CUBO),
                          celdas=d["celdas"], ha=d["ha"], grande=int(d["celdas"] >= 5),
                          calendario=float(clim.loc[f.dayofyear])))
    t = pd.DataFrame(filas)
    t.to_csv(AQUI / "calibracion_servicio_dias.csv", index=False)
    auc = {}
    for anio, g in t.groupby("anio"):
        auc[str(anio)] = {c: float(roc_auc_score(g.grande, g[c]))
                          for c in ("pct_ext_servicio", "pct_ext_cortes2025", "calendario")}
        auc[str(anio)]["spearman_ha"] = float(g.pct_ext_servicio.corr(g.ha, method="spearman"))
        auc[str(anio)]["dias"], auc[str(anio)]["grandes"] = int(len(g)), int(g.grande.sum())
    res["auc_dia_grande"] = auc
    ref = t.pct_ext_servicio.values
    res["referencia"] = dict(dias=int(len(ref)), mediana=float(np.median(ref)),
                             p75=float(np.percentile(ref, 75)), p90=float(np.percentile(ref, 90)),
                             maximo=float(ref.max()))
    res["por_mes"] = t.groupby(["anio", "mes"]).pct_ext_servicio.median().round(3).reset_index().to_dict("records")
    json.dump(res, open(AQUI / "calibracion_servicio.json", "w"), indent=1, ensure_ascii=False)

    print("cortes:", {k: [round(x, 4) for x in v] for k, v in juegos.items()}, "cubo:", [round(x, 4) for x in CUBO])
    for nom, s in res["significado"].items():
        print(f"\n{nom}")
        print(pd.DataFrame(s).round(4).to_string(index=False))
    print("\nAUC día grande:", json.dumps(auc, indent=1))
    print("referencia:", res["referencia"])
    print(pd.DataFrame(res["por_mes"]).to_string(index=False))
    for f in ("2025-08-13", "2025-10-29"):
        v = float(t.loc[t.fecha == f, "pct_ext_servicio"].iloc[0])
        print(f"{f}: EXTREMO {v:.2f} % · percentil en la referencia {100*np.mean(ref <= v):.1f}")


if __name__ == "__main__":
    main()
