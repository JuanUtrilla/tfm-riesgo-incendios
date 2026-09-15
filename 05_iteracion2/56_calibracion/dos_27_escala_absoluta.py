#!/usr/bin/env python3
"""
Dos modelos, paso 27: escala absoluta de niveles para el mapa.

No toca producción ni ningún repo. Lee salida/dos_25_*; escribe
salida/dos_27_*.{csv,json,md}.

El problema
-----------
`dos_riesgo_hoy.py:68` pinta los niveles con percentiles del día:

    CORTES_PCTL = [30, 90, 98]
    NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]

Es decir, el 2 % superior del día es EXTREMO siempre, por construcción. Un
martes muerto de febrero sale con la misma cantidad de rojo que el 15 de
agosto, y el mapa se lee como «estas celdas van a arder». Medido sobre
2015-2024: bajo ese mismo rojo, la puntuación absoluta del p98 de un día de
invierno va de 0,032 a 0,965, es decir, el mismo color para treinta veces
más riesgo.

Lo que hace este paso
---------------------
Fija los cortes en la escala de puntuación, no en el ranking del día. El corte
de EXTREMO es el percentil 98 de la distribución climatológica (todas las
celdas de todos los días de 2015-2024), no el del día. Consecuencias:

  · un día tranquilo no tiene rojo;
  · un día excepcional puede tener mucho más del 2 %;
  · el rojo significa lo mismo en febrero que en agosto.

Los cortes salen del histograma de `dos_25` (3.653 días × 5 modelos × 4.096
bins), que ya está en disco: no hay que volver a tocar el cubo.

Como un color sin significado no arregla nada, cada banda se acompaña de su
frecuencia observada de quema, contada sobre las mismas celdas que forman el
histograma (`es_sub`), con `quem` como numerador.

Uso:
    python dos_27_escala_absoluta.py
"""

import json

import numpy as np
import pandas as pd

import config
from dos_25_barrido_historico import BINS, FIN, INI, MODELOS, NBINS
from dos_26_calibra_si import REGIMEN

SAL = config.salida("dos_27")
# los mismos cortes conceptuales que hoy, pero leídos sobre la climatología
PCTL = [30, 90, 98]
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]
# borde superior de cada bin: el histograma cuenta en [BINS[k], BINS[k+1])
BORDE = np.r_[BINS, 1.0][1:]


def cortes_climatologicos(hist):
    """Puntuación absoluta que deja por debajo el p30, p90 y p98 del clima.

    Se acumula el histograma de todos los días: cada celda-día pesa lo mismo,
    que es justo lo que el percentil del día pierde.
    """
    pool = hist.sum(0)                       # (modelos, bins)
    cs = np.cumsum(pool, axis=-1)
    n = cs[:, -1:]
    out = {}
    for k, nom in enumerate(MODELOS):
        out[nom] = [float(BORDE[np.clip((cs[k] >= n[k] * (q / 100)).argmax(),
                                        0, NBINS - 1)]) for q in PCTL]
    return out


def banda(hist, cortes_nom):
    """Reparte el histograma de cada día en las 4 bandas absolutas."""
    ks = [int(np.clip(np.searchsorted(BINS, c, "right") - 1, 0, NBINS - 1))
          for c in cortes_nom]
    trozos = [hist[:, :ks[0]].sum(1)]
    for a, b in zip(ks, ks[1:]):
        trozos.append(hist[:, a:b].sum(1))
    trozos.append(hist[:, ks[-1]:].sum(1))
    return np.column_stack(trozos)           # (dias, 4)


def main():
    z = np.load(config.salida("dos_25_hist.npz"), allow_pickle=True)
    hist = z["hist"]
    quem = z["quem"]
    cel = pd.read_parquet(config.salida("dos_25_celdas.parquet"))
    dias = pd.date_range(INI, FIN, freq="D")
    print(f"histograma {hist.shape} · quemadas {quem.shape}", flush=True)

    cortes = cortes_climatologicos(hist)
    print("\n=== CORTES ABSOLUTOS (percentiles de la climatología 2015-2024) ===")
    print(f"{'modelo':8} {'MODERADO≥':>10} {'ALTO≥':>9} {'EXTREMO≥':>10}")
    for nom in MODELOS:
        c = cortes[nom]
        print(f"{nom:8} {c[0]:10.4f} {c[1]:9.4f} {c[2]:10.4f}")

    # --- qué significa cada banda: frecuencia observada de quema -----------
    # `quem` lleva todas las celdas de `keep` que arden; el histograma solo
    # cuenta `es_sub`. Para que numerador y denominador hablen de la misma
    # población hay que quedarse con las quemadas que son `es_sub`.
    sub = cel[cel.es_sub][["iy", "ix"]].copy()
    sub["_ok"] = True
    q = pd.DataFrame(quem[:, :3], columns=["ti", "iy", "ix"]).astype(int)
    for k, nom in enumerate(MODELOS):
        q[nom] = quem[:, 4 + k]
    q = q.merge(sub, on=["iy", "ix"], how="left")
    q = q[q._ok.fillna(False)]
    print(f"\nquemadas dentro de la submuestra: {len(q):,} de {len(quem):,}")

    filas = []
    for k, nom in enumerate(MODELOS):
        c = cortes[nom]
        den = banda(hist[:, k, :], c).sum(0).astype(float)   # celdas-día/banda
        num = np.histogram(q[nom].values, bins=[0] + c + [1.0])[0].astype(float)
        tasa = num / np.maximum(den, 1)
        base = num.sum() / den.sum()
        for i, niv in enumerate(NIVELES):
            filas.append(dict(modelo=nom, nivel=niv,
                              corte_inf=round(([0.0] + c)[i], 4),
                              celdas_dia=int(den[i]),
                              pct_celdas=round(100 * den[i] / den.sum(), 2),
                              quemadas=int(num[i]),
                              tasa_pct=round(100 * tasa[i], 4),
                              lift=round(tasa[i] / base, 1)))
    B = pd.DataFrame(filas)
    B.to_csv(f"{SAL}_bandas.csv", index=False)
    print("\n=== QUÉ SIGNIFICA CADA COLOR (frecuencia observada de quema) ===")
    print(B.to_string(index=False))

    # --- la pregunta de fondo: ¿desaparece el rojo en invierno? -------------
    d = pd.DataFrame({"fecha": dias})
    d["mes"] = d.fecha.dt.month
    d["regimen"] = d.mes.map(REGIMEN)
    v = pd.read_csv(f"{config.salida('dos_26')}_verdad.csv", parse_dates=["fecha"])
    d = d.merge(v, on="fecha")
    d["y"] = (d.primer_dia >= 5).astype(int)

    print("\n=== CON ESCALA ABSOLUTA: ¿cuántos días se quedan SIN rojo? ===")
    print(f"{'modelo':8} {'régimen':19} {'días':>5} {'sin EXTREMO':>12} "
          f"{'de los tranquilos':>18} {'de los GRANDES':>15}")
    res = {}
    for k, nom in enumerate(MODELOS):
        bd = banda(hist[:, k, :], cortes[nom])
        d[f"ext_{nom}"] = bd[:, 3]
        for reg in ("invierno-primavera", "transicion", "verano"):
            m = (d.regimen == reg).values
            sinrojo = (d[f"ext_{nom}"].values == 0) & m
            tranq = m & (d.y.values == 0)
            gran = m & (d.y.values == 1)
            print(f"{nom:8} {reg:19} {m.sum():5d} "
                  f"{100*sinrojo.sum()/m.sum():11.1f}% "
                  f"{100*(sinrojo&tranq).sum()/max(tranq.sum(),1):17.1f}% "
                  f"{100*((d[f'ext_{nom}'].values==0)&gran).sum()/max(gran.sum(),1):14.1f}%")
        res[nom] = dict(cortes=cortes[nom], niveles=NIVELES, pctl_clim=PCTL)
    d.to_csv(f"{SAL}_dias.csv", index=False)
    json.dump(res, open(f"{SAL}_cortes.json", "w"), indent=1)
    print(f"\n→ {SAL}_cortes.json · {SAL}_bandas.csv · {SAL}_dias.csv")


if __name__ == "__main__":
    main()
