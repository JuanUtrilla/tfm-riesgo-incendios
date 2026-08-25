#!/usr/bin/env python3
"""
Cómo va la temporada: la figura del veredicto.

NO TOCA PRODUCCIÓN. Lee lo que haya en `salida/` y escribe
salida/dos_19_veredicto.{png,json}.

=============================================================================
POR QUÉ UNA SERIE Y NO UN MAPA
=============================================================================
Los mapas contestan DÓNDE. «¿Cómo va el sistema?» es otra pregunta y su
representación natural es una serie temporal, no una mancha de color: lo que
importa es si la ventaja se sostiene día tras día y cuánto se ha estrechado
el intervalo. Ese número existía desde julio, pero solo dentro de un JSON.

Dos paneles:

  ARRIBA   AUC dentro del día, día a día, de producción y de los candidatos,
           con su media acumulada. El punto es ruidoso —un día con tres
           celdas quemadas no dice nada— y por eso la línea gruesa es la
           media, no el punto.
  ABAJO    La diferencia PAREADA contra producción, acumulada: para cada día
           n se recalcula la media de las diferencias de los días 1..n y su
           IC95 por bootstrap DE DÍAS. Es la lectura honesta del progreso:
           la banda se estrecha con los días y el veredicto llega cuando deja
           de tocar el cero. Si la banda todavía cruza el cero, no hay
           veredicto por mucho que la media sea positiva.

           AVISO METODOLÓGICO: mirar la banda cada día y cantar victoria el
           primero en que despega del cero es *peeking*, y con 74 miradas
           infla el falso positivo. La trayectoria es descriptiva —sirve para
           ver si la ventaja es estable o la sostienen cuatro días— y el
           número que vale es el IC del ÚLTIMO día, fijado de antemano al
           cierre de la temporada.

Fuentes, y hay que distinguirlas porque no son igual de exigentes:
  · `dos_09_temporada2026.csv` — la temporada RETROSPECTIVA (reanálisis para
    todos: misma meteo, cara a cara limpio entre modelos, pero cota superior
    de lo que da la operación). Es la serie larga: 74 días.
  · `puntuacion_effis.csv` — el juez diario en OPERACIÓN, con la previsión
    que de verdad se publicó. Es el que vale, y empezó el 21/08, así que hoy
    tiene pocos días; se dibuja encima con marcador hueco.
  · `veredicto_miteco.csv` y `rankings_justo.csv` — los otros dos jueces, que
    no son series comparables (uno es por incidente y el otro por estación):
    van como texto en el pie, no como línea.
"""

import json
import os

import numpy as np
import pandas as pd

import config

N_BOOT = 2000
SEED = 42
# nombre corto → columna en cada fuente. Producción es la referencia.
SERIES = [
    ("producción", "auc_prod", "auc_produccion", "#444444"),
    ("único EFFIS", "auc_donde_dia_effis", "auc_unico", "#0072B2"),
    ("único EFFIS 1:10", "auc_donde_dia_effis_r10", None, "#009E73"),
    ("pareja", "auc_donde_effis_c×cuando_egif", "auc_pareja", "#D55E00"),
]
REF = "producción"


def acumulada(dif, rng):
    """Media de la diferencia y su IC95, recalculadas día a día."""
    m, lo, hi = [], [], []
    for n in range(1, len(dif) + 1):
        d = dif[:n]
        m.append(d.mean())
        if n < 3:
            lo.append(np.nan); hi.append(np.nan); continue
        b = d[rng.integers(0, n, (N_BOOT, n))].mean(1)
        lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
    return np.array(m), np.array(lo), np.array(hi)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # La serie base es la retrospectiva si está (74 días, todos los modelos);
    # si no —el caso de GitHub Actions, donde dos_09 no se ejecuta— se dibuja
    # con el juez diario en operación, que es el que de verdad importa y el
    # que irá creciendo. Nunca se mezclan en la misma línea: son bancos
    # distintos (reanálisis contra previsión publicada).
    op = None
    ruta_op = config.salida("puntuacion_effis.csv")
    if os.path.exists(ruta_op):
        op = pd.read_csv(ruta_op).sort_values("fecha")
        op["fecha"] = pd.to_datetime(op["fecha"])

    # en el portátil está en salida/; en GitHub, versionado en publicado/
    retro = next((r for r in (config.salida("dos_09_temporada2026.csv"),
                              f"{config.BASE}/publicado/dos_09_temporada2026.csv")
                  if os.path.exists(r)), None)
    if retro:
        d = pd.read_csv(retro).sort_values("fecha")
        d["fecha"] = pd.to_datetime(d["fecha"])
        hay = [(n, c, o, k) for n, c, o, k in SERIES if c in d.columns]
        fuente = "dos_09_temporada2026.csv (retrospectivo, mismo reanálisis)"
    elif op is not None and len(op) >= 3:
        d = op
        hay = [(n, o, None, k) for n, c, o, k in SERIES
               if o and o in op.columns]
        op = None                              # ya es la serie base
        fuente = "puntuacion_effis.csv (operación, previsión publicada)"
    else:
        raise SystemExit("sin serie: falta dos_09_temporada2026.csv y "
                         "puntuacion_effis.csv no tiene días todavía")

    rng = np.random.default_rng(SEED)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    R = {"n_dias": int(len(d)), "fuente": fuente}

    # ---------------------------------------------------------- panel de arriba
    for nom, col, col_op, color in hay:
        y = d[col].values
        a1.plot(d["fecha"], y, ls="", marker="o", ms=2.6, color=color, alpha=0.28)
        a1.plot(d["fecha"], pd.Series(y).expanding().mean(), color=color, lw=2,
                label=f"{nom}  ({np.nanmean(y):.3f})")
        if op is not None and col_op and col_op in op.columns:
            a1.plot(op["fecha"], op[col_op], ls="", marker="o", ms=6, mfc="none",
                    mew=1.4, color=color)
        R[nom] = {"auc_medio": float(np.nanmean(y))}
    a1.axhline(0.5, color="0.7", lw=0.8, ls=":")
    a1.set_ylabel("AUC dentro del día")
    a1.legend(fontsize=8, ncol=2, frameon=False, loc="lower left")
    a1.set_title(f"Temporada 2026 · punto = día · línea = media acumulada · "
                 f"fuente: {fuente}"
                 + ("\nmarcador hueco = día puntuado en operación con la "
                    "previsión publicada" if op is not None else ""),
                 fontsize=10)
    a1.grid(alpha=0.25)

    # ---------------------------------------------------------- panel de abajo
    ref = dict((n, c) for n, c, _o, _k in hay)[REF]
    for nom, col, _col_op, color in hay:
        if nom == REF:
            continue
        dif = (d[col] - d[ref]).values
        m, lo, hi = acumulada(dif, rng)
        a2.plot(d["fecha"], m, color=color, lw=2, label=nom)
        a2.fill_between(d["fecha"], lo, hi, color=color, alpha=0.13, lw=0)
        R[nom].update({"dif_final": float(m[-1]),
                       "ic95_final": [float(lo[-1]), float(hi[-1])],
                       "dias_gana": float((dif > 0).mean()),
                       "significativo": bool(lo[-1] > 0 or hi[-1] < 0),
                       "roza_el_cero": bool(min(abs(lo[-1]), abs(hi[-1])) < 0.005)})
    a2.axhline(0, color="0.35", lw=1)
    a2.set_ylabel("Δ AUC contra producción\n(acumulada, IC95 por bootstrap de días)")
    a2.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    a2.grid(alpha=0.25)

    # ------------------------------------------------------------------- pie
    pie = []
    rj = config.salida("rankings_justo.csv")
    if os.path.exists(rj):
        j = pd.read_csv(rj)
        pie.append(f"cara a cara con previsión real: {len(j)} días "
                   f"(`comparar_rankings_justo.py`)")
    vm = config.salida("veredicto_miteco.csv")
    if os.path.exists(vm):
        m = pd.read_csv(vm)
        cols = [c for c in ("auc10_unico", "auc10_malla") if c in m.columns]
        txt = " · ".join(f"{c[6:]} {m[c].mean():.3f}" for c in cols)
        pie.append(f"juez MITECO: {len(m)} días, {int(m['n_incidentes'].sum())} "
                   f"incidentes · AUC10 medio {txt}")
    if op is None:
        pie.append("juez EFFIS en operación: aún sin días puntuados "
                   "(la cadena empezó el 21/08 y EFFIS tarda 6-9 días)")
    if pie:
        fig.text(0.5, -0.01, "   ·   ".join(pie), ha="center", va="top",
                 fontsize=8, color="0.3")

    fig.tight_layout()
    f = config.salida("dos_19_veredicto.png")
    fig.savefig(f, dpi=150, bbox_inches="tight")
    json.dump(R, open(config.salida("dos_19_veredicto.json"), "w"), indent=1,
              ensure_ascii=False)
    print(f"guardado {f}")
    for n, v in R.items():
        if isinstance(v, dict) and "dif_final" in v:
            print(f"  {n:20s} AUC {v['auc_medio']:.3f} · Δprod {v['dif_final']:+.3f} "
                  f"[{v['ic95_final'][0]:+.3f}, {v['ic95_final'][1]:+.3f}] · "
                  f"gana {v['dias_gana']*100:.0f}% · "
                  + ("roza el cero (no lo cantes)" if v["roza_el_cero"] else
                     ("el IC no toca el cero" if v["significativo"] else "toca el cero")))


if __name__ == "__main__":
    main()
