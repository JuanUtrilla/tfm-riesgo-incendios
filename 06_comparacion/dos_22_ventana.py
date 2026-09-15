#!/usr/bin/env python3
"""
Dos modelos, paso 22: la alerta con ventana. Mide si sirve marcar la zona
con antelación.

dos_20 y dos_21 puntúan el mapa del día D contra el fuego del día D. Un mapa
de riesgo no se usa así: si Guadalajara sale arriba hoy y mañana y arde al
tercer día, el aviso sirvió. Aquí la alerta del día D es la unión del top-k
de los últimos W+1 días, y un incendio cuenta como pillado si alguna de sus
celdas estaba bajo alerta el día en que ardió.

Ojo con el truco: ampliar la ventana siempre sube la captura, porque hay más
territorio bajo alerta. Por eso se compara captura contra coste, y no captura
contra captura: el coste es la fracción media de España bajo alerta ese día.
Una ventana de 3 días con top-1 % que pone el 2,5 % del país en alerta hay
que compararla con el top-2,5 % de un solo día, no con el top-1 %. Si la
curva (coste, captura) de las ventanas se pega a la de un día, la
persistencia no aporta nada: solo estaba ensanchando la mancha.

Se mide también la antelación: cuántos días antes entró la celda en el top-k.

Sigue siendo reanálisis (cota superior, cf. dos_09), no previsión.
No toca producción. Escribe salida/dos_22_ventana.{json,csv,png}.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

import config
import config_expansion as ce
from dos_21_hectareas import CLASES, clase, verdad_por_dia

TOPK = [0.002, 0.005, 0.01, 0.02, 0.05]
VENTANAS = [0, 1, 2, 3, 5, 7]          # días hacia atrás, además del propio D
MAPAS = ["prod", "prod_sin_firms", "donde_dia_effis", "donde_dia_effis_r10",
         "donde_effis_c×cuando_egif", "donde", "fwi_pctl"]


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es = ds["is_spain"].values.astype(bool)
    _, incendios = verdad_por_dia(ds, es)
    ds.close()
    por_dia = {}
    for inc in incendios:
        por_dia.setdefault(inc["fecha"], []).append(inc)

    ficheros = sorted(glob.glob(f"{ce.DATASET}/mapas_2026/*.npz"))
    fechas = [pd.Timestamp(os.path.basename(f)[:10]) for f in ficheros]

    filas = []
    for nom in MAPAS:
        # top-k booleano de cada día, una vez por mapa
        tops = {k: [] for k in TOPK}
        for ruta in ficheros:
            m = np.load(ruta)
            if nom not in m.files:
                tops = None
                break
            v = m[nom]
            ok = np.isfinite(v)
            n = int(ok.sum())
            orden = np.argsort(-v[ok], kind="stable")
            pos = np.empty(n, int)
            pos[orden] = np.arange(n)
            rango = np.full(len(v), n, int)
            rango[ok] = pos
            for k in TOPK:
                tops[k].append(rango < max(1, int(round(k * n))))
        if tops is None:
            continue
        for k in TOPK:
            T = np.array(tops[k])                       # (dias, celdas) bool
            for W in VENTANAS:
                # alerta(D) = unión del top-k en [D-W, D]; días con historia corta se saltan
                coste, cap, ha_in, ha_tot, antel = [], [], 0.0, 0.0, []
                for i, dia in enumerate(fechas):
                    if i < W:
                        continue
                    alerta = T[i - W:i + 1].any(axis=0)
                    coste.append(alerta.mean())
                    for inc in por_dia.get(dia, []):
                        dentro = alerta[inc["celdas"]].any()
                        cap.append((inc["clase"], inc["ha"], bool(dentro)))
                        ha_tot += inc["ha"]
                        if dentro:
                            ha_in += inc["ha"]
                            # primer día de la ventana en que ya estaba marcada
                            for j in range(i - W, i + 1):
                                if T[j][inc["celdas"]].any():
                                    antel.append(i - j)
                                    break
                if not cap:
                    continue
                dcap = pd.DataFrame(cap, columns=["clase", "ha", "dentro"])
                f = dict(mapa=nom, k=k, ventana=W,
                         territorio_alerta=float(np.mean(coste)),
                         cobertura_ha=ha_in / ha_tot if ha_tot else np.nan,
                         antelacion_media=float(np.mean(antel)) if antel else np.nan,
                         n_dias=len(coste))
                for c, _, _ in CLASES:
                    s = dcap[dcap.clase == c]
                    f[f"pilla_{c}"] = float(s.dentro.mean()) if len(s) else np.nan
                    f[f"n_{c}"] = int(len(s))
                filas.append(f)
    df = pd.DataFrame(filas)
    df.to_csv(config.salida("dos_22_ventana.csv"), index=False)
    with open(config.salida("dos_22_ventana.json"), "w") as fh:
        json.dump(df.to_dict("records"), fh, indent=1, ensure_ascii=False)

    for nom in MAPAS:
        g = df[df.mapa == nom]
        if not len(g):
            continue
        print(f"=== {nom}")
        print(f"{'k':>7s} {'vent':>5s} {'España en alerta':>17s} {'cob.HA':>8s} "
              f"{'pilla GRANDE':>13s} {'antelación':>11s}")
        for _, r in g.sort_values(["k", "ventana"]).iterrows():
            print(f"{r.k*100:6.1f}% {int(r.ventana):4d}d {r.territorio_alerta*100:16.1f}% "
                  f"{r.cobertura_ha*100:7.1f}% {r.pilla_GRANDE*100:12.0f}% "
                  f"{r.antelacion_media:10.2f}d")
        print()
    figura(df)


def figura(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ver = ["prod", "donde_dia_effis_r10", "donde_dia_effis", "fwi_pctl"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, campo, tit in [(axes[0], "cobertura_ha", "hectáreas cubiertas"),
                           (axes[1], "pilla_GRANDE", "incendios ≥500 ha pillados")]:
        for nom in ver:
            g = df[df.mapa == nom]
            if not len(g):
                continue
            u = g[g.ventana == 0].sort_values("k")
            ax.plot(u.territorio_alerta * 100, u[campo] * 100, "o-",
                    label=f"{nom} — mismo día")
            for W in [2, 5]:
                w = g[g.ventana == W].sort_values("k")
                ax.plot(w.territorio_alerta * 100, w[campo] * 100, "s--", alpha=.55,
                        color=ax.lines[-1].get_color(),
                        label=f"{nom} — ventana {W}d" if nom == ver[0] else None)
        ax.set_xscale("log")
        ax.set_xlabel("% de España bajo alerta ese día (coste)")
        ax.set_ylabel(f"% de {tit}")
        ax.set_title(f"Captura vs. coste — {tit}", fontsize=10)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("¿Aporta algo mantener la alerta varios días? (2026, EFFIS, reanálisis)")
    fig.tight_layout()
    fig.savefig(config.salida("dos_22_ventana.png"), dpi=130)


if __name__ == "__main__":
    main()
