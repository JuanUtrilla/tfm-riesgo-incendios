#!/usr/bin/env python3
"""F6 - De donde saca cada modelo su ventaja, y la comprobacion de desfase.

(a) Importancia por *gain* de los dos modelos, leida de los .ubj publicados en
    `muestras/modelos/`. `donde_dia_effis` se guardo sin nombres de columna:
    los indices f0..f45 se traducen con la lista canonica
    `muestras/modelos/xgb_v2_prototipo_features.json`, que es la que
    `dos_05_modelos.py:76` usa como FULL para entrenarlo.
    Los porcentajes se escriben ademas en figs/f6_gain.json.
(b) Fraccion de superficie quemada del dia que cae en el 2 % de celdas de mayor
    riesgo, en funcion del desfase entre el mapa y el dia del fuego
    (`salida/dos_23_vispera.json`, corrida limpia).
Uso:  python f6_donde_mira.py   [TFM_MALLA=<ruta a TFM_fuego_malla>]
"""
import json
import os
import pathlib

import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt

from comun import (ANCHO, AZUL, BERMELLON, GRIS, NEGRO, VERDE, MALLA, REPO,
                   SALIDA, guarda)

MODELOS = REPO / "muestras" / "modelos"
VISPERA = pathlib.Path(os.environ.get("TFM_MALLA", MALLA)) / "salida" / "dos_23_vispera.json"
ETIQ = {
    "fwi_anom_sigma": "FWI, anomalía en σ",
    "n_fuegos_10km_mismomes_hist": "fuegos históricos a 10 km,\nmismo mes",
    "fwi_pctl_local": "FWI en percentil local",
    "fwi": "FWI absoluto",
    "rh_min": "humedad relativa mínima",
    "frp_max_50km_7d": "FIRMS, potencia máxima\na 50 km",
    "n_detec_50km_7d": "FIRMS, detecciones a 50 km",
    "pendiente": "pendiente del terreno",
    "clc_agricola": "CORINE, agrícola",
    "clc_matorral": "CORINE, matorral",
    "clc_bosque": "CORINE, bosque",
    "mes": "mes",
    "lst": "temperatura de superficie",
    "ndvi": "NDVI",
    "elevacion": "elevación",
    "dia_anio": "día del año",
}


def gain(ruta, nombres):
    b = xgb.Booster(); b.load_model(str(ruta))
    g = b.get_score(importance_type="gain")
    if b.feature_names is None:
        g = {nombres[int(k[1:])]: v for k, v in g.items()}
    tot = sum(g.values())
    return {k: 100 * v / tot for k, v in g.items()}


def main():
    nombres = json.load(open(MODELOS / "xgb_v2_prototipo_features.json"))
    prod = gain(MODELOS / "xgb_v2_prototipo.ubj", nombres)
    unico = gain(MODELOS / "donde_dia_effis.ubj", nombres)
    json.dump({"xgb_v2_prototipo": prod, "donde_dia_effis": unico},
              open(SALIDA / "f6_gain.json", "w"), indent=1, ensure_ascii=False)

    orden = sorted(set(list(prod)[:0] + nombres),
                   key=lambda n: -(prod.get(n, 0) + unico.get(n, 0)))[:9]
    orden = orden[::-1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(ANCHO, 3.1),
                                 gridspec_kw=dict(width_ratios=[1.35, 1]))
    y = np.arange(len(orden)); w = 0.38
    a1.barh(y - w / 2, [prod.get(n, 0) for n in orden], w, color=AZUL,
            label="producción (iteración 1)")
    a1.barh(y + w / 2, [unico.get(n, 0) for n in orden], w, color=BERMELLON,
            label="único (iteración 2)")
    a1.set_yticks(y); a1.set_yticklabels([ETIQ.get(n, n) for n in orden], fontsize=7)
    a1.set_xlabel("importancia por gain (% del total)")
    a1.legend(frameon=False, loc="lower right", fontsize=7.2)
    a1.grid(axis="y", visible=False)
    a1.set_title("a) qué mira cada modelo", fontsize=8.2, loc="left")

    v = json.load(open(VISPERA))
    series = [("prod", "producción", AZUL, "o"),
              ("donde_dia_effis", "único", BERMELLON, "s"),
              ("donde", "mapa estático", VERDE, "^"),
              ("fwi_pctl", "FWI percentil", GRIS, "D")]
    for clave, etiq, color, marca in series:
        r = sorted([x for x in v if x["mapa"] == clave and x["k"] == 0.02],
                   key=lambda x: x["desfase"])
        a2.plot([x["desfase"] for x in r], [100 * x["cobertura_ha"] for x in r],
                marker=marca, markersize=3.6, linewidth=1.2, color=color, label=etiq)
    a2.set_xticks([0, 1, 2, 3])
    a2.set_xlabel("desfase mapa − fuego (días)")
    a2.set_ylabel("% de hectáreas quemadas\ndentro del 2 % de mayor riesgo")
    a2.set_ylim(0, 56)
    a2.legend(fontsize=6.9, loc="upper right", facecolor="white",
              framealpha=0.9, edgecolor="none")
    a2.set_title("b) comprobación de desfase", fontsize=8.2, loc="left")

    fig.subplots_adjust(wspace=0.55)
    guarda(fig, "f6_donde_mira.png")


if __name__ == "__main__":
    main()
