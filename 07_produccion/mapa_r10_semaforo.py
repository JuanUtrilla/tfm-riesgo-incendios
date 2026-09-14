#!/usr/bin/env python3
"""
El producto final de cada día: dos mapas del r10 y la cifra del día.

  izquierda   el r10 en escala absoluta (cuánto riesgo hay), con la cifra del
              día: % de España en EXTREMO y su posición entre los días de
              referencia
  derecha     el r10 por percentil del día (dónde mirar): el 2 % más alto de
              cada día en EXTREMO, sea el día que sea

Lee lo que deja `dos_riesgo_hoy.py` (salida/mapas_diarios/dos_riesgo_<f>.npz,
clave prob_r10) y escribe salida/mapas_<f>.png y salida/mapas_<f>.json, uno por
día (hoy y mañana), además de mapas_hoy.png y mapas_manana.png. No recalcula
nada.

La escala y la referencia están en `escala_servicio.json`, junto a este
script, y se calibraron sobre los mapas SERVIDOS del replay 2025-2026 (250
días, condición IFS): los cortes de la climatología del cubo (dos_27) marcaban
en servicio entre 2 y 10 veces más EXTREMO que en el cubo el mismo mes. Con los
cortes del servicio, en EXTREMO ardió 1 celda-día de cada 760 (13 veces la
media de las dos temporadas).

Hasta el 14/09/2026 este script aplicaba además un semáforo nacional que
retiraba el nivel EXTREMO los días de p98 bajo. Se quitó: en verano ocultaba el
EXTREMO en uno de cada tres días con incendio grande sin mejorar al calendario
(calibracion_si/semaforo_2026-09-14/README.md).

Uso:
    python 07_produccion/mapa_r10_semaforo.py                 # hoy y mañana
    python 07_produccion/mapa_r10_semaforo.py --fecha 2026-09-11
"""
import argparse
import json
import os
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

import capa_base
import config

ESCALA = json.load(open(pathlib.Path(__file__).resolve().parent / "escala_servicio.json",
                        encoding="utf-8"))
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]
CORTES_ABS = [ESCALA["cortes"][n] for n in NIVELES[1:]]
REFERENCIA = np.array(list(ESCALA["referencia_pct_extremo"]["valores"].values()), float)
MEDIANA_REF = float(np.median(REFERENCIA))
COLORES = ["#ffffd9", "#fed976", "#fd8d3c", "#bd0026"]
CORTES_PCTL = [30, 90, 98]


def niveles_abs(p):
    n = np.full(p.shape, np.nan)
    m = np.isfinite(p)
    n[m] = np.digitize(p[m], CORTES_ABS)
    return n


def niveles_pctl(p):
    n = np.full(p.shape, np.nan)
    m = np.isfinite(p)
    v = p[m]
    pct = v.argsort().argsort() / max(v.size - 1, 1) * 100
    n[m] = np.digitize(pct, CORTES_PCTL)
    return n


def un_dia(fstr, es_esp, origen, etiqueta=None):
    md = config.salida("mapas_diarios")
    f_r10 = f"{md}/dos_riesgo_{fstr}.npz"
    if not os.path.exists(f_r10):
        print(f"  {fstr}: no hay dos_riesgo_{fstr}.npz, se salta")
        return None
    r10 = np.load(f_r10)["prob_r10"].astype(float)
    r10[~es_esp] = np.nan
    val = np.isfinite(r10)

    niv = niveles_abs(r10)
    pct_niv = {NIVELES[i]: float((niv[val] == i).mean() * 100) for i in range(4)}
    pct_ext = pct_niv["EXTREMO"]
    posicion = float((REFERENCIA <= pct_ext).mean() * 100)

    info = {}
    fj = config.salida(f"dos_riesgo_{fstr}.json")
    if os.path.exists(fj):
        info = json.load(open(fj))

    res = {"fecha": fstr,
           "pct_extremo": round(pct_ext, 3),
           "posicion_referencia": round(posicion, 1),
           "dias_referencia": int(REFERENCIA.size),
           "mediana_referencia": round(MEDIANA_REF, 3),
           "niveles_pct": pct_niv,
           "celdas_extremo": int(np.nansum(niv == 3)),
           "p98_r10": round(float(np.nanpercentile(r10, 98)), 5),
           "rama": info.get("rama"), "fwi_medio": info.get("fwi_medio"),
           "fwi_medio_pctl_clim": info.get("fwi_medio_pctl_clim")}
    json.dump(res, open(config.salida(f"mapas_{fstr}.json"), "w"), indent=1,
              ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(COLORES)
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    ny, nx = r10.shape
    fig, axs = plt.subplots(1, 2, figsize=(18.4, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.17, left=0.01, right=0.99, wspace=0.03)

    ax = axs[0]
    im = ax.imshow(niv, origin=origen, cmap=cmap, norm=norm, interpolation="nearest")
    capa_base.dibujar(ax, nx, ny, etiquetas=12, lw=0.9)
    ax.set_axis_off()
    ax.set_title("Cuánto riesgo: escala absoluta\n"
                 + " · ".join(f"{k} {v:.1f} %" for k, v in pct_niv.items()), fontsize=11)
    lineas = [f"EXTREMO: {pct_ext:.2f} % de España",
              f"más que el {posicion:.0f} % de los {REFERENCIA.size} días de referencia",
              f"(mediana de la referencia: {MEDIANA_REF:.1f} %)"]
    if info.get("fwi_medio_pctl_clim") is not None:
        lineas.append(f"FWI medio en el percentil {info['fwi_medio_pctl_clim']:.0f} "
                      "de su climatología")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
                      boundaries=[0, 1, 2, 3, 4], ticks=[0.5, 1.5, 2.5, 3.5])
    cb.ax.set_xticklabels(NIVELES, fontsize=9)
    cb.set_label("cortes fijos calibrados con los mapas servidos de 2025-2026", fontsize=8.5)
    # la cifra del día, debajo de la barra de colores: nunca encima del mapa
    cb.ax.text(0.5, -2.3, "  ·  ".join(lineas[:2]) + "\n" + "  ·  ".join(lineas[2:]),
               transform=cb.ax.transAxes, fontsize=10, color="white", va="top", ha="center",
               linespacing=1.5, bbox=dict(boxstyle="round,pad=0.5", fc="#bd0026", ec="none",
                                          alpha=0.95))

    ax = axs[1]
    im2 = ax.imshow(niveles_pctl(r10), origin=origen, cmap=cmap, norm=norm,
                    interpolation="nearest")
    capa_base.dibujar(ax, nx, ny, etiquetas=12, lw=0.9)
    ax.set_axis_off()
    ax.set_title("Dónde mirar: percentil del día\nniveles p30 / p90 / p98", fontsize=11)
    cb2 = fig.colorbar(im2, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
                       boundaries=[0, 1, 2, 3, 4], ticks=[0.5, 1.5, 2.5, 3.5])
    cb2.ax.set_xticklabels(NIVELES, fontsize=9)
    cb2.set_label("el 2 % más alto de cada día, sea el día que sea", fontsize=8.5)

    sub = f"Riesgo de incendio · modelo r10 · {fstr}" + (f" ({etiqueta})" if etiqueta else "")
    if info.get("rama"):
        sub += f" · {info['rama']}"
    if info.get("fwi_medio") is not None:
        sub += (f" · FWI medio {info['fwi_medio']:.1f} "
                f"(percentil climático {info['fwi_medio_pctl_clim']:.0f})")
    fig.suptitle(sub, fontsize=12.5, y=0.985)
    fig.text(0.5, -0.04,
             "r10: XGBoost entrenado con celdas quemadas EFFIS 2015-2022, negativos del mismo "
             "día (1:10). Meteorología: ERA5-Land hasta D-7 e IFS hasta D+1 en 5,605 nodos, "
             "498,530 celdas de 1 km.",
             ha="center", va="bottom", fontsize=8, color="0.3")
    fig.savefig(config.salida(f"mapas_{fstr}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fstr}: EXTREMO {pct_ext:.2f} % (más que el {posicion:.0f} % de la referencia) · "
          f"{config.salida(f'mapas_{fstr}.png')}", flush=True)
    return res


def main(a):
    if a.fecha:
        fechas, etiquetas = [a.fecha], [None]
    else:
        hoy = pd.Timestamp.now("UTC").tz_localize(None).normalize()
        fechas = [str((hoy + pd.Timedelta(days=h)).date()) for h in a.dias]
        etiquetas = [{0: "hoy", 1: "mañana"}.get(h) for h in a.dias]
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es_esp = ds["is_spain"].values.astype(bool)
    ys = ds["y"].values
    origen = "lower" if ys[1] > ys[0] else "upper"
    ds.close()
    hechos = [r for f, e in zip(fechas, etiquetas) if (r := un_dia(f, es_esp, origen, e))]
    if hechos:
        import shutil
        for nombre, r in zip(("hoy", "manana"), hechos):
            json.dump(r, open(config.salida(f"mapas_{nombre}.json"), "w"), indent=1,
                      ensure_ascii=False)
            shutil.copyfile(config.salida(f"mapas_{r['fecha']}.png"),
                            config.salida(f"mapas_{nombre}.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fecha")
    p.add_argument("--dias", type=int, nargs="+", default=[0, 1])
    main(p.parse_args())
