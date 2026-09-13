#!/usr/bin/env python3
"""
El producto final de cada día: el mapa del r10 en escala absoluta con el
semáforo nacional, y al lado el modelo de referencia (producción sobre la
malla) en percentil del día.

Lee lo que dejan `dos_riesgo_hoy.py` (salida/mapas_diarios/dos_riesgo_<f>.npz,
clave prob_r10) y `riesgo_hoy.py` (riesgo_hoy_<f>.npz, clave prob) y escribe
salida/semaforo_<f>.png y salida/semaforo_<f>.json. No recalcula nada.

Las dos reglas vienen de la calibración sobre los diez años del cubo
(05_iteracion2/56_calibracion/):

  escala absoluta  cortes del r10 en su propia puntuación, leídos de la
                   climatología 2015-2024 (dos_27_escala_absoluta.py ->
                   dos_27_cortes.json): MODERADO >= 0,00178, ALTO >= 0,0736,
                   EXTREMO >= 0,4066. EXTREMO es el 2 % de la historia y en
                   esa banda ardió 1 celda de cada 800 (lift 35,6).
  semaforo         el p98 de la puntuacion del r10 en España ese dia, pasado
                   por la calibracion beta del aviso (dos_26_calibra_si.py ->
                   dos_26_calibracion.json, entrada r10|TODO). Si la
                   probabilidad calibrada de "dia con incendio grande" no
                   llega al umbral, el dia no lleva EXTREMO: esas celdas se
                   pintan como ALTO y el mapa se declara apagado. Con esa
                   regla, en invierno el 44 % de los dias quedan sin rojo y
                   se pierde el 6,2 % de los dias grandes.

Uso:
    python 07_produccion/mapa_r10_semaforo.py                 # hoy y mañana
    python 07_produccion/mapa_r10_semaforo.py --fecha 2026-09-11
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

import capa_base
import config

# --- constantes de la calibracion (copiadas de los JSON, con su origen) ------
CORTES_ABS = [0.0017807815593375862, 0.07355870903046814, 0.40661572679295743]  # dos_27_cortes.json, r10
BETA_SI = (1.9364816488092353, 0.005305402168090773, 0.2418997277088091)        # dos_26_calibracion.json, r10|TODO
UMBRAL_SI = 0.15463618712368413                                                 # idem, umbral_prob
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]
COLORES = ["#ffffd9", "#fed976", "#fd8d3c", "#bd0026"]
CORTES_PCTL = [30, 90, 98]


def p_si(p98):
    """Probabilidad calibrada de dia con incendio grande a partir del p98."""
    s = float(np.clip(p98, 1e-9, 1 - 1e-9))
    a, b, c = BETA_SI
    z = a * np.log(s) + b * (-np.log(1 - s)) + c
    return float(1 / (1 + np.exp(-z)))


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


def un_dia(fstr, ds, es_esp, origen):
    md = config.salida("mapas_diarios")
    f_r10 = f"{md}/dos_riesgo_{fstr}.npz"
    if not os.path.exists(f_r10):
        print(f"  {fstr}: no hay dos_riesgo_{fstr}.npz, se salta")
        return None
    r10 = np.load(f_r10)["prob_r10"].astype(float)
    r10[~es_esp] = np.nan
    f_prod = f"{md}/riesgo_hoy_{fstr}.npz"
    prod = None
    if os.path.exists(f_prod):
        prod = np.load(f_prod)["prob"].astype(float)
        prod[~es_esp] = np.nan

    p98 = float(np.nanpercentile(r10, 98))
    prob_si = p_si(p98)
    aviso = prob_si >= UMBRAL_SI
    niv = niveles_abs(r10)
    val = np.isfinite(r10)
    pct_niv_bruto = {NIVELES[i]: float((niv[val] == i).mean() * 100) for i in range(4)}
    niv_pintado = niv.copy()
    if not aviso:
        niv_pintado[niv_pintado == 3] = 2          # sin aviso no hay EXTREMO
    pct_niv = {NIVELES[i]: float((niv_pintado[val] == i).mean() * 100) for i in range(4)}

    info = {}
    fj = config.salida(f"dos_riesgo_{fstr}.json")
    if os.path.exists(fj):
        info = json.load(open(fj))

    res = {"fecha": fstr, "semaforo": "AVISO" if aviso else "SIN AVISO",
           "p98_r10": round(p98, 5), "prob_dia_grande": round(prob_si, 4),
           "umbral": round(UMBRAL_SI, 4), "niveles_pct": pct_niv,
           "niveles_pct_sin_semaforo": pct_niv_bruto,
           "celdas_extremo": int(np.nansum(niv_pintado == 3)),
           "rama": info.get("rama"), "fwi_medio": info.get("fwi_medio"),
           "fwi_medio_pctl_clim": info.get("fwi_medio_pctl_clim"),
           "hay_referencia": prod is not None}
    json.dump(res, open(config.salida(f"semaforo_{fstr}.json"), "w"), indent=1,
              ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(COLORES)
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    ny, nx = r10.shape
    ncol = 2 if prod is not None else 1
    fig, axs = plt.subplots(1, ncol, figsize=(9.2 * ncol, 8.6), squeeze=False)
    fig.subplots_adjust(top=0.93, bottom=0.04, left=0.01, right=0.99, wspace=0.03)
    axs = axs[0]

    ax = axs[0]
    im = ax.imshow(niv_pintado, origin=origen, cmap=cmap, norm=norm, interpolation="nearest")
    capa_base.dibujar(ax, nx, ny, etiquetas=12, lw=0.9)
    ax.set_axis_off()
    ax.set_title("Riesgo de incendio · modelo r10 · escala absoluta\n"
                 + " · ".join(f"{k} {v:.1f} %" for k, v in pct_niv.items()), fontsize=11)
    if aviso:
        txt, fc = (f"SEMÁFORO: AVISO\nprobabilidad calibrada de día con incendio grande "
                   f"{prob_si:.2f} ≥ {UMBRAL_SI:.2f}"), "#bd0026"
    else:
        txt, fc = (f"SEMÁFORO: SIN AVISO · mapa sin nivel EXTREMO\nprobabilidad calibrada "
                   f"de día con incendio grande {prob_si:.2f} < {UMBRAL_SI:.2f}"), "#2c7a3f"
    ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=9.5, color="white",
            va="bottom", ha="left", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec="none", alpha=0.95))
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
                      boundaries=[0, 1, 2, 3, 4], ticks=[0.5, 1.5, 2.5, 3.5])
    cb.ax.set_xticklabels(NIVELES, fontsize=9)
    cb.set_label("cortes fijos en la puntuación del modelo, aprendidos de 2015-2024", fontsize=8.5)

    if prod is not None:
        ax = axs[1]
        nivp = niveles_pctl(prod)
        im2 = ax.imshow(nivp, origin=origen, cmap=cmap, norm=norm, interpolation="nearest")
        capa_base.dibujar(ax, nx, ny, etiquetas=12, lw=0.9)
        ax.set_axis_off()
        ax.set_title("Referencia: modelo de producción (primera iteración) sobre la misma malla\n"
                     "niveles por percentil del día (p30 / p90 / p98)", fontsize=11)
        cb2 = fig.colorbar(im2, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
                           boundaries=[0, 1, 2, 3, 4], ticks=[0.5, 1.5, 2.5, 3.5])
        cb2.ax.set_xticklabels(NIVELES, fontsize=9)
        cb2.set_label("el 2 % más alto de cada día se pinta EXTREMO, sea el día que sea", fontsize=8.5)

    sub = f"{fstr}"
    if info.get("rama"):
        sub += f" · {info['rama']}"
    if info.get("fwi_medio") is not None:
        sub += f" · FWI medio {info['fwi_medio']:.1f} (percentil climático {info['fwi_medio_pctl_clim']:.0f})"
    fig.suptitle(sub, fontsize=12.5, y=0.985)
    fig.text(0.5, 0.005,
             "r10: XGBoost entrenado con celdas quemadas EFFIS 2015-2022, negativos del mismo día (1:10). "
             "Meteorología: ERA5-Land hasta D-7 e IFS hasta D+1 en 5.605 nodos, 498.530 celdas de 1 km.",
             ha="center", va="bottom", fontsize=8, color="0.3")
    fig.savefig(config.salida(f"semaforo_{fstr}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fstr}: {res['semaforo']} · p98 {p98:.3f} · p(día grande) {prob_si:.2f} · "
          f"EXTREMO {pct_niv['EXTREMO']:.2f} % · {config.salida(f'semaforo_{fstr}.png')}", flush=True)
    return res


def main(a):
    if a.fecha:
        fechas = [a.fecha]
    else:
        hoy = pd.Timestamp.now("UTC").tz_localize(None).normalize()
        fechas = [str((hoy + pd.Timedelta(days=h)).date()) for h in a.dias]
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es_esp = ds["is_spain"].values.astype(bool)
    ys = ds["y"].values
    origen = "lower" if ys[1] > ys[0] else "upper"
    ds.close()
    hechos = [r for f in fechas if (r := un_dia(f, None, es_esp, origen))]
    if hechos:
        json.dump(hechos[0], open(config.salida("semaforo_hoy.json"), "w"), indent=1,
                  ensure_ascii=False)
        import shutil
        shutil.copyfile(config.salida(f"semaforo_{hechos[0]['fecha']}.png"),
                        config.salida("semaforo_hoy.png"))
        if len(hechos) > 1:
            shutil.copyfile(config.salida(f"semaforo_{hechos[1]['fecha']}.png"),
                            config.salida("semaforo_manana.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fecha")
    p.add_argument("--dias", type=int, nargs="+", default=[0, 1])
    main(p.parse_args())
