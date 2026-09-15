#!/usr/bin/env python3
"""
La comparación que decide: malla contra producción, el mismo día.

No toca producción. Lee el .npz que deja `mapa_riesgo_hoy.py` y el que deja
`riesgo_hoy.py`. Escribe salida/comparacion_<fecha>.png y .json.

Qué añade esto a lo ya medido
-----------------------------
Todo lo anterior compara la malla contra el cubo en 2024. Eso demuestra que
la meteo de los nodos reproduce la referencia con la que se entrenó el modelo,
pero no dice nada sobre el sistema que está en producción hoy, que usa otra
fuente (estaciones AEMET), otra interpolación (IDW k=8) y otro denominador
(la climatología del cubo).

Aquí se comparan los dos mapas sobre las mismas celdas y el mismo día. Es lo
único que responde "¿qué cambia para quien mira el mapa?".

Qué se puede y qué no se puede comparar
---------------------------------------
Producción solo guarda la probabilidad, no las features. Así que aquí no se
puede contrastar `fwi_pctl_local` entre los dos (para eso están los módulos
3c y 02b), solo la salida: la probabilidad por celda y el reparto por niveles,
que es lo que se publica.

Tampoco se mide acierto. La verdad-terreno del día son las detecciones FIRMS,
pero FIRMS NRT entra como feature en los dos mapas: usarla de referencia sería
circular. Esto mide acuerdo, no habilidad. La habilidad está medida en
`malla_05b_evaluacion.py` con FIRMS del propio día sobre 2024.

Uso: python comparar_produccion.py [AAAA-MM-DD]
"""

import json
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config

CORTES = [-1, 0.25, 0.55, 0.80, 2]
NIVELES = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]


def main():
    fecha = sys.argv[1] if len(sys.argv) > 1 else \
        str(pd.Timestamp.utcnow().tz_localize(None).date())
    prod = f"{config.FUENTE}/prototipo/cache/malla_prob_{fecha}.npz"
    mio = config.salida(f"riesgo_hoy_{fecha}.npz")
    P = np.load(prod)["prob"].astype(float)
    M = np.load(mio)["prob"].astype(float)

    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es_esp = ds["is_spain"].values.astype(bool)
    ys = ds["y"].values
    ds.close()

    ok = es_esp & np.isfinite(P) & np.isfinite(M)
    p, m = P[ok], M[ok]
    print(f"Comparación {fecha} · {ok.sum():,} celdas peninsulares\n")
    print("=" * 66)
    print(f"{'':<14}{'media':>9}{'mediana':>10}{'p95':>9}{'p99':>9}")
    for nom, v in [("producción", p), ("malla", m)]:
        print(f"{nom:<14}{v.mean():>9.3f}{np.median(v):>10.3f}"
              f"{np.percentile(v, 95):>9.3f}{np.percentile(v, 99):>9.3f}")

    r = float(np.corrcoef(p, m)[0, 1])
    rs = float(pd.Series(p).corr(pd.Series(m), method="spearman"))
    dif = m - p
    print(f"\n  correlación Pearson  {r:.4f}")
    print(f"  correlación Spearman {rs:.4f}   ← el ranking, que es lo que se publica")
    print(f"  diferencia malla − producción: media {dif.mean():+.3f} · "
          f"mediana {np.median(dif):+.3f}")
    print(f"  |diferencia| > 0,20 en {(np.abs(dif) > 0.20).mean() * 100:.1f} % "
          f"de las celdas")

    print("\n" + "=" * 66)
    print("REPARTO POR NIVELES (lo que se publica)")
    print("=" * 66)
    print(f"{'nivel':<12}{'producción':>13}{'malla':>10}{'dif':>9}")
    res = {"fecha": fecha, "celdas": int(ok.sum()), "pearson": r,
           "spearman": rs, "dif_media": float(dif.mean()), "niveles": {}}
    cp = pd.cut(pd.Series(p), CORTES, labels=NIVELES).value_counts()
    cm = pd.cut(pd.Series(m), CORTES, labels=NIVELES).value_counts()
    for n in NIVELES:
        a, b = cp[n] / len(p) * 100, cm[n] / len(m) * 100
        print(f"{n:<12}{a:>12.1f}%{b:>9.1f}%{b - a:>+9.1f}")
        res["niveles"][n] = {"produccion": float(a), "malla": float(b)}

    # ¿cuántas celdas cambian de nivel, y hacia dónde?
    lp = pd.cut(pd.Series(p), CORTES, labels=NIVELES).cat.codes.values
    lm = pd.cut(pd.Series(m), CORTES, labels=NIVELES).cat.codes.values
    cambia = lp != lm
    print(f"\n  celdas que cambian de nivel: {cambia.mean() * 100:.1f} % "
          f"(sube {((lm > lp).mean() * 100):.1f} %, baja "
          f"{((lm < lp).mean() * 100):.1f} %)")
    print(f"  saltos de más de un nivel: {(np.abs(lm - lp) > 1).mean() * 100:.1f} %")
    res["cambia_nivel_pc"] = float(cambia.mean() * 100)
    res["salto_doble_pc"] = float((np.abs(lm - lp) > 1).mean() * 100)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    orig = "lower" if ys[1] > ys[0] else "upper"
    D = np.where(es_esp, M - P, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(26, 8))
    for a, (X, t, kw) in zip(ax, [
            (P, "producción — estaciones AEMET + IDW k=8",
             dict(cmap="YlOrRd", vmin=0, vmax=1)),
            (M, "malla — 5.605 nodos ERA5-Land, sin IDW",
             dict(cmap="YlOrRd", vmin=0, vmax=1)),
            (D, "malla − producción", dict(cmap="RdBu_r", vmin=-.5, vmax=.5))]):
        im = a.imshow(X, origin=orig, **kw)
        a.set_title(t)
        a.set_axis_off()
        fig.colorbar(im, ax=a, fraction=0.035)
    fig.suptitle(f"Riesgo de incendio {fecha} · Pearson {r:.3f} · "
                 f"Spearman {rs:.3f} · cambian de nivel "
                 f"{cambia.mean() * 100:.1f} % de las celdas")
    salida = config.salida(f"comparacion_{fecha}.png")
    fig.savefig(salida, dpi=120, bbox_inches="tight")
    with open(config.salida(f"comparacion_{fecha}.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado: {salida}")


if __name__ == "__main__":
    main()
