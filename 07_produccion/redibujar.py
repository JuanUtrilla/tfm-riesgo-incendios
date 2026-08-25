#!/usr/bin/env python3
"""
Re-dibuja mapas ya publicados con el formato definitivo, SIN reejecutar nada.

NO TOCA PRODUCCIÓN, y tampoco toca los PNG publicados: escribe en
`salida/rerender/` y se publica en `publicado/rerender/`. Los originales se
quedan donde están, que son la traza de lo que se vio aquel día.

=============================================================================
POR QUÉ
=============================================================================
Los arreglos de render (escala discreta, `interpolation="nearest"` con dpi 165,
leyenda única, marcas de EFFIS y MITECO rotuladas) entraron en la corrida del
24/08/2026. Los mapas del 21, 22 y 23 se publicaron con el formato viejo, en el
que el remuestreo del render fundía las celdas EXTREMO sueltas con sus vecinas
—el 74 % de las manchas EXTREMO son de 1-2 celdas—, o sea que se borraba la
mayoría de los avisos distintos. Para la memoria hacen falta los seis días con
el mismo formato.

No hay que recalcular nada: `mapas_diarios/*.npz` (Release `estado`) guarda las
probabilidades de los cuatro modelos, y el percentil se recalcula de ahí.

=============================================================================
LA VERDAD QUE SE DIBUJA ES LA DE AQUEL DÍA, NO LA DE HOY
=============================================================================
`capa_verdad` dibuja los incendios que EFFIS tiene cartografiados **en el
momento de dibujar**. Re-dibujar hoy el mapa del 21/08 pintaría encima
incendios que el 21 todavía no se conocían: quedaría un mapa más bonito y una
figura deshonesta, porque el lector juzgaría el aviso con información que el
sistema no tenía.

Así que se filtra por fecha de conocimiento:

  · **EFFIS**: solo los registros con `LASTUPDATE` anterior a la hora de la
    corrida (04:00 UTC del propio día). Es una aproximación conservadora —
    `LASTUPDATE` es la ÚLTIMA actualización, así que un incendio que ya
    existiera pero se haya retocado después queda fuera. Dibuja de menos,
    nunca de más, que es el error que se puede defender.
  · **MITECO**: solo partes de días anteriores. El parte del día D se publica
    el D+1 hacia las 14 h, así que a las 04:00 del día D lo último conocido es
    el parte del D−1.

Uso:
    python redibujar.py --fechas 2026-08-21 2026-08-22 2026-08-23
    python redibujar.py --fechas 2026-08-21 --verdad-de-hoy   # sin filtrar
"""

import argparse
import json
import os
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

import capa_base
import capa_verdad
import config
from dos_riesgo_hoy import (COLORES_MAPA, CORTES_MAPA, CORTES_PCTL, NIVELES,
                            percentil_dia)

HORA_CORRIDA = pd.Timedelta(hours=4)   # la cadena diaria arranca a las 03:30 UTC


def filtrar_verdad(corte, fuentes=(None, None)):
    """Deja `capa_verdad` viendo solo lo que se conocía a la hora de la corrida.

    Devuelve una función para restaurar los caminos originales.
    """
    orig = (capa_verdad.GEOJSON, capa_verdad.INCIDENTES)
    # el catálogo del que se filtra ha de ser el MÁS COMPLETO que haya (el del
    # Release, no la copia local, que puede ir días por detrás): lo que decide
    # qué se dibuja es `LASTUPDATE`, no qué fichero se abra.
    capa_verdad.GEOJSON = fuentes[0] or capa_verdad.GEOJSON
    capa_verdad.INCIDENTES = fuentes[1] or capa_verdad.INCIDENTES
    base = (capa_verdad.GEOJSON, capa_verdad.INCIDENTES)
    tmp = tempfile.mkdtemp(prefix="rerender_")

    if os.path.exists(base[0]):
        g = json.load(open(base[0]))
        n0 = len(g["features"])
        g["features"] = [f for f in g["features"]
                         if pd.to_datetime(f["properties"].get("LASTUPDATE"),
                                           errors="coerce") <= corte]
        ruta = f"{tmp}/effis.geojson"
        json.dump(g, open(ruta, "w"))
        capa_verdad.GEOJSON = ruta
        print(f"  EFFIS conocido el {corte:%Y-%m-%d %H:%M}: "
              f"{len(g['features'])} de {n0} incendios", flush=True)

    if os.path.exists(base[1]):
        df = pd.read_csv(base[1])
        n0 = len(df)
        df = df[pd.to_datetime(df["fecha"]).dt.normalize()
                <= corte.normalize() - pd.Timedelta(days=1)]
        ruta = f"{tmp}/miteco.csv"
        df.to_csv(ruta, index=False)
        capa_verdad.INCIDENTES = ruta
        print(f"  MITECO conocido: {len(df)} de {n0} incidentes", flush=True)

    def restaurar():
        capa_verdad.GEOJSON, capa_verdad.INCIDENTES = orig
    return restaurar


def dibujar(fstr, npz_dir, dest, ds):
    """El mismo lienzo 2×3 de `dos_riesgo_hoy`, a partir del npz del día."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    d = np.load(f"{npz_dir}/dos_riesgo_{fstr}.npz")
    ys = ds["y"].values
    out = {}
    for nom in ("unico", "pareja", "donde", "cuando"):
        clave = f"prob_{nom}"
        if clave not in d:
            continue
        p = d[clave].astype(np.float32)
        out[f"pctl_{nom}"] = percentil_dia(p.ravel()).reshape(p.shape).astype(np.float32)
    rp = f"{npz_dir}/riesgo_hoy_{fstr}.npz"
    if os.path.exists(rp):
        p = np.load(rp)["prob"].astype(np.float32)
        out["pctl_prod_malla"] = percentil_dia(p.ravel()).reshape(p.shape).astype(np.float32)
        for nom in ("unico", "pareja"):
            if f"pctl_{nom}" in out:
                out[f"dif_{nom}"] = out[f"pctl_{nom}"] - out["pctl_prod_malla"]

    res = {}
    rj = config.salida(f"dos_riesgo_{fstr}.json")
    if os.path.exists(rj):
        res = json.load(open(rj))

    paneles = [("pctl_prod_malla", "PRODUCCIÓN (xgb_v2 sobre la malla)"),
               ("pctl_unico", "ÚNICO · etiqueta EFFIS · muestreo del mismo día"),
               ("pctl_pareja", "DÓNDE × CUÁNDO"),
               ("pctl_donde", "DÓNDE · susceptibilidad EFFIS (estático)")]
    paneles = [p for p in paneles if p[0] in out]
    difs = [("dif_unico", "ÚNICO − PRODUCCIÓN"), ("dif_pareja", "PAREJA − PRODUCCIÓN")]
    difs = [p for p in difs if p[0] in out]

    cmap_niv = ListedColormap(COLORES_MAPA)
    norm_niv = BoundaryNorm(CORTES_MAPA, cmap_niv.N)
    capas = capa_verdad.preparar(ds, fstr)
    nc = 3 if difs else 2
    fig, axs = plt.subplots(2, nc, figsize=(8 * nc, 12))
    rejilla = list(axs.ravel())
    ejes_niv = rejilla[:len(paneles)]
    ejes_dif = rejilla[len(paneles):len(paneles) + len(difs)]
    orig = "lower" if ys[1] > ys[0] else "upper"
    for ax, (key, tit) in zip(ejes_niv, paneles):
        im = ax.imshow(out[key], origin=orig, cmap=cmap_niv, norm=norm_niv,
                       interpolation="nearest")
        capa_base.dibujar(ax, out[key].shape[1], out[key].shape[0],
                          etiquetas=12, lw=0.9)
        capa_verdad.dibujar(ax, capas, lw=0.9, numerar=6)
        ax.set_title(tit, fontsize=11); ax.set_axis_off()
    for ax, (key, tit) in zip(ejes_dif, difs):
        imd = ax.imshow(out[key], origin=orig, cmap="RdBu_r", vmin=-40, vmax=40,
                        interpolation="nearest")
        capa_base.dibujar(ax, out[key].shape[1], out[key].shape[0], etiquetas=0, lw=0.9)
        capa_verdad.dibujar(ax, capas, lw=0.9, numerar=6)
        ax.set_title(tit, fontsize=11); ax.set_axis_off()
    for ax in rejilla[len(paneles) + len(difs):]:
        ax.set_axis_off()
    cb = fig.colorbar(im, ax=ejes_niv[:nc], shrink=0.85, boundaries=CORTES_MAPA,
                      ticks=CORTES_MAPA, spacing="uniform")
    cb.set_label("percentil del día")
    capa_base.niveles_en_barra(cb, cortes=CORTES_PCTL, nombres=NIVELES)
    if ejes_dif:
        cbd = fig.colorbar(imd, ax=ejes_dif, orientation="horizontal", shrink=0.75,
                           pad=0.03, extend="both")
        cbd.set_label("puntos de percentil de diferencia · azul: el candidato "
                      "baja el riesgo · rojo: lo sube", fontsize=9)
    capa_base.leyenda(fig, capa_verdad.handles(capas))
    pie = capa_verdad.resumen(capas, claves=True)
    pie = ((pie + "\n") if pie else "") + (
        "re-render con el formato definitivo · la verdad dibujada es la que se "
        "conocía a las 04:00 UTC de ese día · el PNG original se conserva en publicado/")
    fig.text(0.5, -0.005, pie, ha="center", va="top", fontsize=7.5,
             color="0.25", linespacing=1.5)
    if res:
        capa_base.alerta_global(fig, res.get("fwi_medio_pctl_clim"), res.get("fwi_medio"))
        nv = res.get("niveles_unico", {})
        fig.suptitle(f"{fstr} · {res.get('rama','')} · FWI medio {res['fwi_medio']:.1f} "
                     f"(pctl clim {res['fwi_medio_pctl_clim']:.0f}) · único: "
                     + " · ".join(f"{k} {v:.1f}%" for k, v in nv.items()), fontsize=12)
    else:
        fig.suptitle(fstr, fontsize=12)
    fig.savefig(f"{dest}/dos_riesgo_{fstr}.png", dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"  guardado {dest}/dos_riesgo_{fstr}.png", flush=True)


def main(a):
    npz_dir = a.npz or config.salida("mapas_diarios")
    dest = a.dest or config.salida("rerender")
    os.makedirs(dest, exist_ok=True)
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    for fstr in a.fechas:
        print(f"\n=== {fstr} ===", flush=True)
        restaurar = (lambda: None) if a.verdad_de_hoy else \
            filtrar_verdad(pd.Timestamp(fstr) + HORA_CORRIDA, (a.effis, a.miteco))
        try:
            dibujar(fstr, npz_dir, dest, ds)
        finally:
            restaurar()
    ds.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fechas", nargs="+", required=True)
    p.add_argument("--npz")
    p.add_argument("--dest")
    p.add_argument("--effis", help="geojson EFFIS del que filtrar (por defecto el de salida/)")
    p.add_argument("--miteco", help="csv de incidentes del que filtrar")
    p.add_argument("--verdad-de-hoy", action="store_true",
                   help="no filtrar: dibuja lo que EFFIS sabe HOY (no usar para la memoria)")
    main(p.parse_args())
