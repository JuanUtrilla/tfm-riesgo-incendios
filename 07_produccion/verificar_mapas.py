#!/usr/bin/env python3
"""
Mapas de verificación: el mapa que se publicó, con lo que pasó ese día encima.

`mapas_hoy_manana.py` publica cada madrugada el mapa limpio de hoy y el de
mañana. La verdad llega después: el parte del MITECO del día D sale el D+1
hacia las 14 h, y los perímetros de EFFIS tardan entre seis y nueve días. Este
script vuelve a dibujar los mapas de los últimos días, sin recalcular nada,
con esa verdad encima, y deja el original intacto. Así cada fecha tiene dos
imágenes: `mapas_<fecha>.png` (lo que se predijo) y `verificado_<fecha>.png`
(lo mismo con lo que ocurrió).

Lo que se dibuja encima:
  · perímetros EFFIS con FIREDATE == D (contorno grueso, relleno) y con
    FIREDATE == D+1 en trazo fino discontinuo, porque FIREDATE es la fecha de
    detección por satélite y un fuego de la tarde de D aparece a menudo con D+1;
  · incidentes del parte del MITECO del día D: un triángulo en el centroide
    del municipio, con un error de 5 a 15 km. Por eso la tabla de verificación
    mira el mejor nivel dentro de un radio de 10 km alrededor del incidente,
    el mismo radio con el que puntuaba `dos_15_veredicto_miteco.py`.
FIRMS no se dibuja: es una variable de entrada del modelo (ver `capa_verdad.py`).

Con `--refrescar` descarga antes los perímetros de la temporada (WFS de EFFIS,
`puntuar_effis.refrescar`) y el parte del MITECO del día (`dos_15.descargar`),
y regenera `miteco_incidentes.csv` a partir de todos los partes guardados.
Ningún fallo de descarga tumba el script: se dibuja con lo que haya.

Uso:
    python 07_produccion/verificar_mapas.py --refrescar --ultimos 20   # la cadena
    python 07_produccion/verificar_mapas.py --fechas 2026-09-05 2026-09-06
Salida: salida/verificado_<fecha>.png y salida/verificacion.csv (una fila por
día: en qué nivel cayó cada verdad).
"""
import argparse
import os
import sys
import types
import numpy as np
import pandas as pd
import xarray as xr

import capa_base
import capa_verdad
import config
import mapas_hoy_manana as mhm

RADIO_MITECO = 10      # km = celdas: el radio con el que puntúa dos_15
ULTIMOS = 20           # días hacia atrás que se regeneran en la cadena
COLS_CSV = ["fecha", "lat", "lon", "localizacion", "provincia", "n_medios"]


def refrescar():
    """EFFIS de la temporada + parte MITECO de hoy + CSV de incidentes.

    `puntuar_effis` y `dos_15` importan `historico`, el acumulado de las
    puntuaciones diarias de la temporada 2026, que no viaja en este repositorio.
    Aquí solo se usan sus funciones de descarga, así que se les da un módulo
    vacío."""
    sys.modules.setdefault("historico", types.SimpleNamespace(anotar=lambda *a, **k: None))
    try:
        import puntuar_effis
        puntuar_effis.refrescar()
    except Exception as e:
        print(f"  EFFIS: no refrescado ({type(e).__name__}: {e})", flush=True)
    try:
        import dos_15_veredicto_miteco as d15
        d15.descargar()
        inc = d15.incidentes()
        inc = inc[[c for c in COLS_CSV if c in inc.columns]].copy()
        inc["fecha"] = pd.to_datetime(inc["fecha"]).dt.strftime("%Y-%m-%d")
        inc.to_csv(capa_verdad.INCIDENTES, index=False)
        print(f"  MITECO: {len(inc)} incidentes en el CSV", flush=True)
    except Exception as e:
        print(f"  MITECO: no refrescado ({type(e).__name__}: {e})", flush=True)


def nivel_en_radio(niv, ix, iy, r=RADIO_MITECO):
    ny, nx = niv.shape
    y0, y1 = max(int(iy) - r, 0), min(int(iy) + r + 1, ny)
    x0, x1 = max(int(ix) - r, 0), min(int(ix) + r + 1, nx)
    v = niv[y0:y1, x0:x1]
    return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan


def verdad(ds, fstr):
    """Las capas del día D y la máscara de D+1, con ventana de un día."""
    m1_d, m_d, fichas = capa_verdad.quemado(ds, fstr, dias=1)
    mx, my, mn = capa_verdad.incidentes(ds, fstr, dias=1)
    d1 = str((pd.Timestamp(fstr) + pd.Timedelta(days=1)).date())
    _, m_d1, _ = capa_verdad.quemado(ds, d1, dias=1)
    capas = {"m7": m1_d, "m1": m_d, "fx": np.array([]), "fy": np.array([]),
             "mx": mx, "my": my, "mn": mn, "effis": fichas}
    return capas, m_d1 & ~m_d


def un_dia(fstr, ds, es_esp, origen):
    f = config.salida(f"mapas_diarios/dos_riesgo_{fstr}.npz")
    if not os.path.exists(f):
        return None
    r10 = np.load(f)["prob_r10"].astype(float)
    r10[~es_esp] = np.nan
    val = np.isfinite(r10)
    niv_abs, niv_pct = mhm.niveles_abs(r10), mhm.niveles_pctl(r10)
    pct_niv = {mhm.NIVELES[i]: float((niv_abs[val] == i).mean() * 100) for i in range(4)}
    pct_ext = pct_niv["EXTREMO"]
    pos = float((mhm.REFERENCIA <= pct_ext).mean() * 100)

    capas, m_d1 = verdad(ds, fstr)
    m_d = capas["m1"]
    fila = {"fecha": fstr, "pct_extremo": round(pct_ext, 2), "posicion": round(pos, 0),
            "celdas_effis_D": int(m_d.sum()), "celdas_effis_D1": int(m_d1.sum()),
            "n_miteco": int(len(capas["mx"]))}
    for nombre, niv in (("abs", niv_abs), ("pct", niv_pct)):
        for tag, m in (("D", m_d), ("D1", m_d1)):
            if m.sum():
                v = niv[m & val]
                fila[f"{nombre}_{tag}_extremo"] = round(float((v == 3).mean() * 100), 1)
                fila[f"{nombre}_{tag}_alto_o_mas"] = round(float((v >= 2).mean() * 100), 1)
        if len(capas["mx"]):
            nm = np.array([nivel_en_radio(niv, x, y) for x, y in zip(capas["mx"], capas["my"])])
            fila[f"{nombre}_miteco_extremo_10km"] = round(float(np.mean(nm == 3) * 100), 1)
            fila[f"{nombre}_miteco_alto_o_mas_10km"] = round(float(np.mean(nm >= 2) * 100), 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(mhm.COLORES)
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    ny, nx = r10.shape
    fig, axs = plt.subplots(1, 2, figsize=(18.4, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.17, left=0.01, right=0.99, wspace=0.03)
    for ax, niv, tit in ((axs[0], niv_abs, "Cuánto riesgo: escala absoluta"),
                         (axs[1], niv_pct, "Dónde mirar: percentil del día")):
        im = ax.imshow(niv, origin=origen, cmap=cmap, norm=norm, interpolation="nearest")
        capa_base.dibujar(ax, nx, ny, etiquetas=12, lw=0.9)
        if m_d1.any():
            ax.contour(m_d1.astype(float), levels=[0.5], colors="#00e5ff", linewidths=0.7,
                       linestyles="dashed", origin=origen)
        capa_verdad.dibujar(ax, capas, lw=1.1, numerar=8)
        ax.set_axis_off()
        ax.set_title(tit, fontsize=11)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
                          boundaries=[0, 1, 2, 3, 4], ticks=[0.5, 1.5, 2.5, 3.5])
        cb.ax.set_xticklabels(mhm.NIVELES, fontsize=9)
    capa_base.leyenda(fig, capa_verdad.handles(capas))
    pie = capa_verdad.resumen(capas, claves=True) or ""
    fig.suptitle(f"Verificación · r10 · {fstr} · EXTREMO {pct_ext:.2f} % de España "
                 f"(más que el {pos:.0f} % de la referencia) · dibujado con la verdad conocida "
                 f"el {pd.Timestamp.now('UTC'):%Y-%m-%d}", fontsize=12.5, y=0.985)
    fig.text(0.5, -0.03,
             "Lo que pasó ese día: perímetros EFFIS con fecha D (contorno grueso) y D+1 (trazo "
             "fino discontinuo); triángulos, incidentes del parte MITECO del día D (centroide "
             "del municipio, 5-15 km de error; se juzgan a 10 km).\n" + pie,
             ha="center", va="top", fontsize=8, color="0.3")
    out = config.salida(f"verificado_{fstr}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fstr}: EFFIS D {fila['celdas_effis_D']} celdas, D+1 {fila['celdas_effis_D1']}, "
          f"MITECO {fila['n_miteco']} · {os.path.basename(out)}", flush=True)
    return fila


def main(a):
    if a.refrescar:
        refrescar()
    if a.fechas:
        fechas = a.fechas
    else:
        hoy = pd.Timestamp.now("UTC").tz_localize(None).normalize()
        fechas = [str((hoy - pd.Timedelta(days=h)).date()) for h in range(a.ultimos, 0, -1)]
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es_esp = ds["is_spain"].values.astype(bool)
    ys = ds["y"].values
    origen = "lower" if ys[1] > ys[0] else "upper"
    filas = [r for f in fechas if (r := un_dia(f, ds, es_esp, origen))]
    ds.close()
    if filas:
        csv = config.salida("verificacion.csv")
        df = pd.DataFrame(filas)
        if os.path.exists(csv):     # se conserva lo anterior y se sustituyen las fechas nuevas
            viejo = pd.read_csv(csv)
            df = pd.concat([viejo[~viejo["fecha"].isin(df["fecha"])], df])
        df.sort_values("fecha").to_csv(csv, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fechas", nargs="+")
    p.add_argument("--ultimos", type=int, default=ULTIMOS)
    p.add_argument("--refrescar", action="store_true")
    main(p.parse_args())
