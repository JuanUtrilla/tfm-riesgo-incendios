#!/usr/bin/env python3
"""
Capa base administrativa de los mapas: provincias, comunidades y ciudades.

No toca producción: solo dibuja, no cambia ninguna probabilidad.

Por qué
-------
Los mapas del repo se leían como una mancha de color sobre el contorno de
España: sin referencias, «¿esto es Zamora o León?» no se contesta a ojo, y el
mapa diario se mira precisamente para eso. Se añaden los límites de provincia
y las ciudades para poder situar el riesgo sin abrir otro mapa al lado.

Los polígonos son los mismos del colector de AEMET (`exportar_limites.py`,
Natural Earth 10 m admin-1, dominio público, simplificados a 800 m y ya
proyectados a píxeles de la malla). Se comprobó que la rejilla es idéntica
(1188 × 920, EPSG:3035, origen 2674734,3466 / 2492195,9911, paso 1 km), así
que el `limites.npz` de allí vale aquí tal cual: 48 KB en `estado/`, sin red
y sin shapely en la cadena diaria.

Las etiquetas se limitan a las N ciudades más pobladas (población de
`estado/municipios.json`, emparejada por cercanía a cada capital) porque las
50 capitales rotuladas tapan el mapa: en un panel grande caben unas 25, en
una rejilla de 2×2 caben unas 12. Los puntos sí se dibujan todos.

Si falta `limites.npz` el mapa sale como antes, con un aviso: la cadena diaria
no se cae por la capa base.
"""

import json
import os

import numpy as np

import config

_CACHE = {}


def cargar():
    """Polilíneas y capitales, con su población. Se lee una vez por proceso."""
    if _CACHE:
        return _CACHE
    ruta = config.entrada("limites.npz")
    if not os.path.exists(ruta):
        _CACHE["falta"] = True
        return _CACHE
    lim = np.load(ruta, allow_pickle=False)
    _CACHE.update({k: lim[k] for k in lim.files})
    # población de cada capital: el municipio de `municipios.json` más cercano
    # a menos de 20 km (evita casar por nombre, que difiere en A Coruña,
    # Lleida/Lérida, Vitoria/Gasteiz...).
    hab = np.zeros(len(_CACHE["cap_x"]))
    mun = config.entrada("municipios.json")
    if os.path.exists(mun):
        from pyproj import Transformer
        m = json.load(open(mun))
        la = np.array([float(x["latitud_dec"]) for x in m])
        lo = np.array([float(x["longitud_dec"]) for x in m])
        nh = np.array([float(x.get("num_hab") or 0) for x in m])
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        mx, my = tr.transform(lo, la)
        # a píxeles con la misma rejilla que las polilíneas (paso 1 km)
        px = (np.asarray(mx) - 2674734.3466) / 1000.0
        py = (np.asarray(my) - 2492195.9911) / -1000.0
        for i, (cx, cy) in enumerate(zip(_CACHE["cap_x"], _CACHE["cap_y"])):
            d = np.hypot(px - cx, py - cy)          # píxeles = km
            j = int(np.argmin(d))
            if d[j] < 20:
                hab[i] = nh[j]
    _CACHE["cap_hab"] = hab
    return _CACHE


def dibujar(ax, nx, ny, etiquetas=20, ciudades=True, lw=1.0):
    """Límites de provincia y CCAA + ciudades sobre un `ax` con la malla ya
    pintada en coordenadas de píxel (imshow con origin='upper').

    etiquetas: cuántas ciudades se rotulan, de más a menos poblada.
    """
    import matplotlib.patheffects as pe
    lim = cargar()
    if lim.get("falta"):
        print("aviso: falta estado/limites.npz, mapa sin capa base", flush=True)
        return
    ax.plot(lim["prov_x"], lim["prov_y"], lw=0.35 * lw, color="0.35", alpha=0.55,
            zorder=2.5, solid_capstyle="round")
    ax.plot(lim["ccaa_x"], lim["ccaa_y"], lw=0.9 * lw, color="0.15", alpha=0.85,
            zorder=2.6, solid_capstyle="round")
    if not ciudades:
        return
    cx, cy, nom, hab = (lim["cap_x"], lim["cap_y"], lim["cap_nombre"],
                        lim["cap_hab"])
    dentro = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
    cx, cy, nom, hab = cx[dentro], cy[dentro], nom[dentro], hab[dentro]
    ax.plot(cx, cy, marker="o", ms=2.4 * lw, mfc="white", mec="black",
            mew=0.6 * lw, ls="", zorder=4)
    for i in np.argsort(-hab)[:etiquetas]:
        ax.annotate(nom[i], (cx[i], cy[i]), xytext=(3, 3),
                    textcoords="offset points", fontsize=6 * lw, color="black",
                    zorder=4, path_effects=[pe.withStroke(linewidth=1.6,
                                                          foreground="white")])


def handles():
    """Artistas ficticios para la leyenda: qué es cada línea y cada punto."""
    from matplotlib.lines import Line2D
    return [
        Line2D([], [], color="0.15", lw=1.4, label="límite de comunidad"),
        Line2D([], [], color="0.35", lw=0.7, label="límite de provincia"),
        Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec="black",
               mew=0.8, label="capital de provincia"),
    ]


def leyenda(fig, extra=(), ncol=5):
    """Una sola leyenda para toda la figura: capa base + lo que añada el llamante.

    Va abajo y en horizontal para no comerse mapa; sin marco, porque el fondo
    de la figura ya es blanco.
    """
    h = handles() + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=ncol, fontsize=8,
               frameon=False, borderaxespad=0.6,
               bbox_to_anchor=(0.5, 0.01))


def niveles_en_barra(cb, cortes=(30, 90, 98), nombres=("BAJO", "MODERADO",
                                                       "ALTO", "EXTREMO")):
    """Rotula las bandas de la barra de color con el nombre de su nivel.

    La barra se dibuja con `spacing='uniform'`: cada tramo ocupa lo mismo
    aunque EXTREMO sea el 2 % del rango, que si no queda una rayita ilegible
    justo en el nivel que más importa. Por eso las posiciones se calculan en
    fracción del eje y no en percentil.
    """
    tramos = cb._boundaries if hasattr(cb, "_boundaries") else None
    n = len(tramos) - 1 if tramos is not None else 6
    lim = [0] + [i for i, b in enumerate(tramos[1:-1], 1)
                 if b in cortes] + [n] if tramos is not None else [0, 2, 4, 5, 6]
    for nom, a, b in zip(nombres, lim[:-1], lim[1:]):
        # a la izquierda de la barra: los números de los cortes van a la
        # derecha y ahí se solapaban ("60MODERADO")
        cb.ax.text(-0.25, (a + b) / 2 / n, nom, transform=cb.ax.transAxes,
                   ha="right", va="center", fontsize=7.5, color="0.2")


def alerta_global(fig, pctl, valor=None, cortes=(30, 90, 98),
                  nombres=("BAJO", "MODERADO", "ALTO", "EXTREMO"),
                  colores=("#ffeda0", "#feb24c", "#fd8d3c", "#bd0026"),
                  rect=(0.055, 0.945, 0.20, 0.016)):
    """Termómetro de la alerta global del día en una esquina de la figura.

    El mapa va en percentil del día, así que por construcción todos los días
    pintan el mismo 2 % en EXTREMO: un 12 de agosto de récord y un martes de
    marzo salen idénticos. El mapa no puede decir si hoy es un día malo, y
    hasta ahora eso solo estaba como cifra en el subtítulo. Aquí es un
    elemento gráfico: el percentil del FWI medio de España contra su propia
    climatología del mes (`dos_14_cortes.py` §ALERTA GLOBAL).
    """
    if pctl is None or not np.isfinite(pctl):
        return
    ax = fig.add_axes(rect)
    bordes = [0] + list(cortes) + [100]
    for c, a, b in zip(colores, bordes[:-1], bordes[1:]):
        ax.axvspan(a, b, color=c, lw=0)
    ax.axvline(pctl, color="black", lw=2.0)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.set_xticks(cortes)
    ax.tick_params(labelsize=6, length=2, pad=1)
    for lado in ax.spines.values():
        lado.set_linewidth(0.5)
    i = int(np.searchsorted(cortes, pctl, side="right"))
    txt = f"ALERTA GLOBAL: {nombres[i]} · FWI en el percentil {pctl:.0f} de su climatología"
    if valor is not None and np.isfinite(valor):
        txt += f" (FWI medio {valor:.1f})"
    ax.set_title(txt, fontsize=7.5, loc="left", pad=3, color="0.2")
