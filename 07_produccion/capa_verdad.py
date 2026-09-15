#!/usr/bin/env python3
"""
La verdad del día encima del mapa: perímetros EFFIS y focos FIRMS.

No toca producción: solo dibuja, y ninguna de estas capas entra en el modelo.

Por qué
-------
El mapa diario se mira para saber si el sistema va acertando, y hasta ahora
había que abrir el mapa por un lado y `puntuacion_effis.csv` por otro. Con la
verdad dibujada encima se contesta de un vistazo: si el contorno del incendio
cae sobre la mancha roja, el mapa acertó ese día.

Son dos capas y se leen distinto:

  · Perímetros EFFIS (contorno) de los últimos días: superficie quemada con
    FIREDATE conocida, la misma verdad-terreno que puntuaba `puntuar_effis.py`
    durante la temporada. Llega con retraso (de horas a días), así que sobre
    el mapa de hoy casi siempre estará vacío y lo que se ve es el pasado
    reciente: sirve para contrastar los mapas de ayer y anteayer, que son los
    que ya se publicaron.
  · Incidentes de MITECO (triángulos) del último parte publicado. Es la
    referencia rápida: el parte del día D sale el D+1 hacia las 14 h, así que
    cierra el bucle en 24 h mientras EFFIS tarda 6-9 días. Solo trae los
    incendios con medios del Estado (unos 4 al día) y la posición es el centro
    del municipio, con 5-15 km de error: es un punto, no un perímetro, y así
    se dibuja.

FIRMS no se dibuja (`FOCOS_FIRMS = False`). Se probó y se quitó a propósito:
las detecciones VIIRS entran como variable del modelo (`frp_max_50km_7d`,
`n_detec_50km_7d`), así que pintarlas encima invita a leer «foco sobre rojo =
acierto» cuando el modelo ya las había visto al puntuar; es la misma
circularidad que se evita en las métricas, colada por el mapa. Y tapan: unas
1.000 cruces contra unas 100 celdas de EFFIS. El código se queda por si alguna
vez interesa un mapa de situación (fuego activo en este momento, que eso sí lo
dice FIRMS y no lo dicen las otras dos fuentes), pero para contrastar el mapa
solo valen fuentes independientes del modelo.

Todas las capas son opcionales: si falta el GeoJSON de EFFIS o el CSV de
incidentes, el mapa sale sin ellas y con un aviso. La cadena diaria no se cae
por un adorno.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

import config

GEOJSON = config.salida("effis_ba_season_ES.geojson")
DIAS_EFFIS = 7        # ventana de perímetros dibujados
DIAS_MITECO = 3       # ventana de incidentes del parte
DIAS_FIRMS = 2        # ventana de focos activos (solo si FOCOS_FIRMS)
FOCOS_FIRMS = False   # FIRMS es variable del modelo, no juez: ver la cabecera
INCIDENTES = config.salida("miteco_incidentes.csv")


def quemado(ds, fecha, dias=DIAS_EFFIS):
    """Máscara de celdas con FIREDATE en (fecha − dias, fecha], la del propio
    día (que se dibuja más marcada) y la ficha de cada incendio: dónde cae su
    centro y cómo se llama, para poder rotularlo en el mapa."""
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    vacia = np.zeros((ny, nx), bool)
    if not os.path.exists(GEOJSON):
        print("  capa verdad: sin GeoJSON de EFFIS", flush=True)
        return vacia, vacia, []
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        g = gpd.read_file(GEOJSON)
        col = "FIREDATE" if "FIREDATE" in g.columns else "firedate"
        g["f"] = pd.to_datetime(g[col], errors="coerce", utc=True) \
                   .dt.tz_localize(None).dt.normalize()
        xs, ys = ds["x"].values, ds["y"].values
        px, ay = xs[1] - xs[0], abs(ys[1] - ys[0])
        # esquina noroeste y paso vertical positivo, como en `quemadas()`
        tr = from_origin(xs[0] - px / 2, max(ys[0], ys[-1]) + ay / 2, px, ay)

        def masc(sel):
            if not len(sel):
                return vacia
            sel = sel.set_crs(4326, allow_override=True).to_crs(3035)
            m = rasterize(((geom, 1) for geom in sel.geometry if geom is not None),
                          out_shape=(ny, nx), transform=tr, fill=0,
                          all_touched=True).astype(bool)
            return m[::-1] if ys[1] > ys[0] else m

        fecha = pd.Timestamp(fecha).normalize()
        ventana = g[(g["f"] > fecha - pd.Timedelta(days=dias)) & (g["f"] <= fecha)]
        return masc(ventana), masc(g[g["f"] == fecha]), fichas(ventana, xs, ys)
    except Exception as e:                      # geopandas/rasterio ausentes, etc.
        print(f"  capa verdad: EFFIS no dibujado ({type(e).__name__})", flush=True)
        return vacia, vacia, []


def fichas(sel, xs, ys):
    """(x, y, texto, ha) de cada incendio EFFIS, de mayor a menor superficie."""
    if not len(sel):
        return []
    sel = sel.set_crs(4326, allow_override=True).to_crs(3035)
    ha = pd.to_numeric(sel["AREA_HA"], errors="coerce").fillna(0)
    c = sel.geometry.representative_point()
    ix = (c.x.values - xs[0]) / (xs[1] - xs[0])
    iy = (c.y.values - ys[0]) / (ys[1] - ys[0])
    nom = sel["COMMUNE"].fillna(sel["PROVINCE"]).astype(str).values
    o = np.argsort(-ha.values)
    return [(float(ix[k]), float(iy[k]),
             f"{nom[k]} · {ha.values[k]:,.0f} ha".replace(",", "."),
             float(ha.values[k])) for k in o]


def focos(ds, dias=DIAS_FIRMS):
    """Detecciones FIRMS NRT recientes en píxeles de malla, de la caché que
    deja `riesgo_hoy.firms_nrt` (no se vuelve a llamar a la API)."""
    cs = sorted(glob.glob(config.salida("_firms/nrt_*.csv")))
    if cs and len(cs) > 10:                     # la caché no crece sin fin
        for viejo in cs[:-10]:
            os.remove(viejo)
        cs = cs[-10:]
    if not cs:
        return np.array([]), np.array([])
    df = pd.read_csv(cs[-1])
    if not len(df):
        return np.array([]), np.array([])
    corte = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=dias)
    df = df[pd.to_datetime(df["acq_date"]) >= corte]
    if not len(df):
        return np.array([]), np.array([])
    from pyproj import Transformer
    xs, ys = ds["x"].values, ds["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    fx, fy = tr.transform(df["longitude"].values, df["latitude"].values)
    ix = (np.asarray(fx) - xs[0]) / (xs[1] - xs[0])
    iy = (np.asarray(fy) - ys[0]) / (ys[1] - ys[0])
    ok = (ix >= 0) & (ix < ds.sizes["x"]) & (iy >= 0) & (iy < ds.sizes["y"])
    ix, iy = ix[ok], iy[ok]
    # La consulta a FIRMS cubre -10,35,5,44: entran Marruecos, Argelia y las
    # llamaradas de barcos y plataformas en mar abierto. Como variable eso da
    # igual (el modelo mira un radio de 50 km desde celda española), pero
    # dibujado ensucia el mapa. Se dejan solo los focos a <50 km de España,
    # que incluye los de Portugal pegados a la raya.
    cerca = maximum_filter(ds["is_spain"].values.astype(bool), size=101,
                           mode="constant")
    dentro = cerca[np.rint(iy).astype(int), np.rint(ix).astype(int)]
    return ix[dentro], iy[dentro]


def incidentes(ds, fecha, dias=DIAS_MITECO):
    """Incidentes del parte de MITECO en píxeles de malla, del CSV que deja
    `dos_15_veredicto_miteco.py` (no se reparsean los PDF: es lo caro)."""
    if not os.path.exists(INCIDENTES):
        return np.array([]), np.array([]), np.array([])
    df = pd.read_csv(INCIDENTES)
    if not len(df):
        return np.array([]), np.array([]), np.array([])
    f = pd.to_datetime(df["fecha"]).dt.normalize()
    fecha = pd.Timestamp(fecha).normalize()
    df = df[(f > fecha - pd.Timedelta(days=dias)) & (f <= fecha)]
    if not len(df):
        return np.array([]), np.array([]), np.array([])
    from pyproj import Transformer
    xs, ys = ds["x"].values, ds["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    mx, my = tr.transform(df["lon"].values, df["lat"].values)
    ix = (np.asarray(mx) - xs[0]) / (xs[1] - xs[0])
    iy = (np.asarray(my) - ys[0]) / (ys[1] - ys[0])
    ok = (ix >= 0) & (ix < ds.sizes["x"]) & (iy >= 0) & (iy < ds.sizes["y"])
    return ix[ok], iy[ok], df["localizacion"].astype(str).values[ok]


def preparar(ds, fecha):
    """Todo lo caro (leer el GeoJSON, proyectar) una vez, para N paneles.

    Nunca levanta: esto es un adorno del PNG y la cadena diaria no puede
    caerse porque EFFIS no responda o falte una caché."""
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    vacio = {"m7": np.zeros((ny, nx), bool), "m1": np.zeros((ny, nx), bool),
             "fx": np.array([]), "fy": np.array([]),
             "mx": np.array([]), "my": np.array([]), "mn": np.array([]),
             "effis": []}
    try:
        m7, m1, fichas_effis = quemado(ds, fecha)
        mx, my, mn = incidentes(ds, fecha)
        fx, fy = focos(ds) if FOCOS_FIRMS else (np.array([]), np.array([]))
    except Exception as e:
        print(f"  capa verdad: no se pudo preparar ({type(e).__name__}: {e})",
              flush=True)
        return vacio
    print(f"  capa verdad: EFFIS {int(m7.sum())} celdas en {DIAS_EFFIS} días "
          f"({int(m1.sum())} del propio día) · MITECO {len(mx)} incidentes "
          f"[{DIAS_MITECO} días]"
          + (f" · FIRMS {len(fx)} focos" if FOCOS_FIRMS else ""), flush=True)
    return {"m7": m7, "m1": m1, "fx": fx, "fy": fy,
            "mx": mx, "my": my, "mn": mn, "effis": fichas_effis}


def dibujar(ax, capas, lw=1.0, focos_visibles=True, rotular=0, numerar=0):
    """Perímetros y focos sobre un `ax` con la malla ya pintada.

    rotular: cuántos incendios EFFIS se identifican con su nombre (los mayores
    por superficie) y si se ponen los municipios de MITECO. En el mapa de un
    panel se usa 6.
    numerar: alternativa para la rejilla de 2×2, donde el nombre completo
    saldría cuatro veces y no cabe: al lado de cada marca va una clave corta
    (número para EFFIS, letra para MITECO) y el nombre se resuelve en el pie
    de la figura con `resumen(capas, claves=True)`.
    """
    try:
        _dibujar(ax, capas, lw, focos_visibles, rotular, numerar)
    except Exception as e:
        print(f"  capa verdad: no dibujada ({type(e).__name__}: {e})", flush=True)


def _dibujar(ax, capas, lw, focos_visibles, rotular=0, numerar=0):
    import matplotlib.patheffects as pe
    # cian con filo negro: es el único color que se ve tanto sobre el amarillo
    # pálido del BAJO como sobre el rojo oscuro del EXTREMO (el morado que se
    # probó primero desaparecía justo donde más importa, encima del rojo).
    filo = [pe.withStroke(linewidth=1.6 * lw, foreground="black", alpha=0.6)]
    # burbuja proporcional: sin ella, 1 ha y 6.780 ha se dibujan igual porque
    # a 1 km el perímetro de los pequeños es un píxel
    ef = capas.get("effis", [])
    if ef:
        r = np.array([radio_pt(h) for _x, _y, _t, h in ef]) * lw
        ax.scatter([x for x, _y, _t, _h in ef], [y for _x, y, _t, _h in ef],
                   s=(2 * r) ** 2, facecolors="none", edgecolors="#00E5FF",
                   linewidths=0.9 * lw, alpha=0.9, zorder=4.8)
    for m, ancho, alfa in ((capas["m7"], 1.0, 0.8), (capas["m1"], 1.8, 1.0)):
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=["#00E5FF"],
                       linewidths=ancho * lw, alpha=alfa, zorder=5).set(
                           path_effects=filo)
    if len(capas.get("mx", [])):
        # círculo de 10 km, que es el radio con el que `dos_15` decide si el
        # incidente cayó en zona de riesgo: el punto solo fingía una precisión
        # que no existe (la posición es el centro del municipio, 5-15 km)
        from matplotlib.patches import Circle
        for x, y in zip(capas["mx"], capas["my"]):
            ax.add_patch(Circle((x, y), RADIO_MITECO, fill=False,
                                edgecolor="#111111", lw=0.9 * lw, alpha=0.85,
                                zorder=5.9))
        ax.plot(capas["mx"], capas["my"], ls="", marker="v", ms=4.0 * lw,
                mfc="#111111", mec="white", mew=0.7 * lw, alpha=0.95, zorder=6)
    if rotular or numerar:
        texto = dict(zorder=7, fontweight="bold" if numerar else "normal",
                     fontsize=(7.0 if numerar else 6.5) * lw,
                     path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        n_ef = numerar or rotular
        for i, (x, y, nom, _ha) in enumerate(capas.get("effis", [])[:n_ef], 1):
            ax.annotate(str(i) if numerar else nom, (x, y), xytext=(4, -8),
                        textcoords="offset points", color="#006B7A", **texto)
        for i, (x, y, nom) in enumerate(zip(capas.get("mx", []),
                                            capas.get("my", []),
                                            capas.get("mn", []))):
            ax.annotate(CLAVES[i % len(CLAVES)] if numerar else str(nom).title(),
                        (x, y), xytext=(4, 4), textcoords="offset points",
                        color="#111111", **texto)
    if focos_visibles and FOCOS_FIRMS and len(capas["fx"]):
        ax.plot(capas["fx"], capas["fy"], ls="", marker="x", ms=2.0 * lw,
                mew=0.6 * lw, color="black", alpha=0.7, zorder=5.5)


def handles(capas):
    """Artistas ficticios para la leyenda, solo de lo que se ha dibujado."""
    from matplotlib.lines import Line2D
    h = []
    if capas["m7"].any():
        h.append(Line2D([], [], color="#00E5FF", lw=1.8,
                        label=f"perímetro EFFIS · {DIAS_EFFIS} días "
                              "(grueso: del día)"))
        for ha in HA_LEYENDA:                   # escala de la burbuja
            h.append(Line2D([], [], ls="", marker="o", mfc="none",
                            mec="#00E5FF", mew=0.9, ms=2 * radio_pt(ha),
                            label=f"{ha:,} ha".replace(",", ".")))
    if len(capas.get("mx", [])):
        h.append(Line2D([], [], ls="", marker="v", ms=6, mfc="#111111",
                        mec="white", mew=0.8,
                        label=f"incidente MITECO · {DIAS_MITECO} días · "
                              f"círculo de {RADIO_MITECO} km (el radio con que "
                              "se puntúa)"))
    if FOCOS_FIRMS and len(capas["fx"]):
        h.append(Line2D([], [], color="black", ls="", marker="x", ms=5, mew=1,
                        label=f"foco FIRMS · {DIAS_FIRMS} días "
                              "(variable del modelo, NO juez)"))
    return h


def leyenda(fig, capas):
    """Leyenda solo de estas capas. Para una única leyenda con la capa base,
    usar `capa_base.leyenda(fig, capa_verdad.handles(capas))`."""
    h = handles(capas)
    if h:
        fig.legend(handles=h, loc="lower left", fontsize=8, frameon=False)


CLAVES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RADIO_MITECO = 10        # km = píxeles: el radio con el que puntúa dos_15
HA_LEYENDA = (10, 1_000, 10_000)   # tamaños de referencia de la burbuja


def radio_pt(ha):
    """Radio en puntos de la burbuja de un incendio de `ha` hectáreas.

    Área proporcional a la superficie sería lo correcto de libro, pero a
    escala nacional un fuego de 100 ha saldría a un tercio de punto y no se
    vería: se usa raíz de la superficie (escalado perceptual) con un suelo de
    1,8 pt para que ningún incendio desaparezca, y en la leyenda van tres
    círculos de referencia calculados con esta misma fórmula, que es lo que
    permite descodificar el tamaño."""
    return 1.8 + 0.09 * np.sqrt(np.maximum(ha, 0.0))


def resumen(capas, n=6, claves=False):
    """Texto de una o dos líneas que dice qué es cada marca del mapa.

    En la rejilla de 2×2 rotular sobre el mapa saldría cuatro veces el mismo
    nombre, así que la identificación va aquí, en el pie de la figura.
    """
    lineas = []
    ef = capas.get("effis", [])
    if ef:
        cola = f"  (+{len(ef) - n} más)" if len(ef) > n else ""
        it = [(f"{i}. {t}" if claves else t)
              for i, (_x, _y, t, _h) in enumerate(ef[:n], 1)]
        lineas.append("EFFIS (perímetro quemado) · " + "   ".join(it) + cola)
    mn = list(capas.get("mn", []))
    if mn:
        cola = f"  (+{len(mn) - n} más)" if len(mn) > n else ""
        it = [(f"{CLAVES[i]}. {str(m).title()}" if claves else str(m).title())
              for i, m in enumerate(mn[:n])]
        lineas.append("MITECO (incidente con medios del Estado) · " + "   ".join(it) + cola)
    return "\n".join(lineas)
