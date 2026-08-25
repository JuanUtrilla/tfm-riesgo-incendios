#!/usr/bin/env python3
"""
Puntúa cada día los dos mapas contra el área quemada de EFFIS, y acumula.

NO TOCA PRODUCCIÓN. Escribe salida/puntuacion_effis.csv y, con --informe,
salida/veredicto_acumulado.json.

=============================================================================
QUÉ AGUJERO TAPA
=============================================================================
El cron ya genera los dos mapas cada día, y `comparar_produccion.py` mide si
se PARECEN. Pero parecerse no es acertar: sin esto, el veredicto se quedaría
congelado en los 17 días de julio de 2026 por muchos días que pasaran, que
era justo el argumento para montar el cron en paralelo.

Aquí se puntúa el ACIERTO de cada mapa contra la superficie realmente quemada
y se apila en un CSV. Cada corrida recalcula el veredicto con todos los días.

Ya no hace falta reconstruir producción desde el colector horario (que además
se paró el 27-jul): el cron genera su mapa de verdad.

=============================================================================
LA LATENCIA DE EFFIS MANDA EL DISEÑO
=============================================================================
Cartografiar un perímetro lleva días: entre FIREDATE y LASTUPDATE hay una
mediana de 6 días, media 9 y cola hasta 164. El propio repo de EFFIS avisa de
que los 2-3 últimos días están sistemáticamente incompletos.

Por eso NO se puntúa el día en curso, sino una VENTANA PASADA que se reescribe
entera en cada corrida: los polígonos que llegan tarde se recogen solos y las
filas convergen. Puntuar el día D el propio día D daría cero incendios y
metería ruido a la baja en los dos mapas por igual.

Uso:
    python puntuar_effis.py --refrescar          # baja EFFIS y puntúa
    python puntuar_effis.py --informe            # solo el veredicto
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

import config
from comparar_julio2026 import auc, quemadas

CSV = config.salida("puntuacion_effis.csv")
GEOJSON = config.salida("effis_ba_season_ES.geojson")
BASE_JULIO = config.salida("julio2026_effis.json")
WFS = "https://maps.effis.emergency.copernicus.eu/effis"
CAMPOS = ["id", "FIREDATE", "LASTUPDATE", "COUNTRY", "PROVINCE", "COMMUNE",
          "AREA_HA", "CLASS"]


def refrescar():
    """Perímetros de la temporada en curso, solo España. WFS abierto, sin key."""
    import requests
    try:
        r = requests.get(WFS, timeout=600, params={
            "service": "WFS", "version": "1.1.0", "request": "GetFeature",
            "typename": "ms:modis.ba.poly.season",
            "outputformat": "application/json; subtype=geojson",
            "srsname": "EPSG:4326"})
        r.raise_for_status()
        geo = json.loads(r.content.decode("utf-8"))
    except Exception as e:
        # 21/08/2026: un fallo del WFS no debe tumbar la puntuación ni, peor,
        # dejar un geojson vacío: se sigue con el de la última vez.
        print(f"  EFFIS: el WFS falló ({type(e).__name__}) · se usa el geojson "
              f"anterior", flush=True)
        return
    fs = [f for f in geo.get("features", [])
          if f["properties"].get("COUNTRY") == "ES"]
    if len(fs) < 100:
        print(f"  EFFIS: respuesta sospechosa ({len(fs)} incendios) · se conserva "
              f"el geojson anterior", flush=True)
        return
    for f in fs:
        f["properties"] = {k: f["properties"].get(k) for k in CAMPOS}
    with open(GEOJSON, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": fs}, fh,
                  ensure_ascii=False)
    p = pd.to_datetime([f["properties"]["FIREDATE"] for f in fs],
                       format="ISO8601")
    print(f"  EFFIS: {len(fs)} incendios · {p.min():%F} → {p.max():%F}",
          flush=True)


def mapas(fecha):
    """{nombre: mapa} para esa fecha, o None si faltan producción o malla.

    21/08/2026: se añaden `unico` y `pareja` (modelos de etiqueta EFFIS,
    `dos_riesgo_hoy.py`) cuando existan. Son los candidatos a producción y
    aquí es donde se juzgan cada día, con la misma verdad-terreno."""
    p = f"{config.FUENTE}/prototipo/cache/malla_prob_{fecha}.npz"
    md = config.salida("mapas_diarios")
    m = next((r for r in (f"{md}/riesgo_hoy_{fecha}.npz",
                          config.salida(f"riesgo_hoy_{fecha}.npz"))
              if os.path.exists(r)), None)
    if m is None:
        return None
    out = {"malla": np.load(m)["prob"].astype(float)}
    # producción (AEMET) solo existe en el portátil; en GitHub no hay
    if os.path.exists(p):
        out["produccion"] = np.load(p)["prob"].astype(float)
    d = next((r for r in (f"{md}/dos_riesgo_{fecha}.npz",
                          config.salida(f"dos_riesgo_{fecha}.npz"))
              if os.path.exists(r)), None)
    if d is not None:
        z = np.load(d)
        out["unico"] = z["prob_unico"].astype(float)
        out["pareja"] = z["prob_pareja"].astype(float)
    return out


def puntuar(dias):
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es_esp = ds["is_spain"].values.astype(bool)
    filas = []
    for dia in dias:
        f = str(dia.date())
        par = mapas(f)
        if par is None:
            continue
        quem, ha = quemadas(dia, ds, es_esp, ruta=GEOJSON)
        if quem.sum() == 0:
            continue
        fila = {"fecha": f, "celdas_quemadas": int(quem.sum()),
                "area_ha": round(ha, 1)}
        for nom, A in par.items():
            todas = A[es_esp]
            todas = todas[np.isfinite(todas)]
            q = A[quem & es_esp]
            q = q[np.isfinite(q)]
            if not len(q):
                continue
            orden = np.sort(todas)
            pct = np.searchsorted(orden, q, side="right") / len(orden) * 100
            noq = A[es_esp & ~quem]
            noq = noq[np.isfinite(noq)]
            fila[f"pctl_{nom}"] = round(float(np.median(pct)), 2)
            fila[f"auc_{nom}"] = round(auc(q, noq), 4)
        if "pctl_malla" in fila:
            filas.append(fila)
            print(f"  {f}  quemadas {fila['celdas_quemadas']:4d} celdas "
                  f"({ha:7.0f} ha) · " + " · ".join(
                      f"{n} {fila[f'pctl_{n}']:5.1f}" for n in par
                      if f"pctl_{n}" in fila), flush=True)
    ds.close()
    return filas


def guardar(filas):
    """Reescribe las filas de la ventana; las de fuera se conservan."""
    nuevo = pd.DataFrame(filas)
    if os.path.exists(CSV) and len(nuevo):
        viejo = pd.read_csv(CSV)
        viejo = viejo[~viejo["fecha"].isin(nuevo["fecha"])]
        nuevo = pd.concat([viejo, nuevo], ignore_index=True)
    elif os.path.exists(CSV):
        nuevo = pd.read_csv(CSV)
    if len(nuevo):
        nuevo.sort_values("fecha").to_csv(CSV, index=False)
    return nuevo


def informe(df):
    """Veredicto con TODOS los días acumulados, remuestreando días."""
    julio = []
    if os.path.exists(BASE_JULIO):
        julio = json.load(open(BASE_JULIO))["dias"]
    d = pd.concat([pd.DataFrame(julio), df], ignore_index=True) \
          .drop_duplicates("fecha", keep="last").sort_values("fecha") \
        if len(julio) else df
    if not len(d):
        print("  todavía no hay días puntuados")
        return
    dp = d.dropna(subset=["pctl_produccion"]) if "pctl_produccion" in d else d.iloc[0:0]
    if len(dp) < 3:
        print("  sin días con mapa de producción: solo tabla de medias")
        for nom in ("malla", "unico", "pareja"):
            if f"auc_{nom}" in d:
                print(f"    {nom:8s} pctl {d[f'pctl_{nom}'].mean():5.1f} · "
                      f"AUC {d[f'auc_{nom}'].mean():.3f} · {d[f'auc_{nom}'].notna().sum()} días")
        return
    d = dp
    dif = (d["pctl_malla"] - d["pctl_produccion"]).values
    dau = (d["auc_malla"] - d["auc_produccion"]).values
    w = d["celdas_quemadas"].values.astype(float)
    rng = np.random.default_rng(0)

    def boot(v, peso=None):
        o = [np.average(v[i], weights=peso[i]) if peso is not None
             else v[i].mean()
             for i in (rng.integers(0, len(v), len(v)) for _ in range(20000))]
        return [float(x) for x in np.percentile(o, [2.5, 97.5])]

    ic, ica, icw = boot(dif), boot(dau), boot(dif, w)
    print("\n" + "=" * 68)
    print(f"VEREDICTO ACUMULADO — {len(d)} días · "
          f"{int(w.sum()):,} celdas quemadas")
    print("=" * 68)
    # MEDIA de las medianas diarias, no mediana de medianas: así la tabla
    # cuadra con la diferencia emparejada de abajo (mediana de medianas puede
    # salir con el signo contrario y no es lo que se contrasta).
    print(f"{'mapa':<14}{'pctl medio/día':>16}{'AUC medio':>12}{'n días':>8}")
    for nom in ("produccion", "malla", "unico", "pareja"):
        if f"pctl_{nom}" in d:
            print(f"{nom:<14}{d[f'pctl_{nom}'].mean():>16.1f}"
                  f"{d[f'auc_{nom}'].mean():>12.3f}{d[f'pctl_{nom}'].notna().sum():>8}")
    # los candidatos EFFIS contra producción, solo en los días en que existen
    extra = {}
    for nom in ("unico", "pareja"):
        if f"auc_{nom}" in d and d[f"auc_{nom}"].notna().sum() >= 3:
            s_ = d.dropna(subset=[f"auc_{nom}"])
            da = (s_[f"auc_{nom}"] - s_["auc_produccion"]).values
            ica_ = boot(da)
            extra[nom] = {"dias": int(len(s_)), "dif_auc": float(da.mean()), "ic95_auc": ica_}
            print(f"  {nom} − producción ({len(s_)} días): AUC {da.mean():+.3f} "
                  f"IC95 [{ica_[0]:+.3f}, {ica_[1]:+.3f}] · gana {int((da > 0).sum())}")
    print(f"\n  malla − producción, remuestreando DÍAS:")
    print(f"    percentil            {dif.mean():+5.1f} pts · "
          f"IC95 [{ic[0]:+.1f}, {ic[1]:+.1f}]")
    print(f"    AUC                  {dau.mean():+5.3f}     · "
          f"IC95 [{ica[0]:+.3f}, {ica[1]:+.3f}]")
    print(f"    percentil ponderado  {np.average(dif, weights=w):+5.1f} pts · "
          f"IC95 [{icw[0]:+.1f}, {icw[1]:+.1f}]")
    print(f"\n  la malla gana en {int((dif > 0).sum())} de {len(dif)} días")
    ver = ("la malla gana" if ic[0] > 0 else
           ("la malla pierde" if ic[1] < 0 else
            f"mejor de media, pero con {len(d)} días no se distingue del ruido"))
    print(f"  → {ver}")
    json.dump({"dias": int(len(d)), "celdas": int(w.sum()),
               "dif_pctl": float(dif.mean()), "ic95_pctl": ic,
               "dif_auc": float(dau.mean()), "ic95_auc": ica,
               "dif_ponderada": float(np.average(dif, weights=w)),
               "ic95_ponderada": icw, "veredicto": ver, "candidatos_effis": extra},
              open(config.salida("veredicto_acumulado.json"), "w"),
              indent=1, ensure_ascii=False)


def main(a):
    if a.refrescar:
        refrescar()
    hoy = pd.Timestamp.utcnow().tz_localize(None).normalize()
    filas = [] if a.informe else puntuar(
        pd.date_range(hoy - pd.Timedelta(days=a.retraso + a.ventana),
                      hoy - pd.Timedelta(days=a.retraso), freq="D"))
    df = guardar(filas)
    informe(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refrescar", action="store_true")
    p.add_argument("--informe", action="store_true")
    p.add_argument("--retraso", type=int, default=4,
                   help="días que se dejan madurar antes de puntuar")
    p.add_argument("--ventana", type=int, default=45,
                   help="días hacia atrás que se reescriben en cada corrida")
    main(p.parse_args())
