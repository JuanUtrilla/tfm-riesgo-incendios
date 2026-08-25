#!/usr/bin/env python3
"""
Validación prospectiva del modelo a nivel ESTACIÓN-DÍA.

Complementa (no sustituye) la verificación diaria de `mapa_diario.
verificar_prevision`, que mide el lift usando el foco como unidad. Ese enfoque
tiene dos límites conocidos: un incendio grande aporta cientos de píxeles VIIRS
y domina el día (pseudo-replicación), y el lift solo mira el corte ALTO/EXTREMO.
Aquí la unidad es la estación-día (n≈684/día), lo que permite métricas de
discriminación estándar y, sobre todo, comparar el modelo con baselines.

Etiqueta: y=1 si hubo alguna detección VIIRS a ≤ RADIO km de la estación ese día.

Salidas:
  rankings/validacion_estacion_dia.csv   (una fila por tipo de previsión y día)
  resumen por consola con AUC-ROC, AUC-PR, lift del decil superior y
  calibración por nivel, más los baselines FWI y actividad reciente.

Uso: FIRMS_MAP_KEY=... python3 validar_modelo.py [--radio 25]
"""

import argparse
import json
import os
import re
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAIZ = Path(__file__).parent
RANKINGS = RAIZ / "rankings"
BBOX = "-10,35,5,44"
SATS = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]


# ------------------------------------------------------------------- FIRMS

def focos(desde, hasta):
    """Detecciones VIIRS en [desde, hasta] (ambos incluidos). La API sirve
    como mucho 5 días por petición ('Invalid day range. Expects [1..5]')."""
    key = os.environ["FIRMS_MAP_KEY"]
    trozos = []
    for sat in SATS:
        d = desde
        while d <= hasta:
            n = min(5, (hasta - d).days + 1)
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/"
                   f"{sat}/{BBOX}/{n}/{d:%Y-%m-%d}")
            r = requests.get(url, timeout=120)
            if r.ok and r.text.startswith("latitude"):
                trozos.append(pd.read_csv(StringIO(r.text)))
            else:
                print(f"aviso: {sat} {d:%F} sin datos", file=sys.stderr)
            d += pd.Timedelta(days=n)
    if not trozos:
        raise RuntimeError("FIRMS no devolvió detecciones")
    det = pd.concat(trozos, ignore_index=True)
    det["fecha"] = pd.to_datetime(det["acq_date"])
    return det


# --------------------------------------------------------------- previsiones

def previsiones():
    """Todos los CSV de predicción del repo, con su tipo y el día al que se
    refieren. D0/D1 están SELLADOS por commit antes del día evaluado; retro es
    hindcast (mismo modelo, meteo observada) y ranking_* es el día ya cerrado."""
    filas = []
    for ruta in sorted(RANKINGS.glob("*.csv")):
        m = re.match(r"(prevision_D0|prevision_D1|retro|ranking)_"
                     r"(\d{4}-\d{2}-\d{2})\.csv$", ruta.name)
        if not m:
            continue
        tipo = {"prevision_D0": "D0 (sellada)", "prevision_D1": "D1 (sellada)",
                "retro": "retro (hindcast)",
                "ranking": "ranking (día cerrado)"}[m.group(1)]
        df = pd.read_csv(ruta)
        if "prob" not in df or "lat" not in df:
            continue
        filas.append((tipo, pd.Timestamp(m.group(2)), df))
    return filas


def etiquetar(prev, det_dia, radio_km):
    """y=1 si la estación tiene una detección a ≤ radio_km ese día."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ex, ey = tr.transform(prev["lon"].values, prev["lat"].values)
    if not len(det_dia):
        return np.zeros(len(prev), dtype=int)
    fx, fy = tr.transform(det_dia["longitude"].values,
                          det_dia["latitude"].values)
    d, _ = cKDTree(np.column_stack([fx, fy])).query(np.column_stack([ex, ey]))
    return (d <= radio_km * 1000).astype(int)


# ------------------------------------------------- etiqueta alternativa EFFIS

def perimetros_effis(area_min_ha):
    """Perímetros de área quemada de EFFIS proyectados a EPSG:3035, indexados
    por día de inicio del incendio (FIREDATE). Requiere haber ejecutado
    descargar_effis.py. Etiqueta más limpia que VIIRS: superficie realmente
    cartografiada, no anomalía térmica."""
    from shapely.geometry import shape
    from shapely.ops import transform as sh_transform
    from pyproj import Transformer
    ruta = RAIZ / "data" / "effis_ba_season_ES.geojson"
    if not ruta.exists():
        raise SystemExit(f"falta {ruta}: ejecuta antes descargar_effis.py")
    geo = json.loads(ruta.read_text())
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    por_dia, ultimo = {}, None
    for f in geo["features"]:
        pr = f["properties"]
        if float(pr["AREA_HA"] or 0) < area_min_ha:
            continue
        dia = pd.Timestamp(pd.to_datetime(pr["FIREDATE"],
                                          format="ISO8601").date())
        g = sh_transform(lambda a, b: tr.transform(a, b), shape(f["geometry"]))
        por_dia.setdefault(dia, []).append(g)
        ultimo = max(ultimo or dia, dia)
    return por_dia, ultimo


def etiquetar_effis(prev, polis, radio_km):
    """y=1 si la estación está a ≤ radio_km del perímetro de un incendio que
    empezó ese día."""
    from pyproj import Transformer
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    if not polis:
        return np.zeros(len(prev), dtype=int)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ex, ey = tr.transform(prev["lon"].values, prev["lat"].values)
    arbol = STRtree(polis)
    y = np.zeros(len(prev), dtype=int)
    for i, (x, yy) in enumerate(zip(ex, ey)):
        p = Point(x, yy)
        cand = arbol.query(p.buffer(radio_km * 1000))
        y[i] = int(any(polis[j].distance(p) <= radio_km * 1000 for j in cand))
    return y


# ----------------------------------------------------------------- métricas

def metricas(y, p):
    """AUC-ROC, AUC-PR y lift del decil superior. NaN si el día no tiene
    ambas clases (no se puede discriminar nada)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    base = y.mean()
    out = {"n": len(y), "prevalencia": round(float(base) * 100, 1),
           "auc_roc": np.nan, "auc_pr": np.nan, "lift_pr": np.nan,
           "lift_decil": np.nan}
    if base in (0.0, 1.0):
        return out
    k = max(1, int(round(len(y) * 0.10)))
    top = np.argsort(-p)[:k]
    out.update(auc_roc=round(float(roc_auc_score(y, p)), 3),
               auc_pr=round(float(average_precision_score(y, p)), 3),
               lift_pr=round(float(average_precision_score(y, p) / base), 2),
               lift_decil=round(float(y[top].mean() / base), 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radio", type=float, default=25,
                    help="km alrededor de la estación que cuentan como fuego")
    ap.add_argument("--frp-min", type=float, default=0,
                    help="descarta detecciones con FRP menor (MW): filtra "
                         "industria, quemas agrícolas y falsos positivos débiles")
    ap.add_argument("--solo-fiables", action="store_true",
                    help="descarta detecciones de confianza baja (confidence=l)")
    ap.add_argument("--etiqueta", choices=["firms", "effis"], default="firms",
                    help="firms = detecciones VIIRS · effis = perímetros de "
                         "área quemada cartografiados (más limpio, con latencia)")
    ap.add_argument("--area-min", type=float, default=0,
                    help="ha mínimas del perímetro EFFIS para contar")
    ap.add_argument("--margen-latencia", type=int, default=3,
                    help="días finales que se excluyen con --etiqueta effis: "
                         "los perímetros recientes aún no están cartografiados")
    args = ap.parse_args()

    prev = previsiones()
    if not prev:
        raise SystemExit("no hay CSV de previsión en rankings/")
    dias = [f for _, f, _ in prev]

    if args.etiqueta == "effis":
        polis, ultimo = perimetros_effis(args.area_min)
        corte = ultimo - pd.Timedelta(days=args.margen_latencia)
        prev = [(t, f, df) for t, f, df in prev if f <= corte]
        n_tot = sum(len(v) for v in polis.values())
        print(f"EFFIS: {n_tot} perímetros (≥{args.area_min:.0f} ha) · último "
              f"incendio cartografiado {ultimo:%F} · se validan días ≤ "
              f"{corte:%F} para no penalizar al modelo con la latencia de "
              f"cartografiado\n")
    else:
        det = focos(min(dias), max(dias))
        if args.frp_min:
            det = det[det["frp"] >= args.frp_min]
        if args.solo_fiables:
            det = det[det["confidence"] != "l"]
        print(f"FIRMS: {len(det)} detecciones "
              f"{det['fecha'].min():%F} → {det['fecha'].max():%F}\n")

    filas, acum = [], {}
    for tipo, dia, df in prev:
        if args.etiqueta == "effis":
            y = etiquetar_effis(df, polis.get(dia, []), args.radio)
        else:
            y = etiquetar(df, det[det["fecha"] == dia], args.radio)
        m = metricas(y, df["prob"].values)
        filas.append({"tipo": tipo, "fecha": str(dia.date()),
                      "etiqueta": args.etiqueta, "radio_km": args.radio,
                      "frp_min": args.frp_min,
                      "solo_fiables": args.solo_fiables, **m})
        acum.setdefault(tipo, []).append((df, y))

    val = pd.DataFrame(filas)
    val.to_csv(RANKINGS / "validacion_estacion_dia.csv", index=False)

    print(f"=== POR DÍA (radio {args.radio:.0f} km) ===")
    print(val.to_string(index=False), "\n")

    print("=== AGREGADO (todos los días juntos) ===")
    for tipo, pares in acum.items():
        d = pd.concat([p for p, _ in pares], ignore_index=True)
        y = np.concatenate([yy for _, yy in pares])
        m = metricas(y, d["prob"].values)
        print(f"\n{tipo}: n={m['n']} estaciones-día · "
              f"prevalencia {m['prevalencia']}% · AUC-ROC {m['auc_roc']} · "
              f"AUC-PR {m['auc_pr']} (×{m['lift_pr']} sobre la base) · "
              f"lift decil superior ×{m['lift_decil']}")
        # baselines: ¿aporta el modelo sobre el FWI o sobre 'donde ya ardía'?
        for col, nom in [("fwi_pctl_local", "FWI (percentil local)"),
                         ("fwi", "FWI bruto"),
                         ("n_detec_50km_7d", "focos 50 km 7 días previos")]:
            if col in d and d[col].notna().any():
                b = metricas(y, d[col].fillna(0).values)
                print(f"    baseline {nom:<28} AUC-ROC {b['auc_roc']} · "
                      f"AUC-PR {b['auc_pr']} · lift decil ×{b['lift_decil']}")
        # calibración: qué fracción de estaciones ardió en cada banda
        d = d.assign(_y=y)
        cal = d.groupby("nivel")["_y"].agg(["size", "mean"])
        cal["mean"] = (cal["mean"] * 100).round(1)
        cal.columns = ["estaciones-día", f"% con fuego a <={args.radio:.0f} km"]
        print(cal.reindex(["BAJO", "MODERADO", "ALTO", "EXTREMO"]).dropna()
              .to_string())
        # el AUC agregado mezcla días con prevalencias distintas; la media de
        # los AUC diarios mide solo la discriminación DENTRO de cada día
        dia_auc = val[val["tipo"] == tipo]["auc_roc"].dropna()
        if len(dia_auc):
            print(f"    AUC-ROC medio por día: {dia_auc.mean():.3f} "
                  f"(mediana {dia_auc.median():.3f}, "
                  f"{(dia_auc > 0.5).sum()}/{len(dia_auc)} días > 0,5)")

    print(f"\nguardado: {RANKINGS/'validacion_estacion_dia.csv'}")


if __name__ == "__main__":
    main()
