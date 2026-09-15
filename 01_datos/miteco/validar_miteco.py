# Copia del validador contra el MITECO de la primera versión del proyecto
# (copiada el 21/08/2026 por gh_exportar_estado.py para la cadena diaria).
#!/usr/bin/env python3
"""
Validación del modelo contra los PARTES OFICIALES DE MITECO — la fuente de
verdad más limpia disponible para el verano de 2026.

=============================================================================
POR QUÉ ESTA VALIDACIÓN AÑADE ALGO QUE NINGUNA OTRA DA
=============================================================================
Hasta ahora el modelo se ha validado contra:
  · EGIF   — registro oficial, pero INCOMPLETO desde 2021 (bitácora §2).
  · FIRMS  — detecciones VIIRS. Independientes y en tiempo real, pero: no ven
             fuegos pequeños ni bajo nubes, y dan falsos positivos industriales
             (papeleras, refinerías). Un incendio grande aporta cientos de
             píxeles y domina el día (pseudo-replicación).
  · EFFIS  — perímetros cartografiados: limpio, pero con latencia de días.

Los partes diarios de MITECO son una cuarta fuente, **independiente de las tres
anteriores** y con una propiedad que ninguna tiene: registran los incendios en
los que el Ministerio DESPLEGÓ MEDIOS. Es decir, no miden anomalía térmica ni
superficie: miden **qué incendios fueron lo bastante graves como para movilizar
recursos del Estado**. Eso es justamente lo que un sistema de alerta temprana
debería anticipar.

Sesgos que hay que declarar (van en dirección contraria a los de FIRMS, y por
eso las dos validaciones juntas son más informativas que cualquiera sola):
  + No tiene falsos positivos industriales: cada registro es un incendio real
    atendido por medios del MITECO.
  + Incluye incendios que VIIRS no ve (pequeños, nublados, nocturnos apagados
    rápido) si movilizaron medios.
  − Solo recoge incendios con despliegue del MITECO: sesgo fuerte a incendios
    grandes o amenazantes. NO es un censo de igniciones.
  − La localización es el MUNICIPIO, no una coordenada: hay que geocodificar,
    con el error que eso implica (centro del municipio, no el punto de inicio).
  − El parte no siempre trae fecha de inicio: se usa la primera aparición del
    incidente como proxy de la ignición (modo por defecto).

=============================================================================
ORIGEN DE LOS DATOS  (todo se usa en SOLO LECTURA)
=============================================================================
  · Partes MITECO + parser  → el módulo RAG del TFM
      Su parser (`parseo_y_chuncking.py`, copiado en esta carpeta) se importa
      sin modificar. Los PDF se leen de `TFM_DATOS/miteco/` y todas las
      salidas van a `TFM_DATOS/dataset/`.
  · Previsiones del modelo  → el colector de AEMET (`TFM_COLECTOR`)
      `rankings/prevision_D{0,1}_<fecha>.csv`, SELLADAS por commit de GitHub
      Actions ANTES del día evaluado → validación prospectiva pura.
  · Maestro de municipios   → prototipo/cache/municipios.json (8.122 municipios
      con coordenadas, ya cacheado en este proyecto).

=============================================================================
METODOLOGÍA (idéntica a validar_modelo.py del colector, para que los números
sean comparables entre etiquetas)
=============================================================================
Unidad de análisis: ESTACIÓN-DÍA (~684 estaciones × día).
Etiqueta: y=1 si hay un incendio de MITECO a ≤ radio km de la estación ese día.
Métricas: AUC-ROC, AUC-PR, lift del AUC-PR sobre la prevalencia, lift del decil
superior, y el percentil del riesgo previsto donde ardió (comparable con las
validaciones del TFM: 85,5 en 2021-24 y 84,9 en 2025-26).

Modos de etiquetado:
  --modo inicio  (por defecto) solo la PRIMERA aparición de cada incidente
                 (`incident_key` del parser del módulo RAG) → proxy del día de
                 ignición. Es el test correcto para un modelo de riesgo de
                 ignición y evita que un incendio de 5 días cuente 5 veces.
  --modo todos   todos los incendio-día. Responde a otra pregunta: "¿acierta el
                 modelo los días en que hay incendios activos?".

Uso:
    python3 validar_miteco.py                      # modo inicio, radio 25 km
    python3 validar_miteco.py --modo todos --radio 35

Salidas (nada se sobrescribe de v1/v2/v3):
    dataset/validacion_miteco_<modo>.csv    una fila por tipo de previsión y día
    dataset/validacion_miteco_<modo>.json   agregados y metadatos
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DIR = Path(os.environ.get("TFM_DATOS", str(RAIZ / "datos")))
REPO_COLECTOR = Path(os.environ.get("TFM_COLECTOR", str(DIR / "colector")))
PDFS_MITECO = DIR / "miteco"
RANKINGS = REPO_COLECTOR / "rankings"
MUNICIPIOS = DIR / "prototipo" / "cache" / "municipios.json"
if not MUNICIPIOS.exists():
    MUNICIPIOS = Path(__file__).resolve().parents[1] / "municipios.json"

# Código INE de provincia → nombre, para desambiguar municipios homónimos
# (hay 8.122 municipios y nombres repetidos en provincias distintas).
PROVINCIAS = {
    "01": "alava", "02": "albacete", "03": "alicante", "04": "almeria",
    "05": "avila", "06": "badajoz", "07": "baleares", "08": "barcelona",
    "09": "burgos", "10": "caceres", "11": "cadiz", "12": "castellon",
    "13": "ciudad real", "14": "cordoba", "15": "a coruna", "16": "cuenca",
    "17": "girona", "18": "granada", "19": "guadalajara", "20": "gipuzkoa",
    "21": "huelva", "22": "huesca", "23": "jaen", "24": "leon", "25": "lleida",
    "26": "la rioja", "27": "lugo", "28": "madrid", "29": "malaga",
    "30": "murcia", "31": "navarra", "32": "ourense", "33": "asturias",
    "34": "palencia", "35": "las palmas", "36": "pontevedra",
    "37": "salamanca", "38": "santa cruz de tenerife", "39": "cantabria",
    "40": "segovia", "41": "sevilla", "42": "soria", "43": "tarragona",
    "44": "teruel", "45": "toledo", "46": "valencia", "47": "valladolid",
    "48": "bizkaia", "49": "zamora", "50": "zaragoza", "51": "ceuta",
    "52": "melilla",
}
# MITECO usa a veces el topónimo castellano y el maestro el cooficial (o al revés)
ALIAS_PROV = {
    "orense": "ourense", "la coruna": "a coruna", "coruna": "a coruna",
    "gerona": "girona", "lerida": "lleida", "guipuzcoa": "gipuzkoa",
    "vizcaya": "bizkaia", "araba": "alava", "araba/alava": "alava",
    "islas baleares": "baleares", "illes balears": "baleares",
    "castellon de la plana": "castellon", "castello": "castellon",
    "valencia/valencia": "valencia", "alicante/alacant": "alicante",
    "rioja": "la rioja", "asturias, principado de": "asturias",
    "murcia, region de": "murcia", "navarra, comunidad foral de": "navarra",
}


def norm(s: str | None) -> str:
    """minúsculas, sin tildes, sin puntuación: base de todos los emparejamientos."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# 1. Partes de MITECO (parser del módulo RAG, en solo lectura)
# --------------------------------------------------------------------------- #
def cargar_partes(modo: str) -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parseo_y_chuncking import parse_pdf_directory

    snaps = parse_pdf_directory(PDFS_MITECO)
    df = pd.DataFrame([{
        "snapshot_id": s.snapshot_id, "incident_key": s.incident_key,
        "fecha": pd.Timestamp(s.report_date), "pais": s.country,
        "ccaa": s.autonomous_community, "provincia": s.province,
        "localizacion": s.location, "estado": s.status,
        "n_medios": len(s.assigned_resources),
    } for s in snaps])
    print(f"MITECO: {len(df)} incendio-parte · {df.incident_key.nunique()} "
          f"incidentes únicos · {df.fecha.min():%F} → {df.fecha.max():%F}",
          flush=True)

    df = df[df.pais == "ES"]                       # 2 registros de Portugal
    if modo == "inicio":
        # primera aparición de cada incidente = proxy del día de ignición
        df = df.sort_values("fecha").groupby("incident_key", as_index=False).first()
        print(f"  modo INICIO: {len(df)} igniciones (1 por incidente)", flush=True)
    else:
        print(f"  modo TODOS: {len(df)} incendio-día", flush=True)
    return df


# --------------------------------------------------------------------------- #
# 2. Geocodificación municipio → coordenadas
# --------------------------------------------------------------------------- #
def cargar_municipios() -> pd.DataFrame:
    muni = pd.DataFrame(json.loads(MUNICIPIOS.read_text(encoding="utf-8")))
    muni["cod_prov"] = muni["id"].str.extract(r"id(\d{2})")
    muni["prov_norm"] = muni["cod_prov"].map(PROVINCIAS)
    muni["nombre_norm"] = muni["nombre"].map(norm)
    muni["lat"] = muni["latitud_dec"].astype(float)
    muni["lon"] = muni["longitud_dec"].astype(float)
    return muni[["nombre", "nombre_norm", "prov_norm", "lat", "lon"]]


def geocodificar(fuegos: pd.DataFrame, muni: pd.DataFrame) -> pd.DataFrame:
    """Empareja `localizacion` con un municipio, priorizando la provincia.

    Estrategia en tres pasos, de más fiable a menos:
      1. coincidencia exacta del nombre dentro de la provincia del parte
      2. coincidencia difusa (rapidfuzz) dentro de la provincia
      3. coincidencia difusa a nivel nacional (solo si la provincia no resolvió)
    Se guarda el score y el método para poder auditar y filtrar.
    """
    from rapidfuzz import process, fuzz

    por_prov = {p: g for p, g in muni.groupby("prov_norm")}
    filas = []
    for f in fuegos.itertuples():
        loc = norm(f.localizacion)
        prov = norm(f.provincia)
        prov = ALIAS_PROV.get(prov, prov)
        cand = por_prov.get(prov)

        elegido, score, metodo = None, 0, "sin_match"
        if cand is not None:
            exacto = cand[cand.nombre_norm == loc]
            if len(exacto):
                elegido, score, metodo = exacto.iloc[0], 100, "exacto_provincia"
            else:
                m = process.extractOne(loc, cand.nombre_norm.tolist(),
                                       scorer=fuzz.WRatio, score_cutoff=80)
                if m:
                    elegido, score, metodo = cand.iloc[m[2]], m[1], "difuso_provincia"
        if elegido is None:
            m = process.extractOne(loc, muni.nombre_norm.tolist(),
                                   scorer=fuzz.WRatio, score_cutoff=88)
            if m:
                elegido, score, metodo = muni.iloc[m[2]], m[1], "difuso_nacional"

        filas.append({
            **{c: getattr(f, c) for c in
               ["incident_key", "fecha", "provincia", "localizacion", "estado",
                "n_medios"]},
            "muni": None if elegido is None else elegido.nombre,
            "lat": np.nan if elegido is None else elegido.lat,
            "lon": np.nan if elegido is None else elegido.lon,
            "geo_score": score, "geo_metodo": metodo,
        })

    g = pd.DataFrame(filas)
    print("\nGeocodificación:", flush=True)
    for metodo, n in g.geo_metodo.value_counts().items():
        print(f"  {metodo:20s} {n:4d} ({n/len(g)*100:.0f}%)", flush=True)
    sin = g[g.geo_metodo == "sin_match"]
    if len(sin):
        print(f"  ⚠️ sin geocodificar: {sorted(set(sin.localizacion))[:8]}",
              flush=True)
    return filtrar_peninsula(g.dropna(subset=["lat"]))


def filtrar_peninsula(g: pd.DataFrame) -> pd.DataFrame:
    """Descarta incendios fuera de la España peninsular, DECLARÁNDOLO.

    El modelo se entrenó y opera sobre la malla peninsular de IberFire: en
    Canarias, Baleares, Ceuta y Melilla no hay predicción. Un incendio ahí no
    tendría ninguna estación a ≤25 km, así que nunca contaría como positivo y
    desaparecería en silencio de la estadística. Se filtra de forma explícita
    para que quede escrito qué se ha dejado fuera y por qué — un descarte
    silencioso se leería como "se ha evaluado todo" sin serlo.
    """
    dentro = g.lat.between(35.5, 44.5) & g.lon.between(-9.6, 4.6)
    if (~dentro).any():
        f = g[~dentro]
        print(f"  ⓘ fuera de la península (sin cobertura del modelo), "
              f"descartados {len(f)}: "
              f"{', '.join(f'{r.localizacion} ({r.provincia})' for r in f.itertuples())}",
              flush=True)
    return g[dentro]


# --------------------------------------------------------------------------- #
# 3. Previsiones selladas del colector
# --------------------------------------------------------------------------- #
def previsiones() -> list:
    """CSV de predicción del colector de AEMET, con su tipo y el día evaluado.

    D0/D1 están SELLADOS por commit ANTES del día al que se refieren (D1 es
    forecast puro: se emitió la víspera). `retro` es hindcast y `ranking` es el
    día ya cerrado — se incluyen como contraste, no como validación prospectiva.
    """
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
        if "prob" in df and "lat" in df:
            filas.append((tipo, pd.Timestamp(m.group(2)), df))
    return filas


# --------------------------------------------------------------------------- #
# 4. Etiquetado y métricas (mismas convenciones que validar_modelo.py)
# --------------------------------------------------------------------------- #
def etiquetar(prev: pd.DataFrame, fuegos_dia: pd.DataFrame, radio_km: float):
    """y=1 si la estación tiene un incendio de MITECO a ≤ radio_km ese día."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    if not len(fuegos_dia):
        return np.zeros(len(prev), dtype=int)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ex, ey = tr.transform(prev["lon"].values, prev["lat"].values)
    fx, fy = tr.transform(fuegos_dia["lon"].values, fuegos_dia["lat"].values)
    d, _ = cKDTree(np.column_stack([fx, fy])).query(np.column_stack([ex, ey]))
    return (d <= radio_km * 1000).astype(int)


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    """AUC-ROC, AUC-PR, lift del AUC-PR y lift del decil superior.

    NaN si el día no tiene las dos clases: sin positivos o sin negativos no hay
    nada que discriminar y cualquier número sería inventado.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    base = float(y.mean())
    out = {"n": len(y), "n_pos": int(y.sum()),
           "prevalencia": round(base * 100, 2),
           "auc_roc": np.nan, "auc_pr": np.nan,
           "lift_pr": np.nan, "lift_decil": np.nan, "pctl_mediano": np.nan}
    if base in (0.0, 1.0):
        return out
    k = max(1, int(round(len(y) * 0.10)))
    top = np.argsort(-p)[:k]
    # percentil del riesgo previsto en las estaciones donde ardió
    pctl = pd.Series(p).rank(pct=True).values * 100
    out.update(auc_roc=round(float(roc_auc_score(y, p)), 3),
               auc_pr=round(float(average_precision_score(y, p)), 3),
               lift_pr=round(float(average_precision_score(y, p) / base), 2),
               lift_decil=round(float(y[top].mean() / base), 2),
               pctl_mediano=round(float(np.median(pctl[y == 1])), 1))
    return out


def agregar(sub: pd.DataFrame) -> dict:
    """Agregado por tipo de previsión. Los días sin ambas clases quedan fuera."""
    v = sub.dropna(subset=["auc_roc"])
    if not len(v):
        return {}
    return {
        "dias_evaluables": len(v), "dias_totales": len(sub),
        "n_estacion_dia": int(sub["n"].sum()), "n_positivos": int(sub["n_pos"].sum()),
        "prevalencia_media": round(float(v["prevalencia"].mean()), 2),
        "auc_roc_medio": round(float(v["auc_roc"].mean()), 3),
        "auc_pr_medio": round(float(v["auc_pr"].mean()), 3),
        "lift_pr_medio": round(float(v["lift_pr"].mean()), 2),
        "lift_decil_medio": round(float(v["lift_decil"].mean()), 2),
        "pctl_mediano": round(float(v["pctl_mediano"].median()), 1),
        "dias_auc_roc_mayor_0_5": int((v["auc_roc"] > 0.5).sum()),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("=====")[0])
    ap.add_argument("--modo", choices=["inicio", "todos"], default="inicio",
                    help="inicio = 1 fila por incidente (proxy de ignición); "
                         "todos = todos los incendio-día")
    ap.add_argument("--radio", type=float, default=25,
                    help="km alrededor de la estación que cuentan como fuego")
    ap.add_argument("--geo-score-min", type=float, default=80,
                    help="descarta geocodificaciones con score menor")
    args = ap.parse_args()

    fuegos = cargar_partes(args.modo)
    fuegos = geocodificar(fuegos, cargar_municipios())
    n0 = len(fuegos)
    fuegos = fuegos[fuegos.geo_score >= args.geo_score_min]
    print(f"  usables: {len(fuegos)} (descartados {n0-len(fuegos)} por score "
          f"< {args.geo_score_min})", flush=True)

    prev = previsiones()
    if not prev:
        raise SystemExit(f"no hay CSV de previsión en {RANKINGS}")
    tipos = sorted({t for t, _, _ in prev})
    print(f"\nPrevisiones del colector: {len(prev)} ficheros · tipos: {tipos}",
          flush=True)

    filas = []
    for tipo, dia, df in prev:
        fd = fuegos[fuegos.fecha == dia]
        y = etiquetar(df, fd, args.radio)
        m = metricas(y, df["prob"].values)
        filas.append({"tipo": tipo, "fecha": dia.date().isoformat(),
                      "n_fuegos_miteco": len(fd), **m})
    res = pd.DataFrame(filas).sort_values(["tipo", "fecha"])

    salida_csv = DIR / "dataset" / f"validacion_miteco_{args.modo}.csv"
    res.to_csv(salida_csv, index=False)

    print(f"\n{'='*78}\nRESULTADOS — etiqueta MITECO, modo {args.modo.upper()}, "
          f"radio {args.radio:.0f} km\n{'='*78}", flush=True)
    agregados = {}
    for tipo in tipos:
        sub = res[res.tipo == tipo]
        a = agregar(sub)
        agregados[tipo] = a
        if not a:
            print(f"\n{tipo}: sin días evaluables (ningún día con incendios "
                  f"MITECO y previsión a la vez)", flush=True)
            continue
        print(f"\n{tipo}", flush=True)
        print(f"  días evaluables      {a['dias_evaluables']}/{a['dias_totales']}"
              f"   ({a['n_positivos']} estación-día con fuego de "
              f"{a['n_estacion_dia']}, prevalencia {a['prevalencia_media']}%)")
        print(f"  AUC-ROC medio        {a['auc_roc_medio']:.3f}"
              f"   (0,5 = azar; supera 0,5 en {a['dias_auc_roc_mayor_0_5']}"
              f"/{a['dias_evaluables']} días)")
        print(f"  AUC-PR medio         {a['auc_pr_medio']:.3f}"
              f"   → lift {a['lift_pr_medio']:.2f}× sobre la prevalencia")
        print(f"  lift del decil sup.  {a['lift_decil_medio']:.2f}×"
              f"   (concentración de fuego en el 10% de más riesgo)")
        print(f"  pctl mediano donde ardió  {a['pctl_mediano']:.1f}"
              f"   (azar = 50; TFM con FIRMS: 85,5 y 84,9)", flush=True)

    meta = {
        "etiqueta": "MITECO — partes diarios de actuaciones (despliegue de medios)",
        "modo": args.modo, "radio_km": args.radio,
        "geo_score_min": args.geo_score_min,
        "n_fuegos_usables": len(fuegos),
        "rango_fechas_miteco": [str(fuegos.fecha.min().date()),
                                str(fuegos.fecha.max().date())],
        "procedencia": {
            "partes_y_parser": "módulo RAG del TFM (solo lectura)",
            "previsiones": "colector de AEMET, rankings/*.csv "
                           "sellados por commit de GitHub Actions",
            "municipios": "prototipo/cache/municipios.json (maestro AEMET)",
        },
        "caveats": [
            "MITECO solo registra incendios con despliegue de medios del "
            "Ministerio: sesgo fuerte a incendios grandes; NO es un censo de "
            "igniciones.",
            "La localización es el MUNICIPIO: se geocodifica a su centro, no al "
            "punto de inicio del fuego.",
            "En modo inicio, la primera aparición en los partes es un proxy de "
            "la ignición: un fuego iniciado antes del primer parte disponible "
            "se fecha más tarde de lo real.",
            "Muestra pequeña: pocas decenas de incidentes en el solape con las "
            "previsiones selladas. Los agregados tienen incertidumbre alta.",
        ],
        "agregados": agregados,
    }
    salida_json = DIR / "dataset" / f"validacion_miteco_{args.modo}.json"
    salida_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nGuardado {salida_csv.name} y {salida_json.name} en dataset/",
          flush=True)


if __name__ == "__main__":
    main()
