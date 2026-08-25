#!/usr/bin/env python3
"""
Evaluación operativa del modelo: tres verdades-terreno independientes, cinco
scores en competencia e intervalos de confianza por remuestreo de días.

=============================================================================
QUÉ ARREGLA RESPECTO A LAS VALIDACIONES ANTERIORES
=============================================================================

1. PSEUDO-REPLICACIÓN (el sesgo detectado en la bitácora §25).
   FIRMS da una fila por PÍXEL: un mega-incendio en zona de riesgo alto aporta
   más de mil aciertos y domina el día. Aquí las detecciones se agrupan en
   EVENTOS (DBSCAN espacio-temporal) y cada evento cuenta una vez, en su día de
   PRIMERA detección — que es el día de ignición, no el de máxima extensión.

2. FALTABAN LOS BASELINES QUE IMPORTAN.
   Hasta ahora el modelo competía contra el FWI y contra el azar. En
   verificación operativa el listón real es la CLIMATOLOGÍA: la tasa histórica
   de incendio de esa zona en esa época del año. Es información gratis; un
   sistema que no la bate no aporta nada. Se añade también PERSISTENCIA
   (¿hubo fuego cerca en los últimos 7 días?), el otro baseline barato clásico.

3. NO HABÍA INCERTIDUMBRE.
   Con unas decenas de eventos, un lift de 2,3× tiene un intervalo ancho. Se
   calcula por BOOTSTRAP DE BLOQUES sobre los días (no sobre estación-día: las
   estaciones del mismo día están fuertemente correlacionadas y remuestrearlas
   por separado daría intervalos falsamente estrechos).

4. EL RADIO ERA ARBITRARIO.
   Se repite todo a 10 / 25 / 50 km para comprobar que la conclusión no depende
   de una elección de diseño.

=============================================================================
LAS TRES ETIQUETAS (sus sesgos van en direcciones distintas: si las tres
coinciden, la conclusión es robusta; si discrepan, la discrepancia informa)
=============================================================================
  effis   perímetro de área quemada cartografiado, con fecha estimada de inicio
          y superficie. La mejor: geometría real y umbral de completitud
          explícito. Latencia de días → se excluyen los últimos.
  miteco  partes oficiales de actuaciones: incendios con despliegue de medios
          del Ministerio. Sesgo a grandes; localización a nivel de municipio.
  firms   eventos VIIRS agrupados: cobertura completa y sin criterio humano,
          pero ciego a fuegos pequeños o bajo nubes.

=============================================================================
LOS CINCO SCORES
=============================================================================
  modelo        probabilidad de XGBoost (la previsión sellada)
  fwi           FWI absoluto — el estándar internacional
  fwi_pctl      percentil local del FWI — la contribución del TFM, aislada
  climatologia  tasa histórica de incendio EGIF 2015-2020 a <25 km en esa
                ventana del calendario. SIN meteorología: es "lo que pasa
                normalmente aquí por estas fechas"
  persistencia  detecciones FIRMS a <50 km en los 7 días previos (ya venía en
                los CSV sellados). ⚠️ Con la etiqueta `firms` es parcialmente
                circular (misma fuente): interpretar solo con effis/miteco.

Uso:
    python3 validar_operativo.py                       # todo, radio 25 km
    python3 validar_operativo.py --radios 10 25 50
    python3 validar_operativo.py --area-min 30 --boot 4000

Salidas: dataset/validacion_operativa.{csv,json}  (no sobrescribe nada previo)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path("/home/charredgem/Desktop/Master/TFM_fuego")
RANKINGS = Path("/home/charredgem/Desktop/Master/aemet_horario_verano2026/rankings")
EFFIS = DIR / "dataset" / "effis_ba_season_2026.geojson"
FIRMS = DIR / "dataset" / "firms_ventana_2026.parquet"
EGIF = DIR / "egif_civio.csv"

SEED = 42
MARGEN_LATENCIA = 3      # días finales que EFFIS aún no ha cartografiado
DBSCAN_KM = 6.0          # mismos parámetros que la validación 2021-24 (§18)
DBSCAN_DIAS = 2.4
MIN_DETEC = 3            # clústeres con menos detecciones se descartan
DIAS_FUENTE_FIJA = 8     # píxel con fuego más días que esto = fuente industrial

SCORES = {
    "modelo": "Modelo (XGBoost)",
    "fwi": "FWI absoluto",
    "fwi_pctl": "FWI percentil local",
    "climatologia": "Climatología (histórico EGIF)",
    "persistencia": "Persistencia (fuego reciente)",
    "modelo_clim": "Modelo + climatología",
}
# `modelo_clim` no es una fuente: es la media de los percentiles intra-día del
# modelo y de la climatología. Se incluye porque, si ambos rinden parecido pero
# aciertan en días distintos, combinarlos debería batir a los dos — y eso sería
# una mejora operativa inmediata sin reentrenar nada.


# --------------------------------------------------------------------------- #
# previsiones selladas
# --------------------------------------------------------------------------- #
def previsiones(tipo: str = "prevision_D0") -> list:
    out = []
    for ruta in sorted(RANKINGS.glob(f"{tipo}_*.csv")):
        m = re.match(rf"{tipo}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$", ruta.name)
        if not m:
            continue
        df = pd.read_csv(ruta)
        if "prob" in df and "lat" in df:
            out.append((pd.Timestamp(m.group(1)), df))
    return out


# --------------------------------------------------------------------------- #
# etiquetas
# --------------------------------------------------------------------------- #
def perimetros_effis(area_min: float):
    """Perímetros proyectados a 3035, indexados por día de INICIO."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as sh_transform
    geo = json.loads(EFFIS.read_text())
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    por_dia, ultimo = {}, None
    for f in geo["features"]:
        pr = f["properties"]
        if float(pr.get("AREA_HA") or 0) < area_min:
            continue
        dia = pd.Timestamp(pd.to_datetime(pr["FIREDATE"], format="ISO8601").date())
        g = sh_transform(lambda a, b: tr.transform(a, b), shape(f["geometry"]))
        por_dia.setdefault(dia, []).append(g)
        ultimo = max(ultimo or dia, dia)
    return por_dia, ultimo


def etiqueta_effis(prev, polis, radio_km):
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


def eventos_firms() -> pd.DataFrame:
    """Agrupa las detecciones VIIRS en eventos y devuelve UNA fila por evento:
    su día de primera detección y el centroide de las detecciones de ese día.

    Es la corrección de la pseudo-replicación: el incendio de Zamora que produjo
    1.300 píxeles pasa a valer exactamente lo mismo que el de 3 píxeles.
    """
    from pyproj import Transformer
    from sklearn.cluster import DBSCAN
    det = pd.read_parquet(FIRMS)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x, y = tr.transform(det["longitude"].values, det["latitude"].values)
    det["x"], det["y"] = x, y
    det["dia"] = det["acq_date"].values.astype("datetime64[D]").astype(int)

    # fuentes térmicas fijas (industria): mismo píxel ardiendo muchos días
    det["celda"] = (det.x // 375).astype(int).astype(str) + "_" + \
                   (det.y // 375).astype(int).astype(str)
    dias_por_celda = det.groupby("celda")["dia"].nunique()
    fijas = set(dias_por_celda[dias_por_celda > DIAS_FUENTE_FIJA].index)
    n0 = len(det)
    det = det[~det.celda.isin(fijas)]
    print(f"  filtradas {n0-len(det)} detecciones de {len(fijas)} fuentes fijas "
          f"(industria: mismo píxel >{DIAS_FUENTE_FIJA} días)", flush=True)

    escala = DBSCAN_KM * 1000 / DBSCAN_DIAS          # m por día
    X = np.column_stack([det.x, det.y, det.dia * escala])
    det["cl"] = DBSCAN(eps=DBSCAN_KM * 1000, min_samples=MIN_DETEC).fit_predict(X)
    det = det[det.cl >= 0]

    filas = []
    for cl, g in det.groupby("cl"):
        d0 = g.dia.min()
        prim = g[g.dia == d0]
        filas.append({"lat": prim.latitude.mean(), "lon": prim.longitude.mean(),
                      "fecha": pd.Timestamp(np.datetime64(int(d0), "D")),
                      "n_detec": len(g),
                      "frp_max": float(g.frp.max())})
    ev = pd.DataFrame(filas)
    print(f"  {len(ev)} eventos FIRMS a partir de {len(det)} detecciones "
          f"(mediana {ev.n_detec.median():.0f} detecciones/evento)", flush=True)
    return ev


def etiqueta_puntos(prev, puntos, radio_km):
    """y=1 si la estación tiene un punto (evento/incendio) a ≤ radio km."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    if not len(puntos):
        return np.zeros(len(prev), dtype=int)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ex, ey = tr.transform(prev["lon"].values, prev["lat"].values)
    fx, fy = tr.transform(puntos["lon"].values, puntos["lat"].values)
    d, _ = cKDTree(np.column_stack([fx, fy])).query(np.column_stack([ex, ey]))
    return (d <= radio_km * 1000).astype(int)


# --------------------------------------------------------------------------- #
# baseline climatológico
# --------------------------------------------------------------------------- #
def climatologia(est: pd.DataFrame, radio_km: float, ventana_dias: int = 10):
    """Tasa histórica de incendio: nº medio de incendios EGIF 2015-2020 al año
    a <radio km de la estación, en ±ventana días alrededor de ese día del año.

    Es el baseline honesto de un sistema operativo: "lo que pasa normalmente
    aquí por estas fechas", sin mirar la meteorología del día. Se construye con
    EGIF 2015-2020 —años anteriores a la evaluación— así que no puede contener
    información del periodo evaluado.
    """
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    eg = pd.read_csv(EGIF, usecols=["fecha", "lat", "lng"])
    eg["fecha"] = pd.to_datetime(eg["fecha"], errors="coerce")
    eg = eg.dropna()
    eg = eg[(eg.fecha.dt.year >= 2015) & (eg.fecha.dt.year <= 2020)]
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    fx, fy = tr.transform(eg["lng"].values, eg["lat"].values)
    doy = eg.fecha.dt.dayofyear.values
    arbol = cKDTree(np.column_stack([fx, fy]))

    ex, ey = tr.transform(est["lon"].values, est["lat"].values)
    tabla = {}
    for i, idema in enumerate(est["idema"].values):
        vec = arbol.query_ball_point([ex[i], ey[i]], r=radio_km * 1000)
        tabla[idema] = np.sort(doy[vec]) if vec else np.array([])

    def tasa(idema, dia_anio):
        d = tabla.get(idema)
        if d is None or not len(d):
            return 0.0
        # distancia circular en el calendario
        dd = np.abs(d - dia_anio)
        dd = np.minimum(dd, 365 - dd)
        return float((dd <= ventana_dias).sum()) / 6.0   # 6 años de histórico
    return tasa


# --------------------------------------------------------------------------- #
# métricas + bootstrap de bloques
# --------------------------------------------------------------------------- #
def metricas(t: pd.DataFrame, col: str) -> dict:
    from sklearn.metrics import roc_auc_score
    y, p = t["label"].values, t[col].values
    base = y.mean()
    if base in (0.0, 1.0):
        return {}
    k = max(1, int(round(len(y) * 0.10)))
    top = np.argsort(-p)[:k]
    return {"auc_roc": float(roc_auc_score(y, p)),
            "lift_decil": float(y[top].mean() / base),
            "pctl_mediano": float(np.median(p[y == 1]))}


def _metricas_np(y: np.ndarray, p: np.ndarray) -> dict:
    """Versión numpy pura de las métricas: el bootstrap la llama miles de veces
    y pasar por pandas en cada réplica lo hacía inviable."""
    n = len(y)
    base = y.mean()
    if base in (0.0, 1.0):
        return {}
    orden = np.argsort(-p, kind="stable")
    k = max(1, int(round(n * 0.10)))
    # AUC-ROC por el estadístico de Mann-Whitney sobre rangos (con empates)
    r = np.empty(n, dtype=float)
    idx = np.argsort(p, kind="stable")
    ps = p[idx]
    i = 0
    while i < n:                      # rango medio dentro de cada grupo de empates
        j = i
        while j + 1 < n and ps[j + 1] == ps[i]:
            j += 1
        r[idx[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum(); n0 = n - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"auc_roc": float(auc),
            "lift_decil": float(y[orden[:k]].mean() / base),
            "pctl_mediano": float(np.median(p[y == 1]))}


def bootstrap(t: pd.DataFrame, cols: list, B: int, rng) -> tuple:
    """Bootstrap de bloques por DÍA, para TODOS los scores a la vez.

    Dos decisiones que cambian las conclusiones:

    1. EL BLOQUE ES EL DÍA. Las ~684 estaciones de una misma jornada comparten
       la situación sinóptica: tratarlas como independientes daría intervalos
       mucho más estrechos de lo real y sugeriría una precisión que estos datos
       no tienen.

    2. TODOS LOS SCORES SE MIDEN SOBRE EL MISMO REMUESTREO (emparejado). Es lo
       que permite después calcular la distribución de la DIFERENCIA entre dos
       scores. Comparar dos intervalos por separado y concluir "se solapan,
       luego no hay diferencia" es un error clásico: como ambos se miden sobre
       los mismos días, sus errores están muy correlacionados, y al restar se
       cancela la incertidumbre compartida (qué días tocaron en la réplica). La
       diferencia puede ser claramente distinta de cero aunque los intervalos
       individuales se pisen.

    Devuelve (intervalos_por_score, muestras_por_score).
    """
    dias = list(t["fecha"].unique())
    ys = {d: g["label"].values for d, g in t.groupby("fecha")}
    ps = {c: {d: g[c].values for d, g in t.groupby("fecha")} for c in cols}
    muestras = {c: {"auc_roc": [], "lift_decil": [], "pctl_mediano": []}
                for c in cols}
    ix = np.arange(len(dias))
    for _ in range(B):
        sel = rng.choice(ix, size=len(dias), replace=True)
        y = np.concatenate([ys[dias[i]] for i in sel])
        for c in cols:
            m = _metricas_np(y, np.concatenate([ps[c][dias[i]] for i in sel]))
            for k in muestras[c]:
                muestras[c][k].append(m.get(k, np.nan))
    ic = {c: {k: [round(float(np.nanpercentile(v, 2.5)), 3),
                  round(float(np.nanpercentile(v, 97.5)), 3)]
              for k, v in muestras[c].items() if np.isfinite(v).any()}
          for c in cols}
    return ic, muestras


def contraste(muestras: dict, a: str, b: str, metrica: str = "auc_roc") -> dict:
    """Distribución de la diferencia a − b sobre el MISMO remuestreo.

    `p_mejor` es la fracción de réplicas en las que `a` supera a `b`: la lectura
    directa de "qué probabilidad hay de que vaya mejor que". Es una probabilidad
    BOOTSTRAP, no una posterior bayesiana: mide el respaldo que estos datos dan
    a que `a` sea superior, y hereda las limitaciones de la muestra — aquí,
    pocos días y de un solo régimen estacional.
    """
    da = np.asarray(muestras[a][metrica], dtype=float)
    db = np.asarray(muestras[b][metrica], dtype=float)
    d = da - db
    d = d[np.isfinite(d)]
    if not len(d):
        return {}
    return {"delta": round(float(np.mean(d)), 4),
            "ic95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "p_mejor": round(float((d > 0).mean()), 4)}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radios", type=float, nargs="+", default=[10, 25, 50])
    ap.add_argument("--area-min", type=float, default=10,
                    help="ha mínimas del perímetro EFFIS (umbral de completitud)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tipo", default="prevision_D0")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)

    prev = previsiones(a.tipo)
    if not prev:
        raise SystemExit(f"no hay CSV {a.tipo}_*.csv en {RANKINGS}")
    print(f"Previsiones {a.tipo}: {len(prev)} días "
          f"({prev[0][0]:%F} → {prev[-1][0]:%F})\n", flush=True)

    # --- verdades-terreno ---------------------------------------------------
    print("EFFIS:", flush=True)
    polis, ultimo_effis = perimetros_effis(a.area_min)
    corte = ultimo_effis - pd.Timedelta(days=MARGEN_LATENCIA)
    n_per = sum(len(v) for v in polis.values())
    print(f"  {n_per} perímetros ≥{a.area_min:.0f} ha · último {ultimo_effis:%F} "
          f"→ se validan días ≤ {corte:%F} (margen de latencia {MARGEN_LATENCIA} d)",
          flush=True)

    print("FIRMS (eventos):", flush=True)
    ev_firms = eventos_firms()

    import sys
    sys.argv = ["x"]
    from validar_miteco import cargar_partes, cargar_municipios, geocodificar
    print("MITECO:", flush=True)
    fuegos_miteco = geocodificar(cargar_partes("inicio"), cargar_municipios())
    fuegos_miteco = fuegos_miteco[fuegos_miteco.geo_score >= 80]

    est0 = prev[0][1][["idema", "lat", "lon"]].drop_duplicates()

    resultados, filas_csv = {}, []
    for radio in a.radios:
        print(f"\n{'='*72}\nRADIO {radio:.0f} km\n{'='*72}", flush=True)
        tasa_clim = climatologia(est0, radio)

        for etiqueta in ["effis", "miteco", "firms"]:
            tablas = []
            for dia, df in prev:
                if etiqueta == "effis":
                    if dia > corte:
                        continue
                    y = etiqueta_effis(df, polis.get(dia, []), radio)
                elif etiqueta == "miteco":
                    y = etiqueta_puntos(df, fuegos_miteco[fuegos_miteco.fecha == dia], radio)
                else:
                    y = etiqueta_puntos(df, ev_firms[ev_firms.fecha == dia], radio)
                if y.sum() == 0:
                    continue
                d = df.copy()
                d["fecha"], d["label"] = dia, y
                doy = dia.dayofyear
                d["climatologia"] = [tasa_clim(i, doy) for i in d["idema"]]
                d["persistencia"] = d.get("n_detec_50km_7d", 0)
                d["fwi_pctl"] = d.get("fwi_pctl_local", np.nan)
                # percentil INTRA-DÍA: lo que ordena un operador cada mañana
                for c in SCORES:
                    if c == "modelo_clim":
                        continue
                    src = "prob" if c == "modelo" else c
                    d[c] = d[src].rank(pct=True, na_option="bottom") * 100
                d["modelo_clim"] = (d["modelo"] + d["climatologia"]) / 2
                tablas.append(d)

            if not tablas:
                print(f"\n  {etiqueta}: sin días evaluables", flush=True)
                continue
            t = pd.concat(tablas, ignore_index=True)
            print(f"\n  etiqueta {etiqueta.upper()}: {t.fecha.nunique()} días · "
                  f"{len(t)} estación-día · {int(t.label.sum())} positivos "
                  f"(prevalencia {t.label.mean()*100:.2f}%)", flush=True)
            print(f"    {'score':30s} {'AUC-ROC':>18s} {'lift decil':>18s}",
                  flush=True)
            cols = [c for c in SCORES if metricas(t, c)]
            ic_todos, muestras = bootstrap(t, cols, a.boot, rng)
            for c in cols:
                m, ic = metricas(t, c), ic_todos[c]
                resultados.setdefault(f"radio_{radio:.0f}", {}) \
                          .setdefault(etiqueta, {})[c] = {**m, "ic95": ic}
                filas_csv.append({"radio_km": radio, "etiqueta": etiqueta,
                                  "score": c, "n_dias": t.fecha.nunique(),
                                  "n_pos": int(t.label.sum()),
                                  "prevalencia_pct": round(t.label.mean()*100, 2),
                                  **{k: round(v, 3) for k, v in m.items()},
                                  "auc_ic_bajo": ic["auc_roc"][0],
                                  "auc_ic_alto": ic["auc_roc"][1],
                                  "lift_ic_bajo": ic["lift_decil"][0],
                                  "lift_ic_alto": ic["lift_decil"][1]})
                print(f"    {SCORES[c]:30s} "
                      f"{m['auc_roc']:.3f} [{ic['auc_roc'][0]:.3f}–{ic['auc_roc'][1]:.3f}]"
                      f"   {m['lift_decil']:.2f}× [{ic['lift_decil'][0]:.2f}–{ic['lift_decil'][1]:.2f}]",
                      flush=True)

            # ---- contrastes emparejados: la probabilidad de ir mejor --------
            # Aquí está la respuesta a "¿va bien o va mal?": no en si los
            # intervalos se pisan, sino en la distribución de la DIFERENCIA
            # medida sobre los mismos días.
            if "modelo" in cols:
                print(f"\n    ¿El MODELO va mejor que…?   "
                      f"(Δ AUC-ROC emparejado · P(modelo mejor))", flush=True)
                comps = {}
                for rival in [c for c in cols if c not in ("modelo", "modelo_clim")]:
                    for met in ("auc_roc", "lift_decil"):
                        d = contraste(muestras, "modelo", rival, met)
                        if d:
                            comps.setdefault(rival, {})[met] = d
                    d = comps.get(rival, {}).get("auc_roc")
                    if d:
                        veredicto = ("SÍ" if d["p_mejor"] >= .95 else
                                     "no" if d["p_mejor"] <= .05 else "no concluyente")
                        print(f"      vs {SCORES[rival]:28s} "
                              f"Δ={d['delta']:+.3f} [{d['ic95'][0]:+.3f},{d['ic95'][1]:+.3f}]"
                              f"   P={d['p_mejor']*100:5.1f}%   {veredicto}", flush=True)
                if "modelo_clim" in cols:
                    d = contraste(muestras, "modelo_clim", "modelo", "auc_roc")
                    if d:
                        print(f"      (combinado vs modelo solo:      "
                              f"Δ={d['delta']:+.3f} [{d['ic95'][0]:+.3f},{d['ic95'][1]:+.3f}]"
                              f"   P={d['p_mejor']*100:5.1f}%)", flush=True)
                        comps["_combinado_vs_modelo"] = {"auc_roc": d}
                resultados[f"radio_{radio:.0f}"][etiqueta]["_contrastes"] = comps

    pd.DataFrame(filas_csv).to_csv(DIR / "dataset" / "validacion_operativa.csv",
                                   index=False)
    (DIR / "dataset" / "validacion_operativa.json").write_text(
        json.dumps({"config": vars(a), "margen_latencia_dias": MARGEN_LATENCIA,
                    "n_perimetros_effis": n_per, "n_eventos_firms": len(ev_firms),
                    "n_incendios_miteco": len(fuegos_miteco),
                    "resultados": resultados}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("\nGuardado dataset/validacion_operativa.{csv,json}", flush=True)


if __name__ == "__main__":
    main()
