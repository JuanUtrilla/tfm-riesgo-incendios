#!/usr/bin/env python3
"""
Prepara los datos del panel de validación contra MITECO.

Toma las previsiones selladas del colector y los incendios oficiales del MITECO
(geocodificados por validar_miteco.py) y precalcula todo lo que necesita el
panel HTML, para que este sea autocontenido y no dependa de ningún fichero
externo ni de red.

Qué pregunta responde
Los CSV sellados traen la probabilidad del modelo y también `fwi` (el índice
del sistema canadiense, el estándar internacional) y `fwi_pctl_local` (el
percentil local, la aportación metodológica del TFM). Con eso se puede
comparar de forma prospectiva, sobre incendios oficiales y con predicciones
selladas por commit, si el modelo aporta algo sobre el índice que ya existe.
Es una comprobación complementaria de temporada; la validación del trabajo es
el replay de 2025-2026.

Cómo se puntúa (importa para leer el panel)
La unidad es la estación-día. Cada día, las ~684 estaciones se ordenan por
score y se convierten a percentil dentro de ese día. Después se agrupan todos
los días. Así se mide lo que hace un operador cada mañana ("hoy, ¿a qué zonas
miro?") sin que un día de ola de calor, en el que todo el país puntúa alto,
arrastre la estadística. Es más exigente que agrupar las probabilidades crudas.

Salida: dataset/dashboard_miteco.json  (no sobrescribe nada)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path("/home/charredgem/Desktop/Master/TFM_fuego")
RANKINGS = Path("/home/charredgem/Desktop/Master/aemet_horario_verano2026/rankings")
RADIO_KM = 25.0

# Scores que compiten. El modelo contra el estándar internacional y contra la
# contribución del TFM aislada.
SCORES = {
    "modelo": ("prob", "Modelo (XGBoost)"),
    "fwi": ("fwi", "FWI absoluto (estándar)"),
    "fwi_pctl": ("fwi_pctl_local", "FWI percentil local"),
}


def cargar_fuegos() -> pd.DataFrame:
    """Incendios MITECO geocodificados, modo inicio (proxy de ignición)."""
    import sys
    sys.argv = ["x"]
    from validar_miteco import cargar_partes, cargar_municipios, geocodificar
    f = geocodificar(cargar_partes("inicio"), cargar_municipios())
    return f[f.geo_score >= 80]


def previsiones(tipo_re: str) -> list:
    out = []
    for ruta in sorted(RANKINGS.glob("*.csv")):
        m = re.match(rf"({tipo_re})_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$", ruta.name)
        if not m:
            continue
        df = pd.read_csv(ruta)
        if "prob" in df and "lat" in df:
            out.append((m.group(1), pd.Timestamp(m.group(2)), df))
    return out


def etiquetar(prev: pd.DataFrame, fuegos_dia: pd.DataFrame) -> np.ndarray:
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    if not len(fuegos_dia):
        return np.zeros(len(prev), dtype=int)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ex, ey = tr.transform(prev["lon"].values, prev["lat"].values)
    fx, fy = tr.transform(fuegos_dia["lon"].values, fuegos_dia["lat"].values)
    d, _ = cKDTree(np.column_stack([fx, fy])).query(np.column_stack([ex, ey]))
    return (d <= RADIO_KM * 1000).astype(int)


def construir_tabla(tipo_re: str, fuegos: pd.DataFrame) -> pd.DataFrame:
    """Tabla estación-día con la etiqueta MITECO y el percentil intra-día de
    cada score (0 = el de menos riesgo del día, 100 = el de más)."""
    filas = []
    for tipo, dia, df in previsiones(tipo_re):
        y = etiquetar(df, fuegos[fuegos.fecha == dia])
        d = df.copy()
        d["fecha"] = dia
        d["tipo"] = tipo
        d["label"] = y
        for clave, (col, _) in SCORES.items():
            if col in d:
                # percentil dentro del día: lo que un operador ordena cada mañana
                d[f"pctl_{clave}"] = d[col].rank(pct=True, na_option="bottom") * 100
        filas.append(d)
    return pd.concat(filas, ignore_index=True)


def curva_ganancia(t: pd.DataFrame, clave: str, pasos: int = 101) -> list:
    """% de incendios cubiertos al vigilar el X% de más riesgo de cada día.

    La diagonal (cubrir el X% de incendios con el X% del territorio) es lo que
    daría el azar. Cuanto más se despega la curva, más valor operativo.
    """
    col = f"pctl_{clave}"
    if col not in t:
        return []
    pos = t.loc[t.label == 1, col].values
    if not len(pos):
        return []
    xs = np.linspace(0, 100, pasos)
    # vigilar el X% superior = percentil intra-día >= 100 - X
    return [[round(float(x), 1),
             round(float((pos >= 100 - x).mean() * 100), 2)] for x in xs]


def por_decil(t: pd.DataFrame, clave: str) -> list:
    """Tasa observada de incendio por decil de riesgo previsto (intra-día).

    Sirve para comprobar que el orden significa algo: la tasa debería crecer
    de forma monótona del decil 1 al 10.
    """
    col = f"pctl_{clave}"
    base = float(t.label.mean())
    out = []
    for d in range(10):
        m = (t[col] >= d * 10) & (t[col] < (d + 1) * 10 + (0.001 if d == 9 else 0))
        sub = t[m]
        tasa = float(sub.label.mean()) if len(sub) else 0.0
        out.append({"decil": d + 1, "n": int(len(sub)), "n_pos": int(sub.label.sum()),
                    "tasa_pct": round(tasa * 100, 3),
                    "lift": round(tasa / base, 2) if base else None})
    return out


def metricas_globales(t: pd.DataFrame, clave: str) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    col = f"pctl_{clave}"
    y, p = t.label.values, t[col].values
    base = float(y.mean())
    k = max(1, int(round(len(y) * 0.10)))
    top = np.argsort(-p)[:k]
    return {
        "auc_roc": round(float(roc_auc_score(y, p)), 3),
        "auc_pr": round(float(average_precision_score(y, p)), 4),
        "lift_pr": round(float(average_precision_score(y, p) / base), 2),
        "lift_decil": round(float(y[top].mean() / base), 2),
        "pctl_mediano": round(float(np.median(p[y == 1])), 1),
        "prevalencia_pct": round(base * 100, 2),
    }


def niveles_operativos(t: pd.DataFrame) -> list:
    """Rendimiento en los cortes de alerta que el sistema usa de verdad."""
    out = []
    for nivel, cond in [("EXTREMO", t.nivel == "EXTREMO"),
                        ("ALTO+", t.nivel.isin(["ALTO", "EXTREMO"])),
                        ("MODERADO+", t.nivel.isin(["MODERADO", "ALTO", "EXTREMO"]))]:
        cob = float(cond.mean())
        cap = float(t.loc[cond, "label"].sum() / max(1, t.label.sum()))
        out.append({"nivel": nivel,
                    "pct_territorio": round(cob * 100, 1),
                    "pct_incendios": round(cap * 100, 1),
                    "lift": round(cap / cob, 2) if cob else None})
    return out


def construir_panel(datos_json: str) -> None:
    """Inyecta los datos en la plantilla y escribe el panel autocontenido.

    Se generan dos versiones del mismo contenido:
      · panel_validacion_miteco.html: autónomo (doctype + html/head/body), para
        abrir con doble clic, como el resto de visores del proyecto.
      · dataset/panel_miteco_artifact.html: solo el contenido, sin envoltorio,
        que es lo que espera el publicador de artifacts.
    El JSON va embebido: el panel no depende de red ni de ficheros externos.
    """
    plantilla = (DIR / "panel_miteco_plantilla.html").read_text(encoding="utf-8")
    if plantilla.count("__DATOS__") != 1:
        raise SystemExit("la plantilla debe tener exactamente un marcador __DATOS__")
    contenido = plantilla.replace("__DATOS__", datos_json)
    (DIR / "dataset" / "panel_miteco_artifact.html").write_text(contenido,
                                                                encoding="utf-8")
    estilo, resto = contenido.split("</style>", 1)
    autonomo = ('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                f'{estilo}</style>\n</head>\n<body>\n{resto}\n</body>\n</html>\n')
    ruta = DIR / "panel_validacion_miteco.html"
    ruta.write_text(autonomo, encoding="utf-8")
    print(f"Panel: {ruta.name} ({ruta.stat().st_size/1024:.0f} KB, autocontenido)")


def main() -> None:
    fuegos = cargar_fuegos()
    tD0 = construir_tabla("prevision_D0", fuegos)
    tD1 = construir_tabla("prevision_D1", fuegos)
    print(f"\nD0: {len(tD0)} estación-día, {int(tD0.label.sum())} con incendio")
    print(f"D1: {len(tD1)} estación-día, {int(tD1.label.sum())} con incendio")

    # --- distribución del riesgo, con y sin incendio (discriminación visual) --
    bins = np.arange(0, 101, 5)
    hist = {}
    for et, sub in [("con_incendio", tD0[tD0.label == 1]),
                    ("sin_incendio", tD0[tD0.label == 0])]:
        h, _ = np.histogram(sub["pctl_modelo"], bins=bins)
        hist[et] = {"bins": bins[:-1].tolist(),
                    "pct": (h / max(1, h.sum()) * 100).round(2).tolist(),
                    "n": int(h.sum())}

    # --- serie diaria: MITECO vs FIRMS (la pseudo-replicación de FIRMS) ------
    comp = pd.read_csv(DIR / "dataset" / "comparacion_miteco_vs_firms.csv")
    dias = [{"fecha": r.fecha, "n_miteco": int(r.mit_n),
             "n_focos_firms": int(r.firms_n_focos),
             "pctl_miteco": None if pd.isna(r.mit_pctl) else round(float(r.mit_pctl), 1),
             "pctl_firms": round(float(r.firms_pctl), 1),
             "auc": None if pd.isna(r.mit_auc) else round(float(r.mit_auc), 3)}
            for r in comp.itertuples()]

    # --- mapa: riesgo medio por estación + incendios oficiales ---------------
    est = (tD0.groupby(["idema", "nombre"], as_index=False)
              .agg(lat=("lat", "first"), lon=("lon", "first"),
                   riesgo=("prob", "mean"), pctl=("pctl_modelo", "mean"),
                   n_fuego=("label", "sum")))
    est = est[(est.lat.between(35, 44.5)) & (est.lon.between(-9.6, 4.5))]
    fdia = fuegos[fuegos.fecha.isin(tD0.fecha.unique())]

    payload = {
        "meta": {
            "titulo": "Validación del modelo de riesgo de incendios contra los partes oficiales del MITECO",
            "radio_km": RADIO_KM,
            "n_incidentes": int(len(fuegos)),
            "n_incidentes_en_ventana": int(len(fdia)),
            "rango": [str(tD0.fecha.min().date()), str(tD0.fecha.max().date())],
            "n_dias": int(tD0.fecha.nunique()),
            "n_estacion_dia": int(len(tD0)),
        },
        "metricas": {c: {"D0": metricas_globales(tD0, c),
                         "D1": metricas_globales(tD1, c)} for c in SCORES},
        "etiquetas_score": {c: n for c, (_, n) in SCORES.items()},
        "ganancia": {"D0": {c: curva_ganancia(tD0, c) for c in SCORES},
                     "D1": {c: curva_ganancia(tD1, c) for c in SCORES}},
        "deciles": {"D0": {c: por_decil(tD0, c) for c in SCORES},
                    "D1": {c: por_decil(tD1, c) for c in SCORES}},
        "distribucion": hist,
        "dias": dias,
        "niveles": {"D0": niveles_operativos(tD0), "D1": niveles_operativos(tD1)},
        "mapa": {
            "estaciones": [{"n": r.nombre, "lat": round(r.lat, 3),
                            "lon": round(r.lon, 3), "r": round(r.riesgo, 3),
                            "f": int(r.n_fuego)} for r in est.itertuples()],
            "fuegos": [{"lat": round(r.lat, 3), "lon": round(r.lon, 3),
                        "m": r.muni, "p": r.provincia,
                        "d": str(r.fecha.date())} for r in fdia.itertuples()],
        },
    }

    salida = DIR / "dataset" / "dashboard_miteco.json"
    salida.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado {salida} ({salida.stat().st_size/1024:.0f} KB)")
    construir_panel(salida.read_text(encoding="utf-8"))

    print("\n== Métricas globales (percentil intra-día, etiqueta MITECO) ==")
    for c, (_, nombre) in SCORES.items():
        m = payload["metricas"][c]["D0"]
        print(f"  {nombre:28s} AUC-ROC={m['auc_roc']:.3f}  "
              f"lift decil={m['lift_decil']:.2f}×  pctl={m['pctl_mediano']:.1f}")
    print("\n== Niveles operativos (D0) ==")
    for n in payload["niveles"]["D0"]:
        print(f"  {n['nivel']:10s} {n['pct_territorio']:5.1f}% del territorio → "
              f"{n['pct_incendios']:5.1f}% de los incendios  (lift {n['lift']}×)")


if __name__ == "__main__":
    main()
