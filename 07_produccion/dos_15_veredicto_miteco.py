#!/usr/bin/env python3
"""
Veredicto RÁPIDO: los mapas de cada día juzgados con el parte de MITECO del
día siguiente. Cierra el bucle en 24 h (EFFIS tarda 6-9 días).

NO TOCA PRODUCCIÓN ni TFM-RAG. Descarga el parte a salida/miteco/, lee los
históricos de TFM-RAG en solo lectura, reutiliza su parser y el geocodificador
de `validar_miteco.py` del repo original (importados, no copiados). Escribe
salida/veredicto_miteco.csv y salida/veredicto_miteco.json.

=============================================================================
CÓMO (ver ESTUDIO_MITECO_DIARIO.md para las medidas que lo justifican)
=============================================================================
· El "Parte Definitivo de Intervenciones (día previo)" del día D se publica
  el D+1 hacia las 13-15 h. Se descarga y, si su fecha es nueva, se guarda.
· Incidente = primera aparición de su `incident_key` (el 97 % no trae fecha
  de inicio). Localización = centro del municipio (error 5-15 km).
· Etiqueta: celdas a ≤10 km y ≤25 km del municipio, el día D.
· Mapas del día D: producción (caché del prototipo), malla (`riesgo_hoy`),
  único y pareja (`dos_riesgo_hoy`). Solo el último mapa guardado para esa
  fecha (la pasada D0 pisa la D+1 de la víspera: es lo que se publicó
  para HOY esa mañana).
· Métricas por día: AUC (celdas del incidente contra el resto de España),
  percentil mediano, y si el incidente cae en ALTO/EXTREMO (≥p90).
· Se reescriben los últimos 10 días (por si llega un parte tardío) y se
  recalcula el acumulado con bootstrap por días.

~4 incidentes/día: un día no decide nada; semanas sí.
"""

import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import config
from comparar_julio2026 import auc

RAG = "/home/charredgem/Desktop/Master/TFM-RAG"
PDF_RAG = f"{RAG}/data/raw/miteco"
# sin TFM-RAG / TFM_fuego (GitHub): copias vendorizadas en estado/vendor
VENDOR = f"{config.ESTADO}/vendor"
PDF_MIO = config.salida("miteco")
CSV = config.salida("veredicto_miteco.csv")
RADIOS = (10, 25)
VENTANA = 10


PAGINA = ("https://www.miteco.gob.es/es/biodiversidad/temas/"
          "incendios-forestales/estadisticas-actuaciones.html")


def descargar():
    """El parte de hoy (datos de ayer). Misma página y mismo criterio de
    enlace que el downloader de TFM-RAG, pero con regex en vez de bs4 (el
    entorno tfm_fuego no lo tiene); la fecha del parte la saca su parser."""
    import re
    import unicodedata
    from urllib.parse import urljoin
    import requests
    os.makedirs(PDF_MIO, exist_ok=True)
    sys.path.insert(0, f"{RAG}/src" if os.path.isdir(RAG) else VENDOR)
    try:
        # fecha del parte con el parser de TFM-RAG (su downloader importa bs4,
        # que este entorno no tiene)
        import io, tempfile, pathlib
        if os.path.isdir(RAG):
            from miteco_rag.parseo_y_chuncking import extract_pdf_lines, extract_report_date
        else:
            from parseo_y_chuncking import extract_pdf_lines, extract_report_date
        html = requests.get(PAGINA, timeout=30, headers={"User-Agent": "TFM-fuego-malla/0.1"}).text
        url = None
        for href, txt in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S | re.I):
            t = unicodedata.normalize("NFKD", re.sub("<[^>]+>", " ", txt))
            t = " ".join("".join(c for c in t if not unicodedata.combining(c)).casefold().split())
            if "parte definitivo de intervenciones" in t and "dia previo" in t:
                url = urljoin(PAGINA, href); break
        if url is None:
            raise RuntimeError("enlace al parte no encontrado en la página")
        pdf = requests.get(url, timeout=60).content
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf); tmp_path = pathlib.Path(tmp.name)
        try:
            fecha = extract_report_date("\n".join(l.cleaned_text for l in extract_pdf_lines(tmp_path)))
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"  descarga MITECO falló: {type(e).__name__} {e}", flush=True)
        return None
    ruta = f"{PDF_MIO}/ActuacionesMITECO-definitivo-{fecha}.pdf"
    if not os.path.exists(ruta) and not glob.glob(f"{PDF_RAG}/*{fecha}*.pdf"):
        open(ruta, "wb").write(pdf)
        print(f"  parte nuevo: {fecha} ({len(pdf)//1024} KB)", flush=True)
    else:
        print(f"  parte {fecha}: ya estaba", flush=True)
    return fecha


def incidentes():
    """Primera aparición de cada incidente, geocodificada."""
    if os.path.isdir(RAG):
        sys.path.insert(0, f"{RAG}/src")
        from miteco_rag.parseo_y_chuncking import parse_miteco_pdf
    else:
        sys.path.insert(0, VENDOR)
        from parseo_y_chuncking import parse_miteco_pdf
    import pathlib
    snaps = []
    for d in (PDF_RAG, PDF_MIO):
        for f in sorted(glob.glob(f"{d}/*.pdf")):
            try:
                snaps += parse_miteco_pdf(pathlib.Path(f))
            except Exception as e:
                print(f"  parse falló {os.path.basename(f)}: {type(e).__name__}")
    df = pd.DataFrame([{
        "incident_key": s.incident_key, "fecha": pd.Timestamp(s.report_date),
        "pais": s.country, "provincia": s.province, "localizacion": s.location,
        "estado": s.status, "n_medios": len(s.assigned_resources)} for s in snaps])
    df = df[df.pais == "ES"].sort_values("fecha") \
           .groupby("incident_key", as_index=False).first()
    vm_ruta = f"{config.FUENTE}/validar_miteco.py"
    if not os.path.exists(vm_ruta):
        vm_ruta = f"{VENDOR}/validar_miteco.py"
    spec = importlib.util.spec_from_file_location("vm", vm_ruta)
    vm = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(vm)
    sys.argv = argv
    g = vm.geocodificar(df, vm.cargar_municipios())
    print(f"  incidentes geocodificados: {len(g)} · {g.fecha.min():%F} → {g.fecha.max():%F}")
    return g


def mapas(fecha):
    out = {}
    p = f"{config.FUENTE}/prototipo/cache/malla_prob_{fecha}.npz"
    if os.path.exists(p):
        out["produccion"] = np.load(p)["prob"].astype(float)
    md = config.salida("mapas_diarios")
    m = next((r for r in (f"{md}/riesgo_hoy_{fecha}.npz", config.salida(f"riesgo_hoy_{fecha}.npz"))
              if os.path.exists(r)), None)
    if m:
        out["malla"] = np.load(m)["prob"].astype(float)
    d = next((r for r in (f"{md}/dos_riesgo_{fecha}.npz", config.salida(f"dos_riesgo_{fecha}.npz"))
              if os.path.exists(r)), None)
    if d:
        z = np.load(d)
        out["unico"], out["pareja"] = z["prob_unico"].astype(float), z["prob_pareja"].astype(float)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ventana", type=int, default=VENTANA)
    ap.add_argument("--sin-descarga", action="store_true")
    args = ap.parse_args()
    if not args.sin_descarga:
        descargar()
    inc = incidentes()
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es = ds["is_spain"].values.astype(bool)
    xs, ys = ds["x"].values, ds["y"].values
    ds.close()
    X, Y = np.meshgrid(xs, ys)
    cx, cy = X[es], Y[es]
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    ix_, iy_ = tr.transform(inc["lon"].values, inc["lat"].values)

    hoy = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    desde = hoy - pd.Timedelta(days=args.ventana)
    # los incidentes recientes, geocodificados, para que `capa_verdad` los
    # pinte en el mapa de mañana sin volver a parsear los PDF (que es lo caro
    # de este script). El mapa se dibuja ANTES que este paso en la cadena, así
    # que usa siempre el parte de la corrida anterior: es el último publicado.
    rec = inc[inc.fecha >= hoy - pd.Timedelta(days=15)]
    rec[["fecha", "lat", "lon", "localizacion", "provincia", "n_medios"]] \
        .to_csv(config.salida("miteco_incidentes.csv"), index=False)
    print(f"  incidentes recientes para el mapa: {len(rec)}", flush=True)
    filas = []
    for d, g in inc[inc.fecha >= desde].groupby("fecha"):
        M = mapas(str(d.date()))
        if not M:
            continue
        k = g.index.values
        dist = np.min(np.hypot(cx[:, None] - ix_[k][None], cy[:, None] - iy_[k][None]), 1)
        fila = {"fecha": str(d.date()), "n_incidentes": len(g),
                "municipios": "; ".join(g["localizacion"].astype(str).head(6))}
        for nom, A in M.items():
            a = A[es]; ok = np.isfinite(a)
            orden = np.sort(a[ok])
            for R in RADIOS:
                q = dist <= R * 1000
                if not (q & ok).any():
                    continue
                pct = np.searchsorted(orden, a[ok & q], side="right") / len(orden) * 100
                fila[f"auc{R}_{nom}"] = round(auc(a[ok & q], a[ok & ~q]), 4)
                fila[f"pctl{R}_{nom}"] = round(float(np.median(pct)), 1)
            # ¿el incidente (mejor celda a 10 km) cayó en ALTO/EXTREMO (≥p90)?
            q10 = (dist <= 10_000) & ok
            if q10.any():
                fila[f"p90_{nom}"] = int(np.max(a[q10]) >= np.percentile(a[ok], 90))
        filas.append(fila)
        print(f"  {d.date()} · {len(g)} incidentes · " + " · ".join(
            f"{n} AUC10 {fila.get(f'auc10_{n}', float('nan')):.3f}" for n in M), flush=True)

    nuevo = pd.DataFrame(filas)
    if os.path.exists(CSV):
        viejo = pd.read_csv(CSV)
        if len(nuevo):
            viejo = viejo[~viejo["fecha"].isin(nuevo["fecha"])]
        nuevo = pd.concat([viejo, nuevo], ignore_index=True)
    if not len(nuevo):
        print("  sin días puntuables"); return
    nuevo = nuevo.sort_values("fecha")
    nuevo.to_csv(CSV, index=False)

    # acumulado, bootstrap por días, contra producción
    rng = np.random.default_rng(0)
    R = {"dias": int(len(nuevo)), "incidentes": int(nuevo.n_incidentes.sum())}
    print(f"\n== MITECO acumulado · {len(nuevo)} días · {int(nuevo.n_incidentes.sum())} incidentes ==")
    for nom in ("produccion", "malla", "unico", "pareja"):
        if f"auc10_{nom}" not in nuevo:
            continue
        s = nuevo.dropna(subset=[f"auc10_{nom}"])
        r = {"n_dias": int(len(s)), "auc10": float(s[f"auc10_{nom}"].mean()),
             "auc25": float(s[f"auc25_{nom}"].mean()),
             "pctl10": float(s[f"pctl10_{nom}"].mean()),
             "frac_alto_extremo": float(s[f"p90_{nom}"].mean()) if f"p90_{nom}" in s else None}
        if nom != "produccion" and f"auc10_produccion" in s:
            s2 = s.dropna(subset=["auc10_produccion"])
            dd = (s2[f"auc10_{nom}"] - s2["auc10_produccion"]).values
            if len(dd) >= 3:
                b = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
                r["dif_vs_prod"] = float(dd.mean())
                r["ic95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        R[nom] = r
        print(f"  {nom:11s} AUC10 {r['auc10']:.3f} · AUC25 {r['auc25']:.3f} · pctl {r['pctl10']:.1f}"
              + (f" · Δprod {r['dif_vs_prod']:+.3f} [{r['ic95'][0]:+.3f}, {r['ic95'][1]:+.3f}]"
                 if "dif_vs_prod" in r else ""))
    json.dump(R, open(config.salida("veredicto_miteco.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
