#!/usr/bin/env python3
"""
Hindcast de julio 2026: mapas diarios retrospectivos + verificación contra
los focos FIRMS reales de cada día.

⚠️ HONESTIDAD METODOLÓGICA: estos mapas se evalúan con meteo OBSERVADA del
propio día (nowcast retrospectivo) y se generan a posteriori — NO son
predicciones selladas. Las selladas por commit empiezan el 15/07 (ranking
D-1) y el 22/07 (mapas D0/D+1 con forecast). Por eso los archivos llevan el
sufijo `_retro` y la columna `prevision='retro'` en la verificación.

Salidas:
- mapas/historico/<fecha>_D0_retro.jpg      (mapa + focos FIRMS de ese día)
- rankings/retro_<fecha>.csv                (probabilidad por estación)
- rankings/verificacion_retro_julio.csv     (métrica diaria retro)
- mapas/seguimiento_julio_2026.png          (resumen del mes)
La verificación del 22/07 (previsión D0 SÍ sellada) se añade a
rankings/verificacion_diaria.csv, la misma serie que alimenta el cron.

Uso (local, una vez): AEMET_API_KEY=... FIRMS_MAP_KEY=... python3 retro_julio.py
"""

import json
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

os.environ["PRESUPUESTO_AEMET_S"] = "0"   # días pasados: no hay forecast que pedir

import mapa_diario as md
from mapa_diario import RAIZ, frp_estaciones, generar_mapa, verificar_prevision

DIAS = pd.date_range("2026-07-01", "2026-07-21", freq="D")


def firms_julio():
    """Detecciones FIRMS 26-jun → 25-jul (6 peticiones de 5 días)."""
    key = os.environ["FIRMS_MAP_KEY"]
    trozos = []
    for d0 in ["2026-06-26", "2026-07-01", "2026-07-06", "2026-07-11",
               "2026-07-16", "2026-07-21"]:
        r = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                         f"{key}/VIIRS_NOAA20_NRT/-10,35,5,44/5/{d0}", timeout=60)
        if r.ok and r.text.startswith("latitude"):
            trozos.append(pd.read_csv(StringIO(r.text)))
        else:
            print(f"FIRMS {d0}: sin datos", flush=True)
    det = pd.concat(trozos, ignore_index=True).drop_duplicates()
    det["d"] = pd.to_datetime(det["acq_date"])
    print(f"FIRMS: {len(det)} detecciones {det['d'].min().date()} → "
          f"{det['d'].max().date()}", flush=True)
    return det


def main():
    import xgboost as xgb
    with open(RAIZ / "modelo" / "xgb_v2_prototipo_features.json") as fh:
        feats = json.load(fh)
    modelo = xgb.XGBClassifier()
    modelo.load_model(str(RAIZ / "modelo" / "xgb_v2_prototipo.ubj"))
    est = pd.read_parquet(RAIZ / "modelo" / "estaciones_prototipo.parquet") \
            .set_index("idema")
    det = firms_julio()

    # todas las fechas de una vez: la serie y el FWI se calculan UNA vez por
    # estación y las features salen por índice de fecha
    rks = md.evaluar_estaciones(list(DIAS), est)
    estat = dict(np.load(md.MALLA / "estaticas.npz"))
    estat["is_spain"] = estat["is_spain"].astype(bool)
    (RAIZ / "mapas" / "historico").mkdir(exist_ok=True, parents=True)

    from mapa_diario import UMBRALES
    vers = []
    for dia in DIAS:
        rk = rks[dia]
        rk = rk[np.isfinite(rk["fwi"]) & np.isfinite(rk["t2m_max"])
                & np.isfinite(rk["rh_min"]) & np.isfinite(rk["elevacion"])] \
            .reset_index(drop=True)
        if len(rk) < 100:
            print(f"{dia.date()}: solo {len(rk)} estaciones — omitido")
            continue
        # features FIRMS con la ventana [D-5, D-1] de ESE día (sin el propio
        # día: misma regla anti-circularidad que en producción)
        ventana = det[(det["d"] >= dia - pd.Timedelta(days=5)) & (det["d"] < dia)]
        rk["frp_max_50km_7d"], rk["n_detec_50km_7d"] = frp_estaciones(ventana, rk)
        rk["prob"] = modelo.predict_proba(rk[feats])[:, 1].round(4)
        rk["nivel"] = rk["prob"].map(lambda p: next(n for u, n in UMBRALES if p >= u))

        cols = ["idema", "nombre", "lat", "lon", "prob", "nivel",
                "fuente_forecast", "dias_forecast", "fwi", "fwi_pctl_local",
                "fwi_anom_sigma", "t2m_max", "rh_min", "viento_max",
                "dias_sin_lluvia", "precip_30d", "n_detec_50km_7d"]
        ruta_csv = RAIZ / "rankings" / f"retro_{dia.date()}.csv"
        rk.sort_values("prob", ascending=False)[cols].round(3) \
          .to_csv(ruta_csv, index=False)

        focos_dia = det[det["d"] == dia]
        generar_mapa(dia, rk, estat, ventana, modelo, feats,
                     RAIZ / "mapas" / "historico" / f"{dia.date()}_D0_retro.jpg",
                     det_overlay=focos_dia, etiqueta_focos="ese mismo día")
        v = verificar_prevision(det, dia, ruta_csv)
        if v is not None:
            vers.append({"prevision": "retro", **v})
            print(f"  {dia.date()}: {v['n_focos']} focos · pctl mediano "
                  f"{v['pctl_mediana_en_focos']} · lift {v['lift']}", flush=True)

    pd.DataFrame(vers).to_csv(RAIZ / "rankings" / "verificacion_retro_julio.csv",
                              index=False)

    # el 22/07 SÍ está sellado (prevision_D0 commiteada esa mañana) → va a la
    # serie oficial que alimenta la tabla del README
    v22 = verificar_prevision(det, pd.Timestamp("2026-07-22"),
                              RAIZ / "rankings" / "prevision_D0_2026-07-22.csv")
    if v22 is not None:
        ruta_v = RAIZ / "rankings" / "verificacion_diaria.csv"
        pd.DataFrame([{"prevision": "D0", **v22}]).to_csv(
            ruta_v, mode="a", header=not ruta_v.exists(), index=False)
        print(f"sellada 22/07 (D0): pctl mediano {v22['pctl_mediana_en_focos']} "
              f"· lift {v22['lift']}", flush=True)
        md.tabla_verificacion_readme()

    figura_seguimiento()


def figura_seguimiento():
    """Resumen del mes, pensado para leerse sin contexto: zona verde/roja,
    anotaciones en los días clave y pie con la definición de la métrica."""
    vdf = pd.read_csv(RAIZ / "rankings" / "verificacion_retro_julio.csv")
    vdf["fecha"] = pd.to_datetime(vdf["fecha_focos"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                 gridspec_kw={"hspace": 0.32})

    # ---- panel 1: ¿en qué percentil del riesgo previsto cayeron los fuegos?
    a1.axhspan(50, 100, color="seagreen", alpha=0.08)
    a1.axhspan(0, 50, color="firebrick", alpha=0.07)
    a1b = a1.twinx()
    a1b.bar(vdf["fecha"], vdf["n_focos"], color="lightsteelblue", alpha=0.55,
            label="nº de focos FIRMS ese día (eje derecho)")
    a1b.set_ylabel("nº focos FIRMS")
    a1.plot(vdf["fecha"], vdf["pctl_mediana_en_focos"], "o-", color="firebrick",
            lw=2, zorder=5,
            label="dónde ardió: percentil mediano del riesgo previsto (eje izq.)")
    a1.axhline(50, ls="--", c="gray", lw=1.2)
    a1.set_ylim(0, 100)
    a1.set_ylabel("percentil del riesgo previsto\n(100 = lo más señalado; 50 = azar)")
    a1.text(0.01, 0.93, "✔ punto AQUÍ = ardió donde el modelo señalaba más riesgo",
            transform=a1.transAxes, fontsize=9, color="darkgreen")
    a1.text(0.01, 0.06, "✘ punto AQUÍ = ardió donde el modelo decía riesgo bajo",
            transform=a1.transAxes, fontsize=9, color="darkred")

    mejor = vdf.loc[vdf["pctl_mediana_en_focos"].idxmax()]
    peor = vdf.loc[vdf["pctl_mediana_en_focos"].idxmin()]
    a1.annotate(f"{mejor['fecha']:%d-%b}: {int(mejor['n_focos'])} focos justo\n"
                f"en las zonas señaladas (pctl {mejor['pctl_mediana_en_focos']:.0f})",
                (mejor["fecha"], mejor["pctl_mediana_en_focos"]),
                xytext=(-95, -52), textcoords="offset points", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="darkgreen"),
                color="darkgreen")
    a1.annotate(f"{peor['fecha']:%d-%b}: fuegos sin señal meteo\n"
                f"(pctl {peor['pctl_mediana_en_focos']:.0f} — fallo claro)",
                (peor["fecha"], peor["pctl_mediana_en_focos"]),
                xytext=(12, 30), textcoords="offset points", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="darkred"), color="darkred")
    a1.set_title("Julio 2026 — ¿ardió donde el modelo decía?\n"
                 "(retrospectivo con meteo observada; las predicciones selladas "
                 "por commit empiezan el 15-22/07)", fontsize=12)
    a1.legend(loc="center left", fontsize=8.5)
    a1b.legend(loc="lower right", fontsize=8.5)

    # ---- panel 2: el aviso ALTO/EXTREMO, ¿concentró los fuegos?
    a2.plot(vdf["fecha"], vdf["pct_focos_en_altoextremo"], "o-", lw=2,
            color="darkorange",
            label="% de los focos que cayó en zonas avisadas ALTO/EXTREMO")
    a2.plot(vdf["fecha"], vdf["pct_estaciones_altoextremo"], "s--",
            color="gray", label="% del territorio avisado ALTO/EXTREMO (azar)")
    a2.fill_between(vdf["fecha"], vdf["pct_focos_en_altoextremo"],
                    vdf["pct_estaciones_altoextremo"],
                    where=vdf["pct_focos_en_altoextremo"]
                    >= vdf["pct_estaciones_altoextremo"],
                    color="seagreen", alpha=0.15, interpolate=True)
    a2.set_ylim(0, 100)
    a2.set_ylabel("%")
    a2.set_title("La franja verde entre las dos líneas = lo que el aviso aporta "
                 "sobre el azar (naranja arriba = acierta)", fontsize=10)
    a2.legend(loc="upper left", fontsize=8.5)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    fig.text(0.02, 0.005,
             "Cómo se mide: cada foco FIRMS se empareja con su estación AEMET más "
             "cercana (≤35 km). Percentil 80 = ese fuego ocurrió en un lugar con "
             "más riesgo previsto que el 80% de España ese día.\nSi el modelo no "
             "aportara nada, la línea roja rondaría 50 y las dos líneas de abajo "
             "irían pegadas.", fontsize=8, color="dimgray")
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(RAIZ / "mapas" / "seguimiento_julio_2026.png", dpi=140)
    print("guardado mapas/seguimiento_julio_2026.png")


if __name__ == "__main__":
    import sys
    if "--figura" in sys.argv:
        figura_seguimiento()      # regenerar solo el gráfico (CSV ya existente)
    else:
        main()
