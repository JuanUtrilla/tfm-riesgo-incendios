#!/usr/bin/env python3
"""
Climatología de FWI calculada desde AEMET — arregla el cruce de fuentes.

NO SOBRESCRIBE NADA. Escribe en un directorio NUEVO,
`aemet_horario_verano2026/modelo/clim_fwi_aemet/`, dejando intacto el
`clim_fwi/` original. El cambio en producción es una línea de ruta y es
decisión del usuario, no de este script.

=============================================================================
EL FALLO QUE ARREGLA
=============================================================================
`fwi_pctl_local` = percentil del FWI de hoy dentro de la climatología de esa
estación. En ENTRENAMIENTO los dos términos salen del cubo (ERA5-Land):
coherente. En PRODUCCIÓN el numerador es un FWI de estación AEMET y la
referencia sigue siendo la del cubo. Como el FWI de AEMET corre muy por encima
del del cubo, el percentil se desplaza y satura:

    entrenamiento (jul-ago): mediana 54,4 · 1,0 % de filas en pctl ≥99,9
    producción              : mediana 87,1 · 12,6 % de filas en pctl ≥99,9

La feature deja de discriminar en el techo del ranking, que es justo donde
vive el AUC operativo con 1 % de prevalencia.

=============================================================================
CÓMO SE CONSTRUYE
=============================================================================
Mismo procedimiento que `preparar_prototipo.py` pero con meteo de AEMET:

1. Serie diaria por estación, 2015-2025 (`descargar_historico_aemet.py`).
2. FWI con `calcular_fwi_serie`, EL MISMO código que usa producción — si se
   usara otra implementación el arreglo introduciría un desajuste nuevo.
3. Se guarda, por mes, el vector ORDENADO de valores de FWI, que es lo que
   `ranking_diario.py` espera encontrar en el .npz.

Reglas de calidad, para no emitir climatologías basura:

· El FWI es un integrador recursivo: necesita series continuas. Se procesa
  **año a año**, con spin-up desde el 1 de enero, y se descarta el año
  completo de una estación si le falta más del 20 % de los días.
· Se exigen ≥5 años válidos y ≥60 valores por mes para emitir ese mes. Un mes
  con pocos datos daría percentiles a saltos.
· Si una estación no llega al mínimo, NO se le escribe .npz: `ranking_diario`
  ya devuelve NaN en ese caso y XGBoost gestiona el NaN de forma nativa.
  Mejor sin feature que con una feature mal calibrada.
· Los huecos cortos (≤3 días) se interpolan linealmente en T/HR/viento y se
  rellenan con 0 en precipitación, que es lo que hace un día sin parte de
  lluvia. Los huecos largos parten el año y lo invalidan.

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python construir_clim_fwi_aemet.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).parent
REPO_OP = Path("/home/charredgem/Desktop/Master/aemet_horario_verano2026")
sys.path.insert(0, str(REPO_OP))
from fwi_canadiense import calcular_fwi_serie          # noqa: E402

ENTRADA = DIR / "dataset" / "aemet_historico_2015_2025.parquet"
SALIDA = REPO_OP / "modelo" / "clim_fwi_aemet"
INFORME = DIR / "dataset" / "clim_fwi_aemet_informe.json"

MAX_HUECO = 3          # días consecutivos que se interpolan
FALTA_MAX_ANIO = 0.20  # fracción de días ausentes que invalida un año
MIN_ANIOS = 5
MIN_POR_MES = 60


def serie_anual(s, anio):
    """Serie diaria continua de un año, o None si tiene demasiados huecos."""
    cal = pd.date_range(f"{anio}-01-01", f"{anio}-12-31", freq="D")
    d = s.set_index("fecha").reindex(cal)
    falta = d["tmax"].isna() | d["hr_min"].isna() | d["viento_max"].isna()
    if falta.mean() > FALTA_MAX_ANIO:
        return None
    # un hueco largo rompe la integración de los códigos de sequía
    grupos = (falta != falta.shift()).cumsum()[falta]
    if len(grupos) and grupos.value_counts().max() > MAX_HUECO:
        return None
    for c in ["tmax", "tmin", "hr_min", "viento_max"]:
        d[c] = d[c].interpolate(limit=MAX_HUECO, limit_direction="both")
    d["prec"] = d["prec"].fillna(0.0)
    if d[["tmax", "hr_min", "viento_max"]].isna().any().any():
        return None
    return d.rename_axis("fecha").reset_index()


def main():
    if not ENTRADA.exists():
        raise SystemExit(f"falta {ENTRADA}: ejecuta descargar_historico_aemet.py")
    hist = pd.read_parquet(ENTRADA)
    est = pd.read_parquet(REPO_OP / "modelo" / "estaciones_prototipo.parquet")
    objetivo = set(est["idema"])
    print(f"histórico: {len(hist):,} estación-día · {hist.idema.nunique()} "
          f"estaciones · estaciones del modelo: {len(objetivo)}")

    SALIDA.mkdir(parents=True, exist_ok=True)
    informe = {"emitidas": 0, "descartadas": 0, "por_estacion": {},
               "motivos": {"sin_historico": 0, "pocos_anios": 0, "pocos_meses": 0}}

    for idema, s in hist.groupby("idema"):
        if idema not in objetivo:
            continue
        valores = {m: [] for m in range(1, 13)}
        anios_ok = 0
        for anio in sorted(s["fecha"].dt.year.unique()):
            d = serie_anual(s[s["fecha"].dt.year == anio], anio)
            if d is None:
                continue
            anios_ok += 1
            fwi = calcular_fwi_serie(d["tmax"].values, d["hr_min"].values,
                                     d["viento_max"].values * 3.6,
                                     d["prec"].values,
                                     d["fecha"].dt.month.values)["fwi"]
            mes = d["fecha"].dt.month.values
            ok = np.isfinite(fwi)
            for m in range(1, 13):
                valores[m].append(fwi[ok & (mes == m)])

        if anios_ok < MIN_ANIOS:
            informe["descartadas"] += 1
            informe["motivos"]["pocos_anios" if anios_ok else "sin_historico"] += 1
            continue
        arrays = {f"m{m}": np.sort(np.concatenate(v)) if v else np.array([])
                  for m, v in valores.items()}
        n_meses = sum(len(a) >= MIN_POR_MES for a in arrays.values())
        if n_meses < 6:
            informe["descartadas"] += 1
            informe["motivos"]["pocos_meses"] += 1
            continue
        # un mes con muestra insuficiente se emite vacío: ranking_diario da NaN
        arrays = {k: (a if len(a) >= MIN_POR_MES else np.array([]))
                  for k, a in arrays.items()}
        np.savez_compressed(SALIDA / f"{idema}.npz", **arrays)
        informe["emitidas"] += 1
        informe["por_estacion"][idema] = {
            "anios": int(anios_ok),
            "n_jul": int(len(arrays["m7"])), "n_ago": int(len(arrays["m8"])),
            "p50_jul": round(float(np.median(arrays["m7"])), 1) if len(arrays["m7"]) else None,
            "p99_jul": round(float(np.percentile(arrays["m7"], 99)), 1) if len(arrays["m7"]) else None}

    print(f"\nclimatologías emitidas: {informe['emitidas']} · "
          f"descartadas: {informe['descartadas']} {informe['motivos']}")
    INFORME.write_text(json.dumps(informe, indent=2, ensure_ascii=False))
    print(f"→ {SALIDA}\n→ {INFORME}")


if __name__ == "__main__":
    main()
