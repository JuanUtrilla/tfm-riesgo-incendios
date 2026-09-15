#!/usr/bin/env python3
"""
Ensamblado y control de calidad del dataset del Modelo B.

Entradas: dataset/muestra_maestra_v1.parquet
          dataset/features_cubo_v1.parquet
          dataset/features_historia_v1.parquet
Salida:   dataset/dataset_modelo_v1.parquet  (tabla final de entrenamiento)
          stdout: informe QC completo (se archiva en MODELO_B_BITACORA.md)

Pasos (cada uno documentado para la memoria):
1. Join 1:1 de los tres bloques por id_muestra (assert de cardinalidad).
2. Features derivadas:
   - vpd_max (kPa): Magnus sobre t2m_max y rh_min, proxy de sequedad atmosférica
     que la literatura sitúa como driver dominante de propagación (Fire Ecology 2026).
   - dia_anio: día del año crudo, sin codificación cíclica (árboles, §7.1-N7).
3. Filtrado de negativos ambiguos: pseudo-ausencias con is_near_fire_dia=1
   (fuego EFFIS a <12,5 km en los 10 días previos que el buffer EGIF no vio:
   típicamente incendios del lado portugués o perímetros grandes). Los positivos
   no se filtran (su is_near_fire refleja su propio incendio).
4. QC: rangos físicos, % NaN por feature y split, duplicados, prevalencia final
   por split (el ratio del test queda fijado y reportado, porque AUC-PR no es
   comparable entre prevalencias distintas, §7.3-D5).
5. NaN se dejan como NaN (manejo nativo de XGBoost, §7.1-N3). Sin escalado,
   sin winsorización, sin imputación (§7.5 checklist negativo).
"""

import numpy as np
import pandas as pd

DIR = "/home/charredgem/Desktop/Master/TFM_fuego/dataset"

RANGOS_FISICOS = {
    "fwi": (0, 200), "t2m_max": (-30, 50), "t2m_min": (-40, 45),
    "rh_min": (0, 100), "viento_max": (0, 70), "precip_dia": (0, 400),
    "ndvi": (-1, 1), "lai": (0, 12), "swi010": (0, 100),
    "fwi_pctl_local": (0, 100), "elevacion": (-10, 3600), "pendiente": (0, 90),
    "popdens": (0, 60000), "dias_sin_lluvia": (0, 120),
}


def vpd_kpa(t_c, rh_pct):
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    return es * (1.0 - rh_pct / 100.0)


def main():
    maestra = pd.read_parquet(f"{DIR}/muestra_maestra_v1.parquet")
    cubo = pd.read_parquet(f"{DIR}/features_cubo_v1.parquet")
    historia = pd.read_parquet(f"{DIR}/features_historia_v1.parquet")

    df = (maestra.merge(cubo, on="id_muestra", validate="1:1")
                 .merge(historia, on="id_muestra", validate="1:1"))
    assert len(df) == len(maestra), "se perdieron filas en el join"
    print(f"Join: {len(df)} filas × {len(df.columns)} columnas")

    # --- derivadas ---
    df["vpd_max"] = vpd_kpa(df["t2m_max"], df["rh_min"])
    df["dia_anio"] = pd.DatetimeIndex(df["fecha"]).dayofyear

    # --- filtrado de negativos ambiguos ---
    ambiguos = (df["label"] == 0) & (df["is_near_fire_dia"] == 1)
    print(f"\nNegativos con is_near_fire=1 (EFFIS no visto por buffer EGIF): "
          f"{ambiguos.sum()} ({ambiguos.mean()*100:.2f}% del total) → eliminados")
    df = df[~ambiguos].reset_index(drop=True)

    # --- QC: duplicados ---
    dup = df.duplicated(subset=["ix", "iy", "fecha"]).sum()
    print(f"Duplicados (celda,fecha): {dup}")
    assert dup == 0

    # --- QC: rangos físicos ---
    print("\n== Rangos físicos (violaciones) ==")
    algun_fallo = False
    for col, (lo, hi) in RANGOS_FISICOS.items():
        v = df[col]
        n_bad = int(((v < lo) | (v > hi)).sum())
        if n_bad:
            algun_fallo = True
            print(f"  ⚠️ {col}: {n_bad} fuera de [{lo},{hi}] "
                  f"(min={v.min():.2f}, max={v.max():.2f})")
    if not algun_fallo:
        print("  todas las features dentro de rango")

    # --- QC: NaN por feature (solo las que tienen alguno) ---
    print("\n== % NaN por feature y split (features con >0%) ==")
    no_feat = {"id_muestra", "id_egif", "fecha", "anio", "mes", "split", "label",
               "ix", "iy", "x3035", "y3035", "lat_celda", "lon_celda",
               "superficie", "causa", "is_near_fire_dia", "is_fire_dia"}
    feats = [c for c in df.columns if c not in no_feat]
    tabla_nan = (df.groupby("split")[feats].apply(lambda g: g.isna().mean() * 100)
                   .T.round(2))
    tabla_nan = tabla_nan[(tabla_nan > 0).any(axis=1)]
    print(tabla_nan.to_string() if len(tabla_nan) else "  sin NaN")

    # --- prevalencia final por split (fijada para la memoria) ---
    print("\n== Prevalencia final por split ==")
    t = df.groupby(["split", "label"]).size().unstack()
    t["ratio_neg_pos"] = (t[0] / t[1]).round(3)
    t["prevalencia_%"] = (t[1] / (t[0] + t[1]) * 100).round(2)
    print(t.to_string())

    df.to_parquet(f"{DIR}/dataset_modelo_v1.parquet", index=False)
    print(f"\nGuardado {DIR}/dataset_modelo_v1.parquet "
          f"({len(df)} filas, {len(feats)} features candidatas)")
    print("Features:", sorted(feats))


if __name__ == "__main__":
    main()
