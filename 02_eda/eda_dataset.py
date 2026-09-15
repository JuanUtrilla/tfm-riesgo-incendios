#!/usr/bin/env python3
"""
EDA del dataset del Modelo B, solo sobre el split train (2015-2018), para que
ninguna decisión de diseño se tome mirando val/test (anti-leakage §7.4-V4).

Entrada:  dataset/dataset_modelo_v1.parquet
Salidas:  eda/eda_distribuciones.png   distribuciones por clase de las features principales
          eda/eda_estacionalidad.png   positivos/negativos por mes + día del año
          eda/eda_normalizacion_local.png  el argumento FWI absoluto vs percentil local
          eda/eda_correlaciones.png    matriz de correlación (Spearman) de features
          stdout: estadísticos que van a la bitácora

El gráfico que va a la memoria es eda_normalizacion_local: muestra que el FWI
absoluto de los días de incendio varía mucho entre CCAA (Galicia arde con FWI
bajo, Andalucía necesita FWI alto) pero el percentil local de los días de
incendio es alto en todas → justifica la feature fwi_pctl_local (§7.1-N5).
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR = os.environ.get("TFM_DATOS", str(RAIZ / "datos"))

FEATS_DIST = ["fwi", "fwi_pctl_local", "rh_min", "vpd_max", "t2m_max",
              "dias_sin_lluvia", "precip_30d", "viento_max", "ndvi",
              "n_fuegos_10km_365d", "popdens", "dist_carreteras"]


def main():
    df = pd.read_parquet(f"{DIR}/dataset/dataset_modelo_v1.parquet")
    tr = df[df.split == "train"]
    pos, neg = tr[tr.label == 1], tr[tr.label == 0]
    print(f"EDA sobre train: {len(tr)} filas ({len(pos)} pos / {len(neg)} neg)")

    # --- 1. distribuciones por clase ---
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, col in zip(axes.flat, FEATS_DIST):
        lo, hi = np.nanpercentile(tr[col], [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(neg[col], bins=bins, alpha=0.55, density=True, label="no fuego")
        ax.hist(pos[col], bins=bins, alpha=0.55, density=True, label="fuego")
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Distribución por clase (train 2015-2018)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{DIR}/eda/eda_distribuciones.png", dpi=130)

    print("\nMedianas pos vs neg (train):")
    for col in FEATS_DIST:
        print(f"  {col:22s} pos={pos[col].median():9.2f}  neg={neg[col].median():9.2f}")

    # --- 2. estacionalidad ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    t = tr.groupby(["mes", "label"]).size().unstack(fill_value=0)
    axes[0].bar(t.index - 0.2, t[1], width=0.4, label="fuego")
    axes[0].bar(t.index + 0.2, t[0] / 3, width=0.4, label="no fuego (÷3)")
    axes[0].set_xlabel("mes"); axes[0].set_title("Muestras por mes"); axes[0].legend()
    axes[1].hist(pos["dia_anio"], bins=52, alpha=0.7)
    axes[1].set_xlabel("día del año"); axes[1].set_title("Positivos por día del año")
    fig.tight_layout(); fig.savefig(f"{DIR}/eda/eda_estacionalidad.png", dpi=130)

    # --- 3. normalización local: el gráfico de la memoria ---
    ccaa_top = pos["ccaa"].value_counts().head(8).index
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    datos_abs = [pos.loc[pos.ccaa == c, "fwi"].dropna() for c in ccaa_top]
    datos_pct = [pos.loc[pos.ccaa == c, "fwi_pctl_local"].dropna() for c in ccaa_top]
    etiquetas = [f"CCAA {int(c)}" for c in ccaa_top]
    axes[0].boxplot(datos_abs, tick_labels=etiquetas, showfliers=False)
    axes[0].set_title("FWI ABSOLUTO en días de incendio, por CCAA")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].boxplot(datos_pct, tick_labels=etiquetas, showfliers=False)
    axes[1].set_title("PERCENTIL LOCAL del FWI en días de incendio, por CCAA")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("El riesgo es relativo a la climatología local (train)")
    fig.tight_layout(); fig.savefig(f"{DIR}/eda/eda_normalizacion_local.png", dpi=130)

    print("\nFWI mediano en día de incendio por CCAA (top 8) — absoluto vs percentil:")
    for c in ccaa_top:
        sub = pos[pos.ccaa == c]
        print(f"  CCAA {int(c):2d}: FWI={sub['fwi'].median():5.1f}  "
              f"pctl={sub['fwi_pctl_local'].median():5.1f}  (n={len(sub)})")

    # --- 4. correlaciones (Spearman) para leer SHAP con cuidado ---
    feats_num = [c for c in FEATS_DIST + ["fwi_med_30d", "fwi_anom_sigma",
                 "precip_7d", "swi010", "lai", "elevacion", "pendiente"]
                 if c in tr.columns]
    corr = tr[feats_num].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats_num)), feats_num, rotation=90, fontsize=8)
    ax.set_yticks(range(len(feats_num)), feats_num, fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlación (Spearman) — train")
    fig.tight_layout(); fig.savefig(f"{DIR}/eda/eda_correlaciones.png", dpi=130)

    alto = [(feats_num[i], feats_num[j], corr.iloc[i, j])
            for i in range(len(feats_num)) for j in range(i + 1, len(feats_num))
            if abs(corr.iloc[i, j]) > 0.8]
    print("\nPares con |rho|>0.8 (cuidado al interpretar SHAP):")
    for a, b, r in alto:
        print(f"  {a} ~ {b}: {r:.2f}")


if __name__ == "__main__":
    main()
