#!/usr/bin/env python3
"""Figuras del replay: (1) por temporada y condición, serie acumulada del AUC
medio con banda bootstrap de la diferencia contra producción (estilo dos_19);
(2) las cuatro corridas lado a lado (AUC medio ± IC de la diferencia).
Escribe replay/fig_replay_<año>_<condicion>.png y replay/fig_replay_resumen.png."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
AQUI = os.path.dirname(os.path.abspath(__file__))
MODELOS = {"prod": "producción (malla)", "unico": "único 1:3", "r10": "único 1:10",
           "pareja": "pareja", "donde": "dónde solo", "cuando": "cuándo solo"}
COL = {"prod": "0.3", "unico": "#d62728", "r10": "#ff7f0e", "pareja": "#1f77b4", "donde": "#2ca02c", "cuando": "#9467bd"}
rng = np.random.default_rng(42)

def acumulada(df, nombre):
    con = df[df["celdas_quemadas"] > 0].sort_values("fecha").reset_index(drop=True)
    if len(con) < 3: return
    f = pd.to_datetime(con["fecha"])
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
    for k, lab in MODELOS.items():
        a = con[f"auc_{k}"].values
        ax[0].plot(f, np.nancumsum(a) / np.arange(1, len(a) + 1), color=COL[k], lw=2 if k in ("prod", "unico", "r10", "pareja") else 1, label=lab, alpha=1 if k in ("prod", "unico", "r10", "pareja") else 0.6)
    ax[0].set_ylabel("AUC medio acumulado (días con fuego)"); ax[0].legend(ncol=3, fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Replay {nombre}: {len(con)} días con fuego · {con['area_ha'].sum():,.0f} ha")
    for k in ("unico", "r10", "pareja"):
        d = con[f"auc_{k}"].values - con["auc_prod"].values
        med, lo, hi = [], [], []
        for n in range(1, len(d) + 1):
            x = d[:n]; med.append(x.mean())
            if n >= 3:
                b = np.array([x[rng.integers(0, n, n)].mean() for _ in range(400)])
                lo.append(np.percentile(b, 2.5)); hi.append(np.percentile(b, 97.5))
            else: lo.append(np.nan); hi.append(np.nan)
        ax[1].plot(f, med, color=COL[k], lw=2, label=f"{MODELOS[k]} − producción")
        ax[1].fill_between(f, lo, hi, color=COL[k], alpha=0.12)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("Δ AUC acumulada vs producción"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].text(0.01, 0.04, "el número que vale es el del último día; la banda es IC95 bootstrap por días", transform=ax[1].transAxes, fontsize=7, color="0.4")
    fig.tight_layout(); fig.savefig(f"{AQUI}/replay/fig_replay_{nombre}.png", dpi=140); plt.close(fig)

def resumen():
    V = json.load(open(f"{AQUI}/replay/replay_veredicto.json"))
    corridas = [c for c in ("2025_ifs", "2025_reanalisis", "2026_ifs", "2026_reanalisis") if c in V]
    if not corridas: return
    fig, ax = plt.subplots(1, len(corridas), figsize=(4.2 * len(corridas), 4.6), sharey=True)
    ax = np.atleast_1d(ax)
    for a, c in zip(ax, corridas):
        M = V[c]["modelos"]; ks = list(MODELOS)
        y = [M[k]["auc_medio"] for k in ks]
        err = [[M[k]["auc_medio"] - (M["prod"]["auc_medio"] + M[k]["ic95_dif"][0]) if k != "prod" else 0 for k in ks],
               [(M["prod"]["auc_medio"] + M[k]["ic95_dif"][1]) - M[k]["auc_medio"] if k != "prod" else 0 for k in ks]]
        a.bar(range(len(ks)), y, color=[COL[k] for k in ks], yerr=err, capsize=3)
        a.axhline(M["prod"]["auc_medio"], color="0.3", ls="--", lw=1)
        a.set_xticks(range(len(ks))); a.set_xticklabels([MODELOS[k] for k in ks], rotation=35, ha="right", fontsize=8)
        a.set_title(f"{c.replace('_', ' ')} · {V[c]['n_dias_con_fuego']} días", fontsize=10); a.set_ylim(0.5, 1.0); a.grid(axis="y", alpha=0.3)
    ax[0].set_ylabel("AUC medio por día (barra = IC95 de la Δ vs producción)")
    fig.tight_layout(); fig.savefig(f"{AQUI}/replay/fig_replay_resumen.png", dpi=140); plt.close(fig)

if __name__ == "__main__":
    for csv in sorted(glob.glob(f"{AQUI}/replay/replay_*_*.csv")):
        nom = os.path.basename(csv)[7:-4]
        if nom.startswith("si"): continue
        acumulada(pd.read_csv(csv), nom)
    resumen(); print("figuras:", sorted(os.path.basename(f) for f in glob.glob(f"{AQUI}/replay/fig_*.png")))
