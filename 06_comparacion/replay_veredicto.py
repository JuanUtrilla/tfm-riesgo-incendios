#!/usr/bin/env python3
"""Veredicto del replay (prerregistro en TRAZABILIDAD.md, 02/09/2026 18:10).

Lee replay/replay_<año>_<condicion>.csv y calcula, por modelo:
  AUC medio por día (días con fuego), IC95 bootstrap de días (2.000, semilla 42)
  de la diferencia contra producción-malla, % de días que gana a producción,
  percentil mediano y ponderado por ha, captura en el top-2 % (celdas, ha,
  incendios ≥100 ha). Y la resta IFS − reanálisis por modelo con su IC.
Escribe replay/replay_veredicto.json y replay/replay_veredicto.md.
Uso: python replay_veredicto.py   (usa lo que haya; se puede correr a medias)
"""
import glob, json, os
import numpy as np, pandas as pd
AQUI = os.path.dirname(os.path.abspath(__file__))
MODELOS = ("prod", "unico", "r10", "pareja", "donde", "cuando")
N_BOOT, SEED = 2000, 42

def ic_dif(a, b, rng):
    """IC95 bootstrap por días de mean(a-b)."""
    d = a - b; n = len(d)
    if n < 2: return (np.nan, np.nan)
    m = np.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))

def resumen(df):
    rng = np.random.default_rng(SEED)
    con = df[df["celdas_quemadas"] > 0].copy()
    out = {"n_dias": int(len(df)), "n_dias_con_fuego": int(len(con)),
           "celdas_quemadas": int(df["celdas_quemadas"].sum()), "area_ha": float(df["area_ha"].sum()),
           "modelos": {}}
    for k in MODELOS:
        a = con[f"auc_{k}"].values; p = con["auc_prod"].values
        ok = np.isfinite(a) & np.isfinite(p)
        r = {"auc_medio": float(np.nanmean(a)), "auc_mediana": float(np.nanmedian(a)),
             "dif_vs_prod": float((a[ok] - p[ok]).mean()), "ic95_dif": ic_dif(a[ok], p[ok], rng),
             "dias_gana_prod": float((a[ok] > p[ok]).mean()),
             "pctl_mediano": float(np.nanmedian(con[f"pctl_{k}"])),
             "pctl_ponderado_ha": float(np.nansum(con[f"pctl_ha_{k}"] * con["area_ha"]) / con["area_ha"].sum()),
             "top2_celdas": float(np.nansum(con[f"top2_celdas_{k}"] * con["celdas_quemadas"]) / con["celdas_quemadas"].sum()),
             "top2_ha": float(np.nansum(con[f"top2_ha_{k}"] * con["area_ha"]) / con["area_ha"].sum())}
        g = con[con["n_incendios_100ha"] > 0]
        r["top2_inc100"] = float(np.nansum(g[f"top2_inc100_{k}"] * g["n_incendios_100ha"]) / g["n_incendios_100ha"].sum()) if len(g) else np.nan
        r["ic95_dif"] = [round(x, 4) for x in r["ic95_dif"]]
        out["modelos"][k] = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}
    return out

def main():
    R = {}
    for csv in sorted(glob.glob(f"{AQUI}/replay/replay_*_*.csv")):
        nom = os.path.basename(csv)[7:-4]          # 2025_ifs
        df = pd.read_csv(csv)
        R[nom] = resumen(df); R[nom]["_df"] = df
    # resta ifs − reanálisis por modelo, mismos días
    rng = np.random.default_rng(SEED)
    for anio in ("2025", "2026"):
        if f"{anio}_ifs" in R and f"{anio}_reanalisis" in R:
            a = R[f"{anio}_ifs"]["_df"]; b = R[f"{anio}_reanalisis"]["_df"]
            m = a.merge(b, on="fecha", suffixes=("_i", "_r")); m = m[m["celdas_quemadas_i"] > 0]
            R[f"{anio}_ifs_menos_reanalisis"] = {"n_dias": int(len(m)), "modelos": {}}
            for k in MODELOS:
                x, y = m[f"auc_{k}_i"].values, m[f"auc_{k}_r"].values; ok = np.isfinite(x) & np.isfinite(y)
                R[f"{anio}_ifs_menos_reanalisis"]["modelos"][k] = {
                    "dif_media": round(float((x[ok] - y[ok]).mean()), 4),
                    "ic95": [round(v, 4) for v in ic_dif(x[ok], y[ok], rng)]}
    for v in R.values(): v.pop("_df", None)
    json.dump(R, open(f"{AQUI}/replay/replay_veredicto.json", "w"), indent=1, ensure_ascii=False)
    # markdown
    L = ["# Veredicto del replay (prerregistro 02/09/2026 18:10)\n"]
    for nom, v in R.items():
        if "ifs_menos" in nom:
            L.append(f"\n## {nom}: coste de la previsión (AUC con IFS − AUC con reanálisis), {v['n_dias']} días\n")
            L.append("| modelo | Δ media | IC95 |\n|---|---|---|")
            for k, r in v["modelos"].items(): L.append(f"| {k} | {r['dif_media']:+.3f} | [{r['ic95'][0]:+.3f}, {r['ic95'][1]:+.3f}] |")
            continue
        L.append(f"\n## {nom}: {v['n_dias']} días, {v['n_dias_con_fuego']} con fuego, {v['celdas_quemadas']:,} celdas, {v['area_ha']:,.0f} ha\n")
        L.append("| modelo | AUC medio | Δ vs prod | IC95 | gana a prod | pctl med | pctl pond. ha | top-2 % celdas | top-2 % ha | top-2 % inc ≥100 ha |\n|---|---|---|---|---|---|---|---|---|---|")
        for k, r in v["modelos"].items():
            L.append(f"| {k} | {r['auc_medio']:.3f} | {r['dif_vs_prod']:+.3f} | [{r['ic95_dif'][0]:+.3f}, {r['ic95_dif'][1]:+.3f}] | {r['dias_gana_prod']*100:.0f} % | {r['pctl_mediano']:.1f} | {r['pctl_ponderado_ha']:.1f} | {r['top2_celdas']:.1f} % | {r['top2_ha']:.1f} % | {r['top2_inc100']:.1f} % |")
    open(f"{AQUI}/replay/replay_veredicto.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
