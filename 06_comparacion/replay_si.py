#!/usr/bin/env python3
"""La pregunta del «si» (¿hoy hay aviso o no?), prerregistro 02/09/2026.

Los mapas por percentil siempre pintan un 2 % de «extremo», también en enero, y
el AUC por día solo existe en días con fuego. Aquí se fija un umbral absoluto de
probabilidad por modelo y se aplica igual todos los días:
  u_k = percentil 98 de las probabilidades de todas las celdas de España en los
        días de calibración (jun-ago 2025, condición IFS; versión «barata»: la
        correcta calibraría en los veranos 2015-2024 del cubo, pendiente).
Para cada día (todas las temporadas y condiciones disponibles, días de cero
fuego incluidos): fracción de celdas ≥ u_k; «día con aviso» = fracción ≥ 2 %
(el nivel medio del verano de calibración). Se cruza con «día con fuego»
(≥1 incendio EFFIS; y ≥1 de ≥100 ha) en tabla 2×2, por periodo:
calibración (jun-ago 2025, dentro de muestra), sep-oct 2025 y 2026 (fuera).
Se calcula también la Spearman entre la fracción sobre umbral y las ha quemadas
del día (con ceros), y lo mismo para el índice nacional de FWI (fwi_medio_pctl_clim del
json de cada día), que es la referencia sin modelo.
Modelos: prod, cuando, pareja (probabilidad comparable entre días). El único
queda fuera por construcción.
Escribe replay/replay_si.json y replay/replay_si.md.
"""
import glob, json, os
import numpy as np, pandas as pd, xarray as xr
AQUI = os.path.dirname(os.path.abspath(__file__))
MODELOS = ("prod", "cuando", "pareja")
CAL = ("2025-06-01", "2025-08-31")
PERIODOS = {"calibracion jun-ago 2025 (dentro)": ("2025-06-01", "2025-08-31"),
            "sep-oct 2025 (fuera)": ("2025-09-01", "2025-11-01"),
            "2026 (fuera)": ("2026-05-25", "2026-09-02"),
            "may 2025 (fuera)": ("2025-05-25", "2025-05-31")}

def main():
    import sys; sys.path.insert(0, f"{AQUI}/sandbox_replay"); os.chdir(f"{AQUI}/sandbox_replay")
    import config
    es_esp = xr.open_dataset(config.CUBO, decode_timedelta=False)["is_spain"].values.astype(bool).ravel()
    sub = np.flatnonzero(es_esp)[::7]           # submuestra fija de celdas para el percentil agrupado
    # --- 1. umbral por modelo, agrupado sobre los días de calibración --------
    vals = {k: [] for k in MODELOS}
    for f in sorted(glob.glob(f"{AQUI}/replay/2025/ifs/*.npz")):
        d = os.path.basename(f)[:-4]
        if not (CAL[0] <= d <= CAL[1]): continue
        m = np.load(f)
        for k in MODELOS: vals[k].append(m[f"prob_{k}"].astype(np.float32).ravel()[sub])
    if not vals["prod"]:
        raise SystemExit("aún no hay días de calibración en replay/2025/ifs")
    umbral = {k: float(np.nanpercentile(np.concatenate(vals[k]), 98)) for k in MODELOS}
    n_cal = len(vals["prod"])
    # --- 2. fracción sobre umbral, todos los días y condiciones --------------
    filas = []
    for anio in (2025, 2026):
        ver = pd.read_csv(f"{AQUI}/replay/verdad_{anio}.csv") if os.path.exists(f"{AQUI}/replay/verdad_{anio}.csv") else pd.DataFrame()
        for cond in ("ifs", "reanalisis"):
            for f in sorted(glob.glob(f"{AQUI}/replay/{anio}/{cond}/*.npz")):
                d = os.path.basename(f)[:-4]; m = np.load(f)
                v = ver[ver["fecha"] == d] if len(ver) else ver
                fila = dict(fecha=d, anio=anio, condicion=cond, n_inc=int(len(v)),
                            n_inc100=int((v["area_ha"] >= 100).sum()) if len(v) else 0,
                            area_ha=float(v["area_ha"].sum()) if len(v) else 0.0)
                js = f"{AQUI}/replay/{anio}/{cond}/{d}.json"
                if os.path.exists(js): fila["fwi_pctl_clim"] = json.load(open(js)).get("fwi_medio_pctl_clim", np.nan)
                for k in MODELOS:
                    p = m[f"prob_{k}"].astype(np.float32).ravel()[es_esp]
                    fila[f"frac_{k}"] = float(np.nanmean(p >= umbral[k]) * 100)
                filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(f"{AQUI}/replay/replay_si_dias.csv", index=False)
    # --- 3. tablas 2×2 y correlaciones por periodo (condición IFS) ----------
    from scipy.stats import spearmanr
    R = {"umbral_p98_calibracion": umbral, "n_dias_calibracion": n_cal, "periodos": {}}
    L = [f"# La pregunta del «si» — umbral absoluto (p98 pooled, jun-ago 2025 IFS, {n_cal} días)\n",
         "| modelo | umbral |\n|---|---|"] + [f"| {k} | {u:.4f} |" for k, u in umbral.items()]
    for nom, (a, b) in PERIODOS.items():
        s = df[(df["condicion"] == "ifs") & (df["fecha"] >= a) & (df["fecha"] <= b)]
        if len(s) < 3: continue
        P = {"n_dias": int(len(s)), "dias_con_fuego": int((s["n_inc"] > 0).sum()),
             "dias_con_inc100": int((s["n_inc100"] > 0).sum()), "modelos": {}}
        L.append(f"\n## {nom}: {len(s)} días, {P['dias_con_fuego']} con fuego, {P['dias_con_inc100']} con incendio ≥100 ha\n")
        L.append("| modelo | días con aviso | aviso∧fuego | aviso∧sin fuego | sin aviso∧fuego | sin aviso∧sin fuego | aviso∧inc≥100 / inc≥100 | Spearman frac~ha |\n|---|---|---|---|---|---|---|---|")
        fuego = s["n_inc"] > 0; g100 = s["n_inc100"] > 0
        for k in MODELOS + ("fwi",):
            av = (s[f"frac_{k}"] >= 2.0) if k != "fwi" else (s["fwi_pctl_clim"] >= 90)
            x = s[f"frac_{k}"] if k != "fwi" else s["fwi_pctl_clim"]
            rho = spearmanr(x, s["area_ha"], nan_policy="omit").correlation
            r = dict(dias_aviso=int(av.sum()), aviso_fuego=int((av & fuego).sum()), aviso_sinfuego=int((av & ~fuego).sum()),
                     sinaviso_fuego=int((~av & fuego).sum()), sinaviso_sinfuego=int((~av & ~fuego).sum()),
                     aviso_inc100=int((av & g100).sum()), spearman_frac_ha=round(float(rho), 3))
            P["modelos"][k] = r
            L.append(f"| {k}{' (pctl clim ≥90)' if k=='fwi' else ''} | {r['dias_aviso']} | {r['aviso_fuego']} | {r['aviso_sinfuego']} | {r['sinaviso_fuego']} | {r['sinaviso_sinfuego']} | {r['aviso_inc100']}/{P['dias_con_inc100']} | {r['spearman_frac_ha']:+.2f} |")
        R["periodos"][nom] = P
    json.dump(R, open(f"{AQUI}/replay/replay_si.json", "w"), indent=1, ensure_ascii=False)
    open(f"{AQUI}/replay/replay_si.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
