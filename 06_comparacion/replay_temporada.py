#!/usr/bin/env python3
"""Bucle de temporada sobre replay_dia.py: reanudable (salta los días con npz
en replay/<año>/<condicion>/), sin PNG, 8 hilos. Uso:
   python replay_temporada.py --anio 2025 --condicion ifs [--ini --fin]"""
import argparse, os, subprocess, sys, time
import pandas as pd
AQUI = os.path.dirname(os.path.abspath(__file__))
P = "/home/charredgem/miniconda3/envs/tfm_fuego/bin/python"
RANGO = {2025: ("2025-05-25", "2025-11-01"), 2026: ("2026-05-25", "2026-09-02")}
a = argparse.ArgumentParser(); a.add_argument("--anio", type=int, required=True)
a.add_argument("--condicion", choices=["ifs", "reanalisis"], required=True)
a.add_argument("--ini"); a.add_argument("--fin"); a = a.parse_args()
ini, fin = a.ini or RANGO[a.anio][0], a.fin or RANGO[a.anio][1]
if a.condicion == "reanalisis" and a.anio == 2026:
    fin = min(fin, "2026-08-27")            # el reanálisis del Release llega al 27-ago
env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
dest = f"{AQUI}/replay/{a.anio}/{a.condicion}"
t0 = time.time(); hechos = fallos = 0
for d in pd.date_range(ini, fin, freq="D"):
    f = str(d.date())
    if os.path.exists(f"{dest}/{f}.npz"):
        continue
    r = subprocess.run([P, f"{AQUI}/replay_dia.py", "--fecha", f, "--condicion", a.condicion, "--sin-png"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        fallos += 1
        open(f"{AQUI}/replay/fallos_{a.anio}_{a.condicion}.log", "a").write(f"=== {f}\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}\n")
        print(f"  {f}: FALLO (ver fallos_{a.anio}_{a.condicion}.log)", flush=True)
        continue
    hechos += 1
    res = [l for l in r.stdout.splitlines() if l.startswith("  unico") or l.startswith("  prod")]
    print(f"  {f}: ok · {(time.time()-t0)/max(hechos,1):.0f} s/día · {res[0].strip() if res else ''}", flush=True)
print(f"terminado {a.anio} {a.condicion}: {hechos} días nuevos, {fallos} fallos, {(time.time()-t0)/60:.0f} min", flush=True)
