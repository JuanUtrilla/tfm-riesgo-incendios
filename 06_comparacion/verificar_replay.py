#!/usr/bin/env python3
"""Verificación de punta a punta del replay (PENDIENTE.md §1, 01/09/2026).

Reproduce el mapa del 21-ago-2026 —día con pasada IFS real en local
(salida/ifs_malla_2026-08-21.parquet), mapa servido (mapas_diarios/
dos_riesgo_2026-08-21.npz) y dentro del rango del archivo histórico— y lo
compara con el servido. Si no se parecen, el archivo de entradas no vale
para evaluar modelos y hay que saberlo antes de bajar más temporadas.

Tres diferencias inevitables respecto a la corrida real, que se parchean:
  1. El reanálisis de hoy llega más allá del 14-ago → se trunca al 13-ago
     (la pasada IFS del 21 empieza el 14: ese día el reanálisis acababa el 13).
  2. `firms_nrt` pide siempre los últimos 5 días hasta HOY → se sustituye por
     `comparar_rankings.firms_dia` con ventana 5 ([D−5, D−1] de la fecha
     pasada, cacheado en _firms5/), que devuelve las mismas dos rejillas.
  3. Las salidas pisarían las servidas → se respaldan antes y se restauran
     al final pase lo que pase; las del replay quedan en archivo_ifs/replay_21ago/.

NO toca ningún .py de los repos (los md5 de PROCEDENCIA.md siguen valiendo):
todo es monkeypatch desde fuera. No escribe en ningún veredicto.
"""
import hashlib
import json
import os
import shutil
import sys

REPO = "/home/charredgem/Desktop/Master/TFM_fuego_malla"
AQUI = os.path.dirname(os.path.abspath(__file__))
DEST = f"{AQUI}/replay_21ago"
F = "2026-08-21"
CORTE_RE = "2026-08-13"

os.environ["TFM_FIRMS_DIAS"] = "5"     # la ventana con la que se sirvió agosto
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
import xarray as xr

import config

SERVIDOS = [config.salida(f"mapas_diarios/dos_riesgo_{F}.npz"),
            config.salida(f"mapas_diarios/riesgo_hoy_{F}.npz"),
            config.salida(f"dos_riesgo_{F}.json"),
            config.salida(f"dos_riesgo_{F}.png")]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(DEST, exist_ok=True)
    respaldo = {}
    for p in SERVIDOS:
        if os.path.exists(p):
            b = f"{DEST}/RESPALDO_{os.path.basename(p)}"
            shutil.copy2(p, b)
            respaldo[p] = (b, md5(p))
    print(f"respaldados {len(respaldo)} ficheros servidos", flush=True)

    # reanálisis truncado a lo que había la mañana del 21-ago
    ds = xr.open_dataset(config.entrada("era5land_diario.nc"),
                         decode_timedelta=False)
    fechas = pd.to_datetime(ds["fecha"].values)
    dim = ds["fecha"].dims[0]
    keep = np.where(fechas <= pd.Timestamp(CORTE_RE))[0]
    trunc = f"{DEST}/era5land_trunc_{CORTE_RE}.nc"
    if not os.path.exists(trunc):
        ds.isel({dim: keep}).to_netcdf(trunc)
    print(f"reanálisis truncado: {fechas[keep][-1].date()} ({len(keep)} días) "
          f"→ {trunc}", flush=True)
    ds.close()

    import riesgo_hoy as rh
    rh.REANALISIS = trunc

    import comparar_rankings as cr
    assert cr.VENTANA_FIRMS == 5, cr.VENTANA_FIRMS
    dia = pd.Timestamp(F)
    rh.firms_nrt = lambda dsx: cr.firms_dia(dia, dsx)

    import dos_riesgo_hoy as dr

    class A:
        pasada = F
        dias = [0]

    try:
        dr.main(A())
        # --- comparación servido vs replay --------------------------------
        srv = np.load(f"{DEST}/RESPALDO_dos_riesgo_{F}.npz")
        rep = np.load(config.salida(f"mapas_diarios/dos_riesgo_{F}.npz"))
        from scipy.stats import pearsonr, spearmanr
        print("\n=== servido vs replay ===", flush=True)
        res = {}
        for k in sorted(set(srv.files) & set(rep.files)):
            a = srv[k].astype(float).ravel()
            b = rep[k].astype(float).ravel()
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            pe = float(pearsonr(a, b).statistic)
            sp = float(spearmanr(a, b).correlation)
            dmax = float(np.abs(a - b).max())
            # solape del top-2 % (la punta operativa del ranking)
            n = max(1, int(len(a) * 0.02))
            ta = set(np.argsort(a)[-n:])
            tb = set(np.argsort(b)[-n:])
            top = len(ta & tb) / n
            res[k] = dict(pearson=pe, spearman=sp, max_abs=dmax, top2=top)
            print(f"  {k:14s} pearson {pe:.4f} · spearman {sp:.4f} · "
                  f"max|Δ| {dmax:.4f} · top2% solape {top:.1%}", flush=True)
        json.dump(res, open(f"{DEST}/comparacion.json", "w"), indent=1)
        # las salidas del replay se apartan a DEST
        for p in SERVIDOS:
            if os.path.exists(p) and p in respaldo:
                shutil.move(p, f"{DEST}/REPLAY_{os.path.basename(p)}")
    finally:
        for p, (b, h) in respaldo.items():
            shutil.copy2(b, p)
            assert md5(p) == h, f"restauración corrupta: {p}"
        print(f"\nrestaurados {len(respaldo)} servidos (md5 verificados)",
              flush=True)


if __name__ == "__main__":
    main()
