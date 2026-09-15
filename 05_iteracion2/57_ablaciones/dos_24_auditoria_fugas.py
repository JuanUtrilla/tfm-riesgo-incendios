#!/usr/bin/env python3
"""
Dos modelos, paso 24: auditoría de fuga para todos los modelos.

Nace de los tres hallazgos del 31/08/2026 (ventana FIRMS invertida, ventana
7d/5d train-serve, y ndvi/lst del día D con la firma del incendio dentro).
Los tres se habrían cazado antes con una prueba automática, y este script
es esa prueba.

Cómo se distingue una fuga de una causa
---------------------------------------
En un diseño «dentro del día» la meteo del propio día D es legítima: el calor
y la sequedad causan el incendio. Lo que no es legítimo es una feature que
cambia porque el incendio ya está ardiendo. Estadísticamente las dos cosas
mueven la feature el día D, así que mirar solo el salto no distingue nada.

Lo que sí las separa es la escala espacial. La meteo es regional: el día que
sube la temperatura, sube en toda la comarca, en la celda que arde y en sus
vecinas. El daño del fuego es local: solo la celda que arde. Así que se mide

    índice = media[ Δ_local(D − D−1) − Δ_regional(D − D−1) ]

donde Δ_regional es el mismo salto promediado sobre celdas de control a
50-150 km ese mismo día. Una feature meteorológica honesta da un índice ~0
(sube igual dentro que fuera); una feature contaminada por el propio fuego da
un índice grande (solo se mueve donde arde). Se normaliza por la desviación
del control (z), y se acompaña de la persistencia en D+1: el daño persiste, el
calor del propio fuego revierte.

Después el índice se pondera por el `gain` de cada feature en cada modelo, y
sale el % de cada modelo que se apoya en señal concurrente.

Las features de ventana (`*_7d`, `*_30d`, historia, FIRMS) son limpias por
construcción (`slice(t-7, t)` y `< d`) y se marcan como tales sin medirlas.

No toca producción. Escribe salida/dos_24_auditoria_fugas.{json,csv}.
"""

import json

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

import config
import config_expansion as ce
from dos_05_modelos import FULL

# Criterio principal: ¿puede la fuente ver el fuego?
# La estadística sola no basta. ERA5-Land es un reanálisis: asimila
# observaciones atmosféricas, no incendios; su temperatura del día D no puede
# contener el fuego aunque se desplace localmente (los fuegos empiezan donde
# hace localmente más calor, y eso es causa, no fuga). Los productos
# satelitales de superficie (LST, NDVI, LAI, SWI) sí observan el suelo, y por
# tanto ven la quema el mismo día. Por eso una feature solo es concurrente si
# (a) su fuente observa la superficie y (b) el desplazamiento local es real.
FUENTE = {"FWI": "reanálisis", "t2m_max": "reanálisis", "t2m_min": "reanálisis",
          "RH_min": "reanálisis", "wind_speed_max": "reanálisis",
          "total_precipitation_mean": "reanálisis",
          "LST": "satélite", "NDVI": "satélite", "LAI": "satélite",
          "SWI_010": "satélite"}

# feature del modelo -> variable del cubo leída en el día D
DIA_D = {"fwi": "FWI", "t2m_max": "t2m_max", "t2m_min": "t2m_min",
         "rh_min": "RH_min", "viento_max": "wind_speed_max",
         "precip_dia": "total_precipitation_mean", "lst": "LST",
         "ndvi": "NDVI", "lai": "LAI", "swi010": "SWI_010"}
# derivadas del día D (heredan lo que le pase a su fuente)
DERIVA_D = {"fwi_pctl_local": "fwi", "fwi_anom_sigma": "fwi",
            "vpd_max": "t2m_max", "dias_sin_lluvia": "precip_dia"}
# limpias por construcción: la ventana excluye D, o miran a años anteriores
POR_CONSTRUCCION = {
    "precip_7d", "precip_15d", "precip_30d", "fwi_med_7d", "fwi_max_7d",
    "fwi_med_15d", "fwi_med_30d", "rh_min_med_7d", "t2m_max_med_7d",
    "viento_max_med_7d", "ndvi_med_30d", "n_fuegos_10km_mismomes_hist",
    "frp_max_50km_7d", "n_detec_50km_7d", "rayos_7d"}
ESTATICAS = {"elevacion", "pendiente", "rugosidad", "dist_carreteras",
             "dist_rios", "popdens", "clc_bosque", "clc_matorral",
             "clc_agricola", "clc_artificial", "clc_abierto",
             "clc_agric_hetero", "mes", "dia_anio", "es_festivo", "ccaa"}
# rayos_dia: el propio día D a propósito (tormenta seca antes de la ignición
# de tarde). Se mide igual, y si sale concurrente hay que discutirlo.

MODELOS = ["donde_dia_effis", "donde_dia_effis_r3", "donde_dia_effis_r10",
           "donde_dia_effis_r30", "donde_dia_effis_r60", "donde_dia_effis_r100",
           "donde_dia", "donde_effis", "donde_effis_c", "donde_effis_mix",
           "cuando", "cuando_46", "cuando_effis", "cuando_verano",
           "donde_cel", "donde_cel_hist", "mixto"]
VERANOS = [(2019, 6, 9), (2020, 6, 9), (2021, 6, 9), (2022, 6, 9)]
N_MUESTRA = 600
R_MIN_KM, R_MAX_KM = 50, 150
# Dos varas, porque miden cosas distintas y hacen falta las dos:
#   d  = tamaño de efecto por celda (media / desviación). Cuánto puede
#        explotarlo el modelo en una celda concreta.
#   t  = media / error estándar. Si el desplazamiento es real o es ruido.
# Una feature es concurrente si el desplazamiento es real (|t| ≥ 4) y tiene
# tamaño suficiente para que el árbol lo use (|d| ≥ 0,10). Con |t| ≥ 4 pero
# d minúsculo se marca marginal: real pero probablemente inocuo.
UMBRAL_T, UMBRAL_D = 4.0, 0.10
SEMILLA = 20260831


def indice_concurrencia(ds, es):
    """z de (Δ local − Δ regional) el día D, y persistencia en D+1."""
    t = ds["time"].values.astype("datetime64[D]")
    rng = np.random.default_rng(SEMILLA)
    ys, xs = np.where(es)
    acum = {v: {"d0": [], "d1": []} for v in DIA_D.values()}
    for anio, m0, m1 in VERANOS:
        i0 = int(np.searchsorted(t, np.datetime64(f"{anio}-{m0:02d}-01")))
        i1 = int(np.searchsorted(t, np.datetime64(f"{anio}-{m1:02d}-01")))
        sl = slice(i0 - 1, i1 + 2)
        fuego = ds["is_fire"].isel(time=sl).values.astype(bool)
        prim = fuego[1:-2] & ~fuego[:-3]          # primer día del incendio
        idx = np.argwhere(prim)
        if not len(idx):
            continue
        idx = idx[rng.choice(len(idx), min(N_MUESTRA, len(idx)), replace=False)]
        campos = {v: ds[v].isel(time=sl).values for v in DIA_D.values()}
        for k, y, x in idx:
            # control: mismo día, anillo de 50-150 km, sin fuego en D ni D+1
            for _ in range(60):
                j = rng.integers(len(ys))
                cy, cx = ys[j], xs[j]
                dkm = np.hypot(cy - y, cx - x)     # celda = 1 km
                if not (R_MIN_KM <= dkm <= R_MAX_KM):
                    continue
                if fuego[k + 1, cy, cx] or fuego[k + 2, cy, cx]:
                    continue
                break
            else:
                continue
            for v, A in campos.items():
                dl0 = A[k + 1, y, x] - A[k, y, x]
                dr0 = A[k + 1, cy, cx] - A[k, cy, cx]
                dl1 = A[k + 2, y, x] - A[k, y, x]
                dr1 = A[k + 2, cy, cx] - A[k, cy, cx]
                if np.isfinite(dl0) and np.isfinite(dr0):
                    acum[v]["d0"].append(dl0 - dr0)
                if np.isfinite(dl1) and np.isfinite(dr1):
                    acum[v]["d1"].append(dl1 - dr1)
    out = {}
    for v, d in acum.items():
        a0, a1 = np.array(d["d0"]), np.array(d["d1"])
        s0 = np.std(a0) if len(a0) else np.nan
        out[v] = dict(n=int(len(a0)),
                      dif_D=float(np.mean(a0)) if len(a0) else np.nan,
                      d_D=float(np.mean(a0) / s0) if s0 else np.nan,
                      t_D=float(np.mean(a0) / (s0 / np.sqrt(len(a0)))) if s0 else np.nan,
                      dif_D1=float(np.mean(a1)) if len(a1) else np.nan,
                      d_D1=float(np.mean(a1) / s0) if s0 else np.nan)
    return out


def main():
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    es = ds["is_spain"].values.astype(bool)
    print("midiendo concurrencia local vs regional...", flush=True)
    conc = indice_concurrencia(ds, es)
    ds.close()

    # veredicto por feature
    veredicto = {}
    for f in FULL:
        if f in ESTATICAS:
            veredicto[f] = dict(clase="estática", z_D=None, motivo="no depende de D")
        elif f in POR_CONSTRUCCION:
            veredicto[f] = dict(clase="limpia por construcción", z_D=None,
                                motivo="ventana excluye D / años anteriores")
        else:
            v = DIA_D.get(f) or DIA_D.get(DERIVA_D.get(f, ""), None)
            if f in DERIVA_D:
                v = DIA_D[DERIVA_D[f]]
            if v is None:
                veredicto[f] = dict(clase="sin medir", z_D=None, motivo="")
                continue
            c = conc[v]
            real = abs(c["t_D"]) >= UMBRAL_T
            grande = abs(c["d_D"]) >= UMBRAL_D
            persiste = abs(c["d_D1"]) >= abs(c["d_D"]) * 0.5
            ve = FUENTE[v] == "satélite"
            if not real:
                clase, motivo = "limpia", "sin desplazamiento local"
            elif not ve:
                clase = "efecto local real"
                motivo = "el reanálisis no puede ver el fuego → es causa, no fuga"
            elif grande:
                clase = "CONCURRENTE"
                motivo = ("la fuente observa la superficie y solo se mueve donde arde"
                          + (", persiste en D+1 → daño" if persiste
                             else ", revierte en D+1 → calor del propio fuego"))
            else:
                clase, motivo = "marginal", "satélite, desplazamiento real pero pequeño"
            veredicto[f] = dict(
                clase=clase, d_D=c["d_D"], t_D=c["t_D"], d_D1=c["d_D1"],
                dif_D=c["dif_D"], n=c["n"], cubo=v, fuente=FUENTE[v],
                derivada_de=DERIVA_D.get(f), motivo=motivo)

    print(f"\n{'feature':28s} {'fuente':12s} {'clase':20s} {'Δloc-reg':>10s} "
          f"{'d':>6s} {'t':>7s} {'d(D+1)':>7s}")
    for f in FULL:
        v = veredicto[f]
        if v.get("d_D") is None:
            print(f"{f:28s} {'—':12s} {v['clase']:20s}")
            continue
        print(f"{f:28s} {v['fuente']:12s} {v['clase']:20s} {v['dif_D']:+10.4f} "
              f"{v['d_D']:+6.2f} {v['t_D']:+7.1f} {v['d_D1']:+7.2f}")

    # ponderar por gain de cada modelo
    print(f"\n{'modelo':24s} {'n feats':>8s} {'% gain CONCURRENTE':>20s}  detalle")
    filas = []
    for nom in MODELOS:
        try:
            m = xgb.XGBClassifier(); m.load_model(f"{ce.MODELOS}/{nom}.ubj")
        except Exception:
            continue
        g = m.get_booster().get_score(importance_type="gain")
        nf = m.get_booster().num_features()
        feats = FULL if nf == len(FULL) else FULL[:nf]
        tot = sum(g.values())
        imp = {}
        for k, val in g.items():
            i = int(k[1:]) if k.startswith("f") else FULL.index(k)
            if i < len(feats):
                imp[feats[i]] = val / tot * 100
        malas = {k: v for k, v in imp.items()
                 if veredicto.get(k, {}).get("clase") in ("CONCURRENTE", "marginal")}
        filas.append(dict(modelo=nom, n_features=nf,
                          pct_gain_concurrente=sum(malas.values()),
                          detalle="; ".join(f"{k} {v:.1f}%" for k, v in
                                            sorted(malas.items(), key=lambda x: -x[1]))))
        print(f"{nom:24s} {nf:8d} {sum(malas.values()):19.1f}%  "
              f"{filas[-1]['detalle'][:60]}")

    df = pd.DataFrame(filas)
    df.to_csv(config.salida("dos_24_auditoria_fugas.csv"), index=False)
    json.dump({"umbral_t": UMBRAL_T, "umbral_d": UMBRAL_D,
               "anillo_control_km": [R_MIN_KM, R_MAX_KM],
               "features": veredicto, "modelos": filas},
              open(config.salida("dos_24_auditoria_fugas.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
