#!/usr/bin/env python3
"""
Dos modelos, paso 18: ¿cuántos negativos del mismo día hace falta ver?

No toca producción. Lee el cubo y el disco externo; escribe
<externo>/dataset/{maestra,dataset}_ratio.parquet, los modelos
donde_dia_effis_r<k>.ubj y salida/dos_18_ratio.json.

Por qué
-------
El salto del trabajo (0,744 → 0,828 de AUC dentro del día) vino de cambiar
de dónde salen los negativos: del mismo día en otra celda, en vez de la misma
celda en otro día. Lo que nunca se ha ablacionado es cuántos. `RATIO_NEG = 3`
(dos_02) y `RATIO = 4` (dos_12) son constantes heredadas del muestreador
original y nadie las ha movido.

Importa porque el modelo se evalúa por cómo ordena las ~498.530 celdas de un
día y en entrenamiento solo ve 3 negativos por positivo: en un día con 9
igniciones son 27 celdas de medio millón. El AUC no depende de la prevalencia
(por eso el desbalanceo no invalida nada), pero sí depende de cuántos
negativos difíciles entran en la comparación. A 1:100 el día de entrenamiento
se parece al día del banco de evaluación (1.000 celdas al azar), que es la
pregunta operativa.

Diseño: una sola extracción, escalera anidada
---------------------------------------------
Los mismos positivos de `dos_12` (celdas EFFIS de primer día 2015-2024, tope
de 30 por día y bloque de 100 km, SEED 7: el código es idéntico, así que el
conjunto es el mismo). Se sortean KMAX=110 negativos por positivo (mismo día,
celda al azar de España) y se extraen las features una vez; cada peldaño de
la escalera es un prefijo de esa lista (`k_neg` 0..k-1). Así:

  · 1:3, 1:10, 1:30, 1:100 comparten positivos y negativos (los de k menor
    son un subconjunto de los de k mayor): la diferencia es solo cuántos, no
    cuáles, y el ruido de muestreo no contamina la comparación;
  · se paga una sola extracción de features (2,1 M filas ≈ 8 min con los 8
    procesos de `dos_03`), no cuatro.

`is_near_fire` (fuego EFFIS a <12,5 km en ±10 días) se sigue filtrando en
`dos_03`, y por eso se sortean 110 y no 100: a ratios altos el filtro muerde
más filas y hay que tener colchón para que todos los positivos lleguen a 100.

Árboles fijos: 441 para todos, que es donde paró `donde_dia_effis` (1:3) con
early stopping en `dos_13`. Con más negativos la val cambia de prevalencia y
el early stopping pararía en otro sitio: si se dejara libre, la comparación
sería del ratio y del número de árboles a la vez. Se reporta también la val
aucpr, que no es comparable entre peldaños (prevalencias distintas) y solo
sirve para ver que ninguno diverge.

Banco: el `eval_dia` de `dataset_effis.parquet` (verano 2023 y 2024, 1.000
celdas al azar por día + celdas EFFIS de primer día), sin tocar. Métrica: AUC
dentro del día, IC95 por bootstrap de días, pareado contra el peldaño 1:3.
La prueba externa es la temporada 2026 (`dos_09`, que recoge solo los
modelos `donde_dia_effis_r*`).

Uso:
    python dos_18_ratio.py muestrear      # maestra_ratio.parquet
    python dos_03_features.py ratio       # dataset_ratio.parquet
    python dos_18_ratio.py modelos        # entrena la escalera y evalúa
"""

import json
import sys

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from pyproj import Transformer

import config
import config_expansion as ce
from dos_05_modelos import CLIMA, FULL, GEO, HIST, PARAMS, SEED, metricas_dia
from dos_12_muestrear_effis import (ANIOS, SEED as SEED12, SPLITS,
                                    TOPE_POS_DIA_BLOQUE, split_de)

KMAX = 110                       # negativos sorteados por positivo
ESCALERA = (3, 10, 30, 60, 100)  # peldaños medidos (60 es el último
                                 # completo: el filtro is_near_fire se lleva
                                 # el 8,7 % y el positivo con menos negativos
                                 # vivos se queda en 62. El peldaño 100 es
                                 # «hasta 100» y su reparto por positivo no es
                                 # uniforme (los positivos de zona con mucho
                                 # fuego pierden más negativos), así que la
                                 # escalera limpia es 3→60 y 100 es un extra.
N_ARBOLES = 441                  # donde_dia_effis (1:3) en dos_13
N_BOOT = 2000
MAESTRA = f"{ce.DATASET}/maestra_ratio.parquet"
DATASET = f"{ce.DATASET}/dataset_ratio.parquet"


# ------------------------------------------------------------ muestrear ----
def positivos(ds, xs, ys, esp):
    """Idéntico a dos_12: celdas EFFIS de primer día con tope por bloque."""
    t = ds["time"].values.astype("datetime64[D]")
    anio = t.astype("datetime64[Y]").astype(int) + 1970
    k = np.where((anio >= ANIOS[0]) & (anio <= ANIOS[1]))[0]
    k = np.r_[k[0] - 1, k]
    print("leyendo is_fire 2015-2024 por bloques...", flush=True)
    fuego = []
    for by in range(0, esp.shape[0], 77):
        for bx in range(0, esp.shape[1], 99):
            sy, sx = slice(by, by + 77), slice(bx, bx + 99)
            if not esp[sy, sx].any():
                continue
            f = ds["is_fire"].isel(time=k, y=sy, x=sx).values.astype(bool)
            f &= esp[sy, sx][None]
            primer = f[1:] & ~f[:-1]
            tt, yy, xx = np.where(primer)
            fuego.append(pd.DataFrame({"t": k[1:][tt], "iy": yy + by, "ix": xx + bx}))
    fuego = pd.concat(fuego, ignore_index=True)
    fuego["fecha"] = pd.to_datetime(t[fuego["t"].values])
    fuego["x3035"], fuego["y3035"] = xs[fuego["ix"]], ys[fuego["iy"]]
    fuego["bloque_100km"] = (np.floor(fuego["x3035"] / 1e5) * 1000
                             + np.floor(fuego["y3035"] / 1e5)).astype(int)
    pos = (fuego.sample(frac=1, random_state=SEED12)
                .groupby(["fecha", "bloque_100km"]).head(TOPE_POS_DIA_BLOQUE)
                .reset_index(drop=True))
    print(f"  celdas de primer día: {len(fuego):,} · positivos tras el tope: {len(pos):,}")
    return pos[["fecha", "ix", "iy"]]


def muestrear():
    rng = np.random.default_rng(SEED12)
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    xs, ys = ds["x"].values, ds["y"].values
    esp = ds["is_spain"].values.astype(bool)
    ccaa = ds["AutonomousCommunities"].values
    iy_esp, ix_esp = np.where(esp)
    pos = positivos(ds, xs, ys, esp)
    ds.close()
    n = len(pos)

    j = rng.integers(len(iy_esp), size=(n, KMAX))
    neg = pd.DataFrame({
        "fecha": np.repeat(pos["fecha"].values, KMAX),
        "ix": ix_esp[j].ravel(), "iy": iy_esp[j].ravel(),
        "id_pos": np.repeat(np.arange(n), KMAX),
        "k_neg": np.tile(np.arange(KMAX), n).astype(np.int16),
        "label": np.int8(0)})
    p = pos.assign(id_pos=np.arange(n), k_neg=np.int16(-1), label=np.int8(1))
    df = pd.concat([p, neg], ignore_index=True)          # positivos primero
    antes = len(df)
    df = df.drop_duplicates(["fecha", "ix", "iy"]).reset_index(drop=True)
    print(f"  sorteados {antes:,} · tras quitar repetidos {len(df):,} "
          f"({(antes-len(df))/antes*100:.2f} % colisiones)")

    df["disenio"] = "donde"
    df["anio"] = df["fecha"].dt.year.astype(np.int16)
    df["mes"] = df["fecha"].dt.month.astype(np.int8)
    df["split"] = df["anio"].map(split_de)
    df["x3035"], df["y3035"] = xs[df["ix"].values], ys[df["iy"].values]
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    df["lon_celda"], df["lat_celda"] = tr.transform(df["x3035"].values, df["y3035"].values)
    df["ccaa"] = ccaa[df["iy"].values, df["ix"].values]
    df["bloque_100km"] = (np.floor(df["x3035"] / 1e5) * 1000
                          + np.floor(df["y3035"] / 1e5)).astype(int)
    df["label_egif"] = np.int8(0)
    df["primer_dia"] = df["label"]
    df["origen"] = "muestra"
    df["id_muestra"] = np.arange(len(df))
    df.to_parquet(MAESTRA, index=False)
    print(f"\nguardado {MAESTRA}: {df.shape}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())


# --------------------------------------------------------------- modelos ----
def peldano(d, k):
    """Prefijo de la escalera: los k primeros negativos vivos de cada positivo."""
    neg = d[d["label"] == 0].sort_values(["id_pos", "k_neg"])
    rk = neg.groupby("id_pos").cumcount()
    return pd.concat([d[d["label"] == 1], neg[rk.values < k]])


def modelos():
    cols_cel = ["iy", "ix"] + CLIMA + GEO + HIST + [
        "elev_std", "dist_ferrocarril", "is_natura2000", "clc_arable",
        "clc_urbano", "clc_perm", "egif_1518_10km"] + [f"aspect_{a}" for a in range(1, 9)]
    cel = pd.read_parquet(f"{ce.DATASET}/celdas.parquet")[cols_cel]
    df = pd.read_parquet(DATASET).merge(cel, on=["iy", "ix"], how="left", validate="m:1")
    ev = pd.read_parquet(f"{ce.DATASET}/dataset_effis.parquet")
    ev = ev[ev["disenio"] == "eval_dia"].merge(cel, on=["iy", "ix"], how="left",
                                               validate="m:1").reset_index(drop=True)
    print(f"escalera {df.shape} · banco eval {ev.shape} · "
          f"positivos eval {int((ev.origen == 'effis').sum())}")

    scores, info = {}, {}
    prod = xgb.Booster(); prod.load_model(f"{config.MODELOS}/xgb_v2_prototipo.ubj")
    scores["prod"] = prod.predict(xgb.DMatrix(ev[FULL].values.astype(np.float32),
                                              feature_names=FULL))
    m = xgb.XGBClassifier(); m.load_model(f"{ce.MODELOS}/donde_dia_effis.ubj")
    scores["donde_dia_effis (dos_13, 1:3)"] = m.predict_proba(ev[FULL].values)[:, 1]

    par = dict(PARAMS); par.pop("early_stopping_rounds")
    par["n_estimators"] = N_ARBOLES
    for k in ESCALERA:
        tr = peldano(df[df["split"] == "train"], k)
        va = peldano(df[df["split"] == "val"], k)
        npos = int((tr["label"] == 1).sum())
        nom = f"donde_dia_effis_r{k}"
        mk = xgb.XGBClassifier(**par)
        mk.fit(tr[FULL].values, tr["label"].values,
               eval_set=[(va[FULL].values, va["label"].values)], verbose=False)
        mk.save_model(f"{ce.MODELOS}/{nom}.ubj")
        neg_dia = tr.groupby("fecha").size().sub(tr[tr.label == 1].groupby("fecha").size(),
                                                 fill_value=0).median()
        info[nom] = {"filas_train": int(len(tr)), "positivos": npos,
                     "neg_por_pos": round((len(tr) - npos) / npos, 2),
                     "neg_por_dia_mediana": float(neg_dia),
                     "val_aucpr": float(list(mk.evals_result()["validation_0"].values())[0][-1])}
        print(f"  {nom}: {len(tr):,} filas · {info[nom]['neg_por_pos']} neg/pos · "
              f"mediana {neg_dia:.0f} neg/día · val aucpr {info[nom]['val_aucpr']:.4f}",
              flush=True)
        scores[nom] = mk.predict_proba(ev[FULL].values)[:, 1]

    ref = f"donde_dia_effis_r{ESCALERA[0]}"
    rng = np.random.default_rng(SEED)
    R = {"config": {"kmax": KMAX, "escalera": list(ESCALERA), "n_arboles": N_ARBOLES,
                    "referencia": ref}, "modelos": info}
    for anio in (2024, 2023):
        e = ev[ev["anio"] == anio]
        por = {n: metricas_dia(e, s[e.index.values], "effis").set_index("fecha")
               for n, s in scores.items()}
        R[anio] = {}
        for n, t in por.items():
            b = por[ref]
            com = t.index.intersection(b.index)
            d = (t.loc[com, "auc"] - b.loc[com, "auc"]).values
            bb = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
            R[anio][n] = {"n_dias": int(len(t)), "n_pos": int(t.n_pos.sum()),
                          "auc_medio": float(t.auc.mean()),
                          "lift_decil": float(t.lift.mean()),
                          "dif_vs_ref": float(d.mean()),
                          "ic95": [float(np.percentile(bb, 2.5)),
                                   float(np.percentile(bb, 97.5))],
                          "dias_gana": float((d > 0).mean())}
        print(f"\n== verano {anio} ({'TEST' if anio == 2024 else 'val'}) · "
              f"AUC dentro del día · Δ contra {ref} ==")
        for n, r in sorted(R[anio].items(), key=lambda kv: -kv[1]["auc_medio"]):
            print(f"  {n:32s} AUC {r['auc_medio']:.3f} · lift {r['lift_decil']:.2f} · "
                  f"Δref {r['dif_vs_ref']:+.3f} [{r['ic95'][0]:+.3f}, {r['ic95'][1]:+.3f}]"
                  f" · gana {r['dias_gana']*100:.0f}%")
    json.dump(R, open(config.salida("dos_18_ratio.json"), "w"), indent=1)
    print(f"\nguardado {config.salida('dos_18_ratio.json')}")


if __name__ == "__main__":
    paso = sys.argv[1] if len(sys.argv) > 1 else "modelos"
    {"muestrear": muestrear, "modelos": modelos}[paso]()
