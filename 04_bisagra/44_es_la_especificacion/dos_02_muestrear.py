#!/usr/bin/env python3
"""
Dos modelos, paso 2: la tabla maestra con los tres diseños de muestreo.

No toca producción. Lee EGIF y cubo; escribe en expansión.

Por qué otro muestreador

`muestrear_dataset.py` (iteración 1) empareja cada incendio con tres días
aleatorios de la misma celda. Se midió la consecuencia: el
98,8 % de las celdas tiene exactamente un 25 % de positivos, así que el modelo
no puede aprender el dónde. Este script produce, en una sola tabla y con un
campo `disenio`, todo lo que hace falta para comparar diseños con el mismo
pipeline de features:

  cuando   la muestra maestra v1 tal cual (78.065 filas, id_muestra v1
           conservado en `id_v1`). Se re-extraen sus features con el mismo
           código que los demás, para que la comparación sea limpia.
  donde    cada incendio EGIF (celda X, día D) + 3 celdas al azar de España
           el mismo día D. Uniforme sobre las 498.530 celdas: es literalmente
           la pregunta operativa («¿cuál de estas arde hoy?»). Se rechaza una
           celda si hay un incendio EGIF a <12,5 km en ±10 días (mismo buffer
           que el diseño original), para no etiquetar como ausencia lo que es
           un incendio en curso.
  eval_dia verano (1-jun → 30-sep) de 2019 (val), 2020 (test) y 2022 (año
           récord, fuera del EGIF: solo verdad-terreno EFFIS). Por día:
           1.000 celdas al azar + todos los positivos EGIF del día + hasta 40
           celdas con `is_fire` (EFFIS ≥5 ha) del día. Es el banco de pruebas
           del AUC dentro del día, y no entra en
           ningún entrenamiento.

El diseño mixto (mitad cuándo / mitad dónde) no necesita filas propias: se
monta en el entrenamiento submuestreando `cuando` y `donde`.

Etiquetas en `eval_dia`:
  label_egif   1 si hay ignición EGIF en esa celda ese día
  label_effis  1 si `is_fire`=1 en esa celda ese día (se rellena en el paso 3
               desde el cubo, junto con `is_near_fire`)
  primer_dia   1 si `is_fire`=1 y el día anterior no (aproxima la ignición;
               `is_fire` persiste un 51 % al día siguiente, `dos_00`)

Salida: EXPANSION/dataset/maestra_dos.parquet
"""

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

import config
import config_expansion as ce

SEED = 42
RATIO_NEG = 3
BUFFER_KM = 12.5
BUFFER_DIAS = 10
N_AZAR_DIA = 1000
N_EFFIS_DIA = 40
ANIOS_EVAL = (2019, 2020, 2022)
MESES_EVAL = (6, 9)
SALIDA = f"{ce.DATASET}/maestra_dos.parquet"
V1 = f"{config.FUENTE}/dataset/muestra_maestra_v1.parquet"
SPLITS = {"train": (2015, 2018), "val": (2019, 2019), "test": (2020, 2020)}


def split_de(anio):
    for k, (a, b) in SPLITS.items():
        if a <= anio <= b:
            return k
    return "fuera"


def main():
    rng = np.random.default_rng(SEED)
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    xs, ys = ds["x"].values, ds["y"].values
    esp = ds["is_spain"].values.astype(bool)
    ccaa = ds["AutonomousCommunities"].values
    iy_esp, ix_esp = np.where(esp)
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    # ---- 1. diseño cuando: la v1 tal cual ---------------------------------
    v1 = pd.read_parquet(V1)
    cuando = pd.DataFrame({
        "disenio": "cuando", "id_v1": v1["id_muestra"].values,
        "fecha": pd.to_datetime(v1["fecha"]).dt.normalize().values,
        "ix": v1["ix"].values, "iy": v1["iy"].values,
        "label": v1["label"].values.astype(np.int8),
        "id_egif": v1["id_egif"].values, "superficie": v1["superficie"].values})
    pos = cuando[cuando["label"] == 1].copy()
    print(f"cuando: {len(cuando):,} filas · positivos {len(pos):,}")

    # ---- 2. diseño donde: mismo día, otra celda -----------------------------
    # árbol de todos los incendios EGIF 2015-2020 (coincide con los positivos
    # v1 deduplicados; la deduplicación no cambia el buffer)
    fx, fy = xs[pos["ix"].values], ys[pos["iy"].values]
    fd = pos["fecha"].values.astype("datetime64[D]").astype(int)
    arbol = cKDTree(np.column_stack([fx, fy]))

    filas = []
    n_rech = 0
    for r in pos.itertuples():
        d = np.datetime64(r.fecha, "D").astype(int)
        got = 0
        while got < RATIO_NEG:
            k = rng.integers(len(iy_esp), size=RATIO_NEG * 2)
            for j in k:
                if got == RATIO_NEG:
                    break
                cy, cx = iy_esp[j], ix_esp[j]
                if cy == r.iy and cx == r.ix:
                    continue
                vec = arbol.query_ball_point([xs[cx], ys[cy]], r=BUFFER_KM * 1e3)
                if len(vec) and (np.abs(fd[vec] - d) <= BUFFER_DIAS).any():
                    n_rech += 1
                    continue
                filas.append((r.fecha, cx, cy))
                got += 1
    donde = pd.DataFrame(filas, columns=["fecha", "ix", "iy"])
    donde["disenio"] = "donde"
    donde["label"] = np.int8(0)
    donde = donde.drop_duplicates(["fecha", "ix", "iy"])
    # los positivos del diseño dónde son los mismos incendios
    pos_d = pos.copy()
    pos_d["disenio"] = "donde"
    pos_d = pos_d.drop(columns=["id_v1"])
    donde = pd.concat([pos_d, donde], ignore_index=True)
    print(f"donde: {len(donde):,} filas · rechazos por buffer {n_rech:,}")

    # ---- 3. eval_dia ----------------------------------------------------------
    tiempos = ds["time"].values.astype("datetime64[D]")
    t0 = tiempos[0].astype(int)
    dias = [d for a in ANIOS_EVAL
            for d in pd.date_range(f"{a}-{MESES_EVAL[0]:02d}-01",
                                   f"{a}-{MESES_EVAL[1]:02d}-30", freq="D")]
    dias = np.array(dias, dtype="datetime64[D]")
    ti = dias.astype(int) - t0
    # is_fire de esos días, leído por bloques (el cubo está chunked en tiempo
    # a 521 días: leer el rango de una vez por bloque es lo barato)
    print("  leyendo is_fire de los días de evaluación...", flush=True)
    fuego = []                     # (t_rel, iy, ix, primer_dia)
    sel_t = np.concatenate([ti, ti - 1])          # y el día anterior
    sel_t = np.unique(sel_t)
    pos_t = {t: i for i, t in enumerate(sel_t)}
    for by in range(0, esp.shape[0], 77):
        for bx in range(0, esp.shape[1], 99):
            sy, sx = slice(by, by + 77), slice(bx, bx + 99)
            if not esp[sy, sx].any():
                continue
            f = ds["is_fire"].isel(time=sel_t, y=sy, x=sx).values.astype(bool)
            f &= esp[sy, sx][None]
            for t in ti:
                a = f[pos_t[t]]
                if not a.any():
                    continue
                b = f[pos_t[t - 1]]
                yy, xx = np.where(a)
                for y_, x_ in zip(yy, xx):
                    fuego.append((t, y_ + by, x_ + bx, int(not b[y_, x_])))
    fuego = pd.DataFrame(fuego, columns=["t", "iy", "ix", "primer_dia"])
    print(f"  celdas-día con is_fire en eval: {len(fuego):,} · "
          f"primer día {fuego.primer_dia.sum():,}")

    egif_dia = {}
    for r in pos.itertuples():
        egif_dia.setdefault(np.datetime64(r.fecha, "D"), []).append(
            (r.ix, r.iy, r.id_egif, r.superficie))

    ev = []
    for d, t in zip(dias, ti):
        k = rng.choice(len(iy_esp), N_AZAR_DIA, replace=False)
        for j in k:
            ev.append((d, ix_esp[j], iy_esp[j], "azar", 0, -1, np.nan, 0, 0))
        for cx, cy, ide, sup in egif_dia.get(d, []):
            ev.append((d, cx, cy, "egif", 1, ide, sup, 0, 0))
        fz = fuego[fuego["t"] == t]
        if len(fz) > N_EFFIS_DIA:
            fz = fz.sample(N_EFFIS_DIA, random_state=int(t))
        for r in fz.itertuples():
            ev.append((d, r.ix, r.iy, "effis", 0, -1, np.nan, 1, r.primer_dia))
    ev = pd.DataFrame(ev, columns=["fecha", "ix", "iy", "origen", "label_egif",
                                   "id_egif", "superficie", "label_effis",
                                   "primer_dia"])
    ev["fecha"] = pd.to_datetime(ev["fecha"])
    # una celda-día solo una vez: si el azar cae en un positivo, manda el positivo
    ev = ev.sort_values("origen", key=lambda s: s.map({"egif": 0, "effis": 1,
                                                        "azar": 2}))
    ev = ev.drop_duplicates(["fecha", "ix", "iy"]).reset_index(drop=True)
    ev["disenio"] = "eval_dia"
    ev["label"] = ev["label_egif"].astype(np.int8)
    print(f"eval_dia: {len(ev):,} filas · "
          + " · ".join(f"{k} {v:,}" for k, v in ev.origen.value_counts().items()))

    # ---- 4. unir y completar --------------------------------------------------
    df = pd.concat([cuando, donde, ev], ignore_index=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["anio"] = df["fecha"].dt.year.astype(np.int16)
    df["mes"] = df["fecha"].dt.month.astype(np.int8)
    df["split"] = df["anio"].map(split_de)
    df.loc[df["disenio"] == "eval_dia", "split"] = "eval"
    df["x3035"] = xs[df["ix"].values]
    df["y3035"] = ys[df["iy"].values]
    df["lon_celda"], df["lat_celda"] = tr.transform(df["x3035"].values,
                                                    df["y3035"].values)
    df["ccaa"] = ccaa[df["iy"].values, df["ix"].values]
    df["bloque_100km"] = (np.floor(df["x3035"] / 1e5) * 1000
                          + np.floor(df["y3035"] / 1e5)).astype(int)
    df["id_muestra"] = np.arange(len(df))
    df.to_parquet(SALIDA, index=False)
    print(f"\nguardado {SALIDA}: {df.shape}")
    print(df.groupby(["disenio", "split", "label"]).size().unstack(fill_value=0)
          .to_string())


if __name__ == "__main__":
    main()
