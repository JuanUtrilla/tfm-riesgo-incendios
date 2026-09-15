#!/usr/bin/env python3
"""
Dos modelos, paso 26: calibrar el «si» (aviso de día) con 2015-2024.

No toca producción. Lee salida/dos_25_hist.npz y el cubo; escribe
salida/dos_26_*.{csv,json,md,png}.

Nota (14/09/2026): el semáforo nacional que salía de esta calibración se
evaluó y se descartó; el script se conserva como registro del experimento y
de sus números.

Qué pregunta contesta
---------------------
`replay_si.py` fijó el umbral con jun-ago 2025: 92 días, 86 con fuego. Con esa
prevalencia el umbral no se estima, se extrapola, y la tabla 2×2 queda
degenerada. Aquí hay 3.653 días con la variación real de la tasa base, días
de cero incluidos.

Tres cosas que la literatura obliga a cambiar respecto al plan original:

1. Dos regímenes en vez de uno. En 2015-2024 el 33,9 % de los días de nov-mar
   tienen fuego, y el 85,6 % de las celdas quemadas de dic-abr caen en el
   noroeste (14,4 % del territorio): es la temporada de quemas pastorales de
   Galicia-Asturias-León, con conductor de ignición y no de sequía. La UE
   calibra el FWI con escalas distintas para verano e invierno por esto mismo.
   Un umbral único todo el año queda ciego medio año.

2. Calibración beta en vez de isotónica o analítica. La corrección analítica
   del submuestreo (Elkan/King-Zeng/Dal Pozzolo, p = βp_s/(βp_s − p_s + 1))
   no vale para árboles: arXiv 2412.16209 muestra que la prevalencia estimada
   depende del número de predictores y del ratio, y que los árboles pueden
   estar sesgados hacia la minoritaria. La isotónica sobreajusta con pocos
   positivos y da una escalera, lo peor si lo que se extrae es un corte. Beta
   (Kull et al. 2017) es de tres parámetros y se aprende del dato.

3. El Brier solo no basta. Bajo desbalanceo el Brier lo domina la tasa base.
   Se reporta BSS contra la climatología del día del año, diagrama de
   fiabilidad con bines de igual población, y SEDI (Ferro & Stephenson 2011),
   que es independiente de la tasa base y no degenera con la rareza.

Partición (no se toca después)
------------------------------
  calibración 2015-2021 · validación 2022-2023 · test 2024 intacto
El umbral se elige en calibración, se mira en validación y se reporta en test.

Objetivo del día
----------------
«¿Arde algo?» apenas discrimina: en jun-sep el 56,5 % de los días tienen
fuego y en jul-ago el 63-76 %. El objetivo por defecto es el día grande: ≥5
celdas EFFIS de primer día en España (≥500 ha), que en 2015-2024 pasa el
10,6 % de los días. Se puede cambiar con --umbral-dia.

Uso:
    python dos_26_calibra_si.py                # todo
    python dos_26_calibra_si.py --umbral-dia 1 # objetivo «arde algo»
"""

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr

import config
import config_expansion as ce
from dos_25_barrido_historico import (BINS, CHUNK_X, CHUNK_Y, FIN, INI, MODELOS,
                                      NBINS, NO_PROV)

SAL = config.salida("dos_26")
CALIB = (2015, 2021)
VAL = (2022, 2023)
TEST = (2024, 2024)
REGIMEN = {**{m: "invierno-primavera" for m in (12, 1, 2, 3, 4)},
           **{m: "verano" for m in (6, 7, 8, 9)},
           5: "transicion", 10: "transicion", 11: "transicion"}
N_BOOT = 2000
SEED = 42
BLOQUE = 14      # días por bloque en el bootstrap de bloques móviles
# avisos mínimos en calibración para que SEDI sea estimable (ver el bucle de
# elección de umbral). 20 sobre los 644-2.557 días de cada régimen es un 1-3 %.
MIN_AVISOS = 20


# --------------------------------------------------------------------------
def verdad_diaria(dias):
    """Celdas EFFIS de primer día por jornada, nacional y noroeste.

    Pasada ligera sobre `is_fire` bloque a bloque: primer día = is_fire en t y
    no en t−1, que es lo que puntúa `quemadas()` al comparar con EFFIS.
    """
    import os
    f = f"{SAL}_verdad.csv"
    if os.path.exists(f):
        d = pd.read_csv(f, parse_dates=["fecha"])
        return d
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    tiempos = ds["time"].values.astype("datetime64[D]")
    t0 = tiempos[0].astype(int)
    t_of = dias.values.astype("datetime64[D]").astype(int) - t0
    cel = pd.read_parquet(f"{ce.DATASET}/celdas.parquet", columns=["iy", "ix", "lat", "lon"])
    mun = pd.DataFrame(json.load(open(f"{config.ESTADO}/municipios.json")))
    mun["lat"] = pd.to_numeric(mun.latitud_dec, errors="coerce")
    mun["lon"] = pd.to_numeric(mun.longitud_dec, errors="coerce")
    mun = mun.dropna(subset=["lat", "lon"])
    from scipy.spatial import cKDTree
    _, j = cKDTree(np.c_[mun.lat, mun.lon]).query(np.c_[cel.lat, cel.lon])
    es_no = mun.id_old.str[:2].values[j]
    es_no = pd.Series(es_no).isin(NO_PROV).values
    tot = np.zeros(len(dias), int)
    tot_no = np.zeros(len(dias), int)
    ard = np.zeros(len(dias), int)
    cel["_by"], cel["_bx"] = cel.iy // CHUNK_Y, cel.ix // CHUNK_X
    cel["_no"] = es_no
    for (by, bx), sub in cel.groupby(["_by", "_bx"]):
        sy = slice(by * CHUNK_Y, min((by + 1) * CHUNK_Y, ds.sizes["y"]))
        sx = slice(bx * CHUNK_X, min((bx + 1) * CHUNK_X, ds.sizes["x"]))
        v = ds["is_fire"].isel(y=sy, x=sx).values.reshape(len(tiempos), -1)
        loc = ((sub.iy.values - sy.start) * (sx.stop - sx.start)
               + (sub.ix.values - sx.start))
        v = v[:, loc] > 0
        prim = v[t_of] & ~v[np.maximum(t_of - 1, 0)]
        tot += prim.sum(1)
        tot_no += prim[:, sub._no.values].sum(1)
        ard += v[t_of].sum(1)
    d = pd.DataFrame({"fecha": dias, "primer_dia": tot, "primer_dia_NO": tot_no,
                      "celdas_ardiendo": ard})
    d.to_csv(f, index=False)
    print(f"  verdad diaria → {f}", flush=True)
    return d


def cuantiles_de_hist(hist, qs):
    """Cuantiles exactos (a resolución de bin) desde el histograma del día."""
    cs = np.cumsum(hist, axis=-1)
    n = cs[..., -1:]
    out = {}
    centros = np.r_[BINS, 1.0][1:]
    for q in qs:
        k = (cs >= n * q).argmax(-1)
        out[q] = centros[np.clip(k, 0, NBINS - 1)]
    return out


def frac_sobre(hist, u):
    """Fracción de celdas del día por encima del umbral u."""
    k = int(np.clip(np.searchsorted(BINS, u, "right") - 1, 0, NBINS - 1))
    cs = hist[..., k:].sum(-1)
    return cs / np.maximum(hist.sum(-1), 1)


# --------------------------------------------------------------------------
# calibración beta (Kull et al. 2017): logística sobre log s y −log(1−s)
# --------------------------------------------------------------------------
class Beta:
    def __init__(self):
        self.w = None

    @staticmethod
    def _X(s):
        s = np.clip(s, 1e-9, 1 - 1e-9)
        return np.column_stack([np.log(s), -np.log(1 - s), np.ones_like(s)])

    def fit(self, s, y):
        from sklearn.linear_model import LogisticRegression
        X = self._X(s)[:, :2]
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        self.w = (m.coef_[0], m.intercept_[0])
        self.m = m
        return self

    def predict(self, s):
        return self.m.predict_proba(self._X(s)[:, :2])[:, 1]


def climatologia(d, vent=31):
    """Referencia climatológica del BSS: P(día grande) por día del año.

    Dos cuidados, y los dos importan (05/09/2026):

    1. Solo años de calibración. La media cruda por `doy` sobre los 10 años
       incluía 2024 en su propia referencia. No contamina el modelo (la beta
       se ajusta con 2015-2021), pero sí el baseline contra el que se mide la
       habilidad, y eso es una fuga.
    2. Suavizada. Con 7 años por `doy` la media cruda solo puede valer
       0, 1/7, 2/7…: memoriza el ruido y se convierte en una referencia
       imbatible por accidente. Medido: Brier 0,1454 la cruda contra 0,1617
       la honesta, y por eso todos los modelos daban BSS negativo en verano.
       Una ventana centrada de 31 días es la práctica habitual para
       climatologías diarias y deja la estacionalidad intacta (0,028-0,438).

    El `doy` 366 se rellena por interpolación: solo existe en bisiestos.
    """
    m = d[d.anio.between(*CALIB)].groupby("doy").y.mean().reindex(range(1, 367))
    m = m.interpolate().bfill().ffill()
    # triplicada para que la ventana no tenga borde entre 31-dic y 1-ene
    suav = pd.concat([m, m, m]).rolling(vent, center=True,
                                        min_periods=1).mean().iloc[366:732]
    suav.index = range(1, 367)
    return d.doy.map(suav).values


def bss(p, y, clim):
    """Brier Skill Score contra una referencia climatológica por día del año."""
    b = np.mean((p - y) ** 2)
    b0 = np.mean((clim - y) ** 2)
    return 1 - b / b0 if b0 > 0 else np.nan


def sedi(y, aviso):
    """Symmetric Extremal Dependence Index (Ferro & Stephenson 2011).

    Solo depende de sensibilidad y especificidad: no degenera con la rareza,
    al contrario que POD/FAR/ETS sobre la tabla 2x2.
    """
    a = np.sum(aviso & (y == 1)); b = np.sum(aviso & (y == 0))
    c = np.sum(~aviso & (y == 1)); d = np.sum(~aviso & (y == 0))
    H = a / max(a + c, 1); F = b / max(b + d, 1)
    H = min(max(H, 1e-6), 1 - 1e-6); F = min(max(F, 1e-6), 1 - 1e-6)
    num = np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)
    den = np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H)
    return num / den if den != 0 else np.nan


def indices_boot(n, rng, bloque=1):
    """Índices de remuestreo. `bloque=1` es el bootstrap iid de la casa.

    Con `bloque>1` es un bootstrap por bloques móviles: se remuestrean tramos
    de días consecutivos en vez de días sueltos. Hace falta mirarlo porque los
    días no son independientes (un incendio grande dura varios días y el
    tiempo persiste), y el iid, que supone independencia, estrecha los
    intervalos. Dentro de un régimen los días son contiguos salvo el salto de
    un año al siguiente; el bloque cruza esos saltos alguna vez, y es una
    aproximación asumida.
    """
    if bloque <= 1:
        return rng.integers(0, n, (N_BOOT, n))
    nb = int(np.ceil(n / bloque))
    ini = rng.integers(0, max(n - bloque + 1, 1), (N_BOOT, nb))
    idx = (ini[:, :, None] + np.arange(bloque)[None, None, :]).reshape(N_BOOT, -1)
    return np.clip(idx[:, :n], 0, n - 1)


def sedi_vec(Y, A):
    """SEDI sobre muchas remuestras a la vez. Y y A son (N_BOOT, n)."""
    a = (A & (Y == 1)).sum(1); b = (A & (Y == 0)).sum(1)
    c = (~A & (Y == 1)).sum(1); d = (~A & (Y == 0)).sum(1)
    H = np.clip(a / np.maximum(a + c, 1), 1e-6, 1 - 1e-6)
    F = np.clip(b / np.maximum(b + d, 1), 1e-6, 1 - 1e-6)
    num = np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)
    den = np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H)
    return np.where(den != 0, num / np.where(den == 0, 1, den), np.nan)


def ic_sedi_bss(y, pm, u, clim, rng, bloque=1):
    """IC percentil al 95 % de SEDI y BSS, con el umbral fijo.

    El umbral no se vuelve a elegir en cada remuestra: forma parte del sistema
    ya entrenado, y reelegirlo mediría la variabilidad del procedimiento, no
    la de la métrica que se reporta.
    """
    n = len(y)
    if n < 10:
        return (np.nan,) * 4
    I = indices_boot(n, rng, bloque)
    Y = y[I]
    S = sedi_vec(Y, pm[I] >= u)
    P = pm[I]
    C = clim[I]
    B = 1 - ((P - Y) ** 2).mean(1) / np.maximum(((C - Y) ** 2).mean(1), 1e-12)
    return (np.nanpercentile(S, 2.5), np.nanpercentile(S, 97.5),
            np.nanpercentile(B, 2.5), np.nanpercentile(B, 97.5))


def fiabilidad(p, y, nbin=10):
    """Diagrama de fiabilidad con bines de igual población (no de anchura)."""
    o = np.argsort(p)
    trozos = np.array_split(o, nbin)
    return pd.DataFrame([{"n": len(t), "p_medio": p[t].mean(),
                          "obs": y[t].mean()} for t in trozos if len(t)])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral-dia", type=int, default=5,
                    help="celdas de primer día que definen «día grande»")
    a = ap.parse_args()

    dias = pd.date_range(INI, FIN, freq="D")
    z = np.load(config.salida("dos_25_hist.npz"))
    hist = z["hist"]                     # (dias, modelos, bins)
    hechos = z["hechos"]
    print(f"histograma {hist.shape} · {len(hechos)} bloques · "
          f"celdas/día {int(hist[0, 0].sum()):,}", flush=True)
    v = verdad_diaria(dias)

    qs = [0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    Q = cuantiles_de_hist(hist, qs)
    d = pd.DataFrame({"fecha": dias})
    d["anio"] = d.fecha.dt.year
    d["mes"] = d.fecha.dt.month
    d["doy"] = d.fecha.dt.dayofyear
    d["regimen"] = d.mes.map(REGIMEN)
    d = d.merge(v, on="fecha")
    d["y"] = (d.primer_dia >= a.umbral_dia).astype(int)
    d["y_algo"] = (d.primer_dia >= 1).astype(int)
    d["frac_NO"] = d.primer_dia_NO / d.primer_dia.replace(0, np.nan)
    for k, nom in enumerate(MODELOS):
        for q in qs:
            d[f"p{int(q*1000)}_{nom}"] = Q[q][:, k]
    d.to_csv(f"{SAL}_dias.csv", index=False)

    print(f"\nobjetivo: ≥{a.umbral_dia} celdas de primer día → "
          f"{d.y.mean()*100:.1f} % de los días ({int(d.y.sum())} de {len(d)})")
    print(d.groupby("regimen").agg(dias=("y", "size"), grandes=("y", "sum"),
                                   pct=("y", lambda x: round(100*x.mean(), 1)),
                                   arde_algo=("y_algo", lambda x: round(100*x.mean(), 1))
                                   ).to_string())

    # ---- calibración por régimen -----------------------------------------
    clim = climatologia(d)
    rng = np.random.default_rng(SEED)
    res, tablas = {}, []
    for nom in MODELOS:
        s = d[f"p980_{nom}"].values          # el p98 del día = «≥2 % de celdas»
        for reg in ("verano", "invierno-primavera", "transicion", "TODO"):
            m = np.ones(len(d), bool) if reg == "TODO" else (d.regimen == reg).values
            ca = m & d.anio.between(*CALIB).values
            va = m & d.anio.between(*VAL).values
            te = m & d.anio.between(*TEST).values
            if d.y[ca].sum() < 20 or d.y[te].sum() < 3:
                continue
            b = Beta().fit(s[ca], d.y[ca].values)
            p = b.predict(s)
            for nomp, msk in (("calib", ca), ("val", va), ("test", te)):
                y = d.y[msk].values
                pm = p[msk]
                # umbral elegido solo en calibración: máximo SEDI, pero
                # exigiendo que la tabla 2x2 sea estimable (05/09/2026).
                #
                # SEDI premia el pronóstico rarísimo y perfecto: sin la
                # restricción, `pareja/verano` elegía un umbral con 3 avisos
                # en 854 días (3 aciertos, precisión 1,0) que en test daba
                # cero avisos y SEDI 0; a `prod/transicion` le pasaba igual
                # con 6. Con celdas de la tabla casi vacías el índice no es
                # estimable (su varianza explota), y un aviso que salta tres
                # veces en siete años tampoco sirve como producto operativo.
                if nomp == "calib":
                    reja = np.unique(np.quantile(pm, np.linspace(0.5, 0.999, 200)))
                    viables = reja[[(pm >= t).sum() >= MIN_AVISOS for t in reja]]
                    if len(viables):
                        reja = viables
                    u = reja[np.argmax([sedi(y, pm >= t) for t in reja])]
                av = pm >= u
                s_lo, s_hi, b_lo, b_hi = ic_sedi_bss(y, pm, u, clim[msk], rng)
                sb_lo, sb_hi, _, _ = ic_sedi_bss(y, pm, u, clim[msk], rng,
                                                 bloque=BLOQUE)
                tablas.append(dict(modelo=nom, regimen=reg, parte=nomp,
                                   n=int(msk.sum()), pos=int(y.sum()),
                                   tasa_base=round(y.mean(), 4),
                                   brier=round(np.mean((pm - y) ** 2), 4),
                                   bss=round(bss(pm, y, clim[msk]), 4),
                                   sedi=round(sedi(y, av), 4),
                                   sedi_lo=round(s_lo, 4), sedi_hi=round(s_hi, 4),
                                   sedi_lo_bl=round(sb_lo, 4),
                                   sedi_hi_bl=round(sb_hi, 4),
                                   bss_lo=round(b_lo, 4), bss_hi=round(b_hi, 4),
                                   avisos=int(av.sum()),
                                   aciertos=int((av & (y == 1)).sum()),
                                   falsas=int((av & (y == 0)).sum()),
                                   perdidos=int((~av & (y == 1)).sum()),
                                   umbral_p=round(float(u), 5)))
            res[f"{nom}|{reg}"] = dict(umbral_prob=float(u),
                                       beta=[float(x) for x in b.m.coef_[0]]
                                       + [float(b.m.intercept_[0])])
    T = pd.DataFrame(tablas)
    T.to_csv(f"{SAL}_calibracion.csv", index=False)
    json.dump(res, open(f"{SAL}_calibracion.json", "w"), indent=1)
    print("\n== TEST 2024 ==")
    print(T[T.parte == "test"].to_string(index=False))
    print(f"\n→ {SAL}_dias.csv · {SAL}_calibracion.csv/.json")

    # ---- fiabilidad del mejor por régimen --------------------------------
    lin = []
    for reg in ("verano", "invierno-primavera"):
        sub = T[(T.regimen == reg) & (T.parte == "val")]
        if not len(sub):
            continue
        best = sub.sort_values("bss", ascending=False).iloc[0]
        lin.append(f"\n### {reg}: mejor por BSS en validación = {best.modelo}")
        m = (d.regimen == reg).values & d.anio.between(*TEST).values
        b = Beta().fit(d[f"p980_{best.modelo}"].values[
            (d.regimen == reg).values & d.anio.between(*CALIB).values],
            d.y[(d.regimen == reg).values & d.anio.between(*CALIB).values].values)
        p = b.predict(d[f"p980_{best.modelo}"].values[m])
        lin.append(fiabilidad(p, d.y[m].values, 5).round(3).to_string(index=False))
    open(f"{SAL}_fiabilidad.md", "w").write("\n".join(lin))
    print("\n".join(lin))


if __name__ == "__main__":
    main()
