#!/usr/bin/env python3
"""
¿De verdad el FWI del cubo son las 13 UTC? Verificación antes de tocar nada.

No toca producción. Escribe salida/verificar_hora_fwi.json.

Qué se verifica y por qué hace falta
------------------------------------
`diagnostico_fwi.py` encontró que el FWI del cubo (31,9 de media en julio
2024) se reproduce calculándolo con los valores instantáneos de las 13 UTC
(31,7), y no con el proxy tmax/hrMin que se sirve (39,2) ni con el mediodía
canónico de las 11 UTC (25,1).

Pero eso era un mes, 300 nodos y una comparación de medias. Dos medias pueden
coincidir con distribuciones distintas, y una hora puede acertar en julio por
casualidad. Actuar sobre eso significaría rehacer los 84 meses de la
climatología, así que primero se comprueba con más cuidado:

 · varios meses (julio, agosto, septiembre de 2024)
 · media, mediana, sesgo, correlación y error absoluto
 · la hora que minimiza el sesgo en cada mes, para ver si es estable o si
   cambia con la estación

Si la hora buena se mueve de un mes a otro, no hay una "convención horaria"
que copiar y la hipótesis se cae.

Spin-up. La serie arranca el 1-jun-2024 y solo se evalúan julio, agosto y
septiembre, para que el FWI llegue con memoria a los meses medidos. Junio se
usa de rodaje y no se puntúa.

Uso: python verificar_hora_fwi.py
"""

import json

import numpy as np
import pandas as pd
import xarray as xr

import config
from fwi_canadiense import calcular_fwi_serie
from malla_02_descarga import a_diario, abre, hr_desde_rocio, pide_mes

N_NODOS = 300
HORAS = (11, 12, 13, 14, 15)
MESES_EVAL = (7, 8, 9)


def main():
    n = np.load(config.NODOS)
    la, lo = n["nodo_lat"], n["nodo_lon"]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(la), N_NODOS, replace=False))
    LA = xr.DataArray(la[sel], dims="n")
    LO = xr.DataArray(lo[sel], dims="n")

    # --- horario y diario en los nodos, mes a mes (no cabe todo junto) ---
    H = {h: [] for h in HORAS}
    dias, prec, tmax, hrmin = [], [], [], []
    for m in (6, 7, 8, 9):
        ds = abre(pide_mes(2024, m, list(range(1, 32))))
        p = ds.sel(latitude=LA, longitude=LO, method="nearest")
        t = pd.to_datetime(p["valid_time"].values)
        for h in HORAS:
            k = t.hour == h
            H[h].append(pd.DataFrame(
                {"fecha": t[k].normalize()}).assign(
                _t=list((p["t2m"].values[k] - 273.15)),
                _h=list(hr_desde_rocio(p["t2m"].values[k], p["d2m"].values[k])),
                _v=list(np.sqrt(p["u10"].values[k] ** 2
                                + p["v10"].values[k] ** 2))))
        d = a_diario(ds).sel(latitude=LA, longitude=LO, method="nearest")
        dias.append(pd.to_datetime(d["fecha"].values).normalize())
        prec.append(d["prec"].values)
        tmax.append(d["tmax"].values)
        hrmin.append(d["hr_min"].values)
        ds.close()
        print(f"  {2024}-{m:02d} leído", flush=True)

    fechas = pd.DatetimeIndex(np.concatenate(dias))
    orden = np.argsort(fechas.values)
    fechas = fechas[orden]
    PREC = np.vstack(prec)[orden]
    TMAX = np.vstack(tmax)[orden]
    HRMIN = np.vstack(hrmin)[orden]
    meses = fechas.month.values

    def serie(T, HR, V):
        M = np.full(T.shape, np.nan)
        for j in range(T.shape[1]):
            M[:, j] = calcular_fwi_serie(T[:, j], HR[:, j], V[:, j] * 3.6,
                                         PREC[:, j], meses)["fwi"]
        return M

    def alinea(lst):
        df = pd.concat(lst, ignore_index=True).set_index("fecha")
        df = df.reindex(fechas)
        return (np.array([np.asarray(x) for x in df["_t"]]),
                np.array([np.asarray(x) for x in df["_h"]]),
                np.array([np.asarray(x) for x in df["_v"]]))

    print("  FWI por hora...", flush=True)
    FWI = {h: serie(*alinea(H[h])) for h in HORAS}
    FWI["proxy"] = serie(TMAX, HRMIN,
                         np.full(TMAX.shape, np.nan))  # relleno, se corrige
    # el proxy usa viento_max diario: se recalcula bien
    vmax = []
    for m in (6, 7, 8, 9):
        ds = abre(pide_mes(2024, m, list(range(1, 32))))
        d = a_diario(ds).sel(latitude=LA, longitude=LO, method="nearest")
        vmax.append(d["viento_max"].values)
        ds.close()
    VMAX = np.vstack(vmax)[orden]
    FWI["proxy"] = serie(TMAX, HRMIN, VMAX)

    # --- el cubo en las mismas celdas y días ---
    from pyproj import Transformer
    cu = xr.open_dataset(config.CUBO, decode_timedelta=False)
    xs, ys = cu["x"].values, cu["y"].values
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    X, Y = tr.transform(lo[sel], la[sel])
    ix = np.rint((X - xs[0]) / (xs[1] - xs[0])).astype(int)
    iy = np.rint((Y - ys[0]) / (ys[1] - ys[0])).astype(int)
    t = pd.to_datetime(cu["time"].values)
    a = int(np.where(t == fechas[0])[0][0])
    b = int(np.where(t == fechas[-1])[0][0])
    CUBO = cu["FWI"].isel(time=slice(a, b + 1)).values[:, iy, ix]
    cu.close()

    res = {}
    print("\n" + "=" * 78)
    print("¿QUÉ HORA REPRODUCE EL FWI DEL CUBO? — 2024, "
          f"{N_NODOS} nodos, por mes")
    print("=" * 78)
    print(f"{'mes':>5}{'fuente':>10}{'media':>9}{'cubo':>9}{'sesgo':>9}"
          f"{'p50':>8}{'corr':>8}{'|err| med':>11}")
    for m in MESES_EVAL:
        k = meses == m
        c = CUBO[k]
        fila = {}
        for nom in list(HORAS) + ["proxy"]:
            v = FWI[nom][k]
            o = np.isfinite(v) & np.isfinite(c)
            s = v[o].mean() - c[o].mean()
            fila[str(nom)] = dict(media=float(v[o].mean()), sesgo=float(s),
                                  corr=float(np.corrcoef(c[o], v[o])[0, 1]),
                                  err_abs=float(np.abs(v[o] - c[o]).mean()))
            et = f"{nom} UTC" if nom != "proxy" else "proxy"
            print(f"{m:>5}{et:>10}{v[o].mean():>9.1f}{c[o].mean():>9.1f}"
                  f"{s:>+9.1f}{np.median(v[o]):>8.1f}"
                  f"{fila[str(nom)]['corr']:>8.3f}"
                  f"{fila[str(nom)]['err_abs']:>11.1f}")
        mejor = min((h for h in HORAS), key=lambda h: abs(fila[str(h)]["sesgo"]))
        fila["mejor_hora"] = int(mejor)
        res[f"2024-{m:02d}"] = fila
        print(f"{'':>5}{'→ mejor:':>10}{mejor:>4} UTC "
              f"(sesgo {fila[str(mejor)]['sesgo']:+.1f})")
        print()

    horas = [res[k]["mejor_hora"] for k in res]
    estable = len(set(horas)) == 1
    print("-" * 78)
    print(f"  hora que minimiza el sesgo por mes: {horas}")
    print(f"  → {'ESTABLE: hay convención horaria que copiar' if estable else 'NO ESTABLE: la hipótesis no se sostiene'}")
    res["horas_por_mes"] = horas
    res["estable"] = bool(estable)
    with open(config.salida("verificar_hora_fwi.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado: {config.salida('verificar_hora_fwi.json')}")


if __name__ == "__main__":
    main()
