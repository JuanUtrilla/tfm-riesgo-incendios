#!/usr/bin/env python3
"""
Pipeline de malla, módulo 3: corrección de la precipitación por cuantiles.

No toca producción. Escribe malla_data/mapeo_precip.npz y
dataset/malla_03_cuantiles.json.

Por qué solo la precipitación
-----------------------------
Medido en `malla_02_vs_cubo.py` (jun-sep 2024, 120 estaciones, contra el cubo):

    tmax        sesgo −0,01 °C   r 0,998   → ERA5-Land ya coincide
    hr_min      sesgo +0,05 %    r 0,997   → ERA5-Land ya coincide
    viento_max  sesgo +0,00 m/s  r 0,987   → ERA5-Land ya coincide
    prec        sesgo +0,71 mm   r 0,904   → única que hay que corregir

Ir a ERA5-Land puro arregló el viento (era +0,79 con `era5_seamless`) pero no
la lluvia: ERA5-Land llueve 1,40 mm/día, igual que ERA5 (1,38) y el doble que
el cubo (0,69). No es un problema de unidades: el ratio tiene mediana 1,84 con
cuartiles 1,25 y 2,80, y dividir por 2 empeora la correlación (0,875 vs 0,904).

Por eso se usa un mapeo de cuantiles y no una regla de tres: es monótono
(conserva el orden, y con r=0,904 el orden es bueno) y ajusta la distribución
sin suponer que la relación sea lineal ni constante.

Decisión: mapeo global, no por nodo
-----------------------------------
La discrepancia parece de procesado del cubo (`total_precipitation_mean`), no
un fenómeno local, así que debería ser espacialmente homogénea. Un mapeo global
se ajusta con ~14.000 muestras en vez de ~122 por nodo, y por tanto estima bien
la cola, que es donde se juega el FWI. El script comprueba esa hipótesis
midiendo si el mapeo global deja sesgo residual por estación.

La lluvia es cero la mayor parte de los días, así que el mapeo se ajusta solo
sobre los días húmedos y los ceros se preservan: mapear ceros contra cuantiles
metería lluvia donde no la hubo.

Prueba de aceptación
--------------------
No basta con que cuadren las medias de precipitación: lo que importa es el FWI.
El script recalcula el FWI con ERA5-Land corregido y lo compara con el del
cubo. Criterio: |sesgo| < 1,0 y corr > 0,97 (con `era5_seamless` era −2,01 y
0,976; el reanálisis debería mejorarlo).

Uso: /home/charredgem/miniconda3/envs/tfm_fuego/bin/python malla_03_cuantiles.py
"""

import json

import numpy as np
import pandas as pd
import xarray as xr

from fwi_canadiense import calcular_fwi_serie
from malla_02_descarga import DATA, DIR, a_diario, abre, pide_mes
from prueba_era5_produccion import EVAL, FIN, SPIN, estaciones
from prueba_era5_atribucion import N, cubo_cacheado

Q = np.linspace(0, 1, 501)
UMBRAL = 0.1          # mm: por debajo se considera día seco y se preserva


def main():
    est = estaciones(N)
    fechas = pd.date_range(SPIN, FIN, freq="D")
    meses = fechas.month.values
    m_ev = np.asarray((fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1])
                      & (fechas.year == 2024))
    print(f"{len(est)} estaciones · {m_ev.sum()} días jun-sep 2024\n")

    rutas = [pide_mes(2024, m, list(range(1, 32))) for m in (6, 7, 8, 9)]
    ps = [a_diario(abre(r)) for r in rutas]
    dim = [d for d in ps[0].dims if d not in ("latitude", "longitude")][0]
    ds = xr.concat(ps, dim=dim).rename({dim: "fecha"}).sortby("fecha").load()

    cubo = cubo_cacheado(est, fechas)
    fmap = {pd.Timestamp(t).normalize(): i for i, t in enumerate(ds.fecha.values)}
    idx = [fmap.get(f) for f in fechas]
    L = {}
    for v in ["tmax", "hr_min", "viento_max", "prec"]:
        M = np.full((len(fechas), len(est)), np.nan)
        for j, (_, e) in enumerate(est.iterrows()):
            col = ds[v].sel(latitude=e.lat, longitude=e.lon,
                            method="nearest").values
            M[:, j] = [np.nan if i is None else col[i] for i in idx]
        L[v] = M
    C = {"tmax": cubo["t2m_max"], "hr_min": cubo["RH_min"],
         "viento_max": cubo["wind_speed_max"],
         "prec": cubo["total_precipitation_mean"]}

    # ---- ajuste: adaptación de frecuencia + mapeo de cuantiles -------------
    # Corregir solo la magnitud dejaba la mitad del sesgo (−7,08 → −3,56): el
    # reanálisis grueso tiene "drizzle bias", llueve más días de los que llueve
    # en el cubo, y esos días de más mojan el combustible aunque su magnitud
    # esté bien escalada. Primero se iguala la frecuencia de días húmedos
    # (umbral sobre ERA5-Land tal que P(ERA5>t) = P(cubo>UMBRAL)) y solo
    # después se mapean los cuantiles de los días que sobreviven.
    a, b = C["prec"][m_ev].ravel(), L["prec"][m_ev].ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    f_cubo = (a[ok] >= UMBRAL).mean()
    umbral_era5 = float(np.quantile(b[ok], 1.0 - f_cubo))
    print(f"frecuencia de días húmedos: cubo {f_cubo*100:.1f}% · "
          f"ERA5-Land crudo {(b[ok]>=UMBRAL).mean()*100:.1f}%")
    print(f"  umbral de frecuencia sobre ERA5-Land: {umbral_era5:.3f} mm")
    hum_a = ok & (a >= UMBRAL)
    hum_b = ok & (b >= umbral_era5)
    xs, ys = np.quantile(b[hum_b], Q), np.quantile(a[hum_a], Q)
    print(f"ajuste con {hum_b.sum():,} días húmedos de ERA5-Land "
          f"y {hum_a.sum():,} del cubo (frecuencias ya igualadas)")
    print(f"  ERA5-Land: media {b[hum_b].mean():.2f} · p95 {np.quantile(b[hum_b],.95):.2f}")
    print(f"  cubo     : media {a[hum_a].mean():.2f} · p95 {np.quantile(a[hum_a],.95):.2f}")

    def corrige(x):
        seco = np.isfinite(x) & (x < umbral_era5)
        y = np.where(np.isfinite(x) & ~seco, np.interp(x, xs, ys), x)
        return np.where(seco, 0.0, y)

    Pc = corrige(L["prec"])
    np.savez_compressed(f"{DATA}/mapeo_precip.npz", xs=xs, ys=ys,
                        umbral=UMBRAL, umbral_era5=umbral_era5,
                        ajustado_con="jun-sep 2024, 120 est.")

    # ---- ¿el mapeo global vale, o hace falta por nodo? ---------------------
    ses_est = []
    for j in range(len(est)):
        u, w = C["prec"][m_ev, j], Pc[m_ev, j]
        o = np.isfinite(u) & np.isfinite(w)
        if o.sum() > 30:
            ses_est.append(w[o].mean() - u[o].mean())
    ses_est = np.array(ses_est)
    print(f"\nsesgo residual por estación tras el mapeo global: "
          f"mediana {np.median(ses_est):+.3f} · "
          f"p10 {np.percentile(ses_est,10):+.3f} · "
          f"p90 {np.percentile(ses_est,90):+.3f} mm/día")
    print("  (si p10-p90 fuera ancho, habría que mapear por nodo o por región)")

    # ---- prueba de aceptación: el FWI --------------------------------------
    def fwi_m(src, prec):
        M = np.full((len(fechas), len(est)), np.nan)
        for j in range(len(est)):
            M[:, j] = calcular_fwi_serie(src["tmax"][:, j], src["hr_min"][:, j],
                                         src["viento_max"][:, j] * 3.6,
                                         prec[:, j], meses)["fwi"]
        return M

    ref = fwi_m(C, C["prec"])
    esc = {"ERA5-Land crudo": fwi_m(L, L["prec"]),
           "ERA5-Land + precip corregida": fwi_m(L, Pc)}
    print("\n" + "=" * 70)
    print("PRUEBA DE ACEPTACIÓN — FWI contra el cubo (jun-sep 2024)")
    print("=" * 70)
    print(f"{'configuración':<32}{'sesgo':>9}{'corr':>8}{'máx':>9}")
    print(f"{'CUBO (referencia)':<32}{0.0:>+9.2f}{1.0:>8.3f}"
          f"{np.nanmax(ref[m_ev]):>9.1f}")
    res = {}
    for nom, M in esc.items():
        u, w = ref[m_ev].ravel(), M[m_ev].ravel()
        o = np.isfinite(u) & np.isfinite(w)
        s, r = w[o].mean() - u[o].mean(), np.corrcoef(u[o], w[o])[0, 1]
        print(f"{nom:<32}{s:>+9.2f}{r:>8.3f}{np.nanmax(M[m_ev]):>9.1f}")
        res[nom] = dict(sesgo=float(s), corr=float(r))
    print(f"{'(era5_seamless, §6)':<32}{-2.01:>+9.2f}{0.976:>8.3f}{110.1:>9.1f}")

    s = res["ERA5-Land + precip corregida"]
    pasa = abs(s["sesgo"]) < 1.0 and s["corr"] > 0.97
    print("\n" + ("ACEPTADO" if pasa else "NO PASA") +
          f" — criterio |sesgo|<1,0 y corr>0,97")
    res["acepta"] = bool(pasa)
    res["sesgo_residual_precip_p10_p90"] = [float(np.percentile(ses_est, 10)),
                                            float(np.percentile(ses_est, 90))]
    with open(f"{DIR}/dataset/malla_03_cuantiles.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("\nGuardado: malla_data/mapeo_precip.npz")


if __name__ == "__main__":
    main()
