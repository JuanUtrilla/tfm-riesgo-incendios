#!/usr/bin/env python3
"""
¿Se salva el híbrido corrigiendo el viento del IFS? (§6ter, continuación)

No sobrescribe nada. Escribe dataset/prueba_hibrido_corregido.json.

De dónde viene. El híbrido (ERA5 hasta D−6 + IFS los últimos días) satura el
percentil al 2,2 %, contra el 0,8 % del reanálisis puro y el 0,2 % de
referencia. El diagnóstico: los extremos del FWI salen del ISI, que depende del
FFMC (~1 día) y es exponencial en el viento, y no del DC (memoria ~52 días, que
aporta el reanálisis). Y el viento del IFS va mal contra el cubo: r = 0,745 y
+0,79 m/s de sesgo. La precipitación también (r = 0,748, +0,72 mm).

La corrección. Mapeo de cuantiles IFS → escala del cubo, por estación. Es
monótono, luego conserva el orden (y por tanto la información) y solo reescala.
Es la misma técnica que usa `auditoria_train_serve.py`, allí para simular el
desajuste y aquí para deshacerlo.

Sin fuga: la transformación se ajusta con los meses de 2024 fuera de jun-sep
(feb-may y oct-dic; el archivo del IFS empieza el 2024-02-03, no hay 2023) y se
evalúa sobre jun-sep 2024. En producción se ajustaría sobre el solape histórico
con el cubo y se aplicaría en adelante.

Salvedad: al no haber solape en temporada, la calibración es fuera de
temporada. La distribución del viento en invierno no es la del verano, así que
esto es una cota inferior de lo que daría un mapeo calibrado en jun-sep de
otros años.

Escenarios, todos con L=6 (el retraso real de ERA5-Land):
    IFS crudo · viento corregido · viento+precip · las cuatro variables
más IFS puro con las cuatro corregidas, como cota.

Uso: python prueba_hibrido_corregido.py
"""

import json

import numpy as np
import pandas as pd

from fwi_canadiense import calcular_fwi_serie
from prueba_era5_produccion import DIR, EVAL, FIN, SPIN, estaciones, serie_openmeteo
from prueba_era5_atribucion import N, cubo_cacheado
from prueba_ifs_produccion import serie_ifs
from prueba_hibrido_produccion import paso

L = 6
Q = np.linspace(0, 1, 201)          # rejilla de cuantiles para el mapeo


def ajusta_qm(src, ref, m_fit):
    """Mapeo de cuantiles por estación, ajustado solo con los días de m_fit
    (2024 fuera de temporada)."""
    out = src.copy()
    for j in range(src.shape[1]):
        a, b = src[m_fit, j], ref[m_fit, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 100 or len(b) < 100:
            continue
        xs, ys = np.quantile(a, Q), np.quantile(b, Q)
        ok = np.isfinite(src[:, j])
        out[ok, j] = np.interp(src[ok, j], xs, ys)
    return out


def main():
    est = estaciones(N)
    fechas = pd.date_range(SPIN, FIN, freq="D")
    meses = fechas.month.values
    # El archivo del IFS empieza el 2024-02-03: 2023 está vacío. Se calibra
    # con los meses de 2024 fuera de la ventana de evaluación (feb-may y
    # oct-dic) y se evalúa en jun-sep. No hay fuga, pero sí una salvedad que
    # hay que declarar: la calibración es fuera de temporada, y la
    # distribución del viento de invierno no es la del verano.
    m_fit = np.asarray((fechas.year == 2024)
                       & ~((fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1])))
    m_ev = np.where((fechas.month >= EVAL[0]) & (fechas.month <= EVAL[1])
                    & (fechas.year == 2024))[0]
    print(f"ajuste: {m_fit.sum()} días de 2024 fuera de temporada | evaluación: {len(m_ev)} días "
          f"jun-sep 2024 | {len(est)} estaciones\n")

    cubo = cubo_cacheado(est, fechas)
    om, ifs = serie_openmeteo(est), serie_ifs(est)

    def matriz(df, c):
        return np.column_stack([
            df[df.idema == i].set_index("fecha").reindex(fechas)[c].values
            for i in est["idema"]])

    C = {"tmax": cubo["t2m_max"], "hr_min": cubo["RH_min"],
         "viento_max": cubo["wind_speed_max"],
         "prec": cubo["total_precipitation_mean"]}
    E = {c: matriz(om, c) for c in C}
    F = {c: matriz(ifs, c) for c in C}

    print("  ajustando mapeos de cuantiles (feb-may + oct-dic 2024)...", flush=True)
    FC = {c: ajusta_qm(F[c], C[c], m_fit) for c in C}
    for c in C:
        a = C[c][m_ev].ravel(); b = F[c][m_ev].ravel(); d = FC[c][m_ev].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        print(f"    {c:<12} cubo {a[ok].mean():6.2f} | IFS {b[ok].mean():6.2f} "
              f"→ corregido {d[ok].mean():6.2f}")

    ref = np.full((len(fechas), len(est)), np.nan)
    for j in range(len(est)):
        ref[:, j] = calcular_fwi_serie(C["tmax"][:, j], C["hr_min"][:, j],
                                       C["viento_max"][:, j] * 3.6,
                                       C["prec"][:, j], meses)["fwi"]
    clim = {i: np.load(f"{DIR}/prototipo/clim_fwi/{i}.npz") for i in est["idema"]}

    def evalua(fuente_cola, hibrido=True):
        M = np.full((len(fechas), len(est)), np.nan)
        for j in range(len(est)):
            if hibrido:
                s = calcular_fwi_serie(E["tmax"][:, j], E["hr_min"][:, j],
                                       E["viento_max"][:, j] * 3.6,
                                       E["prec"][:, j], meses)
                for k in m_ev:
                    e = (s["ffmc"][k - L], s["dmc"][k - L], s["dc"][k - L])
                    if not np.all(np.isfinite(e)):
                        continue
                    v = np.nan
                    for q in range(k - L + 1, k + 1):
                        e, v = paso(fuente_cola["tmax"][q, j],
                                    fuente_cola["hr_min"][q, j],
                                    fuente_cola["viento_max"][q, j] * 3.6,
                                    fuente_cola["prec"][q, j], meses[q], e)
                    M[k, j] = v
            else:
                M[:, j] = calcular_fwi_serie(
                    fuente_cola["tmax"][:, j], fuente_cola["hr_min"][:, j],
                    fuente_cola["viento_max"][:, j] * 3.6,
                    fuente_cola["prec"][:, j], meses)["fwi"]
        a, b = ref[m_ev].ravel(), M[m_ev].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        p = []
        for j, i in enumerate(est["idema"]):
            for k in m_ev:
                v = M[k, j]
                if np.isfinite(v):
                    c = clim[i][f"m{meses[k]}"]
                    c = np.sort(c[~np.isnan(c)])
                    if len(c):
                        p.append(np.searchsorted(c, v, side="right") / len(c) * 100)
        p = np.array(p)
        return dict(corr=float(np.corrcoef(a[ok], b[ok])[0, 1]),
                    sesgo=float(b[ok].mean() - a[ok].mean()),
                    maximo=float(np.nanmax(b)),
                    pctl_mediana=float(np.median(p)),
                    pctl_pc_ge999=float((p >= 99.9).mean() * 100))

    mezcla = lambda **kw: {**F, **kw}
    escenarios = {
        "híbrido, IFS crudo": (mezcla(), True),
        "híbrido, viento corregido": (mezcla(viento_max=FC["viento_max"]), True),
        "híbrido, viento+precip": (mezcla(viento_max=FC["viento_max"],
                                          prec=FC["prec"]), True),
        "híbrido, las 4 corregidas": (FC, True),
        "IFS puro, las 4 corregidas": (FC, False),
    }
    res = {}
    for nom, (src, hib) in escenarios.items():
        res[nom] = evalua(src, hib)
        print(f"  {nom} hecho", flush=True)

    print("\n" + "=" * 78)
    print(f"CORRECCIÓN POR CUANTILES (ajuste feb-may+oct-dic 2024 → eval jun-sep 2024, L={L})")
    print("=" * 78)
    print(f"{'configuración':<30}{'corr':>8}{'sesgo':>9}{'máx':>9}"
          f"{'pctl med':>10}{'% >=99,9':>11}")
    print(f"{'CUBO (referencia)':<30}{1.0:>8.3f}{0.0:>+9.2f}"
          f"{np.nanmax(ref[m_ev]):>9.1f}{46.5:>10.1f}{0.2:>11.1f}")
    print(f"{'ERA5 puro (inalcanzable)':<30}{0.976:>8.3f}{-2.01:>+9.2f}"
          f"{110.1:>9.1f}{38.6:>10.1f}{0.8:>11.1f}")
    for nom, r in res.items():
        print(f"{nom:<30}{r['corr']:>8.3f}{r['sesgo']:>+9.2f}{r['maximo']:>9.1f}"
              f"{r['pctl_mediana']:>10.1f}{r['pctl_pc_ge999']:>11.1f}")
    print(f"{'AEMET (producción hoy)':<30}{0.824:>8.3f}{-0.90:>+9.2f}"
          f"{179.2:>9.1f}{36.9:>10.1f}{3.3:>11.1f}")

    with open(f"{DIR}/dataset/prueba_hibrido_corregido.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("\nGuardado: dataset/prueba_hibrido_corregido.json")


if __name__ == "__main__":
    main()
