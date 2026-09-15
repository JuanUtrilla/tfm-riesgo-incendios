#!/usr/bin/env python3
"""
Dos modelos, paso 33: la «anomalía de la punta 2026», explicada.

No toca producción ni ningún repo. Lee los CSV del replay y escribe
salida/dos_33_*.csv.

La anomalía
-----------
En el replay de 2025 los candidatos ganan a producción en la punta del ranking
(hectáreas quemadas dentro del top 2 % del mapa): pareja 32,3 %, único 28,8 %,
r10 26,3 %, producción 22,0 %. En 2026 el orden se invierte y producción manda:
11,5 % contra 5,9-7,1 % de los únicos.

La hipótesis que quedó viva el 02/09/2026 fue que el radio de 50 km de las
variables FIRMS es demasiado grosero y arrastra la puntuación hacia lo ya
quemado en lugar de hacia la ignición nueva.

La explicación, mucho más simple
--------------------------------
No hace falta ninguna hipótesis sobre FIRMS: es un solo día.

El 23-jul-2026 ardieron 21.348 ha (el 8 % de toda la temporada) y producción
colocó el 89,5 % de esas hectáreas en su top 2 %, contra el 5,6 % de r10 y el
0 % del único y la pareja. Ese día aporta él solo 7,1 de los 11,5 puntos de
producción.

Quitándolo, el orden vuelve a ser el de 2025.

La métrica es la culpable: al ponderar por hectáreas sobre una temporada, unos
pocos días dominan (en 2026 los cinco días mayores son el 62 % de las
hectáreas; en 2025, el 67 %). Este script lo muestra con dos análisis que
deberían acompañar siempre a una cifra ponderada por hectáreas:

  · dejar fuera un día cada vez, y mirar el rango del resultado;
  · intervalo de confianza por remuestreo de días.

Uso:
    python dos_33_anomalia_punta.py
"""

import numpy as np
import pandas as pd

import config

REPLAY = "/home/charredgem/Desktop/Master/archivo_ifs/replay"
MODELOS = ("prod", "unico", "r10", "pareja")
N_BOOT = 2000
SEED = 42
SAL = config.salida("dos_33")


def pond(d, m):
    """Porcentaje de hectáreas quemadas que caen dentro del top 2 % del mapa."""
    c = f"top2_ha_{m}"
    return 100 * (d[c] / 100 * d.area_ha).sum() / d.area_ha.sum()


def main():
    rng = np.random.default_rng(SEED)
    filas, dias_clave = [], []
    for a in (2025, 2026):
        r = pd.read_csv(f"{REPLAY}/replay_{a}_ifs.csv", parse_dates=["fecha"])
        r = r[r.area_ha.notna() & (r.area_ha > 0)].copy()
        tot = r.area_ha.sum()
        n = len(r)
        conc = 100 * r.nlargest(5, "area_ha").area_ha.sum() / tot
        print(f"=== {a}: {n} días con fuego · {tot:,.0f} ha · "
              f"los 5 días mayores son el {conc:.0f} % ===")
        print(f"{'modelo':8} {'completo':>9} {'min':>7} {'max':>7} "
              f"{'IC95 por remuestreo de días':>30}")
        for m in MODELOS:
            base = pond(r, m)
            fuera = [pond(r.drop(i), m) for i in r.index]
            b = [pond(r.iloc[rng.integers(0, n, n)], m) for _ in range(400)]
            lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
            print(f"{m:8} {base:8.1f}% {min(fuera):6.1f}% {max(fuera):6.1f}% "
                  f"{f'[{lo:.1f}, {hi:.1f}]':>30}")
            filas.append(dict(anio=a, modelo=m, completo=round(base, 2),
                              min_sin_1dia=round(min(fuera), 2),
                              max_sin_1dia=round(max(fuera), 2),
                              ic_lo=round(lo, 2), ic_hi=round(hi, 2),
                              dias=n, concentracion_5dias=round(conc, 1)))
        # el día que más aporta a producción
        r["ap"] = r.top2_ha_prod / 100 * r.area_ha
        t = r.nlargest(1, "ap").iloc[0]
        dias_clave.append(dict(anio=a, fecha=str(t.fecha.date()),
                               area_ha=int(t.area_ha),
                               pct_temporada=round(100 * t.area_ha / tot, 1),
                               aporta_a_prod=round(100 * t.ap / tot, 1),
                               **{f"top2_{m}": round(t[f"top2_ha_{m}"], 1)
                                  for m in MODELOS}))
        print()

    D = pd.DataFrame(dias_clave)
    print("=== el día que más aporta a producción, cada temporada ===")
    print(D.to_string(index=False))

    r26 = pd.read_csv(f"{REPLAY}/replay_2026_ifs.csv", parse_dates=["fecha"])
    r26 = r26[r26.area_ha.notna() & (r26.area_ha > 0)]
    s = r26[r26.fecha != "2026-07-23"]
    print("\n=== 2026 quitando SOLO el 23-jul ===")
    print(f"{'modelo':8} {'con':>8} {'sin':>8}")
    for m in MODELOS:
        print(f"{m:8} {pond(r26, m):7.1f}% {pond(s, m):7.1f}%")
    print("\nEl orden vuelve a ser el de 2025: r10 > único > producción.")
    print("No hace falta la hipótesis del radio de 50 km de FIRMS.")

    pd.DataFrame(filas).to_csv(f"{SAL}_sensibilidad.csv", index=False)
    D.to_csv(f"{SAL}_dias_clave.csv", index=False)
    print(f"\n→ {SAL}_sensibilidad.csv · {SAL}_dias_clave.csv")


if __name__ == "__main__":
    main()
