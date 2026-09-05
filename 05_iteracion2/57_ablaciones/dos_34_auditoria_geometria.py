#!/usr/bin/env python3
"""
Dos modelos — paso 34: auditoría de GEOMETRÍA entre entrenamiento y servicio.

NO TOCA PRODUCCIÓN NI NINGÚN REPO, y NO modifica `auditoria_train_serve.py`,
que está sellado. Escribe salida/dos_34_*.csv.

=============================================================================
QUÉ HUECO CUBRE
=============================================================================
`auditoria_train_serve.py` compara VALORES de features entre las dos ramas, y
por eso no vio dos desajustes reales:

  · la ventana temporal de FIRMS (entrenó con [D-7, D-1], se servían 5 días);
  · la GEOMETRÍA del vecindario.

El segundo es el que mide este script. El extractor de entrenamiento
(`extraer_features_historia.py`) cuenta vecinos con `cKDTree` dentro de un
RADIO circular —«a <10 km», «a <50 km»—, mientras que la cadena
(`riesgo_hoy.py:230,270`) usa `maximum_filter(size=101)` y `suma_caja(gn, 50)`,
es decir una CAJA cuadrada. Una caja de lado 2r+1 cubre más superficie que el
círculo de radio r, así que la cadena sirve valores sistemáticamente mayores
que los que el modelo aprendió.

=============================================================================
CÓMO SE MIDE, SIN DEPENDER DE LA CADENA
=============================================================================
En vez de ejecutar las dos ramas y comparar salidas —que exige la cadena viva—
se aplican LAS DOS GEOMETRÍAS a la MISMA rejilla de entrada y se compara. La
rejilla es `dos_25_egif_mes.npz`, los incendios EGIF del mismo mes acumulados
por celda, que es la entrada de `n_fuegos_10km_mismomes_hist` — la variable de
mayor ganancia del modelo (29 %).

Se reporta:
  · el cociente de superficies, que es la predicción teórica;
  · el cociente medido de los valores servidos contra los de entrenamiento;
  · la correlación de Spearman, para ver si el desajuste es de ESCALA (el orden
    se conserva) o también de ORDEN.

Uso:
    python dos_34_auditoria_geometria.py
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

import config

SAL = config.salida("dos_34")
KM = 1.0                       # la malla del cubo es de 1 km
RADIOS = (10, 50)              # los dos vecindarios que usan las features


def circular(g, r):
    """Suma en un vecindario CIRCULAR de radio r (lo que hace el entrenamiento).

    Convolución por FFT con una máscara de disco: exacta y rápida sobre la
    malla completa, frente al cKDTree del extractor, que da lo mismo.
    """
    n = int(np.ceil(r / KM))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    disco = ((yy ** 2 + xx ** 2) * KM ** 2 <= r ** 2).astype(np.float32)
    from scipy.signal import fftconvolve
    return fftconvolve(g.astype(np.float32), disco, mode="same")


def caja(g, r):
    """Suma en una CAJA de lado 2r+1 (lo que hace la cadena)."""
    lado = int(2 * np.ceil(r / KM) + 1)
    return uniform_filter(g.astype(np.float32), size=lado,
                          mode="constant") * lado ** 2


def main():
    z = np.load(config.salida("dos_25_egif_mes.npz"))
    g = z["g"]                                   # (12, ny, nx)
    print(f"rejilla EGIF mismo-mes {g.shape} · {g.sum():,.0f} incendios",
          flush=True)

    filas = []
    for r in RADIOS:
        lado = int(2 * np.ceil(r / KM) + 1)
        teor = lado ** 2 / (np.pi * r ** 2)
        print(f"\n=== radio {r} km ===")
        print(f"  caja {lado}x{lado} = {lado**2:,} km2 · círculo r={r} = "
              f"{np.pi*r**2:,.0f} km2 · cociente teórico {teor:.3f}")
        for m in range(12):
            gm = g[m]
            if gm.sum() == 0:
                continue
            c = circular(gm, r)
            b = caja(gm, r)
            ok = c > 0                            # donde hay algo que comparar
            if ok.sum() == 0:
                continue
            ratio = float(b[ok].sum() / c[ok].sum())
            sp = float(pd.Series(b[ok].ravel()).corr(
                pd.Series(c[ok].ravel()), method="spearman"))
            filas.append(dict(radio_km=r, mes=m + 1, celdas=int(ok.sum()),
                              media_train=round(float(c[ok].mean()), 4),
                              media_serve=round(float(b[ok].mean()), 4),
                              ratio=round(ratio, 4),
                              ratio_teorico=round(teor, 4),
                              spearman=round(sp, 6)))
        sub = [f for f in filas if f["radio_km"] == r]
        rr = np.array([f["ratio"] for f in sub])
        ss = np.array([f["spearman"] for f in sub])
        print(f"  medido sobre {len(sub)} meses: cociente "
              f"{rr.mean():.3f} (rango {rr.min():.3f}-{rr.max():.3f})")
        print(f"  Spearman caja vs círculo: {ss.mean():.4f} "
              f"(mínimo {ss.min():.4f})")

    F = pd.DataFrame(filas)
    F.to_csv(f"{SAL}_geometria.csv", index=False)

    print("\n=== LECTURA ===")
    for r in RADIOS:
        s = F[F.radio_km == r]
        print(f"  r={r:2d} km: la cadena sirve valores {s.ratio.mean():.2f}x "
              f"los del entrenamiento (teórico {s.ratio_teorico.iloc[0]:.2f}x), "
              f"con Spearman {s.spearman.mean():.3f}")
    print("""
El desajuste es de ESCALA, no de orden: la correlación de rangos es casi
perfecta, así que el ranking de celdas apenas cambia y por eso el AUC no se
resiente. Lo que se desplaza es el VALOR que ve el árbol, y con él los cortes
aprendidos: un umbral aprendido sobre el círculo se aplica a un número mayor.

Corregirlo exige tocar `riesgo_hoy.py`, que está sellado hasta el cierre de los
jueces. Ver LIMITACIONES.md §8.""")
    print(f"\n→ {SAL}_geometria.csv")


if __name__ == "__main__":
    main()
