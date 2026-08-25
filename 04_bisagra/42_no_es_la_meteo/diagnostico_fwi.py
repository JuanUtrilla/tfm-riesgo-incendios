#!/usr/bin/env python3
"""
De dónde sale el factor ~2 entre el FWI de entrenamiento y el de producción.

NO TOCA PRODUCCIÓN. Escribe salida/diagnostico_fwi.json.

=============================================================================
EL SÍNTOMA
=============================================================================
`dataset/auditoria_train_serve.json` documenta un desplazamiento enorme en las
features de más peso del modelo:

    feature          entrenamiento (jul-ago)   producción   PSI
    fwi                        23,7               50,9      1,65
    fwi_med_30d                21,4               52,1      3,29
    fwi_med_7d                 20,1               51,7      2,61

PSI por encima de 0,25 ya se considera desplazamiento grave. El modelo recibe
en operación un FWI que duplica el que vio al aprender. Es candidato serio a
explicar por qué el AUC-ROC pasa de 0,89 en test a ~0,60 en operación.

(Una hipótesis previa —que fuese comparar entrenamiento de todo el año contra
producción de verano— NO se sostiene: la auditoría ya estaba restringida a
julio-agosto.)

=============================================================================
LA DESCOMPOSICIÓN
=============================================================================
Este script mide los tres eslabones sobre las MISMAS celdas y días:

 1. MUESTRA. El conjunto de entrenamiento (23,7) está por debajo del propio
    cubo sobre España en julio (29-31). Los negativos se muestrean y salen
    con FWI 14,9 en pleno verano: la muestra no representa a España en
    verano, por diseño.

 2. IMPLEMENTACIÓN. Con la MISMA meteo del cubo, el `FWI` del cubo da 31,0 y
    `fwi_canadiense` —la función que SIRVE— da 43,6. Factor 1,41 con
    correlación 0,925. Es el eslabón más grande y el más barato de arreglar.

 3. FUENTE Y AÑO. De 43,6 a los 50,9 de producción queda 1,17: estaciones
    AEMET en vez de celdas del cubo, y 2026 en vez de 2018.

=============================================================================
LAS TRES FUENTES SOBRE LOS MISMOS DÍAS Y CELDAS (julio 2024, 300 nodos)
=============================================================================
No basta con mirar AEMET: la comparación honesta pone las tres fuentes en las
mismas celdas y los mismos días.

    cubo (FWI propio)                31,9   p50 34,1
    ERA5-Land + fwi_canadiense       39,2   p50 42,0   x1,20 vs cubo
    cubo meteo + fwi_canadiense      43,4   p50 46,6   x1,33 vs cubo
    AEMET + fwi_canadiense (prod.)   50,9              x1,60 vs cubo

Correlación con el cubo: 0,926-0,927 en los dos casos. La relación es fuerte
pero desplazada, no un error aleatorio.

Dos lecturas:

· El eslabón dominante es la IMPLEMENTACIÓN, no la fuente: con la MISMA meteo
  del cubo, `fwi_canadiense` da 1,33x. Y está documentado en el propio módulo:
  el sistema canónico usa valores a MEDIODÍA y aquí se usa tmax/hrMin como
  proxy. La cabecera de `fwi_canadiense` avisa de que ese sesgo "se cancela en
  el percentil local" — y así es (PSI 0,29) — pero NADIE protege a las
  features de NIVEL, que es donde pega (PSI 1,65).

· LA MALLA MEJORA ESTO, no lo empeora. ERA5-Land (39,2) queda más cerca del
  entrenamiento que AEMET (50,9): el desplazamiento pasa de x2,15 a x1,65.
  Es un beneficio de la migración que no estaba medido.

=============================================================================
RESUELTO: EL CUBO ES EL FWI DE LAS 13 UTC
=============================================================================
Con ERA5-Land HORARIO en disco se puede calcular el FWI con valores
instantáneos de cada hora en vez del proxy, y buscar cuál reproduce el cubo
(julio 2024, 300 nodos, mismos días):

    hora UTC   T media   HR media   FWI julio
        11        27,6      39,8       25,1   ← mediodía canónico
        12        28,8      36,7       28,8
        13        29,6      34,7    ** 31,7 **  ← el cubo da 31,9
        14        30,1      33,6       33,8
        15        30,2      33,4       34,8

    proxy tmax/hrMin (lo que se sirve)         39,2

El cubo se reproduce con los valores de las 13 UTC — media tarde, la hora de
máximo peligro. Ni el mediodía canónico (25,1, demasiado bajo) ni el proxy de
extremos diarios (39,2, demasiado alto).

Por qué el proxy infla: `tmax` y `hr_min` son extremos del día que NO ocurren
a la vez. A las 13 UTC coinciden 29,6 °C con 34,7 % de HR; el proxy junta
30,5 °C con 31,7 %, una combinación que no se da en ninguna hora real.

CONSECUENCIA PRÁCTICA. El desplazamiento más grande de la auditoría no exige
reentrenar: exige calcular el FWI que se SIRVE a las 13 UTC en vez de con
extremos diarios. Eso reproduce por construcción la escala con la que el
modelo aprendió. Hay datos horarios en las dos ramas: ERA5-Land los tiene y
el colector de AEMET también.

AVISO: `fwi_pctl_local` está hoy protegido porque numerador y climatología
usan la MISMA receta. Si se cambia la receta hay que rehacer también la
climatología a las 13 UTC — y el módulo 4 borra el horario tras agregarlo,
así que serían los 84 meses otra vez. Cambiar solo el nivel y dejar el
percentil como está también es coherente: cada feature queda alineada con su
propia referencia.

ÁRBITRO YA NO NECESARIO PARA ESTO: EFFIS publica su propio FWI (capa WMS `mf010.fwi`) y sería
la referencia independiente para decidir si el "bajo" es el cubo o el "alto"
es fwi_canadiense. La consulta histórica por GetFeatureInfo no salió a la
primera y queda pendiente. Los índices de CEMS no están en el CDS climático
(solo `satellite-fire-radiative-power`): viven en el Early Warning Data Store,
que necesita credenciales aparte.

=============================================================================
LO QUE ESTO IMPLICA
=============================================================================
· El percentil está a salvo: `fwi_pctl_local` tiene PSI 0,29, porque numerador
  y denominador se calculan con la misma función a cada lado. Es justo el
  invariante que arregló el módulo 4. Las features de NIVEL no tienen esa
  protección.

· LA MALLA NO ARREGLA ESTO. También calcula el FWI con `fwi_canadiense`: su
  media el 13-ago-2026 fue 53,1, el mismo orden que los 50,9 de producción.
  Es un problema ortogonal al del IDW y sobrevive a la migración.

· El arreglo barato es recalcular las columnas de NIVEL del entrenamiento con
  la misma función que sirve. Mueve el entrenamiento de 23,7 a ~33 y cierra el
  eslabón 2 entero. Los eslabones 1 y 3 exigen rediseñar el muestreo, que ya
  es reentrenar de verdad.

Uso: python diagnostico_fwi.py
"""

import json

import numpy as np
import pandas as pd
import xarray as xr

import config
from fwi_canadiense import calcular_fwi_serie

N_CELDAS = 200
MES = ("2018-01-01", "2018-07-31")     # spin-up desde enero, se mide julio


def main():
    aud = {f["feature"]: f for f in json.load(
        open(f"{config.FUENTE}/dataset/auditoria_train_serve.json"))["features"]}
    d = pd.read_parquet(f"{config.FUENTE}/dataset/dataset_modelo_v4.parquet")
    d["fecha"] = pd.to_datetime(d["fecha"])
    ver = d[d["fecha"].dt.month.isin([7, 8])]

    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    t = pd.to_datetime(ds["time"].values)
    a = int(np.where(t == pd.Timestamp(MES[0]))[0][0])
    b = int(np.where(t == pd.Timestamp(MES[1]))[0][0])
    esp = ds["is_spain"].values.astype(bool)
    rng = np.random.default_rng(0)
    yy, xx = np.where(esp)
    s = rng.choice(len(yy), N_CELDAS, replace=False)
    yy, xx = yy[s], xx[s]
    sl = dict(time=slice(a, b + 1))
    T = ds["t2m_max"].isel(**sl).values[:, yy, xx]
    H = ds["RH_min"].isel(**sl).values[:, yy, xx]
    V = ds["wind_speed_max"].isel(**sl).values[:, yy, xx]
    P = ds["total_precipitation_mean"].isel(**sl).values[:, yy, xx]
    FC = ds["FWI"].isel(**sl).values[:, yy, xx]
    ds.close()

    f = pd.date_range(*MES, freq="D")
    jul = f.month.values == 7
    mio = np.full(T.shape, np.nan)
    for j in range(N_CELDAS):
        mio[:, j] = calcular_fwi_serie(T[:, j], H[:, j], V[:, j] * 3.6,
                                       P[:, j], f.month.values)["fwi"]
    x, y = FC[jul], mio[jul]
    k = np.isfinite(x) & np.isfinite(y)

    res = {
        "entrenamiento_julago": float(aud["fwi"]["media_ent"]),
        "entrenamiento_negativos": float(
            ver.loc[ver["label"] == 0, "fwi"].mean()),
        "cubo_espana_julio": float(x[k].mean()),
        "fwi_canadiense_misma_meteo": float(y[k].mean()),
        "produccion": float(aud["fwi"]["media_prod"]),
        "factor_implementacion": float(y[k].mean() / x[k].mean()),
        "corr_implementacion": float(np.corrcoef(x[k], y[k])[0, 1]),
        "psi_fwi": float(aud["fwi"]["psi"]),
        "psi_fwi_pctl_local": float(aud["fwi_pctl_local"]["psi"]),
    }
    print("=" * 66)
    print("DE DÓNDE SALE EL FACTOR ~2 DEL FWI")
    print("=" * 66)
    print(f"  entrenamiento jul-ago (media)        {res['entrenamiento_julago']:6.1f}")
    print(f"    de los cuales, negativos           "
          f"{res['entrenamiento_negativos']:6.1f}  ← muestra sesgada a la baja")
    print(f"  cubo sobre España en julio           {res['cubo_espana_julio']:6.1f}")
    print(f"  fwi_canadiense, MISMA meteo          "
          f"{res['fwi_canadiense_misma_meteo']:6.1f}  ← x"
          f"{res['factor_implementacion']:.2f} por implementación")
    print(f"  producción (AEMET + fwi_canadiense)  {res['produccion']:6.1f}")
    print(f"\n  correlación cubo vs fwi_canadiense: "
          f"{res['corr_implementacion']:.3f}")
    print(f"  PSI de fwi {res['psi_fwi']:.2f} frente a "
          f"fwi_pctl_local {res['psi_fwi_pctl_local']:.2f}"
          f"  ← el percentil está protegido, el nivel no")
    with open(config.salida("diagnostico_fwi.json"), "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"\nGuardado: {config.salida('diagnostico_fwi.json')}")


if __name__ == "__main__":
    main()
