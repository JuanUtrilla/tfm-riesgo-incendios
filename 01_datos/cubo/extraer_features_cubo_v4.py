#!/usr/bin/env python3
"""
Runner v4 del extractor de features de IberFire.

NO duplica el extractor ni lo modifica: importa `extraer_features_cubo` y
redirige sus tres rutas a las del reproceso v4. Toda la lógica de extracción
—definiciones de features, ventanas anti-leakage, climatología local, recorrido
por bloques de chunk— es literalmente la misma que produjo v1/v2/v3, que es
justo lo que se necesita: los cambios de métrica entre v3 y v4 deben ser
atribuibles a los datos nuevos, no a un extractor distinto.

⚠️ `DIR_PARTES` DEBE ser un directorio propio. Las partes de v1 están indexadas
por `id_muestra`, y los id_muestra de la muestra v4 son otros: reutilizar el
directorio antiguo mezclaría dos numeraciones y el merge final saldría mal (o,
peor, saldría bien y con las filas cruzadas).

La climatología local sigue siendo 2008-2014: son años previos a TODO el dataset
(2015-2022), así que sigue sin haber fuga temporal al añadir 2021 y 2022.

Resumible: si se corta, se relanza y continúa por el primer bloque sin fichero.

Uso:
    /home/charredgem/miniconda3/envs/tfm_fuego/bin/python extraer_features_cubo_v4.py
"""

import extraer_features_cubo as m

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"

m.RUTA_MAESTRA = f"{DIR}/dataset/muestra_maestra_v4.parquet"
m.DIR_PARTES = f"{DIR}/dataset/_features_parts_v4"
m.RUTA_SALIDA = f"{DIR}/dataset/features_cubo_v4.parquet"

# --------------------------------------------------------------------------- #
# popdens: el cubo solo llega a 2020
# --------------------------------------------------------------------------- #
# IberFire trae popdens_2008 … popdens_2020. Al entrar 2021 y 2022 al dataset,
# `extraer_estaticas` pedía `popdens_2021` y reventaba con KeyError.
#
# La solución NO es un apaño: se arrastra el último año disponible (2020), que es
# EXACTAMENTE lo que ya hace el sistema en producción — `mapa_riesgo_hoy.py:66`
# lee `popdens_2020` fijo para predecir en 2026. Arrastrarlo aquí ALINEA el
# entrenamiento con la inferencia en vez de separarlos; usar el valor real de
# 2021-22 (si existiera) habría creado un desajuste entrenamiento/inferencia.
# La densidad de población varía ~1%/año: el error de arrastrar dos años es
# despreciable frente al beneficio de que ambos lados usen la misma capa.
#
# Se implementa recortando el año SOLO para la extracción de estáticas. No afecta
# al corte CLC (2021 y 2022 siguen siendo ≥2018 tras el recorte, luego CLC_2018).
ANIO_MAX_POPDENS = 2020
_estaticas_original = m.extraer_estaticas


def extraer_estaticas_v4(ds, df):
    n = int((df["anio"] > ANIO_MAX_POPDENS).sum())
    if n:
        print(f"[popdens] {n} filas de años >{ANIO_MAX_POPDENS} usan "
              f"popdens_{ANIO_MAX_POPDENS} (igual que producción)", flush=True)
        df = df.assign(anio=df["anio"].clip(upper=ANIO_MAX_POPDENS))
    return _estaticas_original(ds, df)


m.extraer_estaticas = extraer_estaticas_v4

if __name__ == "__main__":
    print(f"maestra : {m.RUTA_MAESTRA}")
    print(f"partes  : {m.DIR_PARTES}")
    print(f"salida  : {m.RUTA_SALIDA}")
    print(f"clim    : {m.ANIOS_CLIM}\n", flush=True)
    m.main()
