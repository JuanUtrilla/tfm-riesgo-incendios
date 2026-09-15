#!/usr/bin/env python3
"""
Tarea diaria (cron): ranking nacional de riesgo + archivo histórico.

Cada ejecución:
1. Calcula el ranking de las ~705 estaciones (tiempo_real.evaluar_todas,
   forzando recálculo) para el último día completo.
2. Lo archiva en prototipo/historico_predicciones/ranking_<fecha>.parquet.
   Ese archivo acumulado guarda las predicciones emitidas antes de los
   incendios, comparables a final del verano con FIRMS (la validación del
   trabajo es el replay de 2025-2026; esto es un registro de seguimiento).
3. Escribe un resumen (y las estaciones en EXTREMO) en el log.

Instalación (crontab -e):
  30 9 * * * /home/charredgem/Desktop/Master/TFM_fuego/prototipo/cron_ranking.sh
"""

import os
import shutil

import pandas as pd

DIR = "/home/charredgem/Desktop/Master/TFM_fuego"
DIR_HIST = f"{DIR}/prototipo/historico_predicciones"


def main():
    import tiempo_real as trm
    os.makedirs(DIR_HIST, exist_ok=True)
    df = trm.evaluar_todas(forzar=True)
    dia = df["fecha"].mode()[0]
    destino = f"{DIR_HIST}/ranking_{dia}.parquet"
    df.to_parquet(destino, index=False)

    niveles = df["nivel"].value_counts().to_dict()
    extremo = df[df["nivel"] == "EXTREMO"]
    print(f"[{pd.Timestamp.now()}] día {dia}: {len(df)} estaciones · {niveles}")
    if len(extremo):
        print("  ⚠️ EXTREMO en: " + "; ".join(
            f"{r.nombre} ({r.prob:.2f})" for r in extremo.head(12).itertuples()))
    print(f"  archivado: {destino} "
          f"(histórico: {len(os.listdir(DIR_HIST))} días)")


if __name__ == "__main__":
    main()
