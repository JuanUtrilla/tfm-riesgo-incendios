#!/usr/bin/env python3
"""
Verificación del módulo 2: ¿está bien la agregación horaria → diaria de CDS?

No sobrescribe nada. Solo imprime y escribe dataset/malla_02_verificacion.json.

Qué compara: el diario derivado de ERA5-Land horario (CDS) contra el diario que
sirve Open-Meteo, en los puntos de 250 estaciones AEMET y sobre el mismo
periodo (1-may → 13-ago 2026).

Lo que debe salir (el test tiene dos filos):

· `tmax` y `hr_min` deben cuadrar casi exacto. Open-Meteo toma esas dos de
  ERA5-Land, la misma fuente que CDS. Si no cuadran, el fallo está en este
  script: o en la agregación max/min o en la derivación de HR por Magnus desde
  el punto de rocío.

· `prec` y `viento_max` deben salir más bajos en CDS. Open-Meteo los toma de
  ERA5 (31 km), no de ERA5-Land, y en §6 se midió que por eso llueve el doble
  que el cubo (1,38 vs 0,69 mm/día) y sopla de más (4,12 vs 3,63 m/s). Aquí CDS
  sirve ERA5-Land puro en las cuatro, que es justo el arreglo que §6 predijo.
  Si `prec` de CDS se acerca al 0,69 del cubo, la predicción se cumple.

Ojo: el fallo que este test busca por encima de todo es el del acumulado. Si la
regla de `total_precipitation` fuera una suma de horarios en vez del acumulado
diario, `prec` saldría ~24× más alta, y eso se vería aquí de inmediato.

Uso: python malla_02_verificar.py
"""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import requests
import xarray as xr

RAIZ = Path(__file__).resolve().parents[2]
DIR = os.environ.get("TFM_DATOS", str(RAIZ / "datos"))
CACHE = os.path.join(os.environ.get("TFM_SALIDA", str(RAIZ / "salida")), "cache")
INI, FIN = "2026-05-01", "2026-08-13"
N = 250


def main():
    ds = xr.open_dataset(f"{DIR}/malla_data/era5land_diario.nc")
    est = pd.read_parquet(f"{DIR}/prototipo/estaciones_prototipo.parquet") \
        .drop_duplicates("idema").head(N)

    ruta = f"{CACHE}/om_verif_{N}.parquet"
    if os.path.exists(ruta):
        om = pd.read_parquet(ruta)
    else:
        print("  pidiendo diarios a Open-Meteo (1 petición)...", flush=True)
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=dict(latitude=",".join(f"{v:.4f}" for v in est.lat),
                        longitude=",".join(f"{v:.4f}" for v in est.lon),
                        start_date=INI, end_date=FIN,
                        daily=("temperature_2m_max,relative_humidity_2m_min,"
                               "wind_speed_10m_max,precipitation_sum"),
                        models="era5_seamless", timezone="UTC",
                        wind_speed_unit="ms"), timeout=300).json()
        om = pd.concat([
            pd.DataFrame({"idema": i, "fecha": pd.to_datetime(b["daily"]["time"]),
                          "tmax": b["daily"]["temperature_2m_max"],
                          "hr_min": b["daily"]["relative_humidity_2m_min"],
                          "viento_max": b["daily"]["wind_speed_10m_max"],
                          "prec": b["daily"]["precipitation_sum"]})
            for i, b in zip(est.idema, r if isinstance(r, list) else [r])],
            ignore_index=True)
        om.to_parquet(ruta, index=False)

    # CDS en el nodo más cercano a cada estación
    filas = []
    for _, e in est.iterrows():
        p = ds.sel(latitude=e.lat, longitude=e.lon, method="nearest")
        filas.append(pd.DataFrame({
            "idema": e.idema, "fecha": pd.to_datetime(p.fecha.values),
            **{v: p[v].values for v in ["tmax", "hr_min", "viento_max", "prec"]}}))
    cds = pd.concat(filas, ignore_index=True)

    d = cds.merge(om, on=["idema", "fecha"], suffixes=("_cds", "_om"))
    print(f"\nfilas comparadas: {len(d):,} "
          f"({d.idema.nunique()} estaciones × {d.fecha.nunique()} días)\n")
    print("=" * 76)
    print(f"{'variable':<14}{'CDS':>9}{'Open-Meteo':>12}{'sesgo':>9}"
          f"{'corr':>8}   veredicto")
    print("=" * 76)

    esperado = {
        "tmax": ("cuadrar (ambas ERA5-Land)", lambda c, o, r: abs(c - o) < 0.6 and r > 0.98),
        "hr_min": ("cuadrar (ambas ERA5-Land)", lambda c, o, r: abs(c - o) < 3.0 and r > 0.95),
        "viento_max": ("CDS menor (ERA5-Land vs ERA5 31 km)", lambda c, o, r: c < o),
        "prec": ("CDS menor (ERA5-Land vs ERA5 31 km)", lambda c, o, r: c < o),
    }
    res, todo_ok = {}, True
    for v, (nota, test) in esperado.items():
        a, b = d[f"{v}_cds"].values, d[f"{v}_om"].values
        ok = np.isfinite(a) & np.isfinite(b)
        ca, cb = a[ok].mean(), b[ok].mean()
        r = np.corrcoef(a[ok], b[ok])[0, 1]
        pasa = bool(test(ca, cb, r))
        todo_ok &= pasa
        print(f"{v:<14}{ca:>9.2f}{cb:>12.2f}{ca-cb:>+9.2f}{r:>8.3f}   "
              f"{'OK ' if pasa else '!! '}{nota}")
        res[v] = dict(cds=float(ca), open_meteo=float(cb), sesgo=float(ca - cb),
                      corr=float(r), pasa=pasa, esperado=nota)

    print("\n" + "-" * 76)
    p_cds, p_om = res["prec"]["cds"], res["prec"]["open_meteo"]
    print(f"Control del acumulado: prec CDS {p_cds:.2f} mm/día. "
          f"Si se hubieran sumado los\nhorarios en vez de tomar el acumulado "
          f"diario, saldría ~{p_om*24:.0f} mm/día.")
    print(f"Referencia del cubo en §6 (jun-sep 2024): 0,69 mm/día · "
          f"viento 3,63 m/s.\nOpen-Meteo daba 1,38 y 4,12 — el doble de lluvia. "
          f"Si CDS se acerca al cubo,\nse cumple la predicción de §6.")
    print("-" * 76)
    print(f"\n{'TODAS LAS COMPROBACIONES PASAN' if todo_ok else 'HAY COMPROBACIONES QUE FALLAN'}")

    with open(f"{DIR}/dataset/malla_02_verificacion.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
