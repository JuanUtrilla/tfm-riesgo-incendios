"""ERA5-Land horario de mayo a noviembre de 2025 desde el CDS, misma petición
que malla_02_descarga.pide_mes (área, variables, 24 horas, netcdf), pero con
salida FUERA de los repos: archivo_ifs/era5land_cds/era5land_YYYYMM.nc.
Reanudable: salta los meses ya bajados. Tres peticiones en paralelo (el CDS
las encola)."""
import os, sys, calendar, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/charredgem/Desktop/Master/TFM_fuego_malla")
import config
import cdsapi

SAL = "/home/charredgem/Desktop/Master/archivo_ifs/era5land_cds"
MESES = [(2025, m) for m in range(5, 12)]

def mes(am):
    anio, m = am
    ruta = f"{SAL}/era5land_{anio}{m:02d}.nc"
    if os.path.exists(ruta):
        print(f"  {anio}-{m:02d}: ya está", flush=True); return
    ndias = calendar.monthrange(anio, m)[1]
    t0 = time.time()
    print(f"  {anio}-{m:02d}: pidiendo al CDS ({ndias} días)...", flush=True)
    cdsapi.Client(quiet=True).retrieve("reanalysis-era5-land", {
        "variable": config.VARIABLES, "year": str(anio), "month": f"{m:02d}",
        "day": [f"{d:02d}" for d in range(1, ndias + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": config.AREA, "data_format": "netcdf",
    }, ruta + ".tmp")
    os.replace(ruta + ".tmp", ruta)
    open(ruta + ".cobertura", "w").write(str(ndias))
    print(f"  {anio}-{m:02d}: listo · {os.path.getsize(ruta)/1e6:.0f} MB · "
          f"{(time.time()-t0)/60:.0f} min", flush=True)

with ThreadPoolExecutor(3) as ex:
    list(ex.map(mes, MESES))
print("descarga completa", flush=True)
