#!/usr/bin/env python3
"""
Dos modelos, paso 11: los mismos mapas diarios de 2026, evaluados con los
partes de MITECO en vez de con EFFIS.

No toca producción. Lee los mapas que guarda `dos_09` (expansión) y los 52
incidentes geocodificados de `dataset/dashboard_miteco.json` en los datos
externos (TFM_DATOS; solo lectura; geocodificación al centro del municipio,
hecha por `validar_miteco.py`). Escribe salida/dos_11_miteco.{json,csv}.

Por qué: EFFIS es MODIS (perímetros, >=5 ha, latencia) y favorece a quien
acierta los megaincendios. MITECO registra los incendios en los que el Estado
desplegó medios, sin depender de satélite: otro sesgo, otra referencia.
Etiqueta: celdas a <= RADIO km del centro del municipio, el día de primera
aparición del incidente (proxy de ignición). El radio absorbe el error de
geocodificar al centro del municipio. Se reportan 10 y 25 km.
"""
import glob, json, os
import numpy as np, pandas as pd, xarray as xr
from pyproj import Transformer
import config, config_expansion as ce
from comparar_julio2026 import auc

fu = pd.DataFrame(json.load(open(f"{config.FUENTE}/dataset/dashboard_miteco.json"))["mapa"]["fuegos"])
fu["d"] = pd.to_datetime(fu["d"])
ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
es = ds["is_spain"].values.astype(bool); xs, ys = ds["x"].values, ds["y"].values
X, Y = np.meshgrid(xs, ys); cx, cy = X[es], Y[es]
tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
fx, fy = tr.transform(fu["lon"].values, fu["lat"].values)
R = {}; filas = []
for radio in (10, 25):
    for d, g in fu.groupby("d"):
        f = f"{ce.DATASET}/mapas_2026/{d.date()}.npz"
        if not os.path.exists(f):
            continue
        m = np.load(f)
        i = g.index.values
        dist = np.min(np.hypot(cx[:, None] - fx[i][None], cy[:, None] - fy[i][None]), 1)
        q = dist <= radio * 1000
        fila = {"radio": radio, "fecha": str(d.date()), "n_incidentes": len(g), "n_celdas": int(q.sum())}
        for k in m.files:
            A = m[k]; ok = np.isfinite(A)
            fila[f"auc_{k}"] = round(auc(A[ok & q], A[ok & ~q]), 4)
        filas.append(fila)
df = pd.DataFrame(filas); df.to_csv(config.salida("dos_11_miteco.csv"), index=False)
rng = np.random.default_rng(0)
for radio in (10, 25):
    s = df[df.radio == radio]
    print(f"\n== MITECO · radio {radio} km · {len(s)} días · {s.n_incidentes.sum()} incidentes ==")
    R[radio] = {}
    for k in sorted(c[4:] for c in s.columns if c.startswith("auc_")):
        a = s[f"auc_{k}"].values; dd = a - s["auc_prod"].values
        b = dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)
        R[radio][k] = {"auc": float(a.mean()), "dif": float(dd.mean()),
                       "ic95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                       "gana": float((dd > 0).mean())}
        print(f"  {k:24s} AUC {a.mean():.3f} · Δprod {dd.mean():+.3f} [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}] · gana {(dd>0).mean()*100:.0f}%")
json.dump(R, open(config.salida("dos_11_miteco.json"), "w"), indent=1)
