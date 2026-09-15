#!/usr/bin/env python3
"""
Dos modelos, paso 10: el dónde con etiqueta de superficie quemada (EFFIS).

No toca producción. Lee expansión; escribe modelo en expansión y
salida/dos_10_mapa_donde_effis.npz.

Por qué
-------
`dos_09` midió la temporada 2026 sobre perímetros EFFIS: el mapa dónde
entrenado con igniciones EGIF se hunde los días de megaincendio (Luesia, La
Mierla, Navaluenga, Niebla, Riglos: su decil superior contiene el 0-7 % de
las celdas quemadas). El EGIF cuenta igniciones, que se concentran en el
noroeste; la superficie quemada se concentra en otro sitio. Aquí la etiqueta
es «≥1 celda-día `is_fire` (EFFIS ≥5 ha) en 2008-2020», la misma fuente con
la que se compara, y se mira en 2026 si el mapa cambia lo que importa.
Prevalencia por celda ~1,8 %. Para 2019-2020 esta etiqueta está contaminada
(usa esos años): solo vale para 2021+ y para la temporada 2026.
"""
import numpy as np, pandas as pd
import config, config_expansion as ce
from dos_05_modelos import FEATS_DONDE, HIST, entrena

cel = pd.read_parquet(f"{ce.DATASET}/celdas.parquet")
y = ((cel["effis_0814"] + cel["effis_1520"]) > 0).astype(int).values
yv = (cel["effis_2124"] > 0).astype(int).values
print(f"celdas con EFFIS 2008-20: {y.sum():,} ({y.mean()*100:.2f} %)")
out = {"iy": cel["iy"].values, "ix": cel["ix"].values}
for nombre, feats in [("donde_effis", FEATS_DONDE + HIST),
                      ("donde_effis_mix", FEATS_DONDE + HIST + ["egif_1518_10km"])]:
    X = cel[feats].values.astype(np.float32)
    m = entrena(X, y, X, yv, nombre)
    out[nombre] = m.predict_proba(X)[:, 1]
    imp = pd.Series(m.feature_importances_, feats).sort_values(ascending=False)
    print(f"  top {nombre}: {imp.head(6).round(3).to_dict()}")
# ¿se parecen los dos mapas?
from scipy.stats import spearmanr
md = np.load(config.salida("dos_05_mapa_donde.npz"))
print("Spearman donde_egif ~ donde_effis:", round(spearmanr(md["p_donde_hist"], out["donde_effis"]).correlation, 3))
np.savez_compressed(config.salida("dos_10_mapa_donde_effis.npz"), **out)
