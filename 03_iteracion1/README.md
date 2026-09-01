# 3 · Primera iteración: etiqueta EGIF y muestreo caso-control

> **Este capítulo no es un error, es el grupo de control.** Sin él no hay forma
> de demostrar que el rediseño hacía falta ni de cuantificar lo que aportó.

## Qué pregunta responde

«¿Es hoy un día peligroso en esta celda?» — y ese enunciado, que parece una
manera razonable de plantear el problema, es exactamente lo que el capítulo 4
demuestra que no coincide con la pregunta operativa.

## El diseño

- **Positivo**: una ignición registrada en EGIF, en su celda y su fecha.
- **Negativo**: **la misma celda en otros días** (caso-control temporal).
- **Features**: 46 en el modelo servido (la lista canónica son 50 — ver abajo).
- **Modelo**: XGBoost.

## El orden

```
31_muestreo/   muestrear_dataset.py -> ensamblar_dataset.py     (v1)
               muestrear_dataset_v4.py -> ensamblar_dataset_v4.py (EGIF consolidado, Civio 09/07/2026)
32_features/   extraer_features_historia.py                     (historial EGIF, entorno FIRMS, rayos)
33_train/      entrenar_modelo.py        v1/v2, test 2020
               entrenar_modelo_v3.py     reajuste con 2015-2020 completo
               entrenar_modelo_v4.py     EGIF consolidado, test 2022  <- el 0,89 de la memoria
               tuning_optuna.py          40 trials: +0,007. El ROI estaba en las features
               modelo_nativo_estacion.py entrenar donde se sirve
34_produccion/ la cadena que se sirvió a diario (ver abajo)
35_ablaciones/ ablacion_features.py      FWI absoluto vs percentil local: +0,042
               ablacion_v3_firms.py      sustituir la autorregresiva EGIF caducada por FIRMS
```

## Los números de test

| Métrica | Valor |
|---|---|
| AUC-ROC v4, test 2022 | **0,8906** [0,8644, 0,9136] |
| AUC-ROC v1/v2, test 2020 | 0,931 |
| AUC-PR v1/v2, test 2020 | 0,843 (azar 0,263) |
| GroupKFold espacial 100 km | 0,848 ± 0,028 — sin colapso |
| Etiquetas barajadas | 0,231 ≈ azar → pipeline sin fugas |

Guarda ese **0,89**: es el número que la memoria anuncia en la primera página
para avisar de que va a caerse.

## La puesta en producción

`34_produccion/` es lo que se sirvió de verdad desde julio de 2026, y sigue
sirviéndose: ranking nacional por estación (`ranking_diario.py`) y mapas D0/D1
(`mapa_diario.py`), publicados por GitHub Actions en el repositorio del colector
de AEMET. El modelo que corre ahí es **`xgb_v2_prototipo`** — ni v3 ni v4, que
se entrenaron y se midieron pero nunca se desplegaron, para no romper la serie
sellada a mitad de temporada.

### De dónde sale `xgb_v2_prototipo` (46 features, no 50)

Ningún script de este repositorio lo entrena, y no es un descuido de la copia:
fue un **reentrenamiento interactivo del 15-jul-2026** que no se versionó. Su
racional está escrito y cuantificado en [`MODELO_B_BITACORA.md`](MODELO_B_BITACORA.md)
§16: las cuatro autorregresivas intra-celda acumulativas (`n_fuegos_1km_hist`,
`n_fuegos_1km_90d`, `n_fuegos_10km_90d`, `n_fuegos_10km_365d`) arrastran un
artefacto del muestreo caso-control con celda fija —el incendio del positivo
entra en el historial de los negativos posteriores de su propia celda, y el
modelo aprende orden temporal (+0,92 log-odds por `n_fuegos_1km_hist=0` en
pleno Madrid urbano)—, así que se reentrenó v1 sin ellas: AUC-PR 0,828 frente
a 0,843, «coste pequeño, modelo honesto para producción».
[`33_train/verificar_v2.py`](33_train/verificar_v2.py) comprueba con las
muestras que las 46 del JSON servido son exactamente las 50 canónicas
(`features.FULL`) menos esas cuatro, y que su orden coincide con el que lleva
dentro el `.ubj`. Es, además, el primer aviso de la tesis: el diseño del
muestreo ya se estaba colando en el modelo.

## Y entonces se midió

Servido a diario contra superficie quemada real, el 0,89 se convirtió en
**0,56-0,64**. Eso es el capítulo 4.
