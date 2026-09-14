# ¿Mejora el r10 si se entrena con todos los años? No

Resultado **negativo** y **prerregistrado**. 05/09/2026.

## La pregunta

El `donde_dia_effis_r10` que se sirve está entrenado con la partición de
`dos_12_muestrear_effis.py`:

```python
SPLITS = {"train": (2015, 2022), "val": (2023, 2023), "test": (2024, 2024)}
```

O sea **ocho años**: 2023 y 2024 se reservan enteros. Eso era lo correcto
mientras el test fuese 2024. Pero existe una evaluación posterior e
independiente —el replay de 2025 y 2026, con datos que no están en el cubo—, así
que se puede entrenar con **todo 2015-2024** y medir fuera. Son dos años más y un
**+32 % de positivos** (14,861 → 19,552).

## El prerregistro

Escrito a las **11:30 UTC**, antes de ejecutar nada
(`calibracion_si/PRERREGISTRO_r10_todo.md`). Métrica primaria: **AUC medio por
día** sobre los días del replay con IFS, con IC bootstrap al 95 % de la
diferencia pareada. Criterio de éxito: **que el IC no toque el cero**.

Se declaró de antemano que las secundarias (percentil, percentil por hectáreas,
% de ha en el top 2 %) se reportan pero **no deciden**.

## Los dos controles

Sin estos dos, la comparación no valdría nada:

1. `r10_base` (mismo código, train 2015-2022) reproduce el
   `donde_dia_effis_r10.ubj` servido con **Δ máxima 0.00e+00** y Spearman
   **1.000000** sobre 200,000 filas.
2. Sobre los 224 días del replay, los cuatro modelos **no tocados** —`prod`,
   `unico`, `pareja`, `cuando`— dan **Δ máxima 0.00e+00** contra el replay
   original. Lo único que se movió fue el `r10`.

## El resultado

| | AUC medio por día |
|---|---|
| `r10_base` (2015-2022, el servido) | 0.7898 |
| `r10_todo` (2015-2024) | 0.7931 |
| **diferencia pareada** | **+0.0033 · IC95 [−0.0042, +0.0104]** |

**El IC cruza el cero: el criterio prerregistrado no se cumple.** Gana el 54.5 %
de los días, indistinguible del azar.

| temporada | base | todo | diferencia | gana | n |
|---|---|---|---|---|---|
| 2025 | 0.8073 | 0.8118 | +0.0045 [−0.0043, +0.0141] | 58 % | 138 |
| 2026 | 0.7618 | 0.7632 | +0.0014 [−0.0109, +0.0128] | 49 % | 86 |

Secundarias, todas cruzando el cero:

| métrica | base | todo | diferencia |
|---|---|---|---|
| percentil medio | 82.57 | 82.85 | +0.28 [−0.60, +1.17] |
| percentil ponderado por ha | 79.64 | 80.16 | +0.52 [−0.31, +1.29] |
| % ha en top 2 % | 15.53 | 15.92 | +0.38 [−1.53, +2.29] |
| % incendios >100 ha en top 2 % | 29.64 | 29.68 | +0.04 [−4.57, +4.84] |

Todas apuntan en la misma dirección (positiva, pequeña) y ninguna es
significativa: un efecto real pero minúsculo, o ninguno.

## Qué se concluye

1. **El modelo está saturado de datos.** Un 32 % más de positivos no mueve el
   rendimiento. Lo que limita al `r10` no es la cantidad de datos: es la señal
   disponible.
2. **La partición actual se mantiene sin coste.** `train 2015-2022 / val 2023 /
   test 2024` conserva un test honesto reservado y no cuesta rendimiento. No hay
   que elegir entre rigor y precisión.
3. **Blinda las cifras publicadas**: no son un artefacto de haber guardado datos.

Y el contexto que lo ordena todo, sobre los mismos 224 días:

| modelo | AUC |
|---|---|
| producción | 0.7104 |
| pareja | 0.7636 |
| único 1:3 | 0.7872 |
| **r10 (servido)** | **0.7898** |
| r10 con todos los años | 0.7931 |

La distancia r10 → producción (**+0.079**) es **24 veces** la ganancia de
reentrenar con todo (+0.0033). La señal está ahí, no en más años.

## Reproducción

```bash
source entorno.sh
python 05_iteracion2/57_ablaciones/dos_30_r10_todo.py     # entrena los dos
# y después, editando las rutas absolutas del cabecero:
python 06_comparacion/replay_temporada_todo.py --anio 2025 --condicion ifs
python 06_comparacion/replay_temporada_todo.py --anio 2026 --condicion ifs
```

53 min de replay (35 de 2025 + 18 de 2026), 224 días válidos de 250. Los 6
fallos por temporada son los días previos al 1-jun: el archivo IFS empieza el
25-may y el reanálisis debe llegar a D−7.

**Nada de esto tocó la cadena viva**: el modelo nuevo no se sirvió y
`donde_dia_effis_r10.ubj` no se modificó.
