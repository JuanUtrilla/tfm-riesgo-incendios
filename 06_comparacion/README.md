# 6 · Los dos sistemas, sobre los mismos días y las mismas fuentes

## Qué pregunta responde

Si el rediseño aportó algo, tiene que verse comparando los dos pipelines en
igualdad de condiciones: mismos días, misma meteorología, misma verdad-terreno.
Comparar el 0,89 de test con el 0,64 de operación no vale — no son la misma
medida.

## Qué hay aquí

| Script | Qué compara |
|---|---|
| `dos_09_temporada2026.py` | La temporada 2026 día a día contra el área quemada de EFFIS |
| `dos_11_miteco.py` | Los mismos mapas, juzgados con los partes del MITECO |
| `comparar_julio2026.py` | Julio de 2026, 17 días con superficie quemada |
| `comparar_produccion.py` | Malla contra producción, el mismo día |
| `comparar_rankings.py` | Contra los rankings **sellados** por estación |
| `comparar_rankings_justo.py` | Previsión contra previsión: el cara a cara honesto |
| `dos_19_veredicto.py` | La figura del veredicto de la temporada |

## Los resultados

Temporada 2026, AUC contra superficie quemada de EFFIS:

| Sistema | Todos | Días grandes (11) | Sin FIRMS |
|---|---|---|---|
| producción (`xgb_v2`) | 0,647 | 0,692 | 0,611 |
| único (`donde_dia_effis`) | 0,752 | 0,746 | 0,744 |
| **único 1:10** | **0,759** | **0,763** | — |
| pareja | 0,705 | 0,763 | — |

Y con el ratio de negativos ya ajustado a 1:10, el número que se defiende:

> **único EFFIS 1:10 — AUC 0,759 frente a 0,647 de producción.
> Δ +0,112, IC95 [+0,074, +0,151]**, ganando el 76 % de los días. Los tres
> candidatos son significativos y ninguno roza el cero.

## Dos lecturas que conviene no saltarse

**FIRMS aporta mucho menos de lo que parecía.** Producción baja de 0,647 a
0,611 sin esa feature: +0,036, no el +0,126 que se publicó hasta el 31/08/2026
y que era un artefacto de la fuga (ver `docs/BITACORA.md`). Y en los once días
de megaincendio, producción con FIRMS (0,692) y sin ella (0,689) son
indistinguibles: la idea de que «producción acierta los grandes gracias a
FIRMS» no sobrevive al arreglo. Quien acierta los grandes es el único 1:10,
con 0,763.

**El cara a cara justo, previsión contra previsión, 19 días**: ranking sellado
de producción 0,568 · malla con el modelo de producción 0,547 · **único EFFIS
0,605**, ganando 14 de 19 días. Es el único experimento sin retrovisor para
nadie, y la primera vez que el único bate a producción POR ESTACIÓN, que es la
geometría en la que producción juega en casa.
