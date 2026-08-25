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

| Sistema | Todos | Días grandes | Sin FIRMS |
|---|---|---|---|
| producción (`xgb_v2`) | 0,737 | 0,883 | **0,611** |
| único (`donde_dia_effis`) | **0,773** | 0,794 | 0,744 |
| pareja | 0,754 | 0,830 | — |

Y con el ratio de negativos ya ajustado a 1:10, el número que se defiende:

> **único EFFIS 1:10 — AUC 0,785 frente a 0,736 de producción.
> Δ +0,049, IC95 [+0,014, +0,085]**, el único candidato cuyo intervalo no toca
> el cero.

## Dos lecturas que conviene no saltarse

**Producción depende de FIRMS.** Sin esa feature cae a 0,611 (y a 0,520 con el
juez MITECO), mientras el único aguanta en 0,744. FIRMS son focos ya activos:
apoyarse en ella es, en parte, predecir el incendio que ya está ardiendo.

**El cara a cara justo, previsión contra previsión, 19 días**: ranking sellado
de producción 0,568 · malla con el modelo de producción 0,602 · **único EFFIS
0,625**, ganando 16 de 19 días.
