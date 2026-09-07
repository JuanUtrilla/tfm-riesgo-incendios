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

## El replay de 2025 y 2026: el veredicto final

Lo anterior compara los dos sistemas sobre 2026 con **reanálisis para todos**:
una cota superior. El replay hace la comparación como se haría en operación:
día a día, alimentando a los seis modelos con **la pasada IFS disponible la
víspera** (archivada en `archivo_ifs/`, fuera del repo) y juzgando con los
perímetros EFFIS de ese día. Dos temporadas, 250 días, protocolo prerregistrado
en `docs/TRAZABILIDAD.md` (commit `ed7931f`, 02/09/2026 18:10) **antes** de
ejecutar nada. Desde el 06/09/2026 es **el veredicto de la memoria**; los
jueces en vivo del capítulo 7 se citan como validación operativa en curso.

| Script | Qué hace |
|---|---|
| `replay_verdad.py` | La verdad EFFIS por día (celda de 1 km, primer día de cada perímetro) para 2025 y 2026 |
| `replay_dia.py` | Un día: reanálisis truncado a D−7, IFS de archivo, FIRMS ≤ D−1, puntúa los seis modelos. ~15 s |
| `replay_temporada.py` | El bucle de temporada, reanudable (salta los días con `.npz`) |
| `replay_veredicto.py` | AUC medio por día, Δ vs producción, IC95 bootstrap, gana/pierde, top 2 % → `docs/REPLAY_VEREDICTO.md` |
| `replay_si.py` | La pregunta del «si» con umbral p98 (versión previa a la calibración de `56_calibracion`) → `docs/REPLAY_SI.md` |
| `replay_precision.py` | Precisión exacta en la punta y lift → `docs/REPLAY_PRECISION.md` (a posteriori) |
| `replay_radio.py` | Tolerancia espacial de la verdad a 2/4/6 km → `docs/REPLAY_RADIO.md` (a posteriori) |
| `replay_cobertura.py` | Cobertura de incendios con el top dilatado → `docs/REPLAY_COBERTURA.md` (a posteriori) |
| `replay_figuras.py` | Las figuras del veredicto |
| `replay_dia_todo.py`, `replay_temporada_todo.py` | Copias parcheadas para el r10 entrenado con todos los años (`docs/R10_TODOS_LOS_ANIOS.md`) |
| `verificar_replay.py` | El verificador del 01/09: un día de replay contra el mapa operativo del Release |
| `dos_33_anomalia_punta.py` | Por qué producción cubría más hectáreas en la punta de 2026: es un solo día (23-jul) |

**Resultado** (AUC medio por día, condición IFS):

| Temporada | Días con fuego | producción | único | **r10** | pareja | Δ r10 (IC95) | gana |
|---|---|---|---|---|---|---|---|
| 2025 | 138 | 0,743 | 0,806 | **0,807** | 0,794 | +0,064 [+0,027, +0,101] | 60 % |
| 2026 | 86 | 0,658 | 0,757 | **0,762** | 0,716 | +0,104 [+0,065, +0,144] | 67 % |

El coste de la previsión (IFS − reanálisis) es de −0,003 para todos, IC que
cruza el cero. Tablas completas, con los seis modelos y las cuatro condiciones,
en `docs/REPLAY_VEREDICTO.md`.

**Límite, dicho sin disimulo**: los mapas se generaron el 02/09/2026 con la
previsión de cada víspera, pero se generaron *a posteriori*, no sellados día a
día. Está en `docs/LIMITACIONES.md` §3.

## Ejecutar

`dos_09`, `dos_11` y los `comparar_*` leen los mapas de la temporada del disco
externo (`TFM_USB`). El replay necesita además `archivo_ifs/` (pasadas IFS
2025-2026, ERA5-Land 2025, FIRMS, EFFIS 2025), que no está en el repo por
tamaño; `docs/PROCEDENCIA.md` da sus md5. Nada de este capítulo corre con
`muestras/`.
