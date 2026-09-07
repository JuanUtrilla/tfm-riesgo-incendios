# 5 · Segunda iteración: negativos del mismo día y etiqueta EFFIS

## Qué pregunta responde

«¿Cuál de las 498.530 celdas arde hoy?» — la que se hace en operación. Todo el
capítulo consiste en construir un conjunto de entrenamiento que haga
exactamente esa pregunta.

## Los dos cambios, y nada más

| | Iteración 1 | Iteración 2 |
|---|---|---|
| Negativos | la misma celda, otros días | **otras celdas, el mismo día** |
| Etiqueta | ignición EGIF | **superficie quemada EFFIS** (`is_fire` del cubo, celda ≥5 ha) |

Las features son las mismas 46, a propósito: si algo mejora, no se puede
atribuir a información nueva.

## El orden

```
51_celdas/      dos_01_celdas.py          la tabla por celda: censo de las 498.530
52_muestreo/    dos_12_muestrear_effis.py la maestra con etiqueta EFFIS
53_features/    dos_03_features.py        features del cubo + historia
54_analisis/    dos_04_analisis.py        las medidas de la fase 1
55_train/       dos_05_modelos.py         los seis diseños comparados (etiqueta EGIF)
                dos_10_donde_effis.py     el DÓNDE con etiqueta de superficie quemada
                dos_13_modelos_effis.py   los definitivos, test 2024
56_calibracion/ dos_14_cortes.py          cortes p30/p90/p98 -> BAJO/MODERADO/ALTO/EXTREMO
                dos_25 … dos_32           del orden a la probabilidad: el «si», la escala
                                          absoluta y lo que no se puede calibrar (README propio)
57_ablaciones/  dos_18_ratio.py           ¿cuántos negativos del mismo día?
                dos_08_verano.py          ¿entrenar solo con verano?
                dos_17_capa_quemado.py    ¿ayuda saber que una celda ya ardió?
                dos_30_r10_todo.py        el r10 entrenado con todos los años (prerregistrado)
                dos_34_auditoria_geometria.py  caja frente a círculo en el filtro de vecinos
```

## Los tres modelos que salen de aquí

- **único** (`donde_dia_effis`): las 46 features, muestreo del mismo día,
  etiqueta EFFIS. **0,752** sobre la temporada 2026 (**0,759** con ratio 1:10).
- **dónde** (`donde_effis_c`): una fila por celda, sin meteorología. Responde
  «¿esta celda es propensa?» y su mapa es el mismo todos los días.
- **cuándo**: solo las 29 dinámicas, diseño misma-celda-otro-día. Responde
  «¿hoy es peligroso?». La **pareja** es el producto de los dos: 0,705.

## Las ablaciones, que son casi todas negativas

| Pregunta | Respuesta |
|---|---|
| ¿Entrenar solo con verano mejora el producto de verano? | **No**, empeora el DÓNDE |
| ¿Ayuda una capa de «ya quemado»? | **No**: −0,02. La etiqueta EFFIS penaliza bajar el riesgo en la cicatriz |
| ¿Cuántos negativos por positivo? | El 1:3 heredado no estaba justificado. Óptimo en **1:10-1:30**; desde 1:60 empeora |
| ¿Reentrenar el r10 con todos los años (hasta 2024)? | Prerregistrado el 05/09. No cambia el orden; ver `docs/R10_TODOS_LOS_ANIOS.md` |
| ¿Es el filtro de vecinos una caja o un círculo? | Caja en entrenamiento, círculo al servir: desajuste de escala (1,39×), no de orden. `LIMITACIONES.md` §8 |

El ratio merece leerse con cuidado, porque enseña a no cantar victoria: la
ganancia en **AUC medio** (+0,006 a +0,017) es del orden del **ruido del sorteo
de negativos** (0,005, medido re-sorteando la misma receta). Lo que sí mejora de
forma consistente es el **fuego grande**: percentil ponderado por hectáreas
81,7 → 87,6-87,9. Esa es la métrica que se reporta.

## Del orden a la probabilidad (`56_calibracion/`)

El modelo ordena bien, pero un mapa que pinta de rojo el 2 % superior de cada
día se lee como «estas celdas van a arder», y eso es falso los doce meses del
año. `dos_25`-`dos_32` pasan los cinco modelos por los diez años del cubo y
sacan tres cosas: un **aviso de día** («si») con umbral absoluto, que tiene
habilidad en invierno-primavera (BSS +0,30) y **no en verano** (IC cruza el
cero); una **escala absoluta** para el mapa (EXTREMO = 1 de cada 800 celdas
quema, lift 35×); y la constatación, medida tres veces, de que la probabilidad
por celda **no es calibrable en valor absoluto** en un año récord como 2022.
Todo en el README de la carpeta y en `docs/LIMITACIONES.md` §9-10.

## Ejecutar

Estos scripts leen el cubo día a día (29 GB) y los datasets del disco externo
(`TFM_USB`). **No se pueden ejecutar con `muestras/`**, que solo lleva las capas
estáticas. Lo que sí corre con la muestra es la rama de servicio del capítulo 7.
