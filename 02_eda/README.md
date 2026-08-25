# 2 · Análisis exploratorio

> Solo sobre el split de **entrenamiento (2015-2018)**, para que ninguna
> decisión de modelado se tome mirando el test.

## Qué pregunta responde

Cómo se distribuyen las variables, qué relación tienen con el fuego y qué
decisiones de ingeniería de features quedan justificadas antes de entrenar.

## Qué hay aquí

`eda_dataset.py` — el EDA del dataset de la iteración 1, restringido al split
de entrenamiento.

## Lo que se dejó fuera del repositorio

El bloque de visión por satélite —Sentinel-2, D-Fire, los timelapses y las
historias interactivas de los incendios de Sotalvo, Luna y Ponteareas— fue
trabajo real de exploración, pero no entra en el hilo de la memoria y arrastra
decenas de MB de imágenes. Sigue en el repositorio original `TFM_fuego`.

## El hallazgo que sobrevive al capítulo

Las cinco features de satélite (NDVI, LAI, SWI, LST) **no aportan señal**:
eliminarlas cuesta lo mismo que congelarlas por climatología, −0,004 de AUC.
Está medido en `04_bisagra/43_no_es_el_train_serve/ablacion_proxies_operativos.py`.
