# Limitaciones conocidas

Lo que no cierra, lo que se dejó roto a propósito y lo que no se puede afirmar
con los datos que hay. Está escrito para poder copiarse casi tal cual al
capítulo de limitaciones de la memoria.

---

## 1. Producción sirve una versión con un defecto conocido, y es deliberado

Desde el 18/08/2026 se sabe que `ranking_diario.py` usa `np.minimum` donde
debería usar `np.fmin` al combinar las dos fuentes de viento. La diferencia
importa: **`minimum` propaga NaN**, así que a toda estación a la que le falte el
dato de racha el viento le sale NaN, y el cálculo del FWI trata el viento NaN
igual que viento 0.

| | |
|---|---|
| Efecto | FWI 35,7 donde debería ser 58,0 (con 15 km/h de viento) |
| Alcance | **16,3 %** de las estaciones-día; 128 de 858 estaciones a lo largo de la serie |
| Estado | **Sin corregir en producción, a propósito** |

**Por qué no se arregla:** el valor del veredicto de septiembre depende de que
producción sirva durante toda la temporada exactamente la misma versión con la
que empezó. Arreglarlo a mitad de temporada rompería la comparación sellada, que
es justo lo que da fuerza al resultado. El arreglo está escrito y se aplicará al
cerrar la temporada.

**Cómo afecta a la lectura de los resultados:** el baseline contra el que compiten
los candidatos está **penalizado** en un 16 % de sus filas. Toda ventaja medida
sobre producción debe leerse teniendo esto en cuenta.

---

## 2. La comparación de AUC entre fuentes meteorológicas nunca cerró

El Experimento B tenía tres patas. Dos son firmes (el desplazamiento de features
entre ERA5-Land y AEMET, y la saturación del percentil por cruce de fuentes). La
tercera —comparar el AUC de los tres brazos bajo un único pipeline— **falló el
control**: la reconstrucción del brazo *AEMET observada* da 0,589 donde
producción publicó 0,702, sobre los mismos 18 días y las mismas estaciones.

El efecto que se quería medir era **−0,012**, nueve veces menor que ese error de
instrumento. Lo acotado:

- **No es ruido**: perturbar con ruido de la misma magnitud cuesta −0,006. La
  desviación es sistemática.
- **No son huecos de datos**: restringiendo a ventanas de 80 días completas
  (7.898 de 20.445 filas) la discrepancia sigue en −0,113.
- **Está en la cola alta del FWI**: las entradas coinciden casi perfectamente
  (`t2m_max` r=0,998), pero el FWI se desvía más que sus propias entradas
  (r=0,941). Con 1 % de prevalencia, el AUC lo decide el techo del ranking.
- **Sospecha sin confirmar**: producción evalúa el último día completo *de cada
  estación*, que no es el mismo día para todas, mientras la reconstrucción fuerza
  a todas al día objetivo.

**Ningún número de esa comparación se reporta como si valiera.**

---

## 3. Tamaños de muestra: el veredicto todavía no está cerrado

| Juez | Acumulado a 25/08/2026 | Qué falta |
|---|---|---|
| EFFIS | 1 día puntuado | Los 45 días de desfase de cartografiado |
| MITECO | 3 días / 6 incidentes | Acumular temporada |
| Por estación | 1 día / 688 estaciones / 3 positivas | Acumular temporada |

El veredicto acumulado dice literalmente *«mejor de media, pero con 17 días no
se distingue del ruido»*, y así debe citarse hasta el cierre de septiembre. El
resultado con intervalo que no toca el cero es el retrospectivo sobre la
temporada 2026 completa (Δ +0,112, IC95 [+0,074, +0,151]); desde que se
corrigió la fuga de FIRMS del 31/08/2026 también lo son el único 1:3
(+0,105 [+0,067, +0,146]) y la pareja (+0,058 [+0,027, +0,090]). Sigue siendo
un resultado RETROSPECTIVO, con reanálisis para todos: es una cota superior de
lo que daría la operación, no la operación.

---

## 4. El ruido del sorteo de negativos es del orden de las diferencias medidas

Re-sortear los negativos con la misma receta mueve el AUC medio **0,005**. Varias
de las diferencias entre configuraciones caen dentro de ese margen, y por eso no
se reportan como mejoras: en la ablación del ratio, lo que sí se sostiene es el
percentil ponderado por hectáreas (81,7 → 87,6), no el AUC medio.

---

## 5. Limitaciones de las etiquetas

- **EGIF** registra igniciones (punto y fecha de inicio), no superficie quemada.
  Entrenar con ella y validar con área quemada fue el origen del problema de
  especificación de la iteración 1.
- **EFFIS** solo cartografía incendios por encima de cierto tamaño y tarda ~45
  días. Penaliza además bajar el riesgo en una cicatriz reciente, lo que invalidó
  la capa de «ya quemado».
- **MITECO** es rápido y oficial, pero de grano municipal y sin superficie fina;
  captura entre el 45 % y el 81 % de los EFFIS grandes.
- **FIRMS** detecta focos activos, no incendios: sin filtrar por FRP y confianza,
  la etiqueta es ruido (AUC 0,561 frente a 0,715 filtrada).

---

## 6. Features que no aportan

Las cinco features satelitales (NDVI, LAI, SWI, LST) cuestan lo mismo eliminadas
que congeladas por climatología (−0,004). **No aportan señal ni siendo reales.**
Se mantienen documentadas porque el resultado negativo es informativo, no porque
el modelo las necesite.

---

## 7. Alcance geográfico y temporal

Península peninsular únicamente: 498.530 celdas. Quedan fuera Baleares, Canarias,
Ceuta y Melilla. El entrenamiento cubre 2015-2020 (etiquetas EGIF consolidadas) y
la validación en operación, la temporada de 2026.
