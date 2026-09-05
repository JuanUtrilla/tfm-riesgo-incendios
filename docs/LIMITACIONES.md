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

Re-sortear los negativos con la misma receta mueve el AUC medio **0,004**
(0,752 → 0,748 en 2026; 0,809 → 0,804 en test 2024). Varias
de las diferencias entre configuraciones caen dentro de ese margen, y por eso no
se reportan como mejoras: en la ablación del ratio, lo que sí se sostiene es el
percentil ponderado por hectáreas (79,8 → 81,9 con ratio 1:10, cifras limpias
del 31/08/2026; la corrida con fuga decía 81,7 → 87,6), no el AUC medio.

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

## 8. Desajuste train/serve: filtro de caja contra radio circular (05/09/2026)

`riesgo_hoy.py` calcula tres features con un filtro de **caja** donde el
extractor de entrenamiento (`extraer_features_historia.py`, que usa `cKDTree`
con «a <10 km» y «a <50 km») usa **radio circular**. Medido sobre las filas
`eval_dia` del banco de `dos_13`:

| feature | entrenamiento | servicio | ratio | corr | n |
|---|---|---|---|---|---|
| `n_fuegos_10km_mismomes_hist` | 2,80 | 4,06 | **1,50** | 0,971 | 245.166 |
| `frp_max_50km_7d` | 4,79 | 5,84 | **1,54** | 0,910 | 58.308 |
| `n_detec_50km_7d` | 2,63 | 3,41 | **1,36** | 0,952 | 58.308 |

Cuadra con la geometría: caja 21×21 = 441 km² contra círculo r=10 km = 314 km²
(1,40); caja 101×101 = 10.201 km² contra círculo r=50 km = 7.854 km² (1,30).

**La cadena sirve tres variables un 36-54 % mayores que las que los modelos
vieron al entrenar.** Afecta a producción, único, r10 y a la rama `cuando` de la
pareja.

Se documenta y **no se corrige antes del cierre de los jueces**, por la misma
razón que el bug del viento (§1): tocar `riesgo_hoy.py` a mitad de temporada
rompe la serie sellada. La calibración de `56_calibracion` replica el camino de
SERVICIO a propósito, así que sus números son válidos para lo que se sirve.

**Confirmado por una vía independiente** (`dos_34_auditoria_geometria.py`,
05/09/2026). En lugar de ejecutar las dos ramas, se aplican las **dos
geometrías a la misma rejilla de entrada** (la de EGIF mismo-mes, la variable de
mayor ganancia del modelo). Los cocientes medidos clavan los teóricos:

| radio | caja | círculo | cociente teórico | **medido** | Spearman |
|---|---|---|---|---|---|
| 10 km | 441 km² | 314 km² | 1,404 | **1,391** | 0,924 |
| 50 km | 10.201 km² | 7.854 km² | 1,299 | **1,300** | 0,980 |

Dos consecuencias. La primera: la geometría explica por sí sola un factor
1,39× del 1,50× observado en `n_fuegos_10km_mismomes_hist`; el resto procede de
otras diferencias entre las dos ramas. La segunda, y más importante: el
desajuste es **de escala, no de orden** —la correlación de rangos es 0,92-0,98—,
lo que explica que el AUC apenas se resienta. Lo que se desplaza es el valor que
ve el árbol, y con él la posición relativa de los cortes aprendidos.

`auditoria_train_serve.py` no detecta esto porque compara valores, no geometrías
ni ventanas temporales, y está sellado. `dos_34_auditoria_geometria.py` cubre el
hueco sin modificarlo.

## 9. La probabilidad por celda no es calibrable en valor absoluto (05/09/2026)

Con calibración estratificada por mes (`dos_29_calibra_estacional.py`) el error
típico en la punta es de **1,8×**, y en julio parecía llegar a 7×. La primera
redacción de esta sección atribuía ese 7× a que la ventana de calibración
2015-2021 no contenía ningún año extremo, y proponía como arreglo recalibrar
sobre **ventana móvil**.

**Se probó (`dos_31_ventana_movil.py`) y la hipótesis es falsa.** Calibrando con
los K años anteriores a cada año evaluado, el error en jun-sep va **al revés**
de lo previsto:

| modelo | 3 años | 5 años | 7 años | todo lo anterior |
|---|---|---|---|---|
| **r10** | 0,947 | 0,576 | 0,487 | **0,453** |
| prod | 0,790 | 0,665 | 0,623 | **0,517** |

Cuanta menos historia, peor. En julio de 2022 el error empeora de 6,4× a 8,9×.

La causa real es otra. La tasa observada en el top 0,2 % de **julio**, año a año:

| 2015 | 2018 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|
| 1,13 % | 0,03 % | 0,80 % | **12,76 %** | **0,00 %** | 0,07 % |

Mediana **0,53 %**. El julio de 2022 está 24 veces por encima; el de 2023 es
cero. Ninguna curva puntuación→probabilidad puede seguir eso, porque **la
variación no vive en la puntuación de la celda: vive en la intensidad del día y
del año**. El «7× de julio» era, además, un artefacto de promediar 2022 con el
resto: contra la mediana de julio, la calibración acierta dentro de un factor 2.

**El producto de dos capas tampoco lo arregla, y se ha probado.** La
descomposición `P(día grande) × P(celda | día grande)`, con el primer factor de
`dos_26` y el segundo ajustado solo con días grandes
(`dos_32_producto_dos_capas.py`), **empeora**: en jun-sep el error típico pasa de
1,7× (incondicional) a 3,4×. El producto hereda la infraconfianza de la capa
«si» y la multiplica por la de la capa «dónde»; los dos errores se componen en
lugar de cancelarse.

**La causa real es que 2022 está fuera del envolvente histórico.** Tasa en la
punta de julio condicionada a día grande: máximo 2,05 % en los años de
calibración (2015-2021), contra **14,95 % en julio de 2022**, siete veces por
encima. Ninguna calibración ajustada con 2015-2021 puede producir ese número, y
por eso fallan igual los tres intentos: por mes (`dos_29`), ventana móvil
(`dos_31`) y producto de dos capas (`dos_32`). No es un problema de esquema de
calibración: **es extrapolación, no calibración**.

Lo que se sostiene, y es lo que debe ir a la memoria:

- La probabilidad por celda está **bien calibrada en condiciones típicas**:
  contra la mediana de julio (0,53 %) el modelo dice ~0,9 %, dentro de un
  factor 2.
- La calibración es válida para el **orden** y la **magnitud típica**
  («extremo es 1 de cada 1.000», techo 1,6 %).
- **No puede anticipar una temporada récord**, y en un año extremo la
  probabilidad servida se queda corta en un orden de magnitud. Es una limitación
  del planteamiento —calibrar sobre el pasado—, no un defecto corregible con más
  ingeniería, y conviene declararla porque es operativamente relevante.

## 10. El bootstrap iid estrecha los intervalos un 35 % (05/09/2026)

Todos los IC de esta memoria usan bootstrap percentil **iid sobre días**
(`dos_19_veredicto.py:76` y siguientes), que supone días independientes. No lo
son: un incendio grande dura varios días y el tiempo persiste.

Medido en `dos_26_calibra_si.py`, que calcula ambos: el bootstrap por **bloques
móviles de 14 días** da intervalos de SEDI **1,35× más anchos** (mediana sobre
las 20 filas de test).

No invalida nada publicado —el orden entre modelos no cambia y los IC de
invierno siguen excluyendo el cero también con bloques—, pero **los intervalos
del resto de la memoria deben leerse como una cota inferior de la
incertidumbre**, y en los casos al filo la conclusión honesta es que el
intervalo toca el cero.
