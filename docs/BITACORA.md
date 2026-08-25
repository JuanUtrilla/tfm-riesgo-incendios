# Bitácora: del primer modelo en producción al veredicto de la temporada

> Esta bitácora es el **orden cronológico** del trabajo, y existe precisamente
> para que la memoria no tenga que contarlo así. La memoria sigue el orden
> lógico ([`00_marco/README.md`](../00_marco/README.md)); aquí queda la traza
> real, con lo que se probó y no entró, y con las conclusiones que hubo que
> retirar. Cubre del **28 de julio al 25 de agosto de 2026**.
>
> Regla de la casa: **un resultado negativo medido es un resultado.** Casi todo
> lo que hay aquí abajo lo es.

---

## Punto de partida — 28/07/2026

El primer modelo (Modelo B, XGBoost sobre 46 features del cubo IberFire,
etiqueta EGIF, muestreo caso-control) llevaba desde mediados de julio sirviendo
a diario en el repositorio del colector de AEMET: ranking nacional por estación
y mapas D0/D1, publicados por GitHub Actions.

Su primera validación operativa, con **2-3 días de previsiones selladas por
commit**, daba esto:

| Serie | AUC-ROC |
|---|---|
| D0 sellada | 0,622 |
| D1 sellada | 0,629 |
| ranking (día cerrado) | 0,685 |
| retro (hindcast) | 0,594 |

Y una conclusión que parecía buena: **el modelo bate al FWI**. Guárdala, porque
tres semanas después hubo que retirarla.

---

## 1. La grieta: 11/08/2026

### 1.1 Una cuarta fuente de verdad

Hasta entonces la verdad-terreno eran detecciones FIRMS y perímetros EFFIS. Se
añadieron los **partes diarios oficiales de MITECO**, archivados a diario por
un repositorio hermano. Su virtud no es la precisión, es la **latencia**: se
publican al día siguiente, frente a los ~45 días que tarda EFFIS en cartografiar
un incendio. Eso convierte a MITECO en el juez *rápido* de todo lo que vino
después.

### 1.2 El resultado incómodo

Se rehízo la evaluación operativa con baselines e intervalos de confianza. Con
etiqueta FIRMS filtrada (FRP≥20 MW, confianza no baja) el modelo daba un
respetable **0,715**… pero:

| Baseline | AUC D0 |
|---|---|
| FWI percentil | 0,573 |
| FWI bruto | 0,580 |
| **Persistencia** (`n_detec_50km_7d`) | **0,723** |

El modelo **empataba con la persistencia**, un baseline trivial que además es
una de sus propias features. Solo ganaba en D1. Primera señal seria de que el
número de test no medía lo que parecía medir.

---

## 2. Tres semanas de refuerzo: 11-18/08/2026

Antes de aceptar que el problema fuera de diseño, se agotó la vía del ajuste.
Todo esto se hizo, y nada de ello cerró el hueco:

| Trabajo | Qué era | Resultado |
|---|---|---|
| **v3** | Reajuste con 2015-2020 completo | Mejora marginal; dos resultados negativos útiles |
| **v4** | Reproceso con el EGIF consolidado (Civio, 09/07/2026) y test sobre 2022 | Sin cambio de conclusión |
| **Modelo nativo de estación** | Entrenar directamente donde se sirve, en vez de en el cubo | No resuelve el hueco |
| **Auditoría train/serve** | Las 46 features, una por una: ¿le llega a producción lo mismo que vio al entrenar? | Localiza desajustes, ninguno suficiente |
| **Verificación de fuentes meteo** | ¿Qué fuente predice mejor el tiempo? | Insumo para lo que vendría después |

---

## 3. El día que se cayó la conclusión: 18/08/2026

### 3.1 La validación, rehecha con 20 días

Se redescargaron los perímetros EFFIS de la temporada (1.871 incendios,
301.996 ha) y la muestra sellada pasó de 3 días a **20**. Con bootstrap de
**días completos** —no de estaciones, que dentro de un día están correlacionadas
e inflan la precisión artificialmente—:

| Serie | Días | n | Modelo | FWI percentil | Diferencia (IC95) |
|---|---|---|---|---|---|
| D0 sellada | 20 | 13.661 | 0,642 | **0,702** | −0,060 [−0,149, +0,020] |
| D1 sellada | 20 | 13.647 | 0,618 | **0,681** | −0,062 [−0,162, +0,020] |
| ranking (día cerrado) | 25 | 17.173 | **0,707** | 0,668 | +0,039 [−0,047, +0,116] |
| retro (hindcast) | 21 | 13.906 | 0,596 | **0,627** | −0,031 [−0,096, +0,037] |

**Ninguna diferencia es significativa**: el IC95 cruza el cero en las cuatro
series. Con 20 días no hay potencia para separar 0,64 de 0,70, y afirmar lo
contrario en cualquiera de las dos direcciones sería inventar.

Y la retirada explícita de la conclusión del 28/07: al añadir días, el baseline
**pasó de 0,540 a 0,702** mientras el modelo apenas se movió (0,62 → 0,64). La
estimación del modelo era razonable; la del baseline, no. **«El modelo bate al
FWI» era un artefacto de muestra pequeña.**

### 3.2 Descartado: no son las features congeladas

En producción, las features de satélite van por climatología mensual congelada
y los rayos van a cero. Parecía el sospechoso obvio. Se midió sustituyendo cada
dato real por su proxy operativo, en el entorno de entrenamiento donde sí
existen las dos versiones:

| Proxy operativo | Δ AUC en test 2020 |
|---|---|
| Satélite (NDVI/LAI/SWI/LST) → climatología mensual | −0,004 |
| Rayos → 0 | −0,000 (coherente: +0,002) |
| **Los dos a la vez, como hoy en producción** | **−0,0045** |

Cuesta 0,0045. **No explica nada.** De regalo, un hallazgo que sí sirve:
eliminar las cinco features satelitales cuesta lo mismo que congelarlas, o sea
que **no aportan nada ni siendo reales**.

### 3.3 Medido: el desplazamiento de fuente (Experimento B)

En entrenamiento la meteo sale de ERA5-Land dentro del cubo; en producción, de
estación AEMET y forecast municipal. Se descargó ERA5-Land de 2026 (12-abr →
14-ago, 281 MB) y se recalcularon las features con el mismo pipeline, para
comparar las mismas features desde dos fuentes:

| Feature | r | sesgo (ERA5 − AEMET) | MAE |
|---|---|---|---|
| `t2m_max` | 0,899 | −1,13 °C | 1,75 |
| `rh_min` | 0,903 | +1,41 % | 5,52 |
| `precip_30d` | 0,554 | **+8,00 mm** | 10,24 |
| `dias_sin_lluvia` | 0,555 | **−7,91 días** | 17,71 |
| `fwi` | 0,822 | **−10,04** | 13,20 |
| `fwi_pctl_local` | 0,523 | **−16,50 pctl** | 23,84 |

Temperatura y humedad coinciden bien. Lo que no coincide es la **precipitación**:
ERA5-Land llueve 8 mm más en 30 días (sesgo húmedo conocido del reanálisis,
llovizna difusa) y de ahí se propaga en cadena todo lo demás — menos días sin
lluvia, combustible más húmedo, FWI 10 puntos más bajo, percentil 16 puntos más
bajo.

### 3.4 El defecto operativo más grave localizado

Rastreando de dónde salía la climatología del percentil apareció un cruce de
fuentes: `clim_fwi/*.npz` se construyó con la meteo **del cubo** (ERA5-Land),
mientras que en producción el numerador del percentil es un **FWI de AEMET**.
Numerador y referencia de fuentes distintas, y la feature satura:

| | Entrenamiento (jul-ago) | Producción |
|---|---|---|
| Filas en percentil ≥99,9 | 1,0 % | **12,6 %** |
| Mediana nacional del percentil | 54 | **87** |

### 3.5 Lo que quedó abierto, y se dice

La tercera pata del Experimento B —comparar el AUC entre brazos— **no cerró**.
Al reconstruir el brazo *AEMET observada* con código propio para tener los tres
brazos bajo un único pipeline, el control falló: sobre los mismos 18 días y las
mismas estaciones, la reconstrucción da **0,589** donde producción publicó
**0,702**. El efecto que se quería medir era −0,012: **nueve veces menor que el
error del instrumento**.

Lo que sí se acotó: no es ruido (perturbar con ruido de la misma magnitud cuesta
solo −0,006, luego la desviación es sistemática); no son los huecos de datos
(restringir a ventanas completas deja la discrepancia idéntica, −0,113); y está
**concentrada en la cola alta del FWI**, que con 1 % de prevalencia es justo
donde se decide el AUC.

Ningún número de esa comparación se reporta como si valiera. Ver
[`LIMITACIONES.md`](LIMITACIONES.md).

---

## 4. Servir con la misma malla del entrenamiento: 19-20/08/2026

Si la fuente meteorológica desplaza las features, la respuesta natural es dejar
de servir con AEMET + IDW y servir con **la misma ERA5-Land con la que se
entrenó**. De ahí sale todo el pipeline de malla:

- **Nodos y descarga** ERA5-Land, sin interpolación IDW.
- **Previsión IFS** (ECMWF vía Open-Meteo) para D0 y D1, donde el reanálisis
  todavía no existe: medida su saturación, corregido el mapeo IFS→ERA5-Land
  sobre 1.100 nodos, y reajustada la rama de previsión.
- **Climatología de FWI a 7 años**, que baja la saturación a la mitad en los
  días extremos.
- **Comparaciones serias**: 17 días de julio de 2026 con área quemada EFFIS,
  cara a cara previsión contra previsión sobre los 18 días sellados, y contra
  los rankings sellados del prototipo por estación.
- Dos verificaciones que había que hacer antes de tocar nada: **el FWI del cubo
  son las 13 UTC** (estable en tres meses, no es un proxy de extremos) y de
  dónde salía el **factor ~2** entre el FWI de entrenamiento y el de producción.

---

## 5. El giro: 21/08/2026

Aquí el diagnóstico cambia de naturaleza. No es la meteorología, no es el
train/serve, no son las features: es **la especificación**.

Una pasada por el cubo para medir la etiqueta real, y tres diseños de muestreo
comparados, dejan ver lo que la métrica de desarrollo escondía:

- el **98,8 %** de las celdas del conjunto v1 tenía exactamente un 25 % de
  positivos, de modo que el modelo no podía aprender qué celda es más peligrosa
  que otra: **solo qué día lo es**;
- el **AUC caso-control de los dos diseños es idéntico (0,92)** y el **AUC
  dentro del día los separa (0,744 frente a 0,828)**. La métrica con la que se
  desarrolló era literalmente ciega a la diferencia.

El rediseño que sale de ahí: negativos del **mismo día**, y etiqueta **EFFIS**
—la misma con la que se valida— en vez de igniciones EGIF. Resultado sobre la
temporada 2026: **0,773 dentro del día frente a 0,737** de producción.

Ese mismo día se montó la infraestructura para juzgarlo en operación sin
depender de que el portátil esté encendido: la cadena diaria en **GitHub
Actions**, los candidatos sirviendo en paralelo a producción, y los **tres
jueces** — EFFIS (45 días de desfase), MITECO (parte del día siguiente) y el
juez por estación contra el ranking sellado del hermano, por deploy key de
lectura.

---

## 6. Ablaciones: 21-23/08/2026

Con el diseño nuevo en pie, se fue a por los supuestos heredados que nadie había
justificado. Tres de los cuatro son resultados negativos:

| Ablación | Pregunta | Resultado |
|---|---|---|
| **Solo verano** | ¿Entrenar solo con verano mejora el producto de verano? | **No.** Empeora el DÓNDE |
| **Capa «ya quemado»** | ¿Ayuda saber que una celda ardió hace poco? | **No.** −0,02 con la verdad completa; solo vale como overlay |
| **Vegetación consumida** | La cicatriz existe (LAI −40 %), ¿se puede usar? | La etiqueta EFFIS **penaliza** bajar el riesgo; la máscara empeora −0,02 |
| **Ratio de negativos** | El 1:3 venía heredado sin justificación | Óptimo en **1:10-1:30**; a partir de 1:60 empeora |

El ratio merece una nota, porque es donde se aprendió a leer los números con
cuidado: la ganancia en **AUC medio** (+0,006 a +0,017) es del orden del **ruido
del sorteo de negativos** (0,005, medido re-sorteando la misma receta). Lo que
sí mejora de forma consistente es el **fuego grande**: el percentil ponderado
por hectáreas pasa de **81,7 a 87,6-87,9**. Esa es la métrica que importa
operativamente, y la que se reporta.

---

## 7. Dónde está hoy: 24-25/08/2026

La cadena diaria corre sola y en verde (21 pasos, dos pasadas al día). Estado de
los tres jueces:

| Juez | Acumulado | Resultado |
|---|---|---|
| **MITECO** | 3 días / 6 incidentes | único **0,807** > pareja 0,771 > malla 0,598 |
| **EFFIS** | 1 día (4 celdas, 122 ha) | pareja 90,6 · único 73,5 · malla 39,2 (percentil) |
| **Por estación** | 1 día, 688 estaciones | pareja 0,607 · único 0,600 · malla 0,572 · producción 0,507 |

Y el veredicto de temporada, que es el número que se defiende:

> **Único con etiqueta EFFIS y ratio 1:10 — AUC 0,785 frente a 0,736 de
> producción. Δ +0,049, IC95 [+0,014, +0,085].** Es el único candidato cuyo
> intervalo de confianza **no toca el cero**.

El veredicto acumulado sigue siendo prudente a propósito: *«mejor de media, pero
con 17 días no se distingue del ruido»*. El cierre está previsto para mediados
de septiembre, cuando venzan los 45 días de desfase de EFFIS.

---

## Lo que esta bitácora deja para la memoria

1. **El primer modelo no es un error, es el grupo de control.** Sin sus 0,89 de
   test y sus 0,64 de operación no hay forma de demostrar que el rediseño hacía
   falta, ni de cuantificar lo que aportó.
2. **La métrica de desarrollo puede ser ciega.** 0,92 de AUC caso-control en dos
   diseños que difieren en 0,084 de AUC dentro del día es el hallazgo central
   del trabajo.
3. **Los descartes valen tanto como el hallazgo.** Que no sea el satélite
   congelado (−0,0045), que no sean los rayos (−0,000) y que no sea el
   train/serve es lo que hace creíble el diagnóstico de especificación.
4. **Lo que no cerró se dice.** El control fallido del Experimento B y el bug
   sellado en producción van a [`LIMITACIONES.md`](LIMITACIONES.md), no debajo
   de la alfombra.
