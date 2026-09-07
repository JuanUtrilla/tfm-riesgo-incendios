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
temporada 2026: **0,752 dentro del día frente a 0,647** de producción. (Estas
cifras son las corregidas el 31/08/2026; hasta esa fecha se publicaron 0,773 y
0,737, infladas por una fuga de futuro en la ventana de FIRMS —ver la entrada
del 31/08— que favorecía a producción.)

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
del sorteo de negativos** (0,004, medido re-sorteando la misma receta:
0,752 frente a 0,748 en 2026, `dos_09_ventana7.log`; 0,809 frente a 0,804 en
test 2024, `dos_18_ratio.json`). Lo que
sí mejora de forma consistente es el **fuego grande**: el percentil ponderado
por hectáreas pasa de **79,8 (1:3) a 81,9 (1:10)** y 80,6 (1:30)
(`dos_09_ventana7.log`, cifras limpias del 31/08/2026; la corrida original
del 23/08 decía 81,7 → 87,6-87,9, inflada por la fuga de FIRMS de la entrada
§8). Esa es la métrica que importa operativamente, y la que se reporta.

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

> **Único con etiqueta EFFIS y ratio 1:10 — AUC 0,759 frente a 0,647 de
> producción. Δ +0,112, IC95 [+0,074, +0,151].** Gana el 76 % de los días y su
> intervalo de confianza **no toca el cero**; tras corregir la fuga tampoco lo
> tocan el único 1:3 (+0,105) ni la pareja (+0,058).

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

---

## 31/08/2026 — Una fuga de futuro en FIRMS, y por qué el arreglo refuerza el resultado

`comparar_rankings.firms_dia` pedía a la API de FIRMS `/5/{D-1}` creyendo que
la ventana de 5 días iba hacia atrás. La API los cuenta **hacia adelante**
desde la fecha, así que devolvía `[D−1, D+3]`: el mapa del día D llevaba dentro
los focos térmicos del propio incendio y de los tres días siguientes.

**Cómo se detectó.** No leyendo el código, sino mirando si el acierto decaía al
alejarse del día del suceso. Puntuando el fuego del día D con el mapa de D−2,
producción no perdía nada (63 % → 62 % de incendios de ≥500 ha en el top-2 %
del día) mientras que su variante sin FIRMS sí (31 % → 18 %). Un predictor
honesto pierde fuerza con la distancia; uno que está viendo el suceso, no. La
confirmación: los 33 incendios de ≥500 ha tenían un foco FIRMS a menos de 5 km
en su ventana «pasada» (control aleatorio: 1,2 %). Tras el arreglo, 9,1 %.

**Alcance.** Producción en vivo **nunca** estuvo afectada: `riesgo_hoy.py` y el
`firms_api.py` del repo hermano llaman sin fecha de inicio, que son los cinco
últimos días hasta hoy, y en operación el futuro no existe. El entrenamiento
tampoco: `extraer_features_historia.py:117` usa `(vf_d >= d-7) & (vf_d < d)`,
estrictamente anterior a D. La fuga solo se materializó en la evaluación
retrospectiva, porque la caché de junio y julio se descargó en agosto, cuando
el futuro ya existía. Se relanzaron `dos_09`, `dos_11`, `dos_14`, `dos_19`,
`comparar_rankings` y `comparar_rankings_justo`.

**Qué cambia.** Producción baja de 0,737 a **0,647** de AUC medio en 2026 y el
único de 0,773 a **0,752**, de modo que la ventaja pasa de +0,037 con el
intervalo rozando el cero a **+0,105 [+0,065, +0,145]**; el ratio 1:10 llega a
**+0,112 [+0,074, +0,151]**, ganando el 76 % de los días. Dos afirmaciones se
caen enteras: que producción acertaba los megaincendios (0,883 era la fuga; el
valor real es 0,692, y sin FIRMS saca 0,689: la feature no le aportaba nada
ahí) y que la malla batía al ranking sellado en el cara a cara justo (0,602 era
la fuga; limpia da 0,547 y **pierde**). Comprobación de que el recálculo es
correcto: todas las filas *sin FIRMS* y el percentil del FWI no se mueven ni un
dígito.

**Lo que se aprende, que es lo que va a la memoria.** Un error de una línea en
el código de EVALUACIÓN —no en el modelo, no en el entrenamiento, no en
producción— sostuvo durante semanas la conclusión de que la mejora era
marginal. La comprobación que lo destapó es barata y general: *si el acierto no
decae al alejar el predictor del suceso, algo está viendo el suceso*. Se ha
dejado automatizada en `dos_24_auditoria_fugas.py`, que audita los 17 modelos
separando causa de fuga por dos criterios —la escala espacial (¿se mueve solo
donde arde, o también en el anillo de 50-150 km?) y si la fuente puede ver el
fuego (el reanálisis no; el satélite sí)—. Su resultado: ningún modelo tiene
fuga seria, y los tres únicos rasgos concurrentes (`lst`, `ndvi`, `swi010`, del
día D) pesan entre el 2,1 % y el 7,5 % del *gain* según el modelo, **la misma
proporción en todos**, así que no sesgan ninguna comparación pero obligan a
presentar los números del mismo día como cota superior.

## 01/09/2026 — Archivar las entradas para poder juzgar modelos que aún no existen

La cadena diaria guardaba sus **salidas** y tiraba sus **entradas**. Cada
`.npz` de `salida/mapas_diarios/` contiene un único array —`prob`, 920×1188 en
`float16`—: la probabilidad ya calculada por el modelo que se sirvió ese día.
Con eso se puede repuntuar *ese* mapa cuando llega el perímetro de EFFIS, pero
no se puede puntuar un modelo distinto, porque las 46 variables de ese día no
están en ninguna parte.

Y las entradas irrepetibles se perdían. `malla_02b_ifs.py` escribe la pasada del
día en `salida/ifs_malla_<fecha>.parquet`, pero ese patrón no figuraba en la
lista `DIARIO` de `gh_estado.py`, así que no subía al Release y moría con el
runner de Actions. La API de previsión de Open-Meteo solo sirve la pasada
vigente: el pronóstico emitido el 27 de agosto para el 27 de agosto no se puede
volver a pedir. Lo mismo con el NRT de FIRMS, cuyo archivo está reprocesado.

El mecanismo para aprovecharlas ya existía y estaba escrito hace semanas:
`dos_riesgo_hoy.py --pasada <fecha>` sustituye la descarga por la lectura de
`ifs_malla_<fecha>.parquet` (`dos_riesgo_hoy.py:91-93`) y reejecuta la cadena
entera —`series_nodos`, `meteo_dia`, las 46 variables, el *scoring*— como si
fuera ese día. Faltaba únicamente el fichero.

**Lo hecho.** El paso de respaldo del workflow copia ahora también
`ifs_malla_<fecha>.parquet` y `_firms/nrt_<fecha>.csv` al artefacto que ya
existía, con 60 días de retención: unos 270 KB al día. Se hizo ahí y no en
`DIARIO` para no tocar `gh_estado.py`, que va sellado por md5 en
`PROCEDENCIA.md`. Y se pincharon las versiones de `requirements_gh.txt`, que no
fijaba ninguna: archivar la entrada no sirve de nada si dentro de seis meses el
mismo dato pasa por otro XGBoost y no se puede atribuir el cambio.

**El hueco del 15 de agosto al 1 de septiembre.** Es recuperable, al contrario
de lo que parecía: `historical-forecast-api.open-meteo.com` sirve las pasadas
archivadas tal y como se emitieron, y es de donde salió `ifs_historico.parquet`
del retro justo. Se descargó el rango 08-ago → 02-sep en los 5.605 nodos —26
días, 145.730 filas, 477 KB, sin un solo 429— y vive fuera de los repositorios,
en `archivo_ifs/`.

**Qué vale y qué no.** La API histórica devuelve una serie continua por nodo, no
la pasada de un día: los ficheros por fecha salen de cortar `[F-7, F+1]`, con
las mismas columnas que espera `--pasada`. Para D y D+1 es previsión archivada
de verdad; para el tramo pasado es «lo mejor disponible a poco plazo», que es
también lo que devolvía `past_days=7` en la cadena viva. Es una reconstrucción
fiel en lo que importa, no un byte a byte. El archivo exacto empieza el
01/09/2026.

**Y una distinción que conviene no perder.** Un modelo puntuado a posteriori
sobre entradas archivadas no vale lo mismo que uno sellado antes del día: con
la temporada guardada se pueden probar veinte variantes hasta que una gane, y
el intervalo de confianza de la ganadora deja de significar lo que dice. El
archivo es un banco de pruebas y una herramienta de auditoría —sin las cachés
de junio y julio la fuga de FIRMS no se habría podido demostrar—; el veredicto
lo siguen firmando los tres jueces en vivo sobre candidatos comprometidos de
antemano, que es por lo que `donde_dia_effis_r10` se puso a servir el 31/08 en
lugar de limitarse a medirlo en retro.

## 05/09/2026 — Calibración incorporada

Jornada entera en un banco aislado (`~/Desktop/Master/calibracion_si/`), fuera de
todo repositorio, a petición explícita: ningún fichero de `TFM_fuego_malla` ni de
este repo se modificó mientras se producían los resultados.

Lo que salió: los dos regímenes de incendio de España; el aviso de día calibrado
(funciona en invierno, en verano no se distingue de la climatología); la escala
absoluta que da significado al color (EXTREMO = 0,13 %, techo 1,6 %); la
probabilidad por celda mes a mes (0,28 % en diciembre a 10,36 % en julio); y un
resultado negativo prerregistrado: entrenar el r10 con todos los años **no
mejora** (+0,0033, IC [−0,0042, +0,0104]).

Cuatro fallos encontrados y corregidos, todos en scripts que nunca habían
corrido: la climatología del BSS se memorizaba a sí misma e incluía el test; el
umbral por máximo SEDI se iba al extremo de la rejilla (3 avisos en 854 días);
la calibración beta por celda no era monótona (techo 0,000 %) y se sustituyó por
isotónica; `N_BOOT`/`SEED` estaban declarados y sin usar, así que no había IC.

## 06/09/2026 — La validación es el replay

Se montó el disco Expansion buscando el mapa operativo del 21-ago: no existe
(`LIMITACIONES.md` §11). Y se tomó la decisión que cambia el eje del cierre:
**el veredicto de la memoria es el replay de 2025 y 2026**, no los jueces en
vivo, que llevan ~15 días y cuyos intervalos siguen cruzando el cero. Es
defendible por tres cosas: 250 días contra 15, dos temporadas contra una, y el
prerregistro escrito antes de correrlo. Su límite —a posteriori, no sellado—
va escrito en la memoria. Se descarta la Fase 4 del «si» (modelo día-nacional
directo): hay material de sobra y no aporta al argumento. Nada se tocó en
ningún repositorio.

## 07/09/2026 — Poner el repositorio en orden para la memoria

Los tres jueces siguen verdes (única corrida fallida de septiembre, el 04/09,
se repescó el mismo día). MITECO gana un día (13) y el único sigue clavado en
0,69; EFFIS baja dos centésimas en todos; producción sigue delante por
estación sin significación.

Auditoría del repo y arreglos: `01_datos/egif/` estaba vacío (ahora README con
la procedencia del CSV de Civio, que se bajó a mano); `06_comparacion/README`
no mencionaba el replay, que es el veredicto; `05_iteracion2/README` no
mencionaba la calibración; el README raíz y `LIMITACIONES.md` §3 todavía daban
como titular el retrospectivo de 74 días; el 21-ago y los cinco días perdidos
no estaban en `LIMITACIONES.md` (§11 nuevo); `PENDIENTE.md` era un diario de
529 líneas (ahora estado actual + histórico compacto); `docs/MEMORIA/` no
decía cuál de sus cinco documentos era cuál (README nuevo). Y se escribió el
primer borrador divulgativo de la sección de la memoria conjunta
(`docs/MEMORIA/seccion_ML_divulgativa.md`, 5 páginas): titular = replay, r10
como modelo propuesto, producto de dos capas, límites declarados.
