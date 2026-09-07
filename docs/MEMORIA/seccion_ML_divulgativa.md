# Predicción diaria del riesgo de incendio: dónde y cuándo

> Borrador de la sección de *machine learning* de la memoria conjunta. Objetivo:
> 5 páginas, tono divulgativo, sin fundamentos ni código. Escrito el 07/09/2026
> a partir de `docs/TRAZABILIDAD.md`, `docs/REPLAY_*.md`,
> `calibracion_si/RESULTADOS_05SEP.md` y los READMEs por capítulo. Todas las
> cifras están trazadas al final, en «Notas para el autor» (no van a la memoria).
> Los `[FIG n]` y `[TAB n]` marcan el hueco de cada figura o tabla.

---

## 1. La pregunta, y el resultado en un párrafo

La España peninsular cabe en 498.530 celdas de un kilómetro de lado. La pregunta
que responde este módulo es, cada mañana, **cuáles de esas celdas van a arder
hoy**. No es un mapa de «zonas propensas», que sería el mismo todos los días,
sino un ranking diario: el producto es una ordenación de las celdas de más a
menos peligro, y lo único que importa de ella es si las que efectivamente
ardieron estaban arriba.

El camino hasta ahí no fue en línea recta, y esa es la parte del trabajo que
más vale la pena contar. Un primer modelo alcanzó un acierto del 0,89 en su
propio banco de pruebas y, puesto a funcionar cada día y medido contra lo que de
verdad se quemó, cayó a 0,56-0,64. La causa no era que el modelo estuviera mal
ajustado ni que la meteorología en tiempo real fuera peor que la de
entrenamiento: era que **el conjunto de datos con el que se entrenó hacía otra
pregunta**. Corregido eso, y reevaluado sobre las temporadas completas de 2025 y
2026, el nuevo modelo pasa de 0,74 a 0,81 en 2025 y de 0,66 a 0,76 en 2026, y
lo hace prediciendo a un día vista con la misma precisión que si supiera el
tiempo que va a hacer.

Este módulo es el «antes» del sistema conjunto: anticipa el riesgo que la
visión por computador confirma y que el asistente sobre los partes del MITECO
permite consultar.

## 2. Con qué se construye

Cuatro fuentes, y conviene tener claro qué aporta cada una porque el giro del
trabajo está en la diferencia entre dos de ellas.

[TAB 1 — fuentes]

| Fuente | Qué es | Para qué se usa |
|---|---|---|
| **IberFire** | Un «cubo» de datos abierto (Tekniker / UPV-EHU) con 261 variables diarias para cada celda de 1 km desde 2008 hasta 2024: tiempo, vegetación, terreno, usos del suelo, población, carreteras | De aquí salen casi todas las variables del modelo |
| **EGIF** | El registro oficial de incendios del Ministerio: una fila por incendio con su fecha y un **punto** GPS de inicio | La etiqueta del primer modelo |
| **EFFIS** | Los **perímetros** de superficie quemada que Copernicus cartografía por satélite | La etiqueta del segundo modelo, y la verdad contra la que se juzgan los dos |
| **ERA5-Land / IFS** | El reanálisis meteorológico europeo (lo que pasó) y la previsión del ECMWF (lo que va a pasar) | Entrenar con lo que pasó; servir cada día con lo que va a pasar |

El modelo ve **46 variables** por celda y día. Dichas en castellano, son cuatro
familias: *qué tiempo hace* (temperatura, humedad, viento, lluvia de hoy y de los
últimos 7, 15 y 30 días, y el índice de peligro FWI), *qué hay en la celda*
(proporción de bosque y matorral, pendiente, altitud, distancia a carreteras,
población), *qué ha ardido cerca antes* (incendios en 1 y 10 km en los últimos
meses y años, focos térmicos detectados por satélite en la última semana) y *qué
día es*. La lista completa está en el anexo A.

Una de ellas merece una línea porque es una aportación propia del trabajo: el
FWI no se usa solo en valor absoluto, sino como **percentil de la propia celda**.
Un FWI de 40 es rutina en Almería y excepcional en Asturias; lo que predice el
fuego no es el número, sino lo raro que es ese número *ahí*. Esa variable sola
aporta +0,042 de acierto.

## 3. El primer modelo, y por qué falló al salir a la calle

El primer modelo se construyó como se construyen casi todos los de la
literatura. Los positivos son los 19.561 incendios del registro EGIF entre
2015 y 2020, cada uno en su celda y su fecha. Los negativos se sortean tomando
**la misma celda en otros días** en que no ardió, tres por cada positivo. Se
entrena un XGBoost, se afina con 2021 y se prueba con 2022 —el peor año de la
década, con Losacio y Sierra de la Culebra dentro—, y da un AUC de **0,89**. Un
control con etiquetas barajadas confirma que no hay fugas.

Con ese resultado se puso en producción en julio de 2026: cada mañana descarga
la observación de 687 estaciones de AEMET, calcula las 46 variables, puntúa y
publica un ranking nacional y un mapa de riesgo para hoy y mañana. Lleva desde
entonces corriendo sin interrupción, y esa serie sellada es lo que permite todo
lo que viene después.

Y entonces se midió. Contra los perímetros de EFFIS, contra los partes diarios
del MITECO y contra los focos térmicos de satélite, el acierto diario del modelo
en producción estuvo entre **0,56 y 0,64**. Un índice tan simple como el
percentil local del FWI, sin modelo, sacaba 0,70.

[FIG 1 — las dos iteraciones y la bisagra que las separa]

Había tres sospechosos, y los dos primeros se descartaron con números antes de
llegar al tercero:

1. **¿Está mal medido?** No: tres verdades independientes —satélite, cartografía
   y partes oficiales— cuentan lo mismo.
2. **¿Es la meteorología?** El modelo se entrenó con el reanálisis del cubo y se
   sirve con estaciones. Hay diferencias reales —el reanálisis «llueve» 8 mm más
   al mes, y eso baja el FWI 10 puntos—, pero al cuantificarlas no explican la
   caída. Tampoco explica nada lo que cambia entre entrenar y servir: sustituir
   las variables de satélite por su climatología cuesta 0,0045.
3. **Es la pregunta.** Al sortear los negativos en *la misma celda otros días*,
   el 98,8 % de las celdas del conjunto de entrenamiento tenían exactamente un
   25 % de días positivos. El modelo no podía aprender qué celda es más
   peligrosa que otra, porque todas tenían la misma proporción de fuego; solo
   podía aprender **qué día** es peligroso. Respondía a «¿es hoy un día malo en
   esta celda?», y en operación se le preguntaba «¿cuál de las 498.530 celdas
   arde hoy?».

La prueba definitiva es que la métrica de laboratorio era ciega a la
diferencia. Se construyeron dos conjuntos de entrenamiento, uno con negativos
de la misma celda y otro con negativos del mismo día. Medidos cada uno con su
propio banco de pruebas, los dos dan 0,92. Medidos **dentro de cada día**, que
es como se usan, dan 0,744 y 0,828. La segunda medida es la que había que haber
hecho desde el principio, y es la lección del trabajo.

## 4. El segundo modelo: cambiar la pregunta, no el algoritmo

La segunda versión cambia exactamente dos cosas y deja todo lo demás igual: las
mismas 46 variables, el mismo algoritmo con los mismos parámetros, la misma
malla.

- **Los negativos son otras celdas del mismo día.** Así el modelo aprende a
  comparar celdas entre sí, que es lo que se le pide.
- **La etiqueta es la superficie quemada de EFFIS**, la misma con la que se
  juzga. Entrenar con puntos de ignición y evaluar con perímetros era comparar
  dos cosas distintas: solo el 15 % de las igniciones de 25-100 ha coincide
  celda a celda con un perímetro del mismo día.

Se probaron varias configuraciones —un modelo único frente a la pareja de un
modelo de *dónde* (fijo) por uno de *cuándo* (diario), y distintas cantidades
de negativos por positivo— y se seleccionó el **modelo único con diez
negativos por positivo**, «r10» en adelante. No es que las demás fueran mucho
peores: la cantidad de negativos mueve el acierto medio apenas +0,01, del
orden del azar del sorteo; donde el r10 marca la diferencia es en los
incendios grandes, que es donde importa. El detalle de esas pruebas, incluidas
las que salieron mal (entrenar solo con verano empeora; añadir una capa de «ya
quemado» empeora), está en el anexo B.

## 5. Resultados: dos temporadas enteras, a un día vista

Para comparar con justicia hay que poner a los dos modelos ante los mismos días,
la misma meteorología y la misma verdad. Se hizo reproduciendo día a día las
temporadas de 2025 y 2026 (del 1 de junio al 1 de noviembre y al 2 de
septiembre respectivamente), alimentando a ambos con **la previsión que estaba
disponible la víspera** y juzgándolos con los perímetros de EFFIS. El protocolo
se escribió y se registró antes de ejecutar nada.

[TAB 2 — el veredicto]

| Temporada | Días con fuego | Hectáreas | Producción | r10 | Diferencia (IC 95 %) | Días que gana el r10 |
|---|---|---|---|---|---|---|
| 2025 | 138 | 387.000 | 0,743 | **0,807** | +0,064 [+0,027, +0,101] | 60 % |
| 2026 | 86 | 268.000 | 0,658 | **0,762** | +0,104 [+0,065, +0,144] | 67 % |

Tres lecturas para quien no quiera mirar los decimales:

**La mejora es consistente y no es ruido.** El intervalo de confianza no toca el
cero en ninguna de las dos temporadas, y el r10 gana en dos de cada tres días.
Es mayor en 2026, la temporada en que producción lo pasó peor.

**Predecir no cuesta nada.** Cada día se calculó dos veces: con la previsión de
la víspera y, a posteriori, con el tiempo que realmente hizo. La diferencia es
de −0,003, indistinguible de cero. El modelo no depende de acertar el tiempo
al detalle; depende de saber dónde está el combustible y cuánto lleva seco.

**Es un mapa para vigilar, no una alarma por celda.** Que arda una celda
concreta un día concreto es rarísimo: una de cada diez mil. En el 2 % más alto
del mapa del r10 esa tasa se multiplica por doce, y si se mira a 6 km alrededor
—la escala a la que se despliegan medios—, ese 2 % cubre el 44 % de los
incendios de 2025 vigilando el 10 % del territorio. Producción cubría el 29 %
con el 6,6 %. El mapa dice dónde mirar, no dónde va a arder.

[FIG 2 — el mismo día de agosto de 2026 según producción y según el r10, y la resta entre ambos]

Además del replay, el sistema **corre en vivo desde el 21 de agosto** en GitHub
Actions: cada madrugada descarga la previsión, publica los mapas de ambos
modelos y los puntúa contra tres jueces independientes —EFFIS, el parte del
MITECO y el ranking por estación de producción—. A 7 de septiembre acumula
entre 8 y 13 días según el juez; los candidatos van por delante en EFFIS y
MITECO, pero con tan pocos días los intervalos siguen cruzando el cero. Se
cuenta como lo que es: una validación operativa en curso, no el veredicto.

## 6. Del ranking al producto: dos capas

El modelo da a cada celda una nota, y el mapa se pintaba por **posición en el
ranking del día**: el 2 % más alto en rojo, el 8 % siguiente en naranja. Eso
tiene un defecto que se ve enseguida en cuanto sale del verano: el ranking
siempre tiene un primero, así que un martes de febrero sale con tanto rojo como
el 15 de agosto, aunque la nota de esa celda «roja» sea veinte veces menor.

La solución tiene dos piezas. La primera es fijar los colores **en la escala
de la nota**, no en la del día: se pasaron los modelos por los diez años del
cubo —3.653 días— y se anotó a partir de qué nota se entra en el 2 % más alto
*de toda la historia*. Con esa regla el rojo significa algo concreto: en diez
años, de cada 800 celdas pintadas de EXTREMO ardió una, 36 veces más que el
promedio; en las de BAJO, cinco en toda la década. Pero la escala absoluta sola
no apaga el mapa de invierno: las variables que fijan *dónde* (bosque,
pendiente, historial) no cambian con la estación, y las mismas celdas superan
el corte en enero y en agosto. Hace falta la segunda pieza, un **semáforo
nacional**: si la nota más alta de España ese día no supera el umbral que en la
historia separa los días con incendio grande, hoy no es un día de fuego y el
mapa no se pinta. Con esa puerta, en invierno el 44 % de los días quedan sin
rojo y solo se pierde el 6 % de los días grandes.

[FIG 3 — el mismo día de invierno pintado por percentil del día y por escala absoluta]

Al comprobarlo se aprendió algo que no estaba previsto. Ese semáforo, fijado
con datos de 2015-2021, funciona sin retocar en 2025 y 2026. Pero **no tiene
habilidad en verano**: de junio a septiembre casi todos los días pasan del
umbral, y decir «hoy es peligroso» en julio no añade nada al calendario. Donde
sí aporta es en invierno y primavera, cuando distingue el día raro en que arde
la cornisa cantábrica del resto. Por eso el producto final tiene dos capas: el
semáforo dice *si* hoy hay que mirar el mapa, y el mapa dice *dónde*. En verano
el semáforo está siempre en rojo y manda el mapa; en invierno está casi siempre
en verde y evita el mapa rojo falso.

Una precisión que conviene dejar escrita: la nota del modelo **no es una
probabilidad**. Un 0,80 no significa «80 % de que arda». Es una puntuación para
ordenar celdas y compararla con umbrales, y así se presenta siempre: como nivel,
nunca como porcentaje.

## 7. Límites, y lo que se aprendió

Cuatro límites que el lector debe conocer:

- **La evaluación de dos temporadas es a posteriori.** Los mapas de 2025 y 2026
  se generaron con la previsión de cada víspera, pero se generaron *después*,
  no sellados día a día. Lo sellado es la cadena en vivo, y esa aún no tiene
  días suficientes. Las dos medidas dicen lo mismo con distinta fuerza, y se
  reportan ambas.
- **La nota no es calibrable como probabilidad** por celda, y el sistema se
  presenta como ranking y niveles.
- **El sistema en producción sirve una versión con un defecto conocido** en el
  cálculo del viento, detectado en agosto y no corregido a propósito para no
  romper la comparación en marcha. Está cuantificado en el anexo E.
- **Cobertura**: Península y Baleares, sin Canarias; datos de 2026 hasta el 2 de
  septiembre.

Y la lección, que es lo que este módulo aporta más allá de un mapa:

> Un modelo se optimiza para la pregunta que define su conjunto de
> entrenamiento, y esa pregunta la fija la elección de positivos y negativos, no
> la intención de quien lo entrena. Un acierto alto sobre un banco construido con
> el mismo diseño que el entrenamiento no demuestra nada sobre el uso previsto.
> La comprobación que lo detecta —evaluar dentro de cada día, con la misma
> verdad con la que se va a juzgar el sistema— es barata y debería hacerse
> antes de entrenar, no después de dos meses en producción.

El código completo, con cada cifra de esta sección enlazada al script que la
produjo, está en el repositorio `tfm-riesgo-incendios` (anexo F).

---
---

## Notas para el autor (no van a la memoria)

**Presupuesto.** ~2.450 palabras + 2 tablas + 3 figuras ≈ 5-5,5 páginas a 11 pt.
Si sobra, el §5 (resultados) es el último que se recorta; §2 y §4 los primeros.

**Trazabilidad de cada cifra** (ver `docs/TRAZABILIDAD.md`):

| Cifra | Origen |
|---|---|
| 498.530 celdas, 261 variables, 2008-2024 | `BORRADOR_memoria_ML.md` §1.1, `02_eda/ANALISIS_CUBO.md` |
| 46 variables; +0,042 del percentil local | `features.py`; `35_ablaciones/ablacion_features.py` |
| 19.561 incendios EGIF 2015-20; split v4; AUC 0,89 (0,8906 test 2022) | `muestrear_dataset_v4.py`, `T/dataset/metricas_v4.json` |
| 687 estaciones; 0,56-0,64; FWI percentil 0,70 | `04_bisagra/41_*` (0,702 en serie D0) |
| +8 mm/30 d; −0,0045 | `42_no_es_la_meteo/experimento_b_era5.py`; `43_*` |
| 98,8 %; 0,92 / 0,744 vs 0,828; 15 % EGIF-EFFIS | `44_*`, `dos_04_analisis.py` |
| Ratio: +0,006…+0,017, ruido 0,004-0,005 | `dos_18_ratio.py`, `LIMITACIONES.md` §4 |
| Tabla 2 (replay) | `docs/REPLAY_VEREDICTO.md` (2025_ifs, 2026_ifs) |
| Coste previsión −0,003 | `REPLAY_VEREDICTO.md`, bloques ifs_menos_reanalisis |
| Lift ×12, 44 % / 10 % vs 29 % / 6,6 % | `REPLAY_PRECISION.md`, `REPLAY_COBERTURA.md` |
| Jueces a 7-sep | `publicado/historico_veredictos.csv` (run 34099321965) |
| 3.653 días; 1 de cada 800 (lift 35,6×); 5 en BAJO; 44 % / 6,2 %; sin habilidad en verano | `05_iteracion2/56_calibracion/README.md`; `calibracion_si/RESULTADOS_05SEP.md` §2, §4, §6 |
| Bug del viento; 21-ago; 5 días perdidos | `LIMITACIONES.md` §1; `PENDIENTE.md` |

**Decisiones asumidas en el texto** (cambiar si no procede):
1. Titular = replay, no los 74 días retrospectivos.
2. Ventana 2026 cerrada el 02-sep (opción A del plan del 07/09).
3. Modelo propuesto = r10 para el dónde; la pareja para el semáforo se menciona
   sin nombrarla («semáforo nacional»).
4. Los jueces en vivo se citan con fecha; actualizar el párrafo al cierre.
5. El 21-ago excluido del juez EFFIS: no se menciona en el texto, va al anexo E.

**Figuras propuestas:**
- FIG 1: ya existe (`figs/f1_iteraciones.png`).
- FIG 2: existe el mapa de seis paneles de `dos_riesgo_hoy.py`; recortar a tres
  (producción, r10, resta).
- FIG 3: nueva. Un día de invierno del replay pintado con `CORTES_PCTL` y con
  los cortes absolutos de `dos_27`. Hay que generarla.

**Anexos referenciados:** A variables y fuentes · B ablaciones y resultados
negativos · C jueces en vivo al cierre · D calibración del semáforo (BSS/SEDI
por régimen, cortes absolutos) · E incidentes con efecto en los números (fuga
FIRMS, días perdidos 27-31/08, mapa del 21-ago, bug del viento) · F guía del
repositorio.
