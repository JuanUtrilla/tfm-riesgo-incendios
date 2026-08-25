# Cómo contar un trabajo que dio la vuelta a mitad de camino

> Guía de redacción para la memoria, con el texto de la sección bisagra ya
> redactado. 23/08/2026.

## 1. El problema de fondo

La estructura que enseñan —explorar, entrenar, evaluar, concluir— supone que
el trabajo fue en línea recta. Este no lo fue: se entrenó un modelo, se puso
en producción, se midió, y al medir se descubrió que **estaba entrenado para
responder una pregunta distinta de la que se le hacía**. Eso obligó a volver a
los datos y montar un segundo pipeline.

La tentación es contarlo como un diario («primero probé…, luego vi que…»). No
se hace así, pero **tampoco se esconde**. Las dos reglas que resuelven la
tensión:

**Regla 1: se escribe el hilo LÓGICO, no el CRONOLÓGICO.** Da igual en qué
orden ocurrieron las cosas; importa en qué orden hay que leerlas para que la
conclusión sea inevitable. Si el diagnóstico se entendió después de tres
semanas de callejones sin salida, en la memoria el diagnóstico va justo
después de la evidencia que lo sostiene, y los callejones no van.

**Regla 2: el primer modelo no es un error, es el grupo de control.** Sin él
no hay forma de demostrar que el rediseño hacía falta ni de cuantificar lo que
aportó. Un TFM que enseña «monté esto, medí que estaba mal especificado, lo
corregí y volví a medir» vale más que uno que enseña un modelo que funciona,
porque demuestra criterio y no solo ejecución. En la literatura esto tiene
nombre y sección propia: *error analysis* y *ablation*. El fallo, medido y
explicado, **es un resultado**.

## 2. La estructura: dos ciclos y una bisagra

El esquema que funciona para este trabajo es un sándwich: un marco común, dos
iteraciones completas y una bisagra entre ellas que es el corazón de la
memoria.

```
1. Introducción y objetivo        ← aquí se anuncia el giro (§3)
2. Datos y fuentes                 marco común a las dos iteraciones
3. Análisis exploratorio           marco común

4. PRIMERA ITERACIÓN
   4.1 Diseño: pregunta, etiqueta (EGIF), muestreo caso-control
   4.2 Entrenamiento y resultados de test: AUC 0,89
   4.3 Puesta en producción: la cadena diaria, 687 estaciones

5. LA BISAGRA: por qué 0,89 no significaba lo que parecía   ← el capítulo clave
   5.1 La validación operativa: 0,56-0,64 contra área quemada
   5.2 Diagnóstico 1: el muestreo fija la pregunta
   5.3 Diagnóstico 2: la etiqueta de entrenamiento no es la de validación
   5.4 Qué implica: no es un problema de ajuste, es de especificación

6. SEGUNDA ITERACIÓN
   6.1 Rediseño: negativos del mismo día, etiqueta EFFIS
   6.2 Dos modelos (DÓNDE × CUÁNDO) frente a uno con muestreo dónde
   6.3 Entrenamiento y evaluación con el protocolo correcto
   6.4 Ablaciones: ratio de negativos, verano, historial, FIRMS

7. COMPARACIÓN DE LOS DOS PIPELINES        ← la tabla y el mapa de diferencia
8. Producción y validación en curso: los tres jueces
9. Conclusiones y limitaciones
```

Dos avisos sobre este esquema:

- **Los capítulos 2 y 3 son comunes a propósito.** Repetir «datos» dentro de
  cada iteración duplica veinte páginas. Lo que cambia entre iteraciones no
  son los datos: es qué se toma como positivo, qué como negativo y qué como
  verdad.
- **El capítulo 5 es el que se lee.** Es donde está la aportación
  intelectual. Merece figuras propias y números propios, no un párrafo de
  transición.

## 3. Anunciar el giro en la introducción

El error más común al escribir esto es dejar al lector cuarenta páginas
creyendo que el primer modelo es el bueno, y darle el disgusto en el
capítulo 5. Se hace al revés: se avisa en la primera página, y luego el
lector lee la primera iteración sabiendo que va a fallar y buscando por qué.

Plantilla, adaptada al caso:

> Este trabajo desarrolla un sistema de predicción diaria de riesgo de
> incendio forestal para la España peninsular y lo pone en operación durante
> la temporada de 2026. El primer modelo alcanzó un AUC de 0,89 en su
> conjunto de test y, servido a diario contra superficie quemada real, cayó a
> 0,56-0,64. La memoria muestra que la discrepancia no se debe al
> sobreajuste ni al cambio de fuentes meteorológicas, sino a un **error de
> especificación**: el diseño de muestreo y la etiqueta de entrenamiento
> definían una pregunta —«¿es hoy un día peligroso en esta celda?»— distinta
> de la que se evalúa en operación —«¿cuál de las 498.530 celdas arde hoy?»—.
> A partir de ese diagnóstico se rediseña el conjunto de entrenamiento y se
> construye un segundo sistema, que se compara con el primero sobre los mismos
> días y con las mismas fuentes. Las dos iteraciones y su comparación son el
> contenido de la memoria.

Esto hace tres cosas a la vez: da el resultado, nombra la aportación
(diagnóstico de especificación) y justifica que haya dos pipelines.

## 4. Cómo se titula la bisagra

Mal: «Problemas encontrados», «Dificultades», «Limitaciones del primer
enfoque». Suenan a excusa y no dicen nada.

Bien: se titula con **el hallazgo**, en afirmativo.

- «Por qué un AUC de 0,89 no medía lo que hacía falta»
- «El muestreo define la pregunta: caso-control temporal frente a espacial»
- «Entrenar con igniciones y validar con superficie quemada»

Y dentro, cada afirmación con su número al lado. El diagnóstico convence
porque está medido, no porque esté bien argumentado:

- el 98,8 % de las celdas del conjunto v1 tenía exactamente un 25 % de
  positivos, de modo que el modelo no podía aprender qué celda es más
  peligrosa que otra: solo qué día lo es;
- el AUC caso-control de los dos diseños es idéntico (0,92) y el AUC dentro
  del día los separa (0,744 frente a 0,828): la métrica de desarrollo era
  ciega a la diferencia;
- solo el 15 % de los incendios EGIF de 25-100 ha coincide celda a celda con
  un perímetro EFFIS del mismo día: las dos etiquetas no son la misma cosa.

## 5. Qué NO hacer

- **No narres el calendario.** «El 12 de julio probé…» no aporta. La única
  fecha que importa en el texto es la del sellado de producción, porque
  garantiza que no hubo reentrenamiento a mitad de temporada.
- **No cuentes todos los experimentos.** Solo los que sostienen una decisión.
  Los demás, si acaso, en un anexo de resultados negativos —que sí conviene
  tener: «entrenar solo con verano empeora el DÓNDE (−0,028)» le ahorra el
  intento al siguiente.
- **No presentes la segunda iteración como una mejora incremental.** No se
  afinaron hiperparámetros: se cambió la pregunta. Si lo cuentas como «además
  probé otra configuración y fue algo mejor», pierdes la aportación.
- **No escondas que la ventaja es modesta.** +0,037 con el IC rozando el cero
  es lo que hay; escribirlo con su intervalo vale más que redondearlo a
  «mejora significativa». La honestidad estadística se nota y se premia.

## 6. Texto redactado: por qué un segundo pipeline

> *Sección 7 del esquema. Se puede pegar tal cual y ajustar las cifras al
> cierre de temporada.*

### 7. Los dos pipelines, uno al lado del otro

Los dos sistemas comparten las 46 variables, el algoritmo (XGBoost con los
mismos hiperparámetros), la malla de 1 km y la cadena de servicio. Todo lo que
los separa está en tres decisiones de diseño del conjunto de entrenamiento,
y esas tres decisiones son las que explican la diferencia de rendimiento.

| | Pipeline 1 (en producción) | Pipeline 2 (candidato) |
|---|---|---|
| Pregunta que aprende | ¿es hoy un día peligroso **en esta celda**? | ¿**cuál de las celdas** de hoy arde? |
| Negativos | la misma celda, otros días | el mismo día, otras celdas |
| Etiqueta | ignición EGIF (punto, ≥1 ha) | superficie quemada EFFIS (celda-día, ≥5 ha) |
| Métrica de desarrollo | AUC caso-control: 0,89 | AUC dentro del día: 0,81 |
| ¿Coincide con la validación? | No: se valida con área quemada | Sí: misma etiqueta en train y en juicio |
| Dependencia de FIRMS | crítica | baja |
| Arquitectura | un modelo | uno (muestreo dónde) o dos (DÓNDE × CUÁNDO) |

**Lo que gana el segundo pipeline.** Sobre los 74 días de la temporada de
2026, juzgado con los perímetros de EFFIS, el AUC dentro del día pasa de
0,737 a 0,773; con el parte diario de MITECO, de 0,693 a 0,731; y en la única
comparación previsión-contra-previsión disponible —19 días contra el ranking
sellado que producción publicó de verdad—, de 0,568 a 0,625, ganando 16 de los
19 días.

**Lo que gana y no se ve en el AUC.** Producción depende de las detecciones
térmicas de FIRMS hasta el punto de que sin ellas su AUC cae a 0,611 y su
discriminación contra el parte de MITECO desaparece (0,52, indistinguible del
azar). El candidato sin FIRMS se queda en 0,744, es decir, **iguala a
producción *con* FIRMS**. Esto no es un matiz académico: cualquier día con
VIIRS cubierto de nubes o con la API caída —ya ocurrió durante la temporada—
deja al sistema en producción sin su variable más informativa, y al candidato
prácticamente igual. Es la diferencia entre un modelo que ha aprendido dónde y
cuándo se quema el territorio y uno que sigue el fuego activo de ayer.

**Lo que se ve en el mapa.** La figura *(mapa de diferencia)* resta el
percentil del candidato menos el de producción para un día de agosto de 2026.
El patrón no es ruido: el candidato **sube sistemáticamente la cornisa
cantábrica** —Asturias pasa de un percentil medio de 12 a 77, Cantabria de 4 a
58, el País Vasco de 15 a 37— y **baja el interior sur** —Andalucía de 63 a
47, Murcia de 60 a 43—. Es exactamente el efecto que cabe esperar de cambiar
la etiqueta: la ignición EGIF es frecuente en el sur y en el centro, mientras
que la superficie quemada de EFFIS se concentra en el noroeste y la cornisa,
donde los perímetros son grandes. El primer pipeline aprendió dónde *empiezan*
los incendios; el segundo, dónde *se quema* territorio.

Conviene decir también lo que este ejemplo no demuestra. Que el candidato suba
Asturias un 22 de agosto no es automáticamente un acierto: la temporada
cantábrica es de febrero a abril, y en pleno verano una susceptibilidad
estática alta en el norte puede ser un sesgo heredado del reparto anual de la
superficie quemada. Quién tiene razón no lo decide el mapa, lo deciden los
jueces día a día, y por eso el veredicto se acumula durante toda la temporada
en lugar de zanjarse con una figura.

**Lo que cuesta.** El segundo pipeline no es gratis. Su probabilidad no está
calibrada —el producto de dos modelos con prevalencias de diseño distintas no
tiene interpretación probabilística—, y por eso el mapa se sirve en percentil
del día y los niveles se cortan por percentil y no por probabilidad. Además,
en los días de megaincendio producción sigue por delante (0,883 frente a
0,794) precisamente gracias a FIRMS: el fuego que ya arde predice bien el
fuego de mañana. Un sistema operativo maduro probablemente debería combinar
las dos señales en vez de elegir.

## 7. Y la conclusión, que es la lección del trabajo

El cierre no debería ser «el modelo B mejora al modelo A en 0,04 de AUC». Eso
es un número. La lección es metodológica y se puede escribir así:

> El resultado más útil de este trabajo no es el modelo final, sino el
> diagnóstico que lo hizo necesario: **un modelo se optimiza para la pregunta
> que define su conjunto de entrenamiento, y esa pregunta la fija el diseño de
> muestreo y la etiqueta, no la intención de quien lo entrena.** Un AUC alto
> sobre un banco de pruebas construido con el mismo diseño de muestreo que el
> entrenamiento no demuestra nada sobre el uso previsto. La comprobación
> barata que lo detecta —evaluar dentro de cada día, con la misma etiqueta con
> la que se va a juzgar el sistema en operación— debería hacerse antes de
> entrenar, no después de seis meses en producción.

## 8. Texto redactado: la ablación del ratio de negativos

> *Sección 6.4 del esquema, dentro de las ablaciones de la segunda iteración.
> Va después de la que compara los diseños de muestreo, porque su función es
> cerrar la objeción que aquella deja abierta.*

### 6.4. ¿Cuántos negativos? La ablación que aísla la causa

El resultado central de la segunda iteración —pasar de negativos tomados en la
misma celda otros días a negativos tomados el mismo día en otras celdas sube
el AUC dentro del día de 0,744 a 0,828— admite una objeción inmediata: los dos
conjuntos no solo difieren en *dónde* están los negativos, sino que son
conjuntos distintos, y podría ser que la ventaja viniese de la cantidad o del
sorteo y no del diseño. La constante que fija esa cantidad, tres negativos por
positivo, venía heredada del muestreador original y nunca se había movido.

Se construyó para responderlo una **escalera anidada**: los mismos 19.552
positivos, 110 negativos sorteados por positivo, una sola extracción de
variables (2,16 millones de filas) y cada peldaño definido como un *prefijo*
de esa lista, de modo que 1:3 ⊂ 1:10 ⊂ 1:30 ⊂ 1:60. Así la diferencia entre
peldaños es cuántos negativos hay, no cuáles, y el ruido de muestreo no
contamina la comparación. El número de árboles se fija en 441 para todos —el
punto en que se detuvo por *early stopping* el modelo de referencia—, porque
dejándolo libre se estaría comparando a la vez el ratio y la complejidad del
modelo.

| negativos por positivo | negativos/día (mediana) | filas de entrenamiento | AUC test 2024 | AUC temporada 2026 | percentil ponderado por ha, 2026 |
|---|---|---|---|---|---|
| 1:3 (el heredado) | 21 | 59.444 | 0,804 | 0,778 | 83,3 |
| 1:10 | 70 | 163.471 | 0,805 | **0,785** | 87,6 |
| 1:30 | 210 | 460.691 | 0,815 | 0,784 | **87,9** |
| 1:60 | 420 | 906.521 | **0,822** | 0,775 | 85,8 |
| hasta 1:100 | 700 | 1.457.619 | 0,813 | 0,766 | 82,3 |

De aquí salen tres lecturas.

**Primera, y es la que importa para el argumento del trabajo: la cantidad de
negativos es una palanca de segundo orden.** Moverla de 3 a 100 sobre los
mismos positivos y con la misma complejidad de modelo mueve el AUC entre
+0,006 y +0,017 según el banco. La escala del ruido está medida en el mismo
experimento: el modelo de referencia y el peldaño 1:3 son la misma receta con
otro sorteo de negativos, y se llevan 0,005. Es decir, **la ventaja del
rediseño no se explica por el tamaño ni por el sorteo del conjunto: se explica
por la posición de los negativos**, que es lo que se quería demostrar. Una
ablación cuyo resultado es «esto casi no influye» no es un experimento
fallido: es el control que convierte la afirmación anterior en una medida.

**Segunda: hay un óptimo interior, y añadir datos por encima de él perjudica.**
El rendimiento sube de forma monótona hasta 1:30-1:60 y a partir de ahí
retrocede; el peldaño de 1:100 es el peor de los cinco sobre la temporada de
2026 (−0,013 respecto al 1:3, con el intervalo casi excluyendo el cero). La
explicación más plausible es que, con cien celdas sorteadas al azar por
positivo, la inmensa mayoría son negativos triviales —secano, urbano, alta
montaña— y con un presupuesto fijo de árboles el modelo gasta capacidad en
separar lo que ya estaba separado. Es un contraejemplo útil a la intuición de
que más datos siempre ayudan: **más datos ayudan cuando lo que se añade es
difícil**.

**Tercera: donde sí se nota el ratio es en el fuego grande.** El percentil
ponderado por hectáreas quemadas sube de 83,3 a 87,9 y el *lift* del decil
superior de 3,76 a 4,08. El AUC medio por día trata igual un incendio de una
hectárea y uno de siete mil; la ponderación por superficie no, y es ahí donde
el conjunto más rico se traduce en una ordenación mejor de las celdas que de
verdad arden mucho.

La ablación deja un cabo abierto que conviene declarar: al fijar el número de
árboles se elimina un factor de confusión pero se introduce otro, porque no
puede descartarse que el retroceso del peldaño 1:100 se deba a que 441 árboles
se quedan cortos con un millón y medio de filas y no al ratio en sí.
Repetirla con parada temprana por peldaño cerraría el punto.

En cuanto a la recomendación práctica, el peldaño 1:10 sería la elección para
un modelo de cierre —cuesta 163.000 filas en vez de 59.000, no cambia ninguna
otra pieza del pipeline y mejora de forma consistente el fuego grande—, pero
**no se adopta durante la temporada**: cambiar el modelo servido a mitad de
validación invalidaría la comparación sellada contra el sistema en producción,
que es lo que da valor al veredicto acumulado.
