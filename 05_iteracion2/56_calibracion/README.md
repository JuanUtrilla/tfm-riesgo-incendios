# 5.6 · Calibración: del orden de las celdas a una probabilidad

## Qué pregunta responde

El modelo ordena bien las celdas. Un mapa que pinta de rojo el 2 % superior
de cada día se lee como «estas celdas van a arder», y eso es falso casi todo
el año. En esta parte se convirtió el orden en tres productos. Al
cierre, la cadena diaria publica solo la escala absoluta, con cortes
recalibrados sobre los mapas servidos, y la cifra del día; el semáforo se
evaluó y se descartó (`07_produccion/escala_y_cifra/README.md`).

1. Si se pinta hoy: un aviso de día calibrado, el semáforo. Evaluado y descartado.
2. Qué significa el rojo: una escala fija en la puntuación, en vez del percentil del día.
3. Cuánta probabilidad tiene una celda: un número que acompaña al color.

## Con qué se calibra

`replay_si.py` (capítulo 6) fijó un primer umbral del aviso con un solo verano
(jun-ago de 2025: 92 días, 86 con fuego), demasiado pocos días sin fuego para
estimar nada. Aquí se calibró con 3,653 días del cubo, 2015-2024, que incluyen
la variación real de la tasa base y los días de cero fuego.

![Percentil del día frente a escala absoluta, con el semáforo](../../docs/MEMORIA/figs/f10_percentil_vs_absoluto.png)

*El r10 el 13 de agosto y el 29 de octubre de 2025: a la izquierda por percentil del día, a la derecha con la escala fija del cubo y el semáforo, tal como se evaluaron antes de descartar el semáforo.*

## Qué hay aquí

El aviso de día se mide con el BSS (*Brier Skill Score*) y el SEDI
(*Symmetric Extremal Dependence Index*), con intervalos por bootstrap.

| Script | Qué hace |
|---|---|
| `dos_14_cortes.py` | Los cortes originales BAJO/MODERADO/ALTO/EXTREMO (percentiles del día) |
| `dos_25_barrido_historico.py` | Puntúa los 5 modelos sobre el cubo, 2015-2024, y guarda histograma diario + celdas quemadas |
| `dos_26_calibra_si.py` | Calibra el aviso de día por régimen (beta), con SEDI, BSS y bootstrap |
| `dos_27_escala_absoluta.py` | Cortes en la escala de puntuación, leídos de la climatología |
| `dos_28_calibra_celda.py` | Probabilidad por celda (isotónica sobre el censo) |
| `dos_29_calibra_estacional.py` | La misma, estratificada por mes |
| `dos_31_ventana_movil.py` | Prueba si calibrar con los K años anteriores mejora (no mejora) |
| `dos_32_producto_dos_capas.py` | La probabilidad por celda como producto P(día grande) × P(celda \| día grande) |

Todos comparten partición: calibración 2015-2021, validación 2022-2023,
test 2024. El umbral se elige solo en calibración.

## Cómo se ejecuta

```bash
source entorno.sh
python 05_iteracion2/56_calibracion/dos_25_barrido_historico.py   # ~6 min, reanudable
python 05_iteracion2/56_calibracion/dos_26_calibra_si.py
python 05_iteracion2/56_calibracion/dos_27_escala_absoluta.py
python 05_iteracion2/56_calibracion/dos_28_calibra_celda.py
python 05_iteracion2/56_calibracion/dos_29_calibra_estacional.py
python 05_iteracion2/56_calibracion/dos_31_ventana_movil.py
```

`dos_25` necesita el cubo IberFire (`TFM_DATOS`) y deja un checkpoint atómico:
si se corta, se relanza y sigue por donde iba. Los demás leen su salida.

## Los resultados

España tiene dos regímenes de incendio. Entre noviembre y marzo hay fuego el
33.9 % de los días, y marzo se parece a septiembre. El 85.6 % del fuego de
diciembre a abril cae en el noroeste, que es el 14.4 % del territorio: son
quemas pastorales, y en ellas manda la ignición y no la sequía.

El aviso de día funcionaba en invierno y no en verano. La tabla recoge el BSS
y el SEDI por régimen; el intervalo de verano cruza el cero, así que ahí el
aviso no se distingue de la climatología del día del año.

| régimen | BSS (test 2024, r10) | SEDI |
|---|---|---|
| invierno-primavera | +0.297 [+0.198, +0.381] | 0.794 |
| verano | −0.107 [−0.230, +0.022] | 0.368 |

La tabla siguiente da el significado de cada color como frecuencia observada
de quema (r10).

| nivel | % del territorio | quemadas | tasa | lift |
|---|---|---|---|---|
| BAJO | 30.1 % | 5 en diez años | 0.0000 % | 0.0× |
| MODERADO | 59.9 % | 1,062 | 0.0005 % | 0.1× |
| ALTO | 8.0 % | 2,682 | 0.0092 % | 2.6× |
| EXTREMO | 2.0 % | 9,148 | 0.1259 % | 35.6× |

EXTREMO no significa «va a arder»: arde 1 de cada 800. La celda más roja
que llega a existir en diez años no pasa del 1.6 %.

La escala absoluta sola tiene un problema medido. Con corte absoluto, el r10
no tiene ni un día sin rojo en diez años: el «dónde» está dominado por
variables estáticas (CLC, historia, pendiente) y las mismas celdas superan el
corte en enero y en agosto. Se probó una puerta del «si» para apagar el mapa:
si la puntuación del 2 % más alto del día (su p98), pasada por la calibración
del aviso, no llega al umbral, no se pinta EXTREMO. Con ella, en invierno el
44 % de los días se quedan sin rojo perdiendo el 6.2 % de los días grandes (5
o más celdas nuevas quemadas). Esa regla se revisó después con los años fuera
de calibración y con el replay: en verano ocultaba el EXTREMO en uno de cada
tres días con incendio grande, y se descartó
(`07_produccion/escala_y_cifra/README.md`). La cadena publica la escala
absoluta sin puerta y la cifra del día.

La punta, mes a mes (probabilidad de la celda peor clasificada):

| diciembre | febrero | octubre | julio |
|---|---|---|---|
| 0.28 % | 3.74 % | 3.16 % | 10.36 % |

Hay un factor 37 entre diciembre y julio, con el mismo mapa y el mismo orden.
De la celda peor clasificada de un día de diciembre solo se puede decir que
su probabilidad de arder es del 0.3 %.

## Límites de la calibración, medidos

La calibración beta no vale por celda. Sin restricción da coeficiente
negativo en −log(1−s), no es monótona y hunde la cola alta (la celda máxima
salía con 0.000 %). Se usa la isotónica, que es monótona por construcción.

La probabilidad por celda no se puede calibrar en valor absoluto. Se probó la
ventana móvil (`dos_31`) y empeora: en jun-sep el error va de 0.453 con toda
la historia a 0.947 con tres años. La razón es que la punta de julio va del
0.00 % (2023) al 12.76 % (2022), mediana 0.53 %. Esa variación no está en la
puntuación de la celda sino en la intensidad del año. La descomposición
correcta son dos factores, `P(día grande) × P(celda | día grande)`, que es el
producto de dos capas. La calibración sirve para el orden y la magnitud
típica, no para un valor absoluto por celda y día.

El bootstrap iid estrecha los intervalos un 35 % frente al de bloques de 14
días. Esto afecta a cómo se leen los IC del resto de la memoria.

Ver `docs/LIMITACIONES.md`.
