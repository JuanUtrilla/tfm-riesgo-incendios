# 5.6 · Calibración: del orden de las celdas a una probabilidad

## Qué pregunta responde

El modelo ordena bien las celdas, pero un mapa que pinta de rojo el 2 %
superior de cada día se lee como «estas celdas van a arder», y eso es falso
casi todo el año. Este capítulo convierte el orden en tres cosas publicables:

1. Si se pinta hoy: un aviso de día calibrado, el semáforo.
2. Qué significa el rojo: una escala fija en la puntuación, en vez del percentil del día.
3. Cuánta probabilidad tiene una celda: un número, no solo un color.

## Con qué se calibra

`replay_si.py` (capítulo 6) fijó un primer umbral del aviso con un solo verano
(jun-ago de 2025: 92 días, 86 con fuego), demasiado pocos días sin fuego para
estimar nada. Aquí se calibra con 3.653 días del cubo, 2015-2024, con la
variación real de la tasa base y los días de cero fuego incluidos.

## Qué hay aquí

| Script | Qué hace |
|---|---|
| `dos_14_cortes.py` | Los cortes originales BAJO/MODERADO/ALTO/EXTREMO (percentiles del día) |
| `dos_25_barrido_historico.py` | Puntúa los 5 modelos sobre el cubo, 2015-2024, y guarda histograma diario + celdas quemadas |
| `dos_26_calibra_si.py` | Calibra el aviso de día por régimen (beta), con SEDI, BSS y bootstrap |
| `dos_27_escala_absoluta.py` | Cortes en la escala de puntuación, leídos de la climatología |
| `dos_28_calibra_celda.py` | Probabilidad por celda (isotónica sobre el censo) |
| `dos_29_calibra_estacional.py` | La misma, estratificada por mes |
| `dos_31_ventana_movil.py` | ¿Ayuda calibrar con los K años anteriores? (no) |
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
33,9 % de los días, y marzo se parece a septiembre. El 85,6 % del fuego de
diciembre a abril cae en el noroeste, que es el 14,4 % del territorio: quemas
pastorales, es decir, ignición y no sequía.

El aviso de día funciona en invierno y no en verano.

| régimen | BSS (test 2024, r10) | SEDI |
|---|---|---|
| invierno-primavera | +0,297 [+0,198, +0,381] | 0,794 |
| verano | −0,107 [−0,230, +0,022] | 0,368 |

En verano el intervalo cruza el cero: el aviso no se distingue de la
climatología del día del año.

Qué significa cada color (frecuencia observada de quema, r10):

| nivel | % del territorio | quemadas | tasa | lift |
|---|---|---|---|---|
| BAJO | 30,1 % | 5 en diez años | 0,0000 % | 0,0× |
| MODERADO | 59,9 % | 1.062 | 0,0005 % | 0,1× |
| ALTO | 8,0 % | 2.682 | 0,0092 % | 2,6× |
| EXTREMO | 2,0 % | 9.148 | 0,1259 % | 35,6× |

EXTREMO no significa «va a arder»: arde 1 de cada 800. Y la celda más roja
que llega a existir en diez años no pasa del 1,6 %.

La escala absoluta sola no basta. Con corte absoluto, el r10 no tiene ni un
día sin rojo en diez años: el «dónde» está dominado por variables estáticas
(CLC, historia, pendiente) y las mismas celdas superan el corte en enero y en
agosto. Hace falta la puerta del «si» para apagar el mapa: si la puntuación del 2 %
más alto del día (su p98), pasada por la calibración del aviso, no llega al
umbral, no se pinta EXTREMO. Con ella, en invierno el 44 % de los días se
quedan sin rojo perdiendo el 6,2 % de los días grandes (5 o más celdas
nuevas quemadas). Es la regla que aplica `07_produccion/mapa_r10_semaforo.py`.

La punta, mes a mes (probabilidad de la celda peor clasificada):

| diciembre | febrero | octubre | julio |
|---|---|---|---|
| 0,28 % | 3,74 % | 3,16 % | 10,36 % |

Un factor 37 entre diciembre y julio, con el mismo mapa y el mismo orden. Lo
que se puede decir de la celda peor clasificada de un día de diciembre es que
su probabilidad de arder es del 0,3 %.

## Límites de la calibración, medidos

- La calibración beta no vale por celda: sin restricción da coeficiente
  negativo en −log(1−s), no es monótona y hunde la cola alta (la celda máxima
  salía con 0,000 %). Se usa la isotónica, que es monótona por construcción.
- La probabilidad por celda no se puede calibrar en valor absoluto. Se probó la
  ventana móvil (`dos_31`) y empeora: en jun-sep el error va de 0,453 con
  toda la historia a 0,947 con tres años. La razón es que la punta de julio va
  del 0,00 % (2023) al 12,76 % (2022), mediana 0,53 %: esa variación no está
  en la puntuación de la celda, está en la intensidad del año. La descomposición
  correcta son dos factores, `P(día grande) × P(celda | día grande)`, que es el
  producto de dos capas. La calibración sirve para el orden y la magnitud típica,
  no para un valor absoluto por celda y día.
- El bootstrap iid estrecha los intervalos un 35 % frente al de bloques de
  14 días. Afecta a cómo se leen los IC del resto de la memoria.

Ver `docs/LIMITACIONES.md`.
