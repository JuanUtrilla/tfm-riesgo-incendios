# 5.6 · Calibración: del orden de las celdas a una probabilidad

## Qué pregunta responde

El modelo **ordena** celdas muy bien, pero un mapa que pinta de rojo el 2 %
superior de cada día se lee como «estas celdas van a arder» — y eso es falso los
doce meses del año. Este capítulo convierte el orden en tres cosas publicables:

1. **¿Se pinta hoy?** Un aviso de día calibrado (el «si»).
2. **¿Qué significa el rojo?** Una escala absoluta, no un percentil del día.
3. **¿Cuánta probabilidad tiene esta celda?** Un número, no un color.

## Por qué no valía lo que ya había

`replay_si.py` fijó el umbral del aviso con jun-ago de 2025: 92 días, 86 con
fuego. Con esa prevalencia el umbral no se estima, **se extrapola**, y la tabla
2×2 es degenerada. Aquí hay **3.653 días** con la variación real de la tasa
base, días de cero fuego incluidos.

## Qué hay aquí

| Script | Qué hace |
|---|---|
| `dos_14_cortes.py` | Los cortes originales BAJO/MODERADO/ALTO/EXTREMO (percentiles del día) |
| `dos_25_barrido_historico.py` | Puntúa los 5 modelos sobre el cubo, 2015-2024, y guarda histograma diario + celdas quemadas |
| `dos_26_calibra_si.py` | Calibra el aviso de día por régimen (beta), con SEDI, BSS y bootstrap |
| `dos_27_escala_absoluta.py` | Cortes en la escala de puntuación, leídos de la climatología |
| `dos_28_calibra_celda.py` | Probabilidad por celda (isotónica sobre el censo) |
| `dos_29_calibra_estacional.py` | La misma, estratificada por mes |

Todos comparten partición: **calibración 2015-2021 · validación 2022-2023 ·
test 2024**. El umbral se elige solo en calibración.

## Cómo se ejecuta

```bash
source entorno.sh
python 05_iteracion2/56_calibracion/dos_25_barrido_historico.py   # ~6 min, reanudable
python 05_iteracion2/56_calibracion/dos_26_calibra_si.py
python 05_iteracion2/56_calibracion/dos_27_escala_absoluta.py
python 05_iteracion2/56_calibracion/dos_28_calibra_celda.py
python 05_iteracion2/56_calibracion/dos_29_calibra_estacional.py
```

`dos_25` necesita el cubo IberFire (`TFM_DATOS`) y deja un checkpoint atómico:
si se corta, se relanza y sigue por donde iba. Los demás leen su salida.

## Los resultados

**España tiene dos regímenes de incendio.** nov-mar: 33,9 % de días con fuego, y
marzo casi como septiembre. El **85,6 %** del fuego de dic-abr cae en el
noroeste, que es el 14,4 % del territorio: quemas pastorales, conductor de
ignición y no de sequía.

**El aviso de día funciona en invierno y no en verano.**

| régimen | BSS (test 2024, r10) | SEDI |
|---|---|---|
| invierno-primavera | **+0,297 [+0,198, +0,381]** | 0,794 |
| verano | −0,107 [−0,230, **+0,022**] | 0,368 |

En verano el intervalo **cruza el cero**: el «si» no se distingue de la
climatología del día del año. No es que sea peor — es que no aporta.

**Qué significa cada color** (frecuencia observada de quema, r10):

| nivel | % del territorio | quemadas | tasa | lift |
|---|---|---|---|---|
| BAJO | 30,1 % | **5 en diez años** | 0,0000 % | 0,0× |
| MODERADO | 59,9 % | 1.062 | 0,0005 % | 0,1× |
| ALTO | 8,0 % | 2.682 | 0,0092 % | 2,6× |
| EXTREMO | 2,0 % | 9.148 | **0,1259 %** | **35,6×** |

**EXTREMO no es «va a arder»: es 1 de cada 800.** Y la celda más roja que llega
a existir en diez años no pasa del **1,6 %**.

**La escala absoluta no basta.** Con corte absoluto, `r10` no tiene **ni un solo
día sin rojo** en diez años: el «dónde» está dominado por variables estáticas
(CLC, historia, pendiente) y las mismas celdas superan el corte en enero y en
agosto. Hace falta la puerta del «si» para apagar el mapa — y con ella, en
invierno el 44 % de los días se quedan sin rojo perdiendo solo el **6,2 %** de
los días grandes.

**La punta, mes a mes** (probabilidad de la celda peor clasificada):

| diciembre | febrero | octubre | julio |
|---|---|---|---|
| **0,28 %** | 3,74 % | 3,16 % | **10,36 %** |

Un factor **37** entre diciembre y julio, con el mismo mapa y el mismo orden.
Esa es la frase publicable: *«esta es la celda peor clasificada de hoy; su
probabilidad de arder es del 0,3 %»*.

## Lo que no funciona, y está medido

- **La calibración beta no vale por celda**: sin restricción da coeficiente
  negativo en −log(1−s), no es monótona y hunde la cola alta (la celda máxima
  salía con 0,000 %). Se usa isotónica, monótona por construcción.
- **Julio infrapredice por 7×** incluso con calibración por mes (0,88 % contra
  6,22 % real). La ventana 2015-2021 no contiene ningún año extremo y el periodo
  de evaluación sí (2022). Arreglo pendiente: ventana móvil.
- **El bootstrap iid estrecha los intervalos un 35 %** frente al de bloques de
  14 días. Afecta a cómo se leen los IC del resto de la memoria.

Ver `docs/LIMITACIONES.md`.
