# 4 · La bisagra: por qué 0,89 no medía lo que hacía falta

> **El capítulo que se lee.** Aquí está la aportación intelectual del trabajo:
> no un modelo que funciona, sino un diagnóstico de por qué el que había estaba
> mal especificado, medido y no argumentado.

## Qué pregunta responde

El modelo daba 0,89 en test y 0,56-0,64 en operación. ¿Por qué? Se persiguieron
tres explicaciones, en este orden, y **las dos primeras se descartaron con
números**. Esa es la razón de que la tercera sea creíble.

```
41  ¿Está mal medido?      -> No. Tres verdades independientes dicen lo mismo
42  ¿Es la meteorología?   -> No. Hay desplazamiento de fuente, pero no basta
43  ¿Es el train/serve?    -> No. Los proxis operativos cuestan 0,0045
44  Es la ESPECIFICACIÓN   -> La etiqueta y el muestreo definían otra pregunta
```

## 41 · La validación operativa

Cinco scores en competencia contra tres verdades-terreno independientes: focos
FIRMS, perímetros EFFIS y los partes oficiales del MITECO.

| Serie (20 días sellados) | Modelo | FWI percentil | Diferencia (IC95) |
|---|---|---|---|
| D0 sellada | 0,642 | **0,702** | −0,060 [−0,149, +0,020] |
| ranking (día cerrado) | **0,707** | 0,668 | +0,039 [−0,047, +0,116] |

**Ninguna diferencia es significativa**, y el baseline duro no es el FWI crudo:
es el **percentil local** de FWI. La conclusión anterior —«el modelo bate al
FWI», con 2-3 días de muestra— era un artefacto: al añadir días el baseline
subió de 0,540 a 0,702 y el modelo apenas se movió.

## 42 · No es la meteorología

El bloque más grande del repositorio, 24 scripts, porque esta hipótesis era la
más plausible y hubo que agotarla. Se midió el desplazamiento de fuente entre
ERA5-Land y AEMET feature a feature, se probó el IFS, el híbrido y las
correcciones por cuantiles, y se construyó el pipeline de malla que después
sería la base de la iteración 2.

Lo firme que salió de aquí:

- **La precipitación no coincide**: +8,00 mm en 30 días, y de ahí se propaga
  todo hasta un FWI 10 puntos más bajo.
- **La climatología del percentil cruzaba fuentes**: numerador de AEMET,
  referencia del cubo. La feature satura — 12,6 % de las filas en percentil
  ≥99,9 en producción frente al 1,0 % en entrenamiento.

Y lo que **no** cerró, que también se dice: la comparación de AUC entre brazos
falló el control (−0,113 de discrepancia frente a un efecto de −0,012). Ver
[`docs/LIMITACIONES.md`](../docs/LIMITACIONES.md).

`prototipo_TFM_fuego/` guarda la versión de los módulos de malla que vivía en
el repositorio original: es la que produjo los números de este capítulo, y
diverge de la definitiva que hoy sirve en `01_datos/` y `07_produccion/`.

## 43 · No es el train/serve

La auditoría de las 46 features una por una, y la ablación de los proxis que
usa producción:

| Proxy operativo | Δ AUC en test 2020 |
|---|---|
| Satélite → climatología mensual congelada | −0,004 |
| Rayos → 0 | −0,000 |
| **Los dos, como hoy en producción** | **−0,0045** |

No explica el hueco. De regalo: las cinco features satelitales no aportan nada
ni siendo reales.

## 44 · Es la especificación

Dos medidas, y con ellas el trabajo cambia de dirección:

- El **98,8 %** de las celdas del conjunto v1 tenía exactamente un 25 % de
  positivos. El modelo no podía aprender qué celda es más peligrosa que otra:
  solo qué día lo es.
- El **AUC caso-control de los dos diseños es idéntico (0,92)** y el **AUC
  dentro del día los separa (0,744 frente a 0,828)**. La métrica de desarrollo
  era ciega a la diferencia.

No es un problema de ajuste. Es que la pregunta que se entrenó —«¿es hoy
peligroso en esta celda?»— no es la que se evalúa en operación —«¿cuál de las
498.530 celdas arde hoy?»—.
