# Predicción diaria de riesgo de incendio forestal en la España peninsular

Repositorio definitivo del TFM. Contiene el trabajo completo: ingesta y
tratamiento de datos, el primer modelo y su puesta en producción, el
diagnóstico de por qué falló al medirlo en operación, el rediseño, y el
sistema que se está validando durante la temporada de 2026.

## El resultado, en un párrafo

El primer modelo alcanzó **AUC 0,89** en su conjunto de test y, servido a
diario contra superficie quemada real, cayó a **0,56-0,64**. La discrepancia
no era sobreajuste ni cambio de fuente meteorológica: era un **error de
especificación**. El muestreo y la etiqueta de entrenamiento definían la
pregunta *«¿es hoy un día peligroso en esta celda?»*, distinta de la que se
evalúa en operación, *«¿cuál de las 498.530 celdas arde hoy?»*. A partir de
ese diagnóstico se rediseñó el conjunto de entrenamiento y se construyó un
segundo sistema. Reproducidas día a día las temporadas de **2025 y 2026** con
la previsión disponible cada víspera (250 días, protocolo prerregistrado), el
modelo único con etiqueta EFFIS y ratio 1:10 pasa de **0,743 a 0,807** en 2025
(Δ +0,064, IC95 [+0,027, +0,101]) y de **0,658 a 0,762** en 2026 (Δ +0,104,
IC95 [+0,065, +0,144]), ganando dos de cada tres días; y predecir a un día
vista cuesta lo mismo que conocer el tiempo real (Δ −0,003, IC cruza el cero).

## Cómo se lee este repositorio

El árbol es el orden de lectura de la memoria, no el orden en que ocurrieron
las cosas.

| Carpeta | Qué contiene |
|---|---|
| `00_marco/` | El hilo conductor, el glosario y el diagrama de las dos iteraciones |
| `01_datos/` | Ingesta y tratamiento de las nueve fuentes. Marco común a las dos iteraciones |
| `02_eda/` | Análisis exploratorio sobre el split de entrenamiento |
| `03_iteracion1/` | Etiqueta EGIF, muestreo caso-control, entrenamiento y puesta en producción |
| `04_bisagra/` | **El capítulo clave**: por qué 0,89 no medía lo que hacía falta |
| `05_iteracion2/` | Rediseño con etiqueta EFFIS, DÓNDE × CUÁNDO, y las ablaciones |
| `06_comparacion/` | Los dos sistemas sobre los mismos días y las mismas fuentes, y **el replay de 2025-2026** que da el veredicto |
| `07_produccion/` | La cadena diaria y los tres jueces en vivo (validación operativa en curso) |
| `docs/` | [Bitácora cronológica](docs/BITACORA.md), [trazabilidad](docs/TRAZABILIDAD.md), [limitaciones](docs/LIMITACIONES.md), [veredicto del replay](docs/REPLAY_VEREDICTO.md), [estructura](docs/ESTRUCTURA.md), [lo que falta](docs/PENDIENTE.md) y [la memoria](docs/MEMORIA/README.md) |
| `muestras/` | Recorte de julio de 2026 para ejecutar el pipeline sin descargar 44 GB |

## Por dónde empezar a leer

- [`00_marco/README.md`](00_marco/README.md) — el diagrama del hilo y las tres
  preguntas que separan las dos iteraciones.
- [`docs/BITACORA.md`](docs/BITACORA.md) — **la historia completa**, del primer
  modelo en producción al veredicto: qué se midió, qué se descartó y qué
  conclusión hubo que retirar.
- [`docs/TRAZABILIDAD.md`](docs/TRAZABILIDAD.md) — **cada número de la memoria
  con el script que lo produjo** y lo que hace falta para reejecutarlo.
- [`docs/REPLAY_VEREDICTO.md`](docs/REPLAY_VEREDICTO.md) — **el resultado
  final**: seis modelos, dos temporadas, dos condiciones, con su prerregistro
  en `TRAZABILIDAD.md`.
- [`docs/LIMITACIONES.md`](docs/LIMITACIONES.md) — lo que no cierra y lo que se
  dejó roto a propósito.
- [`docs/ESTRUCTURA.md`](docs/ESTRUCTURA.md) — cómo está montado el repo y por
  qué el código no se ha reescrito.
- [`docs/PENDIENTE.md`](docs/PENDIENTE.md) — lo que aún le falta, en orden de
  importancia, y los puntos a decidir en equipo.

## Reproducibilidad

Los datos crudos (cubo IberFire 29 GB, reanálisis, históricos AEMET) no están
en el repositorio: se descargan con los scripts de `01_datos/`. Para trabajar
sin esperar a las descargas, `muestras/` lleva un recorte de julio de 2026 con
el que el pipeline se ejecuta de punta a punta en un portátil.

## Aviso sobre producción

La cadena diaria que puntúa los modelos en vivo **corre en otro repositorio**
(`tfm-fuego-malla`, GitHub Actions) y está sellada hasta el cierre de la
temporada, en septiembre de 2026. Aquí está su código y una copia
documental de su workflow con el `cron` desactivado, para que se pueda leer
sin riesgo de alterar la comparación en curso.

## Autoría

Trabajo Fin de Máster, 2026. Módulo de aprendizaje automático de un sistema
conjunto de apoyo a la gestión de incendios (predicción de riesgo, detección
por visión y consulta de partes). Datos: IberFire (CC-BY 4.0), EGIF vía Civio
(CC BY-SA 3.0), EFFIS y ERA5-Land (Copernicus), FIRMS (NASA), AEMET OpenData.
