# Glosario

Términos que aparecen en todo el repositorio. Las fuentes de datos están
detalladas en [`01_datos/README.md`](../01_datos/README.md).

## Fuentes

- **IberFire (el cubo)** — datacubo diario de la Península en malla regular,
  29 GB. Es la fuente de las features de entrenamiento de las dos iteraciones.
- **EGIF** — Estadística General de Incendios Forestales. Registro oficial de
  **igniciones** (punto y fecha de inicio). Etiqueta de la iteración 1.
- **EFFIS** — European Forest Fire Information System. Perímetros de
  **superficie quemada**. Etiqueta de la iteración 2 y referencia de las evaluaciones.
- **MITECO** — parte diario oficial de incendios. Se publica al día siguiente, frente a los ~45 días de desfase de EFFIS; se usó como segunda referencia durante el desarrollo.
- **ERA5-Land** — reanálisis meteorológico de ECMWF, ~9 km. Alimenta la malla.
- **IFS** — modelo de previsión de ECMWF (vía Open-Meteo). La rama de
  *previsión* para los días D y D+1, donde el reanálisis todavía no existe.
- **FIRMS** — detecciones de focos activos por satélite (VIIRS), NASA. Feature
  de entorno de fuego activo y capa de verdad en los mapas.
- **AEMET** — observación y climatología por estación. La fuente meteorológica
  de la producción de la iteración 1.

## Modelado

- **Celda** — unidad espacial del cubo. La Península peninsular son **498.530**
  celdas.
- **Caso-control temporal** — muestreo de la iteración 1: para cada celda con
  ignición se toman como negativos **otros días de esa misma celda**. Enseña a
  distinguir días, no celdas.
- **Muestreo del mismo día** — muestreo de la iteración 2: los negativos son
  **otras celdas del mismo día**. Es la pregunta que se hace en operación.
- **Pseudoausencias** — celdas-día sin fuego que se usan como negativos. Su
  número relativo a los positivos es el **ratio** (1:3, 1:10, 1:30…), medido
  en `05_iteracion2/57_ablaciones/`. El DÓNDE por celda no las usa: es censo de
  las 498.530 celdas.
- **DÓNDE** — modelo que ordena celdas dentro de un día.
- **CUÁNDO** — modelo que ordena días. El producto de los dos es la «pareja».
- **AUC caso-control** — AUC sobre el conjunto de test tal como se muestreó.
  Fue la métrica de desarrollo de la iteración 1, y es **ciega** a la diferencia
  entre los dos diseños de muestreo (0,92 en ambos).
- **AUC dentro del día** — AUC calculado **día a día**, ordenando las celdas de
  ese día. Es la métrica que corresponde a la pregunta operativa, y sí separa
  los dos diseños (0,744 frente a 0,828).
- **Percentil ponderado por hectáreas** — en qué percentil del mapa cayó lo que
  ardió, pesando cada celda por su superficie quemada. Mide si el modelo acierta
  con los fuegos **grandes**, que es lo que importa operativamente.

## Producción

- **FWI** — Fire Weather Index canadiense (Van Wagner, 1987), con sus
  componentes FFMC, DMC, DC, ISI y BUI. Implementación en
  `01_datos/comun/fwi_canadiense.py`.

- **Sellado** — congelado a propósito durante la temporada: producción sirvió la
  versión con la que empezó, aunque después se le encontrara un defecto, para
  no alterar la serie publicada. Ver `docs/LIMITACIONES.md`.
