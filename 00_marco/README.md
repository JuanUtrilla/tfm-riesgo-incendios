# El marco: un trabajo que dio la vuelta a mitad de camino

## El hilo

```
                     FUENTES Y TRATAMIENTO  (01_datos, 02_eda)
                     comunes a las dos iteraciones
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                                                 │
  ITERACIÓN 1  (03)                                 ITERACIÓN 2  (05)
  etiqueta EGIF (ignición)                          etiqueta EFFIS (superficie
  muestreo caso-control temporal                    quemada, la misma con la que
  «¿es hoy peligroso en esta celda?»                se valida)
  AUC test 0,89                                     negativos del MISMO día
        │                                           «¿cuál de las 498.530 celdas
        │  se pone en producción                     arde hoy?»
        │  687 estaciones, mapa diario                       │
        ▼                                                    │
  BISAGRA  (04)                                              │
  operación: AUC 0,56-0,64                                   │
   41  se mide contra tres verdades independientes           │
   42  ¿es la meteorología? → NO (ERA5, IFS, híbrido, FWI)   │
   43  ¿es el train/serve?  → NO (auditoría de 46 features)  │
   44  es la ESPECIFICACIÓN → etiqueta y muestreo            │
        └───────────────── obliga a ────────────────────────┘
                                 │
                                 ▼
                    COMPARACIÓN  (06)  mismos días, mismas fuentes
                                 │
                                 ▼
                    PRODUCCIÓN Y JUECES  (07)  veredicto de septiembre
```

## Las dos reglas de redacción

**Se escribe el hilo lógico, no el cronológico.** Importa en qué orden hay que
leer las cosas para que la conclusión sea inevitable, no en qué orden
ocurrieron. Los callejones sin salida no van a la memoria; van a
`docs/BITACORA.md`, que es donde este repositorio se diferencia de la memoria.

**El primer modelo no es un error, es el grupo de control.** Sin él no hay
forma de demostrar que el rediseño hacía falta ni de cuantificar lo que
aportó. El fallo, medido y explicado, es un resultado.

El desarrollo completo de estas dos reglas, con el texto de la sección bisagra
ya redactado, está en [`docs/MEMORIA/hilo_conductor.md`](../docs/MEMORIA/hilo_conductor.md).

## Las tres preguntas que separan las dos iteraciones

|  | Iteración 1 | Iteración 2 |
|---|---|---|
| **Qué es un positivo** | ignición registrada en EGIF | celda con superficie quemada en EFFIS |
| **Qué es un negativo** | la misma celda en otros días (caso-control temporal) | otras celdas del **mismo día** |
| **Contra qué se valida** | superficie quemada, partes de MITECO | lo mismo — y ahora coincide con el entrenamiento |

La tercera fila es el trabajo entero: en la iteración 1 la etiqueta de
entrenamiento y la de validación eran cosas distintas, y la métrica de
desarrollo era ciega a esa diferencia.
