# 7 · Producción y los tres jueces

> **La cadena viva no corre desde aquí.** Sigue en `tfm-fuego-malla` (GitHub
> Actions), sellada hasta el cierre de la temporada en septiembre de 2026. Aquí
> está su código para leerlo y ejecutarlo, no para publicarlo.

## Qué pregunta responde

Si los candidatos son mejores, hay que demostrarlo **hacia adelante**, sobre
días que nadie había visto al entrenar, y con jueces que no sean el propio
modelo. Eso es lo que produce el veredicto de septiembre.

## Cómo funciona un día

```
gh_reanalisis.py     ERA5-Land incremental (hasta D-7)
malla_02b_ifs.py     previsión IFS en los 5.605 nodos (D-6 -> D+1)
riesgo_hoy.py        mapa con el modelo de PRODUCCIÓN sobre la malla
dos_riesgo_hoy.py    mapas de los CANDIDATOS: único, pareja y dónde
puntuar_effis.py     juez EFFIS   (superficie quemada, ventana de 45 días)
dos_15_...miteco.py  juez MITECO  (parte del día siguiente)
dos_16_...estaciones juez por ESTACIÓN (contra el ranking sellado del hermano)
dos_19_veredicto.py  la figura acumulada
gh_estado.py         guarda el estado en un Release
```

Dos pasadas al día: la larga a las 03:30 UTC (~1 h 15 min, descarga incluida) y
una corta a las 13:45 para recoger el parte del MITECO.

## Los tres jueces, y por qué son tres

| Juez | Latencia | Qué mide | Su defecto |
|---|---|---|---|
| **EFFIS** | ~45 días | Superficie quemada cartografiada | Lento; solo incendios grandes |
| **MITECO** | 1 día | Incidentes con medios del Estado | Grano municipal, ~4 al día |
| **Por estación** | 1 día | El ranking sellado de producción, 688 estaciones | Puntos, no celdas |

Ninguno es suficiente solo. Coincidiendo, son difíciles de discutir.

## Estado a 07/09/2026 (corrida 34099321965)

AUC medio acumulado. «malla» es el modelo de producción servido sobre la malla
del cubo; «producción» es el ranking real por estación del sistema en
operación.

| Juez | Acumulado | malla | único | r10 | pareja | producción |
|---|---|---|---|---|---|---|
| EFFIS | 9 días / 74 celdas | 0,506 | **0,745** | 0,722 (3 d) | 0,735 | — |
| MITECO | 13 días / 34 incidentes | 0,516 | **0,692** | 0,674 (5 d) | 0,651 | — |
| Por estación | 8 días / 65 positivas | 0,578 | 0,571 | 0,504 (3 d) | 0,594 | **0,645** |

Los candidatos van por delante en EFFIS y MITECO; en el juez por estación
producción sigue ganando y ningún intervalo excluye el cero (único: Δ −0,074,
IC [−0,213, +0,073]). Con 8-13 días, esto es **validación operativa en curso**.
Desde el 06/09/2026 el veredicto de la memoria no sale de aquí sino del
**replay de 2025 y 2026** (`06_comparacion/`), que cubre 250 días. Los jueces
siguen acumulando solos y se citan al cierre como lo que son.

Fuente: `publicado/historico_veredictos.csv` del repositorio de la cadena.

## Días que no cuentan, y por qué

- **27-31 de agosto de 2026**: los mapas diarios se perdieron en un incidente de
  sincronización del estado (`docs/LIMITACIONES.md` §11). No se regeneran: un
  mapa regenerado no es el mapa sellado.
- **21 de agosto de 2026**: el mapa que hay en el Release es una regeneración del
  31/08, no el operativo (se buscó el original el 06/09 en el disco congelado y
  no existe). **Se excluye del juez EFFIS** y se documenta.

## Los mapas

`dos_riesgo_hoy.py` dibuja seis paneles: arriba producción, único y pareja en
**percentil del día** (no en probabilidad: las tres escalas no son comparables,
y la pareja además multiplica dos); abajo el DÓNDE estático y las **restas**
contra producción en puntos de percentil. Los mapas de nivel se parecen todos;
es la resta la que enseña dónde cambia la decisión operativa.

`redibujar.py` regenera un mapa ya publicado con el formato actual sin
reejecutar nada, filtrando la verdad a la que se conocía ese día.

## Ejecutar con la muestra

Esta es la parte del repositorio que **sí corre con `muestras/`**: es la misma
rama que ejecuta GitHub Actions, que tampoco tiene el cubo de 29 GB.

```bash
source entorno.sh
python 07_produccion/riesgo_hoy.py
```
