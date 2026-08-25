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

## Estado a 25/08/2026

| Juez | Acumulado | Resultado |
|---|---|---|
| MITECO | 3 días / 6 incidentes | único **0,807** > pareja 0,771 > malla 0,598 |
| EFFIS | 1 día (4 celdas, 122 ha) | pareja 90,6 · único 73,5 · malla 39,2 (percentil) |
| Por estación | 1 día, 688 estaciones | pareja 0,607 · único 0,600 · malla 0,572 · producción 0,507 |

El veredicto acumulado sigue diciendo, a propósito, *«mejor de media, pero con
17 días no se distingue del ruido»*.

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
