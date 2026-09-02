# Trazabilidad de `seccion_modelo.tex`

Cada afirmación cuantitativa de la sección, con el artefacto o el fichero que la
produce. Convención de rutas: sin prefijo = `tfm-riesgo-incendios/`;
`M/` = `TFM_fuego_malla/`; `T/` = `TFM_fuego/`. Escrito el 31/08/2026.

**Juego de cifras oficial adoptado:** el de la **ventana FIRMS de 7 días**
(`M/salida/dos_09_temporada2026.{csv,json}` y `dos_19_veredicto.json`), que es el
que usan el README y `docs/TRAZABILIDAD.md` del repositorio. Se descarta el juego
de ventana 5 (`dos_09_ventana5.json`) que usaba `docs/CONTEXTO_SECCION_MODELO.md`.
Esta decisión resuelve la pregunta 1 de `PLAN_SECCION.md` §6 y es la única que la
sección da por zanjada por su cuenta; todas las demás quedan marcadas abajo.

## Datos y preparación

| Afirmación en el texto | Evidencia |
|---|---|
| 498.530 celdas peninsulares de 1 km | `02_eda/ANALISIS_CUBO.md` §2 |
| «en operacion cae a 0,56-0,64» | rango resumen de `00_marco/README.md:21` y `docs/CONTEXTO_SECCION_MODELO.md:126`; los dos extremos son 0,568 (ranking sellado por estacion, `M/salida/rankings_justo.json`) y 0,647 (produccion dentro del dia en 2026, `M/salida/dos_09_temporada2026.json`) |
| Cubo IberFire: NetCDF 30,3 GB, 261 variables, 1.188×920, EPSG:3035, 2007-2024 | `02_eda/ANALISIS_CUBO.md` §0; `01_datos/cubo/extraer_features_cubo.py` |
| 45 de las 46 variables salen del cubo | `01_datos/cubo/extraer_features_cubo.py`; `02_eda/ANALISIS_CUBO.md` §0 |
| El cubo es ERA5-Land remuestreado: corr 0,987-0,998, sesgo nulo | `T/dataset/malla_02_vs_cubo.json` (lo produce `04_bisagra/42_no_es_la_meteo/prototipo_TFM_fuego/malla_02_vs_cubo.py`) |
| Precipitación del cubo = ERA5-Land / 2,02 | ídem |
| El FWI del cubo son las 13 UTC | `M/salida/verificar_hora_fwi.json`, de `04_bisagra/42_no_es_la_meteo/verificar_hora_fwi.py` |
| EGIF: 19.561 pares celda-día 2015-2020; 888 / 226 / 23 registros en 2021-2023 | `02_eda/ANALISIS_CUBO.md` §1; `03_iteracion1/31_muestreo/muestrear_dataset.py:9-16` |
| EFFIS `is_fire` ≥5 ha, diario, 2008-2024 | `02_eda/ANALISIS_CUBO.md` §1 |
| Solape EGIF/EFFIS: 2,6 % (5-25 ha) y 59 % (>500 ha) | `02_eda/ANALISIS_CUBO.md` §1 |
| Prevalencia 2-6 por 100.000 en verano normal; 37,96 en el verano de 2022 | `02_eda/ANALISIS_CUBO.md` §2 (tabla por año, columna verano) |
| Las 46 variables y sus cinco bloques | `01_datos/comun/features.py:28-43`; `muestras/modelos/xgb_v2_prototipo_features.json` (46 nombres, verificado) |
| Regla antifuga con desigualdad estricta `(vf_d >= d-7) & (vf_d < d)` | `03_iteracion1/32_features/extraer_features_historia.py:117`; `05_iteracion2/53_features/dos_03_features.py:147-151` |
| FIRMS nunca es etiqueta | `01_datos/firms/firms_api.py`; `01_datos/firms/extraer_historia_firms_v3.py` |

## Del problema al modelo

| Afirmación | Evidencia |
|---|---|
| Iteración 1: caso-control celda fija, 1:3, buffer 12,5 km y ±10 días | `03_iteracion1/31_muestreo/muestrear_dataset.py:37-39` |
| XGBoost: 3.000 árboles, profundidad 6, lr 0,05, parada temprana 100, semilla 42 | `03_iteracion1/33_train/entrenar_modelo.py:44,61-65`; heredados sin cambio en `05_iteracion2/55_train/dos_05_modelos.py:71-75` |
| Partición temporal sin solape (2015-2018 / 2019 / 2020 y 2015-2020 / 2021 / 2022) | `muestrear_dataset.py:46`; `T/dataset/metricas_v4.json` (`"protocolo"`) |
| AUC-ROC 0,8906 IC95 [0,8644, 0,9136] en test 2022 | `T/dataset/metricas_v4.json` (`test_2022.v4`) |
| FWI en percentil 0,7528 en el mismo test | ídem (`test_2022["FWI percentil"]`) |
| Medianas del train: FWI pctl 84,33 / 47,00; `popdens` y `dist_carreteras` idénticas | `02_eda/figuras/eda_resumen.log`; **es la fuente de la figura F3** |
| 53.563 filas de train, 13.577 positivos | `02_eda/figuras/eda_resumen.log` línea 1 |
| 98,8 % de celdas con exactamente 25 % de positivos | `M/CUANDO_Y_DONDE.md:38` («celdas con EXACTAMENTE 0,25 → 98,8 %»), producido por `04_bisagra/44_es_la_especificacion/dos_00_cubo_etiquetas.py` |
| Caso-control 0,922 y 0,916 | `M/salida/dos_05_metricas.json` → `auc_global_test_caso_control`: `cuando_46_en_test_cuando` 0,9216 y `donde_46_en_test_donde` 0,9162 |
| Dentro del día 0,745 vs 0,828, separación 0,082 | ídem, `2020_egif`: `cuando_46` 0,7455 y `donde_dia` 0,8275. **Nota:** `docs/TRAZABILIDAD.md:83` cita «0,744 vs 0,828», que es `prod` (0,7443) frente a `donde_dia`; aquí se usa el par de modelos que solo difiere en el muestreo, que es lo que la figura F4 dibuja |
| v4 − v3 = −0,0012 [−0,0032, +0,0007] | `T/dataset/metricas_v4.json` (`v4_menos_v3`) |
| Congelar el satélite cuesta −0,0041 [−0,0064, −0,0006] | `T/dataset/ablacion_proxies_operativos.json` (`SAT mismatch`) |
| Modelo nativo de estación, descartado | `03_iteracion1/33_train/modelo_nativo_estacion.py`; `T/dataset/nativo_estacion_metricas.json` |
| Servir con ERA5-Land nativo, descartado | `T/dataset/experimento_b_resultados.json`; `docs/LIMITACIONES.md` §2 |
| Iteración 2: positivo = primer día EFFIS, tope 30 por (día, bloque 100 km) | `05_iteracion2/52_muestreo/dos_12_muestrear_effis.py:47,79,91` |
| Negativos del mismo día, otra celda al azar | `dos_12_muestrear_effis.py:111-118` |
| Split train 2015-2022 / val 2023 / test 2024; 2026 externo | `dos_12_muestrear_effis.py:49` |
| Banco `eval_dia`: 1.000 celdas al azar por día + hasta 60 EFFIS | `dos_12_muestrear_effis.py:51,120-134` |
| El modelo único usa las mismas 46 variables (`FULL`) | `dos_05_modelos.py:76`; `05_iteracion2/55_train/dos_13_modelos_effis.py:83-85` |
| Tres variables concurrentes (NDVI, LST, humedad superficial), 4,6 % del gain | `05_iteracion2/57_ablaciones/dos_24_auditoria_fugas.py` → `M/salida/dos_24_auditoria_fugas.json` (`modelos[0].pct_gain_concurrente` = 4,62; detalle «lst 1,9 %; ndvi 1,6 %; swi010 1,2 %») |
| Al servir, esas capas se sustituyen por climatología mensual 2020-24 | `07_produccion/riesgo_hoy.py:187` (`mensual()`) |
| Fuga de FIRMS: la API cuenta 5 días hacia adelante; corregida el 31/08/2026 | `06_comparacion/comparar_rankings.py:74,77-102` (`VENTANA_FIRMS` y docstring de `firms_dia`); `docs/BITACORA.md:293-312` |
| Producción y entrenamiento estaban limpios | `extraer_features_historia.py:117`; `07_produccion/riesgo_hoy.py` (llama sin fecha) |
| La ventaja pasa de +0,037 a +0,105 tras el arreglo | `M/salida/rankings_justo_CON_FUGA.json` frente a `M/salida/dos_19_veredicto.json`; `docs/BITACORA.md` §«cifras limpias» |

## Evaluación

| Afirmación | Evidencia |
|---|---|
| Definición del AUC dentro del día y del lift del decil | `dos_05_modelos.py:107-140` (`metricas_dia`), docstring `:39-46` |
| Bootstrap de 2.000 remuestreos de días, nunca de celdas | `dos_05_modelos.py:90,148`; `dos_13_modelos_effis.py:36,43-54` |
| Test 2024: prod 0,792 · único 0,809 [−0,024, +0,063] · pareja 0,762 · FWI 0,636 | `M/salida/dos_13_metricas.json` (`"2024"`) |
| Temporada 2026: 74 días, 4.843 celdas quemadas, 260.753 ha | `M/salida/dos_09_temporada2026.csv` (74 filas, suma de `celdas_quemadas` y `area_ha`) |
| prod 0,647 · único 0,752 (+0,105 [+0,067, +0,146], 70 %) · 1:10 0,759 (+0,112 [+0,074, +0,151], 76 %) · pareja 0,705 | `M/salida/dos_19_veredicto.json`; log íntegro en `M/salida/dos_09_ventana7.log:463-486`. Los IC del texto son los de `dos_19_veredicto.json`, que remuestrea aparte; el log da IC ligeramente distintos ([+0,065, +0,145] y [+0,072, +0,151]) por el sorteo del bootstrap |
| Escalera del ratio en 2026: 0,748 / 0,759 / 0,753 / 0,748 / 0,745 | `M/salida/dos_09_ventana7.log:482-486`. **[PENDIENTE: `M/salida/dos_18_ratio.json` es del 23/08, anterior a la corrección de la fuga, y no cubre 2026; si se quiere citar `dos_18` hay que reejecutarlo]** |
| Tabla 1: AUC-PR en validación de 0,950 (1:3) a 0,737 (1:100) | `M/salida/dos_18_ratio.json` → `modelos[*].val_aucpr` |
| Ruido del sorteo de negativos = 0,004 | `M/salida/dos_18_ratio.json` → `2024["donde_dia_effis (dos_13, 1:3)"].dif_vs_ref` = 0,00435 frente a `donde_dia_effis_r3`, que es la misma receta re-sorteada |
| Percentil ponderado por hectáreas 76,2 → 79,8 → 81,9 | `M/salida/dos_09_ventana7.log:463,480,483` |
| Entrenar solo con verano: −0,028 | `M/salida/dos_08_verano.json` (`2020_egif_donde_verano.dif_vs_ref` = −0,0282 [−0,0381, −0,0187]) |
| Capa «ya quemado»: −0,021 | `M/salida/dos_17_capa_quemado.json` (`todo_prod.capas.effis_30.dif_auc` = −0,0213 [−0,0389, −0,0083]) |
| FIRMS aporta +0,036 | `M/salida/dos_09_ventana7.log:463,467`: prod 0,647 frente a `prod_sin_firms` 0,611 |
| Gain del único: 29,2 % `n_fuegos_10km_mismomes_hist`, 14,3 % CLC, 5,8 % pendiente | `figs/f6_gain.json`, calculado por `figs/scripts/f6_donde_mira.py` desde `muestras/modelos/donde_dia_effis.ubj` |
| Gain de producción: 20,5 % `fwi_anom_sigma` | ídem, desde `muestras/modelos/xgb_v2_prototipo.ubj` |
| Curva de desfase del top-2 % | `06_comparacion/dos_23_vispera.py` → `M/salida/dos_23_vispera.json`, filas con `k = 0.02` |

## Limitaciones

| Afirmación | Evidencia |
|---|---|
| Comparación justa previsión-contra-previsión, 19 días: ranking 0,568 · único 0,605 (gana 14/19, [−0,020, +0,085]) · por celda 0,755 vs 0,669 ([+0,022, +0,156]) | `M/salida/rankings_justo.json` (`auc_ranking`, `candidatos_effis.unico`) |
| Desajuste de ventana FIRMS: entrenó con 7 días, servicio pedía 5; corregirlo aporta +0,003 | `extraer_features_historia.py:117` frente a `07_produccion/riesgo_hoy.py`; medición en `M/salida/dos_09_ventana5.json` (prod 0,644) contra `dos_09_temporada2026.json` (prod 0,647) |
| No hay `environment.yml` ni `requirements.txt` | comprobado: no existen en el repositorio; `docs/PENDIENTE.md` §1 |
| No hay curva PR ni calibración de la iteración 2 | existen `pr_curves_test.png` y `calibracion_test.png` para la iteración 1 (`entrenar_modelo.py:193,201`); no hay equivalente en `dos_05`/`dos_13` |
| La variante 1:10 ya está servida y recogida por los tres jueces | `M/dos_riesgo_hoy.py` (puntúa `prob_r10`); `docs/BITACORA.md` entrada del 31/08/2026 |

## Marcas pendientes de verificación por el autor

Ninguna cifra de la sección está estimada ni inferida: todas salen de los
artefactos de las tablas anteriores. Lo que queda abierto es de decisión, no de
dato.

- **[PENDIENTE: procedencia de `xgb_v2_prototipo`]** — La sección describe el
  modelo de producción por lo que se puede leer de él (46 variables, 240 árboles,
  gain dominado por `fwi_anom_sigma`), pero **no existe en ninguno de los tres
  repositorios un script que escriba `xgb_v2_prototipo.ubj`**. Se ha verificado
  que su lista de 46 variables es exactamente `features.py:FULL` (50) menos las
  cuatro autorregresivas intra-celda; lo que no consta es con qué partición y con
  qué etiqueta se entrenó. El texto evita afirmarlo. Si el tribunal pregunta por
  el entrenamiento del grupo de control, hace falta esa respuesta.
- **[RESUELTO el 01/09/2026: `dos_20`…`dos_24` ya están en el repositorio]** —
  La figura F6b y la cifra del 4,6 % de gain concurrente salen de
  `dos_23_vispera.json` y `dos_24_auditoria_fugas.json`. Los cinco scripts se
  copiaron con sus md5 a `06_comparacion` (`dos_20`, `dos_21`, `dos_22`,
  `dos_23`) y a `05_iteracion2/57_ablaciones` (`dos_24`); ver la actualización
  del 01/09/2026 en `docs/PROCEDENCIA.md`. Como los diez scripts de la
  iteración 2 que ya estaban, necesitan el cubo y el disco de expansión: entran
  por trazabilidad, no como demostración ejecutable.
- **[PENDIENTE: reejecutar `dos_18_ratio` sobre 2026]** — ver la nota de la tabla
  de evaluación. Hoy la escalera del ratio en 2026 solo existe como líneas de log.
- **[PENDIENTE: racional del tope de 30 positivos y del radio de 50 km]** — los
  dos parámetros están en el código (`dos_12_muestrear_effis.py:47` y el nombre
  `frp_max_50km_7d`) pero no tienen justificación escrita en ningún sitio. El
  texto los describe sin justificarlos.
- **[PENDIENTE: hueco de los cinco días de mapa perdidos, 27-31/08/2026]** — no
  se menciona en la sección porque pertenece al capítulo de producción y jueces;
  tampoco está todavía en `docs/LIMITACIONES.md`.
- **[CITA NECESARIA: IberFire]** — el cubo se describe sin cita. La referencia
  que da `docs/CONTEXTO_SECCION_MODELO.md` es Ercibengoa et al. (2025),
  arXiv:2505.00837, CC-BY 4.0.
- **[CITA NECESARIA: FWI canadiense]** — Van Wagner (1987), según la misma fuente.
- **[CITA NECESARIA: XGBoost]** — Chen y Guestrin (2016).
- **[CITA NECESARIA: pseudo-ausencias en modelos de susceptibilidad]** — sostiene
  la frase «el diseño habitual en la literatura de susceptibilidad»;
  Barbet-Massin et al. (2012).
- **[CITA NECESARIA: fuga de información y reproducibilidad]** — sostiene el
  párrafo del control de fuga; Kapoor y Narayanan (2023).
- **[PENDIENTE: norma de citas]** — no consta en ningún repositorio qué estilo
  exige la universidad ni si hay normativa sobre uso de IA. Ninguna de las cinco
  referencias anteriores está escrita en la sección: se citan aquí porque las dio
  el autor en `docs/CONTEXTO_SECCION_MODELO.md`, y hay que unificarlas en el
  fichero de bibliografía del documento completo.

## Figuras y cómo regenerarlas

Todas a 300 dpi y 16 cm de ancho, sin título dentro de la figura, paleta
Okabe-Ito. Desde `docs/MEMORIA/figs/scripts/`:

```
~/miniconda3/envs/tfm_fuego/bin/python f1_iteraciones.py
~/miniconda3/envs/tfm_fuego/bin/python f2_disenios.py
~/miniconda3/envs/tfm_fuego/bin/python f3_separacion.py
~/miniconda3/envs/tfm_fuego/bin/python f4_metrica_ciega.py
~/miniconda3/envs/tfm_fuego/bin/python f5_temporada2026.py
~/miniconda3/envs/tfm_fuego/bin/python f6_donde_mira.py
```

| Figura | Fuente de datos | Nota |
|---|---|---|
| F1 `f1_iteraciones.png` | ninguna: estructural, redibuja `00_marco/README.md:3-32` | |
| F2 `f2_disenios.png` | ninguna: esquema de `dos_12_muestrear_effis.py:97-134` | |
| F3 `f3_separacion.png` | `02_eda/figuras/eda_resumen.log` | dentro del repo definitivo |
| F4 `f4_metrica_ciega.png` | `M/salida/dos_05_metricas.json` | |
| F5 `f5_temporada2026.png` | `M/salida/dos_09_temporada2026.csv` + `dos_19_veredicto.json` | La banda se recalcula con la misma receta que `dos_19` (bootstrap de días, N=2000, semilla 42); la anotación del último día se lee del JSON sellado para que texto y figura no se separen por el sorteo del bootstrap (recálculo local: +0,1117 [+0,0746, +0,1538] frente a +0,1117 [+0,0745, +0,1509] guardado) |
| F6 `f6_donde_mira.png` | `muestras/modelos/*.ubj` + `M/salida/dos_23_vispera.json`; escribe `figs/f6_gain.json` | el script que produce ese JSON es `06_comparacion/dos_23_vispera.py`, en el repositorio desde el 01/09/2026 |

Los scripts leen `TFM_fuego_malla` a través de la variable de entorno
`TFM_MALLA` (por defecto, la carpeta hermana del repositorio). Si esos artefactos
entran algún día en `tfm-riesgo-incendios/salida/`, basta con apuntar `TFM_MALLA`
al propio repositorio.
