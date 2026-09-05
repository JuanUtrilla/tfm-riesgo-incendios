# Procedencia de cada fichero

Este repositorio **no reescribe código**: cada script es una copia literal del
repositorio donde se ejecutó y produjo los números de la memoria. Lo único que
se ha adaptado son las rutas, y solo en los cuatro módulos compartidos de
`01_datos/comun/` (ver [`ESTRUCTURA.md`](ESTRUCTURA.md)).

La razón es de evidencia, no de pereza: si se reescribiera `dos_18_ratio.py`,
el 87,6 de percentil ponderado que aparece en la memoria ya no lo habría
producido el código que se enseña. Se conserva el original y se documenta de
dónde viene.

Origen: **T** = `TFM_fuego` · **M** = `TFM_fuego_malla` · **P** =
`aemet-horario-verano2026`. El hash es el `md5` (8 primeros dígitos) del
fichero en su repo de origen el 25/08/2026, para poder comprobar que la copia
es literal.

## Homónimos que divergen

Cinco ficheros existen con el mismo nombre en `TFM_fuego` y en
`TFM_fuego_malla` con contenido distinto: son el **prototipo** de la malla y su
**versión definitiva**. Se conservan los dos, porque los dos produjeron
resultados citados: el prototipo sostiene el capítulo de la bisagra («no es la
meteorología») y la versión definitiva es la que sirve hoy. El prototipo vive
en `04_bisagra/42_no_es_la_meteo/prototipo_TFM_fuego/`.

`fwi_canadiense.py` aparecía en los tres repos: las copias de `TFM_fuego` y del
repo de producción son idénticas entre sí, y la de `TFM_fuego_malla` solo
cambia en que su bloque de demostración lee la ruta del cubo de `config` en vez
de tenerla escrita. Se conserva **una sola**, la de `TFM_fuego_malla`, en
`01_datos/comun/`.

## Qué se ha dejado fuera

Veintiún scripts del bloque de visión por satélite (Sentinel-2, D-Fire,
timelapses y los casos de Sotalvo, Luna y Ponteareas). Es trabajo real, pero no
aparece en el hilo conductor de la memoria y arrastra 55 MB de imágenes. Sigue
en `TFM_fuego`.

## El inventario


### `01_datos/aemet` — 7 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `aemet_descarga_historico.py` | TFM_fuego | `abbbb282` |
| `aemet_descarga_key2.py` | TFM_fuego | `e098508e` |
| `aemet_descarga_key3.py` | TFM_fuego | `ca9b803c` |
| `aemet_horario_collector.py` | aemet_horario_verano2026 | `910dee39` |
| `descargar_datos_2025_2026.py` | TFM_fuego | `07ac5278` |
| `descargar_historico_aemet.py` | TFM_fuego | `3c43221f` |
| `observacion_horaria.py` | TFM_fuego | `2dfdc914` |

### `01_datos/comun` — 1 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `exportar_limites.py` | aemet_horario_verano2026 | `7d7379a6` |

### `01_datos/cubo` — 3 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `exportar_malla_gh.py` | TFM_fuego | `ed0d96fa` |
| `extraer_features_cubo.py` | TFM_fuego | `a654779c` |
| `extraer_features_cubo_v4.py` | TFM_fuego | `04b0c96f` |

### `01_datos/effis` — 1 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `descargar_effis.py` | aemet_horario_verano2026 | `04384cdd` |

### `01_datos/era5land` — 6 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `descargar_cape_era5.py` | TFM_fuego | `0ade9ad7` |
| `descargar_era5_2026.py` | TFM_fuego | `0f0eb172` |
| `malla_01_nodos.py` | TFM_fuego_malla | `692e0041` |
| `malla_02_descarga.py` | TFM_fuego_malla | `a7a43500` |
| `malla_04_climatologia.py` | TFM_fuego_malla | `f2d70834` |
| `malla_06_descarga_historico.py` | TFM_fuego_malla | `68fc867e` |

### `01_datos/extra` — 2 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `descargar_features_extra.py` | TFM_fuego | `e3eba05f` |
| `ganaderia_descarga.py` | TFM_fuego | `77eda75d` |

### `01_datos/firms` — 4 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `extraer_historia_firms_v3.py` | TFM_fuego | `2224aa05` |
| `firms_api.py` | aemet_horario_verano2026 | `646c27ef` |
| `firms_descarga.py` | TFM_fuego | `3864ef77` |
| `focos_activos.py` | aemet_horario_verano2026 | `27a4c944` |

### `02_eda` — 1 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `eda_dataset.py` | TFM_fuego | `64f1d307` |

### `03_iteracion1` — 1 fichero (copia del 01/09/2026)

| Fichero | Origen | md5 |
|---|---|---|
| `MODELO_B_BITACORA.md` | TFM_fuego | `0f668a23` |

La bitácora de la iteración 1 (779 líneas, 14/07→11/08): es la única
procedencia escrita de `xgb_v2_prototipo`, el modelo servido en producción,
cuyo reentrenamiento (§16) fue interactivo y no dejó script.

### `03_iteracion1/31_muestreo` — 4 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `ensamblar_dataset.py` | TFM_fuego | `557fae54` |
| `ensamblar_dataset_v4.py` | TFM_fuego | `555315bb` |
| `muestrear_dataset.py` | TFM_fuego | `376ca55f` |
| `muestrear_dataset_v4.py` | TFM_fuego | `a18233a8` |

### `03_iteracion1/32_features` — 2 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `extraer_features_historia.py` | TFM_fuego | `e321ddac` |
| `extraer_features_historia_v4.py` | TFM_fuego | `7f95377a` |

### `03_iteracion1/33_train` — 6 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `entrenar_modelo.py` | TFM_fuego | `447554b0` |
| `verificar_v2.py` | escrito para este repositorio (01/09/2026) | — |
| `entrenar_modelo_v3.py` | TFM_fuego | `4ded7410` |
| `entrenar_modelo_v4.py` | TFM_fuego | `bdc1c692` |
| `modelo_nativo_estacion.py` | TFM_fuego | `a477be3b` |
| `tuning_optuna.py` | TFM_fuego | `994199c8` |

### `03_iteracion1/34_produccion` — 11 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `mapa_diario.py` | aemet_horario_verano2026 | `7d3f1734` |
| `mapa_riesgo_dia.py` | TFM_fuego | `1c19f696` |
| `mapa_riesgo_hoy.py` | TFM_fuego | `f47e58ff` |
| `predecir_punto.py` | TFM_fuego | `caff6798` |
| `preparar_prototipo.py` | TFM_fuego | `62a171d0` |
| `publicar_mapas_gh.py` | TFM_fuego | `f877e252` |
| `ranking_diario.py` | aemet_horario_verano2026 | `d941462e` |
| `ranking_diario_cron.py` | TFM_fuego | `bbb99401` |
| `retro_julio.py` | aemet_horario_verano2026 | `e21c5315` |
| `tiempo_real.py` | TFM_fuego | `3cfc5f88` |
| `validar_modelo.py` | aemet_horario_verano2026 | `fb77c1fc` |

### `03_iteracion1/35_ablaciones` — 2 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `ablacion_features.py` | TFM_fuego | `19fb1fe2` |
| `ablacion_v3_firms.py` | TFM_fuego | `a904933f` |

### `04_bisagra/41_validacion_operativa` — 7 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `dashboard_miteco_datos.py` | TFM_fuego | `ba8bfabd` |
| `descargar_verdad_operativa.py` | TFM_fuego | `4a42ab67` |
| `extraer_evento.py` | TFM_fuego | `77c715ef` |
| `validar_eventos_estaciones.py` | TFM_fuego | `e37d005c` |
| `validar_eventos_firms.py` | TFM_fuego | `5304bee9` |
| `validar_miteco.py` | TFM_fuego | `ae29060c` |
| `validar_operativo.py` | TFM_fuego | `8934c2ba` |

### `04_bisagra/42_no_es_la_meteo` — 23 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `construir_clim_fwi_aemet.py` | TFM_fuego | `a7e39211` |
| `diagnostico_fwi.py` | TFM_fuego_malla | `06c0ceb1` |
| `experimento_b_era5.py` | TFM_fuego | `c60a7ed1` |
| `experimento_b_fase2.py` | TFM_fuego | `4eac4933` |
| `malla_02_verificar.py` | TFM_fuego | `544fc9ab` |
| `malla_02_vs_cubo.py` | TFM_fuego | `99a8e42d` |
| `malla_02b_correccion.py` | TFM_fuego_malla | `d6a58bf0` |
| `malla_02b_hibrido.py` | TFM_fuego_malla | `ddb601fe` |
| `malla_02b_ifs.py` | TFM_fuego_malla | `c1607b2f` |
| `malla_02b_prueba.py` | TFM_fuego_malla | `80063163` |
| `malla_02b_reajuste.py` | TFM_fuego_malla | `0422b1cd` |
| `malla_03_cuantiles.py` | TFM_fuego | `6cc4a23e` |
| `malla_03b_fwi_qm.py` | TFM_fuego | `5b5ded7a` |
| `malla_03c_aceptacion.py` | TFM_fuego | `dba35e77` |
| `malla_05_riesgo.py` | TFM_fuego_malla | `0406d9d5` |
| `malla_05b_evaluacion.py` | TFM_fuego_malla | `94a8fb26` |
| `prueba_era5_atribucion.py` | TFM_fuego | `1e5b37be` |
| `prueba_era5_produccion.py` | TFM_fuego | `93d82327` |
| `prueba_hibrido_corregido.py` | TFM_fuego | `b8897932` |
| `prueba_hibrido_produccion.py` | TFM_fuego | `d0026180` |
| `prueba_ifs_produccion.py` | TFM_fuego | `e100b6a5` |
| `verificar_fuentes_meteo.py` | TFM_fuego | `9e5341bd` |
| `verificar_hora_fwi.py` | TFM_fuego_malla | `03ed0c9d` |

### `04_bisagra/42_no_es_la_meteo/prototipo_TFM_fuego` — 5 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `malla_01_nodos.py` | TFM_fuego | `1130f871` |
| `malla_02_descarga.py` | TFM_fuego | `b4cd16ea` |
| `malla_04_climatologia.py` | TFM_fuego | `89ee1705` |
| `malla_05_riesgo.py` | TFM_fuego | `a6d1ad2a` |
| `malla_05b_evaluacion.py` | TFM_fuego | `afb6013d` |

### `04_bisagra/43_no_es_el_train_serve` — 3 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `ablacion_proxies_operativos.py` | TFM_fuego | `3514a799` |
| `auditoria_train_serve.py` | TFM_fuego | `2eda1751` |
| `dos_06_deriva.py` | TFM_fuego_malla | `cf471ddc` |

### `04_bisagra/44_es_la_especificacion` — 2 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `dos_00_cubo_etiquetas.py` | TFM_fuego_malla | `9f4fcdff` |
| `dos_02_muestrear.py` | TFM_fuego_malla | `810dadc5` |

### `05_iteracion2/51_celdas` — 1 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `dos_01_celdas.py` | TFM_fuego_malla | `70dc3f66` |

### `05_iteracion2/52_muestreo` — 1 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `dos_12_muestrear_effis.py` | TFM_fuego_malla | `4efefb1f` |

### `05_iteracion2/53_features` — 1 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `dos_03_features.py` | TFM_fuego_malla | `7b9c9de0` |

### `05_iteracion2/54_analisis` — 1 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `dos_04_analisis.py` | TFM_fuego_malla | `1e9f8d2d` |

### `05_iteracion2/55_train` — 3 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `dos_05_modelos.py` | TFM_fuego_malla | `7f7391b7` |
| `dos_10_donde_effis.py` | TFM_fuego_malla | `881e4389` |
| `dos_13_modelos_effis.py` | TFM_fuego_malla | `7d536c87` |

### `05_iteracion2/56_calibracion` — 1 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `dos_14_cortes.py` | TFM_fuego_malla | `015bf04c` |

### `05_iteracion2/57_ablaciones` — 4 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `dos_08_verano.py` | TFM_fuego_malla | `49ee55c8` |
| `dos_17_capa_quemado.py` | TFM_fuego_malla | `24f36389` |
| `dos_18_ratio.py` | TFM_fuego_malla | `efa65daa` |
| `dos_24_auditoria_fugas.py` | TFM_fuego_malla | `a7b165fa` |

### `06_comparacion` — 11 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `comparar_julio2026.py` | TFM_fuego_malla | `c26881e0` |
| `comparar_produccion.py` | TFM_fuego_malla | `937820f1` |
| `comparar_rankings.py` | TFM_fuego_malla | `fefc67f1` |
| `comparar_rankings_justo.py` | TFM_fuego_malla | `106b1e6d` |
| `dos_09_temporada2026.py` | TFM_fuego_malla | `845e98ec` |
| `dos_11_miteco.py` | TFM_fuego_malla | `ea381a5d` |
| `dos_19_veredicto.py` | TFM_fuego_malla | `4de2f9cb` |
| `dos_20_aciertos.py` | TFM_fuego_malla | `b44b6946` |
| `dos_21_hectareas.py` | TFM_fuego_malla | `caef566d` |
| `dos_22_ventana.py` | TFM_fuego_malla | `31b5d7c0` |
| `dos_23_vispera.py` | TFM_fuego_malla | `1dd0e6be` |

### `07_produccion` — 13 ficheros

| Fichero | Origen | md5 |
|---|---|---|
| `capa_base.py` | TFM_fuego_malla | `ce1591a2` |
| `capa_verdad.py` | TFM_fuego_malla | `3c0b9cb5` |
| `dos_07_mapa_hoy.py` | TFM_fuego_malla | `879a1af7` |
| `dos_15_veredicto_miteco.py` | TFM_fuego_malla | `cde26ca7` |
| `dos_16_juez_estaciones.py` | TFM_fuego_malla | `67d97248` |
| `dos_riesgo_hoy.py` | TFM_fuego_malla | `bfa6f6a9` |
| `gh_estado.py` | TFM_fuego_malla | `7dc13951` |
| `gh_exportar_estado.py` | TFM_fuego_malla | `0afbd6a5` |
| `gh_mensual_rapido.py` | TFM_fuego_malla | `c05c5f07` |
| `gh_reanalisis.py` | TFM_fuego_malla | `278c7ac2` |
| `puntuar_effis.py` | TFM_fuego_malla | `2d17322d` |
| `redibujar.py` | TFM_fuego_malla | `29bddbe7` |
| `riesgo_hoy.py` | TFM_fuego_malla | `5cc7b58c` |

## Actualización del 01/09/2026 — entran `dos_20`…`dos_24`

Faltaban los cinco. La memoria ya citaba dos de sus salidas —el 4,6 % de
importancia concurrente de `dos_24_auditoria_fugas.json` y la curva de desfase
de `dos_23_vispera.json`— sin que el código que las produce viajara con el
repositorio, que es exactamente lo que este documento existe para impedir.

  · `dos_20_aciertos.py`, `dos_21_hectareas.py`, `dos_22_ventana.py`,
    `dos_23_vispera.py` → `06_comparacion`, junto a `dos_09`, `dos_11` y
    `dos_19`, que es donde vive la evaluación de la temporada 2026.
  · `dos_24_auditoria_fugas.py` → `05_iteracion2/57_ablaciones`, con `dos_17`
    y `dos_18`.

No son ejecutables con `muestras/`: necesitan el cubo IberFire y el disco de
expansión, igual que los diez scripts de `05_iteracion2` que ya estaban aquí.
Se incluyen por trazabilidad número→script, no como demostración.

Dos dependencias que ahora quedan cerradas dentro del repositorio:
`dos_23_vispera.py` importa `CLASES` y `verdad_por_dia` de `dos_21_hectareas.py`,
y `dos_24_auditoria_fugas.py` importa `FULL` de `dos_05_modelos.py`, que ya
estaba en `05_iteracion2/55_train`.

Siguen siendo copias literales: los md5 de las cinco copias coinciden con los
de los originales en `TFM_fuego_malla`.

## Actualización del 31/08/2026 — corrección de una fuga

Siete ficheros de `TFM_fuego_malla` cambiaron el 31/08/2026 y sus copias se han
rehecho, con los hashes de esa fecha en vez de los del 25/08:

  · `comparar_rankings.py` — **corrige una fuga de futuro**. `firms_dia` pedía a
    la API de FIRMS `/5/{D-1}` creyendo que la ventana iba hacia atrás; FIRMS
    cuenta los días HACIA ADELANTE, así que devolvía `[D-1, D+3]` y el mapa del
    día D llevaba dentro los focos del propio incendio. La versión copiada el
    25/08 (`3b866e80`) TENÍA ESE BUG: los números que produjo están inflados a
    favor de producción (AUC 0,737 en vez de 0,647) y no deben citarse.
    Producción en vivo nunca estuvo afectada, y el entrenamiento tampoco.
  · `puntuar_effis.py`, `dos_15_veredicto_miteco.py`, `dos_16_juez_estaciones.py`,
    `dos_riesgo_hoy.py` — añaden `donde_dia_effis_r10` como cuarto candidato.
  · `gh_estado.py` — guarda contra pisar el estado diario del Release.
  · `gh_reanalisis.py` — comprueba que el reanálisis cubre hasta D−6 sin huecos.

El principio del repositorio no cambia: siguen siendo copias literales, y el
hash permite comprobarlo. Lo que cambia es a qué fecha del original apuntan.

---

## Incorporación del 05/09/2026 — calibración y ablación de datos

Ocho ficheros nuevos, con un **origen distinto a los tres anteriores**:

Origen **C** = `~/Desktop/Master/calibracion_si/`, un banco de trabajo aislado
fuera de todo repositorio. Se montó así a petición explícita: copia de los `.py`
de `TFM_fuego_malla` (commit `9f92cda`) en un `sandbox/` con su propio `salida/`,
de modo que **ni un solo fichero de los repos se modificó** mientras se
producían estos resultados (`git status` de `TFM_fuego_malla` y de este repo
quedó vacío toda la jornada). Su bitácora, con los md5 antes y después de cada
arreglo, está en `calibracion_si/BITACORA.md`.

### `05_iteracion2/56_calibracion` — 7 ficheros nuevos

| Fichero | Origen | md5 |
|---|---|---|
| `dos_25_barrido_historico.py` | C | `2721d70d` |
| `dos_26_calibra_si.py` | C | `46bdecff` |
| `dos_27_escala_absoluta.py` | C | `c0cd7a9b` |
| `dos_28_calibra_celda.py` | C | `76d1b90f` |
| `dos_29_calibra_estacional.py` | C | `2a8c2340` |
| `dos_31_ventana_movil.py` | C | `63f6e93c` |
| `dos_32_producto_dos_capas.py` | C | `65eee377` |

### `05_iteracion2/57_ablaciones` — 2 ficheros nuevos

| Fichero | Origen | md5 |
|---|---|---|
| `dos_30_r10_todo.py` | C | `5eef1cb0` |
| `dos_34_auditoria_geometria.py` | C | `0b6c998a` |

### `06_comparacion` — 2 ficheros nuevos

| Fichero | Origen | md5 |
|---|---|---|
| `replay_dia_todo.py` | C | `10c8a16e` |
| `replay_temporada_todo.py` | C | `457ae2f1` |

**Salvedad sobre estos dos últimos.** Son copias parcheadas de
`archivo_ifs/replay_dia.py` y `replay_temporada.py`, y a diferencia del resto
del repositorio **llevan rutas absolutas escritas** (`AQUI`, `SANDBOX`,
`MODELO_NUEVO`). No se han adaptado a variables de entorno **a propósito**: son
los que produjeron los números de `R10_TODOS_LOS_ANIOS.md`, y tocarlos rompería
el vínculo entre el código y el resultado, que es justamente el principio de
este repositorio. Para ejecutarlos hay que editar esas tres constantes. Los
scripts del replay original (`archivo_ifs/replay_*.py`) siguen **pendientes** de
incorporar; ver `PENDIENTE.md`.

El principio no cambia: copias literales, y el hash permite comprobarlo.

---

## Incorporación del 05/09/2026 (2) — el replay de 2025 y 2026

Catorce ficheros más, con **otro origen**:

Origen **A** = `~/Desktop/Master/archivo_ifs/`, el directorio de trabajo del
replay, fuera de todo repositorio (los datos que maneja —IFS archivado de dos
temporadas, ERA5-Land, FIRMS, EFFIS— pesan decenas de GB y nunca han estado en
git). Cierra el pendiente «copiar los 9 scripts del replay al repo».

### `06_comparacion` — 11 ficheros nuevos

| Fichero | Origen | md5 |
|---|---|---|
| `replay_dia.py` | A | `2a060518` |
| `replay_temporada.py` | A | `29d666ba` |
| `replay_verdad.py` | A | `79bbf76b` |
| `replay_veredicto.py` | A | `60bd15e2` |
| `replay_figuras.py` | A | `ab2e227d` |
| `replay_si.py` | A | `18e0d3c0` |
| `replay_precision.py` | A | `71288dea` |
| `replay_radio.py` | A | `15a6c6a6` |
| `replay_cobertura.py` | A | `9cf54e1c` |
| `verificar_replay.py` | A | `3aa798d4` |
| `dos_33_anomalia_punta.py` | C | `6fe5559f` |

### `01_datos` — 4 ficheros nuevos

| Fichero | Destino | Origen | md5 |
|---|---|---|---|
| `bajar_ifs_2025.py` | `01_datos/ifs` | A | `7ba0d828` |
| `rellenar_ifs.py` | `01_datos/ifs` | A | `2f16da70` |
| `bajar_effis_2025.py` | `01_datos/effis` | A | `bc046172` |
| `bajar_era5land_2025.py` | `01_datos/era5land` | A | `11cdd482` |

**Salvedad de ejecución, importante.** Estos scripts **no corren desde el árbol
del repositorio**. `replay_dia.py` calcula `AQUI` desde su propia ubicación y
espera encontrar al lado un `sandbox_replay/` (copia de los `.py` de
`TFM_fuego_malla`) y los ficheros de entrada (`ifs_archivo_<año>.parquet`,
`era5land_diario_<año>.nc`, los parquets de FIRMS, los GeoJSON de EFFIS). Aquí
están **como evidencia de qué código produjo los números**, que es el propósito
de este repositorio, no como una copia ejecutable. Para reproducirlos hay que
situarlos en un directorio con esas entradas, como describe
`README_replay.md` en `archivo_ifs/`.

No se han adaptado las rutas por la misma razón que en el resto del repositorio:
tocarlas rompería el vínculo entre el código y el resultado publicado.
