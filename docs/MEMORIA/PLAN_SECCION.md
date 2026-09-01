# Plan de la sección de modelado — documento de trabajo

> Fase de lectura y planificación. **No contiene prosa de la memoria.**
> Escrito el 31/08/2026 recorriendo `tfm-riesgo-incendios` (repo definitivo),
> `TFM_fuego_malla` (iteración 2 y evaluación), `TFM_fuego` (iteración 1 y
> bisagra) y `aemet_horario_verano2026` (cadena sellada de producción).
>
> Convención de rutas: sin prefijo = `tfm-riesgo-incendios/`; `M/` =
> `TFM_fuego_malla/`; `T/` = `TFM_fuego/`.
>
> Todo lo que aparece aquí se ha verificado abriendo el fichero o el artefacto
> citado. Lo que no se ha podido verificar está en §3 o en §6, nunca dado por
> bueno.

---

## 1. Inventario de evidencia

### 1.1 Pipeline de datos e ingesta

| Afirmación que podría sustentarse | Evidencia | Tipo |
|---|---|---|
| El *backbone* es el cubo IberFire y de él salen 45 de las 46 variables | `01_datos/cubo/extraer_features_cubo.py`; `02_eda/ANALISIS_CUBO.md` §0 | código + resultado |
| El cubo **es** ERA5-Land reprocesado (corr 0,987-0,998, sesgo 0), salvo precipitación (/2,02) | `04_bisagra/42_no_es_la_meteo/prototipo_TFM_fuego/malla_02_vs_cubo.py` → `T/dataset/malla_02_vs_cubo.json` | código + resultado guardado |
| El FWI del cubo son las 13 UTC, estable en tres meses | `04_bisagra/42_no_es_la_meteo/verificar_hora_fwi.py` → `M/salida/verificar_hora_fwi.json` | código + resultado |
| EGIF y EFFIS se solapan poco a nivel celda-día (2,6 % / 14,6 % / 34 % / 59 % por tramo de tamaño) | `02_eda/ANALISIS_CUBO.md` §1, producido por `dos_03` | resultado guardado |
| Inventario de etiquetas: 19.561 celda-día EGIF 2015-20; 21.785 celdas-día EFFIS; 1.895 polígonos / 307.850 ha en 2026 | `02_eda/ANALISIS_CUBO.md` §1 | resultado |
| FIRMS se usa como entorno y nunca como etiqueta | `01_datos/firms/firms_api.py`, `01_datos/firms/extraer_historia_firms_v3.py` | código |
| Rutas y resolución de datos por variables de entorno con caída a `muestras/` | `01_datos/comun/config.py:88-101` (`entrada()`), `entorno.sh:19-24` | configuración |

### 1.2 Ingeniería de variables

| Afirmación | Evidencia | Tipo |
|---|---|---|
| El modelo servido usa **46** variables, en orden fijado por un JSON | `muestras/modelos/xgb_v2_prototipo_features.json` (46 nombres, verificado) | artefacto |
| La lista canónica de features y su troceado por bloques | `01_datos/comun/features.py:28-43` | código |
| El orden de columnas se comprueba contra el original antes de predecir (XGBoost no valida nombres) | `01_datos/comun/features.py:52-76` (`verifica()`) | código |
| **Regla antifuga de la iteración 1**: todas las ventanas son estrictamente anteriores a D | `03_iteracion1/32_features/extraer_features_historia.py:112-116` — `(vz_d >= d-90) & (vz_d < d)`, `(vf_d >= d-7) & (vf_d < d)` | código |
| **Regla antifuga de la iteración 2**: idéntica | `05_iteracion2/53_features/dos_03_features.py:147-159` — `atras_90`, `atras_365`, `mismomes` con `vz_d < a0`, FIRMS `(vf_d >= d-7) & (vf_d < d)` | código |
| Normalización local del FWI (percentil contra climatología de la celda) es la contribución propia de la iteración 1 | `01_datos/comun/fwi_canadiense.py`; `04_bisagra/42_no_es_la_meteo/construir_clim_fwi_aemet.py` | código |
| Las cuatro satelitales se sustituyen al servir por climatología mensual 2020-24 | `07_produccion/riesgo_hoy.py:187-203` (`mensual()`) | código |

### 1.3 Definición del modelo y protocolo de entrenamiento

| Afirmación | Evidencia | Tipo |
|---|---|---|
| **Iteración 1**: XGBoost, 50 features, `SEED=42`, `n_estimators=3000`, `lr=0,05`, `max_depth=6`, early stopping 100, split train ≤2018 / val 2019 / test 2020 | `03_iteracion1/33_train/entrenar_modelo.py:44,61-65,105-110` | código |
| **Iteración 1 (v4)**: train 2015-2020 / val 2021 / test 2022, selección `tuned` por AP en val | `T/dataset/metricas_v4.json` (`"protocolo"`, `"seleccion"`) | resultado guardado |
| **Iteración 2**: los mismos hiperparámetros exactos, `SEED=42`, `eval_metric="aucpr"` | `05_iteracion2/55_train/dos_05_modelos.py:71-75` | código |
| Función única de entrenamiento con early stopping en validación | `05_iteracion2/55_train/dos_05_modelos.py:93-99` (`entrena`) | código |
| **Muestreo de la iteración 2**: `SEED=7`, `RATIO=4` (se recorta a 3), tope de **30** positivos por (día, bloque 100 km), split **train 2015-2022 / val 2023 / test 2024** | `05_iteracion2/52_muestreo/dos_12_muestrear_effis.py:45-51` | código |
| Positivo = primer día de `is_fire` (`f[1:] & ~f[:-1]`), no cualquier día ardiendo | `dos_12_muestrear_effis.py:79` | código |
| Los dos diseños de negativos conviven en la misma tabla maestra: `cuando` (misma celda, otro día del split) y `donde` (mismo día, otra celda) | `dos_12_muestrear_effis.py:97-118` | código |
| Banco de evaluación `eval_dia`: 1.000 celdas al azar por día + hasta 60 celdas EFFIS, veranos 2023 y 2024 | `dos_12_muestrear_effis.py:50-51,120-134` | código |
| El modelo único `donde_dia_effis` y la pareja `donde_effis_c × cuando_effis` se entrenan en el mismo script | `05_iteracion2/55_train/dos_13_modelos_effis.py:75-102` | código |
| El `donde_effis_c` usa etiqueta «≥1 `is_fire` 2008-2022» y valida en 2023, para no contaminar el test 2024 | `dos_13_modelos_effis.py:87-96` y su docstring `:13-16` | código |

### 1.4 Métricas registradas

| Número | Artefacto | Tipo |
|---|---|---|
| **Iteración 1, v1**: AUC-ROC 0,9311 · AUC-PR 0,8426 · Brier 0,0906 · TSS 0,6975 en test 2020; prevalencia 0,2628; n_train 53.563 / val 14.489 / test 8.614; CV espacial 0,8476 ± 0,0279 | `T/dataset/metricas_v1.json` | resultado guardado |
| **Escalera de baselines** A→D en el mismo JSON (FWI 0,7285 → XGB completo 0,9311) | `T/dataset/metricas_v1.json` | resultado |
| **Iteración 1, v4**: AUC-ROC **0,8906** IC95 [0,8644, 0,9136] en test 2022; FWI percentil 0,7528; v4−v3 = −0,0012 | `T/dataset/metricas_v4.json` | resultado |
| Importancia por *gain* de v4: `fwi_anom_sigma` 9,51 % · `n_fuegos_10km_mismomes_hist` 8,62 % · `fwi_pctl_local` 6,31 % | `T/dataset/metricas_v4.json` (`importancia_gain_pct`) | resultado |
| SHAP top-15 de v1, encabezado por `n_fuegos_10km_mismomes_hist` (0,612) | `T/dataset/metricas_v1.json` (`shap_top15`) | resultado |
| Ablación de la normalización local: M_abs 0,6684 vs M_ambos 0,7103 de AUC-PR | `T/dataset/ablaciones_v1.json` | resultado |
| Ablación de proxis operativos: satélite −0,0041 [−0,0064, −0,0006], los dos juntos −0,0045 | `T/dataset/ablacion_proxies_operativos.json` | resultado |
| **Iteración 2, test 2024/val 2023** por modelo (AUC dentro del día, lift, Δ vs prod, IC95) | `M/salida/dos_13_metricas.json` | resultado |
| **Ablación del ratio**: escalera 3/10/30/60/100, val AUC-PR y AUC por día en 2023 y 2024 | `M/salida/dos_18_ratio.json` | resultado (**no incluye 2026** — ver §3) |
| **Veredicto de temporada 2026** (74 días, 4.843 celdas quemadas, 260.753 ha, 01/06→15/08): prod 0,647 · único 0,752 (Δ+0,105 [+0,065, +0,145], gana 70 %) · único 1:10 **0,7587** (Δ+0,1117 [+0,0745, +0,1509], gana 75,7 %) · pareja 0,705 | `M/salida/dos_19_veredicto.json` (31/08 17:46) y `M/salida/dos_09_temporada2026.json` | resultado guardado |
| La serie diaria completa que alimenta el veredicto | `M/salida/dos_09_temporada2026.csv` (74 filas) | resultado |
| Cara a cara justo previsión-contra-previsión, 19 días | `M/salida/rankings_justo.json` (+ `rankings_justo_CON_FUGA.json` conservado como contraste) | resultado |
| Jueces en operación | `M/salida/puntuacion_effis.csv`, `veredicto_miteco.json`, `veredicto_estaciones.json`, `historico_veredictos.csv` | resultado sellado |
| Medianas de EDA en train (`fwi_pctl_local` 84,33 pos vs 47,00 neg; `popdens` y `dist_carreteras` con **la misma mediana**) | `02_eda/figuras/eda_resumen.log` | log |

### 1.5 Semillas y reproducibilidad

| Afirmación | Evidencia | Tipo |
|---|---|---|
| Semilla del muestreo: `SEED = 7`; del entrenamiento y del bootstrap: `SEED = 42` | `dos_12_muestrear_effis.py:45`; `dos_05_modelos.py:71`; `dos_13_modelos_effis.py:43` | código |
| Bootstrap **de días**, N=2000, para todos los IC de la iteración 2 | `dos_05_modelos.py:91` (`N_BOOT`), `dos_13_modelos_effis.py:36,43-54` | código |
| Bootstrap agrupado por `bloque_100km`, N=2000, en la iteración 1 | `T/dataset/metricas_v4.json` (`bootstrap`) | resultado |
| El código no se reescribió: 120 ficheros con su md5 de origen | `docs/PROCEDENCIA.md` — **verificado: 119 de 120 hashes coinciden** con el fichero en su ruta actual | configuración |
| La ventana de FIRMS es parametrizable y por defecto 7 días, con caché separada por ventana | `06_comparacion/comparar_rankings.py:74,103-121` | código |
| El error de fuga y su corrección están documentados en el propio código | `06_comparacion/comparar_rankings.py:78-102` (docstring de `firms_dia`) | código |
| **No hay `environment.yml` ni `requirements.txt`** en el repo definitivo | comprobado: no existe ninguno de los dos; `docs/PENDIENTE.md` §1 lo reconoce | ausencia verificada |

---

## 2. Narrativa real del proyecto

Reconstruida con `git log` de los tres repositorios, los ficheros deprecados y
las bitácoras. **La cronología no es la del hilo lógico de la memoria**, y esa
diferencia es en sí misma material del capítulo.

### 2.1 La secuencia efectiva

`TFM_fuego` solo tiene **5 commits** (`c6b2777` … `2deba87`): el trabajo de la
iteración 1 está registrado en prosa, no en historia de git
(`T/MODELO_B_BITACORA.md`, 779 líneas, entradas del 14/07 al 11/08).
`TFM_fuego_malla` tiene **82 commits** del 19/08 al 31/08 y es donde se lee la
secuencia real. `tfm-riesgo-incendios` tiene **11 commits**, todos de
construcción del repositorio (25/08 y 31/08): no aporta narrativa de
investigación.

1. **14-15/07** — dataset, primer modelo y prototipo en tiempo real
   (`T/MODELO_B_BITACORA.md:7,194`). Muestreo caso-control con celda fija.
2. **28/07** — primera validación operativa con 2-3 días sellados; conclusión
   publicada: *«el modelo bate al FWI»* (`docs/BITACORA.md`, punto de partida).
3. **11/08** — se añade MITECO como cuarta verdad-terreno y aparece el
   resultado incómodo: el modelo (0,715 con FIRMS filtrado) **empata con la
   persistencia** `n_detec_50km_7d` (0,723), que además es una de sus propias
   features (`T/MODELO_B_BITACORA.md:532,659`).
4. **11-18/08** — tres semanas agotando la vía del ajuste: v3
   (`entrenar_modelo_v3.py`), v4 con EGIF consolidado
   (`entrenar_modelo_v4.py`), modelo nativo de estación
   (`modelo_nativo_estacion.py`), auditoría train/serve
   (`auditoria_train_serve.py`). **Ninguno cierra el hueco.**
5. **18/08** — la validación rehecha con 20 días retira la conclusión del
   28/07: el baseline FWI pasa de 0,540 a 0,702 mientras el modelo apenas se
   mueve. *«El modelo bate al FWI» era un artefacto de muestra pequeña.*
6. **19-20/08** — pipeline de malla ERA5-Land, para servir con la misma fuente
   con la que se entrenó (`8329d4d` … `f6366b5`).
7. **21/08** — el giro: la medición del cubo y los tres diseños de muestreo
   (`5f8fd3a`, `4374858`, `855f8c5`). El diagnóstico pasa de meteorología a
   **especificación**.
8. **21-23/08** — ablaciones (`c30c939` verano, `92223ad` capa quemado,
   `745fac2` ratio) y montaje de los tres jueces en GitHub Actions.
9. **25/08** — se construye el repositorio definitivo (`fc4a7fb`).
10. **31/08** — se detecta y corrige la **fuga de futuro de FIRMS**
    (`604f568`, `0d456e0`) y se relanza toda la evaluación retrospectiva
    (`3ef7ab9`, y en el definitivo `c6ca73a`, `6af10fa`).

### 2.2 Callejones sin salida y descartes — todos medidos

Esto es lo que impide que la sección quede «sospechosamente limpia». Cada uno
tiene número y fichero:

| Descarte | Por qué se probó | Resultado medido | Evidencia |
|---|---|---|---|
| **v3** (reajuste con 2015-2020 completo) | ¿faltaban datos? | mejora marginal | `T/dataset/metricas_v3.json` |
| **v4** (EGIF consolidado, test 2022) | ¿era la etiqueta incompleta? | v4−v3 = **−0,0012** [−0,0032, +0,0007] | `T/dataset/metricas_v4.json` |
| **Modelo nativo de estación** | entrenar donde se sirve | no resuelve el hueco | `33_train/modelo_nativo_estacion.py`, `T/dataset/nativo_estacion_metricas.json` |
| **Proxis operativos congelados** | sospechoso obvio del train/serve | **−0,0045** los dos juntos | `T/dataset/ablacion_proxies_operativos.json` |
| **Features satelitales** | ¿aportan algo siendo reales? | eliminarlas cuesta lo mismo que congelarlas (−0,004): **no aportan** | ídem |
| **Servir con ERA5-Land en malla** | el sesgo húmedo de +8,00 mm | pipeline construido y servido, pero **no era la causa** | `M/malla_0*.py`, `T/dataset/experimento_b_resultados.json` |
| **Experimento B, tercera pata** | comparar AUC entre fuentes | **el control falló** (0,589 reconstruido vs 0,702 publicado); no se reporta | `docs/LIMITACIONES.md` §2, `T/dataset/experimento_b_fase2_resultados.json` |
| **Entrenar solo con verano** | el producto es de verano | empeora el DÓNDE | `M/salida/dos_08_verano.json` |
| **Capa «ya quemado»** | la cicatriz existe (LAI −40 %) | **−0,02**: la etiqueta EFFIS penaliza bajar el riesgo | `M/salida/dos_17_capa_quemado.json` |
| **Ratio 1:3 heredado** | nadie lo había justificado | óptimo interior 1:10-1:30; 1:100 empeora | `M/salida/dos_18_ratio.json` |
| **La pareja DÓNDE × CUÁNDO** | era la arquitectura *recomendada* en el encargo original | la bate el **modelo único**: 0,705 vs 0,752 en 2026 | `M/salida/dos_19_veredicto.json` |
| **Bloque de visión (Sentinel-2, D-Fire)** | descartado del alcance | 21 scripts fuera del repo definitivo | `docs/PROCEDENCIA.md` «Qué se ha dejado fuera» |

### 2.3 Dos conclusiones que hubo que retirar

Son el material más valioso de la sección y las dos están documentadas:

- **«El modelo bate al FWI»** (28/07) → retirada el 18/08 al pasar de 3 a 20
  días. Causa: estimación inestable del baseline, no del modelo.
- **«La ventaja del rediseño es marginal (+0,037)»** y **«producción acierta
  los megaincendios (0,883)»** → retiradas el 31/08 al corregir la fuga de
  FIRMS. Valores limpios: Δ**+0,105** y **0,692**. El arreglo *refuerza* la
  tesis, y se conserva el fichero contaminado (`rankings_justo_CON_FUGA.json`)
  como contraste.

La detección de la fuga es en sí misma un método reutilizable, ya automatizado:
*si el acierto no decae al alejar el predictor del suceso, algo está viendo el
suceso* (`M/dos_24_auditoria_fugas.py`, `M/salida/dos_24_auditoria_fugas.json`).

---

## 3. Huecos y riesgos

### 3.1 Métricas que NO están guardadas y habría que reejecutar

1. **La ablación del ratio sobre 2026 no existe como artefacto.**
   `M/salida/dos_18_ratio.json` (23/08 17:49, **anterior** a la corrección de
   la fuga) contiene solo 2023 y 2024, y no contiene ninguna métrica ponderada
   por hectáreas. Las cifras limpias de 2026 por ratio existen **solo como
   líneas de log** (`M/salida/dos_09_ventana7.log:482-486`). Si la memoria
   reporta la escalera del ratio en 2026, hay que reejecutar `dos_18` o citar
   `dos_09`, no `dos_18`.
2. **El percentil ponderado por hectáreas «81,7 → 87,6-87,9» está obsoleto.**
   Procede de `M/RESULTADOS_DOS_MODELOS.md:358-393` (23/08, con fuga). Las
   cifras limpias son **79,4 (único 1:3) → 81,8-81,9 (1:10)**
   (`M/salida/dos_09_ventana7.log`). `docs/BITACORA.md` §6 y
   `docs/LIMITACIONES.md` §4 **todavía citan las viejas**;
   `docs/TRAZABILIDAD.md` ya cita las nuevas. Hay que unificar antes de
   escribir.
3. **Curvas PR, calibración y SHAP de la iteración 2 no existen.** La
   iteración 1 tiene `pr_curves_test.png`, `calibracion_test.png` y
   `shap_summary.png` (generadas por `entrenar_modelo.py:192-201`); la
   iteración 2 solo guarda AUC/lift por día y el *gain*. No hay curva PR ni
   calibración del `donde_dia_effis`.
4. **No hay medida de coste computacional** (tiempo de entrenamiento, memoria)
   de ningún modelo, en ningún artefacto.
5. **Sin `environment.yml` ni `requirements.txt`**, ninguna cifra es
   reejecutable tal cual. Es el hueco que `docs/PENDIENTE.md` §1 marca como el
   más grave, y sigue abierto.
6. **Cuatro días de mapa diario (28-31/08) perdidos** y decididos como no
   reconstruibles: esos días no podrán puntuarse cuando lleguen sus perímetros
   EFFIS. No está escrito en `LIMITACIONES.md` todavía.

### 3.2 Decisiones cuya justificación no consta en el repositorio

Solo el autor puede aportarlas (ver §6):

- Por qué `xgb_v2_prototipo` —el modelo **realmente servido en producción**—
  tiene 46 features y no las 50 de `entrenar_modelo.py`. **No existe ningún
  script que entrene o escriba `xgb_v2_prototipo.ubj`**: se ha buscado
  `save_model` en los tres repositorios y solo aparecen `xgb_v1`,
  `xgb_v1_tuned`, `xgb_v3`, `xgb_v4` y los `dos_*`. El JSON de 46 nombres
  coincide con las 50 menos las cuatro autorregresivas intra-celda
  (`n_fuegos_1km_90d`, `n_fuegos_10km_90d`, `n_fuegos_10km_365d`,
  `n_fuegos_1km_hist`), pero eso es inferencia mía, no evidencia.
- Los hiperparámetros de XGBoost se heredan idénticos de la iteración 1 a la
  iteración 2 (`dos_05_modelos.py:72-75`) sin retuneo ni justificación escrita.
- El tope de **30** positivos por (día, bloque de 100 km)
  (`dos_12_muestrear_effis.py:47`) y el radio de **50 km** de las features de
  FIRMS no tienen racional escrito.
- Por qué el split de la iteración 2 llega a 2022 en train (incluyendo el año
  récord) mientras el de la iteración 1 acababa en 2018/2020.
- Ninguna cita bibliográfica está en el repositorio: no hay fichero de
  bibliografía en `tfm-riesgo-incendios`.

### 3.3 Discrepancias entre la documentación y lo que el código hace

Todas verificadas ejecutando la comprobación:

| # | Documentación dice | El código hace | Evidencia |
|---|---|---|---|
| 1 | «Ninguna ruta personal vive ya en el código» (`docs/ESTRUCTURA.md:58`, `entorno.sh:16`) | **69 de los 127 ficheros `.py`** contienen `/home/charredgem` codificado | `grep -rl "/home/charredgem" --include=*.py` |
| 2 | «46 features» en README y borradores | `01_datos/comun/features.py:43` define `FULL` con **50** nombres; el modelo servido usa las 46 del JSON. Quien cuente en `features.py` obtiene 50 | `features.py:28-43` vs `xgb_v2_prototipo_features.json` |
| 3 | El veredicto se cita con dos juegos de cifras distintos | Son **dos corridas con ventana de FIRMS distinta**: ventana 5 → prod 0,644 / único 0,751 / 1:10 0,758 (`dos_09_ventana5.json`); ventana 7 → prod 0,647 / único 0,752 / 1:10 0,7587 (`dos_09_temporada2026.json`, que es la que lee `dos_19`). **La canónica es la de 7 días** | ambos JSON y sus logs |
| 4 | `docs/CONTEXTO_SECCION_MODELO.md` (escrito ayer) | usa el juego de **ventana 5**; hay que rectificarlo al de ventana 7 | ídem |
| 5 | `docs/LIMITACIONES.md` §7: «El entrenamiento cubre 2015-2020» | La iteración 2 entrena **2015-2022** (`dos_12_muestrear_effis.py:49`). La frase solo vale para la iteración 1 | código |
| 6 | El desajuste de ventana FIRMS «ya está arreglado» | Está arreglado en **evaluación** (`comparar_rankings.py:74`, por defecto 7) pero **la cadena de servicio sigue pidiendo 5 días** (`07_produccion/riesgo_hoy.py:222-234`) | código |
| 7 | `docs/PROCEDENCIA.md` ubica `malla_02b_ifs.py` en `04_bisagra/42_no_es_la_meteo/` | El fichero está en `01_datos/ifs/`. Es el **único** de los 120 que no cuadra: los otros 119 md5 verifican | comprobación de los 120 hashes |
| 8 | `dos_09_temporada2026.py:33` documenta «FIRMS [D−5, D−1]» | tras el arreglo la ventana por defecto es 7 días; el docstring quedó desactualizado | código |
| 9 | `docs/TRAZABILIDAD.md` atribuye a `dos_18_ratio.py` el percentil ponderado por hectáreas | ese script **no calcula ni guarda** ninguna métrica ponderada por hectáreas | `dos_18_ratio.py:245` y el JSON |
| 10 | La bitácora cita `dos_24_auditoria_fugas.py` | **cinco scripts posteriores al 25/08 no están en el repo definitivo**: `dos_20_aciertos`, `dos_21_hectareas`, `dos_22_ventana`, `dos_23_vispera`, `dos_24_auditoria_fugas`, más `historico.py`. Los números del top-2 % por hectáreas y la auditoría de fugas no tienen script en el repositorio que acompaña a la memoria | comparación de árboles |

**Ninguno de estos huecos se rellena con valores inventados.** Los puntos 3, 9
y 10 son bloqueantes para escribir la sección de evaluación: hay que fijar
antes qué juego de cifras es el oficial y qué scripts entran en el repo.

---

## 4. Plan de figuras

Seis piezas. Ninguna decorativa; ninguna captura de código.

### F1 · Diagrama de las dos iteraciones y la bisagra
- **Pregunta que responde**: ¿por qué hay dos pipelines y qué los separa?
  Es la figura que evita que el lector crea que el primer modelo es el bueno.
- **Existe**: como diagrama ASCII en `00_marco/README.md:3-32`. **Hay que
  generarla** como vectorial (TikZ o SVG); el contenido ya está fijado.
- **Datos**: ninguno, es estructural. Se dibuja a mano sobre el ASCII existente.
- **Pie preliminar**: «Las dos iteraciones y la bisagra. Lo que cambia entre
  ellas no son los datos ni las variables (46 en las dos), sino qué se toma
  como positivo, qué como negativo y qué como verdad.»

### F2 · Los tres diseños de muestreo, sobre la misma celda y el mismo día
- **Pregunta**: ¿en qué se diferencia físicamente «caso-control con celda fija»
  de «negativos del mismo día»? Es el concepto central y ahora mismo solo está
  en prosa.
- **Existe**: **no**. Hay que generarla.
- **Datos y script**: esquema con tres paneles (celda×día) construido a partir
  de la lógica de `dos_12_muestrear_effis.py:97-118`; opcionalmente con
  recuentos reales de `maestra_effis.parquet` (necesita `TFM_USB`).
- **Pie**: «Los tres diseños de muestreo. El caso-control temporal (izquierda)
  fija la celda y varía el día; el diseño *dónde* (centro) fija el día y varía
  la celda; el banco de evaluación (derecha) enfrenta las celdas quemadas del
  día contra 1.000 celdas al azar de ese mismo día.»

### F3 · Caracterización de los datos: la separación de clases en train
- **Pregunta**: ¿qué separa un positivo de un negativo, y qué no?
  Incluye el aviso temprano: `popdens` y `dist_carreteras` tienen **la misma
  mediana** en positivos y negativos, que es el primer síntoma del problema de
  especificación.
- **Existe**: `02_eda/figuras/eda_distribuciones.png` y
  `eda_normalizacion_local.png`; los números en `eda_resumen.log`.
- **Datos y script**: `02_eda/eda_dataset.py` sobre el split de train (53.563
  filas). Requiere `TFM_USB`.
- **Pie**: «Distribución de las variables discriminantes en el conjunto de
  entrenamiento (53.563 filas, 13.577 positivos). El FWI en percentil local
  separa (84,3 frente a 47,0 de mediana) mientras las estáticas de exposición
  humana no separan nada: en el diseño caso-control con celda fija, positivo y
  negativo comparten celda.»

### F4 · El diagnóstico: la métrica de desarrollo es ciega
- **Pregunta**: ¿por qué un AUC de 0,89 no anunciaba el fallo? Es la figura del
  hallazgo central.
- **Existe**: **no** como figura; los números sí
  (`M/salida/dos_05_metricas.json`: AUC caso-control 0,92 = 0,92; AUC dentro
  del día 0,744 vs 0,828).
- **Datos y script**: barras pareadas de dos diseños × dos métricas, a partir
  de `dos_05_metricas.json`; script nuevo de una veintena de líneas, sin
  recómputo.
- **Pie**: «La misma pareja de modelos bajo dos métricas. En AUC caso-control
  son indistinguibles (0,92 los dos); en AUC dentro del día, que es la métrica
  de la pregunta operativa, se separan 0,084. La métrica con la que se
  desarrolló la iteración 1 no podía ver la diferencia.»

### F5 · El resultado central: la temporada 2026, día a día
- **Pregunta**: ¿la ventaja del rediseño se sostiene, o la sostienen cuatro
  días? Panel superior AUC diario con media acumulada; panel inferior
  diferencia pareada acumulada con IC95 de bootstrap de días.
- **Existe**: **sí**, `M/salida/dos_19_veredicto.png` (31/08 17:46, cifras
  limpias). Hay que copiarla al repo definitivo, que hoy no tiene ni una
  figura de resultados (`docs/PENDIENTE.md` §3).
- **Datos y script**: `06_comparacion/dos_19_veredicto.py` sobre
  `dos_09_temporada2026.csv` (74 días, 4.843 celdas quemadas, 260.753 ha).
- **Pie**: «Temporada 2026 (1-jun a 15-ago, 74 días). Arriba, AUC dentro del
  día; abajo, la diferencia pareada acumulada contra producción con su IC95 por
  bootstrap de días. El número que se defiende es el del último día, fijado de
  antemano: Δ +0,112 [+0,074, +0,151].»
- **Aviso a incluir en el texto**: la propia figura advierte que mirar la banda
  cada día es *peeking* (`dos_19_veredicto.py:28-33`).

### F6 · Análisis de error / importancia: qué mira cada modelo y dónde falla
- **Pregunta**: ¿de dónde saca el modelo su ventaja, y qué le queda por
  resolver? Dos paneles: (a) importancia por *gain* comparada
  —`n_fuegos_10km_mismomes_hist` domina en el único, mientras en v4 dominaba
  `fwi_anom_sigma`—; (b) la comprobación de fuga: acierto del top-2 % en
  función del desfase D−0 / D−1 / D−2, que es lo que destapó el error.
- **Existe**: panel (a) **no** como figura, los números sí
  (`T/dataset/metricas_v4.json` y el *gain* del único); panel (b) tiene datos
  en `M/salida/dos_24_auditoria_fugas.json` y `dos_20_aciertos.json`, pero
  **su script no está en el repo definitivo** (§3.3 #10).
- **Pie**: «Izquierda: importancia por *gain*. Derecha: fracción de incendios
  de ≥500 ha en el 2 % de celdas de mayor riesgo, según el desfase entre el
  mapa y el día del fuego. Un predictor honesto pierde acierto al alejarse; el
  que no lo pierde está viendo el suceso.»

**Tabla T1** (no es figura, pero ocupa presupuesto): comparación de las dos
iteraciones en una tabla — etiqueta, muestreo, split, n, métrica de desarrollo,
métrica operativa. Fuente: `metricas_v4.json` + `dos_19_veredicto.json`.

---

## 5. Esqueleto con presupuesto

Presupuesto total: **2.000-2.400 palabras** de texto corrido + 5-7 figuras.
Referencia: el borrador actual `BORRADOR_memoria_ML.tex` tiene ~2.750 palabras
a 10 pt, así que este plan implica **recortar entre un 13 % y un 27 %**,
sacando a tabla lo que hoy es prosa descriptiva de fuentes.

| § | Subsección | Palabras | Figuras |
|---|---|---|---|
| 1 | **Encuadre y pregunta operativa** — de la susceptibilidad al ordenamiento de 498.530 celdas dentro del día; anuncio del giro en la primera frase | **320** | F1 |
| 2 | **Datos y preparación** — 2.1 fuentes y etiquetas en una tabla (EGIF vs EFFIS, solape 2,6-59 %); 2.2 unidad muestral y regla antifuga; 2.3 las 46 variables por bloques | **450** | F3, T1 |
| 3 | **Metodología hasta obtener el modelo** — 3.1 iteración 1: caso-control, XGBoost, split, AUC 0,89; 3.2 la bisagra: 98,8 % / 0,92=0,92 / 0,744 vs 0,828; 3.3 rediseño: etiqueta EFFIS, negativos del mismo día, único vs pareja | **650** | F2, F4 |
| 4 | **Evaluación** — 4.1 protocolo (AUC dentro del día, bootstrap de días, semillas); 4.2 test 2024 y temporada 2026; 4.3 ablaciones: ratio, verano, FIRMS, capa quemado | **550** | F5 |
| 5 | **Limitaciones y acciones posteriores** — la fuga de FIRMS y el método que la destapó; el bug sellado del viento (16,3 %); el ruido del sorteo (0,005); train/serve de la ventana; los tres jueces y el cierre de septiembre | **380** | F6 |
| | **Total** | **2.350** | 6 figuras + 1 tabla |

Reparto contra el objetivo del enunciado: encuadre 320 (obj. 300-350) · datos
450 (400-500) · metodología 650 (600-700) · evaluación 550 (500-600) ·
limitaciones 380 (300-400). **Total 2.350 palabras**, dentro de la horquilla
2.000-2.400.

Reglas de recorte si se pasa: primero se comprimen las fichas de fuente de §2
(van al capítulo de datos, que es anterior), después la descripción de la
iteración 1 en §3.1 (el detalle está en su propio capítulo). **§3.2 no se
recorta**: es la aportación.

---

## 6. Preguntas para el autor

Ordenadas por impacto en la calidad del texto.

1. **¿Qué juego de cifras es el oficial de la memoria: ventana FIRMS de 5 o de
   7 días?** Existen las dos corridas limpias y difieren (prod 0,644 vs 0,647;
   único 1:10 0,758 vs 0,7587; gana 73 % vs 76 %). `dos_19_veredicto.json`, el
   README y `TRAZABILIDAD.md` usan la de 7; `CONTEXTO_SECCION_MODELO.md` usa la
   de 5. Sin esta decisión no se puede escribir §4.
2. **¿Cómo se entrenó `xgb_v2_prototipo`, el modelo que sirve producción?** No
   hay ningún script en los tres repositorios que lo escriba, y su lista de
   features (46) no coincide con la de `entrenar_modelo.py` (50). ¿Es v1
   reentrenado sin las cuatro autorregresivas intra-celda? La sección tiene que
   describir el modelo del grupo de control y ahora mismo no puedo hacerlo sin
   inventar.
3. **¿Entran los cinco scripts posteriores al 25/08 (`dos_20`…`dos_24`) en el
   repositorio definitivo?** De ellos salen la auditoría de fugas y las cifras
   por hectáreas del top-2 %. Si no entran, esas cifras no deberían citarse en
   la memoria.
4. **¿Se reejecuta `dos_18_ratio` sobre 2026 limpio, o la escalera del ratio se
   cita desde `dos_09`?** El JSON actual es anterior a la corrección de la fuga
   y no cubre 2026.
5. **¿Cuál es la justificación del tope de 30 positivos por (día, bloque de
   100 km) y del radio de 50 km de FIRMS?** Un tribunal preguntará por los dos
   y no hay racional escrito.
6. **¿Se retunean los hiperparámetros para la iteración 2, o se declara
   explícitamente que se heredan de la iteración 1 para aislar el efecto del
   diseño?** Lo segundo es defendible y más fuerte, pero hay que decirlo.
7. **¿Qué norma de citas exige la universidad y hay normativa sobre uso de
   IA?** Sigue sin constar en ningún repositorio, y condiciona el formato de la
   sección entera.
8. **¿Se documenta el hueco de los cuatro días de mapa perdidos (28-31/08) en
   la memoria, o solo en el repositorio?** Afecta a cuántos días podrá puntuar
   el juez EFFIS en septiembre.

---

## Recuento

Palabras planificadas para la sección: **2.350** (objetivo 2.000-2.400).
Este documento de trabajo no cuenta contra ese presupuesto.
