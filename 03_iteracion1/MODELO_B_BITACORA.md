# Bitácora del Modelo B — traza de transformaciones de datos
> Registro cronológico de CADA transformación, decisión y comprobación sobre los datos,
> con el racional de cada una. Material fuente para la sección de metodología de la memoria.
> Convención: cada entrada dice QUÉ se hizo, POR QUÉ, CON QUÉ script y QUÉ salió.
> Documento central: [`README.md`](README.md) · Hoja de ruta: [`PROXIMOS_PASOS.md`](PROXIMOS_PASOS.md) §⭐⭐

## 14/07/2026 — Sesión nocturna: del muestreo al primer modelo

### 1. Decisión de pivote: IberFire como backbone de features
- **Contexto**: deadline adelantado al 1-ago. La extracción masiva
  ERA5-Land+FWI vía CDS/EWDS para ~21k puntos no cabe en plazo
  (cola CDS ~20-30 min/fichero).
- **Inspección previa** (`scratchpad/inspeccion_iberfire*.py`): el datacubo local
  `iberfire/IberFire.nc` (30.3 GB, 1 km × 1 día, 2007-12→2024-12, EPSG:3035, 261 variables)
  valida como fuente: extracción de serie completa por celda en 0.1-0.2 s (chunking interno
  [521,77,99]), FWI sin NaN, `is_fire=1` en Sotalvo el 14-ago-2021 con FWI 63.7 y HR mín 14.3
  (coherente con la validación evento-primero), y el desplome del FWI post-Ponteareas coincide
  con los diluvios reales del 16-17-oct-2017.
- **Huecos asumidos del cubo**: sin componentes FWI (FFMC/DMC/DC/ISI/BUI), sin VPD (se deriva
  de t2m+HR), sin rayos/CAPE (se cubren con WGLC/ERA5 propios).

### 2. Hallazgo crítico: el EGIF está incompleto desde 2021
- Conteo por año del `egif_civio.csv` (2015→2023): 4124 / 2432 / 5223 / 2004 / 3796 / 2287 /
  **888 / 226 / 23**. Los tres últimos años son artefacto del lag de consolidación del registro
  oficial (2022 fue el peor año real de la década según GWIS/MODIS).
- **Consecuencia**: usar 2021-2023 contaminaría el dataset por partida doble: faltan positivos
  y, peor, las pseudo-ausencias sorteadas en esos años podrían ser días de incendio real sin
  registrar. **Decisión: el dataset del modelo usa solo 2015-2020.**

### 3. Muestreo (tabla maestra) — `muestrear_dataset.py` → `dataset/muestra_maestra_v1.parquet`
- **Positivos**: 19,561 incendios EGIF 2015-2020 con coordenadas, proyectados a EPSG:3035 y
  asignados a su celda 1 km de IberFire (vecino más cercano al centro de celda); deduplicados
  por (celda, fecha) conservando el de mayor superficie. Se descartan los que caen fuera de la
  malla peninsular (`is_spain=0`: Canarias, mar).
- **Pseudo-ausencias 1:3** (ratio elegido dentro del rango 1:1-1:10 respaldado por la literatura,
  Barbet-Massin 2012): misma celda que un positivo, día aleatorio uniforme dentro de los años del
  MISMO split. Racional: al fijar la celda se empareja la susceptibilidad espacial y el modelo
  aprende a discriminar el CUÁNDO; la fecha uniforme deja la estacionalidad como señal aprendible
  (los papers la sitúan entre los drivers principales).
- **Buffer de exclusión**: un día no puede ser negativo si hay un incendio EGIF a <12.5 km en
  ±10 días (réplica del criterio `is_near_fire` de IberFire: caja 25×25 celdas × 10 días).
  Implementado con cKDTree sobre las coordenadas 3035 de los fuegos.
- **Split temporal congelado ANTES de extraer features** (anti-leakage, Kapoor & Narayanan 2023):
  train 2015-2018 / val 2019 / test 2020. Columna `bloque_100km` (rejilla de 100 km) para el CV
  espacial posterior (GroupKFold).
- **Resultado**: 78,065 filas (train 40,649/13,577, val 11,090/3,720, test 6,765/2,264;
  ratio neg:pos 2.99 en los tres splits), 76 bloques espaciales. Determinista (SEED=42).
- **Spot-check contra el cubo** (300+300 muestras): 3.0% de negativos con `is_near_fire=1`
  (fuegos del lado portugués o perímetros EFFIS grandes que el EGIF no ve → se filtrarán en el
  ensamblado); 1.3% de positivos con `is_fire=1` (esperado: la máscara de IberFire es EFFIS
  ≥5 ha y la mayoría de incendios EGIF son <1 ha → correcto etiquetar con EGIF, no con `is_fire`).
- **Verificación colateral**: `firms_iberia_2015_2024.parquet` íntegro (256,468 filas,
  2015→ene-2025, 0 nulos en coordenadas) — deuda de SCRIPTS.md §B cerrada.

### 4. Extracción de features del cubo — `extraer_features_cubo.py` → `dataset/features_cubo_v1.parquet`
- **Acceso por bloques de chunk** (77×99 celdas × serie temporal completa, ~2 GB RAM transitorios,
  93 bloques × ~4 s): evita lecturas aleatorias celda a celda. Total: ~7 min para 78,065 filas × 40 cols.
- **Features del día D** (información operacionalmente disponible vía forecast): fwi, t2m_max/min,
  rh_min, viento_max, precip_dia, lst, ndvi, lai, swi010, es_festivo.
- **Ventanas hacia atrás EXCLUYENDO el día D** (anti-leakage §7.4): precip 7/15/30d (suma),
  fwi_med 7/15/30d + fwi_max_7d, rh_min/t2m_max/viento_max medias 7d, ndvi_med_30d,
  dias_sin_lluvia (consecutivos con precip<1 mm desde D hacia atrás, tope 120).
- **Normalización local (contribución del TFM, §7.1-N5)**: fwi_pctl_local y fwi_anom_sigma
  calculados contra la climatología 2008-2014 del MISMO MES en la MISMA celda del propio cubo —
  años anteriores al dataset (2015-2020) → sin fuga temporal, y sin necesidad de descargar
  climatologías externas.
- **Estáticas**: elevación, pendiente, rugosidad, dist. a carreteras/ríos, popdens del AÑO de la
  fila (el cubo trae 2008-2020), proporciones CLC del corte más cercano sin mirar al futuro
  (CLC_2012 para ≤2017, CLC_2018 para ≥2018).

### 5. Historial de fuego, FIRMS y rayos — `extraer_features_historia.py` → `features_historia_v1.parquet`
- **Autorregresivas del EGIF 2008-2020** (driver nº1 en ocurrencia agregada, arXiv:2508.09896),
  estrictamente < fecha: n_fuegos a <1.5 km (90d e histórico), a <10 km (90d, 365d) y mismo-mes
  de años anteriores. Una consulta KDTree por celda única (14,806), no por fila.
- **frp_max_50km_7d / n_detec_50km_7d** (FIRMS VIIRS): ventana [D-7, D-1] — NUNCA el día D
  (circularidad con la propia ignición). Sin detección = 0 real, no missing. Caveat: ventana
  truncada en ene-2015 (<1% filas).
- **Rayos WGLC** (0.5°, vecino más cercano): densidad día D + suma 7d previos.

### 6. Ensamblado + QC — `ensamblar_dataset.py` → `dataset/dataset_modelo_v1.parquet` (76,666 filas × 51 features)
- Join 1:1 validado de los 3 bloques; derivadas: vpd_max (Magnus, kPa) y dia_anio (crudo, sin
  sin/cos — árboles, §7.1-N7).
- **Filtrado**: 1,399 pseudo-ausencias (1.79%) con is_near_fire=1 (fuego EFFIS <12.5 km / 10 días
  previos que el buffer EGIF no vio: lado portugués o perímetros grandes) → eliminadas.
- **QC**: 0 duplicados, 0 violaciones de rangos físicos. NaN: ≤2.2% (meteo ERA5 en celdas
  costeras) — se dejan como NaN nativo (§7.1-N3), sin imputar/escalar/winsorizar (§7.5).
- **Prevalencia FIJADA y reportada** (§7.3-D5): test 26.28% (ratio 2.805), train 25.35%, val 25.67%.

### 7. EDA (solo train 2015-2018) — `eda_dataset.py` → `eda/eda_*.png`
- **Separación clara pos/neg** en las dinámicas: FWI mediano 12.4 vs 1.8; percentil local 84 vs 47;
  HR mín 40% vs 59%; VPD 1.42 vs 0.73 kPa; días sin lluvia 9 vs 2; precip_30d 17 vs 36 mm.
- **El gráfico de la normalización** (eda_normalizacion_local.png): las CCAA atlánticas arden con
  FWI mediano 3-14 y las mediterráneas con 38-41, pero el PERCENTIL local converge a 73-92 en
  todas → la justificación empírica de fwi_pctl_local con nuestros propios datos.
- **Hallazgo de diseño (importante para leer el SHAP)**: popdens/dist_carreteras tienen medianas
  IDÉNTICAS en pos y neg — consecuencia deliberada del muestreo con celda fija: las estáticas no
  pueden discriminar marginalmente; el modelo discrimina el CUÁNDO (la susceptibilidad espacial
  la aporta el muestreo, no el modelo). Fichar para la memoria: el modelo es de riesgo temporal
  condicionado a la localización.
- **Multicolinealidad esperada** (|rho|>0.8): fwi~vpd~rh_min~t2m_max; pctl~anom (0.96); ndvi~lai
  (0.90) — XGBoost la tolera; para SHAP, interpretar por grupos.

### 8. Entrenamiento — `entrenar_modelo.py` → `modelos/xgb_v1.ubj`, `dataset/metricas_v1.json`
Protocolo: split temporal congelado (early stopping y umbral TSS elegidos en val 2019; test 2020
intocado hasta el final), prevalencia real sin `scale_pos_weight` (salida como probabilidad,
§7.3-D3), calibración isotónica ajustada en val.

**Escalera de baselines (test 2020, prevalencia 26.28%):**

| Peldaño | AUC-PR | AUC-ROC | Brier | TSS |
|---|---|---|---|---|
| A. FWI absoluto (sin ML) | 0.488 | 0.729 | — | 0.279 |
| A2. FWI pctl local (sin ML) | 0.481 | 0.746 | — | 0.349 |
| B. XGB solo-FWI | 0.491 | 0.736 | 0.174 | 0.289 |
| C. XGB meteo+FWI | 0.716 | 0.882 | 0.119 | 0.619 |
| **D. XGB completo (50 feats)** | **0.843** | **0.931** | **0.091** | **0.698** |
| D calibrado (isotónico) | 0.833 | 0.931 | 0.092 | — |

- Cada peldaño gana al anterior → el requisito de defensa (XGBoost >> FWI puro) se cumple con
  margen (+0.35 AUC-PR).
- La calibración isotónica NO mejora el Brier (0.0906→0.0920): al entrenar con prevalencia real
  sin rebalanceo, el modelo ya sale bien calibrado (consistente con van den Goorbergh 2022) —
  se reporta la curva de fiabilidad y se mantiene la salida cruda.
- **CV espacial** (GroupKFold 5 por bloque_100km, 2015-2019): AUC-PR 0.848 ± 0.028 — sin colapso
  al dejar zonas fuera (el temido gap espacial de la literatura queda acotado en nuestro diseño,
  en parte porque el muestreo de celda fija ya condiciona la localización).
- **SHAP (test)**: dominan las autorregresivas (n_fuegos_10km_mismomes_hist 1º, n_fuegos_10km_90d
  3º, n_fuegos_1km_hist 4º) — replica el hallazgo de Portugal (arXiv:2508.09896) — y
  **fwi_anom_sigma es 2º, por encima del FWI crudo (7º)**: la normalización local aporta.

### 9. Ablaciones y controles — `ablacion_features.py` → `dataset/ablaciones_v1.json`

| Experimento | AUC-PR test | Lectura |
|---|---|---|
| M_abs (meteo+FWI absoluto, 19f) | 0.668 | |
| M_pctl (meteo+FWI local, 16f) | 0.650 | El percentil NO sustituye al absoluto… |
| M_ambos (21f) | 0.710 | …pero COMPLEMENTA: +0.042 sobre M_abs |
| solo_fwi_abs (5f) | 0.570 | |
| solo_fwi_local (2f) | 0.473 | |
| **solo_calendario (4f)** | **0.476** | La estacionalidad sola ya rinde como el FWI solo (consistente con Fire 9(6):217) — parte del rendimiento del modelo es "cuándo/dónde", honestidad para la memoria |
| **calendario+fwi_local (6f)** | **0.749** | 6 features baten al meteo completo de 21 — el percentil local condensa la interacción zona×peligro |
| D completo (50f) | 0.843 | referencia |
| D sin autorregresivas 90d (48f) | 0.839 | **El sesgo del buffer en n_fuegos_*_90d NO es load-bearing** (−0.004) |

- **Control anti-fuga**: etiquetas barajadas → AUC-PR 0.231 ≈ prevalencia (0.263). Pipeline limpio.
- Caveat honesto para la memoria (§7.3-D5): estos AUC-PR son a prevalencia 26% (diseño 1:3);
  NO son comparables con papers a prevalencia real (~1e-3). El AUC-ROC 0.93 sí es comparable
  y está por encima del rango típico 0.82-0.84 — en parte porque nuestra tarea (discriminar
  el CUÁNDO con celda emparejada) es más fácil que la ocurrencia libre.

### 10. Ficha del dataset
Definición exacta de las 66 columnas, fuentes y garantías anti-fuga: [`dataset/DATASET_CARD.md`](dataset/DATASET_CARD.md).

### 11. Tuning bayesiano acotado — `tuning_optuna.py` → `dataset/tuning_optuna_v1.json`, `modelos/xgb_v1_tuned.ubj`
- 40 trials de Optuna (TPE) optimizando AUC-PR en val 2019 (test intocado durante la búsqueda).
- Mejores params: lr 0.017, max_depth 10, min_child_weight 3, subsample 0.89, colsample 0.50,
  lambda 0.72, gamma 0.01. Val: 0.8666 vs 0.8647 del base.
- **Test 2020: AUC-PR 0.849 / ROC 0.933 / Brier 0.089** (base: 0.843/0.931/0.091).
- Lectura: +0.007 — confirma que el ROI estaba en las features, no en el tuning (como anticipaba
  la revisión de literatura). Se conservan ambos modelos.

### 12. Demo out-of-time: mapa de riesgo nacional — `mapa_riesgo_dia.py` → `eda/mapa_riesgo_<fecha>.png`
- Reconstruye las 50 features del modelo para las ~500k celdas peninsulares de un día dado
  (mismas definiciones que el pipeline; autorregresivas por convolución sobre la malla en vez de
  KDTree; caveat: EGIF llega a 2020 → en fechas posteriores esos conteos van a la baja) y pinta
  la probabilidad con las detecciones FIRMS del día como verdad-terreno (el EGIF de 2022 está
  incompleto, FIRMS es independiente).
- **17-jul-2022 (pico de la ola, año NUNCA visto por el modelo)**: los clusters FIRMS de
  Galicia/León/frontera portuguesa caen sobre las zonas más oscuras. Riesgo mediano en celdas
  con detección: 0.803 vs 0.731 nacional (percentil 72). Lectura crítica: en un día donde media
  España está en riesgo extremo real, la discriminación intra-día es moderada — el valor del
  modelo está en discriminar días/zonas, no dentro del percentil 99 de una ola histórica.
  Parte de las detecciones azules caen en Portugal/mar (fuera de la máscara de predicción).
- **Contraste 15-ene-2022**: riesgo mediano nacional 0.117 (vs 0.731 el 17-jul) — el modelo
  separa regímenes estacionales con claridad. La discriminación intra-día de enero es débil
  (percentil 53), esperable: las detecciones FIRMS de invierno son mayormente quemas agrícolas
  y fuentes industriales de calor.

### 13. Consulta puntual con explicación — `predecir_punto.py`
- Reconstruye las 50 features para UNA (lat, lon, fecha) desde cubo+EGIF+FIRMS+WGLC y predice
  con `xgb_v1_tuned.ubj` + SHAP local. Germen del prototipo de defensa.
- **Validación out-of-time con Sotalvo** (14-ago-2021, año fuera del entrenamiento):
  probabilidad **0.869** (FWI 63.7 = percentil local 100, HR 14%, drivers SHAP: percentil/anomalía
  local del FWI y las detecciones FIRMS del entorno la semana previa). El mismo punto el
  14-ene-2021: **0.036**. Contraste 24×.

### Pendientes que quedan abiertos (15/07/2026 madrugada)
- CAPE/ERA5 goteando en background (5/48 ficheros al cierre; `descargar_cape_era5.py` resumible).
- Prototipo tiempo real: mapeo colector horario AEMET → features del modelo (los nombres/sesgos
  difieren; fichado en ESTRATEGIA_DATOS §4). `predecir_punto.py` ya resuelve el caso histórico.
- Opcionales de mejora: componentes FWI de EWDS, ganadería (join por unidad admin.), sensibilidad
  al ratio de negativos 1:1/1:10, calibración con más años cuando el EGIF 2021-22 se consolide.

## 15/07/2026 — Prototipo en tiempo real: diseño y construcción

### 14. Diseño del prototipo (decisiones y racional)

**Problema central**: el modelo se entrenó con features de rejilla (IberFire/ERA5, hasta 2024-12)
pero el "hoy" (jul-2026) solo es observable vía la red de estaciones AEMET. El mapeo diseñado:

| Feature del modelo | Fuente en tiempo real | Decisión |
|---|---|---|
| t2m_max/min, rh_min, viento_max, precip_dia, vpd_max | **AEMET diario** (API climatologías diarias, lag 3-4 días) + **colector horario** (agregado diario para los últimos días + hoy parcial) | mapeo directo: tmax/tmin, hrMin, racha→viento_max, prec |
| fwi + ventanas 7/15/30d | **FWI calculado por nosotros** (sistema canadiense Van Wagner completo: FFMC/DMC/DC/ISI/BUI/FWI) sobre la serie diaria AEMET desde el 1-may (≈75 días de spin-up → DC convergido, memoria ~52 d) | módulo propio `fwi_canadiense.py`, validado contra el FWI del cubo (misma meteo → correlación por celda) |
| fwi_pctl_local, fwi_anom_sigma | percentil vs climatología 2008-2014 **calculada con NUESTRO FWI sobre la meteo del cubo** en la celda de la estación — mismo algoritmo en numerador y denominador → cancela el sesgo de implementación; queda el sesgo estación-vs-rejilla (documentado como limitación) | precomputado por estación |
| ndvi, lai, swi010, lst | **climatología mensual del cubo 2020-2024** en la celda (no observables en tiempo real sin GEE) | precomputado; alternativa NaN descartada: la climatología es mejor proxy que la ausencia |
| estáticas (elevación…CLC, popdens) | cubo, celda de la estación (popdens 2020, CLC 2018) | precomputado |
| autorregresivas EGIF | conteos hasta 2020 (registro incompleto después) | caveat documentado: van a la baja |
| frp/n_detec 50km 7d | **FIRMS NRT API** (key ya en `.env`) con fallback a NaN si falla | módulo con caché |
| rayos_dia/7d | no disponible (WGLC llega a 2023; MTG-LI no integrado) → **NaN** (XGBoost lo maneja) | honesto; mejora futura con `eumdac` |
| es_festivo | fin de semana + festivos nacionales (librería `holidays`) | festivos autonómicos omitidos (aprox.) |

**Arquitectura** (4 piezas):
1. `fwi_canadiense.py` — FWI system puro numpy, validado contra el cubo.
2. `preparar_prototipo.py` — precálculo por estación (celda, estáticas, climatologías FWI propia
   y de vegetación, autorregresivas) → `prototipo/estaciones_prototipo.parquet`. Se ejecuta 1 vez.
3. `tiempo_real.py` — en caliente: API AEMET diaria (con caché) + colector horario + FIRMS NRT →
   serie FWI → vector de 50 features → probabilidad + SHAP.
4. `app_prototipo.py` — Gradio, 3 pestañas: Tiempo real (estación→riesgo+condiciones+SHAP),
   Histórico (lat/lon/fecha, motor de `predecir_punto.py`), Mapas (PNGs generados).

**Niveles de alerta**: cortes sobre la probabilidad (prevalencia de diseño 25%) fijados con los
cuantiles de las predicciones en val 2019: Bajo/Moderado/Alto/Extremo. Se documentará el corte.

**Limitaciones asumidas (para la memoria)**: sesgo estación puntual vs celda 9 km (mitigado por
la normalización local), FWI con aproximación tmax/hrMin en vez de valores a mediodía (práctica
común), autorregresivas congeladas en 2020, sin rayos en tiempo real.

### 15. Construcción del prototipo (implementación del diseño §14)
- **`fwi_canadiense.py`** validado contra el FWI oficial del cubo (misma meteo de entrada,
  60 celdas aleatorias, jun-dic 2014): correlación mediana **0.944** (p10 0.885); sesgo +8.9
  puntos (esperado: tmax/HRmin en vez de valores de mediodía) — irrelevante para el modelo porque
  el percentil local se calcula contra climatología generada con la MISMA implementación.
- **Niveles de alerta validados en val 2019** (prob a prevalencia de diseño 25%):
  | Corte | Nivel | % días-celda ≥ corte | Precisión | Recall |
  |---|---|---|---|---|
  | ≥0.25 | MODERADO+ | 34.8% | 0.65 | 0.88 |
  | ≥0.55 | ALTO+ | 22.0% | 0.81 | 0.70 |
  | ≥0.80 | EXTREMO | 13.5% | 0.93 | 0.49 |
- Verificaciones operativas previas: la API de climatologías diarias de AEMET sirve 2026 con lag
  3-4 días (probado: hasta 11-jul con todos los campos), y el colector horario (git pull) llega
  a hoy — juntos cubren la serie 1-may→hoy que necesita el spin-up del FWI (DC ~52 días).

### 16. Puesta en producción: tres bugs encontrados por lectura crítica (y valiosos para la memoria)

1. **NaN en features sin missing en entrenamiento** → predicción corrupta (Madrid salió EXTREMO
   0.83 con anomalía −2.3σ; el SHAP delató +0.91 log-odds de un `n_detec_50km_7d=NaN`). Regla
   aprendida: XGBoost solo gestiona bien el NaN en features donde VIO NaN al entrenar; en el
   resto lo manda por una rama arbitraria. Fix: fallbacks semánticos (0 = moda/sin detección).
2. **Racha ≠ viento del FWI**: usar la racha (70-90 km/h) en el sistema FWI infla el ISI
   exponencialmente (salían FWI medios de 129). El análogo correcto de `wind_speed_max` del cubo
   es el máx. de MEDIAS horarias del colector; para los días de la API diaria, min(1.5×velmedia,
   racha).
3. **Evaluar "hoy" con el día a medias**: a las 01 UTC el colector solo trae horas nocturnas →
   Tmax de madrugada (26°C evaluados vs 32.7°C de mediana nacional real). Fix: evaluar el último
   día COMPLETO (≥18 h). Mejora futura fichada: riesgo de HOY real con el forecast de AEMET
   (predicción municipio diaria).

**Hallazgo metodológico (cuantificado)**: las autorregresivas intra-celda acumulativas
(`n_fuegos_1km_hist`, `_10km_365d`, `_90d`) llevan un **artefacto del muestreo misma-celda**: el
incendio del positivo entra en el historial de los negativos posteriores de su celda → el modelo
aprende orden temporal ("historial 0 = pre-incendio = riesgo"), visible como +0.92 log-odds por
`n_fuegos_1km_hist=0` en pleno Madrid urbano. **Se reentrenó `xgb_v2_prototipo` sin las 4
features contaminadas: AUC-PR 0.828 (−0.015 vs 0.843)** — coste pequeño, modelo honesto para
producción. `n_fuegos_10km_mismomes_hist` se conserva (solo cuenta años estrictamente
anteriores → sin artefacto). Para la memoria: sección de limitaciones del diseño de negativos.

### 17. Prueba end-to-end del prototipo (día evaluado: 14-jul-2026, real)
| Estación | Tmax | HRmin | FWI (absoluto) | pctl local | Riesgo |
|---|---|---|---|---|---|
| Madrid Retiro | 35.9° | 15% | **68.4** | 56 | MODERADO (0.35) |
| Ourense | 32.6° | 33% | **41.3** | **91** | **ALTO (0.58)** |
| Zamora | 35.3° | 10% | 64.4 | 77 | ALTO (0.55) |
| Huelva Este | 28.9° | 54% | 43.8 | 6 | MODERADO (0.33) |

**La tesis del TFM funcionando en producción**: Madrid con FWI absoluto 68 queda MODERADO (es su
percentil 56 de julio); Ourense con FWI 41 sale ALTO (su percentil 91). El umbral absoluto habría
invertido el ranking. La app Gradio (`app_prototipo.py`, puerto 7860) sirve las 3 pestañas.

### 18. Validación out-of-time 2021-2024 con eventos FIRMS — `validar_eventos_firms.py` + visor
**La validación externa formal del modelo** (años nunca vistos: entrenó 2015-2018; verdad-terreno
FIRMS, independiente del EGIF que está incompleto 2021+):
- **Eventos**: DBSCAN espacio-temporal (6 km / ~2.4 días) sobre las detecciones FIRMS 2021-2024,
  filtrando fuentes estáticas industriales (píxel con detección en >15 días del cuatrienio —
  lección de la comprobación NRT: Cee, papeleras) y clusters <3 detecciones o >30 días →
  **1,710 eventos**.
- **Métrica**: para cada evento, serie DIARIA de probabilidad del modelo de producción
  (xgb_v2_prototipo) en su celda durante todo su año (features idénticas al pipeline) →
  percentil del día de inicio dentro del año.
- **Resultado agregado: mediana percentil 85.5 · 58% de los incendios en >p80 (azar: 20%) ·
  86% en >p50 (azar: 50%)** — el modelo concentra los incendios reales de años no vistos en
  sus picos de riesgo.
- **Los 8 famosos, todos ≥p84**: Sotalvo 99.5 · Losacio 99.7 · Bermeja 96.4 · Bejís 96.4 ·
  Vall d'Ebo 93.4 · Ateca 91.5 · **Asturias mar-2023: 98.4 con FWI 13.5** — el modelo capta la
  oleada cantábrica de primavera que el FWI absoluto NO ve (viento sur + sequedad relativa):
  el argumento definitivo de que el modelo aporta sobre el índice canónico.
- **Visor**: `eventos_firms_visor.html` (generado por `generar_visor_eventos.py`, Leaflet+Chart.js,
  4.6 MB autocontenido): mapa filtrable por año/tamaño/famosos, color = percentil, clic → curva
  anual de riesgo con la ventana del incendio sombreada + histograma de percentiles en vivo.
- Caveats documentados: FIRMS no detecta <1 ha (los 1,710 son incendios con señal térmica
  satelital, sesgo a medianos/grandes); autorregresivas EGIF congeladas en 2020; rayos=0 en 2024.

### 19. Operacionalización (15/07/2026 tarde): validación 2025-26, cron prospectivo y forecast D+1/D+2

**A. Validación 2025-2026 vía estaciones** (`descargar_datos_2025_2026.py` + `validar_eventos_estaciones.py`
+ `generar_visor_eventos.py --periodo 2025`): el cubo termina en 2024, así que para 2025-26 las
features salen de la MISMA tubería estación→features del prototipo (FWI propio con spin-up desde
nov-2024, vegetación climatológica, autorregresivas congeladas, FIRMS/rayos en moda) — valida a la
vez el modelo y el puente de producción. Eventos: FIRMS NOAA-20 2025→hoy (SP+NRT), DBSCAN idéntico
al de 2021-24, emparejados a la estación válida más cercana (≤35 km). Percentil dentro del año de
la estación (2026 parcial ene→hoy: caveat).

**B. Validación PROSPECTIVA (cron diario)** (`ranking_diario_cron.py` + `prototipo/cron_ranking.sh`,
crontab 09:30): cada mañana archiva el ranking de las ~705 estaciones en
`prototipo/historico_predicciones/ranking_<fecha>.parquet`. Ese histórico son predicciones emitidas
ANTES de los incendios → al final del verano, comparar con FIRMS = validación prospectiva pura
(imposible sobreajustar). Primer punto archivado: 14-jul-2026 (5 EXTREMO: Manilva, Pamplona,
Toledo, Chandrexa de Queixa, Arévalo). ⚠️ Caveat: el cron solo corre con el portátil encendido —
alternativa GitHub Actions (patrón del colector) fichada.

**C. Forecast D+1/D+2 — VIABLE, implementado** (investigación + PoC 15/07): la predicción municipal
de AEMET (`/prediccion/especifica/municipio/diaria/{ine}`) da tmax/tmin, HR mín/máx, viento (km/h)
y probabilidad de precipitación a 7 días; el maestro de municipios (8.122, con coordenadas) permite
mapear estación→municipio más cercano. Implementación en `tiempo_real.construir_features(horizonte)`:
la serie observada se extiende con los días previstos y el FWI SE PROPAGA (los códigos
FFMC/DMC/DC arrastran su memoria desde lo observado) — exactamente como operan EFFIS/ECMWF.
Decisiones: precip prevista = 0 mm si prob<60%, 2 mm si ≥60% (la predicción no da cantidad;
sesgo conservador hacia el riesgo); viento previsto = máx. de periodos / 3.6. Como el último día
completo es "ayer", horizonte 1 = HOY previsto y horizonte 2 = MAÑANA → objetivo tiempo real
+1/+2 días CUMPLIDO. Probado (Ourense 15/07): ayer ALTO 0.71 → hoy previsto ALTO 0.67 → mañana
0.58 (sube la humedad). En la app: selector "Ayer observado / HOY previsto / Mañana previsto".
Límite honesto: la calidad a D+1/D+2 hereda el error de la predicción meteo municipal; no se
recomienda pasar de D+2 (la precipitación como probabilidad degrada el DC a más plazo).

### 20. Resultados de la validación 2025-2026 vía estaciones (§19-A ejecutada)
- Datos: AEMET diario `todasestaciones` 521,959 filas (867 estaciones, nov-24→11-jul-26; el
  colector cubre el hueco) + FIRMS NOAA-20 75,470 detecciones (ene-25→15-jul-26, API ahora
  limita 5 días/petición). Probabilidad diaria del modelo: 374,059 estación-día (695 estaciones).
- Eventos: 1,633 clusters; 1,192 con estación a ≤35 km (mediana 12.4 km); 1,153 evaluables.
- **Resultado: mediana percentil 84.9 · 58% >p80 · 90% >p50 — REPLICA la validación del cubo
  (85.5 / 58% / 86%)**: el puente estación→features del prototipo no degrada el modelo.
- La ola histórica de agosto-2025 (León/Zamora/Ourense, eventos de >2,000 detecciones y
  FRP acumulado 50-86 GW): percentiles 75-97 — el sistema la habría señalado.
- Visor: `eventos_firms_visor_2025_2026.html` (mismo formato que el de 2021-24, con ayuda).

### 21. Cron migrado a GitHub Actions + fix del emparejamiento (15/07 tarde)
- **Fix de robustez en `validar_eventos_estaciones.py`**: el emparejador evento→estación solo
  probaba la más cercana; si esa tenía hueco de datos el día del fuego, el evento se perdía en
  silencio (~40 eventos, incluido el gran incendio de Almería del 10-jul-2026, 158 detecciones).
  Ahora prueba hasta 4 estaciones a ≤35 km → 1,177 eventos (los agregados no cambian: 84.9 /
  58% >p80). **Almería 10-jul: percentil 99.5 del año en Albox** (prob 0.77) — el modelo lo
  tenía como uno de los 1-2 días de más riesgo de 2026 en la zona.
- **Validación prospectiva blindada**: el ranking diario ahora corre en **GitHub Actions** en el
  repo privado del colector (`aemet-horario-verano2026`, workflow `ranking_diario.yml`, 08:30 UTC
  diario): job autocontenido (modelo + climatologías + estáticas en `modelo/`, 14 MB) que
  commitea `rankings/ranking_<fecha>.csv` — **cada predicción queda sellada con fecha por GitHub**,
  inmune a que el portátil esté apagado y a toda sospecha de predicción a posteriori. Probado en
  local antes del push (reproduce el ranking del cron local: mismas 5 EXTREMO el 14-jul).
  El cron local de las 09:30 se mantiene como respaldo. Pendiente del usuario: añadir el secret
  `FIRMS_MAP_KEY` en GitHub (opcional; sin él la feature de fuego cercano cae a 0).

### 22. Mapas D0/D+1 migrados a GitHub Actions — el sistema completo corre sin portátil (22/07 noche)
- **Problema**: los mapas nacionales del README del colector (`mapa_D0/D1.jpg`) los generaba
  `mapa_riesgo_hoy.py` en local (cron 09:30) → se congelaban con el portátil apagado (5 días
  parados a 21/07). El cubo IberFire (29 GB) parecía atarlos al portátil.
- **Clave del diseño**: el mapa NO necesita el cubo en runtime. De sus 46 features, las ~20 que
  salen del cubo son estáticas (elevación, CLC, población...) o climatología mensual 2020-24
  (NDVI/LAI/SWI/LST + densidad EGIF mismo-mes) → se congelan UNA VEZ en npz
  (`exportar_malla_gh.py` → `modelo/malla/` del repo del colector, ~35 MB: estáticas 15.5 MB +
  4 meses × 5 MB + mapeo estación→municipio). La meteo diaria ya salía de AEMET.
- **`mapa_diario.py`** (repo colector, workflow `mapa_diario.yml` 04:45 UTC diario): serie de 80
  días de todas las estaciones → forecast municipal AEMET (1 fetch/municipio, throttling 1.2 s,
  reintentos con espera si 429; en la prueba: 663 municipios, 2 sin predicción, cero 429) → FWI
  propagado → features por estación para D0 y D1 (guardarraíl: se descartan estaciones con >5
  días propagados) → IDW a la malla 1 km + capas congeladas + FIRMS NRT → XGBoost → JPG.
- **Salidas por ejecución**: `mapas/mapa_D{0,1}.jpg` (README, visibles en la app móvil de GitHub),
  tabla top-20 de estaciones inyectada en el README (marcadores `<!-- RANKING:... -->`), y
  `rankings/prevision_D{0,1}_<fecha>.csv` con las ~683 estaciones — **la previsión de mañana
  queda sellada por el commit ANTES de que ocurra** (validación prospectiva del forecast, no solo
  del nowcast como el ranking §21).
- **Capa de verdad-terreno en el mapa**: focos FIRMS de los últimos 5 días dibujados encima
  (ayer en aspas, D-5…D-2 en círculos; azul para no confundir con la escala). Regla: mostrar ≠
  alimentar — como feature siguen entrando solo D-5…D-1 (circularidad documentada §14).
- **Prueba de punta a punta en local (22/07 23:31 UTC)**: 683 estaciones válidas, D0 con 17
  EXTREMO (Alhama de Aragón 0.905, Tudela, Monreal — coherente con el ranking del 20/07,
  correlación 0.735), focos de ayer cayendo sobre las zonas rojas (Zaragoza, Madrid-Guadalajara).
  Primera ejecución programada: 23/07 04:45 UTC. El cron local de mapas queda desactivado
  (`prototipo/cron_ranking.sh`); `mapa_riesgo_hoy.py` pasa a uso manual (figuras de la memoria).
- **Endurecimiento tras la primera ejecución en Actions (23/07 madrugada)**: el run manual murió
  por timeout (90 min) con solo ~300/663 municipios — **desde los runners de GitHub (IPs de
  datacenter) AEMET devuelve 429 ~5× más a menudo que desde casa**. Solución en dos niveles:
  (a) presupuesto de tiempo para AEMET (`PRESUPUESTO_AEMET_S`, 35 min por defecto; el orden de
  consulta se baraja por hash del idema para que si corta, la cobertura AEMET siga siendo
  nacional y no "las provincias que empiezan por 0"); (b) las estaciones restantes caen a
  **Open-Meteo en bloque** (~100 estaciones/petición, segundos, sin key; da además cantidad de
  lluvia real, no probabilidad). AEMET sigue siendo la fuente primaria/oficial; el CSV lleva
  columna `fuente_forecast` para trazar qué estación usó qué. Probado el fallback forzando
  presupuesto 60 s: 600+ estaciones vía Open-Meteo, mapas correctos, y el D0 del 23/07 replica
  la distribución que el D1 del run anterior había previsto (61% ALTO). Cordura adicional:
  respetar el límite/minuto de Open-Meteo (pausa entre lotes, backoff 45 s).
  **Verificación final en Actions (run 29971191325, 23/07 01:12-01:52 UTC, verde)**: 40 min,
  355 municipios AEMET dentro del presupuesto + 315 estaciones vía Open-Meteo (100% resueltas),
  0 estaciones sin predicción, 682 con features válidas; el bot commiteó mapas, tabla y CSVs
  (7f9dec5). Consumo de Actions estimado: ~1,500-1,600 min/mes de los 2,000 gratuitos del repo
  privado (bajar `PRESUPUESTO_AEMET_S` si se acerca al tope).

### 23. Verificación predicción→realidad visible + hindcast de julio (23/07 madrugada)
- **Verificación automática diaria** (en `mapa_diario.py`, cron 04:45): cada mañana se contrasta
  la previsión SELLADA por commit el día anterior (D0 y D1 por separado — D1 es forecast puro)
  con los focos FIRMS reales de ayer. Emparejamiento foco→estación ≤35 km (criterio de la
  validación del TFM). Métricas: pctl mediano del riesgo previsto donde ardió, % de focos en
  zonas ALTO/EXTREMO vs % del territorio (base), y **lift** = ratio de ambos. Se acumula en
  `rankings/verificacion_diaria.csv` y se enseña como tabla en el README (móvil).
- **Hindcast de julio** (`retro_julio.py`, una vez, local): mapas 1-21 jul con meteo observada
  + focos FIRMS del propio día encima (`mapas/historico/<fecha>_D0_retro.jpg`), CSV por día y
  resumen mensual `mapas/seguimiento_julio_2026.png`. Etiquetado RETRO en todo — no confundir
  con lo sellado (ranking desde 15/07, mapas D0/D1 desde 22/07).
- **Resultados julio (21 días retro)**: pctl mediano del riesgo donde ardió ~81 (mediana del
  mes), lift 1.4-3.2 la mayoría de días; los mejores días son los de más fuego (17-jul: 540
  focos, pctl 99.2, lift 3.2 — el enjambre cae sobre la mancha de Navarra/Aragón). Días de
  fallo claros y documentables: 3-jul (lift 0.15), 12-jul (pctl 2.9) y 8-jul (0.84) — fuegos
  sin señal meteo, consistente con la limitación ya conocida (§16). **Primera verificación
  sellada (22/07 D0): 274 focos, pctl mediano 95.0, 66.7% en ALTO+ vs 39.8% base, lift 1.67.**
- FIRMS NRT admite fechas pasadas en trozos de 5 días (`/5/<fecha>`) — usado para el hindcast
  (26-jun→25-jul, 7,664 detecciones).
- **Capa de focos activos (23/07)**: workflow ligero `focos_activos.yml` (11/14/17 UTC) que
  publica `mapas/focos_activos.jpg` — VIIRS NOAA-20+21 (375 m, ~4 pasadas/día, NRT ~3 h) sobre
  relieve, hoy en rojo / ayer en naranja. Evidencia independiente del modelo (mostrar ≠
  alimentar): cubre el hueco de que el mapa de riesgo se congela a las 06:45. El gráfico de
  seguimiento de julio se rehízo autoexplicativo (zonas verde/roja, anotaciones, pie).

## 11/08/2026 — v3: reajuste con todos los datos y dos resultados negativos útiles

### 24. Modelo v3 — `entrenar_modelo_v3.py`, `extraer_historia_firms_v3.py`, `ablacion_v3_firms.py`
Motivación y criterio en [`EVALUACION_DETALLADA.md`](EVALUACION_DETALLADA.md) §10. **Nada se
sobrescribe**: todo sale con sufijo `_v3`. Principio que ordena la sesión: *el protocolo de
evaluación y el modelo de producción no tienen por qué ser el mismo objeto* — se evalúa con el
split congelado, se reporta ESE número, y se despliega un modelo reajustado con todo.

**A. Reajuste con 2015-2020 completo (la razón de ser de v3).** v1/v2 se entrenaron solo con train
2015-2018 (53,563 filas): val 2019 y test 2020 (23,103 filas, **+43%**, y los dos años más
recientes) quedaban sin usar. El modelo final de v3 se ajusta con las 76,666 filas.
- Regla que no se rompe: el modelo final **NO se evalúa sobre 2020** (lo tiene dentro). Las
  métricas que lo amparan son las de su **gemelo del protocolo** (train 2015-2018, mismas features
  y mismos hiperparámetros): **AUC-PR 0.832 · ROC 0.923 · Brier 0.096 · TSS 0.677** (v2: 0.828).
- Control de cordura (no es una métrica): correlación del ranking sobre 2020 entre el gemelo del
  protocolo y el modelo final = **0.925** (0.975 en la variante ligera).

**B. Sustituir la autorregresiva EGIF congelada por FIRMS: PROBADO Y DESCARTADO.**
`n_fuegos_10km_mismomes_hist` es la nº1 en SHAP y en producción está congelada en 2020. Se
construyó su equivalente FIRMS —no caducable— con tasas normalizadas por años de registro
disponibles (`firms_diasfuego_mismomes_tasa`, `firms_diasfuego_tasa`, `firms_frp_p95_hist`,
`firms_anios_previos`; la normalización funciona: las medias por año quedan planas, 0.61-0.65, sin
tendencia espuria). Ablación (`dataset/ablacion_v3_firms.json`, AUC-PR en val 2019):

| Variante | AUC-PR val | Lectura |
|---|---|---|
| A · v2 completo con EGIF (46f) | 0.8543 | lo que se mide hoy, no lo que se opera |
| B · **sin** la autorregresiva (45f) | 0.8401 | cota inferior del futuro |
| C · FIRMS reemplaza al EGIF (49f) | 0.8400 | **no recupera nada (−0.0001)** |
| D · EGIF + FIRMS (50f) | 0.8521 | tampoco son complementarias |

- **El reemplazo FIRMS no aporta.** Explicación: bajo muestreo de celda fija, una propensión
  estructural que es casi constante por celda no puede discriminar; y lo que sí varía —el entorno
  de fuego reciente— ya entra por `frp_max_50km_7d`/`n_detec_50km_7d`, que **ya son FIRMS y ya
  están vivas** en producción.
- **El resultado útil es el otro**: eliminar por completo la autorregresiva EGIF cuesta solo
  **−0.014 AUC-PR** (A−B). Es decir, **la feature congelada en 2020 NO es load-bearing y el
  problema de caducidad está acotado en −0.014**, muy por debajo de lo que se temía. Corrige la
  estimación de EVALUACION_DETALLADA §10.3.c (que lo daba como "impacto alto").

**C. Hallazgo metodológico: la prevalencia varía MUCHO por año.** Al montar la CV temporal
apareció que las pseudo-ausencias se sortearon con fecha uniforme dentro del **split**, no del
**año**; como los positivos siguen el calendario real de incendios, la prevalencia por año va del
**15.2% (2018) al 37.0% (2017)**:

| Año | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|---|
| Prevalencia | 30.3% | 18.0% | **37.0%** | **15.2%** | 25.7% | 26.3% |

- Consecuencia directa: **el AUC-PR crudo NO es comparable entre años** (su suelo *es* la
  prevalencia). El fold de 2018 "se hunde" a 0.659 sobre todo por eso. Se reporta **AUC-PR
  normalizado** = (AP − prev)/(1 − prev) y AUC-ROC.
- CV temporal rolling-origin (2015-16→17, 2015-17→18, 2015-18→19, 2015-19→20):
  **AUC-PR normalizado 0.737 ± 0.081 · AUC-ROC 0.916 ± 0.015**. El AUC-ROC es notablemente estable;
  2018 sigue siendo el año más difícil ya normalizado (0.598 vs 0.76-0.80) — año húmedo y con pocos
  incendios, coherente con la limitación conocida de los fuegos sin señal meteorológica.
- ⚠️ Esto no invalida nada de v1/v2: el protocolo congelado usa val=2019 y test=2020 como años
  ENTEROS, y dentro de cada uno la prevalencia es la de diseño. Solo afecta a comparar años entre sí.

**D. Cortes de alerta recalculados out-of-fold** (§10.1-b confirmado empíricamente). Los de v2
(0.25 / 0.55 / 0.80) salieron de los cuantiles de val 2019; al entrar 2019 en entrenamiento esas
predicciones serían in-sample. Los de v3 se derivan de predicciones **out-of-fold de 2019-2020**
(prevalencia 25.9%, la de diseño; se excluyen 2017 y 2018 para que un artefacto del muestreo no
desplace los cortes), fijando la **misma cobertura operativa** que v2:

| Nivel | Cobertura | v2 | **v3** | v3_lite | Precisión | Recall |
|---|---|---|---|---|---|---|
| MODERADO+ | 34.8% | 0.25 | **0.282** | 0.295 | 0.64 | 0.86 |
| ALTO+ | 22.0% | 0.55 | **0.490** | 0.509 | 0.80 | 0.68 |
| EXTREMO | 13.5% | 0.80 | **0.675** | 0.712 | 0.92 | 0.48 |

Los cortes se mueven de verdad (EXTREMO 0.80 → 0.675): **reutilizar los de v2 habría hecho que el
sistema avisara de menos**. Confirma que recalcularlos no era una formalidad.

**E. Variante ligera — decisión de ingeniería documentada.** Los hiperparámetros de Optuna
(depth 10, lr 0.017 → 590 árboles) ganan **+0.0014 AUC-PR en val** sobre los de base… y multiplican
por 13 el fichero: **10.8 MB vs 0.8 MB**. El modelo viaja dentro del repo del colector para el job
diario de GitHub Actions (§22), donde cada MB se paga.

| | Test 2020 (protocolo) | CV norm. | Tamaño | Árboles |
|---|---|---|---|---|
| `xgb_v3.ubj` | AP 0.8316 · ROC 0.9234 | 0.737 ± 0.081 | 10.8 MB | 590 |
| **`xgb_v3_lite.ubj`** | AP 0.8278 · ROC 0.9216 | 0.736 ± 0.081 | **0.8 MB** | 212 |

Diferencia indistinguible (+0.004 AP, +0.0003 en CV; correlación de ranking entre ambos 0.967).
**Recomendación de despliegue: `xgb_v3_lite.ubj`** — 13× más pequeño por un decimal que está dentro
del ruido. Se conservan los dos con sus metadatos y cortes propios.

**F. Salidas** (ninguna pisa nada previo): `modelos/xgb_v3.ubj`, `modelos/xgb_v3_lite.ubj` +
`*_metadata.json` (features, params, cortes, métricas que los amparan, advertencias de uso),
`dataset/metricas_v3.json`, `dataset/ablacion_v3_firms.json`,
`dataset/features_firms_hist_v3.parquet`, `dataset/oof_v3*.parquet`, `eda/pr_curves_v3*.png`,
`eda/shap_summary_v3*.png`, logs en `dataset/*_v3.log`.

**G. Pendiente para pasar v3 a producción** (no hecho en esta sesión, requiere tocar el prototipo
y el repo del colector): apuntar `tiempo_real.py` / `mapa_riesgo_hoy.py` / el job de Actions al
nuevo modelo **y a sus nuevos cortes** (van juntos, no se pueden mezclar), y revalidar con
`validar_eventos_firms.py` / `validar_eventos_estaciones.py` para comprobar que los percentiles
agregados (85.5 / 84.9) se mantienen.

## 11/08/2026 (tarde) — Cuarta fuente de verdad: los partes oficiales de MITECO

### 25. Validación contra MITECO — `validar_miteco.py`
**Qué se ha incorporado.** El repo [TFM-RAG de Atomas9](https://github.com/Atomas9/TFM-RAG)
(clonado en `../TFM-RAG`, **usado en solo lectura, sin modificar ni escribir nada en él**) monta un
RAG sobre los **partes diarios de actuaciones en incendios forestales del MITECO** y trae un parser
de PDF que extrae, por incendio: comunidad, provincia, **localización (municipio)**, estado
(ACTIVO/CONTROLADO/ESTABILIZADO/EXTINGUIDO), medios asignados y fecha del parte, con un
`incident_key` que agrupa observaciones del mismo incendio en días distintos.

**Por qué es la fuente de verdad que faltaba.** Las tres etiquetas anteriores tienen huecos
conocidos: el EGIF está incompleto desde 2021; FIRMS no ve fuegos pequeños ni bajo nubes y mete
falsos positivos industriales; EFFIS es limpio pero llega con días de retraso. MITECO mide otra
cosa y **es independiente de las tres**: registra los incendios en los que el Estado **desplegó
medios**, es decir, los que fueron lo bastante graves como para movilizar recursos. Eso es
justamente lo que un sistema de alerta debería anticipar. Sus sesgos van en dirección contraria a
los de FIRMS, y por eso las dos juntas informan más que cualquiera sola.

**Datos.** 30 partes → **217 incendio-parte, 114 incidentes únicos** (5-jul → 10-ago-2026).
Geocodificación municipio → coordenadas contra el maestro AEMET de 8,122 municipios ya cacheado
(`prototipo/cache/municipios.json`), priorizando la provincia del parte: **96% de coincidencias
exactas, 4% difusas, 0 sin resolver**. Se cruzan con las previsiones **selladas por commit** del
repo del colector. Metodología calcada de `validar_modelo.py` (unidad = estación-día, radio 25 km)
para que los números sean comparables entre etiquetas.

**Resultado (modo INICIO: primera aparición de cada incidente = proxy de la ignición; el test
correcto para un modelo de riesgo de ignición, y evita que un fuego de 5 días cuente 5 veces):**

| Previsión | Días | Prevalencia | AUC-ROC | Lift decil sup. | Pctl mediano donde ardió |
|---|---|---|---|---|---|
| **D0 (sellada)** | 15/17 | 1.17% | **0.644** | **2.15×** | **72.3** |
| **D1 (sellada, forecast puro)** | 15/17 | 1.39% | 0.607 | 1.47× | 65.3 |
| ranking (día cerrado) | 19/21 | 1.19% | 0.651 | 3.57× | 71.5 |
| retro (hindcast) | 9/21 | 1.96% | 0.546 | 1.45× | 57.1 |

**El modelo discrimina de verdad contra la fuente oficial**, y además **a prevalencia realista
(1.2%)**, no a la del diseño (26%): AUC-ROC 0.644 con 14 de 15 días por encima de 0.5, y los
incendios caen en el percentil 72 del riesgo previsto frente al 50 que daría el azar. D1 —previsión
pura emitida la víspera— aguanta pero pierde (0.607 / p65), coherente con que hereda el error de la
predicción meteorológica municipal.

### ⚠️ El hallazgo importante: los números con FIRMS estaban INFLADOS
Comparando **las mismas previsiones en los mismos 13 días** con las dos verdades-terreno
(`dataset/comparacion_miteco_vs_firms.csv`):

| | MITECO | FIRMS |
|---|---|---|
| Unidades contadas | 36 incidentes | 6,756 focos |
| Pctl mediano donde ardió | **72.3** | **89.5** |

El patrón es sistemático y tiene una explicación clara, **ya fichada como limitación conocida en el
propio `validar_modelo.py` del colector**: FIRMS cuenta **un registro por píxel VIIRS**, así que un
único incendio grande situado en zona de riesgo alto aporta cientos o miles de filas y **domina el
día (pseudo-replicación)**. Se ve en la tabla: los días de mega-incendio (23-30 jul, 700-1,300
focos) FIRMS da percentil 94-99 y MITECO 58-92; los días tranquilos (1-6 ago, ~100 focos) se
invierte y MITECO da MÁS que FIRMS (52-84 vs 50-61).

**Consecuencia para la memoria:** las cifras de las validaciones §18 y §20 (mediana percentil 85.5
y 84.9) **están medidas por foco y por tanto sobreestiman la capacidad de discriminación por
incendio**. No son falsas —responden a "¿dónde cae la actividad de fuego detectada?"— pero la
pregunta que interesa a un servicio de emergencias es "¿dónde empieza un incendio que exige
medios?", y ahí la respuesta honesta es **percentil 72, no 85**. Hay que reportar las dos y explicar
la diferencia: es un punto fuerte de rigor, no una debilidad.

**Caveats declarados** (todos en `dataset/validacion_miteco_*.json`):
- MITECO solo recoge incendios con despliegue del Ministerio → **sesgo fuerte a incendios grandes;
  no es un censo de igniciones**. Y que un fuego se haga grande depende también de la respuesta, el
  viento del momento y el terreno, no solo del riesgo de ignición que modela el sistema.
- La localización es el **municipio**: se geocodifica a su centro, no al punto de inicio.
- La primera aparición en los partes es un **proxy** de la ignición: un fuego iniciado antes del
  primer parte se fecha más tarde de lo real.
- **Muestra pequeña**: 13-15 días evaluables y 36 incidentes en el solape con las previsiones
  selladas. Días con 1-3 incendios dan lifts de decil de 0.00 por puro ruido (3, 4 y 6 de agosto).
  Los agregados tienen incertidumbre alta y conviene reejecutar al cerrar el verano.

**Salidas** (nada pisado): `dataset/validacion_miteco_{inicio,todos}.csv` y `.json`,
`dataset/comparacion_miteco_vs_firms.csv`. Modo `--modo todos` (todos los incendio-día, otra
pregunta: "¿acierta los días con incendios activos?"): D0 AUC-ROC 0.633 · lift decil 2.81 · p70.5.

**Dependencias añadidas al entorno** (no al repo de Atomas9): `pymupdf`, `pypdf`, `rapidfuzz`,
`unidecode`.

### 26. Panel de validación y el contraste prospectivo contra el FWI (11/08/2026)
`dashboard_miteco_datos.py` + `panel_miteco_plantilla.html` → `panel_validacion_miteco.html`
(autocontenido, doble clic) y `dataset/panel_miteco_artifact.html` (versión publicable).

**El hallazgo que faltaba por explotar**: los CSV sellados del colector no traen solo `prob` —
traen también **`fwi` y `fwi_pctl_local`**. Es decir, permiten comparar **prospectivamente**, sobre
incendios oficiales y con predicciones selladas por commit antes de los hechos, si el modelo aporta
sobre el índice canónico. Es la tesis del TFM sometida a su prueba más dura, y hasta ahora solo se
había contrastado retrospectivamente sobre el dataset de diseño (escalera de baselines §8).

**Puntuación**: unidad estación-día; cada día las ~684 estaciones se ordenan y se convierten a
**percentil DENTRO de ese día**. Así se mide lo que hace un operador ("hoy, ¿a qué zonas miro?") y
se evita que una ola de calor nacional —donde todo puntúa alto— infle la estadística. Es más
exigente que agrupar probabilidades crudas.

| Score | AUC-ROC | Lift decil sup. | Pctl mediano | % incendios con el 10% del territorio |
|---|---|---|---|---|
| **Modelo (XGBoost)** | **0.640** | **2.33×** | **72.2** | **23%** |
| FWI percentil local (contribución del TFM) | 0.588 | 1.58× | 61.4 | 10% |
| FWI absoluto (estándar internacional) | 0.559 | 1.08× | 60.7 | 11% |

- **El modelo bate al estándar internacional en validación prospectiva sobre verdad-terreno
  oficial.** El FWI absoluto tiene un lift de decil de **1.08×**: en el 10% del territorio que
  marca como más peligroso caen prácticamente los incendios que tocarían por azar — a efectos
  operativos de priorización diaria, **no discrimina**. El modelo concentra 2.33×.
- **El percentil local solo (2 features) ya bate al FWI absoluto** (1.58× vs 1.08×): confirma
  prospectivamente la ablación §9, que era retrospectiva.
- **Niveles operativos**: EXTREMO cubre el **3.2% del territorio y captura el 10.0%** de los
  incendios (lift 3.08×); ALTO+ el 37.4% → 59.2% (1.58×); MODERADO+ el 89.0% → 93.3% (1.05×, es
  decir, el corte bajo apenas informa: casi todo el país queda dentro).

**Filtro declarado**: 1 incendio (TELDE, Las Palmas) queda fuera de la península, donde el modelo no
tiene cobertura. Se descarta **de forma explícita** (`filtrar_peninsula`), no en silencio: un
descarte callado se leería como "se ha evaluado todo" sin serlo.

**Diseño del panel**: paleta y reglas de la guía de visualización (validador de daltonismo ejecutado
en modo claro y oscuro, todas las comprobaciones pasadas); tema claro/oscuro por tokens; cada
gráfico con vista de tabla (requisito de accesibilidad y relieve del aviso de contraste del aqua);
tooltips accesibles por teclado. Estructurado por preguntas —¿discrimina? ¿aporta sobre el índice?
¿qué pasa en los avisos? ¿por qué con satélite parecía mejor?— en vez de por tipo de gráfico.
Publicado como artifact: https://claude.ai/code/artifact/33e83812-9060-46b0-ae33-d04163395066

⚠️ Los caveats de §25 (muestra pequeña, sesgo a incendios grandes, geocodificación al centro del
municipio, día de ignición como proxy) aplican igual y están escritos en el propio panel.

## 11/08/2026 (noche) — Evaluación operativa con baselines e intervalos: el resultado incómodo

### 27. `descargar_verdad_operativa.py` + `validar_operativo.py`
Se atacan los tres defectos que quedaban en la evaluación: la pseudo-replicación de FIRMS, la
ausencia de los baselines que importan y la ausencia de incertidumbre. **3 verdades-terreno × 6
scores × 3 radios, con intervalos de confianza.** Ningún repo externo se modifica.

**A. Datos nuevos** (a `dataset/`, no al repo del colector): EFFIS reproducido vía WFS abierto
(**1,841 perímetros** de España, 13-ene→11-ago-2026, con `FIREDATE` y `AREA_HA`; el fichero que
había en el colector se cortaba el 27-jul) y FIRMS del periodo (**26,461 detecciones**, la copia
local llegaba solo a 15-jul).

**B. Pseudo-replicación corregida**: las detecciones se agrupan con DBSCAN (6 km / 2.4 d, mismos
parámetros que §18), se filtran fuentes térmicas fijas (mismo píxel >8 días → 96 focos
industriales, 2,140 detecciones) y **cada evento cuenta UNA vez, en su día de primera detección**.
23,329 detecciones → **405 eventos** (mediana 5 detecciones/evento). El mega-incendio de 1,300
píxeles pasa a valer lo mismo que el de 3.

**C. Los baselines que faltaban.** `climatologia` = nº medio de incendios EGIF 2015-2020 al año a
<radio km de la estación en ±10 días de calendario (sin meteorología: "lo que pasa normalmente aquí
por estas fechas"; construida con años anteriores → no puede contener información del periodo
evaluado). `persistencia` = detecciones FIRMS a <50 km en los 7 días previos, que ya venía en los
CSV sellados.

**D. Incertidumbre**: bootstrap de bloques **por día** (2,000 réplicas), no por estación-día —
las ~684 estaciones de una jornada comparten la situación sinóptica y remuestrearlas por separado
daría intervalos falsamente estrechos.

#### Resultados (previsión D0 sellada, radio 25 km, AUC-ROC [IC 95%] · lift del decil superior)

| Score | EFFIS (166 pos.) | MITECO (120 pos.) | FIRMS-eventos (304 pos.) |
|---|---|---|---|
| **Modelo (XGBoost)** | 0.627 [0.56–0.70] · 2.89× | **0.640** [0.60–0.68] · 2.33× | 0.562 [0.48–0.65] · 2.11× |
| FWI absoluto | 0.559 [0.49–0.63] · 1.21× | 0.559 [0.50–0.61] · 1.08× | 0.511 [0.44–0.58] · 1.28× |
| FWI percentil local | 0.574 · 1.15× | 0.588 · 1.58× | 0.550 · 1.48× |
| **Climatología (gratis)** | 0.607 · 1.69× | 0.602 · **2.67×** | **0.602** · 1.81× |
| **Persistencia (gratis)** | **0.640** · **2.95×** | 0.616 · 2.25× | 0.614 · 1.78× |
| **Modelo + climatología** | 0.639 · 2.65× | **0.649** · 2.42× | 0.598 · 2.01× |

#### Lo que se confirma
**El modelo bate al FWI en las 9 combinaciones** (3 etiquetas × 3 radios), y la conclusión es
estable al radio (10/25/50 km) y coherente entre fuentes con sesgos opuestos. La tesis del TFM
—que el ML aporta sobre el índice canónico— queda **verificada prospectivamente, sobre verdad-terreno
oficial, con predicciones selladas antes de los hechos y con intervalos de confianza**. Es la
validación más fuerte del trabajo. El FWI absoluto tiene lift 1.08-1.28×: para priorizar cada
mañana, prácticamente no discrimina.

#### ⚠️ Lo que NO se confirma (y hay que llevar a la memoria)
**El modelo no demuestra aportar sobre baselines gratuitos.** Climatología y persistencia igualan
o superan al modelo en varias celdas: persistencia gana en EFFIS (0.640 vs 0.627), climatología
gana en FIRMS (0.602 vs 0.562) y da más lift que el modelo en MITECO (2.67× vs 2.33×).
**Todos los intervalos se solapan ampliamente**: con 15-17 días nada está separado
estadísticamente. La lectura honesta no es "el modelo falla", sino **"en esta ventana no está
demostrado que aporte sobre saber dónde suele arder y dónde ardió la semana pasada"**.

Matices que hay que dar a la vez:
- **La ventana es el peor escenario posible para el modelo**: 15-17 días de pleno agosto, con media
  España en riesgo alto. Ya estaba fichado en §12 ("en un día donde media España está en riesgo
  extremo real, la discriminación intra-día es moderada"). El valor del modelo debería estar en
  separar regímenes y en los hombros de temporada, no dentro del percentil 99 de una ola.
- **Ambos baselines están DENTRO del modelo**: `n_fuegos_10km_mismomes_hist` ≈ climatología,
  `frp_max_50km_7d`/`n_detec_50km_7d` ≈ persistencia. Que el modelo no los bata sugiere que las
  features meteorológicas aportan poco en este régimen, o que el modelo infrapondera la señal
  autorregresiva respecto a lo que sería óptimo operativamente.
- **Conexión con §24**: la climatología del modelo está **congelada en 2020** en producción. Una
  climatología recalculada la bate. El sustituto FIRMS que se probó y descartó en §24 falló
  probablemente por estar mal construido (propensión casi constante por celda), no porque la idea
  fuera mala: una climatología estacional EGIF con ventana de calendario **sí** funciona.
- `persistencia` frente a la etiqueta `firms` es **parcialmente circular** (misma fuente): esa celda
  no se debe interpretar. Frente a EFFIS y MITECO no lo es.

#### La mejora accionable
**`modelo + climatología`** (media de los dos percentiles intra-día, sin reentrenar nada) es el
mejor o empatado en la mayoría de celdas: **0.649 en MITECO-25 km, 0.650 en EFFIS-50, 0.651 en
MITECO-50**, por encima del modelo solo en todas ellas. Es una mejora gratuita e inmediata para
producción, y sugiere que el camino no es tocar el modelo sino **combinarlo con una climatología
viva**.

**Salidas**: `dataset/validacion_operativa.{csv,json,log}`,
`dataset/effis_ba_season_2026.geojson`, `dataset/firms_ventana_2026.parquet`.

### 28. Contrastes emparejados: la probabilidad de que el modelo vaya mejor (11/08/2026)

**Corrección metodológica a §27.** Allí se concluyó "todos los intervalos se solapan, luego nada
está separado estadísticamente". **Eso es un error clásico**: comparar dos intervalos de confianza
por separado no es un contraste. Dos IC pueden pisarse y aun así la diferencia entre ambas
cantidades ser decisivamente distinta de cero, porque ambos scores se miden **sobre los mismos
días** y sus errores están muy correlacionados — al restar se cancela la incertidumbre compartida
(qué días tocaron en la réplica).

**Lo correcto**: remuestrear la DIFERENCIA de forma emparejada. En `validar_operativo.py`, la
función `bootstrap()` evalúa ahora todos los scores sobre el **mismo** remuestreo de días y
`contraste()` devuelve la distribución de Δ y **`p_mejor` = fracción de réplicas en las que el
modelo supera al rival**. Ésa es la lectura directa de *"¿qué probabilidad hay de que vaya mejor?"*.
Es una probabilidad **bootstrap**, no una posterior bayesiana: mide el respaldo que estos datos dan
a la superioridad, y hereda las limitaciones de la muestra (pocos días, un solo régimen estacional).

**Resultados (D0 sellada, radio 25 km, Δ AUC-ROC emparejado · P(modelo mejor)):**

| El modelo vs… | EFFIS | MITECO | FIRMS-eventos |
|---|---|---|---|
| **FWI absoluto** | +0.070 [+0.034,+0.114] · **100 %** | +0.082 [+0.032,+0.134] · **100 %** | +0.051 [+0.018,+0.087] · **100 %** |
| FWI percentil local | +0.056 [+0.002,+0.126] · **97.9 %** | +0.052 [−0.036,+0.130] · 88.1 % | +0.014 · 66.0 % |
| Climatología | +0.020 · 70.2 % | +0.037 · 84.5 % | −0.040 · 16.5 % |
| Persistencia | −0.013 · 40.7 % | +0.023 · 78.7 % | −0.049 · 15.5 % |
| *(combinado vs modelo solo)* | +0.012 · 69.8 % | +0.009 · 66.6 % | +0.036 · **96.4 %** |

**Lo que cambia respecto a §27:**
- **Contra el FWI el resultado es DECISIVO, no ambiguo.** P = 100 % en las tres etiquetas y el
  intervalo de la diferencia no toca el cero en ninguna. §27 lo infravaloró al mirar los IC
  individuales, que sí se solapaban. **La tesis del TFM queda confirmada con un contraste formal.**
- Contra el **percentil local**, el modelo gana con EFFIS (97.9 %) pero no de forma concluyente con
  las otras dos: coherente con que el percentil local es una *parte* del modelo.
- Contra **climatología y persistencia** sigue sin haber evidencia: 70 % / 85 % / 17 % y 41 % / 79 %
  / 16 %. Ni se demuestra que gane ni que pierda; con FIRMS la evidencia apunta en contra
  (P = 16.5 % → un 83.5 % de respaldo a que la climatología es MEJOR).
- El **blend modelo+climatología** tiene su mejor evidencia con FIRMS (96.4 %) y solo apoyo débil
  con las otras dos. Sigue siendo la mejora más barata, pero tampoco está demostrada.

**Umbral de lectura usado en el script**: P ≥ 95 % → "SÍ"; P ≤ 5 % → "no"; en medio, "no
concluyente". Es deliberadamente conservador y equivale a un contraste de una cola al 5 %.
