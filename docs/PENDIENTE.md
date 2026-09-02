# Qué le falta a este repositorio

> Escrito el 25/08/2026 mirando el repo como lo miraría un tribunal: con la
> memoria al lado y quince minutos. Para discutir en equipo antes de repartirlo.

Lo que ya está: los 120 scripts por capítulos con su procedencia
([`PROCEDENCIA.md`](PROCEDENCIA.md)), un README por capítulo, la tabla de
trazabilidad número → script ([`TRAZABILIDAD.md`](TRAZABILIDAD.md)), la bitácora
([`BITACORA.md`](BITACORA.md)), las limitaciones ([`LIMITACIONES.md`](LIMITACIONES.md))
y 48 MB de muestra de julio de 2026.

Lo que falta se ordena aquí por **cuánto cambia la impresión de quien lo abre**,
no por dificultad.

---

## 1. No se puede montar el entorno — el agujero más grave

**No hay `requirements.txt` ni `environment.yml`.** El código importa xarray,
xgboost, pyproj, geopandas, rasterio, cdsapi y una docena más, y no hay ni una
versión declarada. Sin eso, «reproducible» es una afirmación sin respaldo: el
primero que clone se queda en el primer `import`.

**Qué hacer:** congelar el entorno conda `tfm_fuego`, que es donde se ejecutó
todo, en `environment.yml`, y un `requirements.txt` mínimo para quien use pip.

**Coste:** 15 minutos. **Bloquea:** todo lo demás de esta lista.

## 2. La promesa de «clona y corre» está escrita pero no verificada

`muestras/` permite ejecutar la rama de servicio —la misma que corre en GitHub
Actions, que tampoco tiene el cubo—, pero **no hay un comando que lo demuestre**.
En la primera prueba, el script que se intentó (`malla_05_riesgo.py`) falló
porque lee el cubo día a día. Es el límite honesto de la muestra y está
documentado, pero deja la promesa sin ejemplo.

**Qué hacer:** un `demo_muestras.py` que con julio de 2026 construya las
features en los 5.605 nodos, puntúe los tres modelos sobre las 498.530 celdas y
escriba un mapa y un JSON. Un solo comando, en el README principal.

**Coste:** media jornada. **Decisión de equipo:** ¿qué día de julio se elige
como ejemplo? La segunda quincena es lo razonable, porque las ventanas móviles
de 30 días arrancan incompletas.

## 3. El repositorio no enseña ni un resultado

Hay cincuenta números en tablas de texto y cuatro figuras del EDA. **No está el
mapa de seis paneles, ni la figura del veredicto, ni el mapa de diferencia
contra producción** — que es la pieza que mejor explica el trabajo entero. Un
tribunal casi nunca ejecuta: mira.

**Qué hacer:** embeber las tres figuras clave en el README principal y en los
capítulos 6 y 7, y una carpeta `ejemplos/` con la salida de la demo ya
commiteada para quien no ejecute nada.

**Coste:** 1 hora.

## 4. GitHub Actions: sí, pero no la cadena diaria

Conviene separar dos cosas que se confunden:

**La cadena diaria no se mueve todavía.** Sigue en `tfm-fuego-malla` hasta el
cierre de la temporada. Moverla a mitad de veredicto arriesga días de serie por
nada. Aquí va como copia documental con el `cron` desactivado.

**Pero sí debería haber un workflow propio y distinto:** un test que en cada
push monte el entorno en una máquina limpia y ejecute la demo sobre `muestras/`.
Eso da un **badge verde en el README**, y ese badge es la prueba más persuasiva
que existe de que el repo funciona: no lo dice el autor, lo dice una máquina que
parte de cero. De paso protege contra romper un import sin enterarse.

**Coste:** 2 horas contando la pelea habitual con las dependencias geoespaciales
en el runner. **Decisión de equipo:** si el CI tarda demasiado en instalar
geopandas/rasterio, ¿se recorta la demo a la parte que solo necesita numpy y
xgboost?

## 5. La cadena en producción es un resultado, y no se está enseñando

Que el trabajo tenga un sistema corriendo en producción desde julio, con tres
jueces independientes acumulando veredicto y sin depender de que un portátil
esté encendido, es infrecuente en un TFM y vale más que un AUC bonito. En el
repositorio se cuenta; no se **enseña**.

**Qué hacer:** enlazar corridas reales de Actions, el Release `estado` y la
serie de mapas publicados, con fechas. Y decidir qué se hace en septiembre,
cuando la temporada cierre y la cadena pueda mudarse aquí.

## 6. Trámite que se nota cuando no está

- **LICENSE** — no hay ninguna.
- **Autoría**: nombre, titulación, año, tutor.
- **Cómo citar las fuentes de terceros**: ERA5-Land y EFFIS son de Copernicus
  y piden atribución; IberFire, EGIF (Civio) y los partes del MITECO tienen sus
  propias condiciones.
- La tabla del README principal promete un `docs/DECISIONES.md` que **no
  existe**: o se escribe, o se quita la promesa. Buena parte de su contenido ya
  está repartido entre [`ESTRUCTURA.md`](ESTRUCTURA.md) y los README de
  capítulo.

**Coste:** media hora.

---

## Para hablarlo mañana

1. **¿Público o privado al entregar?** Hoy es privado. Si va a ser público hay
   que revisar antes que no viaje ninguna clave (CDSAPI, FIRMS, AEMET) ni ruta
   personal, y confirmar que las licencias de los datos permiten redistribuir
   la muestra de 48 MB.
2. **¿Quién ejecuta qué?** La demo y el CI se pueden repartir; el `environment.yml`
   lo tiene que generar quien tenga el entorno bueno.
3. **¿Se funde el capítulo 2 con el 1?** El EDA quedó con dos partes bien
   distintas (el cubo y el dataset) y aguanta como capítulo propio, pero es una
   decisión de la memoria, no del repo.
4. **Septiembre**: qué se hace con la cadena diaria al cerrar la temporada, y
   si entonces se aplica el arreglo del viento (`np.minimum` → `np.fmin`) que
   está sellado a propósito — ver [`LIMITACIONES.md`](LIMITACIONES.md) §1.

---

# Actualización del 01/09/2026 — el archivo de entradas

## Lo que se cerró hoy

- **`dos_20`…`dos_24` entran al repositorio** (`06_comparacion` los cuatro de
  evaluación, `05_iteracion2/57_ablaciones` la auditoría de fugas), copia
  literal con md5. Cierra el pendiente que tenía `TRAZABILIDAD.md` y la marca
  roja de `seccion_modelo.tex`. La memoria ya no cita ninguna cifra cuyo código
  se quede fuera.
- **La sección de modelado gana el análisis exploratorio y la ingeniería de
  variables** (§1.2 y §1.3, 7 páginas) con dos figuras nuevas generadas por
  código, `f7_prevalencia` y `f8_normalizacion`.
- **La cadena archiva sus entradas irrepetibles**: el artefacto diario se lleva
  ahora `ifs_malla_<fecha>.parquet` y el NRT de FIRMS, ~270 KB/día.
- **`requirements_gh.txt` pinchado** en `TFM_fuego_malla` con las versiones de
  la corrida que produjo los veredictos publicados.
- **Hueco 08-ago → 02-sep recuperado** de la API histórica de Open-Meteo, en
  `archivo_ifs/`, fuera de los repositorios.

## Lo que queda, por orden de lo que más cambia la impresión

### 1. El replay nunca se ha ejecutado de punta a punta

`dos_riesgo_hoy.py --pasada <fecha>` está escrito y el archivo ya existe, pero
**no se ha corrido ni una sola vez con estos ficheros**. Hasta que no se haga,
que el archivo «sirve para evaluar otros modelos» es una expectativa razonada,
no un hecho comprobado. Falta cortar la serie continua en ficheros por pasada
—`[F-7, F+1]`, mismas columnas—, colocarlos donde `ruta_cache` los busca y
reproducir un día del que ya se conozca el mapa servido, comparándolo con el
`.npz` guardado. Si el mapa reproducido no se parece al servido, el archivo no
vale para lo que se pretende y hay que saberlo ahora.

**Coste:** una tarde. **Bloquea:** todo lo demás de esta lista.

### 2. El archivo vive en un artefacto de 60 días, no en un sitio permanente

El respaldo diario caduca. Para que la temporada 2026 siga siendo replayable
dentro de un año hay que llevar `ifs_malla_*.parquet` y `_firms/` a la lista
`DIARIO` de `gh_estado.py`, que los subiría al Release. Tiene un coste
explícito: `gh_estado.py` está sellado por md5 en `PROCEDENCIA.md` y habría que
reindexarlo. La alternativa es bajar el artefacto a mano cada pocas semanas,
que funciona hasta que alguien se olvida.

**Decisión pendiente**, no tarea.

### 3. Se puede archivar la temporada entera, y probablemente convenga

Hoy el archivo cubre 08-ago → 02-sep. La API histórica permite bajar hacia
atrás sin más límite que la cuota: 5.605 nodos × 1 unidad por tramo de 14 días,
contra 10.000 al día. La temporada completa desde el 25 de mayo (D-7 del 1 de
junio) son ocho tramos, unas 45.000 unidades: **cuatro o cinco noches** con el
descargador reanudable, que cede el turno a la cadena viva. En disco no llega a
3 MB. Es el complemento natural de `mapas_2026` del USB, que tiene 74 días de
mapas ya puntuados de junio y julio pero ninguna de sus entradas.

Con eso, cualquier modelo futuro podría medirse sobre la temporada completa en
condiciones de servicio, y no solo sobre las tres últimas semanas.

### 4. Los modelos evaluados en sombra necesitan su propia tabla

`historico_veredictos.csv` y `veredicto_estaciones.csv` solo admiten días con
mapa operativo, por la regla de no contaminar el juez con retro. Un modelo
reevaluado sobre el archivo **no puede escribir ahí**. Hace falta un fichero
aparte y que la memoria diga con claridad cuál es cuál: el archivo explora, la
cadena en vivo decide.

### 5. Correcciones menores ya identificadas

- `TRAZABILIDAD.md` habla de «cuatro días de mapa perdidos, 28-31/08». En el
  Release el salto va del **26-ago al 1-sep**: son **cinco**, el 27 también.
- Sigue abierto reejecutar `dos_18_ratio` sobre 2026 (el JSON es del 23/08,
  anterior a la corrección de la fuga).
- El punto 1 de esta lista —`environment.yml` en este repositorio— sigue
  abierto. Lo pinchado hoy es el entorno de la cadena en `TFM_fuego_malla`, que
  es otro fichero y otro repositorio.
- El cron de las 03:13 lleva días saliendo con 5-6 h de retraso; el 01/09 se
  lanzó la corrida a mano y luego entró la programada, duplicando la cadena
  entera (~75 min de cuota y dos pasadas de Open-Meteo el mismo día).

---

# Plan del 02/09/2026 — congelar las cifras (y por qué)

## El diagnóstico

La sensación de que «los números cambian todo el rato» tiene causas concretas y
enumerables, no es deriva:

1. **La fuga de FIRMS** (corregida el 31/08) relanzó toda la evaluación
   retrospectiva de golpe. Evento único e irreversible: las cifras limpias son
   las finales, y `dos_24_auditoria_fugas.py` vigila que no haya otra.
2. **Dos corridas paralelas conviven sin oficial declarada**: ventana FIRMS de
   5 días (prod 0,644 · único 0,751 · 1:10 0,758) y de 7 (0,647 · 0,752 ·
   0,759). No es un número que cambia: son dos experimentos sin bautizar.
3. **Documentos con cifras muertas sin purgar** citándose unos a otros.
4. Lo que cambia **porque debe**: los tres jueces en vivo, hasta mediados de
   septiembre.

Lo demás está congelado y tiene fuente de verdad:

| Bloque | Fuente de verdad | Estado |
|---|---|---|
| Iteración 1 (train/val/test) | `T/dataset/metricas_v1.json`, `metricas_v4.json` | congelado desde julio/agosto |
| Iteración 2 (test 2024) | `M/salida/dos_13_metricas.json` | congelado |
| Retro 2026 (74 días, 1-jun→15-ago) | `dos_09_temporada2026.json` + `dos_19_veredicto.json` (31/08) | congelado post-fuga; la ventana no crece |
| Modelo servido | `.ubj` con md5 + `33_train/verificar_v2.py` | congelado |
| Jueces en vivo | `historico_veredictos.csv` y compañía | el ÚNICO que se mueve, por diseño |

Comprobado el 01/09: el `seccion_modelo.tex` clava contra los artefactos casi
al tercer decimal (ICs de `dos_19`, escalera del ratio de `dos_09_ventana7.log`,
los tres concurrentes de `dos_24`). Los números no bailan; los documentos
llevan retraso unos respecto a otros.

## Las tres tareas del día

> **Reproducción verificada el 02/09/2026**: v1, v4, dos_05, dos_13 y dos_18
> reentrenados en copia aislada, cero discrepancias y `.ubj` idénticos;
> `xgb_v2_prototipo` reconstruido árbol a árbol (`reconstruir_v2.py`);
> `environment.yml` añadido desde el conda `tfm_fuego`. Ver TRAZABILIDAD.md.

> **Hecho el 02/09/2026** (las tres, más los cambios del `.tex` de abajo salvo
> la decisión sobre el 21-ago). El ruido del sorteo quedó anclado en **0,004**
> (`dos_18_ratio.json` 2024 y `dos_09_ventana7.log` 2026); el 76,2 de producción
> por hectáreas es `pctl_ponderado_ha` de `dos_09_temporada2026.json`.

1. **Declarar la ventana 7 como juego oficial** (es la que leen `dos_19`, el
   README y `TRAZABILIDAD.md`). La de 5 pasa a ser explícitamente «la ablación
   del desajuste train/serve» y nunca una cifra citada suelta.
2. **Pasada de purga**: buscar toda cifra con fuga o de ventana 5 en los `.md`
   y corregirla o marcarla «(obsoleta, ver X)». Puntos ya localizados:
   - `docs/BITACORA.md` §6 y `docs/LIMITACIONES.md` §4: citan el
     «81,7 → 87,6-87,9» por hectáreas (con fuga). Limpio: 79,8 (único 1:3) →
     81,9 (1:10), de `dos_09_ventana7.log:480-483`.
   - `docs/CONTEXTO_SECCION_MODELO.md`: usa el juego de ventana 5 entero.
   - `docs/TRAZABILIDAD.md`: «cuatro días de mapa perdidos 28-31/08» → son
     **cinco** (27-31/08; en el Release el salto va del 26-ago al 1-sep).
3. **Sellar por escrito**: una línea al principio de `TRAZABILIDAD.md` —
   «cifras selladas el 31/08/2026 tras la corrección de la fuga; solo la
   sección de jueces en vivo se actualiza hasta el cierre de septiembre». A
   partir de ahí, un número que no coincida con esa tabla es un error del
   documento, no una duda sobre el resultado.

## Cambios pendientes en `docs/MEMORIA/seccion_modelo.tex` (revisión del 01/09)

- **Error sustantivo (L211-213)**: «Obtiene 0,8906 … y con ese número se puso
  en producción». Lo servido es `xgb_v2_prototipo` (v1 sin las 4
  autorregresivas contaminadas, AUC-PR 0,828 vs 0,843, 15-jul), NO el v4 del
  0,8906, que se midió en agosto y nunca se desplegó. La historia verdadera
  refuerza el hilo: es el primer aviso de que el muestreo se colaba en el
  modelo (`03_iteracion1/MODELO_B_BITACORA.md` §16 y `verificar_v2.py`).
- **Matiz en fig. 1 y tabla T1**: «las mismas 46» vale para el servido; la
  iteración 1 entrenó con 50. Nota al pie citando `verificar_v2.py`.
- Erratas: L201 «se puntuan» → «puntúan»; L401 «se público» → «se publicó».
- Faltan de §5 (esqueleto del plan): el bug sellado del viento (16,3 % de
  estaciones-día) y los cinco días de mapa perdidos (+ decidir si se cuenta el
  21-ago regenerado, ver abajo).
- Dos cifras por anclar a su línea de log: el 76,2 de producción en percentil
  por hectáreas y el 0,004 de ruido del sorteo (otra fuente decía 0,005).

## Replay de las temporadas 2025 y 2026 (plan del 02/09/2026, pendiente de datos)

**Objetivo.** Los modelos que los jueces en vivo señalen como mejores a
mediados de septiembre se evalúan sobre la temporada 2025 entera y sobre la
2026 completa desde el 25 de mayo, en condiciones de servicio: reanálisis
hasta D−7, previsión IFS para D−6…D+1, focos FIRMS anteriores a D, verdad
EFFIS. Es «como haber tenido la cadena corriendo el año pasado», con tres
diferencias que hay que escribir: (1) el archivo de Open-Meteo guarda cada
día con su previsión de menor antelación, no la pasada emitida esa mañana
(medido el 21-ago: Spearman 0,96, solape 65-74 % en el top-2 %); (2) el replay
no reproduce los fallos operativos (días perdidos); (3) solo hay juez EFFIS (y
MITECO si se reconstruye de los partes históricos); no hay juez por estación.

**Cómo se lee.** Jueces en vivo 2026 = la prueba sellada, la que decide.
Replay 2025 = confirmación fuera de muestra (ningún modelo entrenó con 2025 ni
2026; nadie ha mirado 2025). Replay 2026 completo = cobertura de toda la
temporada con la misma metodología, NO una prueba independiente (los jueces
que eligieron puntúan esos mismos días). Elegir con 2026 en vivo, confirmar
con 2025; no elegir sobre 2025. Replay se compara con replay, nunca con la
serie servida.

**Dos condiciones por modelo.** Con IFS (servicio) y con ERA5-Land del propio
día (cota superior, como `dos_09`); la resta mide el coste de la previsión.
Etiquetas: producción es EGIF; únicos 1:3 y 1:10 son EFFIS; la pareja es
mixta (dónde EFFIS × cuándo EGIF). Con juez EFFIS los únicos juegan en casa.

**Datos, todos fuera de los repos en `~/Desktop/Master/archivo_ifs/`:**

| Pieza | 2025 | 2026 |
|---|---|---|
| IFS archivado (`ifs_archivo_2025.parquet`) | descargando, ~1 día | 08-ago→02-sep hecho; 25-may→07-ago repartido: repo `ifs-descarga-remota` (otro PC / Actions / portátil) |
| ERA5-Land horario (`era5land_cds/`) | may-nov 2025 desde CDS, en curso | may-jul en `TFM_fuego/malla_data/_cds`; ago-sep al cierre |
| FIRMS (`firms_2025/`, `firms_2026/`) | **hecho**: 46.356 focos VIIRS SP, año completo | **hecho**: 19.664 focos VIIRS NRT, 01-may→02-sep |
| EFFIS (`effis_ba_2025_ES.geojson`) | **hecho**, 1.359 perímetros | GeoJSON de temporada de la cadena |

**Trabajo que queda.** Adaptar `archivo_ifs/verificar_replay.py` (los tres
parches: truncar reanálisis a D−7, `firms_dia` con los parquets de archivo en
vez de la API, respaldar/restaurar salidas) para leer ERA5-Land de
`era5land_cds/` y FIRMS de los parquets; medir un día en local (en Actions son
~40 min/día) antes de lanzar ~160 días × 2 temporadas × 2 condiciones; MITECO
2025 desde los partes históricos de TFM-RAG si se quiere segundo juez.
Salida: un `replay_<temporada>_<condicion>.csv` por corrida + veredicto con
bootstrap por días como `dos_19`.

## En paralelo: las descargas de 2025 y el replay

- **EFFIS 2025: hecho** (01/09). `~/Desktop/Master/archivo_ifs/
  effis_ba_2025_ES.geojson`, 1.359 perímetros ES del año completo, 65 ≥500 ha.
  Capa anual `ms:modis.ba.poly.2025` del WFS, misma consulta probada del repo.
- **IFS verano 2025: descargando.** `archivo_ifs/bajar_ifs_2025.py`
  (25-may→01-nov 2025, 5.605 nodos × 12 tramos ≈ 67.260 unidades) →
  `ifs_archivo_2025.parquet`. Al cierre del 01/09 iba por el 18 % y esperaba
  cuota (429); reanuda solo tras medianoche UTC y cede el paso a la cadena.
  Revisar: `tail archivo_ifs/bajar_2025.log` y `pgrep -af bajar_ifs_2025`;
  si murió, relanzar (es reanudable por nodo). Estimación: 2-3 días.
- **El replay quedó verificado el 01/09** (`archivo_ifs/verificar_replay.py`,
  resultados en `archivo_ifs/replay_21ago/`): corre de punta a punta con tres
  parches (reanálisis truncado a D−7, `firms_nrt` sustituido por `firms_dia`
  —ignora `--pasada` y pediría FIRMS de hoy—, respaldo/restauración de
  salidas). La estática clava 1,0000; las dinámicas ~0,96 de Spearman pero
  **solo 65-74 % de solape en el top-2 %** entre dos replays con estados de
  entrada ligeramente distintos: un modelo evaluado en replay se compara con
  otro replay, nunca se mezcla con la serie servida.
- **Hallazgo pendiente de decisión**: el mapa del 21-ago del Release NO es el
  operativo — es una regeneración del 31/08 (única con `prob_r10`; la huella
  es `_firms/nrt_2026-08-31.csv`) que el `push --base` del incidente subió.
  Los días 20 y 22-26 sí son originales. Como `puntuar_effis` recalcula con
  todos los días, el juez puntuará ese mapa retro cuando lleguen sus
  perímetros: excluir el día, recuperar el original del USB Expansion
  (congelado el 21/08), o documentarlo.
