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
