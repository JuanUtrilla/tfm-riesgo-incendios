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
