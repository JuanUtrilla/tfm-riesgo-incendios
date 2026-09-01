# Contexto de redacción — sección de modelado

> Bloque `<variables>` relleno con la información real de este trabajo, para
> generar/reescribir la sección de modelado de la memoria. Escrito el
> 31/08/2026. Las cifras son las **limpias** posteriores a la corrección de la
> fuga de FIRMS de ese mismo día (ver `docs/BITACORA.md`).

```
<variables>
  TITULO_SECCION: "Modelado predictivo del riesgo diario de incendio forestal:
                   de la susceptibilidad al ordenamiento operativo de celdas"
                  (título corto para índice: «Modelado predictivo del riesgo
                   diario de incendio». El subtítulo importa: la sección NO es
                   un modelo de susceptibilidad estática, es un modelo que
                   ordena 498.530 celdas dentro de cada día.)

  CAPITULO_EN_EL_TFM: Bloque de aprendizaje automático, autocontenido, que
                      condensa los capítulos 4 a 7 de la memoria larga
                      (estructura de dos ciclos con bisagra descrita en
                      `docs/MEMORIA/hilo_conductor.md`):
                        · 4. Primera iteración (etiqueta EGIF, muestreo
                             caso-control, AUC 0,89 en test)
                        · 5. LA BISAGRA — por qué 0,89 no medía lo que hacía
                             falta (error de especificación)
                        · 6. Segunda iteración (etiqueta EFFIS, negativos del
                             mismo día, DÓNDE × CUÁNDO, ablaciones)
                        · 7. Comparación de los dos pipelines
                      ANTES va: cap. 2 «Datos y fuentes» (nueve fuentes,
                      IberFire como backbone, licencias) y cap. 3 «Análisis
                      exploratorio» — de ahí que aquí los datos se resuman en
                      una tabla y NO se repitan fichas de fuente.
                      DESPUÉS va: cap. 8 «Producción y validación en curso:
                      los tres jueces» (cadena diaria en GitHub Actions, jueces
                      EFFIS / MITECO / estaciones) y cap. 9 «Conclusiones y
                      limitaciones» — de ahí que aquí NO se detalle la
                      infraestructura operativa ni el veredicto de septiembre,
                      solo se anuncie.
                      NO REPETIR: fichas de fuentes, licencias, EDA extenso,
                      arquitectura de Actions, discusión ética.

  EXTENSION_OBJETIVO: 6-7 páginas a 10 pt, A4, márgenes 2,1 cm (≈ 3.500-4.000
                      palabras + 4 tablas + 3-4 figuras). Es el presupuesto ya
                      cumplido por `docs/MEMORIA/BORRADOR_memoria_ML.tex`
                      (8 secciones: Datos · EDA · Variables e ingeniería ·
                      Entrenamiento · Evaluación del primer modelo y el
                      diagnóstico · Segunda iteración · Evaluación del segundo
                      ciclo · Repositorio y conclusión).
                      Reparto orientativo: 1 p. datos+EDA, 1 p. variables,
                      1 p. entrenamiento, 1,5 p. bisagra/diagnóstico (es el
                      núcleo, no recortar), 1,5 p. segunda iteración y
                      evaluación, 0,5 p. cierre.

  FORMATO_SALIDA: LaTeX (.tex), artículo `\documentclass[10pt,a4paper]{article}`,
                  compilado con `pdflatex` (dos pasadas), paquetes ya fijados en
                  el preámbulo existente (booktabs, capt-of, microtype,
                  hyperref, geometry). Convenciones vivas del fichero:
                  `\var{}` para nombres de fichero/variable, `\pdte{}` para
                  marcas de pendiente y `\fig{}` para figuras aún no dibujadas
                  (ambas se anulan en la versión final). Tablas con `booktabs`
                  y `\captionof{table}`. El borrador largo en Markdown
                  (`BORRADOR_memoria_ML.md`, 751 líneas) es el material del que
                  se recorta; el .tex es la salida.

  IDIOMA: español académico, impersonal («se entrena», «se mide»), sin primera
          persona ni tono de diario de laboratorio. Términos técnicos en inglés
          en cursiva la primera vez (*early stopping*, *ablation*, *gain*).
          Aviso de codificación: el .tex actual está escrito SIN tildes por
          compatibilidad; mantener ese criterio o pasar el fichero entero a
          UTF-8 acentuado, pero no mezclar.

  NORMA_CITAS: autor-año estilo APA 7 (es lo asumido en el índice de la memoria,
               `TFM_fuego/MEMORIA/00_INDICE.md`), con la bibliografía común al
               final del documento, no en esta sección. Convención de fuentes de
               datos (dataset, versión, licencia, DOI/arXiv) en el apéndice de
               fuentes. Citas mínimas imprescindibles aquí: Ercibengoa et al.
               (2025) IberFire, arXiv:2505.00837, CC-BY 4.0; Van Wagner (1987)
               FWI canadiense; Chen & Guestrin (2016) XGBoost; Barbet-Massin
               et al. (2012) pseudo-ausencias; Kapoor & Narayanan (2023) fuga y
               reproducibilidad. PENDIENTE: confirmar la norma exacta que exige
               la universidad y unificar el fichero de bibliografía.

  RUTA_SALIDA: docs/MEMORIA/  (dentro de `tfm-riesgo-incendios`)
               · fuente LaTeX: docs/MEMORIA/BORRADOR_memoria_ML.tex
               · borrador largo de apoyo: docs/MEMORIA/BORRADOR_memoria_ML.md
               · versión previa en Markdown: docs/MEMORIA/capitulo_ML_donde_cuando.md
               · figuras: docs/MEMORIA/figuras/ y 02_eda/figuras/
                 (ambas ya en `\graphicspath`)

  LECTOR_OBJETIVO: tribunal de máster con formación técnica general, sin
                   conocimiento previo de este repositorio ni del dominio de
                   incendios. Hay que dar por NO sabido: qué es el FWI, qué es
                   un datacubo, qué es EFFIS/EGIF/FIRMS, y por qué el AUC dentro
                   del día no es el AUC caso-control. Se puede dar por sabido:
                   AUC-ROC/PR, gradient boosting, split temporal, validación
                   cruzada. Cada afirmación del diagnóstico debe ir con su
                   número al lado: convence porque está medido, no porque esté
                   bien argumentado.

  RESTRICCIONES_UNIVERSIDAD: PENDIENTE DE CONFIRMAR (normativa sobre uso de IA,
                   plantilla obligatoria, límite de páginas y estilo de citas).
                   Restricciones autoimpuestas que sí están fijadas y hay que
                   respetar al redactar:
                     · ninguna cifra se escribe de memoria: toda métrica sale de
                       un JSON/CSV de `salida/` y está trazada en
                       `docs/TRAZABILIDAD.md` (número → script que lo produjo);
                     · el código no se reescribe (copias literales con md5 en
                       `docs/PROCEDENCIA.md`), así que el texto no puede
                       describir un pipeline distinto del publicado;
                     · las limitaciones y los huecos conocidos van escritos, no
                       omitidos (`docs/LIMITACIONES.md`);
                     · el repositorio es privado y acompaña a la memoria: se
                       puede citar por ruta de fichero, no por URL pública.
</variables>
```

---

## Material de apoyo: lo que esta sección tiene que contar

Resumen de lo que hay en el repositorio, para no tener que releerlo entero al
redactar. Fuente de cada cifra: `docs/TRAZABILIDAD.md`.

### El hilo en una frase

Un primer modelo alcanza **AUC 0,89** en test y, servido a diario contra
superficie quemada real, cae a **0,56-0,64**. La caída no es sobreajuste ni
cambio de fuente meteorológica: es un **error de especificación** — el muestreo
y la etiqueta definían la pregunta «¿es hoy un día peligroso en esta celda?»,
distinta de la que se evalúa en operación, «¿cuál de las 498.530 celdas arde
hoy?». Se rediseña el conjunto de entrenamiento y se construye un segundo
sistema, que se compara con el primero sobre los mismos días y las mismas
fuentes.

### Marco común (viene del capítulo anterior, aquí solo se resume)

- **Unidad muestral:** (celda de 1 km, día). **498.530** celdas peninsulares.
- **Backbone:** cubo **IberFire** (NetCDF 30,3 GB, 261 variables, EPSG:3035,
  1.188×920, 2007-2024). Meteorología = ERA5-Land remuestreado a 1 km;
  vegetación Copernicus CLMS; terreno EU-DEM; usos del suelo CORINE.
- **46 variables** en ambas iteraciones — la comparación es del *diseño*, no de
  las variables.
- Dos avisos que condicionan todo: el **FWI del cubo son las 13 UTC** (servirlo
  a otra hora mete un factor ~2) y **ERA5-Land llueve de más** (+8,00 mm en 30
  días frente a AEMET), de donde sale un FWI diez puntos más bajo.

### Iteración 1 — el grupo de control

- Etiqueta **EGIF** (19.561 igniciones 2015-2020, punto y no perímetro;
  incompleto desde 2021: 888/226/23 registros — eso fija la ventana temporal).
- Muestreo **caso-control con celda fija**: pseudo-ausencias en la misma celda,
  fecha uniforme, ratio 1:3, buffer 12,5 km × ±10 días.
- Resultado: **AUC 0,8906** (test 2022, IC [0,8644, 0,9136]) para v4.
  *Ojo a no confundir:* el 0,931 es v1/v2 en test 2020; producción sirve
  `xgb_v2_prototipo`, no v3 ni v4.
- Se pone en producción sobre 687 estaciones AEMET y se mide.

### La bisagra — el diagnóstico, con números

- El **98,8 %** de las celdas del conjunto v1 tenía exactamente un 25 % de
  positivos: el modelo no podía aprender qué celda es más peligrosa que otra,
  solo qué día lo es. El muestreo fija la pregunta.
- El **AUC caso-control de los dos diseños es idéntico (0,92)** y el AUC dentro
  del día los separa (**0,744 frente a 0,828**): la métrica de desarrollo era
  ciega a la diferencia.
- La etiqueta de entrenamiento no es la de validación: solo el **15 %** de los
  incendios EGIF de 25-100 ha coincide celda a celda con un perímetro EFFIS ese
  día.
- Conclusión: no es un problema de ajuste, es de especificación.

### Iteración 2 — el rediseño

- Etiqueta **EFFIS** (`is_fire` del cubo, celda quemada ≥5 ha), la misma que
  juzga en operación. EGIF es mejor etiqueta de ignición pero se consolida con
  2-4 años de retraso: nunca podrá juzgar un sistema vivo.
- **Negativos del mismo día** que el positivo (`dos_12`), tope de 30 positivos
  por día×bloque de 100 km, primer día EFFIS de cada incendio.
- Split **train 2015-2022 / val 2023 / test 2024**, y **2026 como externo**.
- Modelo **XGBoost** (`donde_dia_effis`), 46 variables. *Gain*:
  `n_fuegos_10km_mismomes_hist` **29 %** (domina), CLC ~10 %, pendiente 5,8 %.
- Alternativa evaluada y descartada: la **pareja DÓNDE × CUÁNDO** (dos modelos
  multiplicados) frente al **modelo único** con muestreo dónde.

### Evaluación (cifras limpias del 31/08/2026)

| comparación (temporada 2026) | AUC | Δ vs producción | IC95 | días ganados |
|---|---|---|---|---|
| producción (iteración 1) | 0,644 | — | — | — |
| **único `donde_dia_effis`** | **0,751** | +0,106 | [+0,067, +0,148] | 70 % |
| **único, negativos 1:10** | **0,758** | +0,114 | [+0,074, +0,154] | 73 % |
| pareja DÓNDE × CUÁNDO | 0,702 | — | — | — |

- **Ablación del ratio de negativos** (`dos_18`): óptimo interior en 1:10-1:30;
  a partir de 1:60 empeora. La ganancia en AUC medio (+0,006 a +0,017) es del
  orden del ruido del sorteo de negativos (0,005, medido re-sorteando); lo que
  sí mejora consistentemente es el fuego grande: percentil ponderado por
  hectáreas 81,7 → 87,6-87,9.
- **Ablación FIRMS:** aporta menos de lo que se creía (+0,033, no +0,126) y en
  la punta del ranking producción *con* FIRMS es peor que sin ella (top-2 %:
  11,1 % de hectáreas frente a 19,4 %). Hipótesis viva: el radio de 50 km
  arrastra la punta hacia lo ya quemado en vez de hacia la ignición nueva.
- **Entrenar solo con verano** pierde en el DÓNDE (`dos_08`).
- **Comparación justa previsión-contra-previsión** (`comparar_rankings_justo`,
  19 días 22-jul → 15-ago): ranking sellado 0,568, único **0,605**
  (Δ +0,036, gana 14/19 días), malla 0,547, pareja 0,548; por celda único 0,755
  vs malla 0,669 (Δ +0,086 [+0,022, +0,156]). Es el único experimento sin
  retrovisor para nadie.

### Lo que hay que contar aunque incomode (va en la sección, no escondido)

- **Fuga de futuro en FIRMS, detectada y corregida el 31/08/2026.** La API de
  FIRMS cuenta los 5 días *hacia adelante* desde la fecha pedida: el mapa
  retrospectivo del día D llevaba dentro los focos del propio incendio. Los 33
  incendios de ≥500 ha tenían un foco a <5 km en su ventana «pasada» (control
  aleatorio 1,2 %); tras el arreglo, 9,1 %. **Producción nunca estuvo
  afectada** y el **entrenamiento estaba limpio** (`extraer_features_historia.py:117`
  usa `(vf_d >= d-7) & (vf_d < d)`); solo se materializó en la evaluación
  retrospectiva, que se relanzó entera. El arreglo *refuerza* la tesis: la
  ventaja del segundo modelo pasa de +0,037 (rozando el cero) a +0,106.
- **Firma del fuego en el día D.** En el cubo, la celda que arde el día D ya
  trae ese mismo día la señal del incendio: ΔNDVI (D − D−1) = −0,0312 frente a
  −0,0033 en el control; ΔLST +0,91 K frente a +0,02 K. Pesan el **5,5 % del
  gain**. No sesga la comparación entre modelos (ambos comparten receta) pero
  **infla los absolutos** de toda evaluación del mismo día, y es además un
  desajuste *train/serve*: al servir se sustituyen por una climatología mensual
  plana 2020-24.
- **Desajuste de ventana FIRMS:** se entrenó con [D−7, D−1] y la tubería de
  servicio pedía 5 días. Medido: corregirlo da +0,003 de AUC — ruido.
- **Cuatro días de mapas perdidos** (28-31/08/2026) por un incidente de
  publicación de estado; reconstruirlos exigía ~56.000 unidades de Open-Meteo
  compitiendo con la cadena viva. Decidido no hacerlo y documentar el hueco.

### Lo que NO va en esta sección (es del capítulo siguiente)

Cadena diaria en GitHub Actions (`malla_diaria.yml`, 03:30 y 13:45 UTC), los
tres jueces en operación (EFFIS a 45 días, MITECO parte D+1, y el juez por
estación contra el ranking sellado del repositorio hermano), y el veredicto
acumulado esperado a mediados de septiembre de 2026.
