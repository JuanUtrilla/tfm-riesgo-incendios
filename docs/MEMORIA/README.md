# La memoria

Esta carpeta contiene la sección de aprendizaje automático de la memoria
conjunta del Trabajo Fin de Máster (TFM): cuatro páginas de memoria y el anexo B,
de diez páginas, que remite a este repositorio para el detalle.

| Fichero | Qué es |
|---|---|
| [`memoria_ML.pdf`](memoria_ML.pdf) | La sección y su anexo, compilados |
| `memoria_ML.tex` | Su fuente LaTeX. Se compila con `pdflatex memoria_ML.tex` dos veces desde esta carpeta |
| `tabla_algoritmos.tex` | Las dos tablas de la comparación de algoritmos (anexo B.2.1), generadas con `05_iteracion2/57_ablaciones/dos_35_algoritmos.py` y `dos_35_replay.py` |
| `figs/` | Las figuras de la memoria y de los README, con los scripts que las generan en `figs/scripts/` y en `07_produccion/escala_y_cifra/` |

El anexo B tiene cinco apartados:

| Apartado | Contenido |
|---|---|
| B.1 | Datos y análisis exploratorio |
| B.2 | Modelos: por qué XGBoost, las dos iteraciones, la comparación de los seis modelos y las alternativas descartadas |
| B.3 | Escala absoluta y cifra del día |
| B.4 | Limitaciones |
| B.5 | Repositorio, trazabilidad y reproducibilidad |

Las figuras del cuerpo son `pagina_2_imagen_1_Im0.jpg` (diseños de muestreo, de
`f2_disenios.py`), `pagina_3_imagen_1_Im1.jpg` (valores SHAP, *SHapley Additive
exPlanations*, del r10, de `f11_shap_r10.py`) y `fig3_dos_mapas.png` (los dos
mapas, de `07_produccion/escala_y_cifra/fig3_dos_mapas.py`). El anexo usa además
`f17_malla_nodos.png`, `anexo_a_prevalencia.png`, `anexo_a_normalizacion.png`,
`f3_separacion.png`, `f18_escalera_it1.png`, `f4_metrica_ciega.png` y
`anexo_b_boxplot.png`, cada una con su script en `figs/scripts/`.

Ninguna cifra de la memoria se escribió a mano. Cada número sale de un CSV o
JSON producido por un script del repositorio, y
[`../TRAZABILIDAD.md`](../TRAZABILIDAD.md) indica cuál.
