# La memoria

Esta carpeta contiene la sección de aprendizaje automático de la memoria
conjunta del Trabajo Fin de Máster (TFM). Son cuatro páginas de memoria y cinco
de anexos (datos y variables, benchmark y alternativas, escala y cifra del día,
repositorio y cadena diaria, limitaciones), que remiten a este repositorio para
el detalle.

| Fichero | Qué es |
|---|---|
| [`memoria_ML.pdf`](memoria_ML.pdf) | La sección compilada |
| `memoria_ML.tex` | Su fuente LaTeX. Se compila con `pdflatex memoria_ML.tex` dos veces desde esta carpeta |
| `figs/` | Las figuras de la memoria y de los README, con los scripts que las generan en `figs/scripts/` y en `07_produccion/escala_y_cifra/` |

Las figuras de la memoria son `pagina_2_imagen_1_Im0.jpg` (diseños de
muestreo, de `f2_disenios.py`), `pagina_3_imagen_1_Im1.jpg` (valores SHAP,
*SHapley Additive exPlanations*, del r10, de `f11_shap_r10.py`),
`fig3_dos_mapas.png` (los dos mapas, de
`07_produccion/escala_y_cifra/fig3_dos_mapas.py`), `anexo_a_prevalencia.png` y
`anexo_a_normalizacion.png` (anexo A) y `anexo_b_boxplot.png` (anexo B, de
`anexo_b_boxplot.py`).

Ninguna cifra de la memoria se escribió a mano. Cada número sale de un CSV o
JSON producido por un script del repositorio, y
[`../TRAZABILIDAD.md`](../TRAZABILIDAD.md) indica cuál.
