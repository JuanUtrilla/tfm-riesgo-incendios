# La memoria

La sección de aprendizaje automático de la memoria conjunta del TFM: cuatro
páginas de memoria y dos de anexos, que remiten a este repositorio para el
detalle.

| Fichero | Qué es |
|---|---|
| [`memoria_ML.pdf`](memoria_ML.pdf) | La sección compilada |
| `memoria_ML.tex` | Su fuente LaTeX. Se compila con `pdflatex memoria_ML.tex` dos veces desde esta carpeta |
| `figs/` | Las figuras de la memoria y de los README, con los scripts que las generan en `figs/scripts/` y en `07_produccion/escala_y_cifra/` |

Figuras de la memoria: `pagina_2_imagen_1_Im0.jpg` (diseños de muestreo, de
`f2_disenios.py`), `pagina_3_imagen_1_Im1.jpg` (SHAP del r10, de
`f11_shap_r10.py`), `fig3_dos_mapas.png` (los dos mapas, de
`07_produccion/escala_y_cifra/fig3_dos_mapas.py`), `anexo_a_prevalencia.png` y
`anexo_b_replay.png` (anexos A y B).

Ninguna cifra de la memoria se escribió a mano: cada número sale de un CSV o
JSON que produjo un script del repositorio, y
[`../TRAZABILIDAD.md`](../TRAZABILIDAD.md) dice cuál.
