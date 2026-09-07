# Los documentos de la memoria, y cuál es cuál

Hay varios borradores porque el formato de la memoria cambió por el camino. El
que se entrega es el primero; el resto es cantera para los anexos.

| Documento | Qué es | Estado |
|---|---|---|
| **`seccion_ML_divulgativa.md`** | La sección de ML de la memoria **conjunta** (RAG + visión + ML, 20 páginas): 5 páginas, divulgativa, sin fundamentos. Titular = replay 2025-2026 | Borrador del 07/09/2026, **el vigente** |
| `seccion_modelo.tex` / `.pdf` | Versión técnica de la sección (9 páginas, LaTeX autocontenido, figuras f1-f8). Superada por el formato divulgativo | Cantera para los anexos B y D |
| `TRAZABILIDAD.md` | Cada cifra de `seccion_modelo.tex` con su artefacto. Complementa `../TRAZABILIDAD.md`, que es la del repositorio entero | Vigente para lo que se recicle del .tex |
| `BORRADOR_memoria_ML.md` / `.tex` / `.pdf` | Borrador largo por capítulos (10-11 páginas) con el presupuesto de páginas antiguo | Superado |
| `capitulo_ML_donde_cuando.md` | Primer capítulo autocontenido (21/08), anterior a la fuga de FIRMS: **sus cifras están desactualizadas** | Histórico |
| `hilo_conductor.md` | Guía de redacción: hilo lógico y no cronológico, la iteración 1 como grupo de control, el texto de la bisagra | Vigente como guía |
| `PLAN_SECCION.md` | Inventario de evidencia del 31/08 recorriendo los cuatro repos | Histórico |
| `figs/` | Las figuras y los scripts que las generan (`figs/scripts/`) | f1-f8 vigentes; falta la del invierno (percentil vs. absoluto) |

Regla de oro, heredada de todas las versiones: **ninguna cifra se escribe de
memoria**. Cada número sale de un CSV/JSON que produjo un script del repo, y
`../TRAZABILIDAD.md` dice cuál.
