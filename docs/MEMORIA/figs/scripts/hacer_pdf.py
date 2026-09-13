#!/usr/bin/env python3
"""Convierte seccion_ML_divulgativa.md en PDF (markdown + weasyprint; no hay pandoc).
Las «Notas para el autor», separadas por una doble raya, van en pagina aparte."""
import pathlib
import markdown
from weasyprint import HTML

D = pathlib.Path(__file__).resolve().parents[2]
src = (D / "seccion_ML_divulgativa.md").read_text(encoding="utf8")
body, notas = src.split("\n---\n---\n")
md = lambda t: markdown.markdown(t, extensions=["tables", "sane_lists"])
css = """
@page { size: A4; margin: 1.8cm 2cm; @bottom-center { content: counter(page); font: 8pt serif; color:#666 } }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10pt; line-height: 1.28; color:#111 }
h1 { font-size: 15pt; margin: 0 0 .2em } h2 { font-size: 11.5pt; margin: .9em 0 .3em; border-bottom: 1px solid #bbb; padding-bottom: 2px }
p { margin: 0 0 .5em; text-align: justify }
blockquote { margin: .5em 1em; padding-left: .7em; border-left: 3px solid #999; color:#333; font-style: italic }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: .3em 0 .8em }
th, td { border: 1px solid #999; padding: 2px 5px; vertical-align: top } th { background: #eee }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt }
img { display:block; max-width: 100%; margin: .6em auto .2em }
p:has(> img) + p { font-size: 8.5pt; color:#333; text-align: justify; margin: 0 .6em .9em }
.notas { page-break-before: always; font-size: 8.5pt; color:#333 } .notas h2 { border: 0 }
"""
ancho = {"f10_percentil_vs_absoluto.png": "62%", "f2_disenios.png": "64%", "f11_shap_r10.png": "74%"}
cuerpo = md(body)
for f, w in ancho.items():
    cuerpo = cuerpo.replace(f'src="figs/{f}"', f'src="figs/{f}" style="width:{w}"')
html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{cuerpo}<div class='notas'>{md(notas)}</div></body></html>"
HTML(string=html, base_url=str(D)).write_pdf(D / "seccion_ML_divulgativa.pdf")
print("escrito", D / "seccion_ML_divulgativa.pdf")
