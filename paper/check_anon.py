# -*- coding: utf-8 -*-
"""Verifica que las PDF de TMLR NO llevan identidad y que las de arXiv SI."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")
P = r"e:\ARTICULOS-CIENTIFICOS\MIURACOGNITIVE\miuracognitive\paper"

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

IDENT = ["Arrabal", "arrabal", "Fern\u00e1ndez", "Fernandez", "ual.es",
         "Almer", "CIAIMBITAL", "ifernan", "fmarrabal",
         "Montoya", "Alcayde", "pagilm", "aalcayde",
         "F.M.A.-C", "F.G.M", "A.A.:", "I.F.:"]


def texto(p, paginas=None):
    r = PdfReader(p)
    pgs = r.pages if paginas is None else r.pages[:paginas]
    return "\n".join((pg.extract_text() or "") for pg in pgs)


print("=== TMLR (doble ciego): NO debe aparecer identidad ===")
mal = False
for n in ("cognition", "governor"):
    t = texto(f"{P}\\tmlr\\tmlr-submission-{n}.pdf")
    hits = sorted({k for k in IDENT if k in t})
    anon = "Anonymous authors" in t or "double-blind" in t
    estado = "LIMPIO" if not hits else f"FUGA: {hits}"
    if hits:
        mal = True
    print(f"  {n:10s} banner_anonimo={anon}  {estado}")

print("\n=== arXiv (preprint): SI debe aparecer identidad ===")
for n, tex in (("cognition", "main"), ("governor", "governor_en")):
    t = texto(f"{P}\\arxiv\\{n}\\{tex}.pdf") if False else None
    # las pdf del staging se borran al empaquetar; usamos las del repo
    t = texto(f"{P}\\{tex}.pdf", paginas=1)
    falta = [k for k in ("Arrabal", "Montoya", "Alcayde", "Fern",
                         "CIAIMBITAL", "Engineering", "corresponding",
                         "fmarrabal", "pagilm", "aalcayde", "ifernan")
             if k not in t]
    print(f"  {n:10s} "
          + ("OK: los 4 autores, ambas afiliaciones y el corresponding"
             if not falta else "FALTA: " + str(falta)))

print("\nVEREDICTO:", "ANONIMIZACION CORRECTA" if not mal
      else "*** HAY FUGA DE IDENTIDAD, NO ENVIAR ***")
