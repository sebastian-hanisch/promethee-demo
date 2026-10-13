"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import pmt_constants as C
from pmt_evaluation import Analysis, de

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "↔": "<->", "–": "-", "—": "-", "τ": "tau", "σ": "sigma", "δ": "delta", "θ": "theta", "Φ": "Phi", "⁺": "+", "⁻": "-", "≥": ">=",
            "≤": "<=", "−": "-", "€": "EUR", "·": "*"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    n = s.n
    names = [f"A{i + 1}" for i in range(n)]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("PROMETHEE: Nettofluss statt Zyklen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    params = ", ".join(f"{k} = {de(getattr(s, k), 2)}" for k in C.KIND_PARAMS[s.kind]) or "ohne Schwelle"
    pdf.multi_cell(0, 5, _s(f"Alternativen n = {n}, Kriterien m = {s.m}, Praeferenzfunktion: {C.KIND_SHORT[s.kind]} ({params}), Entscheider: {C.DECIDER_LABELS[s.decider]}, "
                            f"Streuung des ersten Kriteriums delta = {de(s.delta, 2)}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    r2, rs = a.results["promethee"], a.results["summe"]
    pdf.multi_cell(0, 5, _s(f"PROMETHEE II: Platz 1 {names[r2.top1]} ({'getroffen' if r2.hit else 'nicht getroffen'}), tau {de(r2.tau, 2)}. Summe: Platz 1 {names[rs.top1]} "
                            f"({'getroffen' if rs.hit else 'nicht getroffen'}), tau {de(rs.tau, 2)}. Bester nach Nutzen: {names[a.best]}. "
                            f"Unvergleichbare Paare in PROMETHEE I: {int(a.R.sum() / 2)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Fluesse je Alternative (nach Nettofluss geordnet)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for rank, i in enumerate(a.order, start=1):
        pdf.cell(0, 5, _s(f"{rank}. {names[i]}: Phi+ {de(a.plus[i], 3)}, Phi- {de(a.minus[i], 3)}, Phi {de(a.phi[i], 3)}; Summe {de(rs.score[i], 3)}"),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der Entscheider ist ein simuliertes Orakel, kein Mensch. Quelle: promethee-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
