"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pmt_constants as C
import pmt_promethee as pm
from pmt_evaluation import Analysis, de

ARROW = {1: "↑", -1: "↓"}
BORDA_LABEL = "Gewichtete Borda-Zählung"


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def criterion_label(a: Analysis, k: int) -> str:
    inst = a.case.matrix
    return f"K{k + 1} {ARROW[int(inst.directions[k])]} {inst.units[k]}"


def build_preference_curve(kind: str, q: float, p: float, s: float) -> go.Figure:
    """Die Präferenzfunktion P(d) über der Differenz d in Anteilen der Spannweite."""
    d = np.linspace(-0.25, 1.0, 400)
    fig = go.Figure(go.Scatter(x=d, y=pm.preference(kind, d, q, p, s), mode="lines", line=dict(color=C.COLOR_FLOW, width=3), name=C.KIND_SHORT[kind]))
    fig.update_xaxes(title="Differenz d (Anteil der Spannweite)")
    fig.update_yaxes(title="Präferenz P(d)", range=[-0.05, 1.05])
    fig.update_layout(title=dict(text=f"Präferenzfunktion: {C.KIND_SHORT[kind]}", font=dict(size=13)), showlegend=False)
    return _fixed(fig, 260)


def build_flow_plane(a: Analysis) -> go.Figure:
    """Plusfluss gegen Minusfluss: a übertrifft b in PROMETHEE I, wenn a rechts (mehr Plusfluss) und unterhalb (weniger Minusfluss) von b liegt; die Diagonale ist der Nettofluss."""
    names = alt_names(a.settings.n)
    fig = go.Figure()
    lo = float(min(a.plus.min(), a.minus.min()))
    hi = float(max(a.plus.max(), a.minus.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="rgba(150,150,150,0.6)", dash="dot"), name="Nettofluss 0", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.plus, y=a.minus, mode="markers+text", text=names, textposition="top center", textfont=dict(size=9),
                             marker=dict(size=10, color=[C.COLOR_BEST if i == a.best else C.COLOR_FLOW for i in range(a.settings.n)]), name="Alternativen",
                             hovertemplate="%{text}: Φ⁺ %{x:.3f}, Φ⁻ %{y:.3f}<extra></extra>"))
    pad = 0.05 * (hi - lo + 1e-9)
    fig.update_xaxes(title="Plusfluss Φ⁺ (mehr ist besser)", range=[lo - pad, hi + pad])
    fig.update_yaxes(title="Minusfluss Φ⁻ (weniger ist besser)", range=[hi + pad, lo - pad])
    fig.update_layout(title=dict(text="Flüsse (orange: bester nach Nutzen)", font=dict(size=13)))
    return _fixed(fig, 380)


def build_net_flow(a: Analysis) -> go.Figure:
    names = alt_names(a.settings.n)
    order = a.order
    colors = [C.COLOR_BEST if i == a.best else C.COLOR_FLOW for i in order]
    fig = go.Figure(go.Bar(x=[names[i] for i in order], y=a.phi[order], marker_color=colors))
    fig.update_yaxes(title="Nettofluss Φ")
    fig.update_layout(title=dict(text="PROMETHEE II: Nettofluss", font=dict(size=13)))
    return _fixed(fig, 320)


def build_relation_matrix(a: Analysis) -> go.Figure:
    """PROMETHEE I: Zeile übertrifft Spalte (→), beidseitig gleich (=), widersprüchlich unvergleichbar (leer)."""
    n = a.settings.n
    names = alt_names(n)
    z = np.where(a.P, 2, np.where(a.I, 1, 0))
    text = [["" if i == j else ("→" if a.P[i, j] else "=" if a.I[i, j] else "?" if a.R[i, j] else "") for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(z=z, x=names, y=names, text=text, texttemplate="%{text}", zmin=0, zmax=2, showscale=False,
                               colorscale=[[0, "rgba(150,150,150,0.15)"], [0.5, "#e0b040"], [1, C.COLOR_FLOW]], hovertemplate="%{y} gegenüber %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text="PROMETHEE I: Zeile übertrifft Spalte", font=dict(size=13)))
    return _fixed(fig, 360)


def build_table(a: Analysis) -> pd.DataFrame:
    n = a.settings.n
    names = alt_names(n)
    rank = lambda v: np.argsort(np.argsort(-np.asarray(v), kind="stable"), kind="stable") + 1  # noqa: E731
    df = pd.DataFrame({
        "Alternative": names, "Φ⁺": a.plus, "Φ⁻": a.minus, "Nettofluss Φ": a.phi, "Rang (PROMETHEE II)": rank(a.phi),
        "übertroffen von (PROMETHEE I)": [", ".join(names[j] for j in np.flatnonzero(a.P[:, i])) or "-" for i in range(n)],
        "unvergleichbar mit": [", ".join(names[j] for j in np.flatnonzero(a.R[i])) or "-" for i in range(n)],
        "Rang Summe": rank(a.results["summe"].score), "Rang Entscheider": rank(a.case.utility), "nichtdominiert": np.where(a.front, "ja", "nein"),
    })
    if a.settings.decider == "untergrenze":
        df["zulässig"] = np.where(a.case.acceptable, "ja", "nein")
    return df


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 3)) for c in ("Φ⁺", "Φ⁻", "Nettofluss Φ")})


def pair_details(a: Analysis, i: int, j: int) -> pd.DataFrame:
    """Je Kriterium: Werte (Min-Max), Differenz, Präferenz P(d) von i über j und von j über i, Gewicht."""
    s = a.settings
    names = alt_names(s.n)
    d = a.N[i] - a.N[j]
    return pd.DataFrame({
        "Kriterium": [f"K{k + 1}" for k in range(s.m)], "Gewicht": a.case.w, f"{names[i]} (Min-Max)": a.N[i], f"{names[j]} (Min-Max)": a.N[j], "Differenz d": d,
        f"P({names[i]} über {names[j]})": pm.preference(s.kind, d, s.q, s.p, s.s), f"P({names[j]} über {names[i]})": pm.preference(s.kind, -d, s.q, s.p, s.s),
    })


def style_pair(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 3)) for c in df.columns if c != "Kriterium"})


def build_method_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.settings.n)
    labels = {"summe": "Gewichtete Summe (Min-Max)", "promethee": f"PROMETHEE II ({C.KIND_SHORT[a.settings.kind]})", "borda": BORDA_LABEL}
    rows = []
    for k, label in labels.items():
        r = a.results[k]
        rows.append({"Verfahren": label, "Platz 1": names[r.top1] + (" (Gleichstand)" if r.tie_at_top else ""), "Trifft den Besten": "ja" if r.hit else "nein",
                     "τ gegen Nutzen": de(r.tau, 2), "Nutzenverlust": de(r.loss, 3)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Experimente

def identity_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Kriterien m": r["m"], "gewöhnlich = Borda": f"{de(100 * r['borda'], 1)} %", "gewöhnlich = Summe der Perzentil-Ränge": f"{de(100 * r['percentile'], 1)} %",
                          "V-Form (p = Spannweite) = Summe": f"{de(100 * r['vsum'], 1)} %", "Gegenprobe V-Form p = 0,5 = Summe": f"{de(100 * r['vsum_p05'], 1)} %",
                          "größte Abweichung Φ gegen Borda": f"{r['diff_borda']:.1e}".replace(".", ","), "größte Abweichung Φ gegen Summe": f"{r['diff_vsum']:.1e}".replace(".", ",")}
                         for r in rows])


def agreement_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Funktion A": r["a"], "Funktion B": r["b"], "gleicher Platz 1": f"{de(100 * r['same_top1'], 1)} %",
                          "mittleres τ": "-" if r["tau"] != r["tau"] else de(r["tau"], 2)} for r in rows])


def build_dial(rows: list[dict], m: int = 5) -> go.Figure:
    """Platz 1 getroffen je Funktion, nach Entscheider."""
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig = go.Figure()
    for dec in dict.fromkeys(r["decider"] for r in rows):
        rs = {r["method"]: r for r in rows if r["decider"] == dec and r["m"] == m}
        fig.add_trace(go.Bar(x=["Summe" if k == "summe" else k for k in methods], y=[100 * rs[k]["hit"] for k in methods], name=C.DECIDER_SHORT[dec]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    fig.update_xaxes(tickangle=-25)
    return _fixed(fig, 420)


def dial_table(rows: list[dict]) -> pd.DataFrame:
    methods = list(dict.fromkeys(r["method"] for r in rows))
    cols = list(dict.fromkeys((r["decider"], r["m"]) for r in rows))
    out = []
    for k in methods:
        row = {"Funktion": "Summe (Min-Max)" if k == "summe" else k}
        for dec, m in cols:
            r = next(x for x in rows if x["decider"] == dec and x["m"] == m and x["method"] == k)
            row[f"{C.DECIDER_SHORT[dec]}, m = {m}"] = f"{de(100 * r['hit'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)


def partial_table(rows: list[dict]) -> pd.DataFrame:
    nan = lambda v: "-" if v != v else f"{de(100 * v, 1)} %"  # noqa: E731
    return pd.DataFrame([{"Funktion": r["method"], "strikt": nan(r["p"]), "unvergleichbar": nan(r["r"]), "Zyklus": nan(r["cycle"]), "Sieger von II übertroffen": nan(r["winner_outranked"]),
                          "Bester nicht übertroffen": nan(r["best_free"])} for r in rows])


def build_partial(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r["method"] for r in rows], y=[100 * r["r"] for r in rows], name="unvergleichbare Paare", marker_color=C.COLOR_LINE))
    fig.add_trace(go.Bar(x=[r["method"] for r in rows], y=[100 * r["cycle"] for r in rows], name="Matrizen mit Zyklus", marker_color=C.COLOR_WARN))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Anteil (%)", range=[0, 102])
    fig.update_xaxes(tickangle=-25)
    return _fixed(fig, 420)


def addition_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Verfahren": r["method"], "gekippte alte Paare": f"{de(100 * r['flipped_pairs'], 2)} %", "Matrizen mit mindestens einem gekippten Paar": f"{de(100 * r['any_flip'], 1)} %",
                          "Platz 1 der alten wechselt": f"{de(100 * r['top_changes'], 1)} %"} for r in rows])


def build_addition(rows: list[dict]) -> go.Figure:
    fig = go.Figure(go.Bar(x=[r["method"] for r in rows], y=[100 * r["any_flip"] for r in rows], marker_color=C.COLOR_WARN))
    fig.update_yaxes(title="Matrizen mit mindestens einem gekippten Paar (%)", range=[0, 102])
    fig.update_xaxes(tickangle=-25)
    return _fixed(fig, 340)


def floor_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Verfahren": r["method"], "Wahl unter der Untergrenze": f"{de(100 * r['unacceptable'], 1)} %", "zulässiger Bester getroffen": f"{de(100 * r['hit'], 1)} %"} for r in rows])


def stability_table(rows: list[dict]) -> pd.DataFrame:
    methods = list(dict.fromkeys(r["method"] for r in rows))
    sig = sorted({r["sigma"] for r in rows})
    out = []
    for k in methods:
        row = {"Verfahren": k}
        for sg in sig:
            row[f"σ = {de(sg, 1)}"] = f"{de(100 * next(x for x in rows if x['sigma'] == sg and x['method'] == k)['stay'], 1)} %"
        out.append(row)
    return pd.DataFrame(out)
