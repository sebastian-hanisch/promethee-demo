"""PROMETHEE: Nettofluss statt Zyklen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Linie Mehrkriterien-Ranking (Kind von ELECTRE). Dieselbe neutrale Entscheidungsmatrix; ein simulierter Entscheider kennt seine Gewichte und seinen
Nutzen. PROMETHEE vergleicht je Paar mit einer Präferenzfunktion und ordnet nach Plus-, Minus- und Nettofluss; die Funktion ist der Regler zwischen Zählen und Summieren.
Kern: pmt_promethee.py, pmt_score.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import pmt_constants as C
from pmt_evaluation import (
    Settings,
    analyse,
    de,
    experiment_addition,
    experiment_agreement,
    experiment_dial,
    experiment_floor,
    experiment_identities,
    experiment_partial,
    experiment_stability,
    verdict_dependence,
    verdict_dial,
    verdict_identities,
    verdict_partial,
)
from pmt_pdf_export import build_pdf
from pmt_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from pmt_promethee import uses
from pmt_visualization import (
    addition_table,
    agreement_table,
    alt_names,
    build_addition,
    build_dial,
    build_flow_plane,
    build_method_table,
    build_net_flow,
    build_partial,
    build_preference_curve,
    build_relation_matrix,
    build_table,
    dial_table,
    floor_table,
    identity_table,
    pair_details,
    partial_table,
    stability_table,
    style_pair,
    style_table,
)

st.set_page_config(page_title="PROMETHEE – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Die vier Experimente rechnen …")
def _experiments():
    return {
        "identities": experiment_identities(),
        "agreement": experiment_agreement(),
        "dial": experiment_dial(),
        "partial": experiment_partial(),
        "addition": experiment_addition(),
        "floor": experiment_floor(),
        "stability": experiment_stability(),
    }


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🌊 PROMETHEE: Nettofluss statt Zyklen")
st.markdown(
    """
**PROMETHEE ist kein weiteres Verfahren neben der Summe, sondern ein Regler: Die Präferenzfunktion bestimmt, wie stark eine Differenz zählt.** Dies ist das siebte Stück der Linie
**Mehrkriterien-Ranking**: Wie bei ELECTRE wird je **Paar** verglichen, aber nicht mit einer Schwelle, sondern mit einer **Präferenzfunktion** P(d), die jede Differenz in eine Stärke
zwischen 0 und 1 übersetzt. Aus den Stärken entstehen der **Plusfluss** (wie sehr a die anderen überbietet), der **Minusfluss** (wie sehr a überboten wird) und der **Nettofluss**.
An den Enden des Reglers stehen zwei bekannte Verfahren: **Zählen** (die gewöhnliche Funktion: die gewichtete Borda-Zählung) und **Summieren** (V-Form über die ganze Spannweite: die
gewichtete Summe). Der Entscheider ist simuliert und kennt seinen Nutzen.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert PROMETHEE", expanded=True):
    st.markdown(
        """
1. **Präferenzfunktion:** Für jedes Paar (a, b) und jedes Kriterium wird die Differenz d der Min-Max-normierten Werte in eine Stärke P(d) ∈ [0, 1] übersetzt. Sechs Standardfunktionen:
   gewöhnlich, U-Form, V-Form, Stufen, linear, Gauß; die Schwellen q (Indifferenz), p (Präferenz) und s sind Anteile der Spannweite.
2. **Präferenzgrad** π(a, b): die gewichtete Summe der Stärken über die Kriterien.
3. **Flüsse:** Φ⁺(a) = mittlerer Präferenzgrad von a über die anderen, Φ⁻(a) = mittlerer Präferenzgrad der anderen über a, Nettofluss Φ = Φ⁺ − Φ⁻.
4. **PROMETHEE II** ordnet nach Φ: immer eine vollständige Rangfolge. **PROMETHEE I** übertrifft a b nur, wenn a weder weniger Plus- noch mehr Minusfluss hat; widersprechen
   sich die Flüsse, bleibt das Paar **unvergleichbar**. Zyklen sind ausgeschlossen.
5. **Die Enden des Reglers:** Die gewöhnliche Funktion zählt nur, wer besser ist (Borda); die V-Form mit p = ganze Spannweite zählt die Größe der Differenz (die Summe).
6. **Gemessen wird** am simulierten Entscheider; die Funktion passt zum Entscheider oder nicht (Experiment 2).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen zur Wahl stehen.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist.")
    delta = st.selectbox("Streuung des ersten Kriteriums", C.DELTA_CHOICES, key="delta_select", format_func=lambda d: "unverändert" if d == 1.0 else f"{de(100 * d, 0)} % der Streuung",
                         help="Wie stark sich die Alternativen im ersten Kriterium unterscheiden; der Entscheider selbst ändert sich dadurch nicht.")
    decider = st.selectbox("Entscheider (simuliert)", C.DECIDERS, key="decider_select", format_func=lambda k: C.DECIDER_LABELS[k],
                           help="Wie der verborgene Entscheider bewertet. Die Güte-Entscheider bewerten den Rang im Kriterium, der Rohwerte-Entscheider die Größe.")
    st.markdown("**Präferenzfunktion**")
    kind = st.selectbox("Funktion", C.KINDS, key="kind_select", format_func=lambda k: C.KIND_LABELS[k],
                        help="Wie eine Differenz in eine Stärke übersetzt wird. Gewöhnlich = Zählen (Borda); V-Form mit p = 1,0 = die gewichtete Summe.")
    q = st.selectbox("Indifferenzschwelle q", C.Q_CHOICES, key="q_select", format_func=lambda v: de(v, 2), disabled=not uses(kind, "q"),
                     help="Differenzen bis q (Anteil der Spannweite) zählen nicht. Nur bei U-Form, Stufen und linear.")
    p = st.selectbox("Präferenzschwelle p", C.P_CHOICES, key="p_select", format_func=lambda v: de(v, 2), disabled=not uses(kind, "p"),
                     help="Ab p (Anteil der Spannweite) zählt eine Differenz voll. p = 1,0 ist die ganze Spannweite: dann ist der Nettofluss die gewichtete Summe. Nur bei V-Form, Stufen und linear.")
    s = st.selectbox("Gauß-Parameter s", C.S_CHOICES, key="s_select", format_func=lambda v: de(v, 2), disabled=not uses(kind, "s"),
                     help="Streuung der Gauß-Funktion (Anteil der Spannweite). Nur bei Gauß.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "kind_select": kind, "q_select": float(q), "p_select": float(p), "s_select": float(s), "decider_select": decider,
                   "delta_select": float(delta), "seed_input": int(seed)})

settings = Settings(int(n), int(m), kind, float(q), float(p), float(s), decider, float(delta), int(seed))
a = _analysis(settings)
names = alt_names(settings.n)
R2, RS = a.results["promethee"], a.results["summe"]
pairs = settings.n * (settings.n - 1) / 2

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Platz 1: PROMETHEE II", names[R2.top1], delta=f"bester nach Nutzen: {names[a.best]}", delta_color="off",
          help="Alternative mit dem größten Nettofluss; bei Gleichstand die kleinere Nummer.")
c2.metric("Platz 1: Summe", names[RS.top1], delta=f"τ gegen Nutzen: {de(RS.tau, 2)} (PROMETHEE {de(R2.tau, 2)})", delta_color="off",
          help="Gewichtete Summe auf Min-Max-normierten Werten mit denselben Gewichten; τ ist Kendalls τ-b der ganzen Rangfolge gegen den verborgenen Nutzen.")
c3.metric("Unvergleichbare Paare", f"{int(a.R.sum() / 2)} von {int(pairs)}", delta=f"strikt entschieden: {de(100 * a.P.sum() / pairs, 0)} %", delta_color="off",
          help="PROMETHEE I: Paare mit widersprüchlichen Flüssen (der Plusfluss der einen ist größer, ihr Minusfluss aber auch).")
c4.metric("Bester nicht übertroffen", "ja" if a.best_free else "nein", delta="Platz 1 von II ist stets unübertroffen", delta_color="off",
          help="Ob in PROMETHEE I keine Alternative den Besten des Entscheiders strikt übertrifft. Der Sieger von PROMETHEE II wird nie übertroffen (Eigenschaft der Definition).")

st.caption("Fluss-Ebene: Je weiter rechts oben eine Alternative liegt (mehr Plusfluss, weniger Minusfluss), desto besser; a übertrifft b in PROMETHEE I, wenn a in beiden Richtungen mindestens "
           "so gut liegt wie b. Matrix: → a übertrifft b, = beide Flüsse gleich, ? widersprüchlich (unvergleichbar).")
left, right = st.columns([2, 3])
with left:
    st.plotly_chart(build_preference_curve(settings.kind, settings.q, settings.p, settings.s), width="stretch", key="curve_chart")
    st.plotly_chart(build_net_flow(a), width="stretch", key="netflow_chart")
with right:
    st.plotly_chart(build_flow_plane(a), width="stretch", key="plane_chart")
    st.plotly_chart(build_relation_matrix(a), width="stretch", key="relation_chart")

with st.expander("Ein Paar im Einzelnen: Präferenz je Kriterium"):
    pa, pb = st.columns(2)
    ia = pa.selectbox("Alternative a", list(range(settings.n)), index=0, format_func=lambda i: names[i], key="pair_a")
    ib = pb.selectbox("Alternative b", list(range(settings.n)), index=min(1, settings.n - 1), format_func=lambda i: names[i], key="pair_b")
    if ia == ib:
        st.info("Bitte zwei verschiedene Alternativen wählen.")
    else:
        st.dataframe(style_pair(pair_details(a, int(ia), int(ib))), hide_index=True, width="stretch")
        st.caption(f"Präferenzgrad π({names[ia]}, {names[ib]}) = {de(float(a.pi[ia, ib]), 3)}, π({names[ib]}, {names[ia]}) = {de(float(a.pi[ib, ia]), 3)} "
                   f"(die gewichtete Summe der Spalten „P(… über …)“).")

tab_t, tab_m = st.tabs(["Alle Alternativen", "Verfahren im Vergleich"])
with tab_t:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch")
    if settings.decider == "untergrenze":
        st.caption("Zulässig sind Alternativen mit einer Güte von mindestens 0,1 in jedem Kriterium. PROMETHEE hat kein Veto und keine Untergrenze.")
with tab_m:
    st.dataframe(build_method_table(a), hide_index=True, width="stretch")
    st.caption("Dieselbe Matrix, derselbe Entscheider. Die gewichtete Borda-Zählung ist der Nettofluss der gewöhnlichen Funktion (Experiment 1).")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="promethee.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was kostet „statt Zyklen“?")
if R2.top1 == RS.top1:
    st.info(f"PROMETHEE II und die Summe wählen dieselbe Alternative ({names[R2.top1]}); die Rangfolgen {'sind gleich' if (a.order == (-RS.score).argsort(kind='stable')).all() else 'unterscheiden sich weiter hinten'}.")
else:
    st.warning(f"PROMETHEE II wählt {names[R2.top1]}, die Summe {names[RS.top1]}; der Beste des Entscheiders ist {names[a.best]} "
               f"({'PROMETHEE trifft' if R2.hit else 'die Summe trifft' if RS.hit else 'keines trifft'}).")
if a.R.sum() > 0:
    st.caption(f"PROMETHEE I lässt {int(a.R.sum() / 2)} Paare unvergleichbar; PROMETHEE II ordnet sie trotzdem, aus dem Nettofluss. Kein Zyklus, aber die Ordnung hängt an allen anderen Alternativen "
               "(Experiment 4).")
else:
    st.caption("In PROMETHEE I ist jedes Paar entschieden; der Nettofluss ordnet vollständig. Die Ordnung hängt an allen anderen Alternativen (Experiment 4).")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption("200 feste Matrizen (SWEEP_SEEDS), n = 20, Min-Max, wahre Gewichte, unabhängig von den Reglern. Die Experimente rechnen zusammen rund 10 Sekunden und starten erst auf Klick.")
if st.button("▶ Experimente rechnen", key="exp_start"):
    st.session_state["exp_rows"] = _experiments()
exp = st.session_state.get("exp_rows")
if exp:
    st.markdown("### Experiment 1: Zählen und Summieren sind zwei Punkte derselben Familie")
    st.caption("Anteil der Matrizen mit gleicher Rangfolge, je 200 Matrizen, n = 20; darunter, wie oft zwei Funktionen denselben Platz 1 wählen (5 Kriterien, Güte/linear).")
    st.dataframe(identity_table(exp["identities"]), hide_index=True, width="stretch")
    verdict(verdict_identities(exp["identities"], exp["agreement"]))
    st.dataframe(agreement_table(exp["agreement"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 2: Der Regler und der Entscheider")
    st.caption("Platz 1 getroffen, 5 Kriterien, vier Entscheider. Standardfehler 1,3 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten sind nicht belastbar.")
    st.plotly_chart(build_dial(exp["dial"], 5), width="stretch", key="dial_chart")
    verdict(verdict_dial(exp["dial"]))
    st.dataframe(dial_table(exp["dial"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 3: Die Teilordnung")
    st.caption("PROMETHEE I auf denselben Matrizen wie ELECTRE I: Anteile der ungeordneten Paare, Zyklen, und ob der Sieger von II oder der Beste des Entscheiders übertroffen wird (5 Kriterien).")
    st.plotly_chart(build_partial(exp["partial"]), width="stretch", key="partial_chart")
    verdict(verdict_partial(exp["partial"]))
    st.dataframe(partial_table(exp["partial"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 4: Abhängigkeit von den anderen Alternativen, Untergrenze, Stabilität")
    st.caption("Eine Alternative kommt dazu (n = 20 → 21): Anteil der alten Paare mit gekippter Ordnung. Dies ist eine Vorschau, kein Protokoll der Rangumkehr.")
    st.plotly_chart(build_addition(exp["addition"]), width="stretch", key="addition_chart")
    st.dataframe(addition_table(exp["addition"]), hide_index=True, width="stretch")
    verdict(verdict_dependence(exp["addition"], exp["floor"], exp["stability"]))
    fl, st_ = st.columns(2)
    with fl:
        st.markdown("**Entscheider mit Untergrenze** (kein Kriterium unter 0,1 Güte): Wahl unter der Untergrenze, 5 Kriterien.")
        st.dataframe(floor_table(exp["floor"]), hide_index=True, width="stretch")
    with st_:
        st.markdown("**Bleibt Platz 1 unter Gewichtsstörung?** (lognormal, 10 Störungen je Matrix, Güte/linear)")
        st.dataframe(stability_table(exp["stability"]), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Güte-Entscheider bewerten den Rang** (Quantil im Kriterium) | Dass Zählen dort gewinnen kann, ist plausibel: Die Borda-Zählung summiert Ränge; gemessen ist es nur für Güte/linear bei 5 Kriterien. Für einen Entscheider, der Größen bewertet, kehrt es sich um (Experiment 2). | – |
| **Einheitliche Schwellen** | Echte Anwendungen setzen q, p und s je Kriterium und mischen die Funktionstypen; hier ist es ein Anteil der Spannweite für alle. | – |
| **Die Gewichte sind bekannt** | In der Demo sind es die wahren Gewichte des Entscheiders; wie man sie erhebt, steht in den Vorgängern. | Gewichte erheben, AHP |
| **Ein Verfahren ohne Veto** | PROMETHEE kennt keine Untergrenze und kein Veto; unzulässige Alternativen werden gewählt (Experiment 4). Absolute Anforderungen sind Profile. | Zuordnung statt Rangfolge |
| **Die Alternativen sind gegeben** | Der Nettofluss hängt von jeder anderen Alternative ab; Experiment 4 ist nur eine Vorschau mit einer hinzugefügten Alternative. | Rangumkehr |
| **Kriterien zählen einzeln** | Die Zählung (gewöhnliche Funktion) zählt jedes Kriterium als eine Stimme; doppelte Kriterien zählen doppelt. | Kriterien-Wechselwirkung |
| **Eine Gewichtung, ein Punkt** | Ob die Rangfolge unter allen verträglichen Gewichten bestehen bleibt, zeigt nur die Stabilitätstabelle grob. | Unsichere Gewichte |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Normierung.** Min-Max: $n_{ij} = (x_{ij} - \min_k x_{kj}) / (\max_k x_{kj} - \min_k x_{kj})$, höher ist besser; Gewichte $w_j \ge 0$, $\sum_j w_j = 1$.

**Präferenzfunktionen** auf $d = n_{aj} - n_{bj}$ (Schwellen als Anteil der Spannweite): gewöhnlich $P(d) = \mathbb{1}[d > 0]$; U-Form $\mathbb{1}[d > q]$; V-Form
$\min(1, \max(0, d/p))$; Stufen $0$ für $d \le q$, $\tfrac12$ für $q < d \le p$, $1$ darüber; linear $\min(1, \max(0, (d - q)/(p - q)))$; Gauß $1 - e^{-d^2/(2 s^2)}$ für $d > 0$.

**Präferenzgrad und Flüsse.** $\pi(a, b) = \sum_j w_j\, P_j(n_{aj} - n_{bj})$; $\Phi^{+}(a) = \frac{1}{n-1} \sum_{b \ne a} \pi(a, b)$; $\Phi^{-}(a) = \frac{1}{n-1} \sum_{b \ne a} \pi(b, a)$;
$\Phi(a) = \Phi^{+}(a) - \Phi^{-}(a)$. Es gilt $\sum_a \Phi(a) = 0$.

**PROMETHEE II.** Rangfolge nach $\Phi$. **PROMETHEE I.** $a\, P\, b \iff \Phi^{+}(a) \ge \Phi^{+}(b) \wedge \Phi^{-}(a) \le \Phi^{-}(b)$ mit mindestens einer strikten Ungleichung; $a\, I\, b$,
wenn beide Flüsse gleich sind; sonst unvergleichbar. Die Relation ist die Schnittmenge zweier schwacher Ordnungen und damit transitiv; ein Zyklus ist ausgeschlossen. Gilt $a\, P\, b$,
folgt $\Phi(a) > \Phi(b)$; der Sieger von PROMETHEE II wird daher nie übertroffen.

**Zwei Grenzfälle.** Mit der gewöhnlichen Funktion ist $\Phi(a) = \sum_j w_j\, \frac{|\{b : n_{bj} < n_{aj}\}| - |\{b : n_{bj} > n_{aj}\}|}{n - 1}$, die gewichtete **Borda-Zählung**; sie ist affin in
$\sum_j w_j \cdot (\text{Rang}_j)$, der gewichteten Summe der Perzentil-Ränge. Mit der V-Form und $p = 1$ (die ganze Spannweite) ist $\pi(a, b) = \sum_j w_j (n_{aj} - n_{bj})^{+}$ und
$\Phi(a) = \frac{n\, U_a - \sum_b U_b}{n - 1}$ mit der gewichteten Summe $U_a = \sum_j w_j n_{aj}$: dieselbe Rangfolge wie die Summe.

**Literatur:** PROMETHEE I und II wurden von Jean-Pierre Brans und Mitarbeitern in den frühen 1980er Jahren eingeführt; Titel und Fundstellen sind hier nicht geprüft und deshalb
nicht angegeben.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html)."
)
