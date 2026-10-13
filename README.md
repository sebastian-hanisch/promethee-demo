# PROMETHEE: Nettofluss statt Zyklen – Streamlit-Demo

<!-- Nach dem Deploy hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-promethee-demo.streamlit.app/)** -->

Siebtes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind von
[ELECTRE](https://github.com/sebastian-hanisch/electre-demo). Wie dort wird je **Paar** von Alternativen verglichen, aber nicht mit einer Schwelle, sondern mit einer
**Präferenzfunktion** P(d), die jede Differenz in eine Stärke zwischen 0 und 1 übersetzt. Aus den Stärken entstehen **Plusfluss**, **Minusfluss** und **Nettofluss**; PROMETHEE II ordnet nach
dem Nettofluss (immer vollständig), PROMETHEE I übertrifft nur, wo beide Flüsse zustimmen (eine Halbordnung ohne Zyklen).

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der **Entscheider** kennt seine Gewichte und
seinen Nutzen (linear, konkav oder Cobb-Douglas in der Güte, linear in den Rohwerten, oder linear mit Untergrenze); er ist ein Orakel, kein Mensch.

## Warum dieses Problem

**PROMETHEE ist kein weiteres Verfahren neben der Summe, sondern ein Regler: Die Präferenzfunktion bestimmt, wie stark eine Differenz zählt.** An den Enden stehen zwei bekannte Verfahren:
Die gewöhnliche Funktion **zählt** (gewichtete Borda-Zählung), die V-Form über die ganze Spannweite **summiert** (die gewichtete Summe). Die Kernfrage der Demo: Welche Funktion zum
Entscheider passt, und was kostet „statt Zyklen“?

## Befunde und Korrekturen gegenüber dem Plan

- **Die Brücke zur Turnier-Linie ist eine Identität, keine Analogie.** Mit der gewöhnlichen Funktion ist der Nettofluss die gewichtete Borda-Zählung und die gewichtete Summe der
  Perzentil-Ränge (Φ = 2 · Perzentil-Summe − 1); Copeland nur als Spezialfall. Mit der V-Form und p = Spannweite ist er die gewichtete Summe: Φ = (n · Summe − Σ Summe)/(n − 1).
- **„PROMETHEE ist besser als die Summe“ lässt sich nicht sagen.** Bei einem Entscheider, der den Rang im Kriterium bewertet (Güte), gewinnt das Zählen (5 Kriterien: 82,5 % gegen 67,0 %);
  bei einem Entscheider in den Rohwerten das Summieren (96,5 % gegen 64,5 %). Dass Zählen in der Güte-Welt gewinnt, folgt aus der Konstruktion: Die Güte ist ein Quantil.
- **„Statt Zyklen“ ist ein Austausch.** PROMETHEE I hat nie einen Zyklus (Eigenschaft der Definition, in jeder Zelle 0,0 %) und 0,1 bis 14,5 % unvergleichbare Paare; ELECTRE I mit c\* = 0,6
  hat auf denselben Matrizen in 38,5 % einen Zyklus und 20,4 % unvergleichbare Paare. Der Preis: Die Ordnung hängt stärker von den anderen Alternativen ab.
- **Die Vorschau auf die Rangumkehr ist deutlich:** Kommt eine Alternative dazu, kippt bei der gewöhnlichen Funktion in 89,0 % der Matrizen mindestens ein Paar der alten (Summe: 13,5 %),
  und Platz 1 wechselt in 5,0 % (Summe: 0,5 %). Die V-Form über die ganze Spannweite verhält sich wie die Summe. Das Protokoll gehört in ein eigenes Stück.
- **Kein Veto:** Bei einer Untergrenze wählt PROMETHEE in 12,5 bis 18,0 % der Fälle eine unzulässige Alternative (Summe: 11,0 %).
- **Die Stabilität gegen Gewichtsstörung ist nicht besser.** Die gewöhnliche Funktion ist um 1,5 bis 1,9 Punkte weniger stabil als die Summe; V-Form 0,5 liegt gleichauf. Kein belastbarer
  Unterschied bei 200 Matrizen.
- **Die Stabilitätszahlen weichen von der Phase 0 ab,** weil die Demo die Störung mit dem SplitMix64-Strom zieht (die Phase 0 rechnete mit `numpy.default_rng`); die Richtung ist dieselbe.
- **Die Experimente starten auf Klick** (rund 5 Sekunden), nicht beim Laden.

## Modell

- **Matrix:** n Alternativen (5 bis 40, Standard 8), m Kriterien (3 bis 8), Rohwerte mit Richtung und Einheit; optional ein Kriterium auf den Anteil δ seiner Streuung gestaucht.
- **Normierung und Gewichte:** nur Min-Max; die wahren Gewichte des Entscheiders (die Erhebung steht in den Vorgängern).
- **Präferenzfunktionen** auf der Differenz d der normierten Werte, Schwellen als Anteil der Spannweite: gewöhnlich, U-Form (q), V-Form (p), Stufen (q, p), linear (q, p), Gauß (s).
- **Flüsse:** π(a, b) = Σ_j w_j P_j(d_j); Φ⁺(a) = Σ_b π(a, b)/(n − 1); Φ⁻(a) = Σ_b π(b, a)/(n − 1); Φ = Φ⁺ − Φ⁻. **PROMETHEE I:** a P b, wenn Φ⁺(a) ≥ Φ⁺(b) und Φ⁻(a) ≤ Φ⁻(b)
  mit mindestens einer strikten Ungleichung; beide gleich: I; sonst unvergleichbar.
- **Entscheider mit Untergrenze:** Nutzen linear in der Güte; Alternativen mit einer Güte unter 0,1 in irgendeinem Kriterium sind unzulässig und liegen im Nutzen unter allen zulässigen.
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), n = 20, unabhängig vom Regler n.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, derselbe Entscheider, alle Funktionen:** Vergleiche zwischen Funktionen sind gepaart.
2. **Messgröße ist die Entscheidung** (Platz 1, Kendalls τ-b, Nutzenverlust) in vier Welten; keine Aussage „besser“ ohne die Welt.
3. **Teilordnung:** Anteile der ungeordneten Paare, Zyklen, ob der Sieger von II übertroffen wird; ELECTRE I (c\* = 0,6, ohne Veto) auf denselben Matrizen als Vergleich.
4. **Vorschau Rangumkehr:** eine hinzugefügte Alternative, Anteil der alten Paare mit gekippter Ordnung.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. 200 feste Matrizen, n = 20, wahre Gewichte, Min-Max. **Der Standardfehler einer Trefferquote ist 1,3 bis 3,5 Punkte; Unterschiede unter etwa 6 Punkten
sind nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Sind die zwei Identitäten exakt? (3, 5, 8 Kriterien; je 200 Matrizen) | Gewöhnlich = gewichtete Borda-Zählung = Perzentil-Summe und V-Form (p = Spannweite) = Summe: gleiche Rangfolge in 100 % von 600 Fällen, Abweichung höchstens 4·10⁻¹⁶. Gegenprobe V-Form p = 0,5 hat die Rangfolge der Summe in 6 / 2 / 1 %. | `test_the_two_identities_hold_in_every_case` |
| Wie oft wählen zwei Funktionen denselben Platz 1? (5 Kriterien, Güte/linear) | gewöhnlich gegen V (p = 1,0): 64,5 % (τ 0,82); V 0,25 gegen 0,5: 90,0 %; 0,5 gegen 1,0: 90,0 %; 0,25 gegen 1,0: 85,5 %; linear gegen V 0,5: 95,5 %; gewöhnlich gegen U-Form: 69,5 %; gewöhnlich gegen Summe: 64,5 %. | `test_agreement_between_functions` |
| Platz 1 getroffen, 5 Kriterien (Güte/linear, konkav, Cobb-Douglas, Rohwerte) | **Summe 67,0 / 72,5 / 69,0 / 96,5 %; gewöhnlich 82,5 / 76,0 / 69,5 / 64,5 %**; V 0,25: 77,0 / 79,0 / 74,5 / 85,5 %; V 0,5: 69,5 / 75,0 / 72,5 / 91,0 %; V 1,0 = Summe; linear (q 0,05, p 0,5) 66,0 / 71,0 / 70,0 / 90,5 %; Gauß (s 0,25) 65,5 / 71,5 / 70,0 / 92,5 %. | `test_first_place_for_every_function` |
| Dasselbe bei 8 Kriterien | Summe 78,5 / 78,0 / 67,5 / 93,0 %; gewöhnlich 82,5 / 76,0 / 66,0 / 67,0 %. Rohwerte, 3 Kriterien: gewöhnlich 75,5 %, Summe 94,0 %. | `test_first_place_for_every_function` |
| Wer gewinnt in welcher Welt? (5 Kriterien) | Zählen in der Güte-Welt +15,5 Punkte gegenüber der Summe, Summieren in der Rohwert-Welt +32,0 Punkte; in der Rohwert-Welt steigt die V-Form monoton von gewöhnlich über p 0,25 und 0,5 zur Summe. | `test_counting_wins_in_the_rank_world…`, `test_the_v_shape_moves_monotonically…` |
| PROMETHEE I: Teilordnung (5 Kriterien; je Funktion) | strikt 85,5 bis 99,9 %, beidseitig 0,0 %, **unvergleichbar 0,1 / 10,0 / 12,5 / 12,2 / 8,5 / 10,9 / 14,5 / 12,8 %** (gewöhnlich, U, Stufen, linear, V 0,25, V 0,5, V 1,0, Gauß); Zyklus 0,0 %; Sieger von II nie übertroffen. Der Beste ist nicht übertroffen in 78,0 bis 86,5 %. | `test_partial_order_shares` |
| ELECTRE I auf denselben Matrizen (c\* = 0,6, kein Veto) | strikt 79,6 %, unvergleichbar 20,4 %, **Zyklus 38,5 %**. | `test_electre_reference_on_the_same_matrices` |
| Eine Alternative kommt dazu (n 20 → 21; 5 Kriterien): gekippte Paare / Matrizen mit gekipptem Paar / Platz 1 wechselt | Summe 0,20 / 13,5 / 0,5 %; **gewöhnlich 1,13 / 89,0 / 5,0 %**; V 0,5: 0,46 / 52,0 / 3,5 %; V 1,0 = Summe; linear: 0,45 / 48,0 / 2,5 %. | `test_adding_one_alternative` |
| Untergrenze θ = 0,1 (40,9 % der Alternativen unzulässig): Wahl unzulässig / zulässiger Bester getroffen | Summe 11,0 / 64,5 %; gewöhnlich 18,0 / 71,0 %; V 0,5: 12,5 / 66,0 %; V 0,25: 14,0 / 71,5 %; Gauß: 12,5 / 62,0 %. | `test_choices_under_the_floor` |
| Bleibt Platz 1 unter Gewichtsstörung? (σ = 0,2 / 0,4; Güte/linear) | Summe 90,3 / 80,8 %; gewöhnlich 88,8 / 78,9 %; V 0,5: 90,2 / 81,7 %; V 1,0 = Summe. | `test_stability_of_first_place` |

## Ehrliche Grenzen

- **Die Güte-Entscheider bewerten den Rang.** Dass Zählen dort gewinnt, folgt aus der Konstruktion (Güte ist ein Quantil, die Borda-Zählung summiert Ränge). Die Aussage ist „Funktion und
  Entscheider müssen zusammenpassen“, nicht „Zählen ist besser“.
- **Einheitliche Schwellen.** Echte Anwendungen setzen q, p und s je Kriterium und mischen die Funktionstypen; hier ist es ein Anteil der Spannweite für alle.
- **Eine Vorschau auf die Rangumkehr,** kein Protokoll: eine hinzugefügte Alternative mit zufälligen Werten; ob sie die Spannweite ändert, wird nicht getrennt.
- **Nur PROMETHEE I und II.** GAIA, PROMETHEE V (Auswahl unter Nebenbedingungen) und VI (Intervalle) und die Gruppenvarianten sind nicht Teil; die Gewichte sind bekannt.
- **Kleine Unterschiede sind Rauschen.** Die Reihenfolge der Funktionen in den Güte-Spalten (konkav, Cobb-Douglas), die Untergrenzen-Zahlen und die Stabilität liegen innerhalb von zwei
  Standardfehlern. Quellenangaben nur aus Zusammenfassungen.

## Verifikation

- **Kern:** die sechs Präferenzfunktionen, Präferenzgrade, Flüsse, PROMETHEE I (P, I, R) und die Borda-Zählung gegen Handrechnung auf den Mini-Instanzen (`tests/conftest.py`: vier
  Alternativen mit w = (0,5; 0,3; 0,2), Φ = 0,8 / 0,2 / 0 / −1 bei gewöhnlicher Funktion; ein Fall, in dem Zählen einen Gleichstand und Summieren einen Sieger liefert; ein Fall mit einem
  unvergleichbaren Paar).
- **Orakel** (`tests/test_oracle_promethee.py`): Flüsse gegen eine Definitionsschleife für alle sechs Funktionen, Σ Φ = 0; **gewöhnlich = gewichtete Borda-Zählung = 2 · Perzentil-Summe − 1**;
  **V-Form mit p = Spannweite = Summe** (Gegenprobe p = 0,5); PROMETHEE I ist transitiv, asymmetrisch und zyklenfrei, jedes Paar liegt in genau einer Klasse, der Sieger von II wird nie
  übertroffen, strikte Präferenz impliziert größeren Nettofluss; Gegenprobe: ELECTRE hat auf denselben Matrizen Zyklen; Monotonie der Funktionen.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), alle Presets, jede Funktion, jeder Entscheider, deaktivierte Schwellen, Permalink, PDF und Experimente
  erst nach Klick, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `pmt_constants.py` | Regler-Grenzen, feste Annahmen, Konfigurationen der Experimente, Farben, Presets |
| `pmt_promethee.py` | Präferenzfunktionen, Präferenzgrade, Flüsse, PROMETHEE I, Borda-Zählung, Vergleichswert ELECTRE |
| `pmt_score.py` | Min-Max, Perzentil-Ränge, Summe, Nutzen des Entscheiders (auch mit Untergrenze), Dominanz, Kendalls τ-b |
| `pmt_evaluation.py` | Fall der App, vier Experimente, Verdikte |
| `pmt_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `pmt_visualization.py` | Präferenzkurve, Fluss-Ebene, Relationsmatrix, Tabellen und Plotly-Abbildungen (Achsen gesperrt) |
| `pmt_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; `scipy` für die Zyklenprüfung) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanzen, PROMETHEE, Score, Orakel, Generator, Auswertung, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Rangumkehr als Protokoll** (Hinzufügen, Entfernen, Duplizieren): Experiment 4 ist nur eine Vorschau; das Protokoll hat ein eigenes Stück.
- **GAIA und PROMETHEE V/VI:** eine Projektion, die Auswahl unter Nebenbedingungen und Intervalle sind eigene Verfahren.
- **Schwellen je Kriterium und gemischte Funktionen:** ein Regler für alle Kriterien.
- **Absolute Untergrenzen:** PROMETHEE kennt kein Veto; absolute Anforderungen sind Profile, die ein eigenes Stück behandelt.
- **Unsichere Gewichte als Bereich:** eine Gewichtung, ein Punkt; die Verteilung über den Gewichtsraum steht beim SMAA.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
