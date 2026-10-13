"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 5, 40, 8            # Standard 8: Die Fluss-Ebene bleibt lesbar
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5
SEED_MAX, DEFAULT_SEED = 999999, 2           # Seed 2: Der Standardfall zeigt mindestens ein unvergleichbares Paar, beide Verfahren treffen (je Preset geprüft)

KINDS = ("usual", "ushape", "vshape", "level", "linear", "gauss")
KIND_LABELS = {
    "usual": "Gewöhnlich (usual): jede Überlegenheit zählt gleich viel",
    "ushape": "U-Form: erst ab der Indifferenzschwelle q",
    "vshape": "V-Form: linear bis zur Präferenzschwelle p",
    "level": "Stufen (level): halbe Präferenz zwischen q und p",
    "linear": "Linear: null bis q, linear bis p",
    "gauss": "Gauß: glatt, Wendepunkt bei s",
}
KIND_SHORT = {"usual": "gewöhnlich", "ushape": "U-Form", "vshape": "V-Form", "level": "Stufen", "linear": "linear", "gauss": "Gauß"}
KIND_PARAMS = {"usual": (), "ushape": ("q",), "vshape": ("p",), "level": ("q", "p"), "linear": ("q", "p"), "gauss": ("s",)}
DEFAULT_KIND = "vshape"

Q_CHOICES = (0.0, 0.05, 0.1, 0.2)             # Indifferenzschwelle q (Anteil der Spannweite)
P_CHOICES = (0.25, 0.5, 0.75, 1.0)            # Präferenzschwelle p (Anteil der Spannweite; 1,0 = ganze Spannweite: die Summe)
S_CHOICES = (0.15, 0.25, 0.5)                 # Gauß-Parameter s
DEFAULT_Q, DEFAULT_P, DEFAULT_S = 0.1, 0.5, 0.25

DECIDERS = ("guete_linear", "guete_konkav", "guete_cobb", "roh_linear", "untergrenze")
DECIDER_LABELS = {
    "guete_linear": "Linear in der Güte (Quantil innerhalb des Kriteriums)",
    "guete_konkav": "Konkav in der Güte (Wurzel, gewichtet)",
    "guete_cobb": "Cobb-Douglas in der Güte (Logarithmus, gewichtet)",
    "roh_linear": "Linear in den Rohwerten (feste Nenn-Spannweite)",
    "untergrenze": "Linear in der Güte, mit Untergrenze (kein Kriterium unter 0,1)",
}
DECIDER_SHORT = {"guete_linear": "Güte/linear", "guete_konkav": "Güte/konkav", "guete_cobb": "Güte/Cobb-Douglas", "roh_linear": "Rohwerte/linear",
                 "untergrenze": "Güte/Untergrenze"}
DECIDER_PARTS = {"guete_linear": ("guete", "linear"), "guete_konkav": ("guete", "konkav"), "guete_cobb": ("guete", "cobb"), "roh_linear": ("roh", "linear"),
                 "untergrenze": ("guete", "linear")}
DEFAULT_DECIDER = "guete_linear"
THETA = 0.1                                    # Untergrenze der Güte des Entscheiders mit Untergrenze

DELTA_CHOICES = (1.0, 0.5, 0.2, 0.05)         # Anteil der Streuung des ersten Kriteriums (1 = unverändert)
DEFAULT_DELTA = 1.0

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 20                                     # Experimente: feste Instanz mit 20 Alternativen, unabhängig vom Regler
EXP_M = (3, 5, 8)
EXP_M_MAIN = 5
EXP_DECIDERS = ("guete_linear", "guete_konkav", "guete_cobb", "roh_linear")
EXP_SIGMAS = (0.2, 0.4)                        # Streuung der lognormalen Gewichtsstörung
EXP_DRAWS = 10                                 # Störungen je Matrix
ELECTRE_CSTAR = 0.6                            # Vergleichswert aus dem Stück zu ELECTRE: Konkordanzschwelle ohne Veto

# Konfigurationen der Experimente: (Name, Funktion, q, p, s)
CONFIGS = (
    ("usual", "usual", 0.0, 1.0, 0.25),
    ("ushape(q=0,1)", "ushape", 0.1, 1.0, 0.25),
    ("level(q=0,1, p=0,5)", "level", 0.1, 0.5, 0.25),
    ("linear(q=0,05, p=0,5)", "linear", 0.05, 0.5, 0.25),
    ("vshape(p=0,25)", "vshape", 0.0, 0.25, 0.25),
    ("vshape(p=0,5)", "vshape", 0.0, 0.5, 0.25),
    ("vshape(p=1,0)", "vshape", 0.0, 1.0, 0.25),
    ("gauss(s=0,25)", "gauss", 0.0, 1.0, 0.25),
)

COLOR_FLOW = "#2F6B65"
COLOR_BEST = "#D68A2E"
COLOR_SUM = "#8A96A6"
COLOR_WARN = "#c0392b"
COLOR_LINE = "#4c78a8"
COLOR_OTHER = "rgba(150,150,150,0.55)"

PRESETS = {
    "Standardfall (V-Form, p = 0,5)": {"n": DEFAULT_N, "m": DEFAULT_M, "kind": DEFAULT_KIND, "q": DEFAULT_Q, "p": DEFAULT_P, "s": DEFAULT_S,
                                       "decider": DEFAULT_DECIDER, "delta": 1.0, "seed": DEFAULT_SEED},
    "Zählen": {"n": 8, "m": 5, "kind": "usual", "q": 0.1, "p": 0.5, "s": 0.25, "decider": "guete_linear", "delta": 1.0, "seed": 6},
    "Summieren": {"n": 8, "m": 5, "kind": "vshape", "q": 0.1, "p": 1.0, "s": 0.25, "decider": "roh_linear", "delta": 1.0, "seed": 6},
    "Zwei ohne Urteil": {"n": 8, "m": 5, "kind": "vshape", "q": 0.1, "p": 0.5, "s": 0.25, "decider": "guete_linear", "delta": 1.0, "seed": 13},
}
PRESET_HELP = {
    "Standardfall (V-Form, p = 0,5)": "8 Alternativen, 5 Kriterien, V-Form mit p = 0,5 der Spannweite: eine Zwischenstufe zwischen Zählen und Summieren; Nettofluss und Summe wählen denselben Besten.",
    "Zählen": "Gewöhnliche Funktion (jede Überlegenheit zählt gleich): Ein Entscheider, der den Rang im Kriterium bewertet, wird getroffen, die gewichtete Summe verfehlt ihn.",
    "Summieren": "V-Form mit p = ganze Spannweite, Entscheider in den Rohwerten: Der Nettofluss ist die gewichtete Summe und trifft; die gewöhnliche Funktion verfehlt den Besten.",
    "Zwei ohne Urteil": "PROMETHEE I: Ein Paar von Alternativen hat widersprüchliche Flüsse (der Plusfluss der einen ist größer, der Minusfluss auch) und bleibt unvergleichbar; PROMETHEE II ordnet es trotzdem.",
}
