"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte. Nur numpy und scipy (über pmt_promethee), kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pmt_constants as C
import pmt_promethee as pm
import pmt_score as sc
from entscheidungsmatrix_generator import SplitMix64, make_matrix

NOISE_OFFSET = 20011       # Strom der Gewichtsstörungen: SplitMix64(seed + NOISE_OFFSET)


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    kind: str = C.DEFAULT_KIND
    q: float = C.DEFAULT_Q
    p: float = C.DEFAULT_P
    s: float = C.DEFAULT_S
    decider: str = C.DEFAULT_DECIDER
    delta: float = C.DEFAULT_DELTA
    seed: int = C.DEFAULT_SEED


@dataclass
class Case:
    """Eine Matrix mit simuliertem Entscheider: vorzeichenbehaftete Rohwerte nach Verschmälerung, wahre Gewichte, Nutzen, zulässige Alternativen (Untergrenze)."""
    matrix: object
    goodness: np.ndarray
    S: np.ndarray
    w: np.ndarray
    utility: np.ndarray
    acceptable: np.ndarray


def make_case(n: int, m: int, seed: int, delta: float = 1.0, decider: str = "guete_linear") -> Case:
    scale, form = C.DECIDER_PARTS[decider]
    x = make_matrix(n, m, 0.0, SplitMix64(seed))
    g, _, S = sc.build_case(x.goodness, x.directions, m, delta)
    theta = C.THETA if decider == "untergrenze" else None
    return Case(x, g, S, x.weights, sc.decider_utility(scale, form, g, S, x.weights, m, theta), sc.acceptable(g, C.THETA))


@dataclass
class Result:
    score: np.ndarray          # höher ist besser
    top1: int
    hit: bool
    tau: float
    loss: float
    tie_at_top: bool


def _result(case: Case, score: np.ndarray, best: int) -> Result:
    t = sc.top1(score)
    return Result(score, t, t == best, sc.kendall_tau_b(score, case.utility), sc.utility_loss(case.utility, t), int((score == score.max()).sum()) > 1)


@dataclass
class Analysis:
    settings: Settings
    case: Case
    N: np.ndarray
    pi: np.ndarray             # Präferenzgrade
    plus: np.ndarray
    minus: np.ndarray
    phi: np.ndarray
    P: np.ndarray              # PROMETHEE I
    I: np.ndarray
    R: np.ndarray
    order: np.ndarray          # PROMETHEE II: Alternativen von der besten zur schlechtesten
    best: int
    results: dict              # "summe", "promethee", "borda"
    front: np.ndarray
    best_free: bool            # der Beste des Entscheiders wird in PROMETHEE I von keiner Alternative übertroffen


def analyse(settings: Settings) -> Analysis:
    s = settings
    case = make_case(s.n, s.m, s.seed, s.delta, s.decider)
    N = sc.minmax(case.S)
    pi = pm.preference_degrees(N, case.w, s.kind, s.q, s.p, s.s)
    plus, minus, phi = pm.flows(pi)
    P, I, R = pm.promethee_one(plus, minus)
    best = sc.top1(case.utility)
    results = {
        "summe": _result(case, sc.weighted_score(N, case.w), best),
        "promethee": _result(case, phi, best),
        "borda": _result(case, pm.weighted_borda(N, case.w), best),
    }
    return Analysis(s, case, N, pi, plus, minus, phi, P, I, R, pm.net_flow_order(phi), best, results, sc.front_mask(case.goodness), not P[:, best].any())


def config_scores(N: np.ndarray, w: np.ndarray, configs=C.CONFIGS) -> dict:
    """Nettofluss je Konfiguration (Name -> Phi) auf derselben Matrix."""
    return {name: pm.flows(pm.preference_degrees(N, w, kind, q, p, s))[2] for name, kind, q, p, s in configs}


# ----------------------------------------------------------------------------- Experiment 1: Identitäten

AGREEMENT_PAIRS = (("usual", "vshape(p=1,0)"), ("vshape(p=0,25)", "vshape(p=0,5)"), ("vshape(p=0,5)", "vshape(p=1,0)"), ("vshape(p=0,25)", "vshape(p=1,0)"),
                   ("linear(q=0,05, p=0,5)", "vshape(p=0,5)"), ("usual", "ushape(q=0,1)"))


def _same_order(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable")))


def experiment_identities(ms=C.EXP_M, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """usual = gewichtete Borda-Zählung = gewichtete Summe der Perzentil-Ränge; V-Form mit p = Spannweite = gewichtete Summe: gleiche Rangfolge und größte Abweichung."""
    rows = []
    for m in ms:
        acc = {k: [] for k in ("borda", "percentile", "vsum", "vsum_alt")}
        diff = {"borda": 0.0, "vsum": 0.0}
        for s in seeds:
            c = make_case(n, m, s)
            N = sc.minmax(c.S)
            phi_u = pm.flows(pm.preference_degrees(N, c.w, "usual"))[2]
            phi_v = pm.flows(pm.preference_degrees(N, c.w, "vshape", p=1.0))[2]
            bo = pm.weighted_borda(N, c.w)
            ws = N @ c.w
            pw = sc.percentile_normalize(c.S) @ c.w
            acc["borda"].append(_same_order(phi_u, bo))
            acc["percentile"].append(_same_order(phi_u, pw))
            acc["vsum"].append(_same_order(phi_v, ws))
            acc["vsum_alt"].append(_same_order(pm.flows(pm.preference_degrees(N, c.w, "vshape", p=0.5))[2], ws))      # Gegenprobe: p = 0,5
            diff["borda"] = max(diff["borda"], float(np.abs(phi_u - bo).max()))
            diff["vsum"] = max(diff["vsum"], float(np.abs(phi_v - (n * ws - ws.sum()) / (n - 1)).max()))
        rows.append({"m": m, "borda": float(np.mean(acc["borda"])), "percentile": float(np.mean(acc["percentile"])), "vsum": float(np.mean(acc["vsum"])),
                     "vsum_p05": float(np.mean(acc["vsum_alt"])), "diff_borda": diff["borda"], "diff_vsum": diff["vsum"], "cases": len(seeds)})
    return rows


def experiment_agreement(m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Wie oft wählen zwei Funktionen denselben Platz 1 (Güte/linear)?"""
    acc = {pr: ([], []) for pr in AGREEMENT_PAIRS}
    same_sum = []
    for s in seeds:
        c = make_case(n, m, s)
        N = sc.minmax(c.S)
        sco = config_scores(N, c.w)
        for a, b in AGREEMENT_PAIRS:
            acc[(a, b)][0].append(sc.top1(sco[a]) == sc.top1(sco[b]))
            acc[(a, b)][1].append(sc.kendall_tau_b(sco[a], sco[b]))
        same_sum.append(sc.top1(sco["usual"]) == sc.top1(N @ c.w))
    rows = [{"a": a, "b": b, "same_top1": float(np.mean(x)), "tau": float(np.mean(y))} for (a, b), (x, y) in acc.items()]
    rows.append({"a": "usual", "b": "Summe", "same_top1": float(np.mean(same_sum)), "tau": float("nan")})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: der Regler und der Entscheider

def experiment_dial(deciders=C.EXP_DECIDERS, ms=C.EXP_M, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for dec in deciders:
        for m in ms:
            hits = {}
            for s in seeds:
                c = make_case(n, m, s, 1.0, dec)
                N = sc.minmax(c.S)
                best = sc.top1(c.utility)
                cand = {"summe": N @ c.w, **config_scores(N, c.w)}
                for k, v in cand.items():
                    st = hits.setdefault(k, ([], []))
                    st[0].append(sc.top1(v) == best)
                    st[1].append(sc.kendall_tau_b(v, c.utility))
            for k, (h, t) in hits.items():
                p_ = float(np.mean(h))
                rows.append({"decider": dec, "m": m, "method": k, "hit": p_, "tau": float(np.mean(t)), "se": float(np.sqrt(p_ * (1 - p_) / len(seeds)))})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: die Teilordnung

def experiment_partial(m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS, configs=C.CONFIGS) -> list[dict]:
    """PROMETHEE I: Anteile der ungeordneten Paare, Zyklen, Sieger von II übertroffen?, Bester nicht übertroffen; dazu ELECTRE I (c* = 0,6, ohne Veto) auf denselben Matrizen."""
    pairs = n * (n - 1) / 2
    cases = [make_case(n, m, s) for s in seeds]
    norm = [sc.minmax(c.S) for c in cases]
    rows = []
    for name, kind, q, p, s_ in configs:
        acc = {k: [] for k in ("p", "i", "r", "cycle", "winner_outranked", "best_free")}
        for c, N in zip(cases, norm):
            plus, minus, phi = pm.flows(pm.preference_degrees(N, c.w, kind, q, p, s_))
            P, I, R = pm.promethee_one(plus, minus)
            acc["p"].append(P.sum() / pairs)
            acc["i"].append(I.sum() / 2 / pairs)
            acc["r"].append(R.sum() / 2 / pairs)
            acc["cycle"].append(pm.has_cycle(P))
            acc["winner_outranked"].append(bool(P[:, int(np.argmax(phi))].any()))
            acc["best_free"].append(not P[:, sc.top1(c.utility)].any())
        rows.append({"method": name, **{k: float(np.mean(v)) for k, v in acc.items()}})
    pe, ie, re, ce = [], [], [], []
    for c, N in zip(cases, norm):
        S = pm.electre_reference(N, c.w, C.ELECTRE_CSTAR)
        Pe = S & ~S.T
        Ie = S & S.T
        Re = ~S & ~S.T
        np.fill_diagonal(Re, False)
        pe.append(Pe.sum() / pairs)
        ie.append(Ie.sum() / 2 / pairs)
        re.append(Re.sum() / 2 / pairs)
        ce.append(pm.has_cycle(S))
    rows.append({"method": "ELECTRE I (c* = 0,6, kein Veto)", "p": float(np.mean(pe)), "i": float(np.mean(ie)), "r": float(np.mean(re)), "cycle": float(np.mean(ce)),
                 "winner_outranked": float("nan"), "best_free": float("nan")})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Abhängigkeit, Untergrenze, Stabilität

ADDITION_CONFIGS = (("Summe (Min-Max)", None), ("usual", ("usual", 0.0, 1.0, 0.25)), ("vshape(p=0,5)", ("vshape", 0.0, 0.5, 0.25)), ("vshape(p=1,0)", ("vshape", 0.0, 1.0, 0.25)),
                    ("linear(q=0,05, p=0,5)", ("linear", 0.05, 0.5, 0.25)))


def _score(N: np.ndarray, w: np.ndarray, cfg) -> np.ndarray:
    if cfg is None:
        return N @ w
    kind, q, p, s = cfg
    return pm.flows(pm.preference_degrees(N, w, kind, q, p, s))[2]


def experiment_addition(m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Eine Alternative kommt dazu (n -> n + 1, die ersten n bleiben): Anteil der alten Paare mit gekippter Ordnung, Matrizen mit mindestens einem gekippten Paar, Platz 1 der
    alten n wechselt. Vorschau, kein Protokoll der Rangumkehr."""
    iu = np.triu_indices(n, 1)
    rows = []
    for name, cfg in ADDITION_CONFIGS:
        share, any_flip, top = [], [], []
        for s in seeds:
            c = make_case(n + 1, m, s)
            new = _score(sc.minmax(c.S), c.w, cfg)[:n]
            old = _score(sc.minmax(c.S[:n]), c.w, cfg)
            flip = (np.sign(new[:, None] - new[None, :])[iu] * np.sign(old[:, None] - old[None, :])[iu]) < 0
            share.append(flip.mean())
            any_flip.append(flip.any())
            top.append(int(np.argmax(new)) != int(np.argmax(old)))
        rows.append({"method": name, "flipped_pairs": float(np.mean(share)), "any_flip": float(np.mean(any_flip)), "top_changes": float(np.mean(top))})
    return rows


FLOOR_CONFIGS = (("Summe", None), ("usual", ("usual", 0.0, 1.0, 0.25)), ("vshape(p=0,5)", ("vshape", 0.0, 0.5, 0.25)), ("vshape(p=0,25)", ("vshape", 0.0, 0.25, 0.25)),
                 ("gauss(s=0,25)", ("gauss", 0.0, 1.0, 0.25)))


def experiment_floor(m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Entscheider mit Untergrenze: Anteil der Wahlen unter der Untergrenze und Treffer auf den zulässigen Besten. PROMETHEE hat kein Veto."""
    bad = {k: [] for k, _ in FLOOR_CONFIGS}
    hit = {k: [] for k, _ in FLOOR_CONFIGS}
    unacc = []
    for s in seeds:
        c = make_case(n, m, s, 1.0, "untergrenze")
        if not c.acceptable.any():
            continue
        unacc.append(1 - c.acceptable.mean())
        N = sc.minmax(c.S)
        best = sc.top1(c.utility)
        for k, cfg in FLOOR_CONFIGS:
            t = int(np.argmax(_score(N, c.w, cfg)))
            bad[k].append(not c.acceptable[t])
            hit[k].append(t == best)
    return [{"method": k, "unacceptable": float(np.mean(bad[k])), "hit": float(np.mean(hit[k])), "unacceptable_share": float(np.mean(unacc))} for k, _ in FLOOR_CONFIGS]


STABILITY_CONFIGS = (("Summe", None), ("usual", ("usual", 0.0, 1.0, 0.25)), ("vshape(p=0,5)", ("vshape", 0.0, 0.5, 0.25)), ("vshape(p=1,0)", ("vshape", 0.0, 1.0, 0.25)))


def experiment_stability(sigmas=C.EXP_SIGMAS, m=C.EXP_M_MAIN, n=C.EXP_N, seeds=C.SWEEP_SEEDS, draws=C.EXP_DRAWS) -> list[dict]:
    """Bleibt Platz 1, wenn die Gewichte lognormal gestört werden (w' proportional w exp(sigma eps))? Entscheider Güte/linear."""
    rows = []
    for sg in sigmas:
        stay = {k: [] for k, _ in STABILITY_CONFIGS}
        for s in seeds:
            c = make_case(n, m, s)
            N = sc.minmax(c.S)
            base = {k: int(np.argmax(_score(N, c.w, cfg))) for k, cfg in STABILITY_CONFIGS}
            rng = SplitMix64(s + NOISE_OFFSET)
            for _ in range(draws):
                w2 = c.w * np.exp(sg * rng.normal(m))
                w2 = w2 / w2.sum()
                for k, cfg in STABILITY_CONFIGS:
                    stay[k].append(int(np.argmax(_score(N, w2, cfg))) == base[k])
        for k, _ in STABILITY_CONFIGS:
            rows.append({"sigma": sg, "method": k, "stay": float(np.mean(stay[k]))})
    return rows


# ----------------------------------------------------------------------------- Verdikte (live aus den Zeilen)

def _row(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def verdict_identities(ident: list[dict], agree: list[dict]) -> tuple[str, str]:
    total = sum(r["cases"] for r in ident)
    lo = min(min(r["borda"], r["percentile"], r["vsum"]) for r in ident)
    p05 = max(r["vsum_p05"] for r in ident)
    a = _row(agree, a="usual", b="Summe")
    return "info", (f"Mit der gewöhnlichen Funktion ist der Nettofluss die gewichtete Borda-Zählung und die gewichtete Summe der Perzentil-Ränge, mit der V-Form und p = Spannweite die gewichtete Summe: "
                    f"gleiche Rangfolge in {de(100 * lo, 0)} % von {total} Fällen je Identität. Gegenprobe: V-Form mit p = 0,5 hat die Rangfolge der Summe in höchstens {de(100 * p05, 0)} %. "
                    f"Platz 1 der gewöhnlichen Funktion und der Summe stimmen in {de(100 * a['same_top1'], 0)} % überein.")


def verdict_dial(rows: list[dict]) -> tuple[str, str]:
    m = C.EXP_M_MAIN
    u_g = _row(rows, decider="guete_linear", m=m, method="usual")["hit"]
    s_g = _row(rows, decider="guete_linear", m=m, method="summe")["hit"]
    u_r = _row(rows, decider="roh_linear", m=m, method="usual")["hit"]
    s_r = _row(rows, decider="roh_linear", m=m, method="summe")["hit"]
    return "warning", (f"Bei {m} Kriterien trifft das Zählen (gewöhnliche Funktion) Platz 1 eines Entscheiders, der linear im Rang im Kriterium bewertet, in {de(100 * u_g, 0)} % der Fälle "
                       f"(Summe {de(100 * s_g, 0)} %), eines Entscheiders in den Rohwerten in {de(100 * u_r, 0)} % (Summe {de(100 * s_r, 0)} %). Keine Funktion gewinnt in beiden Welten; "
                       "die Wahl der Funktion ist eine Annahme über den Entscheider.")


def verdict_partial(rows: list[dict]) -> tuple[str, str]:
    pf = [r for r in rows if not r["method"].startswith("ELECTRE")]
    e = next(r for r in rows if r["method"].startswith("ELECTRE"))
    return "info", (f"PROMETHEE I hat in keiner der {len(pf)} Konfigurationen einen Zyklus (Anteil {de(100 * max(r['cycle'] for r in pf), 1)} %) und {de(100 * min(r['r'] for r in pf), 1)} bis "
                    f"{de(100 * max(r['r'] for r in pf), 1)} % unvergleichbare Paare; der Sieger von PROMETHEE II wird nie übertroffen. ELECTRE I mit c* = 0,6 hat auf denselben Matrizen in "
                    f"{de(100 * e['cycle'], 0)} % einen Zyklus und {de(100 * e['r'], 0)} % unvergleichbare Paare.")


def verdict_dependence(add: list[dict], floor: list[dict], stab: list[dict]) -> tuple[str, str]:
    u = _row(add, method="usual")
    s = _row(add, method="Summe (Min-Max)")
    fl = [r["unacceptable"] for r in floor if r["method"] != "Summe"]
    top = max(r["sigma"] for r in stab)
    st_u = _row(stab, sigma=top, method="usual")["stay"]
    st_s = _row(stab, sigma=top, method="Summe")["stay"]
    return "warning", (f"Kommt eine Alternative dazu, kippt bei der gewöhnlichen Funktion in {de(100 * u['any_flip'], 0)} % der Matrizen mindestens ein Paar der alten (Summe: {de(100 * s['any_flip'], 0)} %), "
                       f"im Mittel {de(100 * u['flipped_pairs'], 1)} % der Paare (Summe: {de(100 * s['flipped_pairs'], 1)} %), und Platz 1 wechselt in {de(100 * u['top_changes'], 1)} % (Summe: {de(100 * s['top_changes'], 1)} %). Ohne Veto wählt PROMETHEE bei einer Untergrenze in "
                       f"{de(100 * min(fl), 0)} bis {de(100 * max(fl), 0)} % der Fälle Unzulässiges. Gegen Gewichtsstörung (σ = {de(top, 1)}) zeigt sich bei Platz 1 kein belastbarer Unterschied "
                       f"({de(100 * st_u, 0)} % gegen {de(100 * st_s, 0)} %).")
