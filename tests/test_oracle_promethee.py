"""Orakel (DEMO-PLAYBOOK §12): unabhängige Rechenwege für den Nettofluss, die Identitäten (Borda, Perzentil-Summe, Summe), die Halbordnung und die Monotonie.

Jede Eigenschaft hat eine Gegenprobe, die zeigt, dass der Test etwas trennen kann (V-Form mit p = 0,5 ist nicht die Summe; ELECTRE hat Zyklen, PROMETHEE I nie)."""

import numpy as np
import pytest

import pmt_promethee as pm
import pmt_score as sc


def _case(rng, n=9, m=4):
    S = rng.uniform(1.0, 100.0, size=(n, m)) * rng.uniform(0.2, 40.0, size=m)
    return sc.minmax(S), S, rng.dirichlet(np.ones(m))


def _same_order(a, b):
    return np.array_equal(np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable"))


# ----------------------------------------------------------------------------- 1. Nettofluss gegen die Definition

@pytest.mark.parametrize("kind,kw", [("usual", {}), ("ushape", {"q": 0.1}), ("vshape", {"p": 0.5}), ("level", {"q": 0.1, "p": 0.5}), ("linear", {"q": 0.05, "p": 0.5}),
                                     ("gauss", {"s": 0.25})])
def test_flows_equal_the_definition_loop_and_sum_to_zero(kind, kw):
    rng = np.random.default_rng(1)
    for _ in range(20):
        N, _, w = _case(rng)
        n, m = N.shape
        P = {"usual": lambda d, **k: float(d > 0), "ushape": lambda d, q=0, **k: float(d > q), "vshape": lambda d, p=1, **k: min(1.0, max(0.0, d / p)),
             "level": lambda d, q=0, p=1, **k: 1.0 if d > p else 0.5 if d > q else 0.0,
             "linear": lambda d, q=0, p=1, **k: min(1.0, max(0.0, (d - q) / (p - q))), "gauss": lambda d, s=0.25, **k: 0.0 if d <= 0 else 1 - np.exp(-d * d / (2 * s * s))}[kind]
        pi = np.array([[0.0 if a == b else sum(w[j] * P(N[a, j] - N[b, j], **kw) for j in range(m)) for b in range(n)] for a in range(n)])
        plus, minus, phi = pm.flows(pm.preference_degrees(N, w, kind, **kw))
        assert plus == pytest.approx(pi.sum(axis=1) / (n - 1), abs=1e-12) and minus == pytest.approx(pi.sum(axis=0) / (n - 1), abs=1e-12)
        assert phi.sum() == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------- 2. Identitäten

def test_usual_is_the_weighted_borda_count_and_the_sum_of_percentile_ranks():
    rng = np.random.default_rng(2)
    for _ in range(100):
        N, S, w = _case(rng, n=int(rng.integers(4, 25)), m=int(rng.integers(2, 8)))
        phi = pm.flows(pm.preference_degrees(N, w, "usual"))[2]
        assert phi == pytest.approx(pm.weighted_borda(N, w), abs=1e-12)
        assert _same_order(phi, sc.percentile_normalize(S) @ w)
        assert phi == pytest.approx(2.0 * (sc.percentile_normalize(S) @ w) - 1.0, abs=1e-9)          # Phi = 2 * Perzentil-Summe - 1 (Summe w = 1)


def test_v_shape_with_the_full_range_is_the_weighted_sum_and_half_range_is_not():
    rng = np.random.default_rng(3)
    differing = 0
    for _ in range(200):
        N, _, w = _case(rng, n=12, m=5)
        ws = N @ w
        phi = pm.flows(pm.preference_degrees(N, w, "vshape", p=1.0))[2]
        assert phi == pytest.approx((len(N) * ws - ws.sum()) / (len(N) - 1), abs=1e-12) and _same_order(phi, ws)
        differing += not _same_order(pm.flows(pm.preference_degrees(N, w, "vshape", p=0.5))[2], ws)
    assert differing > 100


def test_a_function_that_is_larger_everywhere_never_lowers_a_preference_degree():
    rng = np.random.default_rng(4)
    for _ in range(30):
        N, _, w = _case(rng)
        low = pm.preference_degrees(N, w, "ushape", q=0.3)
        mid = pm.preference_degrees(N, w, "ushape", q=0.1)
        high = pm.preference_degrees(N, w, "usual")
        assert (mid >= low - 1e-12).all() and (high >= mid - 1e-12).all()


def test_scaling_all_weights_scales_the_flows_and_keeps_the_order():
    rng = np.random.default_rng(5)
    N, _, w = _case(rng)
    a = pm.flows(pm.preference_degrees(N, w, "vshape", p=0.5))[2]
    b = pm.flows(pm.preference_degrees(N, 3.0 * w, "vshape", p=0.5))[2]
    assert b == pytest.approx(3.0 * a, abs=1e-12) and _same_order(a, b)


# ----------------------------------------------------------------------------- 3. PROMETHEE I ist eine Halbordnung

@pytest.mark.parametrize("kind,kw", [("usual", {}), ("vshape", {"p": 0.5}), ("level", {"q": 0.1, "p": 0.5}), ("gauss", {"s": 0.25})])
def test_promethee_one_is_a_partial_order_without_cycles(kind, kw):
    rng = np.random.default_rng(6)
    for _ in range(40):
        N, _, w = _case(rng, n=10, m=5)
        plus, minus, phi = pm.flows(pm.preference_degrees(N, w, kind, **kw))
        P, I, R = pm.promethee_one(plus, minus)
        n = len(N)
        assert not P.diagonal().any() and not (P & P.T).any()                      # irreflexiv, asymmetrisch
        assert not ((P[:, :, None] & P[None, :, :]) & ~P[:, None, :]).any()         # transitiv
        assert not pm.has_cycle(P)
        assert (P | P.T | I | R | np.eye(n, dtype=bool)).all()                    # jedes Paar liegt in genau einer Klasse
        top = int(np.argmax(phi))
        assert not P[:, top].any()                                                  # der Sieger von II wird nie übertroffen
        for a, b in zip(*np.nonzero(P)):
            assert phi[a] > phi[b] - 1e-12


def test_electre_has_cycles_where_promethee_one_has_none():
    """Gegenprobe: dieselben Matrizen, ELECTRE I mit c* = 0,5 hat Zyklen, PROMETHEE I nie."""
    rng = np.random.default_rng(7)
    electre = promethee = 0
    for _ in range(60):
        N, _, w = _case(rng, n=10, m=5)
        electre += pm.has_cycle(pm.electre_reference(N, w, 0.5))
        plus, minus, _ = pm.flows(pm.preference_degrees(N, w, "usual"))
        promethee += pm.has_cycle(pm.promethee_one(plus, minus)[0])
    assert electre > 20 and promethee == 0


def test_ties_and_identical_alternatives_are_indifferent_not_incomparable():
    N = np.array([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    w = np.array([0.5, 0.5])
    plus, minus, _ = pm.flows(pm.preference_degrees(N, w, "vshape", p=1.0))
    P, I, R = pm.promethee_one(plus, minus)
    assert I[0, 1] and I[1, 0] and not R[0, 1]
