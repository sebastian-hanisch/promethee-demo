"""PROMETHEE-Bausteine: Präferenzfunktionen, Präferenzgrade, Flüsse, PROMETHEE I, Borda-Zählung, Zyklen (je Einheit gegen Handwerte der Mini-Instanzen)."""

import numpy as np
import pytest

import pmt_constants as C
import pmt_promethee as pm


def test_preference_functions_hand_values():
    d = np.array([-0.2, 0.0, 0.05, 0.1, 0.3, 0.5, 0.6, 1.0])
    assert pm.preference("usual", d).tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    assert pm.preference("ushape", d, q=0.1).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]            # erst über q
    assert pm.preference("vshape", d, p=0.5).tolist() == pytest.approx([0, 0, 0.1, 0.2, 0.6, 1, 1, 1])
    assert pm.preference("level", d, q=0.1, p=0.5).tolist() == [0, 0, 0, 0, 0.5, 0.5, 1, 1]
    assert pm.preference("linear", d, q=0.1, p=0.5).tolist() == pytest.approx([0, 0, 0, 0, 0.5, 1, 1, 1])
    g = pm.preference("gauss", np.array([-0.1, 0.0, 0.25, 1.0]), s=0.25)
    assert g[0] == 0.0 and g[1] == 0.0 and g[2] == pytest.approx(1 - np.exp(-0.5)) and g[3] == pytest.approx(1 - np.exp(-8.0))
    with pytest.raises(ValueError):
        pm.preference("sinus", d)


def test_preference_functions_are_zero_below_zero_and_monotone_and_bounded():
    d = np.linspace(-1, 1, 401)
    for kind, kw in (("usual", {}), ("ushape", {"q": 0.1}), ("vshape", {"p": 0.5}), ("level", {"q": 0.1, "p": 0.5}), ("linear", {"q": 0.05, "p": 0.5}), ("gauss", {"s": 0.25})):
        v = pm.preference(kind, d, **kw)
        assert (v[d <= 0] == 0).all() and (np.diff(v) >= -1e-12).all() and v.min() >= 0 and v.max() <= 1


def test_preference_degrees_and_flows_usual(abcd):
    N, w = abcd
    pi = pm.preference_degrees(N, w, "usual")
    assert pi[0, 1] == pytest.approx(0.7) and pi[1, 0] == pytest.approx(0.3) and pi[1, 2] == pytest.approx(0.5) and pi[2, 1] == pytest.approx(0.5)
    assert np.diag(pi).tolist() == [0, 0, 0, 0] and pi[0, 3] == pytest.approx(1.0) and pi[3, 0] == 0.0
    plus, minus, phi = pm.flows(pi)
    assert plus == pytest.approx([0.9, 0.6, 0.5, 0.0]) and minus == pytest.approx([0.1, 0.4, 0.5, 1.0]) and phi == pytest.approx([0.8, 0.2, 0.0, -1.0], abs=1e-12)
    assert phi.sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("kind,kw,expected", [
    ("vshape", {"p": 1.0}, [0.48, 0.2133, -0.0533, -0.64]),
    ("vshape", {"p": 0.5}, [0.6867, 0.3333, -0.0933, -0.9267]),
    ("linear", {"q": 0.1, "p": 0.5}, [0.6583, 0.3667, -0.1167, -0.9083]),
])
def test_net_flow_hand_values(abcd, kind, kw, expected):
    N, w = abcd
    phi = pm.flows(pm.preference_degrees(N, w, kind, **kw))[2]
    assert phi == pytest.approx(expected, abs=5e-5) and phi.sum() == pytest.approx(0.0, abs=1e-12)
    assert pm.net_flow_order(phi).tolist() == [0, 1, 2, 3]


def test_v_shape_with_full_range_is_an_affine_function_of_the_sum(abcd):
    N, w = abcd
    phi = pm.flows(pm.preference_degrees(N, w, "vshape", p=1.0))[2]
    ws = N @ w
    assert phi == pytest.approx((len(N) * ws - ws.sum()) / (len(N) - 1))


def test_weighted_borda_equals_the_usual_net_flow(abcd, xyz):
    N, w = abcd
    assert pm.weighted_borda(N, w) == pytest.approx([0.8, 0.2, 0.0, -1.0], abs=1e-12)
    assert pm.weighted_borda(N, w) == pytest.approx(pm.flows(pm.preference_degrees(N, w, "usual"))[2])
    X, wx = xyz
    assert pm.weighted_borda(X, wx) == pytest.approx([0.0, 0.0, 0.0])


def test_counting_ties_and_summing_separate_on_the_xyz_instance(xyz):
    X, w = xyz
    assert pm.flows(pm.preference_degrees(X, w, "usual"))[2] == pytest.approx([0.0, 0.0, 0.0])
    phi = pm.flows(pm.preference_degrees(X, w, "vshape", p=1.0))[2]
    assert phi == pytest.approx([-0.05, -0.05, 0.10]) and int(np.argmax(phi)) == 2


def test_promethee_one_on_the_incomparable_instance(incomparable):
    N, w = incomparable
    plus, minus, phi = pm.flows(pm.preference_degrees(N, w, "vshape", p=0.5))
    assert plus == pytest.approx([0.4667, 0.4667, 0.4267, 0.3067], abs=5e-5) and minus == pytest.approx([0.3867, 0.3867, 0.4667, 0.4267], abs=5e-5)
    assert phi == pytest.approx([0.08, 0.08, -0.04, -0.12], abs=5e-5)
    P, I, R = pm.promethee_one(plus, minus)
    assert P.astype(int).tolist() == [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert I.astype(int).tolist() == [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert R.astype(int).tolist() == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]


def test_promethee_one_orders_pairs_that_both_flows_agree_on():
    plus = np.array([0.9, 0.5, 0.5])
    minus = np.array([0.1, 0.5, 0.3])
    P, I, R = pm.promethee_one(plus, minus)
    assert P[0, 1] and P[0, 2] and P[2, 1] and not P[1, 0] and not I.any() and not R.any()


def test_promethee_one_with_one_flow_exactly_tied():
    """Gleicher Einfluss Phi- bei größerem Phi+: a übertrifft b (Gleichheit zählt als 'höchstens'), das Paar ist NICHT unvergleichbar.
    Fehlt die Toleranz auf der richtigen Seite (`<= b - eps` statt `<= b + eps`), fiele ein exakter Gleichstand durch und das Paar wäre unvergleichbar."""
    plus = np.array([0.6, 0.2])
    minus = np.array([0.3, 0.3])
    P, I, R = pm.promethee_one(plus, minus)
    assert P.astype(int).tolist() == [[0, 1], [0, 0]]
    assert not I.any() and not R.any()
    P, I, R = pm.promethee_one(np.array([0.4, 0.4]), np.array([0.1, 0.5]))       # Gleichstand in Phi+, Phi- verschieden
    assert P.astype(int).tolist() == [[0, 1], [0, 0]] and not I.any() and not R.any()


def test_flows_for_a_single_alternative_are_zero():
    plus, minus, phi = pm.flows(np.zeros((1, 1)))
    assert plus.tolist() == minus.tolist() == phi.tolist() == [0.0]


def test_has_cycle():
    A = np.zeros((4, 4), dtype=bool)
    A[0, 1] = A[1, 2] = True
    assert not pm.has_cycle(A)
    A[2, 0] = True
    assert pm.has_cycle(A)


def test_electre_reference_matches_the_concordance_threshold(abcd):
    N, w = abcd
    S = pm.electre_reference(N, w, 0.6)
    assert S.astype(int).tolist() == [[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    assert pm.electre_reference(N, w, 0.7)[0, 1]                      # c(A, B) = 0,7: inklusive
    assert not pm.electre_reference(N, w, 0.71)[0, 1]


def test_uses_reports_the_parameters_of_each_function():
    assert [pm.uses(k, "q") for k in C.KINDS] == [False, True, False, True, True, False]
    assert [pm.uses(k, "p") for k in C.KINDS] == [False, False, True, True, True, False]
    assert [pm.uses(k, "s") for k in C.KINDS] == [False, False, False, False, False, True]


def test_flows_for_exactly_two_alternatives():
    """Mutationstest: Der Zweig n < 2 darf bei genau zwei Alternativen nicht greifen (Division durch n - 1 = 1)."""
    pi = np.array([[0.0, 0.7], [0.2, 0.0]])
    plus, minus, phi = pm.flows(pi)
    assert plus.tolist() == [0.7, 0.2] and minus.tolist() == [0.2, 0.7] and phi == pytest.approx([0.5, -0.5])
    N = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert pm.flows(pm.preference_degrees(N, np.array([0.6, 0.4]), "usual"))[2] == pytest.approx([0.2, -0.2])
