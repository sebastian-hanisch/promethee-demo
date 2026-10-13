"""Jede Zahl aus README.md und App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SWEEP_SEEDS (200 feste Matrizen), n = 20, nie über einen einzelnen Seed. Prozentwerte mit Toleranz 0,5 Punkte (kippende Paare 0,02), damit Rundung und
Bibliotheksversionen (CI installiert ungepinnt) nichts kippen. Der Standardfehler einer Trefferquote ist 1,3 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten sind NICHT belastbar
und stehen nirgends als Aussage.
"""

import pytest

import pmt_constants as C
import pmt_evaluation as ev


@pytest.fixture(scope="module")
def identities():
    return ev.experiment_identities()


@pytest.fixture(scope="module")
def agreement():
    return ev.experiment_agreement()


@pytest.fixture(scope="module")
def dial():
    return ev.experiment_dial()


@pytest.fixture(scope="module")
def partial():
    return ev.experiment_partial()


@pytest.fixture(scope="module")
def addition():
    return ev.experiment_addition()


@pytest.fixture(scope="module")
def floor():
    return ev.experiment_floor()


@pytest.fixture(scope="module")
def stability():
    return ev.experiment_stability()


def cell(rows, **keys):
    hits = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(hits) == 1, (keys, len(hits))
    return hits[0]


def pct(x):
    return 100 * x


# ----------------------------------------------------------------------------- Experiment 1: Identitäten

def test_the_two_identities_hold_in_every_case(identities):
    assert sum(r["cases"] for r in identities) == 600
    for r in identities:
        assert r["borda"] == 1.0 and r["percentile"] == 1.0 and r["vsum"] == 1.0
        assert r["diff_borda"] < 1e-12 and r["diff_vsum"] < 1e-12


def test_half_range_v_shape_is_rarely_the_ranking_of_the_sum(identities):
    assert [round(pct(r["vsum_p05"]), 0) for r in sorted(identities, key=lambda r: r["m"])] == [6.0, 2.0, 1.0]


@pytest.mark.parametrize("a,b,top1,tau", [
    ("usual", "vshape(p=1,0)", 64.5, 0.82), ("vshape(p=0,25)", "vshape(p=0,5)", 90.0, 0.94), ("vshape(p=0,5)", "vshape(p=1,0)", 90.0, 0.95),
    ("vshape(p=0,25)", "vshape(p=1,0)", 85.5, 0.91), ("linear(q=0,05, p=0,5)", "vshape(p=0,5)", 95.5, 0.98), ("usual", "ushape(q=0,1)", 69.5, 0.87),
])
def test_agreement_between_functions(agreement, a, b, top1, tau):
    r = cell(agreement, a=a, b=b)
    assert pct(r["same_top1"]) == pytest.approx(top1, abs=0.5) and r["tau"] == pytest.approx(tau, abs=0.01)


def test_usual_and_the_sum_agree_on_first_place_in_about_two_thirds(agreement):
    assert pct(cell(agreement, a="usual", b="Summe")["same_top1"]) == pytest.approx(64.5, abs=0.5)


# ----------------------------------------------------------------------------- Experiment 2: der Regler

DIAL = {
    ("guete_linear", 3): {"summe": 85.0, "usual": 76.5, "ushape(q=0,1)": 80.0, "level(q=0,1, p=0,5)": 82.0, "linear(q=0,05, p=0,5)": 81.5, "vshape(p=0,25)": 83.5,
                          "vshape(p=0,5)": 86.5, "vshape(p=1,0)": 85.0, "gauss(s=0,25)": 82.0},
    ("guete_linear", 5): {"summe": 67.0, "usual": 82.5, "ushape(q=0,1)": 71.5, "level(q=0,1, p=0,5)": 67.5, "linear(q=0,05, p=0,5)": 66.0, "vshape(p=0,25)": 77.0,
                          "vshape(p=0,5)": 69.5, "vshape(p=1,0)": 67.0, "gauss(s=0,25)": 65.5},
    ("guete_linear", 8): {"summe": 78.5, "usual": 82.5, "ushape(q=0,1)": 76.5, "level(q=0,1, p=0,5)": 77.0, "linear(q=0,05, p=0,5)": 77.0, "vshape(p=0,25)": 82.0,
                          "vshape(p=0,5)": 79.0, "vshape(p=1,0)": 78.5, "gauss(s=0,25)": 76.5},
    ("guete_konkav", 5): {"summe": 72.5, "usual": 76.0, "ushape(q=0,1)": 73.0, "level(q=0,1, p=0,5)": 71.0, "linear(q=0,05, p=0,5)": 71.0, "vshape(p=0,25)": 79.0,
                          "vshape(p=0,5)": 75.0, "vshape(p=1,0)": 72.5, "gauss(s=0,25)": 71.5},
    ("guete_cobb", 5): {"summe": 69.0, "usual": 69.5, "ushape(q=0,1)": 71.0, "level(q=0,1, p=0,5)": 67.0, "linear(q=0,05, p=0,5)": 70.0, "vshape(p=0,25)": 74.5,
                        "vshape(p=0,5)": 72.5, "vshape(p=1,0)": 69.0, "gauss(s=0,25)": 70.0},
    ("roh_linear", 3): {"summe": 94.0, "usual": 75.5, "ushape(q=0,1)": 76.0, "level(q=0,1, p=0,5)": 84.5, "linear(q=0,05, p=0,5)": 87.0, "vshape(p=0,25)": 83.5,
                        "vshape(p=0,5)": 90.5, "vshape(p=1,0)": 94.0, "gauss(s=0,25)": 89.0},
    ("roh_linear", 5): {"summe": 96.5, "usual": 64.5, "ushape(q=0,1)": 82.5, "level(q=0,1, p=0,5)": 88.0, "linear(q=0,05, p=0,5)": 90.5, "vshape(p=0,25)": 85.5,
                        "vshape(p=0,5)": 91.0, "vshape(p=1,0)": 96.5, "gauss(s=0,25)": 92.5},
    ("roh_linear", 8): {"summe": 93.0, "usual": 67.0, "ushape(q=0,1)": 75.0, "level(q=0,1, p=0,5)": 87.5, "linear(q=0,05, p=0,5)": 89.5, "vshape(p=0,25)": 86.0,
                        "vshape(p=0,5)": 89.0, "vshape(p=1,0)": 93.0, "gauss(s=0,25)": 89.5},
}


@pytest.mark.parametrize("key", sorted(DIAL))
def test_first_place_for_every_function(dial, key):
    dec, m = key
    for method, hit in DIAL[key].items():
        assert pct(cell(dial, decider=dec, m=m, method=method)["hit"]) == pytest.approx(hit, abs=0.5), method


def test_the_full_range_v_shape_hits_exactly_as_often_as_the_sum(dial):
    for dec in C.EXP_DECIDERS:
        for m in C.EXP_M:
            assert cell(dial, decider=dec, m=m, method="vshape(p=1,0)")["hit"] == cell(dial, decider=dec, m=m, method="summe")["hit"]


def test_counting_wins_in_the_rank_world_and_summing_in_the_raw_world(dial):
    g = pct(cell(dial, decider="guete_linear", m=5, method="usual")["hit"] - cell(dial, decider="guete_linear", m=5, method="summe")["hit"])
    r = pct(cell(dial, decider="roh_linear", m=5, method="summe")["hit"] - cell(dial, decider="roh_linear", m=5, method="usual")["hit"])
    assert g == pytest.approx(15.5, abs=0.5) and r == pytest.approx(32.0, abs=0.5)


def test_the_v_shape_moves_monotonically_between_counting_and_summing_in_the_raw_world(dial):
    for m in C.EXP_M:
        h = [cell(dial, decider="roh_linear", m=m, method=k)["hit"] for k in ("usual", "vshape(p=0,25)", "vshape(p=0,5)", "vshape(p=1,0)")]
        assert h == sorted(h)


# ----------------------------------------------------------------------------- Experiment 3: Teilordnung

PARTIAL = {   # (strikt %, unvergleichbar %, Bester nicht übertroffen %)
    "usual": (99.9, 0.1, 82.5), "ushape(q=0,1)": (90.0, 10.0, 85.0), "level(q=0,1, p=0,5)": (87.5, 12.5, 82.0), "linear(q=0,05, p=0,5)": (87.8, 12.2, 78.5),
    "vshape(p=0,25)": (91.5, 8.5, 86.5), "vshape(p=0,5)": (89.1, 10.9, 81.5), "vshape(p=1,0)": (85.5, 14.5, 81.0), "gauss(s=0,25)": (87.2, 12.8, 78.0),
}


@pytest.mark.parametrize("method", sorted(PARTIAL))
def test_partial_order_shares(partial, method):
    p, r, free = PARTIAL[method]
    row = cell(partial, method=method)
    assert pct(row["p"]) == pytest.approx(p, abs=0.5) and pct(row["r"]) == pytest.approx(r, abs=0.5) and pct(row["best_free"]) == pytest.approx(free, abs=0.5)
    assert row["i"] == 0.0 and row["cycle"] == 0.0 and row["winner_outranked"] == 0.0


def test_electre_reference_on_the_same_matrices(partial):
    e = cell(partial, method="ELECTRE I (c* = 0,6, kein Veto)")
    assert pct(e["cycle"]) == pytest.approx(38.5, abs=0.5) and pct(e["r"]) == pytest.approx(20.4, abs=0.5) and pct(e["p"]) == pytest.approx(79.6, abs=0.5)


# ----------------------------------------------------------------------------- Experiment 4: Abhängigkeit, Untergrenze, Stabilität

ADD = {"Summe (Min-Max)": (0.20, 13.5, 0.5), "usual": (1.13, 89.0, 5.0), "vshape(p=0,5)": (0.46, 52.0, 3.5), "vshape(p=1,0)": (0.20, 13.5, 0.5),
       "linear(q=0,05, p=0,5)": (0.45, 48.0, 2.5)}


@pytest.mark.parametrize("method", sorted(ADD))
def test_adding_one_alternative(addition, method):
    pairs, anyf, top = ADD[method]
    row = cell(addition, method=method)
    assert pct(row["flipped_pairs"]) == pytest.approx(pairs, abs=0.02) and pct(row["any_flip"]) == pytest.approx(anyf, abs=0.5) and pct(row["top_changes"]) == pytest.approx(top, abs=0.5)


def test_the_full_range_v_shape_flips_exactly_as_often_as_the_sum(addition):
    assert cell(addition, method="vshape(p=1,0)")["any_flip"] == cell(addition, method="Summe (Min-Max)")["any_flip"]
    assert cell(addition, method="usual")["any_flip"] > 5 * cell(addition, method="Summe (Min-Max)")["any_flip"]


FLOOR = {"Summe": (11.0, 64.5), "usual": (18.0, 71.0), "vshape(p=0,5)": (12.5, 66.0), "vshape(p=0,25)": (14.0, 71.5), "gauss(s=0,25)": (12.5, 62.0)}


@pytest.mark.parametrize("method", sorted(FLOOR))
def test_choices_under_the_floor(floor, method):
    bad, hit = FLOOR[method]
    row = cell(floor, method=method)
    assert pct(row["unacceptable"]) == pytest.approx(bad, abs=0.5) and pct(row["hit"]) == pytest.approx(hit, abs=0.5)
    assert pct(row["unacceptable_share"]) == pytest.approx(40.9, abs=0.1)


STAB = {0.2: {"Summe": 90.3, "usual": 88.8, "vshape(p=0,5)": 90.2, "vshape(p=1,0)": 90.3}, 0.4: {"Summe": 80.8, "usual": 78.9, "vshape(p=0,5)": 81.7, "vshape(p=1,0)": 80.8}}


@pytest.mark.parametrize("sigma", sorted(STAB))
def test_stability_of_first_place(stability, sigma):
    for method, v in STAB[sigma].items():
        assert pct(cell(stability, sigma=sigma, method=method)["stay"]) == pytest.approx(v, abs=0.6), method


def test_the_full_range_v_shape_is_exactly_as_stable_as_the_sum(stability):
    for sg in C.EXP_SIGMAS:
        assert cell(stability, sigma=sg, method="vshape(p=1,0)")["stay"] == cell(stability, sigma=sg, method="Summe")["stay"]


def test_experiment_sizes_are_the_documented_ones():
    assert len(C.SWEEP_SEEDS) == 200 and C.EXP_N == 20 and C.ELECTRE_CSTAR == 0.6
