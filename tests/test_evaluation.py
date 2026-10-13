"""Auswertung: Fall der App (Invarianten), Entscheider mit Untergrenze, Experimente im Kleinen, Verdikte."""

import numpy as np
import pytest

import pmt_constants as C
import pmt_evaluation as ev
import pmt_promethee as pm
import pmt_score as sc


def test_make_case_is_deterministic_and_the_floor_decider_ranks_unacceptable_last():
    a, b = ev.make_case(20, 5, 7), ev.make_case(20, 5, 7)
    assert (a.S == b.S).all() and (a.w == b.w).all() and a.utility == pytest.approx(a.goodness @ a.w)
    f = ev.make_case(20, 5, 7, 1.0, "untergrenze")
    ok = f.acceptable
    assert ok.any() and (~ok).any() and f.utility[ok].min() > f.utility[~ok].max()
    assert f.acceptable.tolist() == (f.goodness >= C.THETA).all(axis=1).tolist()


def test_analyse_basic_invariants():
    a = ev.analyse(ev.Settings())
    n = a.settings.n
    assert a.pi.shape == (n, n) and not a.pi.diagonal().any() and a.phi.sum() == pytest.approx(0.0, abs=1e-12)
    assert a.phi == pytest.approx(a.plus - a.minus) and a.order.tolist() == pm.net_flow_order(a.phi).tolist()
    assert a.P.sum() + a.I.sum() / 2 + a.R.sum() / 2 == n * (n - 1) / 2
    assert a.best == int(np.argmax(a.case.utility)) and a.best_free == (not a.P[:, a.best].any())
    assert set(a.results) == {"summe", "promethee", "borda"} and a.results["promethee"].top1 == int(a.order[0])
    for r in a.results.values():
        assert r.hit == (r.top1 == a.best) and r.loss >= 0 and -1 <= r.tau <= 1


def test_the_usual_function_gives_the_borda_ranking_in_the_app_case():
    for seed in range(1, 12):
        a = ev.analyse(ev.Settings(kind="usual", seed=seed))
        assert a.results["borda"].score == pytest.approx(a.phi, abs=1e-12)


def test_the_full_range_v_shape_gives_the_ranking_of_the_sum_in_the_app_case():
    for seed in range(1, 12):
        a = ev.analyse(ev.Settings(kind="vshape", p=1.0, seed=seed))
        assert np.array_equal(a.order, np.argsort(-a.results["summe"].score, kind="stable"))


def test_the_winner_of_promethee_two_is_never_outranked_in_promethee_one():
    for seed in range(1, 40):
        for kind in C.KINDS:
            a = ev.analyse(ev.Settings(kind=kind, seed=seed))
            assert not a.P[:, a.order[0]].any()


def test_config_scores_cover_every_configuration(abcd):
    N, w = abcd
    scores = ev.config_scores(N, w)
    assert list(scores) == [c[0] for c in C.CONFIGS]
    assert scores["usual"] == pytest.approx([0.8, 0.2, 0.0, -1.0], abs=1e-12)


def test_experiment_identities_small_run():
    rows = ev.experiment_identities(ms=(3,), seeds=tuple(range(200000, 200012)))
    r = rows[0]
    assert r["borda"] == r["percentile"] == r["vsum"] == 1.0 and r["vsum_p05"] < 1.0 and r["diff_borda"] < 1e-12 and r["diff_vsum"] < 1e-12 and r["cases"] == 12


def test_experiment_agreement_and_dial_small_runs():
    seeds = tuple(range(200000, 200010))
    agree = ev.experiment_agreement(seeds=seeds)
    assert len(agree) == len(ev.AGREEMENT_PAIRS) + 1 and all(0 <= r["same_top1"] <= 1 for r in agree)
    dial = ev.experiment_dial(deciders=("roh_linear",), ms=(3,), seeds=seeds)
    assert {r["method"] for r in dial} == {"summe"} | {c[0] for c in C.CONFIGS}
    s = next(r for r in dial if r["method"] == "summe")["hit"]
    assert next(r for r in dial if r["method"] == "vshape(p=1,0)")["hit"] == s


def test_experiment_partial_small_run_has_no_cycles():
    rows = ev.experiment_partial(seeds=tuple(range(200000, 200010)))
    pf = [r for r in rows if not r["method"].startswith("ELECTRE")]
    assert len(pf) == len(C.CONFIGS) and all(r["cycle"] == 0.0 and r["winner_outranked"] == 0.0 and r["i"] == 0.0 for r in pf)
    assert rows[-1]["method"].startswith("ELECTRE") and 0 <= rows[-1]["cycle"] <= 1


def test_experiment_addition_floor_and_stability_small_runs():
    seeds = tuple(range(200000, 200010))
    add = ev.experiment_addition(seeds=seeds)
    full = next(r for r in add if r["method"] == "vshape(p=1,0)")
    sm = next(r for r in add if r["method"] == "Summe (Min-Max)")
    assert full == {**sm, "method": "vshape(p=1,0)"}
    fl = ev.experiment_floor(seeds=seeds)
    assert len(fl) == len(ev.FLOOR_CONFIGS) and all(0 <= r["unacceptable"] <= 1 for r in fl)
    st = ev.experiment_stability(sigmas=(0.0,), seeds=seeds, draws=3)
    assert all(r["stay"] == 1.0 for r in st)


def test_experiments_are_deterministic():
    seeds = tuple(range(200000, 200006))
    assert ev.experiment_stability(sigmas=(0.2,), seeds=seeds, draws=3) == ev.experiment_stability(sigmas=(0.2,), seeds=seeds, draws=3)


def test_verdicts_read_their_numbers_from_the_rows():
    ident = [{"m": m, "borda": 1.0, "percentile": 1.0, "vsum": 1.0, "vsum_p05": 0.06, "diff_borda": 0.0, "diff_vsum": 0.0, "cases": 200} for m in (3, 5, 8)]
    agree = [{"a": "usual", "b": "Summe", "same_top1": 0.645, "tau": float("nan")}]
    kind, text = ev.verdict_identities(ident, agree)
    assert kind == "info" and "100 %" in text and "600 Fällen" in text and "6 %" in text and "64 %" in text
    dial = [{"decider": d, "m": 5, "method": k, "hit": h} for d, k, h in (("guete_linear", "usual", 0.825), ("guete_linear", "summe", 0.67), ("roh_linear", "usual", 0.645), ("roh_linear", "summe", 0.965))]
    kind, text = ev.verdict_dial(dial)
    assert kind == "warning" and "82 %" in text and "67 %" in text and "64 %" in text and "96 %" in text and "Keine Funktion" in text
    part = [{"method": "usual", "p": 0.999, "i": 0.0, "r": 0.001, "cycle": 0.0, "winner_outranked": 0.0, "best_free": 0.8},
            {"method": "vshape", "p": 0.855, "i": 0.0, "r": 0.145, "cycle": 0.0, "winner_outranked": 0.0, "best_free": 0.8},
            {"method": "ELECTRE I (c* = 0,6, kein Veto)", "p": 0.8, "i": 0.0, "r": 0.204, "cycle": 0.385, "winner_outranked": float("nan"), "best_free": float("nan")}]
    kind, text = ev.verdict_partial(part)
    assert kind == "info" and "0,0 %" in text and "0,1 bis 14,5 %" in text and "38 %" in text and "20 %" in text
    add = [{"method": "Summe (Min-Max)", "flipped_pairs": 0.002, "any_flip": 0.135, "top_changes": 0.005}, {"method": "usual", "flipped_pairs": 0.0113, "any_flip": 0.89, "top_changes": 0.05}]
    floor = [{"method": "Summe", "unacceptable": 0.11, "hit": 0.6, "unacceptable_share": 0.4}, {"method": "usual", "unacceptable": 0.18, "hit": 0.7, "unacceptable_share": 0.4},
             {"method": "vshape(p=0,5)", "unacceptable": 0.125, "hit": 0.6, "unacceptable_share": 0.4}]
    stab = [{"sigma": 0.4, "method": "usual", "stay": 0.789}, {"sigma": 0.4, "method": "Summe", "stay": 0.808}]
    kind, text = ev.verdict_dependence(add, floor, stab)
    assert kind == "warning" and "89 %" in text and "14 %" in text and "5,0 %" in text and "12 bis 18 %" in text and "79 %" in text and "81 %" in text
