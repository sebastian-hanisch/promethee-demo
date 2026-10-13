"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import numpy as np

import pmt_constants as C
import pmt_evaluation as ev
import pmt_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["kind"] in C.KINDS and p["q"] in C.Q_CHOICES and p["p"] in C.P_CHOICES and p["s"] in C.S_CHOICES, name
        assert p["decider"] in C.DECIDERS and p["delta"] in C.DELTA_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (V-Form, p = 0,5)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice_and_every_configuration_is_valid():
    assert set(C.KIND_LABELS) == set(C.KINDS) == set(C.KIND_SHORT) == set(C.KIND_PARAMS)
    assert set(C.DECIDER_LABELS) == set(C.DECIDERS) == set(C.DECIDER_SHORT) == set(C.DECIDER_PARTS)
    for name, kind, q, p, s in C.CONFIGS:
        assert kind in C.KINDS and 0 <= q < p <= 1.0 and s > 0, name
    assert len({c[0] for c in C.CONFIGS}) == len(C.CONFIGS)


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis("Standardfall (V-Form, p = 0,5)")
    assert a.results["promethee"].hit and a.results["summe"].hit and a.R.sum() >= 2

    a = _analysis("Zählen")
    assert a.settings.kind == "usual" and a.results["promethee"].hit and not a.results["summe"].hit
    assert np.allclose(a.results["borda"].score, a.phi)

    a = _analysis("Summieren")
    b = _analysis("Summieren", kind="usual")
    assert a.settings.p == 1.0 and a.results["promethee"].hit and not b.results["promethee"].hit
    assert np.array_equal(a.order, np.argsort(-a.results["summe"].score, kind="stable"))

    a = _analysis("Zwei ohne Urteil")
    assert a.R.sum() == 2 and a.results["promethee"].hit
    i, j = (int(x) for x in np.argwhere(a.R)[0])
    assert a.plus[i] != a.plus[j] and (a.plus[i] - a.plus[j]) * (a.minus[i] - a.minus[j]) > 0     # Plusfluss größer und Minusfluss größer: widersprüchlich
    assert a.phi[i] != a.phi[j]                                                                     # PROMETHEE II ordnet das Paar trotzdem
