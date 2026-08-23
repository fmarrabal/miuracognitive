from aha_benchmark import _aggregate


def _cell(variant, seed, survival, blank, lesion, cue):
    metrics = {
        "survival_rate": survival,
        "violation_rate": 0.01,
        "mean_homeostatic_deficit": 0.02,
        "mean_absolute_homeostatic_error": 0.03,
        "anticipatory_event_rate": blank,
        "blank_initiative_rate": blank,
        "action_rate": 0.2,
        "false_action_fraction": 0.1,
        "prediction_mse": 0.01,
        "prediction_event_contrast": 0.2,
        "prediction_target_correlation": 0.9,
    }
    return {
        "variant": variant,
        "training": {"seed": seed},
        "conditions": {"normal": metrics},
        "causal_effects": {
            "prediction_on_survival": lesion,
            "prediction_on_violation_rate": lesion,
            "prediction_on_blank_initiative": lesion,
            "valid_cue_on_survival": cue,
            "valid_cue_on_blank_initiative": cue,
            "need_transplant": {
                "mean_target_probability_effect": 0.2,
                "mean_goal_flip_rate": 1.0,
            },
        },
    }


def test_confirmatory_family_is_paired_holm_corrected_and_directional():
    results = []
    for seed in range(8):
        results.append(_cell(
            "gating_wm", seed, survival=0.9, blank=0.8,
            lesion=0.7, cue=0.6))
        results.append(_cell(
            "reactive", seed, survival=0.3, blank=0.2,
            lesion=0.0, cue=0.0))
    family = _aggregate(results)["confirmatory_primary_family"]
    assert family["n_paired_seeds"] == 8
    assert family["status"] == "pass"
    assert all(family["criterion_passed"].values())
    assert set(family["holm_adjusted_p"]) == {
        "gating_minus_reactive_survival",
        "gating_minus_reactive_blank_initiative",
        "prediction_lesion_on_gating_survival",
        "valid_cue_on_gating_survival",
    }


def test_confirmatory_family_fails_if_one_effect_has_wrong_direction():
    results = []
    for seed in range(8):
        results.append(_cell(
            "gating_wm", seed, survival=0.9, blank=0.1,
            lesion=0.7, cue=0.6))
        results.append(_cell(
            "reactive", seed, survival=0.3, blank=0.2,
            lesion=0.0, cue=0.0))
    family = _aggregate(results)["confirmatory_primary_family"]
    assert family["status"] == "fail"
    assert not family["criterion_passed"][
        "gating_minus_reactive_blank_initiative"]
