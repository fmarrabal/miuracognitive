from goal_discovery_benchmark import _aggregate


def _cell(variant, seed, *, success, lesion, shuffle, reflection,
          transplant_switch=1.0, transplant_success=1.0,
          feasible=1.0, unique=1.0, probe_distance=0.2):
    normal = {
        "goal_success_rate": success,
        "mean_target_center_distance": 0.1,
        "mean_target_response": 0.98,
        "dominant_restored_rate": success,
        "mean_homeostatic_improvement": 0.2,
        "mean_final_absolute_error": 0.1,
        "feasible_goal_rate": feasible,
        "continuous_unique_goal_fraction": unique,
        "mean_normalized_probe_distance": probe_distance,
    }
    causal = {
        "feedback_on_goal_success": lesion,
        "valid_outcomes_on_goal_success": shuffle,
        "goal_content_on_success": reflection,
        "feedback_on_homeostatic_improvement": 0.1,
        "body_transplant_switch_rate": transplant_switch,
        "body_transplant_new_goal_success": transplant_success,
        "body_transplant_goal_shift": 0.3,
    }
    return {
        "variant": variant,
        "training": {"seed": seed},
        "conditions": {"normal": normal},
        "causal_effects": causal,
    }


def test_phase3_confirmatory_family_passes_effects_and_guardrails():
    results = []
    for seed in range(8):
        results.append(_cell(
            "discoverer", seed, success=0.9, lesion=0.5,
            shuffle=0.45, reflection=0.6))
        results.append(_cell(
            "no_feedback", seed, success=0.3, lesion=0.0,
            shuffle=0.0, reflection=0.2))
    family = _aggregate(results)["confirmatory_primary_family"]
    assert family["status"] == "pass"
    assert all(family["criterion_passed"].values())
    assert all(family["guardrails"].values())


def test_phase3_confirmatory_family_rejects_continuous_decoration():
    results = []
    for seed in range(8):
        results.append(_cell(
            "discoverer", seed, success=0.9, lesion=0.5,
            shuffle=0.45, reflection=0.6,
            unique=0.4 if seed == 0 else 1.0))
        results.append(_cell(
            "no_feedback", seed, success=0.3, lesion=0.0,
            shuffle=0.0, reflection=0.2))
    family = _aggregate(results)["confirmatory_primary_family"]
    assert family["status"] == "fail"
    assert not family["guardrails"][
        "continuous_unique_each_seed_at_least_0.95"]
