from endogenous_goals_benchmark import _aggregate


def _cell(variant, seed, *, mild, lesion, rotation, crisis=1.0):
    normal = {
        "survival_rate": 1.0,
        "violation_rate": 0.0,
        "mean_absolute_homeostatic_error": 0.1,
        "mean_project_completions": 1.5,
        "initial_goal_selection_rate": 1.0,
        "initial_action_selection_rate": 1.0,
        "mild_commitment_rate": mild,
        "mild_goal_persistence_rate": mild,
        "mild_completion_rate": mild,
        "critical_switch_rate": crisis,
        "critical_goal_switch_rate": crisis,
        "critical_rescue_rate": crisis,
        "work_switch_rate": 0.3,
        "goal_action_alignment_rate": 1.0,
        "minor_alias_max_abs_gap": 1e-8,
        "minor_update_gate_at_conflict": 0.02,
        "critical_update_gate_at_conflict": 0.82,
    }
    causal = {
        "memory_on_mild_completion": lesion,
        "memory_on_mild_commitment": lesion,
        "memory_on_survival": 0.0,
        "goal_content_on_minor_completion": rotation,
        "goal_rotation_follow_rate": 0.5,
        "goal_rotation_changed_action_rate": 0.5,
        "body_transplant_goal_selection_rate": 1.0,
        "body_transplant_action_selection_rate": 1.0,
    }
    return {
        "variant": variant,
        "training": {"seed": seed},
        "conditions": {"normal": normal},
        "causal_effects": causal,
    }


def test_phase2_confirmatory_family_passes_only_with_effects_and_guardrails():
    results = []
    for seed in range(8):
        results.append(_cell(
            "goal_memory", seed, mild=1.0, lesion=0.5, rotation=0.5))
        results.append(_cell(
            "reactive", seed, mild=0.5, lesion=0.0, rotation=0.1))
    family = _aggregate(results)["confirmatory_primary_family"]
    assert family["status"] == "pass"
    assert all(family["criterion_passed"].values())
    assert all(family["guardrails"].values())


def test_phase2_confirmatory_family_fails_blind_perseveration_guardrail():
    results = []
    for seed in range(8):
        results.append(_cell(
            "goal_memory", seed, mild=1.0, lesion=0.5,
            rotation=0.5, crisis=0.8 if seed == 0 else 1.0))
        results.append(_cell(
            "reactive", seed, mild=0.5, lesion=0.0, rotation=0.1))
    family = _aggregate(results)["confirmatory_primary_family"]
    assert family["status"] == "fail"
    assert not family["guardrails"][
        "critical_rescue_each_seed_at_least_0.95"]
