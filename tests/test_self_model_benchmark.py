from self_model_benchmark import _aggregate


def _cell(variant, seed, *, recovery, lesion, shuffle, rotation,
          pre=1.0, backup=1.0, prediction=1e-5,
          transplant=1.0, transplant_switch=1.0, radius=0.95):
    normal = {
        "pre_damage_recovery_rate": pre,
        "post_damage_recovery_rate": recovery,
        "mean_recovery_latency": 2.0,
        "backup_switch_rate": backup,
        "failed_primary_repeat_rate": 0.0,
        "mean_post_damage_target_error": 0.1,
        "post_damage_viability_rate": 1.0,
        "post_adaptation_prediction_mse": prediction,
        "post_adaptation_oracle_match_rate": 1.0,
        "final_damaged_actuator_model_mse": 0.0,
        "passive_spectral_radius": radius,
    }
    lesion_condition = normal | {
        "post_damage_recovery_rate": recovery - lesion}
    causal = {
        "model_update_on_recovery": lesion,
        "valid_efference_on_recovery": shuffle,
        "model_content_on_recovery": rotation,
        "model_update_on_target_error": 0.1,
        "model_update_on_prediction_mse": 0.01,
        "damage_transplant_recovery_rate": transplant,
        "damage_transplant_backup_switch_rate": transplant_switch,
    }
    return {
        "variant": variant,
        "evaluation": {"seed": seed},
        "conditions": {"normal": normal, "update_lesion": lesion_condition},
        "causal_effects": causal,
    }


def test_phase4_confirmatory_family_passes_effects_and_guardrails():
    results = []
    for seed in range(8):
        results.append(_cell(
            "self_model", seed, recovery=1.0, lesion=0.4,
            shuffle=0.4, rotation=0.5))
        results.append(_cell(
            "stale_model", seed, recovery=0.6, lesion=0.0,
            shuffle=0.0, rotation=0.0))
    family = _aggregate(results)["confirmatory_primary_family"]
    assert family["status"] == "pass"
    assert all(family["criterion_passed"].values())
    assert all(family["guardrails"].values())


def test_phase4_confirmatory_family_rejects_unstable_body():
    results = []
    for seed in range(8):
        results.append(_cell(
            "self_model", seed, recovery=1.0, lesion=0.4,
            shuffle=0.4, rotation=0.5,
            radius=1.01 if seed == 0 else 0.95))
        results.append(_cell(
            "stale_model", seed, recovery=0.6, lesion=0.0,
            shuffle=0.0, rotation=0.0))
    family = _aggregate(results)["confirmatory_primary_family"]
    assert family["status"] == "fail"
    assert not family["guardrails"][
        "passive_wave_radius_each_seed_below_1"]
