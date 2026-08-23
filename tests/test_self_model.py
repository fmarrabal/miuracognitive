import torch

from data.self_model import (
    SelfModelDataset, SelfModelEnvironmentConfig,
    damage_transplant, wave_companion_spectral_radius)
from model.self_model import SelfModelControllerConfig, WaveBodySelfModel


def _cfg():
    return SelfModelEnvironmentConfig()


def _agent(variant="self_model"):
    cfg = _cfg()
    return WaveBodySelfModel(
        SelfModelControllerConfig(n_nodes=cfg.n_nodes, variant=variant), cfg)


def test_passive_wave_body_is_strictly_stable():
    assert wave_companion_spectral_radius(_cfg()) < 1.0


def test_self_model_lesion_is_exact_stale_control():
    cfg = _cfg()
    scenario = SelfModelDataset(cfg, seed=50).batch(32)
    adaptive = _agent("self_model").rollout(
        scenario, update_mode="lesion")
    stale = _agent("stale_model").rollout(scenario)
    for key in ("states_after", "actions", "final_effect_model"):
        torch.testing.assert_close(adaptive[key], stale[key], rtol=0, atol=0)


def test_adaptive_model_identifies_failed_actuator_but_stale_does_not():
    cfg = _cfg()
    scenario = SelfModelDataset(cfg, seed=51).batch(32)
    rows = torch.arange(scenario.batch_size)
    primary = 1 + scenario.target_node
    adaptive = _agent("self_model").rollout(scenario)
    stale = _agent("stale_model").rollout(scenario)
    adaptive_column = adaptive["final_effect_model"][rows, :, primary]
    stale_column = stale["final_effect_model"][rows, :, primary]
    truth = scenario.damaged_effects[rows, :, primary]
    torch.testing.assert_close(adaptive_column, truth, rtol=0, atol=0)
    assert float((stale_column - truth).square().mean()) > 0.001


def test_damage_transplant_changes_hidden_failure_not_intact_body():
    cfg = _cfg()
    scenario = SelfModelDataset(cfg, seed=52).batch(16)
    transplanted = damage_transplant(scenario, cfg)
    torch.testing.assert_close(
        scenario.intact_effects, transplanted.intact_effects, rtol=0, atol=0)
    assert torch.equal(
        transplanted.target_node, (scenario.target_node + 1) % cfg.n_nodes)
    assert not torch.equal(
        scenario.damaged_effects, transplanted.damaged_effects)


def test_rollout_is_finite_and_does_not_receive_damage_flag():
    cfg = _cfg()
    scenario = SelfModelDataset(cfg, seed=53).batch(16)
    result = _agent().rollout(scenario)
    assert torch.isfinite(result["states_after"]).all()
    assert torch.isfinite(result["executed_prediction_mse"]).all()
    assert result["actions"].min() >= 0
    assert result["actions"].max() < cfg.n_actions
