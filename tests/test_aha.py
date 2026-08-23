import torch

from data.aha import (AHAEnvironmentConfig, AHAScenario,
                      AnticipatoryHomeostasisDataset,
                      future_disturbance_targets)
from eval.aha import aha_metrics
from model.aha import AHAControllerConfig, AnticipatoryHomeostaticAgent


def _cfg(horizon=16):
    return AHAEnvironmentConfig(
        horizon=horizon, n_events=2, cue_lead=5,
        action_delay=2, prediction_horizon=3)


def _empty_scenario(cfg, batch_size=2):
    return AHAScenario(
        initial_levels=torch.full((batch_size, cfg.n_needs), 0.75),
        setpoints=torch.full((batch_size, cfg.n_needs), 0.75),
        cues=torch.zeros(batch_size, cfg.horizon, cfg.n_needs),
        disturbances=torch.zeros(batch_size, cfg.horizon, cfg.n_needs),
    )


def test_aha_events_have_earlier_type_matched_cues():
    cfg = _cfg()
    dataset = AnticipatoryHomeostasisDataset(cfg, seed=7)
    scenario = dataset.batch(8, hazards=True)
    scenario.validate(cfg)
    events = scenario.disturbances.nonzero(as_tuple=False)
    assert len(events) == 8 * cfg.n_events
    for row, tick, need in events.tolist():
        torch.testing.assert_close(
            scenario.cues[row, tick - cfg.cue_lead, need],
            scenario.disturbances[row, tick, need])


def test_future_target_covers_action_delay_but_is_not_an_input():
    cfg = _cfg()
    disturbances = torch.zeros(1, cfg.horizon, cfg.n_needs)
    event_tick = 10
    disturbances[0, event_tick, 1] = 0.35
    target = future_disturbance_targets(
        disturbances, cfg.prediction_horizon)
    assert target[0, event_tick - cfg.action_delay, 1] == 0.35
    assert target[0, event_tick, 1] == 0.0


def test_predictor_prefix_cannot_see_future_disturbance():
    cfg = _cfg()
    torch.manual_seed(11)
    agent = AnticipatoryHomeostaticAgent(AHAControllerConfig(
        n_needs=cfg.n_needs, d_model=24, d_predictor=16,
        variant="gating_wm")).eval()
    scenario = _empty_scenario(cfg, batch_size=2)
    scenario.cues[0, :6] = scenario.cues[1, :6]
    scenario.disturbances[1, 9, 2] = 0.37
    prediction = agent.predict_disturbances(scenario)
    torch.testing.assert_close(
        prediction[0, :7], prediction[1, :7], rtol=0, atol=0)


def test_shared_aha_predictor_initialization_is_paired_across_variants():
    models = []
    for variant in AnticipatoryHomeostaticAgent.VARIANTS:
        torch.manual_seed(2026)
        models.append(AnticipatoryHomeostaticAgent(AHAControllerConfig(
            d_model=24, d_predictor=16, variant=variant)))
    reference = dict(models[0].named_parameters())
    names = [name for name in reference
             if name.startswith(("predictor.", "predictor_head."))]
    assert names
    for model in models[1:]:
        candidate = dict(model.named_parameters())
        for name in names:
            torch.testing.assert_close(
                reference[name], candidate[name], rtol=0, atol=0)


def test_hbp_aha_starts_as_exact_identity_over_shared_controller():
    cfg = _cfg()
    torch.manual_seed(2027)
    control = AnticipatoryHomeostaticAgent(AHAControllerConfig(
        n_needs=cfg.n_needs, d_model=24, d_predictor=16,
        variant="gating_wm")).eval()
    torch.manual_seed(2027)
    hbp = AnticipatoryHomeostaticAgent(AHAControllerConfig(
        n_needs=cfg.n_needs, d_model=24, d_predictor=16,
        variant="hbp_full")).eval()
    scenario = AnticipatoryHomeostasisDataset(
        cfg, seed=2027).batch(4, hazards=True)
    base = control.rollout(scenario, cfg, hard_actions=True)
    candidate = hbp.rollout(scenario, cfg, hard_actions=True)
    torch.testing.assert_close(
        base["goal_logits"], candidate["goal_logits"], rtol=0, atol=0)
    torch.testing.assert_close(
        base["levels"], candidate["levels"], rtol=0, atol=0)


def test_reactive_controller_is_invariant_to_predictive_cues():
    cfg = _cfg()
    torch.manual_seed(12)
    agent = AnticipatoryHomeostaticAgent(AHAControllerConfig(
        n_needs=cfg.n_needs, d_model=24, d_predictor=16,
        variant="reactive")).eval()
    a = _empty_scenario(cfg)
    b = a.clone()
    b.cues[:, 2, 1] = 0.35
    out_a = agent.rollout(a, cfg, hard_actions=True)
    out_b = agent.rollout(b, cfg, hard_actions=True)
    torch.testing.assert_close(out_a["actions"], out_b["actions"],
                               rtol=0, atol=0)
    # El predictor sí cambia: lo lesionado es su acceso al ejecutivo.
    assert not torch.equal(out_a["predictions"], out_b["predictions"])


def test_action_effect_arrives_only_after_declared_delay():
    cfg = _cfg(horizon=14)
    torch.manual_seed(13)
    agent = AnticipatoryHomeostaticAgent(AHAControllerConfig(
        n_needs=cfg.n_needs, d_model=24, d_predictor=16,
        variant="reactive")).eval()
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.zero_()
        agent.goal_head.bias[1] = 10.0  # siempre restaura necesidad 0
    scenario = _empty_scenario(cfg, batch_size=1)
    result = agent.rollout(scenario, cfg, hard_actions=True)
    levels = result["levels"][0, :, 0]
    # Antes de que llegue la primera acción sólo opera el drenaje basal.
    assert levels[1] < levels[0]
    assert levels[cfg.action_delay] < levels[cfg.action_delay - 1]
    assert levels[cfg.action_delay + 1] > levels[cfg.action_delay]


def test_hbp_aha_rollout_is_finite_and_differentiable():
    cfg = _cfg()
    torch.manual_seed(14)
    agent = AnticipatoryHomeostaticAgent(AHAControllerConfig(
        n_needs=cfg.n_needs, d_model=24, d_predictor=16,
        variant="hbp_full")).train()
    scenario = AnticipatoryHomeostasisDataset(
        cfg, seed=14).batch(4, hazards=True)
    result = agent.rollout(scenario, cfg, hard_actions=False)
    loss = result["losses"]["total"]
    loss.backward()
    assert torch.isfinite(result["levels"]).all()
    assert agent.hbp.raw_zeta.grad is not None
    assert torch.isfinite(agent.hbp.raw_zeta.grad).all()


def test_blank_initiative_requires_correct_pre_event_action():
    cfg = _cfg()
    scenario = _empty_scenario(cfg, batch_size=1)
    event_tick, need = 10, 2
    scenario.cues[0, event_tick - cfg.cue_lead, need] = 0.35
    scenario.disturbances[0, event_tick, need] = 0.35
    action_ids = torch.zeros(1, cfg.horizon, dtype=torch.long)
    action_ids[0, event_tick - cfg.action_delay] = need + 1
    actions = torch.nn.functional.one_hot(
        action_ids, cfg.n_actions).float()
    rollout = {
        "levels": torch.full(
            (1, cfg.horizon + 1, cfg.n_needs), 0.75),
        "action_ids": action_ids,
        "predictions": future_disturbance_targets(
            scenario.disturbances, cfg.prediction_horizon),
        "prediction_targets": future_disturbance_targets(
            scenario.disturbances, cfg.prediction_horizon),
        "actions": actions,
    }
    metrics = aha_metrics(rollout, scenario, cfg)
    assert metrics["anticipatory_event_rate"] == 1.0
    assert metrics["blank_initiative_rate"] == 1.0
