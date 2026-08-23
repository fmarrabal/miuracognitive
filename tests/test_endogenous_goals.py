import torch

from data.endogenous_goals import (
    EndogenousGoalDataset, EndogenousGoalEnvironmentConfig)
from eval.endogenous_goals import endogenous_goal_metrics
from model.endogenous_goals import (
    EndogenousGoalAgent, EndogenousGoalControllerConfig)


def _cfg():
    return EndogenousGoalEnvironmentConfig()


def _agent(variant="goal_memory"):
    return EndogenousGoalAgent(EndogenousGoalControllerConfig(
        n_needs=3, d_model=24, variant=variant))


def test_minor_conflict_is_exactly_history_aliased_after_gap():
    cfg = _cfg()
    scenario = EndogenousGoalDataset(
        cfg, seed=10).batch(48, critical_mode="minor")
    rows = torch.arange(scenario.batch_size)
    # Estado justo antes de decidir en t=2: dos drenajes y el pulso de t=1.
    levels = (scenario.initial_levels - 2 * cfg.basal_drain
              - scenario.disturbances[:, 1])
    primary = levels[rows, scenario.initial_target]
    rival = levels[rows, scenario.rival_target]
    torch.testing.assert_close(primary, rival, rtol=0, atol=1e-7)
    assert torch.all(scenario.opportunities[:, 1] == 0)
    assert torch.all(scenario.opportunities[:, 2] == 1)


def test_target_metadata_is_not_a_policy_input():
    cfg = _cfg()
    torch.manual_seed(11)
    agent = _agent().eval()
    scenario = EndogenousGoalDataset(cfg, seed=11).batch(8)
    changed = scenario.clone()
    changed.initial_target = torch.roll(changed.initial_target, 1)
    changed.rival_target = torch.roll(changed.rival_target, 1)
    changed.critical = ~changed.critical
    a = agent.rollout(scenario, cfg, hard_actions=True)
    b = agent.rollout(changed, cfg, hard_actions=True)
    torch.testing.assert_close(a["levels"], b["levels"], rtol=0, atol=0)
    torch.testing.assert_close(a["actions"], b["actions"], rtol=0, atol=0)
    torch.testing.assert_close(a["goal_probs"], b["goal_probs"], rtol=0, atol=0)


def test_project_progress_survives_gaps_and_requires_three_work_actions():
    cfg = _cfg()
    torch.manual_seed(12)
    agent = _agent().eval()
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.zero_()
        agent.goal_proposal.bias[0] = 10.0
    scenario = EndogenousGoalDataset(
        cfg, seed=12).batch(2, critical_mode="minor")
    scenario.disturbances.zero_()
    result = agent.rollout(scenario, cfg, hard_actions=True)
    assert torch.all(result["action_ids"][:, 0] == 1)
    assert torch.all(result["action_ids"][:, 1] == 0)
    assert torch.all(result["action_ids"][:, 2] == 1)
    assert torch.all(result["progress"][:, 2, 0] == 1)  # tras gap t=1
    assert torch.all(result["completions"][:, :4, 0] == 0)
    assert torch.all(result["completions"][:, 4, 0] == 1)


def test_goal_rotation_changes_project_and_prevents_old_deadline():
    cfg = _cfg()
    torch.manual_seed(13)
    agent = _agent().eval()
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.zero_()
        agent.goal_proposal.bias[0] = 10.0
        # Puerta cerrada: la rotación causal persiste después de t=2.
        agent.goal_update[-1].bias.fill_(-10.0)
    scenario = EndogenousGoalDataset(
        cfg, seed=13).batch(2, critical_mode="minor")
    scenario.disturbances.zero_()
    result = agent.rollout(
        scenario, cfg, hard_actions=True, goal_rotation_tick=2)
    assert torch.all(result["action_ids"][:, 0] == 1)
    assert torch.all(result["action_ids"][:, 2] == 2)
    assert torch.all(result["completions"][:, :5, 0] == 0)


def test_memory_lesion_is_exact_reactive_control_with_paired_weights():
    cfg = _cfg()
    torch.manual_seed(14)
    memory = _agent("goal_memory").eval()
    torch.manual_seed(14)
    reactive = _agent("reactive").eval()
    scenario = EndogenousGoalDataset(cfg, seed=14).batch(10)
    lesioned = memory.rollout(
        scenario, cfg, hard_actions=True, memory_mode="lesion")
    control = reactive.rollout(scenario, cfg, hard_actions=True)
    for key in ("levels", "actions", "goal_probs", "update_gates"):
        torch.testing.assert_close(
            lesioned[key], control[key], rtol=0, atol=0)


def test_endogenous_goal_rollout_is_finite_and_differentiable():
    cfg = _cfg()
    torch.manual_seed(15)
    agent = _agent().train()
    scenario = EndogenousGoalDataset(cfg, seed=15).batch(16)
    result = agent.rollout(scenario, cfg, hard_actions=False)
    result["losses"]["total"].backward()
    assert torch.isfinite(result["levels"]).all()
    assert agent.goal_update[-1].weight.grad is not None
    assert torch.isfinite(agent.goal_update[-1].weight.grad).all()


def test_metrics_detect_persistence_switch_and_alias():
    cfg = _cfg()
    torch.manual_seed(16)
    agent = _agent().eval()
    scenario = EndogenousGoalDataset(
        cfg, seed=16).batch(12, critical_mode="minor")
    result = agent.rollout(scenario, cfg, hard_actions=True)
    metrics = endogenous_goal_metrics(result, scenario, cfg)
    assert metrics["minor_alias_max_abs_gap"] < 1e-6
    assert 0.0 <= metrics["mild_commitment_rate"] <= 1.0
    assert 0.0 <= metrics["goal_action_alignment_rate"] <= 1.0
