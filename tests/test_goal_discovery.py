import torch

from data.goal_discovery import (
    GoalDiscoveryDataset, GoalDiscoveryEnvironmentConfig,
    affordance_response)
from model.goal_discovery import (
    ContinuousGoalDiscoverer, GoalDiscoveryControllerConfig)


def _cfg():
    return GoalDiscoveryEnvironmentConfig()


def _agent(variant="discoverer"):
    return ContinuousGoalDiscoverer(GoalDiscoveryControllerConfig(
        n_needs=3, goal_dim=2, n_probes=4, d_model=32,
        variant=variant))


def test_affordance_response_is_maximal_at_continuous_center():
    centers = torch.tensor([[[0.2, -0.3], [-0.5, 0.4], [0.7, 0.1]]])
    at_center = affordance_response(centers[:, 0], centers, sigma=0.62)
    shifted = affordance_response(
        centers[:, 0] + torch.tensor([[0.3, 0.2]]), centers, sigma=0.62)
    assert at_center[0, 0] == 1.0
    assert shifted[0, 0] < at_center[0, 0]


def test_first_probe_cannot_see_hidden_landscape_centers():
    cfg = _cfg()
    torch.manual_seed(21)
    agent = _agent().eval()
    a = GoalDiscoveryDataset(cfg, seed=21).batch(8)
    b = a.clone()
    b.centers = torch.roll(b.centers, shifts=1, dims=1)
    out_a = agent.rollout(a, cfg)
    out_b = agent.rollout(b, cfg)
    torch.testing.assert_close(
        out_a["queries"][:, 0], out_b["queries"][:, 0], rtol=0, atol=0)
    assert not torch.equal(
        out_a["probe_responses"][:, 0],
        out_b["probe_responses"][:, 0])


def test_no_feedback_control_goal_is_invariant_to_hidden_centers():
    cfg = _cfg()
    torch.manual_seed(22)
    agent = _agent("no_feedback").eval()
    a = GoalDiscoveryDataset(cfg, seed=22).batch(8)
    b = a.clone()
    b.centers = torch.roll(b.centers, shifts=1, dims=1)
    out_a = agent.rollout(a, cfg)
    out_b = agent.rollout(b, cfg)
    torch.testing.assert_close(
        out_a["queries"], out_b["queries"], rtol=0, atol=0)
    torch.testing.assert_close(
        out_a["goal"], out_b["goal"], rtol=0, atol=0)


def test_feedback_lesion_is_exact_paired_no_feedback_control():
    cfg = _cfg()
    torch.manual_seed(23)
    discoverer = _agent("discoverer").eval()
    torch.manual_seed(23)
    control = _agent("no_feedback").eval()
    scenario = GoalDiscoveryDataset(cfg, seed=23).batch(12)
    lesion = discoverer.rollout(scenario, cfg, feedback_mode="lesion")
    baseline = control.rollout(scenario, cfg)
    for key in ("queries", "goal", "final_levels"):
        torch.testing.assert_close(
            lesion[key], baseline[key], rtol=0, atol=0)


def test_goals_and_reflections_are_always_feasible():
    cfg = _cfg()
    torch.manual_seed(24)
    agent = _agent().eval()
    scenario = GoalDiscoveryDataset(cfg, seed=24).batch(16)
    for mode in ("normal", "reflect"):
        goal = agent.rollout(scenario, cfg, goal_mode=mode)["goal"]
        assert torch.all(goal >= scenario.feasible_low)
        assert torch.all(goal <= scenario.feasible_high)


def test_rbf_trilateration_recovers_unseen_continuous_centers():
    cfg = _cfg()
    scenario = GoalDiscoveryDataset(cfg, seed=26).batch(32)
    unit = torch.tensor([
        [0.84, 0.50], [0.50, 0.84],
        [0.16, 0.50], [0.50, 0.16]])
    queries = (scenario.feasible_low.unsqueeze(1)
               + unit.unsqueeze(0) * (
                   scenario.feasible_high
                   - scenario.feasible_low).unsqueeze(1))
    responses = torch.stack([
        affordance_response(queries[:, idx], scenario.centers,
                            cfg.response_sigma)
        for idx in range(cfg.n_probes)], dim=1)
    inferred = ContinuousGoalDiscoverer._infer_centers(
        queries, responses, cfg.response_sigma)
    torch.testing.assert_close(inferred, scenario.centers,
                               rtol=2e-4, atol=2e-4)


def test_goal_discovery_rollout_is_finite_and_differentiable():
    cfg = _cfg()
    torch.manual_seed(25)
    agent = _agent().train()
    scenario = GoalDiscoveryDataset(cfg, seed=25).batch(16)
    result = agent.rollout(scenario, cfg)
    result["losses"]["total"].backward()
    assert torch.isfinite(result["goal"]).all()
    assert agent.query_head[-1].weight.grad is not None
    assert torch.isfinite(agent.query_head[-1].weight.grad).all()


def test_decoder_sigma_sensitivity_does_not_change_world_responses():
    cfg = _cfg()
    torch.manual_seed(27)
    agent = _agent().eval()
    scenario = GoalDiscoveryDataset(cfg, seed=27).batch(8)
    exact = agent.rollout(scenario, cfg, decoder_sigma_scale=1.0)
    mismatched = agent.rollout(scenario, cfg, decoder_sigma_scale=0.85)
    torch.testing.assert_close(
        exact["probe_responses"], mismatched["probe_responses"],
        rtol=0, atol=0)
    assert not torch.equal(exact["goal"], mismatched["goal"])
