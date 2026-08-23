"""Métricas e intervenciones para descubrimiento continuo de metas."""

from __future__ import annotations

import statistics

import torch

from data.goal_discovery import (
    GoalDiscoveryDataset, GoalDiscoveryEnvironmentConfig,
    GoalDiscoveryScenario, affordance_response, body_transplant)
from model.goal_discovery import ContinuousGoalDiscoverer


def goal_discovery_metrics(
        rollout: dict[str, torch.Tensor],
        scenario: GoalDiscoveryScenario,
        cfg: GoalDiscoveryEnvironmentConfig) -> dict[str, float]:
    goal = rollout["goal"].detach().float().cpu()
    response = rollout["goal_response"].detach().float().cpu()
    final_levels = rollout["final_levels"].detach().float().cpu()
    queries = rollout["queries"].detach().float().cpu()
    initial = scenario.initial_levels.detach().float().cpu()
    setpoints = scenario.setpoints.detach().float().cpu()
    centers = scenario.centers.detach().float().cpu()
    target = scenario.dominant_need.detach().cpu()
    low = scenario.feasible_low.detach().float().cpu()
    high = scenario.feasible_high.detach().float().cpu()
    rows = torch.arange(scenario.batch_size)
    target_center = centers[rows, target]
    target_distance = (goal - target_center).norm(dim=-1)
    target_response = response[rows, target]
    target_final = final_levels[rows, target]
    initial_error = (initial - setpoints).abs().mean(dim=-1)
    final_error = (final_levels - setpoints).abs().mean(dim=-1)
    feasible = ((goal >= low) & (goal <= high)).all(dim=-1)
    success = (target_response >= 0.90) & (target_final >= 0.70)
    normalized_goal = (goal - low) / (high - low)
    rounded = torch.round(normalized_goal * 10_000).to(torch.int64)
    unique_fraction = len(torch.unique(rounded, dim=0)) / scenario.batch_size

    pairwise = []
    for left in range(cfg.n_probes):
        for right in range(left + 1, cfg.n_probes):
            width = (high - low).norm(dim=-1).clamp_min(1e-6)
            pairwise.append(
                (queries[:, left] - queries[:, right]).norm(dim=-1) / width)
    mean_probe_distance = float(torch.stack(pairwise, dim=1).mean())
    return {
        "goal_success_rate": float(success.float().mean()),
        "mean_target_center_distance": float(target_distance.mean()),
        "mean_target_response": float(target_response.mean()),
        "dominant_restored_rate": float((target_final >= 0.70).float().mean()),
        "mean_homeostatic_improvement": float(
            (initial_error - final_error).mean()),
        "mean_final_absolute_error": float(final_error.mean()),
        "feasible_goal_rate": float(feasible.float().mean()),
        "continuous_unique_goal_fraction": float(unique_fraction),
        "mean_normalized_probe_distance": mean_probe_distance,
        "n_episodes": scenario.batch_size,
    }


def _mean_metrics(cells: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in cells[0] if key != "n_episodes"]
    return {key: statistics.mean(float(cell[key]) for cell in cells)
            for key in keys} | {
                "n_episodes": sum(int(cell["n_episodes"]) for cell in cells)}


@torch.no_grad()
def evaluate_goal_discovery(
        agent: ContinuousGoalDiscoverer,
        dataset: GoalDiscoveryDataset,
        device: torch.device,
        *, n_batches: int = 10,
        batch_size: int = 128) -> dict:
    cfg = dataset.cfg
    normal_cells, lesion_cells, shuffle_cells, reflect_cells = [], [], [], []
    transplant_switch, transplant_new_success, transplant_shift = [], [], []
    records = []
    agent.eval()
    for batch_idx in range(n_batches):
        scenario = dataset.batch(batch_size).to(device)
        normal_rollout = agent.rollout(scenario, cfg)
        lesion_rollout = agent.rollout(
            scenario, cfg, feedback_mode="lesion")
        shuffle_rollout = agent.rollout(
            scenario, cfg, feedback_mode="shuffle")
        reflect_rollout = agent.rollout(
            scenario, cfg, goal_mode="reflect")
        normal = goal_discovery_metrics(normal_rollout, scenario, cfg)
        lesion = goal_discovery_metrics(lesion_rollout, scenario, cfg)
        shuffled = goal_discovery_metrics(shuffle_rollout, scenario, cfg)
        reflected = goal_discovery_metrics(reflect_rollout, scenario, cfg)

        transplanted = body_transplant(scenario, cfg)
        transplanted_rollout = agent.rollout(transplanted, cfg)
        rows = torch.arange(batch_size, device=device)
        old_target = scenario.dominant_need
        new_target = transplanted.dominant_need
        new_goal = transplanted_rollout["goal"]
        new_distance = (new_goal - scenario.centers[rows, new_target]).norm(
            dim=-1)
        old_distance = (new_goal - scenario.centers[rows, old_target]).norm(
            dim=-1)
        new_response = affordance_response(
            new_goal, scenario.centers, cfg.response_sigma)[rows, new_target]
        transplant_switch.append(float((
            new_distance < old_distance).float().mean()))
        transplant_new_success.append(float((new_response >= 0.90).float().mean()))
        transplant_shift.append(float((
            new_goal - normal_rollout["goal"]).norm(dim=-1).mean()))

        normal_cells.append(normal)
        lesion_cells.append(lesion)
        shuffle_cells.append(shuffled)
        reflect_cells.append(reflected)
        records.append({
            "batch": batch_idx,
            "normal": normal,
            "feedback_lesion": lesion,
            "outcome_shuffle": shuffled,
            "goal_reflection": reflected,
            "body_transplant_switch_rate": transplant_switch[-1],
            "body_transplant_new_goal_success": transplant_new_success[-1],
            "body_transplant_goal_shift": transplant_shift[-1],
        })

    normal = _mean_metrics(normal_cells)
    lesion = _mean_metrics(lesion_cells)
    shuffled = _mean_metrics(shuffle_cells)
    reflected = _mean_metrics(reflect_cells)
    return {
        "variant": agent.cfg.variant,
        "design": {
            "external_goal_supplied": False,
            "center_visible_to_policy": False,
            "goal_catalog_exists": False,
            "goal_space": "continuous_2d_with_episode_specific_bounds",
            "probe_coordinates_chosen_by_agent": True,
            "training_center_supervision": False,
            "primary_mechanism_intervention": "feedback_lesion",
        },
        "conditions": {
            "normal": normal,
            "feedback_lesion": lesion,
            "outcome_shuffle": shuffled,
            "goal_reflection": reflected,
        },
        "causal_effects": {
            "feedback_on_goal_success": (
                normal["goal_success_rate"] - lesion["goal_success_rate"]),
            "valid_outcomes_on_goal_success": (
                normal["goal_success_rate"] - shuffled["goal_success_rate"]),
            "goal_content_on_success": (
                normal["goal_success_rate"] - reflected["goal_success_rate"]),
            "feedback_on_homeostatic_improvement": (
                normal["mean_homeostatic_improvement"]
                - lesion["mean_homeostatic_improvement"]),
            "body_transplant_switch_rate": statistics.mean(transplant_switch),
            "body_transplant_new_goal_success": statistics.mean(
                transplant_new_success),
            "body_transplant_goal_shift": statistics.mean(transplant_shift),
        },
        "records": records,
    }


@torch.no_grad()
def evaluate_decoder_sensitivity(
        agent: ContinuousGoalDiscoverer,
        dataset: GoalDiscoveryDataset,
        device: torch.device,
        *, sigma_scales: tuple[float, ...] = (0.85, 1.0, 1.15),
        n_batches: int = 5,
        batch_size: int = 128) -> dict[str, dict[str, float]]:
    """Audita error moderado en la inductive bias, sin reajustar pesos."""

    cells = {scale: [] for scale in sigma_scales}
    agent.eval()
    for _ in range(n_batches):
        scenario = dataset.batch(batch_size).to(device)
        for scale in sigma_scales:
            rollout = agent.rollout(
                scenario, dataset.cfg, decoder_sigma_scale=scale)
            cells[scale].append(goal_discovery_metrics(
                rollout, scenario, dataset.cfg))
    return {f"sigma_scale_{scale:.2f}": _mean_metrics(values)
            for scale, values in cells.items()}
