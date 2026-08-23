"""Intervenciones y métricas para la fase 2 de metas endógenas."""

from __future__ import annotations

import statistics

import torch

from data.endogenous_goals import (
    EndogenousGoalDataset, EndogenousGoalEnvironmentConfig,
    EndogenousGoalScenario, body_transplant_scenario)
from model.endogenous_goals import EndogenousGoalAgent


def endogenous_goal_metrics(
        rollout: dict[str, torch.Tensor],
        scenario: EndogenousGoalScenario,
        cfg: EndogenousGoalEnvironmentConfig) -> dict[str, float]:
    levels = rollout["levels"].detach().float().cpu()
    actions = rollout["action_ids"].detach().cpu()
    goals = rollout["goal_ids"].detach().cpu()
    gates = rollout["update_gates"].detach().float().cpu()
    completions = rollout["completions"].detach().float().cpu()
    initial = scenario.initial_target.detach().cpu()
    rival = scenario.rival_target.detach().cpu()
    critical = scenario.critical.detach().bool().cpu()
    post = levels[:, 1:]
    setpoints = scenario.setpoints.detach().float().cpu()
    batch = scenario.batch_size
    rows = torch.arange(batch)
    work = list(cfg.work_ticks)
    first_three = work[:cfg.project_length]
    crisis_work_tick = work[1]
    crisis_completion_tick = work[cfg.project_length]

    initial_action = actions[:, work[0]] == initial + 1
    initial_goal = goals[:, work[0]] == initial
    mild = ~critical
    persistence = torch.stack([
        actions[:, tick] == initial + 1 for tick in first_three], dim=1
    ).all(dim=1)
    goal_persistence = torch.stack([
        goals[:, tick] == initial for tick in range(
            work[0], first_three[-1] + 1)], dim=1).all(dim=1)
    primary_completion = completions[rows, :, initial].bool()
    rival_completion = completions[rows, :, rival].bool()
    mild_complete_by_deadline = primary_completion[
        :, :first_three[-1] + 1].any(dim=1)
    critical_switch = actions[:, crisis_work_tick] == rival + 1
    critical_goal_switch = goals[:, crisis_work_tick] == rival
    critical_rescue = rival_completion[
        :, :crisis_completion_tick + 1].any(dim=1)

    violation = post < cfg.viability_threshold
    survival = ~violation.flatten(1).any(dim=1)
    abs_error = (post - setpoints.unsqueeze(1)).abs()
    total_completions = completions.sum(dim=(1, 2))
    work_actions = actions[:, work]
    switches = (work_actions[:, 1:] != work_actions[:, :-1]).float()
    action_goal_alignment = torch.stack([
        actions[:, tick] == goals[:, tick] + 1 for tick in work], dim=1)

    # En episodios minor, al empezar t=2 ambos niveles son idénticos aunque
    # sólo el objetivo inicial tiene progreso físico oculto.
    alias_gap = (levels[rows, crisis_work_tick, initial]
                 - levels[rows, crisis_work_tick, rival]).abs()

    def subset_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].float().mean()) if mask.any() else 0.0

    return {
        "survival_rate": float(survival.float().mean()),
        "violation_rate": float(violation.float().mean()),
        "mean_absolute_homeostatic_error": float(abs_error.mean()),
        "mean_project_completions": float(total_completions.mean()),
        "initial_goal_selection_rate": float(initial_goal.float().mean()),
        "initial_action_selection_rate": float(initial_action.float().mean()),
        "mild_commitment_rate": subset_mean(persistence, mild),
        "mild_goal_persistence_rate": subset_mean(goal_persistence, mild),
        "mild_completion_rate": subset_mean(
            mild_complete_by_deadline, mild),
        "critical_switch_rate": subset_mean(critical_switch, critical),
        "critical_goal_switch_rate": subset_mean(
            critical_goal_switch, critical),
        "critical_rescue_rate": subset_mean(critical_rescue, critical),
        "work_switch_rate": float(switches.mean()),
        "goal_action_alignment_rate": float(
            action_goal_alignment.float().mean()),
        "minor_alias_max_abs_gap": (
            float(alias_gap[mild].max()) if mild.any() else 0.0),
        "minor_update_gate_at_conflict": subset_mean(
            gates[:, crisis_work_tick], mild),
        "critical_update_gate_at_conflict": subset_mean(
            gates[:, crisis_work_tick], critical),
        "n_episodes": batch,
    }


def _mean_metrics(cells: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in cells[0] if key != "n_episodes"]
    return {key: statistics.mean(float(cell[key]) for cell in cells)
            for key in keys} | {
                "n_episodes": sum(int(cell["n_episodes"]) for cell in cells)}


@torch.no_grad()
def evaluate_endogenous_goals(
        agent: EndogenousGoalAgent,
        dataset: EndogenousGoalDataset,
        device: torch.device,
        *, n_batches: int = 10,
        batch_size: int = 96) -> dict:
    cfg = dataset.cfg
    normal_cells, lesion_cells = [], []
    minor_cells, rotation_cells = [], []
    rotation_follow, rotation_changed = [], []
    records = []
    agent.eval()
    for batch_idx in range(n_batches):
        scenario = dataset.batch(batch_size, critical_mode="mixed").to(device)
        normal_rollout = agent.rollout(
            scenario, cfg, hard_actions=True)
        lesion_rollout = agent.rollout(
            scenario, cfg, hard_actions=True, memory_mode="lesion")
        normal = endogenous_goal_metrics(normal_rollout, scenario, cfg)
        lesion = endogenous_goal_metrics(lesion_rollout, scenario, cfg)

        minor_scenario = dataset.batch(
            batch_size, critical_mode="minor").to(device)
        minor_rollout = agent.rollout(
            minor_scenario, cfg, hard_actions=True)
        rotated_rollout = agent.rollout(
            minor_scenario, cfg, hard_actions=True,
            goal_rotation_tick=cfg.work_ticks[1])
        minor = endogenous_goal_metrics(minor_rollout, minor_scenario, cfg)
        rotated = endogenous_goal_metrics(
            rotated_rollout, minor_scenario, cfg)
        tick = cfg.work_ticks[1]
        rotation_follow.append(float((
            rotated_rollout["action_ids"][:, tick]
            == rotated_rollout["goal_ids"][:, tick] + 1
        ).float().mean()))
        rotation_changed.append(float((
            rotated_rollout["action_ids"][:, tick]
            != minor_rollout["action_ids"][:, tick]
        ).float().mean()))

        normal_cells.append(normal)
        lesion_cells.append(lesion)
        minor_cells.append(minor)
        rotation_cells.append(rotated)
        records.append({
            "batch": batch_idx,
            "normal": normal,
            "memory_lesion": lesion,
            "minor_normal": minor,
            "goal_rotation": rotated,
            "rotation_follow_rate": rotation_follow[-1],
            "rotation_changed_action_rate": rotation_changed[-1],
        })

    transplant = body_transplant_scenario(
        cfg, batch_per_need=max(8, batch_size // cfg.n_needs)).to(device)
    transplant_rollout = agent.rollout(
        transplant, cfg, hard_actions=True)
    target = transplant.initial_target
    transplant_goal = float((
        transplant_rollout["goal_ids"][:, 0] == target).float().mean())
    transplant_action = float((
        transplant_rollout["action_ids"][:, 0] == target + 1).float().mean())

    normal = _mean_metrics(normal_cells)
    lesion = _mean_metrics(lesion_cells)
    minor = _mean_metrics(minor_cells)
    rotation = _mean_metrics(rotation_cells)
    return {
        "variant": agent.cfg.variant,
        "design": {
            "external_goal_supplied": False,
            "target_metadata_visible_to_policy": False,
            "physical_project_progress_visible_to_policy": False,
            "mandatory_gap_between_work_actions": True,
            "minor_state_is_history_aliased": True,
            "primary_mechanism_intervention": "memory_lesion",
        },
        "conditions": {
            "normal": normal,
            "memory_lesion": lesion,
            "minor_normal": minor,
            "goal_rotation": rotation,
        },
        "causal_effects": {
            "memory_on_mild_completion": (
                normal["mild_completion_rate"]
                - lesion["mild_completion_rate"]),
            "memory_on_mild_commitment": (
                normal["mild_commitment_rate"]
                - lesion["mild_commitment_rate"]),
            "memory_on_survival": (
                normal["survival_rate"] - lesion["survival_rate"]),
            "goal_content_on_minor_completion": (
                minor["mild_completion_rate"]
                - rotation["mild_completion_rate"]),
            "goal_rotation_follow_rate": statistics.mean(rotation_follow),
            "goal_rotation_changed_action_rate": statistics.mean(
                rotation_changed),
            "body_transplant_goal_selection_rate": transplant_goal,
            "body_transplant_action_selection_rate": transplant_action,
        },
        "records": records,
    }
