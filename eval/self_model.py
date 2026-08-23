"""Métricas e intervenciones causales para la fase 4 de automodelo."""

from __future__ import annotations

import statistics

import torch

from data.self_model import (
    SelfModelDataset, SelfModelEnvironmentConfig, SelfModelScenario,
    damage_transplant)
from model.self_model import WaveBodySelfModel


def self_model_metrics(rollout: dict, scenario: SelfModelScenario,
                       cfg: SelfModelEnvironmentConfig) -> dict[str, float]:
    states = rollout["states_after"].detach().float().cpu()
    actions = rollout["actions"].detach().cpu()
    oracle = rollout["oracle_actions"].detach().cpu()
    prediction_mse = rollout["executed_prediction_mse"].detach().float().cpu()
    target = scenario.target_node.detach().cpu()
    rows = torch.arange(scenario.batch_size)
    target_state = states[rows, :, target]
    damage = cfg.damage_tick
    recovery_window = target_state[:, damage:damage + 5].abs()
    recovered = recovery_window <= 0.18
    recovery_success = recovered.any(dim=1)
    first_recovery = torch.where(
        recovered.any(dim=1),
        recovered.float().argmax(dim=1).float(),
        torch.full((scenario.batch_size,), 5.0))
    backup = 1 + cfg.n_nodes + target
    primary = 1 + target
    post_actions = actions[:, damage + 1:damage + 5]
    backup_switch = (post_actions == backup.unsqueeze(1)).any(dim=1)
    primary_repeat = (post_actions == primary.unsqueeze(1)).float().mean(dim=1)
    post_slice = slice(damage + 1, cfg.horizon)
    post_target_error = target_state[:, post_slice].abs()
    pre_recovery = target_state[:, :5].abs().min(dim=1).values <= 0.18
    oracle_match = (actions[:, post_slice] == oracle[:, post_slice])
    # Excluye el primer fallo: mide si, tras observar sorpresa, el modelo ya
    # predice correctamente los comandos que decide ejecutar.
    post_prediction_mse = prediction_mse[:, post_slice]
    final_model = rollout["final_effect_model"].detach().float().cpu()
    final_primary = final_model[rows, :, primary]
    true_primary = scenario.damaged_effects.detach().float().cpu()[
        rows, :, primary]
    return {
        "pre_damage_recovery_rate": float(pre_recovery.float().mean()),
        "post_damage_recovery_rate": float(recovery_success.float().mean()),
        "mean_recovery_latency": float(first_recovery.mean()),
        "backup_switch_rate": float(backup_switch.float().mean()),
        "failed_primary_repeat_rate": float(primary_repeat.mean()),
        "mean_post_damage_target_error": float(post_target_error.mean()),
        "post_damage_viability_rate": float(
            (post_target_error <= 0.55).float().mean()),
        "post_adaptation_prediction_mse": float(post_prediction_mse.mean()),
        "post_adaptation_oracle_match_rate": float(
            oracle_match.float().mean()),
        "final_damaged_actuator_model_mse": float(
            (final_primary - true_primary).square().mean()),
        "passive_spectral_radius": float(
            rollout["passive_spectral_radius"]),
        "n_episodes": scenario.batch_size,
    }


def _mean_metrics(cells: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in cells[0] if key != "n_episodes"]
    return {key: statistics.mean(float(cell[key]) for cell in cells)
            for key in keys} | {
                "n_episodes": sum(int(cell["n_episodes"]) for cell in cells)}


@torch.no_grad()
def evaluate_self_model(
        agent: WaveBodySelfModel,
        dataset: SelfModelDataset,
        device: torch.device,
        *, n_batches: int = 10,
        batch_size: int = 128) -> dict:
    cfg = dataset.cfg
    cells = {name: [] for name in (
        "normal", "update_lesion", "efference_shuffle", "model_rotation")}
    transplant_cells = []
    records = []
    agent.eval()
    for batch_idx in range(n_batches):
        scenario = dataset.batch(batch_size).to(device)
        rollouts = {
            "normal": agent.rollout(scenario),
            "update_lesion": agent.rollout(
                scenario, update_mode="lesion"),
            "efference_shuffle": agent.rollout(
                scenario, update_mode="shuffle"),
            "model_rotation": agent.rollout(
                scenario, model_mode="rotate"),
        }
        batch_record = {"batch": batch_idx}
        for name, rollout in rollouts.items():
            metrics = self_model_metrics(rollout, scenario, cfg)
            cells[name].append(metrics)
            batch_record[name] = metrics
        transplanted = damage_transplant(scenario, cfg)
        transplant_metrics = self_model_metrics(
            agent.rollout(transplanted), transplanted, cfg)
        transplant_cells.append(transplant_metrics)
        batch_record["damage_transplant"] = transplant_metrics
        records.append(batch_record)

    conditions = {name: _mean_metrics(values)
                  for name, values in cells.items()}
    transplant = _mean_metrics(transplant_cells)
    normal = conditions["normal"]
    lesion = conditions["update_lesion"]
    shuffled = conditions["efference_shuffle"]
    rotated = conditions["model_rotation"]
    return {
        "variant": agent.cfg.variant,
        "design": {
            "damage_flag_visible_to_policy": False,
            "true_actuator_matrix_visible_to_policy": False,
            "target_node_visible_to_policy": False,
            "efference_copy_available": True,
            "passive_body_model": "validated_damped_wave_recurrence",
            "primary_mechanism_intervention": "post_damage_update_lesion",
            "equal_action_and_compute_budget": True,
        },
        "conditions": conditions | {"damage_transplant": transplant},
        "causal_effects": {
            "model_update_on_recovery": (
                normal["post_damage_recovery_rate"]
                - lesion["post_damage_recovery_rate"]),
            "valid_efference_on_recovery": (
                normal["post_damage_recovery_rate"]
                - shuffled["post_damage_recovery_rate"]),
            "model_content_on_recovery": (
                normal["post_damage_recovery_rate"]
                - rotated["post_damage_recovery_rate"]),
            "model_update_on_target_error": (
                lesion["mean_post_damage_target_error"]
                - normal["mean_post_damage_target_error"]),
            "model_update_on_prediction_mse": (
                lesion["post_adaptation_prediction_mse"]
                - normal["post_adaptation_prediction_mse"]),
            "damage_transplant_recovery_rate": transplant[
                "post_damage_recovery_rate"],
            "damage_transplant_backup_switch_rate": transplant[
                "backup_switch_rate"],
        },
        "records": records,
    }
