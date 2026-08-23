"""Evaluación causal del benchmark AHA."""

from __future__ import annotations

import statistics

import torch

from data.aha import (AHAEnvironmentConfig, AHAScenario,
                      AnticipatoryHomeostasisDataset)
from model.aha import AnticipatoryHomeostaticAgent


def aha_metrics(rollout: dict[str, torch.Tensor], scenario: AHAScenario,
                cfg: AHAEnvironmentConfig) -> dict[str, float]:
    """Métricas conductuales; no dependen de pérdidas de entrenamiento."""

    levels = rollout["levels"].detach().float().cpu()
    action_ids = rollout["action_ids"].detach().cpu()
    predictions = rollout["predictions"].detach().float().cpu()
    targets = rollout["prediction_targets"].detach().float().cpu()
    disturbances = scenario.disturbances.detach().float().cpu()
    cues = scenario.cues.detach().float().cpu()
    setpoints = scenario.setpoints.detach().float().cpu()
    post = levels[:, 1:]
    batch, horizon, n_needs = disturbances.shape

    violation = post < cfg.viability_threshold
    survival = ~violation.flatten(1).any(dim=1)
    deficit = torch.relu(setpoints.unsqueeze(1) - post)
    event_positions = disturbances.nonzero(as_tuple=False)
    positive_prediction = targets > 0
    pred_positive = (float(predictions[positive_prediction].mean())
                     if positive_prediction.any() else 0.0)
    pred_negative = (float(predictions[~positive_prediction].mean())
                     if (~positive_prediction).any() else 0.0)
    flat_prediction, flat_target = predictions.flatten(), targets.flatten()
    if (flat_prediction.std(unbiased=False) < 1e-8
            or flat_target.std(unbiased=False) < 1e-8):
        prediction_corr = 0.0
    else:
        prediction_corr = float(
            ((flat_prediction - flat_prediction.mean())
             * (flat_target - flat_target.mean())).mean()
            / (flat_prediction.std(unbiased=False)
               * flat_target.std(unbiased=False)))

    # Ventanas válidas: después del cue (ya ausente) y lo bastante pronto para
    # que la acción demorada llegue como máximo en el tick del evento.
    preparation_mask = torch.zeros(
        batch, horizon, n_needs, dtype=torch.bool)
    anticipated, blank_initiated, leads = [], [], []
    for row_t, event_t_t, need_t in event_positions:
        row, event_t, need = int(row_t), int(event_t_t), int(need_t)
        start = max(0, event_t - cfg.cue_lead + 1)
        stop = max(start, event_t - cfg.action_delay)
        preparation_mask[row, start:stop + 1, need] = True
        correct_times = [tick for tick in range(start, stop + 1)
                         if int(action_ids[row, tick]) == need + 1]
        anticipated.append(bool(correct_times))
        if correct_times:
            first = correct_times[0]
            leads.append(event_t - first)
            blank = (float(cues[row, first].abs().sum()) == 0.0
                     and float(disturbances[row, first].abs().sum()) == 0.0
                     and float(levels[row, first, need])
                     >= cfg.viability_threshold)
            blank_initiated.append(blank)
        else:
            leads.append(0)
            blank_initiated.append(False)

    need_action = action_ids > 0
    relevant_action = torch.zeros_like(need_action)
    for need in range(n_needs):
        relevant_action |= ((action_ids == need + 1)
                            & preparation_mask[:, :, need])
    false_actions = need_action & ~relevant_action
    n_actions = int(need_action.sum())

    return {
        "survival_rate": float(survival.float().mean()),
        "violation_rate": float(violation.float().mean()),
        "mean_viability_margin": float(
            (post - cfg.viability_threshold).mean()),
        "mean_homeostatic_deficit": float(deficit.mean()),
        "mean_absolute_homeostatic_error": float(
            (post - setpoints.unsqueeze(1)).abs().mean()),
        "action_rate": float(need_action.float().mean()),
        "false_action_fraction": (
            float(false_actions.sum()) / n_actions if n_actions else 0.0),
        "anticipatory_event_rate": (
            sum(anticipated) / len(anticipated) if anticipated else 0.0),
        "blank_initiative_rate": (
            sum(blank_initiated) / len(blank_initiated)
            if blank_initiated else 0.0),
        "mean_correct_action_lead": (
            statistics.mean(leads) if leads else 0.0),
        "prediction_mse": float((predictions - targets).square().mean()),
        "prediction_positive_mean": pred_positive,
        "prediction_negative_mean": pred_negative,
        "prediction_event_contrast": pred_positive - pred_negative,
        "prediction_target_correlation": prediction_corr,
        "n_events": int(len(anticipated)),
    }


def _mean_metrics(cells: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in cells[0] if key != "n_events"]
    return {key: statistics.mean(cell[key] for cell in cells)
            for key in keys} | {
                "n_events": sum(int(cell["n_events"]) for cell in cells)}


def _shuffle_cues(scenario: AHAScenario, generator: torch.Generator) -> AHAScenario:
    """Rompe cue→mundo conservando exactamente el multiconjunto de cues."""

    out = scenario.clone()
    if scenario.batch_size > 1:
        permutation = torch.randperm(
            scenario.batch_size, generator=generator,
            device="cpu").to(scenario.cues.device)
        out.cues = out.cues.index_select(0, permutation)
    else:
        out.cues = torch.roll(out.cues, shifts=1, dims=1)
    return out


@torch.no_grad()
def need_transplant_effect(agent: AnticipatoryHomeostaticAgent,
                           scenario: AHAScenario,
                           cfg: AHAEnvironmentConfig) -> dict[str, float]:
    """Interviene sólo el cuerpo interno bajo un exterior nulo e idéntico."""

    base = scenario.clone()
    base.cues.zero_()
    base.disturbances.zero_()
    # Todos parten viables. Para cada necesidad comparamos déficit vs saciedad
    # conservando exactamente setpoints, mundo y resto de variables.
    effects, flips = [], []
    for need in range(cfg.n_needs):
        deficit = base.clone()
        satiated = base.clone()
        deficit.setpoints.fill_(0.75)
        satiated.setpoints.copy_(deficit.setpoints)
        deficit.initial_levels.fill_(0.78)
        satiated.initial_levels.copy_(deficit.initial_levels)
        deficit.initial_levels[:, need] = 0.55
        satiated.initial_levels[:, need] = 0.95
        low = agent.rollout(deficit, cfg, hard_actions=True)
        high = agent.rollout(satiated, cfg, hard_actions=True)
        # Probabilidad antes de que ninguna acción pueda modificar el cuerpo.
        p_low = low["action_probs"][:, 0, need + 1]
        p_high = high["action_probs"][:, 0, need + 1]
        effects.append(float((p_low - p_high).mean()))
        a_low = low["action_ids"][:, 0]
        a_high = high["action_ids"][:, 0]
        flips.append(float(((a_low == need + 1) &
                            (a_high != need + 1)).float().mean()))
    return {
        "mean_target_probability_effect": statistics.mean(effects),
        "mean_goal_flip_rate": statistics.mean(flips),
        "per_need_probability_effect": effects,
        "per_need_goal_flip_rate": flips,
    }


@torch.no_grad()
def evaluate_aha(agent: AnticipatoryHomeostaticAgent,
                 dataset: AnticipatoryHomeostasisDataset,
                 device: torch.device,
                 *, n_batches: int = 10,
                 batch_size: int = 64,
                 seed: int = 20260721) -> dict:
    """Evalúa mundo normal, lesiones y control estacionario pareados."""

    cfg = dataset.cfg
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    records = []
    normal_cells, lesion_cells, shuffled_cells, stationary_cells = [], [], [], []
    transplant_cells = []
    hbp_lesion_cells = []
    agent.eval()

    for batch_idx in range(n_batches):
        scenario = dataset.batch(batch_size, hazards=True).to(device)
        normal = aha_metrics(
            agent.rollout(scenario, cfg, hard_actions=True), scenario, cfg)
        lesion = aha_metrics(
            agent.rollout(scenario, cfg, hard_actions=True,
                          prediction_mode="lesion"), scenario, cfg)
        shuffled_scenario = _shuffle_cues(scenario, generator)
        shuffled = aha_metrics(
            agent.rollout(shuffled_scenario, cfg, hard_actions=True),
            shuffled_scenario, cfg)
        stationary_scenario = dataset.batch(
            batch_size, hazards=False).to(device)
        stationary = aha_metrics(
            agent.rollout(stationary_scenario, cfg, hard_actions=True),
            stationary_scenario, cfg)
        transplant = need_transplant_effect(
            agent, stationary_scenario, cfg)
        hbp_lesion = None
        if agent.use_hbp:
            hbp_lesion = aha_metrics(
                agent.rollout(scenario, cfg, hard_actions=True,
                              hbp_mode="lesion"), scenario, cfg)
            hbp_lesion_cells.append(hbp_lesion)

        normal_cells.append(normal)
        lesion_cells.append(lesion)
        shuffled_cells.append(shuffled)
        stationary_cells.append(stationary)
        transplant_cells.append(transplant)
        records.append({
            "batch": batch_idx, "normal": normal,
            "prediction_lesion": lesion, "cue_shuffle": shuffled,
            "stationary": stationary, "need_transplant": transplant,
            "hbp_modulation_lesion": hbp_lesion,
        })

    normal = _mean_metrics(normal_cells)
    lesion = _mean_metrics(lesion_cells)
    shuffled = _mean_metrics(shuffled_cells)
    stationary = _mean_metrics(stationary_cells)
    hbp_lesion = (_mean_metrics(hbp_lesion_cells)
                  if hbp_lesion_cells else None)
    transplant = {
        "mean_target_probability_effect": statistics.mean(
            x["mean_target_probability_effect"] for x in transplant_cells),
        "mean_goal_flip_rate": statistics.mean(
            x["mean_goal_flip_rate"] for x in transplant_cells),
    }
    return {
        "variant": agent.cfg.variant,
        "design": {
            "external_goal_supplied": False,
            "future_disturbance_visible_to_policy": False,
            "raw_cue_visible_to_executive": False,
            "predictor_input": "current_cue_and_previous_disturbance_only",
            "action_effect_delay": cfg.action_delay,
            "cue_lead": cfg.cue_lead,
            "prediction_horizon": cfg.prediction_horizon,
            "primary_mechanism_intervention": "prediction_lesion",
            "paired_worlds_across_interventions": True,
        },
        "conditions": {
            "normal": normal,
            "prediction_lesion": lesion,
            "cue_shuffle": shuffled,
            "stationary": stationary,
            "hbp_modulation_lesion": hbp_lesion,
        },
        "causal_effects": {
            "prediction_on_survival": (
                normal["survival_rate"] - lesion["survival_rate"]),
            "prediction_on_violation_rate": (
                lesion["violation_rate"] - normal["violation_rate"]),
            "prediction_on_blank_initiative": (
                normal["blank_initiative_rate"]
                - lesion["blank_initiative_rate"]),
            "valid_cue_on_survival": (
                normal["survival_rate"] - shuffled["survival_rate"]),
            "valid_cue_on_blank_initiative": (
                normal["blank_initiative_rate"]
                - shuffled["blank_initiative_rate"]),
            "need_transplant": transplant,
            "hbp_modulation_on_survival": (
                normal["survival_rate"] - hbp_lesion["survival_rate"]
                if hbp_lesion is not None else None),
            "hbp_modulation_on_blank_initiative": (
                normal["blank_initiative_rate"]
                - hbp_lesion["blank_initiative_rate"]
                if hbp_lesion is not None else None),
        },
        "records": records,
    }
