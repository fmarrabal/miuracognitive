"""Automodelo explícito de consecuencias motoras sobre un cuerpo de onda."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data.self_model import (
    SelfModelEnvironmentConfig, SelfModelScenario,
    canonical_actuator_effects, wave_companion_spectral_radius)
from model.hbp import build_chain_laplacian


@dataclass(frozen=True)
class SelfModelControllerConfig:
    n_nodes: int = 5
    variant: str = "self_model"       # self_model | stale_model
    model_update_gain: float = 1.0
    action_cost: float = 0.002


class WaveBodySelfModel(nn.Module):
    """Predice efectos de acciones, detecta daño y replantea el control.

    La matriz real de actuadores sólo aparece en el paso de la planta. La
    política ve estado/velocidad y mantiene ``effect_model``: una estimación de
    qué fuerza produce cada copia eferente. La actualiza desde el residuo entre
    el siguiente estado observado y la predicción de la dinámica pasiva.
    """

    VARIANTS = ("self_model", "stale_model")

    def __init__(self, cfg: SelfModelControllerConfig,
                 env_cfg: SelfModelEnvironmentConfig):
        super().__init__()
        if cfg.variant not in self.VARIANTS:
            raise ValueError(f"variant debe pertenecer a {self.VARIANTS}")
        if cfg.n_nodes != env_cfg.n_nodes:
            raise ValueError("n_nodes incompatible")
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.register_buffer("laplacian", build_chain_laplacian(cfg.n_nodes))
        self.register_buffer(
            "nominal_effect_model", canonical_actuator_effects(env_cfg))

    def passive_prediction(self, current: torch.Tensor,
                           previous: torch.Tensor) -> torch.Tensor:
        cfg = self.env_cfg
        stiffness = (cfg.omega0 ** 2 * current
                     + cfg.wave_speed ** 2 * torch.einsum(
                         "ij,bj->bi", self.laplacian, current))
        damping = (2.0 * cfg.zeta * cfg.omega0 / cfg.dt) * (
            current - previous)
        return (2.0 * current - previous
                - cfg.dt ** 2 * (stiffness + damping))

    def _choose_action(self, passive: torch.Tensor,
                       effect_model: torch.Tensor,
                       true_effects: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        # Contrafactuales (B,A,N): mismo futuro pasivo, una fuerza distinta.
        predicted_next = passive.unsqueeze(1) + effect_model.transpose(1, 2)
        costs = predicted_next.square().mean(dim=-1)
        costs[:, 1:] += self.cfg.action_cost
        action = costs.argmin(dim=-1)

        true_next = passive.unsqueeze(1) + true_effects.transpose(1, 2)
        true_costs = true_next.square().mean(dim=-1)
        true_costs[:, 1:] += self.cfg.action_cost
        oracle_action = true_costs.argmin(dim=-1)
        return action, oracle_action

    def rollout(self, scenario: SelfModelScenario,
                *, update_mode: str = "normal",
                model_mode: str = "normal") -> dict[str, torch.Tensor | float]:
        if update_mode not in ("normal", "lesion", "shuffle"):
            raise ValueError("update_mode debe ser normal, lesion o shuffle")
        if model_mode not in ("normal", "rotate"):
            raise ValueError("model_mode debe ser normal o rotate")
        cfg = self.env_cfg
        scenario.validate(cfg)
        batch = scenario.batch_size
        device = scenario.intact_effects.device
        rows = torch.arange(batch, device=device)
        current = torch.zeros(batch, cfg.n_nodes, device=device)
        previous = torch.zeros_like(current)
        effect_model = self.nominal_effect_model.unsqueeze(0).expand(
            batch, -1, -1).clone()

        before_history, after_history = [], []
        action_history, oracle_history = [], []
        predicted_history, actual_history, error_history = [], [], []
        model_target_history = []

        for tick in range(cfg.horizon):
            if tick == 0:
                current[rows, scenario.target_node] -= scenario.first_shock
                previous[rows, scenario.target_node] -= scenario.first_shock
            if tick == cfg.damage_tick:
                # Segunda carga estandarizada: fija el mismo déficit con
                # velocidad local nula, en vez de sumarlo a la fase residual
                # de la primera oscilación (que podría cancelar la crisis).
                current[rows, scenario.target_node] = -scenario.damage_shock
                previous[rows, scenario.target_node] = -scenario.damage_shock
                if model_mode == "rotate":
                    nonzero = torch.roll(effect_model[:, :, 1:], shifts=1,
                                         dims=-1)
                    effect_model = torch.cat(
                        [effect_model[:, :, :1], nonzero], dim=-1)

            active_effects = (scenario.intact_effects
                              if tick < cfg.damage_tick
                              else scenario.damaged_effects)
            passive = self.passive_prediction(current, previous)
            action, oracle_action = self._choose_action(
                passive, effect_model, active_effects)
            predicted_effect = effect_model.transpose(1, 2)[rows, action]
            actual_effect = active_effects.transpose(1, 2)[rows, action]
            next_state = passive + actual_effect
            prediction_error = (predicted_effect - actual_effect).square().mean(
                dim=-1)

            can_update = (tick < cfg.damage_tick
                          or (self.cfg.variant == "self_model"
                              and update_mode != "lesion"))
            if can_update:
                update_action = action
                if update_mode == "shuffle" and tick >= cfg.damage_tick:
                    # Conserva el residuo observado pero rompe su copia eferente:
                    # se atribuye al comando siguiente, nunca al ejecutado.
                    update_action = torch.where(
                        action == 0, action,
                        1 + (action % (cfg.n_actions - 1)))
                active_rows = rows[update_action != 0]
                if active_rows.numel() > 0:
                    columns = update_action[active_rows]
                    old = effect_model[active_rows, :, columns]
                    observed = actual_effect[active_rows]
                    gain = self.cfg.model_update_gain
                    effect_model = effect_model.clone()
                    effect_model[active_rows, :, columns] = (
                        (1.0 - gain) * old + gain * observed)

            before_history.append(current.clone())
            after_history.append(next_state.clone())
            action_history.append(action)
            oracle_history.append(oracle_action)
            predicted_history.append(predicted_effect)
            actual_history.append(actual_effect)
            error_history.append(prediction_error)
            model_target_history.append(
                effect_model[rows, :, 1 + scenario.target_node])
            previous, current = current, next_state

        return {
            "states_before": torch.stack(before_history, dim=1),
            "states_after": torch.stack(after_history, dim=1),
            "actions": torch.stack(action_history, dim=1),
            "oracle_actions": torch.stack(oracle_history, dim=1),
            "predicted_action_effects": torch.stack(
                predicted_history, dim=1),
            "actual_action_effects": torch.stack(actual_history, dim=1),
            "executed_prediction_mse": torch.stack(error_history, dim=1),
            "damaged_primary_model_history": torch.stack(
                model_target_history, dim=1),
            "final_effect_model": effect_model,
            "passive_spectral_radius": wave_companion_spectral_radius(cfg),
        }
