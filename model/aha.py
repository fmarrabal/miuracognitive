"""Controladores para Anticipatory Homeostatic Agency (AHA).

La arquitectura separa explícitamente:

1. predictor causal de perturbaciones futuras (alostasis);
2. ejecutivo recurrente compartido por todos los controles;
3. selección endógena de prioridad: no-op o una de las necesidades;
4. HBP opcional de primer/segundo orden que modula esa prioridad.

La primera fase de AHA usa un vocabulario de necesidades interpretable y fijo.
Por tanto prueba *selección* endógena e iniciativa anticipatoria, no todavía el
descubrimiento abierto de nuevas metas.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.aha import (AHAEnvironmentConfig, AHAScenario,
                      future_disturbance_targets)
from .hbp import HBPConfig, HomeostaticBackgroundProcessor


@dataclass
class AHAControllerConfig:
    n_needs: int = 3
    d_model: int = 64
    d_predictor: int = 48
    variant: str = "hbp_full"  # reactive | gating_wm | hbp_first | hbp_full
    action_temperature: float = 0.7
    hbp_priority_gain: float = 2.0
    violation_weight: float = 12.0
    beta_prediction: float = 1.0
    prediction_positive_weight: float = 5.0
    beta_intero: float = 0.002
    beta_homeo: float = 0.0001
    beta_stab: float = 0.01

    @property
    def n_actions(self) -> int:
        return self.n_needs + 1


def _aha_hbp_config(variant: str, n_needs: int) -> HBPConfig:
    # Tres necesidades no requieren el VEI d_h=64 del transformer completo.
    # Conservamos las proporciones fisio/afecto/umbral con d_h=32.
    cfg = HBPConfig(n_nodes=n_needs, d_f=8, d_e=8, d_u=16, d_intero=4)
    if variant == "hbp_first":
        cfg.order = 1
        cfg.zeta_init = 2.5
        cfg.zeta_min = 2.0
        cfg.c_init = 0.3
        cfg.c_max = 0.4
        cfg.omega0_init = 0.3
        cfg.omega0_max = 0.45
    elif variant == "hbp_full":
        cfg.order = 2
        cfg.zeta_init = 0.5
        cfg.zeta_min = 0.05
        cfg.c_init = 0.4
        cfg.c_max = 0.7
    else:
        raise ValueError(f"variante HBP desconocida: {variant}")
    return cfg


class AnticipatoryHomeostaticAgent(nn.Module):
    """Agente mínimo persistente para la primera fase de AHA."""

    VARIANTS = ("reactive", "gating_wm", "hbp_first", "hbp_full")

    def __init__(self, cfg: AHAControllerConfig):
        super().__init__()
        if cfg.variant not in self.VARIANTS:
            raise ValueError(f"variant debe pertenecer a {self.VARIANTS}")
        self.cfg = cfg
        n, d = cfg.n_needs, cfg.d_model

        # Predictor alostático. Sólo ve cue_t y la perturbación ya observada en
        # t-1; nunca disturbance_t ni el futuro que intenta predecir.
        self.predictor = nn.GRUCell(2 * n, cfg.d_predictor)
        self.predictor_head = nn.Linear(cfg.d_predictor, n)

        # El ejecutivo no recibe la señal exógena cruda: recibe su estimación de
        # amenaza, además de interocepción y la acción previa.
        executive_in = 4 * n + cfg.n_actions
        self.obs_proj = nn.Sequential(
            nn.Linear(executive_in, d), nn.Tanh())
        self.executive = nn.GRUCell(d, d)
        self.goal_head = nn.Linear(d, cfg.n_actions)

        # Inicialización del camino compartido antes de construir el HBP para
        # conservar emparejamiento bit a bit entre variantes con la misma seed.
        for module in (self.predictor_head, self.obs_proj, self.goal_head):
            module.apply(self._init_weights)
        # Los eventos son escasos; softplus(0)=0.693 sería una predicción
        # inicial absurdamente alta y dominaría las primeras actualizaciones.
        nn.init.constant_(self.predictor_head.bias, -3.0)

        self.use_hbp = cfg.variant in ("hbp_first", "hbp_full")
        if self.use_hbp:
            hcfg = _aha_hbp_config(cfg.variant, n)
            self.hbp = HomeostaticBackgroundProcessor(hcfg)
            self.intero_norm = nn.LayerNorm(4)
            self.executive_to_hbp = nn.Linear(d, hcfg.d_h)
            self.hbp_priority = nn.Linear(hcfg.d_h, 1, bias=False)
            for module in (self.hbp, self.intero_norm,
                           self.executive_to_hbp, self.hbp_priority):
                module.apply(self._init_weights)
            # El contraste debe empezar exactamente en el controlador base.
            # La rama HBP se abre por gradiente; no inyecta un sesgo aleatorio
            # antes de haber observado una sola trayectoria.
            nn.init.zeros_(self.hbp_priority.weight)
            self.hbp.init_physics()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def predictor_parameters(self) -> list[nn.Parameter]:
        return list(self.predictor.parameters()) + list(
            self.predictor_head.parameters())

    @property
    def controller_parameters(self) -> list[nn.Parameter]:
        predictor_ids = {id(p) for p in self.predictor_parameters}
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in predictor_ids]

    def pin_fp32(self) -> "AnticipatoryHomeostaticAgent":
        if self.use_hbp:
            self.hbp.pin_fp32()
        return self

    def predict_disturbances(self, scenario: AHAScenario) -> torch.Tensor:
        """Predicción online (B,T,N) construida sólo desde cada prefijo."""

        batch, horizon, n = scenario.cues.shape
        hidden = torch.zeros(
            batch, self.cfg.d_predictor,
            device=scenario.cues.device, dtype=scenario.cues.dtype)
        previous_disturbance = torch.zeros(
            batch, n, device=scenario.cues.device,
            dtype=scenario.cues.dtype)
        predictions = []
        for tick in range(horizon):
            inp = torch.cat(
                [scenario.cues[:, tick], previous_disturbance], dim=-1)
            hidden = self.predictor(inp, hidden)
            predictions.append(F.softplus(self.predictor_head(hidden)))
            # disturbance_t sólo se hace observable después de predecir/actuar.
            previous_disturbance = scenario.disturbances[:, tick]
        return torch.stack(predictions, dim=1)

    def prediction_loss(self, scenario: AHAScenario,
                        env_cfg: AHAEnvironmentConfig) -> torch.Tensor:
        target = future_disturbance_targets(
            scenario.disturbances, env_cfg.prediction_horizon)
        prediction = self.predict_disturbances(scenario)
        weight = torch.ones_like(target)
        weight = torch.where(
            target > 0, weight * self.cfg.prediction_positive_weight, weight)
        return (weight * (prediction - target).square()).mean()

    def _hbp_bias(self, executive: torch.Tensor, error: torch.Tensor,
                  velocity: torch.Tensor, prediction: torch.Tensor,
                  previous_action: torch.Tensor) -> tuple[torch.Tensor,
                                                           torch.Tensor]:
        effort_by_need = previous_action[:, 1:]
        raw_intero = torch.stack(
            [error, -velocity, prediction, effort_by_need], dim=-1)
        intero = self.intero_norm(raw_intero)
        ext = self.executive_to_hbp(executive).unsqueeze(1).expand(
            -1, self.cfg.n_needs, -1)
        self.hbp.step(intero, ext_force=ext)
        displacement = self.hbp.h_t - self.hbp.h_star.unsqueeze(0)
        bias = self.hbp_priority(displacement).squeeze(-1)
        intero_loss = self.hbp.interoception_loss(intero.detach()).float()
        return bias, intero_loss

    def rollout(self, scenario: AHAScenario,
                env_cfg: AHAEnvironmentConfig,
                *, hard_actions: bool = False,
                straight_through: bool = True,
                prediction_mode: str = "normal",
                hbp_mode: str = "normal") -> dict[str, torch.Tensor]:
        """Ejecuta una vida corta sin acceso a metas o eventos futuros.

        ``prediction_mode='lesion'`` conserva exactamente el mismo cómputo del
        predictor pero sustituye su salida por cero antes del ejecutivo. Es la
        intervención causal principal sobre el mecanismo anticipatorio.
        """

        if prediction_mode not in ("normal", "lesion"):
            raise ValueError("prediction_mode debe ser normal o lesion")
        if hbp_mode not in ("normal", "lesion"):
            raise ValueError("hbp_mode debe ser normal o lesion")
        scenario.validate(env_cfg)
        if env_cfg.n_needs != self.cfg.n_needs:
            raise ValueError("n_needs incompatible entre entorno y agente")

        levels = scenario.initial_levels
        batch, n = levels.shape
        device, dtype = levels.device, levels.dtype
        previous_levels = levels
        previous_action = torch.zeros(
            batch, self.cfg.n_actions, device=device, dtype=dtype)
        previous_action[:, 0] = 1.0
        executive = torch.zeros(
            batch, self.cfg.d_model, device=device, dtype=dtype)
        effect_queue = [torch.zeros_like(levels)
                        for _ in range(env_cfg.action_delay)]

        if self.use_hbp:
            self.hbp.reset_state(batch, device=device, dtype=dtype)

        predictions = self.predict_disturbances(scenario)
        controlled_predictions = predictions
        if self.cfg.variant == "reactive" or prediction_mode == "lesion":
            controlled_predictions = torch.zeros_like(predictions)

        level_history = [levels]
        action_probs_history, action_history = [], []
        goal_logits_history, intero_losses = [], []

        for tick in range(env_cfg.horizon):
            velocity = levels - previous_levels
            error = scenario.setpoints - levels
            prediction = controlled_predictions[:, tick]
            executive_features = torch.cat(
                [levels, error, velocity, prediction, previous_action], dim=-1)
            executive = self.executive(
                self.obs_proj(executive_features), executive)
            logits = self.goal_head(executive)

            if self.use_hbp:
                hbp_bias, intero_loss = self._hbp_bias(
                    executive, error, velocity, prediction, previous_action)
                if hbp_mode == "lesion":
                    # Se integra el mismo campo y se conserva su coste; sólo se
                    # corta la flecha causal VEI -> prioridad de acción.
                    hbp_bias = torch.zeros_like(hbp_bias)
                need_logits = (logits[:, 1:] +
                               self.cfg.hbp_priority_gain * hbp_bias)
                logits = torch.cat([logits[:, :1], need_logits], dim=-1)
                intero_losses.append(intero_loss)

            probs = F.softmax(
                logits / max(self.cfg.action_temperature, 1e-4), dim=-1)
            hard = F.one_hot(
                probs.argmax(dim=-1), self.cfg.n_actions).to(dtype)
            if hard_actions:
                action = hard
            elif straight_through:
                action = hard + probs - probs.detach()
            else:
                action = probs

            arriving_effect = effect_queue.pop(0)
            effect_queue.append(env_cfg.action_effect * action[:, 1:])
            previous_levels = levels
            levels = torch.clamp(
                levels + arriving_effect - env_cfg.basal_drain
                - scenario.disturbances[:, tick], 0.0, 1.0)

            level_history.append(levels)
            action_probs_history.append(probs)
            action_history.append(hard)
            goal_logits_history.append(logits)
            previous_action = action

        levels_by_time = torch.stack(level_history, dim=1)
        action_probs = torch.stack(action_probs_history, dim=1)
        actions = torch.stack(action_history, dim=1)
        goal_logits = torch.stack(goal_logits_history, dim=1)
        post_levels = levels_by_time[:, 1:]
        deviations = post_levels - scenario.setpoints.unsqueeze(1)
        deficits = F.relu(-deviations)
        violations = F.relu(env_cfg.viability_threshold - post_levels)
        action_effort = 1.0 - action_probs[:, :, 0]

        # Homeostasis bilateral: tanto el déficit como la sobrecompensación
        # sostenida son desviaciones. Sin este término, la política degeneraría
        # en actuar siempre y saturar todos los niveles en 1.
        homeostatic = deviations.square().mean()
        viability = violations.square().mean()
        effort = action_effort.mean()
        control_loss = (homeostatic
                        + self.cfg.violation_weight * viability
                        + env_cfg.action_cost * effort)
        target = future_disturbance_targets(
            scenario.disturbances, env_cfg.prediction_horizon)
        predictor_mse = F.mse_loss(predictions, target)
        total = control_loss + self.cfg.beta_prediction * predictor_mse
        losses = {
            "control": control_loss,
            "homeostatic": homeostatic,
            "viability": viability,
            "effort": effort,
            "prediction": predictor_mse,
        }
        if self.use_hbp:
            intero = torch.stack(intero_losses).mean()
            homeo = self.hbp.homeostatic_loss()
            stab = self.hbp.stability_penalty()
            total = (total + self.cfg.beta_intero * intero
                     + self.cfg.beta_homeo * homeo
                     + self.cfg.beta_stab * stab)
            losses.update({"intero": intero, "hbp_homeo": homeo,
                           "stability": stab})
        losses["total"] = total

        return {
            "levels": levels_by_time,
            "action_probs": action_probs,
            "actions": actions,
            "action_ids": actions.argmax(dim=-1),
            "goal_logits": goal_logits,
            "predictions": predictions,
            "prediction_targets": target,
            "losses": losses,
        }
