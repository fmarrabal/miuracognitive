"""Controlador de metas endógenas con prioridad latente persistente."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.endogenous_goals import (
    EndogenousGoalEnvironmentConfig, EndogenousGoalScenario)


@dataclass
class EndogenousGoalControllerConfig:
    n_needs: int = 3
    d_model: int = 64
    variant: str = "goal_memory"       # reactive | goal_memory
    proposal_temperature: float = 0.55
    action_temperature: float = 0.70
    goal_action_gain: float = 4.0
    viability_weight: float = 12.0
    beta_proposal: float = 0.020
    beta_alignment: float = 0.020
    beta_gate: float = 0.020
    beta_goal_change: float = 0.002
    beta_completion: float = 0.010
    switch_urgency_margin: float = 0.05
    gate_urgency_gain: float = 40.0

    @property
    def n_actions(self) -> int:
        return self.n_needs + 1


class EndogenousGoalAgent(nn.Module):
    """Elige una prioridad desde el cuerpo y puede mantenerla entre gaps.

    Ambos brazos ejecutan las mismas redes. ``reactive`` sustituye cada tick el
    estado latente por la propuesta actual; ``goal_memory`` aprende una puerta
    que arbitra entre esa propuesta y la prioridad anterior. La política nunca
    recibe ``initial_target``, ``rival_target``, ``critical`` ni el progreso
    físico del proyecto.
    """

    VARIANTS = ("reactive", "goal_memory")

    def __init__(self, cfg: EndogenousGoalControllerConfig):
        super().__init__()
        if cfg.variant not in self.VARIANTS:
            raise ValueError(f"variant debe pertenecer a {self.VARIANTS}")
        self.cfg = cfg
        n, d = cfg.n_needs, cfg.d_model
        # niveles + error + acción previa + disponibilidad de trabajo
        obs_dim = 2 * n + cfg.n_actions + 1
        self.obs_proj = nn.Sequential(nn.Linear(obs_dim, d), nn.Tanh())
        self.goal_proposal = nn.Linear(d, n)
        self.goal_update = nn.Sequential(
            nn.Linear(d + n, max(16, d // 2)), nn.Tanh(),
            nn.Linear(max(16, d // 2), 1))
        self.action_head = nn.Sequential(
            nn.Linear(d + n, d), nn.Tanh(), nn.Linear(d, cfg.n_actions))
        for module in (self.obs_proj, self.goal_proposal,
                       self.goal_update, self.action_head):
            module.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _straight_through_one_hot(probs: torch.Tensor,
                                  enabled: bool) -> torch.Tensor:
        hard = F.one_hot(
            probs.argmax(dim=-1), probs.shape[-1]).to(probs.dtype)
        return hard + probs - probs.detach() if enabled else probs

    def rollout(self, scenario: EndogenousGoalScenario,
                env_cfg: EndogenousGoalEnvironmentConfig,
                *, hard_actions: bool = False,
                straight_through: bool = True,
                memory_mode: str = "normal",
                goal_rotation_tick: int | None = None) -> dict[str, torch.Tensor]:
        if memory_mode not in ("normal", "lesion"):
            raise ValueError("memory_mode debe ser normal o lesion")
        scenario.validate(env_cfg)
        if env_cfg.n_needs != self.cfg.n_needs:
            raise ValueError("n_needs incompatible")
        if goal_rotation_tick is not None and not (
                0 <= goal_rotation_tick < env_cfg.horizon):
            raise ValueError("goal_rotation_tick fuera del horizonte")

        levels = scenario.initial_levels
        batch, n = levels.shape
        dtype, device = levels.dtype, levels.device
        previous_action = torch.zeros(
            batch, self.cfg.n_actions, device=device, dtype=dtype)
        previous_action[:, 0] = 1.0
        goal_state = torch.full(
            (batch, n), 1.0 / n, device=device, dtype=dtype)
        progress = torch.zeros_like(levels)

        level_history = [levels]
        progress_history = [progress]
        action_probs_history, action_history = [], []
        proposal_history, goal_history, gate_history = [], [], []
        completion_history = []
        proposal_losses, alignment_losses, gate_losses, change_losses = [], [], [], []

        for tick in range(env_cfg.horizon):
            opportunity = scenario.opportunities[:, tick:tick + 1]
            error = scenario.setpoints - levels
            features = torch.cat(
                [levels, error, previous_action, opportunity], dim=-1)
            encoded = self.obs_proj(features)
            proposal_logits = self.goal_proposal(encoded)
            proposal_probs = F.softmax(
                proposal_logits / max(self.cfg.proposal_temperature, 1e-4),
                dim=-1)
            proposal_state = self._straight_through_one_hot(
                proposal_probs, straight_through and not hard_actions)
            urgency_now = F.relu(error).detach()
            current_goal_urgency = (
                goal_state.detach() * urgency_now).sum(
                    dim=-1, keepdim=True)
            best_urgency = urgency_now.max(dim=-1, keepdim=True).values
            urgency_advantage = best_urgency - current_goal_urgency
            learned_gate_residual = self.goal_update(
                torch.cat([encoded, goal_state], dim=-1))
            gate = torch.sigmoid(
                learned_gate_residual
                + self.cfg.gate_urgency_gain * (
                    urgency_advantage - self.cfg.switch_urgency_margin))

            old_goal = goal_state
            if tick == 0:
                candidate = proposal_state
                gate = torch.ones_like(gate)
            elif self.cfg.variant == "reactive" or memory_mode == "lesion":
                # Se ejecuta la puerta pero se corta la recurrencia causal.
                candidate = proposal_state
            else:
                candidate = gate * proposal_state + (1.0 - gate) * goal_state
            candidate = candidate / candidate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            goal_state = self._straight_through_one_hot(
                candidate, straight_through and not hard_actions)
            if goal_rotation_tick == tick:
                # Intervención de contenido: misma intensidad, otra necesidad.
                goal_state = torch.roll(goal_state, shifts=1, dims=-1)

            logits = self.action_head(torch.cat([encoded, goal_state], dim=-1))
            logits = torch.cat([
                logits[:, :1],
                logits[:, 1:] + self.cfg.goal_action_gain * goal_state,
            ], dim=-1)
            # En los gaps se ejecuta todo el controlador, pero el mundo sólo
            # permite no-op. Así la última acción no transporta la meta.
            masked_logits = logits.clone()
            unavailable = opportunity.squeeze(-1) < 0.5
            masked_logits[unavailable, 1:] = -1e4
            action_probs = F.softmax(
                masked_logits / max(self.cfg.action_temperature, 1e-4), dim=-1)
            hard = F.one_hot(
                action_probs.argmax(dim=-1), self.cfg.n_actions).to(dtype)
            if hard_actions:
                action = hard
            elif straight_through:
                action = hard + action_probs - action_probs.detach()
            else:
                action = action_probs

            work = opportunity
            need_action = action[:, 1:]
            work_candidate = (progress + 1.0) * need_action
            next_progress = work * work_candidate + (1.0 - work) * progress
            soft_completion = torch.sigmoid(
                8.0 * (next_progress - env_cfg.project_length + 0.5))
            hard_completion = (
                next_progress >= env_cfg.project_length).to(dtype)
            completion = (hard_completion + soft_completion
                          - soft_completion.detach())
            completion = completion * work
            next_progress = next_progress * (1.0 - completion)

            levels = torch.clamp(
                levels + env_cfg.project_effect * completion
                - env_cfg.basal_drain
                - scenario.disturbances[:, tick], 0.0, 1.0)

            urgency = F.relu(error).detach()
            urgency_target = F.softmax(12.0 * urgency, dim=-1)
            proposal_losses.append(-(urgency_target
                                     * proposal_probs.clamp_min(1e-8).log()
                                     ).sum(dim=-1).mean())
            # Incluye la masa no-op: una meta que nunca llega a acción no
            # satisface el enlace meta→conducta.
            alignment = -(goal_state.detach()
                          * action_probs[:, 1:].clamp_min(1e-8).log()
                          ).sum(dim=-1)
            alignment_losses.append(
                (alignment * opportunity.squeeze(-1)).sum()
                / opportunity.sum().clamp_min(1.0))
            current_urgency = (old_goal.detach() * urgency).sum(dim=-1)
            best_urgency_flat = urgency.max(dim=-1).values
            gate_target = ((best_urgency_flat - current_urgency)
                           > self.cfg.switch_urgency_margin).to(dtype)
            gate_losses.append(F.binary_cross_entropy(
                gate.squeeze(-1), gate_target))
            change_losses.append((goal_state - old_goal).abs().mean())

            level_history.append(levels)
            progress_history.append(next_progress)
            action_probs_history.append(action_probs)
            action_history.append(hard)
            proposal_history.append(proposal_probs)
            goal_history.append(goal_state)
            gate_history.append(gate.squeeze(-1))
            completion_history.append(hard_completion)
            previous_action = action
            progress = next_progress

        levels_by_time = torch.stack(level_history, dim=1)
        action_probs_by_time = torch.stack(action_probs_history, dim=1)
        actions = torch.stack(action_history, dim=1)
        proposals = torch.stack(proposal_history, dim=1)
        goals = torch.stack(goal_history, dim=1)
        gates = torch.stack(gate_history, dim=1)
        completions = torch.stack(completion_history, dim=1)
        progress_by_time = torch.stack(progress_history, dim=1)

        post = levels_by_time[:, 1:]
        deviations = post - scenario.setpoints.unsqueeze(1)
        violations = F.relu(env_cfg.viability_threshold - post)
        work_mask = scenario.opportunities
        effort = ((1.0 - action_probs_by_time[:, :, 0]) * work_mask).sum()
        effort = effort / work_mask.sum().clamp_min(1.0)
        homeostatic = deviations.square().mean()
        viability = violations.square().mean()
        proposal_loss = torch.stack(proposal_losses).mean()
        alignment_loss = torch.stack(alignment_losses).mean()
        gate_loss = torch.stack(gate_losses[1:]).mean()
        goal_change = torch.stack(change_losses[1:]).mean()
        completion_rate = completions.sum(dim=-1).mean()
        total = (homeostatic + self.cfg.viability_weight * viability
                 + env_cfg.action_cost * effort
                 + self.cfg.beta_proposal * proposal_loss
                 + self.cfg.beta_alignment * alignment_loss
                 + self.cfg.beta_gate * gate_loss
                 + self.cfg.beta_goal_change * goal_change
                 - self.cfg.beta_completion * completion_rate)

        return {
            "levels": levels_by_time,
            "progress": progress_by_time,
            "action_probs": action_probs_by_time,
            "actions": actions,
            "action_ids": actions.argmax(dim=-1),
            "proposal_probs": proposals,
            "proposal_ids": proposals.argmax(dim=-1),
            "goal_probs": goals,
            "goal_ids": goals.argmax(dim=-1),
            "update_gates": gates,
            "completions": completions,
            "losses": {
                "total": total,
                "homeostatic": homeostatic,
                "viability": viability,
                "effort": effort,
                "proposal": proposal_loss,
                "alignment": alignment_loss,
                "gate": gate_loss,
                "goal_change": goal_change,
                "completion": completion_rate,
            },
        }
