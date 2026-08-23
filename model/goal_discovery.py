"""Explorador recurrente que propone objetivos continuos no enumerados."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.goal_discovery import (
    GoalDiscoveryEnvironmentConfig, GoalDiscoveryScenario,
    affordance_response)


@dataclass
class GoalDiscoveryControllerConfig:
    n_needs: int = 3
    goal_dim: int = 2
    n_probes: int = 4
    d_model: int = 96
    variant: str = "discoverer"        # discoverer | no_feedback
    beta_response_prediction: float = 0.050
    beta_probe_diversity: float = 0.010
    min_probe_distance: float = 0.22
    urgency_temperature: float = 12.0
    probe_residual_scale: float = 0.25
    goal_residual_scale: float = 0.05


class ContinuousGoalDiscoverer(nn.Module):
    """Explora un paisaje causal y sintetiza una coordenada-objetivo.

    ``centers`` pertenece a la dinámica del mundo: nunca se concatena a una
    entrada. El agente sólo recibe las respuestas de las sondas que él mismo
    elige. El control ``no_feedback`` ejecuta las mismas sondas y la misma RBF,
    pero corta esa respuesta antes del estado recurrente.
    """

    VARIANTS = ("discoverer", "no_feedback")

    def __init__(self, cfg: GoalDiscoveryControllerConfig):
        super().__init__()
        if cfg.variant not in self.VARIANTS:
            raise ValueError(f"variant debe pertenecer a {self.VARIANTS}")
        self.cfg = cfg
        n, d, g = cfg.n_needs, cfg.d_model, cfg.goal_dim
        body_dim = 2 * n + 2 * g
        self.body_proj = nn.Sequential(
            nn.Linear(body_dim, d), nn.Tanh())
        self.probe_embedding = nn.Embedding(cfg.n_probes, d)
        self.query_head = nn.Sequential(
            nn.Linear(d, d), nn.Tanh(), nn.Linear(d, g))
        recurrent_in = g + n + 1
        self.feedback_proj = nn.Sequential(
            nn.Linear(recurrent_in, d), nn.Tanh())
        self.explorer = nn.GRUCell(d, d)
        self.goal_head = nn.Sequential(
            nn.Linear(d, d), nn.Tanh(), nn.Linear(d, g))
        self.response_predictor = nn.Sequential(
            nn.Linear(d + g, d), nn.Tanh(), nn.Linear(d, n))
        angles = torch.arange(cfg.n_probes, dtype=torch.float32) * (
            2.0 * torch.pi / cfg.n_probes)
        anchors = 0.5 + 0.34 * torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=-1)
        anchors = anchors.clamp(0.05, 0.95)
        self.register_buffer(
            "probe_anchor_logits", torch.logit(anchors), persistent=True)
        for module in (self.body_proj, self.probe_embedding, self.query_head,
                       self.feedback_proj, self.goal_head,
                       self.response_predictor):
            module.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    @staticmethod
    def _to_feasible(unit_point: torch.Tensor,
                     scenario: GoalDiscoveryScenario) -> torch.Tensor:
        point = (scenario.feasible_low + unit_point
                 * (scenario.feasible_high - scenario.feasible_low))
        # Protege la invariante también frente a un ulp de redondeo cuando la
        # coordenada unitaria es exactamente 0 o 1 en FP32.
        return torch.maximum(
            torch.minimum(point, scenario.feasible_high),
            scenario.feasible_low)

    @staticmethod
    def _infer_centers(queries: torch.Tensor, responses: torch.Tensor,
                       sigma: float) -> torch.Tensor:
        """Trilatera centros RBF sin observarlos ni supervisarlos.

        Para ``r=exp(-||q-c||²/(2 sigma²))``, restar la distancia
        cuadrada del primer sondeo elimina ``||c||²`` y deja un sistema
        lineal sobredeterminado. ``pinv`` mantiene el mecanismo diferenciable.
        """

        squared_distance = -2.0 * sigma ** 2 * torch.log(
            responses.clamp(1e-6, 1.0))
        delta_q = queries[:, 1:] - queries[:, :1]
        matrix = -2.0 * delta_q
        query_norm = queries.square().sum(dim=-1)
        rhs = (squared_distance[:, 1:] - squared_distance[:, :1]
               - (query_norm[:, 1:] - query_norm[:, :1]).unsqueeze(-1))
        # (B,D,P-1) @ (B,P-1,N) -> (B,D,N) -> (B,N,D)
        return (torch.linalg.pinv(matrix) @ rhs).transpose(1, 2)

    def rollout(self, scenario: GoalDiscoveryScenario,
                env_cfg: GoalDiscoveryEnvironmentConfig,
                *, feedback_mode: str = "normal",
                goal_mode: str = "normal",
                decoder_sigma_scale: float = 1.0) \
            -> dict[str, torch.Tensor]:
        if feedback_mode not in ("normal", "lesion", "shuffle"):
            raise ValueError("feedback_mode debe ser normal, lesion o shuffle")
        if goal_mode not in ("normal", "reflect"):
            raise ValueError("goal_mode debe ser normal o reflect")
        if decoder_sigma_scale <= 0:
            raise ValueError("decoder_sigma_scale debe ser positivo")
        scenario.validate(env_cfg)
        if (env_cfg.n_needs != self.cfg.n_needs
                or env_cfg.goal_dim != self.cfg.goal_dim
                or env_cfg.n_probes != self.cfg.n_probes):
            raise ValueError("configuración de entorno y agente incompatible")

        error = scenario.setpoints - scenario.initial_levels
        body_features = torch.cat([
            scenario.initial_levels, error,
            scenario.feasible_low, scenario.feasible_high], dim=-1)
        state = self.body_proj(body_features)
        batch = scenario.batch_size
        query_units, queries, responses, controlled_responses = [], [], [], []

        for probe_idx in range(env_cfg.n_probes):
            index = torch.full(
                (batch,), probe_idx, device=state.device, dtype=torch.long)
            residual = self.query_head(state + self.probe_embedding(index))
            query_unit = torch.sigmoid(
                self.probe_anchor_logits[index]
                + self.cfg.probe_residual_scale * residual)
            query = self._to_feasible(query_unit, scenario)
            response = affordance_response(
                query, scenario.centers, env_cfg.response_sigma)
            controlled = response
            if self.cfg.variant == "no_feedback" or feedback_mode == "lesion":
                controlled = torch.zeros_like(response)
            elif feedback_mode == "shuffle":
                controlled = torch.roll(response, shifts=1, dims=0)
            step = torch.full(
                (batch, 1), (probe_idx + 1) / env_cfg.n_probes,
                device=state.device, dtype=state.dtype)
            recurrent_features = torch.cat(
                [query_unit, controlled, step], dim=-1)
            state = self.explorer(
                self.feedback_proj(recurrent_features), state)
            query_units.append(query_unit)
            queries.append(query)
            responses.append(response)
            controlled_responses.append(controlled)

        queries_by_time = torch.stack(queries, dim=1)
        controlled_by_time = torch.stack(controlled_responses, dim=1)
        estimated_centers = self._infer_centers(
            queries_by_time, controlled_by_time,
            env_cfg.response_sigma * decoder_sigma_scale)
        estimated_centers = torch.maximum(
            torch.minimum(estimated_centers,
                          scenario.feasible_high.unsqueeze(1)),
            scenario.feasible_low.unsqueeze(1))
        target_need = error.argmax(dim=-1)
        rows = torch.arange(batch, device=state.device)
        estimated_target = estimated_centers[rows, target_need]
        estimated_unit = ((estimated_target - scenario.feasible_low)
                          / (scenario.feasible_high
                             - scenario.feasible_low).clamp_min(1e-6))
        goal_unit = torch.clamp(
            estimated_unit
            + self.cfg.goal_residual_scale * torch.tanh(
                self.goal_head(state)), 0.0, 1.0)
        if goal_mode == "reflect":
            # Conserva factibilidad y distancia al centro de la caja, pero
            # cambia exclusivamente el contenido del objetivo propuesto.
            goal_unit = 1.0 - goal_unit
        goal = self._to_feasible(goal_unit, scenario)
        goal_response = affordance_response(
            goal, scenario.centers, env_cfg.response_sigma)
        final_levels = torch.clamp(
            scenario.initial_levels
            + env_cfg.restoration_scale * goal_response, 0.0, 1.0)

        query_units_by_time = torch.stack(query_units, dim=1)
        responses_by_time = torch.stack(responses, dim=1)
        predicted_response = torch.sigmoid(self.response_predictor(
            torch.cat([state, goal_unit], dim=-1)))
        response_prediction = F.mse_loss(
            predicted_response, goal_response.detach())

        if env_cfg.n_probes > 1:
            # Sólo pares dentro de cada mundo; mezclar episodios produciría una
            # falsa diversidad entre paisajes distintos.
            penalties = []
            for left in range(env_cfg.n_probes):
                for right in range(left + 1, env_cfg.n_probes):
                    distance = (query_units_by_time[:, left]
                                - query_units_by_time[:, right]).norm(dim=-1)
                    penalties.append(F.relu(
                        self.cfg.min_probe_distance - distance).square())
            diversity = torch.stack(penalties, dim=1).mean()
        else:
            diversity = torch.zeros((), device=state.device)
        deviations = final_levels - scenario.setpoints
        # El cuerpo, no una etiqueta de evaluación, decide qué restaurar
        # primero. La softmax suave evita que mejorar dos necesidades menores
        # esconda el fracaso sobre el déficit que amenaza la viabilidad.
        urgency = torch.softmax(
            F.relu(error) * self.cfg.urgency_temperature, dim=-1).detach()
        homeostatic = (urgency * deviations.square()).sum(dim=-1).mean()
        total = (homeostatic
                 + self.cfg.beta_response_prediction * response_prediction
                 + self.cfg.beta_probe_diversity * diversity)
        return {
            "query_units": query_units_by_time,
            "queries": queries_by_time,
            "probe_responses": responses_by_time,
            "controlled_probe_responses": controlled_by_time,
            "estimated_centers": estimated_centers,
            "estimated_target_need": target_need,
            "goal_unit": goal_unit,
            "goal": goal,
            "goal_response": goal_response,
            "predicted_goal_response": predicted_response,
            "final_levels": final_levels,
            "urgency": urgency,
            "losses": {
                "total": total,
                "homeostatic": homeostatic,
                "response_prediction": response_prediction,
                "probe_diversity": diversity,
            },
        }
