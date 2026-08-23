"""Paisajes continuos para descubrir objetivos factibles no enumerados."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GoalDiscoveryEnvironmentConfig:
    n_needs: int = 3
    goal_dim: int = 2
    n_probes: int = 4
    response_sigma: float = 0.62
    restoration_scale: float = 0.46
    setpoint: float = 0.75
    dominant_min: float = 0.28
    dominant_max: float = 0.38
    secondary_min: float = 0.54
    secondary_max: float = 0.64
    other_min: float = 0.72
    other_max: float = 0.80

    def validate(self) -> None:
        if self.n_needs < 3:
            raise ValueError("goal discovery requiere al menos tres necesidades")
        if self.goal_dim != 2:
            raise ValueError("esta primera prueba continua usa goal_dim=2")
        if self.n_probes < 3:
            raise ValueError("se requieren al menos tres sondas")
        if self.response_sigma <= 0 or self.restoration_scale <= 0:
            raise ValueError("sigma y restoration_scale deben ser positivos")


@dataclass
class GoalDiscoveryScenario:
    initial_levels: torch.Tensor    # (B,N)
    setpoints: torch.Tensor         # (B,N)
    feasible_low: torch.Tensor      # (B,D), visible
    feasible_high: torch.Tensor     # (B,D), visible
    centers: torch.Tensor           # (B,N,D), dinámica oculta
    dominant_need: torch.Tensor     # (B,), sólo evaluación

    @property
    def batch_size(self) -> int:
        return int(self.initial_levels.shape[0])

    def to(self, device: torch.device | str) -> "GoalDiscoveryScenario":
        return GoalDiscoveryScenario(
            initial_levels=self.initial_levels.to(device),
            setpoints=self.setpoints.to(device),
            feasible_low=self.feasible_low.to(device),
            feasible_high=self.feasible_high.to(device),
            centers=self.centers.to(device),
            dominant_need=self.dominant_need.to(device),
        )

    def clone(self) -> "GoalDiscoveryScenario":
        return GoalDiscoveryScenario(
            initial_levels=self.initial_levels.clone(),
            setpoints=self.setpoints.clone(),
            feasible_low=self.feasible_low.clone(),
            feasible_high=self.feasible_high.clone(),
            centers=self.centers.clone(),
            dominant_need=self.dominant_need.clone(),
        )

    def validate(self, cfg: GoalDiscoveryEnvironmentConfig) -> None:
        b = self.batch_size
        if tuple(self.initial_levels.shape) != (b, cfg.n_needs):
            raise ValueError("initial_levels incompatible")
        if tuple(self.setpoints.shape) != (b, cfg.n_needs):
            raise ValueError("setpoints incompatible")
        if tuple(self.feasible_low.shape) != (b, cfg.goal_dim):
            raise ValueError("feasible_low incompatible")
        if tuple(self.feasible_high.shape) != (b, cfg.goal_dim):
            raise ValueError("feasible_high incompatible")
        if tuple(self.centers.shape) != (b, cfg.n_needs, cfg.goal_dim):
            raise ValueError("centers incompatible")
        if tuple(self.dominant_need.shape) != (b,):
            raise ValueError("dominant_need incompatible")
        tensors = (self.initial_levels, self.setpoints, self.feasible_low,
                   self.feasible_high, self.centers)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("escenario contiene NaN/Inf")
        if not (self.feasible_high > self.feasible_low).all():
            raise ValueError("región factible vacía")
        if ((self.centers < self.feasible_low.unsqueeze(1)).any()
                or (self.centers > self.feasible_high.unsqueeze(1)).any()):
            raise ValueError("centro fuera de la región factible")


class GoalDiscoveryDataset:
    """Cada batch contiene paisajes continuos nuevos casi seguramente únicos."""

    def __init__(self, cfg: GoalDiscoveryEnvironmentConfig | None = None,
                 seed: int = 0):
        self.cfg = cfg or GoalDiscoveryEnvironmentConfig()
        self.cfg.validate()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def batch(self, batch_size: int) -> GoalDiscoveryScenario:
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        c = self.cfg
        low = -1.0 + 0.4 * torch.rand(
            batch_size, c.goal_dim, generator=self.generator)
        high = 0.6 + 0.4 * torch.rand(
            batch_size, c.goal_dim, generator=self.generator)
        unit_centers = 0.06 + 0.88 * torch.rand(
            batch_size, c.n_needs, c.goal_dim, generator=self.generator)
        centers = low.unsqueeze(1) + unit_centers * (
            high - low).unsqueeze(1)

        levels = c.other_min + (c.other_max - c.other_min) * torch.rand(
            batch_size, c.n_needs, generator=self.generator)
        dominant = torch.arange(batch_size) % c.n_needs
        dominant = dominant[torch.randperm(
            batch_size, generator=self.generator)]
        for row in range(batch_size):
            target = int(dominant[row])
            levels[row, target] = c.dominant_min + (
                c.dominant_max - c.dominant_min) * torch.rand(
                    (), generator=self.generator)
            secondary_candidates = [need for need in range(c.n_needs)
                                    if need != target]
            secondary = secondary_candidates[int(torch.randint(
                0, len(secondary_candidates), (1,),
                generator=self.generator))]
            levels[row, secondary] = c.secondary_min + (
                c.secondary_max - c.secondary_min) * torch.rand(
                    (), generator=self.generator)
        scenario = GoalDiscoveryScenario(
            initial_levels=levels,
            setpoints=torch.full_like(levels, c.setpoint),
            feasible_low=low,
            feasible_high=high,
            centers=centers,
            dominant_need=dominant,
        )
        scenario.validate(c)
        return scenario


def affordance_response(points: torch.Tensor, centers: torch.Tensor,
                        sigma: float) -> torch.Tensor:
    """Respuesta homeostática RBF de una coordenada continua, (B,N)."""

    if points.ndim != 2 or centers.ndim != 3:
        raise ValueError("points debe ser (B,D) y centers (B,N,D)")
    squared_distance = (points.unsqueeze(1) - centers).square().sum(dim=-1)
    return torch.exp(-squared_distance / (2.0 * sigma ** 2))


def body_transplant(scenario: GoalDiscoveryScenario,
                    cfg: GoalDiscoveryEnvironmentConfig) \
        -> GoalDiscoveryScenario:
    """Conserva el paisaje y cambia sólo qué necesidad domina el cuerpo."""

    out = scenario.clone()
    new_target = (scenario.dominant_need + 1) % cfg.n_needs
    out.initial_levels.fill_(cfg.setpoint + 0.02)
    rows = torch.arange(out.batch_size, device=out.initial_levels.device)
    out.initial_levels[rows, new_target] = (
        0.5 * (cfg.dominant_min + cfg.dominant_max))
    out.dominant_need = new_target
    out.validate(cfg)
    return out
