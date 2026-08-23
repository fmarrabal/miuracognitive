"""Entorno de compromiso para metas endógenas sostenidas.

No se entrega ``goal_id``. El cuerpo inicial hace urgente una necesidad y el
agente debe completar un proyecto de varias acciones. Entre acciones hay gaps
obligatorios que borran la última acción observable. Tras el primer paso, un
distractor leve deja dos necesidades exactamente empatadas: el estado presente
ya no identifica qué proyecto tiene progreso. En episodios críticos, en cambio,
una necesidad rival cae lo suficiente como para justificar abandonar el plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EndogenousGoalEnvironmentConfig:
    n_needs: int = 3
    horizon: int = 12
    project_length: int = 3
    project_effect: float = 0.31
    basal_drain: float = 0.002
    action_cost: float = 0.012
    viability_threshold: float = 0.30
    setpoint: float = 0.75
    primary_min: float = 0.46
    primary_max: float = 0.50
    rival_min: float = 0.63
    rival_max: float = 0.67
    other_min: float = 0.74
    other_max: float = 0.80
    critical_post_min: float = 0.33
    critical_post_max: float = 0.37
    critical_fraction: float = 0.5

    @property
    def n_actions(self) -> int:
        return self.n_needs + 1

    @property
    def work_ticks(self) -> tuple[int, ...]:
        return tuple(range(0, self.horizon, 2))

    def validate(self) -> None:
        if self.n_needs < 3:
            raise ValueError("fase 2 requiere al menos tres necesidades")
        if self.horizon < 2 * self.project_length + 2:
            raise ValueError("horizon insuficiente para proyectos sostenidos")
        if self.project_length < 2:
            raise ValueError("project_length debe exigir persistencia")
        if not 0.0 < self.critical_fraction < 1.0:
            raise ValueError("critical_fraction debe estar en (0,1)")
        if self.primary_max >= self.rival_min:
            raise ValueError("la prioridad inicial debe ser inequívoca")
        if self.critical_post_max >= self.primary_min:
            raise ValueError("la crisis debe invertir inequívocamente la urgencia")


@dataclass
class EndogenousGoalScenario:
    initial_levels: torch.Tensor       # (B,N)
    setpoints: torch.Tensor            # (B,N)
    disturbances: torch.Tensor         # (B,T,N)
    opportunities: torch.Tensor        # (B,T), 1 sólo en work ticks
    initial_target: torch.Tensor        # (B,), sólo evaluación
    rival_target: torch.Tensor          # (B,), sólo evaluación
    critical: torch.Tensor              # (B,), sólo evaluación

    @property
    def batch_size(self) -> int:
        return int(self.initial_levels.shape[0])

    def to(self, device: torch.device | str) -> "EndogenousGoalScenario":
        return EndogenousGoalScenario(
            initial_levels=self.initial_levels.to(device),
            setpoints=self.setpoints.to(device),
            disturbances=self.disturbances.to(device),
            opportunities=self.opportunities.to(device),
            initial_target=self.initial_target.to(device),
            rival_target=self.rival_target.to(device),
            critical=self.critical.to(device),
        )

    def clone(self) -> "EndogenousGoalScenario":
        return EndogenousGoalScenario(
            initial_levels=self.initial_levels.clone(),
            setpoints=self.setpoints.clone(),
            disturbances=self.disturbances.clone(),
            opportunities=self.opportunities.clone(),
            initial_target=self.initial_target.clone(),
            rival_target=self.rival_target.clone(),
            critical=self.critical.clone(),
        )

    def validate(self, cfg: EndogenousGoalEnvironmentConfig) -> None:
        b = self.batch_size
        if tuple(self.initial_levels.shape) != (b, cfg.n_needs):
            raise ValueError("initial_levels incompatible")
        if tuple(self.setpoints.shape) != (b, cfg.n_needs):
            raise ValueError("setpoints incompatible")
        if tuple(self.disturbances.shape) != (b, cfg.horizon, cfg.n_needs):
            raise ValueError("disturbances incompatible")
        if tuple(self.opportunities.shape) != (b, cfg.horizon):
            raise ValueError("opportunities incompatible")
        for tensor in (self.initial_target, self.rival_target, self.critical):
            if tuple(tensor.shape) != (b,):
                raise ValueError("metadatos incompatibles")
        policy_tensors = (self.initial_levels, self.setpoints,
                          self.disturbances, self.opportunities)
        if not all(torch.isfinite(x).all() for x in policy_tensors):
            raise ValueError("escenario contiene NaN/Inf")
        if (self.initial_target == self.rival_target).any():
            raise ValueError("target inicial y rival deben diferir")


class EndogenousGoalDataset:
    """Genera mundos pareados sin una meta exógena como entrada."""

    def __init__(self, cfg: EndogenousGoalEnvironmentConfig | None = None,
                 seed: int = 0):
        self.cfg = cfg or EndogenousGoalEnvironmentConfig()
        self.cfg.validate()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def batch(self, batch_size: int, *, critical_mode: str = "mixed") \
            -> EndogenousGoalScenario:
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        if critical_mode not in ("mixed", "minor", "critical"):
            raise ValueError("critical_mode debe ser mixed, minor o critical")
        c = self.cfg
        setpoints = torch.full((batch_size, c.n_needs), c.setpoint)
        levels = c.other_min + (c.other_max - c.other_min) * torch.rand(
            batch_size, c.n_needs, generator=self.generator)
        initial_target = torch.empty(batch_size, dtype=torch.long)
        rival_target = torch.empty(batch_size, dtype=torch.long)

        # Tipos balanceados y luego permutados: no existe una asociación fija
        # entre índice de necesidad y papel conductual.
        primary_types = torch.arange(batch_size) % c.n_needs
        primary_types = primary_types[torch.randperm(
            batch_size, generator=self.generator)]
        for row in range(batch_size):
            primary = int(primary_types[row])
            candidates = [need for need in range(c.n_needs)
                          if need != primary]
            rival = candidates[int(torch.randint(
                0, len(candidates), (1,), generator=self.generator))]
            initial_target[row], rival_target[row] = primary, rival
            levels[row, primary] = c.primary_min + (
                c.primary_max - c.primary_min) * torch.rand(
                    (), generator=self.generator)
            levels[row, rival] = c.rival_min + (
                c.rival_max - c.rival_min) * torch.rand(
                    (), generator=self.generator)

        if critical_mode == "minor":
            critical = torch.zeros(batch_size, dtype=torch.bool)
        elif critical_mode == "critical":
            critical = torch.ones(batch_size, dtype=torch.bool)
        else:
            n_critical = round(batch_size * c.critical_fraction)
            flags = torch.cat([
                torch.ones(n_critical, dtype=torch.bool),
                torch.zeros(batch_size - n_critical, dtype=torch.bool)])
            critical = flags[torch.randperm(
                batch_size, generator=self.generator)]

        disturbances = torch.zeros(batch_size, c.horizon, c.n_needs)
        # El pulso ocurre durante el primer gap (t=1) y se observa al decidir
        # en t=2. En minor deja rival y objetivo inicial exactamente iguales;
        # esa igualdad elimina del estado presente la historia del proyecto.
        for row in range(batch_size):
            primary = int(initial_target[row])
            rival = int(rival_target[row])
            if bool(critical[row]):
                post = c.critical_post_min + (
                    c.critical_post_max - c.critical_post_min) * torch.rand(
                        (), generator=self.generator)
                magnitude = levels[row, rival] - post
            else:
                magnitude = levels[row, rival] - levels[row, primary]
            disturbances[row, 1, rival] = magnitude

        opportunities = torch.zeros(batch_size, c.horizon)
        opportunities[:, list(c.work_ticks)] = 1.0
        scenario = EndogenousGoalScenario(
            levels, setpoints, disturbances, opportunities,
            initial_target, rival_target, critical)
        scenario.validate(c)
        return scenario


def body_transplant_scenario(
        cfg: EndogenousGoalEnvironmentConfig,
        batch_per_need: int = 32) -> EndogenousGoalScenario:
    """Mismo exterior nulo; sólo cambia qué variable corporal tiene déficit."""

    batch = cfg.n_needs * batch_per_need
    levels = torch.full((batch, cfg.n_needs), cfg.setpoint + 0.03)
    targets = torch.arange(cfg.n_needs).repeat_interleave(batch_per_need)
    levels[torch.arange(batch), targets] = cfg.primary_min
    rivals = (targets + 1) % cfg.n_needs
    opportunities = torch.zeros(batch, cfg.horizon)
    opportunities[:, list(cfg.work_ticks)] = 1.0
    scenario = EndogenousGoalScenario(
        initial_levels=levels,
        setpoints=torch.full_like(levels, cfg.setpoint),
        disturbances=torch.zeros(batch, cfg.horizon, cfg.n_needs),
        opportunities=opportunities,
        initial_target=targets,
        rival_target=rivals,
        critical=torch.zeros(batch, dtype=torch.bool),
    )
    scenario.validate(cfg)
    return scenario
