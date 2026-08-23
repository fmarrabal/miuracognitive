"""Planta corporal de onda amortiguada para probar automodelado causal."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from model.hbp import build_chain_laplacian


@dataclass(frozen=True)
class SelfModelEnvironmentConfig:
    n_nodes: int = 5
    horizon: int = 22
    damage_tick: int = 11
    dt: float = 1.0
    omega0: float = 0.12
    zeta: float = 0.40
    wave_speed: float = 0.15
    primary_effect: float = 0.38
    backup_effect: float = 0.27
    actuator_scale_min: float = 0.88
    actuator_scale_max: float = 1.12
    damage_scale: float = 0.0
    shock_min: float = 0.68
    shock_max: float = 0.80

    @property
    def n_actions(self) -> int:
        # no-op + actuadores +/- primarios y de respaldo por nodo.
        return 1 + 4 * self.n_nodes

    def validate(self) -> None:
        if self.n_nodes < 3:
            raise ValueError("se requieren al menos tres nodos corporales")
        if not 2 <= self.damage_tick <= self.horizon - 4:
            raise ValueError("damage_tick no deja ventanas pre/post suficientes")
        if min(self.dt, self.omega0, self.zeta,
               self.wave_speed) <= 0:
            raise ValueError("los parámetros de onda deben ser positivos")
        if not 0 <= self.damage_scale < 1:
            raise ValueError("damage_scale debe representar pérdida de eficacia")


@dataclass
class SelfModelScenario:
    # Matrices pertenecientes a la planta y nunca entran en la política.
    intact_effects: torch.Tensor       # (B,N,A), dinámica oculta
    damaged_effects: torch.Tensor      # (B,N,A), dinámica oculta
    target_node: torch.Tensor          # (B,), sólo evaluación
    first_shock: torch.Tensor          # (B,)
    damage_shock: torch.Tensor         # (B,)

    @property
    def batch_size(self) -> int:
        return int(self.target_node.shape[0])

    def to(self, device: torch.device | str) -> "SelfModelScenario":
        return SelfModelScenario(
            intact_effects=self.intact_effects.to(device),
            damaged_effects=self.damaged_effects.to(device),
            target_node=self.target_node.to(device),
            first_shock=self.first_shock.to(device),
            damage_shock=self.damage_shock.to(device),
        )

    def clone(self) -> "SelfModelScenario":
        return SelfModelScenario(
            intact_effects=self.intact_effects.clone(),
            damaged_effects=self.damaged_effects.clone(),
            target_node=self.target_node.clone(),
            first_shock=self.first_shock.clone(),
            damage_shock=self.damage_shock.clone(),
        )

    def validate(self, cfg: SelfModelEnvironmentConfig) -> None:
        b, n, a = self.batch_size, cfg.n_nodes, cfg.n_actions
        if tuple(self.intact_effects.shape) != (b, n, a):
            raise ValueError("intact_effects incompatible")
        if tuple(self.damaged_effects.shape) != (b, n, a):
            raise ValueError("damaged_effects incompatible")
        if tuple(self.target_node.shape) != (b,):
            raise ValueError("target_node incompatible")
        if tuple(self.first_shock.shape) != (b,):
            raise ValueError("first_shock incompatible")
        if tuple(self.damage_shock.shape) != (b,):
            raise ValueError("damage_shock incompatible")
        if not (torch.isfinite(self.intact_effects).all()
                and torch.isfinite(self.damaged_effects).all()):
            raise ValueError("efectos no finitos")


def canonical_actuator_effects(
        cfg: SelfModelEnvironmentConfig,
        *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Prior corporal nominal, (N,A); no contiene qué actuador será dañado."""

    effects = torch.zeros(
        cfg.n_nodes, cfg.n_actions, device=device, dtype=torch.float32)
    for node in range(cfg.n_nodes):
        effects[node, 1 + node] = cfg.primary_effect
        effects[node, 1 + cfg.n_nodes + node] = cfg.backup_effect
        effects[node, 1 + 2 * cfg.n_nodes + node] = -cfg.primary_effect
        effects[node, 1 + 3 * cfg.n_nodes + node] = -cfg.backup_effect
    return effects


class SelfModelDataset:
    def __init__(self, cfg: SelfModelEnvironmentConfig | None = None,
                 seed: int = 0):
        self.cfg = cfg or SelfModelEnvironmentConfig()
        self.cfg.validate()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def batch(self, batch_size: int) -> SelfModelScenario:
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        cfg = self.cfg
        canonical = canonical_actuator_effects(cfg)
        scale = cfg.actuator_scale_min + (
            cfg.actuator_scale_max - cfg.actuator_scale_min) * torch.rand(
                batch_size, cfg.n_actions, generator=self.generator)
        scale[:, 0] = 1.0
        intact = canonical.unsqueeze(0) * scale.unsqueeze(1)
        target = torch.arange(batch_size) % cfg.n_nodes
        target = target[torch.randperm(batch_size, generator=self.generator)]
        damaged = intact.clone()
        rows = torch.arange(batch_size)
        damaged[rows, :, 1 + target] *= cfg.damage_scale
        first_shock = cfg.shock_min + (
            cfg.shock_max - cfg.shock_min) * torch.rand(
                batch_size, generator=self.generator)
        damage_shock = cfg.shock_min + (
            cfg.shock_max - cfg.shock_min) * torch.rand(
                batch_size, generator=self.generator)
        scenario = SelfModelScenario(
            intact_effects=intact,
            damaged_effects=damaged,
            target_node=target,
            first_shock=first_shock,
            damage_shock=damage_shock,
        )
        scenario.validate(cfg)
        return scenario


def damage_transplant(scenario: SelfModelScenario,
                      cfg: SelfModelEnvironmentConfig) -> SelfModelScenario:
    """Mismo cuerpo intacto; mueve shock y daño al nodo siguiente."""

    out = scenario.clone()
    target = (scenario.target_node + 1) % cfg.n_nodes
    out.target_node = target
    out.damaged_effects = out.intact_effects.clone()
    rows = torch.arange(out.batch_size, device=target.device)
    out.damaged_effects[rows, :, 1 + target] *= cfg.damage_scale
    out.validate(cfg)
    return out


def wave_companion_spectral_radius(cfg: SelfModelEnvironmentConfig) -> float:
    """Radio exacto de la planta pasiva discretizada, sin actuadores."""

    laplacian = build_chain_laplacian(cfg.n_nodes)
    identity = torch.eye(cfg.n_nodes)
    stiffness = cfg.dt ** 2 * (
        cfg.omega0 ** 2 * identity + cfg.wave_speed ** 2 * laplacian)
    damping = 2.0 * cfg.zeta * cfg.omega0 * cfg.dt
    zero = torch.zeros_like(identity)
    companion = torch.cat([
        torch.cat([(2.0 - damping) * identity - stiffness,
                   -(1.0 - damping) * identity], dim=1),
        torch.cat([identity, zero], dim=1),
    ], dim=0)
    return float(torch.linalg.eigvals(companion).abs().max())
