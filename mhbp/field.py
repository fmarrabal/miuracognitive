"""
mHBP — HomeostaticField: un campo homeostático individual.

Responsabilidades (§14 del plan): estado (u, w), parámetros seguros, operadores
K/C/G, energía propia, diagnóstico. La INTEGRACIÓN la realiza el integrador
global de CoupledMultiscaleHBP (el acoplamiento exige un paso conjunto);
el campo expone su parte y su geometría.

Estado: u = h − h* ∈ (B, N, d)  y velocidad escalada w = τ·u̇ ∈ (B, N, d).
"""
from __future__ import annotations
from dataclasses import dataclass, field as dc_field
import torch
import torch.nn as nn

from .operators import FieldParams, build_graph, build_KCG, apply_op


@dataclass
class FieldConfig:
    name: str = "field"
    n_nodes: int = 8
    d: int = 8
    graph: str = "chain"
    tau_init: float = 1.0          # INFORMATIVO: las τ efectivas las fija
                                   # MHBPConfig.taus/Timescales (única fuente de verdad)
    # cajas seguras (se pasan a FieldParams)
    omega_init: float = 0.5
    omega_min: float = 0.1
    omega_max: float = 2.0
    zeta_init: float = 0.7
    zeta_min: float = 0.05
    c_init: float = 0.3
    c_max: float = 0.7
    D_init: float = 0.0
    D_max: float = 0.3
    b_init: float = 0.05
    b_max: float = 0.3
    beta_init: float = 0.02
    beta_max: float = 0.1
    f_max: float = 0.5             # cota del forzamiento interoceptivo (ISS, Prop. 2)


class HomeostaticField(nn.Module):
    """Un campo del mHBP. No integra por sí solo: expone operadores y estado."""

    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        self.name = cfg.name
        L, A = build_graph(cfg.graph, cfg.n_nodes)
        self.register_buffer("L", L)
        self.register_buffer("A", A)
        self.params = FieldParams(
            cfg.d, omega_init=cfg.omega_init, omega_min=cfg.omega_min,
            omega_max=cfg.omega_max, zeta_init=cfg.zeta_init, zeta_min=cfg.zeta_min,
            c_init=cfg.c_init, c_max=cfg.c_max, D_init=cfg.D_init, D_max=cfg.D_max,
            b_init=cfg.b_init, b_max=cfg.b_max, beta_init=cfg.beta_init,
            beta_max=cfg.beta_max)
        # punto de consigna homeostático (aprendible, cuasi-estático)
        self.h_star = nn.Parameter(0.02 * torch.randn(cfg.n_nodes, cfg.d, dtype=torch.float32))
        # estado dinámico (no-parámetro); reset_state lo crea con el batch correcto
        self.register_buffer("u", torch.zeros(1, cfg.n_nodes, cfg.d), persistent=False)
        self.register_buffer("w", torch.zeros(1, cfg.n_nodes, cfg.d), persistent=False)
        self._B = None

    # ------------------------------------------------------------------ #
    def reset_state(self, batch: int, device=None, dtype=torch.float64):
        device = device or self.h_star.device
        self.u = torch.zeros(batch, self.cfg.n_nodes, self.cfg.d, device=device, dtype=dtype)
        self.w = torch.zeros_like(self.u)
        self._B = batch

    def operators(self, dtype=torch.float64):
        """(K, C, G) actuales, (d,N,N)."""
        return build_KCG(self.params, self.L, self.A, dtype=dtype)

    # ------------------------------------------------------------------ #
    def energy(self, K: torch.Tensor | None = None) -> torch.Tensor:
        """Energía propia ½‖w‖² + ½⟨u,Ku⟩ por muestra del batch. (B,)"""
        if K is None:
            K, _, _ = self.operators(dtype=self.u.dtype)
        kin = 0.5 * (self.w ** 2).sum(dim=(1, 2))
        Ku = apply_op(K, self.u)
        pot = 0.5 * (self.u * Ku).sum(dim=(1, 2))
        return kin + pot

    def node_mean(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(ū, w̄): medias sobre nodos, (B, d). Entrada de interfaz/actuadores."""
        return self.u.mean(dim=1), self.w.mean(dim=1)

    def diagnostics(self) -> dict[str, float]:
        with torch.no_grad():
            return {
                "u_norm": float(self.u.norm()),
                "w_norm": float(self.w.norm()),
                "u_max": float(self.u.abs().max()) if self.u.numel() else 0.0,
                "omega_mean": float(self.params.omega.mean()),
                "zeta_mean": float(self.params.zeta.mean()),
                "c": float(self.params.c),
                "b": float(self.params.b),
                "beta": float(self.params.beta),
                "finite": bool(torch.isfinite(self.u).all() and torch.isfinite(self.w).all()),
            }

    def check_parameters(self) -> dict[str, bool]:
        """Comprobaciones estructurales de las cajas (para StabilityMonitor)."""
        p = self.params
        with torch.no_grad():
            return {
                "omega_in_box": bool((p.omega > p.omega_min - 1e-9).all()
                                     and (p.omega < p.omega_max + 1e-9).all()),
                "zeta_positive": bool((p.zeta >= p.zeta_min - 1e-9).all()),
                "L_symmetric": bool(torch.allclose(self.L, self.L.T)),
                "A_antisymmetric": bool(torch.allclose(self.A, -self.A.T)),
            }
