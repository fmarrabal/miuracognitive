"""
mHBP — Alostasis: setpoints del campo rápido gobernados por los campos lentos.

Δh*_fast(t) = ε_a · tanh( Ψ(ū_risk, ū_delib, ū_res, ctx) )   (§6 de MATH_SPEC)

Se implementa como forzamiento F_allo = K_fast · Δh*. PRECISIÓN (hallazgo del
árbitro): esto equivale EXACTAMENTE a desplazar el setpoint h*→h*+Δh* solo sin
acoplamiento; con acoplamiento el atractor real es 𝓚⁻¹·embed(K_fast·Δh*)
(desviación O(‖𝓑‖/λ_min(𝓚)), ~2% en el MVP). Lo que se garantiza SIEMPRE:
‖Δh*‖_∞ ≤ ε_a ⇒ F_allo acotado ⇒ ISS (Prop. 2): la alostasis mantiene el
estado ACOTADO. Con ganancia alta de Ψ el lazo cerrado puede sostener ciclos
límite acotados (verificado adversarialmente): "acotado" ≠ "punto fijo".
La condición de pequeña ganancia del lazo es trabajo de Fase 2.

Anti-bypass (§6): Ψ acotada por tanh, ganancia ε_a pequeña, penalización de
tasa y de norma en `regularizer()`; ablation con enabled=False.

Estado interno (_last/_prev shift): se LIMPIA en reset (sin fugas de grafo entre
episodios) y _prev se guarda DETACHED (la penalización de tasa no encadena
grafos de backwards distintos) — hallazgos major del code-review.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .operators import apply_op


class Allostasis(nn.Module):
    """Ψ pequeña y acotada: campos lentos → desplazamiento de setpoint del rápido.
    `slow_dims`: dimensiones latentes de los campos lentos (admite d heterogéneo)."""

    def __init__(self, d_fast: int, n_fast_nodes: int, slow_dims: list[int],
                 eps_a: float = 0.2, hidden: int = 32, ctx_dim: int = 0,
                 enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.eps_a = float(eps_a)
        self.n_fast_nodes, self.d = n_fast_nodes, d_fast
        self.ctx_dim = ctx_dim
        in_dim = sum(slow_dims) + ctx_dim
        self.psi = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, n_fast_nodes * d_fast),
        )
        nn.init.normal_(self.psi[-1].weight, std=0.02)
        nn.init.zeros_(self.psi[-1].bias)
        self._last_shift = None
        self._prev_shift = None

    def reset(self):
        """Limpia el estado entre episodios (evita fugas de grafo/regularizador)."""
        self._last_shift = None
        self._prev_shift = None

    def forward(self, slow_means: list[torch.Tensor], K_fast: torch.Tensor,
                ctx: torch.Tensor | None = None) -> torch.Tensor | None:
        """slow_means: [(B,d_q)] de los campos lentos. K_fast: (d,N,N) del campo
        rápido. Devuelve F_allo (B, N_fast, d) o None si está deshabilitada."""
        if not self.enabled:
            return None
        parts = list(slow_means)
        if self.ctx_dim > 0:
            if ctx is None:
                ctx = torch.zeros(slow_means[0].shape[0], self.ctx_dim,
                                  device=slow_means[0].device)
            parts.append(ctx)
        x = torch.cat(parts, dim=-1)
        w_dtype = self.psi[0].weight.dtype
        shift = self.eps_a * torch.tanh(self.psi(x.to(w_dtype)))
        shift = shift.view(-1, self.n_fast_nodes, self.d).to(slow_means[0].dtype)
        # prev DETACHED: la penalización de tasa no debe re-atravesar el grafo
        # del tick anterior (RuntimeError 'backward a second time' si no)
        self._prev_shift = (self._last_shift.detach()
                            if self._last_shift is not None else None)
        self._last_shift = shift
        return apply_op(K_fast, shift)                      # F_allo = K·Δh*

    def regularizer(self) -> torch.Tensor:
        """Penalización anti-bypass: norma + tasa de cambio del setpoint."""
        if self._last_shift is None:
            return torch.zeros(())
        pen = (self._last_shift ** 2).mean()
        if self._prev_shift is not None and self._prev_shift.shape == self._last_shift.shape:
            pen = pen + ((self._last_shift - self._prev_shift) ** 2).mean()
        return pen

    # -------- checkpoint del estado dinámico -------- #
    def save_state(self) -> dict:
        return {"last": None if self._last_shift is None else self._last_shift.detach().clone(),
                "prev": None if self._prev_shift is None else self._prev_shift.detach().clone()}

    def load_state(self, st: dict):
        self._last_shift = None if st["last"] is None else st["last"].clone()
        self._prev_shift = None if st["prev"] is None else st["prev"].clone()
