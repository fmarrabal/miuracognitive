"""
mHBP — Actuadores con proyección segura e intervención causal.

Forma general (§10 del plan):  a = Π_A[ a_0 + Σ_q W_q · [ū_q ; w̄_q] ]
con Π_A proyección al intervalo del actuador (sigmoid escalada) ⇒ ACOTACIÓN POR
CONSTRUCCIÓN, no por clipping. La contribución de cada campo se conserva por
separado (matriz de influencia campo→actuador + intervenciones causales).

Actuadores del MVP (Fase 1): halting (sesgo continuo), profundidad máxima
(relajación continua + STE), gate de herramienta [0,1], escala de presupuesto.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ActuatorSpec:
    name: str
    lo: float
    hi: float
    a0: float               # valor de reposo (en unidades del actuador)
    discrete: bool = False  # True -> round con straight-through en el forward


MVP_ACTUATORS = [
    ActuatorSpec("halt_bias", -3.0, 3.0, 0.0),
    ActuatorSpec("depth_max", 2.0, 24.0, 8.0, discrete=True),
    ActuatorSpec("tool_gate", 0.0, 1.0, 0.1),
    ActuatorSpec("budget_scale", 0.25, 4.0, 1.0),
]


def _inv_sigmoid_t(y: torch.Tensor) -> torch.Tensor:
    y = y.clamp(1e-6, 1 - 1e-6)
    return torch.log(y / (1 - y))


class ActuatorHead(nn.Module):
    """Cabeza de actuadores sobre los campos. Entrada por campo: [ū_q; w̄_q] (2d).
    Salida: dict {nombre: (B,)} + registro de saturación + intervenciones."""

    def __init__(self, field_cfgs, specs=None, init_scale: float = 0.1):
        super().__init__()
        self.specs = list(specs or MVP_ACTUATORS)
        self._cfgs = list(field_cfgs)
        self.Q = len(field_cfgs)
        self.nA = len(self.specs)
        # una matriz por campo: (nA, 2d) — separable para la matriz de influencia
        self.W = nn.ParameterList([
            nn.Parameter(init_scale * torch.randn(self.nA, 2 * c.d) / (2 * c.d) ** 0.5)
            for c in field_cfgs])
        # sesgo pre-proyección que realiza a_0 exactamente con campos en reposo
        z0 = []
        for sp in self.specs:
            frac = (sp.a0 - sp.lo) / (sp.hi - sp.lo)
            z0.append(float(_inv_sigmoid_t(torch.tensor(frac))))
        self.register_buffer("z0", torch.tensor(z0))
        # intervenciones causales (§21): congelar campos / fijar actuadores /
        # permutar las entradas de campo (¿qué actuador depende de qué campo?)
        self._frozen_fields: set[int] = set()
        self._overrides: dict[str, float] = {}
        self._perm: list[int] | None = None
        self.saturation_frac = {sp.name: 0.0 for sp in self.specs}

    # ---------------- intervención causal ---------------- #
    def freeze_field(self, q: int, frozen: bool = True):
        (self._frozen_fields.add if frozen else self._frozen_fields.discard)(q)

    def override(self, name: str, value: float | None):
        if value is None:
            self._overrides.pop(name, None)
        else:
            self._overrides[name] = float(value)

    def permute_fields(self, perm: list[int] | None):
        """Permuta qué campo alimenta cada cabeza W_q (None = identidad). Sólo
        válida entre campos con la misma d. Intervención del §21."""
        if perm is not None and sorted(perm) != list(range(self.Q)):
            raise ValueError(f"permutación inválida: {perm}")
        self._perm = list(perm) if perm is not None else None

    # ---------------- forward ---------------- #
    def forward(self, pooled: list[tuple[torch.Tensor, torch.Tensor]]):
        """pooled: [(ū_q, w̄_q)] con (B, d) cada uno. Devuelve (acciones, info)."""
        B = pooled[0][0].shape[0]
        dev = pooled[0][0].device
        if self._perm is not None:
            pooled = [pooled[self._perm[q]] for q in range(self.Q)]
        z = self.z0.to(dev).to(pooled[0][0].dtype).unsqueeze(0).expand(B, -1).clone()
        contrib = torch.zeros(self.Q, self.nA, device=dev)      # influencia media |·|
        for q, ((ubar, wbar), Wq) in enumerate(zip(pooled, self.W)):
            if q in self._frozen_fields:
                continue
            x = torch.cat([ubar, wbar], dim=-1)                  # (B, 2d)
            dz = x @ Wq.to(x.dtype).T                            # (B, nA)
            z = z + dz
            with torch.no_grad():
                contrib[q] = dz.abs().mean(0)
        acts, sat = {}, {}
        for j, sp in enumerate(self.specs):
            y = torch.sigmoid(z[:, j])
            a = sp.lo + (sp.hi - sp.lo) * y                      # Π_A: acotado SIEMPRE
            if sp.discrete:
                a = a + (a.round() - a).detach()                 # STE
            if sp.name in self._overrides:
                a = torch.full_like(a, self._overrides[sp.name])
            acts[sp.name] = a
            with torch.no_grad():
                sat[sp.name] = float(((y < 0.01) | (y > 0.99)).float().mean())
        self.saturation_frac = sat
        return acts, {"influence": contrib, "saturation": sat}
