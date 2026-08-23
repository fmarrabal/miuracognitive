"""
mHBP — Operadores de grafo y parametrización segura de cada campo.

Implementa la §2 de MATH_SPEC.md: por campo q y dimensión latente ℓ,
    K_qℓ = ω_qℓ² I + c_q² L_q           (rigidez, ≻0 por construcción)
    C_qℓ = 2 ζ_qℓ ω_qℓ I + D_q L_q      (disipación, ≻0 por construcción)
    G_qℓ = b_q A_q + β_q A_q³           (antisimétrica exacta)

Los parámetros crudos viven en FP32 SIEMPRE (lección del bug BF16 del HBP:
con |raw|~0.3-1.6 el ULP/2 de BF16 supera el paso de Adam y congela el
entrenamiento). El cómputo del campo va en el dtype que pida el llamador.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Grafos reguladores
# --------------------------------------------------------------------------- #
def build_graph(kind: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (L, A): laplaciano combinatorio (simétrico PSD) y matriz de
    advección orientada (antisimétrica) del grafo pedido. FP64."""
    Adj = torch.zeros(n, n, dtype=torch.float64)
    Adv = torch.zeros(n, n, dtype=torch.float64)
    if kind == "chain":
        for i in range(n - 1):
            Adj[i, i + 1] = Adj[i + 1, i] = 1.0
            Adv[i, i + 1] = 1.0
            Adv[i + 1, i] = -1.0
    elif kind == "ring":
        for i in range(n):
            j = (i + 1) % n
            Adj[i, j] = Adj[j, i] = 1.0
            Adv[i, j] = 1.0
            Adv[j, i] = -1.0
    elif kind == "star":
        for i in range(1, n):
            Adj[0, i] = Adj[i, 0] = 1.0
            Adv[0, i] = 1.0
            Adv[i, 0] = -1.0
    elif kind == "complete":
        for i in range(n):
            for j in range(i + 1, n):
                Adj[i, j] = Adj[j, i] = 1.0
                Adv[i, j] = 1.0
                Adv[j, i] = -1.0
    else:
        raise ValueError(f"grafo desconocido: {kind}")
    L = torch.diag(Adj.sum(1)) - Adj
    return L, Adv


def _inv_sigmoid(y: float) -> float:
    y = min(max(y, 1e-6), 1 - 1e-6)
    return math.log(y / (1.0 - y))


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(max(y, 1e-6)))


# --------------------------------------------------------------------------- #
# Parámetros físicos de UN campo (squash a cajas seguras)
# --------------------------------------------------------------------------- #
class FieldParams(nn.Module):
    """Parámetros aprendibles de un campo dentro de rangos seguros.
    ω, ζ por dimensión latente (d,); c, D, b, β escalares del campo."""

    def __init__(self, d: int, *, omega_init=0.5, omega_min=0.1, omega_max=2.0,
                 zeta_init=0.7, zeta_min=0.05, c_init=0.3, c_max=0.7,
                 D_init=0.0, D_max=0.3, b_init=0.0, b_max=0.3,
                 beta_init=0.0, beta_max=0.1):
        super().__init__()
        self.omega_min, self.omega_max = float(omega_min), float(omega_max)
        self.zeta_min = float(zeta_min)
        self.c_max, self.D_max = float(c_max), float(D_max)
        self.b_max, self.beta_max = float(b_max), float(beta_max)
        span = self.omega_max - self.omega_min
        # crudos SIEMPRE FP32 (ver docstring del módulo)
        self.raw_omega = nn.Parameter(torch.full(
            (d,), _inv_sigmoid((omega_init - omega_min) / span), dtype=torch.float32))
        self.raw_zeta = nn.Parameter(torch.full(
            (d,), _inv_softplus(zeta_init - zeta_min), dtype=torch.float32))
        self.raw_c = nn.Parameter(torch.tensor(
            _inv_sigmoid(c_init / c_max if c_max > 0 else 0.5), dtype=torch.float32))
        self.raw_D = nn.Parameter(torch.tensor(
            _inv_sigmoid(D_init / D_max if D_max > 0 else 0.5), dtype=torch.float32))
        self.raw_b = nn.Parameter(torch.tensor(
            _inv_sigmoid(b_init / b_max if b_max > 0 else 0.5), dtype=torch.float32))
        self.raw_beta = nn.Parameter(torch.tensor(
            _inv_sigmoid(beta_init / beta_max if beta_max > 0 else 0.5), dtype=torch.float32))

    @property
    def omega(self):   # (d,)
        return self.omega_min + (self.omega_max - self.omega_min) * torch.sigmoid(self.raw_omega)

    @property
    def zeta(self):    # (d,)
        return self.zeta_min + F.softplus(self.raw_zeta)

    @property
    def c(self):
        return self.c_max * torch.sigmoid(self.raw_c) if self.c_max > 0 else self.raw_c * 0.0

    @property
    def D(self):
        return self.D_max * torch.sigmoid(self.raw_D) if self.D_max > 0 else self.raw_D * 0.0

    @property
    def b(self):
        return self.b_max * torch.sigmoid(self.raw_b) if self.b_max > 0 else self.raw_b * 0.0

    @property
    def beta(self):
        return self.beta_max * torch.sigmoid(self.raw_beta) if self.beta_max > 0 else self.raw_beta * 0.0

    def pin_fp32(self):
        """Re-ancla los crudos a FP32 tras un cast global del modelo."""
        for p in self.parameters():
            p.data = p.data.float()
        return self


# --------------------------------------------------------------------------- #
# Construcción de las matrices K, C, G por dimensión
# --------------------------------------------------------------------------- #
def build_KCG(params: FieldParams, L: torch.Tensor, A: torch.Tensor,
              dtype=torch.float64):
    """Devuelve (K, C, G) con forma (d, N, N) en el dtype pedido.
    K y C simétricas ≻0, G antisimétrica — por construcción (§2)."""
    om = params.omega.to(dtype)          # (d,)
    ze = params.zeta.to(dtype)
    cc = params.c.to(dtype)
    DD = params.D.to(dtype)
    bb = params.b.to(dtype)
    be = params.beta.to(dtype)
    L = L.to(dtype)
    A = A.to(dtype)
    N = L.shape[0]
    I = torch.eye(N, dtype=dtype, device=L.device)
    K = (om ** 2).view(-1, 1, 1) * I + (cc ** 2) * L            # (d,N,N)
    C = (2.0 * ze * om).view(-1, 1, 1) * I + DD * L
    A3 = A @ A @ A
    G = bb * A + be * A3                                        # (N,N) compartida
    G = G.unsqueeze(0).expand(K.shape[0], N, N)                 # (d,N,N)
    return K, C, G


def apply_op(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Aplica un operador (d,N,N) a un estado (B,N,d): por dimensión latente.
    Devuelve (B,N,d)."""
    # x: (B,N,d) -> (d,N,B); M @ · -> (d,N,B) -> (B,N,d)
    return torch.einsum("dij,bjd->bid", M, x)
