"""Sustrato de campo homeostático 2D con dinámica de Navier-Stokes (vorticidad-
stream-escalar), la extensión a lámina real de la propuesta de Curro.

Sobre una malla H×W (grafo-rejilla), h juega el papel de VORTICIDAD ω:
  · ELÍPTICO (Poisson en el grafo):  ∇²ψ = −ω  ⟹  ψ = L⁺ω   (L discretiza −∇²)
  · Velocidad incompresible:  u = ∂ψ/∂y,  v = −∂ψ/∂x   (∇·u=0 por construcción)
  · ADVECCIÓN del escalar pasivo S por ese flujo (semi-Lagrangiano: traza el
    punto de partida y bilerpa -> INCONDICIONALMENTE ESTABLE y diferenciable)
    + difusión D·∇²S.
Ahora el jacobiano J(ψ,·) SÍ está bien definido (hay x,y). El flujo lo esculpe
el input (ω = forzamiento aprendido); S transporta información a lo largo de las
streamlines que el campo genera. Esto es lo que ninguna física LOCAL puede: mover
información direccionalmente por rutas auto-determinadas y no-locales.

Modo "kinematic" (por defecto): el modelo fija ω (control); ψ,flujo,S se siguen.
Modo "dynamic": ω se auto-advecta (NS turbulenta completa; caótico, opcional).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_laplacian(H: int, W: int, walls: torch.Tensor | None = None) -> torch.Tensor:
    """Laplaciano combinatorio (D−A) de la rejilla H×W con vecindad-4. Positivo
    semidefinido, discretiza −∇². Con `walls` (H,W bool) los nodos-pared se
    desconectan (obstáculos: el flujo no los atraviesa)."""
    N = H * W
    A = torch.zeros(N, N)
    def idx(i, j): return i * W + j
    blocked = walls.reshape(-1) if walls is not None else torch.zeros(N, dtype=torch.bool)
    for i in range(H):
        for j in range(W):
            a = idx(i, j)
            if blocked[a]:
                continue
            for di, dj in ((0, 1), (1, 0)):
                ni, nj = i + di, j + dj
                if ni < H and nj < W and not blocked[idx(ni, nj)]:
                    b = idx(ni, nj)
                    A[a, b] = A[b, a] = 1.0
    D = torch.diag(A.sum(1))
    return D - A


@dataclass
class Flow2DConfig:
    H: int = 16
    W: int = 16
    dt: float = 0.6
    vel_scale: float = 1.0     # escala física velocidad->desplazamiento (CFL)
    diffusion: float = 0.02
    conserve_mass: bool = True # renormaliza ∫S tras la advección (factor detached)
    nonneg: bool = True        # clamp S>=0 (canales one-hot; evita masa negativa)
    max_disp: float = 1.5      # tope de desplazamiento por paso (celdas; CFL)
    mode: str = "kinematic"    # "kinematic" | "dynamic" (dynamic solo forward)
    nu: float = 0.01           # viscosidad (solo dynamic)


class Flow2DField(nn.Module):
    def __init__(self, cfg: Flow2DConfig, walls: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        H, W = cfg.H, cfg.W
        L = grid_laplacian(H, W, walls)
        self.register_buffer("L", L)
        self.register_buffer("L_pinv", torch.linalg.pinv(L))   # Poisson global
        # coordenadas normalizadas base para grid_sample (align_corners=True)
        ys = torch.linspace(-1, 1, H)
        xs = torch.linspace(-1, 1, W)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("base_grid", torch.stack([gx, gy], dim=-1))  # (H,W,2)

    # ---- ω (B,H,W) -> ψ (B,H,W) por Poisson elíptico global ----
    def poisson(self, omega: torch.Tensor) -> torch.Tensor:
        B = omega.shape[0]
        flat = omega.reshape(B, -1) @ self.L_pinv.T.to(omega.dtype)   # ψ = L⁺ ω
        return flat.reshape(B, self.cfg.H, self.cfg.W)

    # ---- velocidad incompresible desde ψ: u_x=∂ψ/∂y, u_y=−∂ψ/∂x ----
    def velocity(self, psi: torch.Tensor):
        dpsi_dy = torch.zeros_like(psi)
        dpsi_dx = torch.zeros_like(psi)
        dpsi_dy[:, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) * 0.5   # ∂ψ/∂y (fila=y)
        dpsi_dx[:, :, 1:-1] = (psi[:, :, 2:] - psi[:, :, :-2]) * 0.5   # ∂ψ/∂x (col=x)
        return dpsi_dy, -dpsi_dx                        # (u_x, u_y)

    # ---- advección semi-Lagrangiana de un campo por (u_x,u_y) ----
    def advect(self, field: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, H, W = field.shape
        # desplazamiento en unidades de celda -> coords normalizadas [-1,1].
        # x = columna (W), y = fila (H): disp_x usa la VELOCIDAD-X (ux), disp_y la -Y.
        # Se topa a max_disp celdas/paso (CFL: evita que una ω sin acotar salte
        # >2 celdas y rompa el semi-Lagrangiano).
        cells_x = (cfg.vel_scale * cfg.dt * ux).clamp(-cfg.max_disp, cfg.max_disp)
        cells_y = (cfg.vel_scale * cfg.dt * uy).clamp(-cfg.max_disp, cfg.max_disp)
        disp_x = cells_x * (2.0 / (W - 1))
        disp_y = cells_y * (2.0 / (H - 1))
        depart = self.base_grid.unsqueeze(0).to(field.dtype).clone()   # (1,H,W,2)
        depart = depart.expand(B, H, W, 2).clone()
        depart[..., 0] = depart[..., 0] - disp_x       # traza hacia atrás
        depart[..., 1] = depart[..., 1] - disp_y
        # BILINEAL (bicúbico introducía S<0 espuria que realimentaba el renorm) +
        # borde ABSORBENTE (zeros): semántica de ruteo sin artefactos de reflexión.
        out = F.grid_sample(field.unsqueeze(1), depart, mode="bilinear",
                            padding_mode="zeros", align_corners=True).squeeze(1)
        if cfg.nonneg:
            out = out.clamp_min(0.0)                    # canales one-hot: S>=0
        if cfg.conserve_mass:
            # el semi-Lagrangiano NO conserva ∫; se renormaliza. El factor va
            # DETACHED: sin detach, ∂(s_old/s_new)/∂ω explotaba el gradiente a
            # ~1e16 en 15 ticks (crítica del panel). Conserva masa en forward,
            # gradiente estable.
            B = out.shape[0]
            s_old = field.reshape(B, -1).sum(-1)
            s_new = out.reshape(B, -1).sum(-1).clamp_min(1e-6)
            out = out * (s_old / s_new).view(B, 1, 1).detach()
        return out

    def diffuse(self, field: torch.Tensor) -> torch.Tensor:
        B = field.shape[0]
        lap = (field.reshape(B, -1) @ self.L.T.to(field.dtype)).reshape(field.shape)
        return field - self.cfg.diffusion * lap        # explícito (D·∇²=−D·L)

    def step(self, S: torch.Tensor, omega: torch.Tensor,
             omega_prev: torch.Tensor | None = None):
        """Un tick. Devuelve (S', omega', psi, (u,v)). En kinematic omega lo fija
        el llamador; en dynamic ω se auto-advecta + difunde (NS)."""
        psi = self.poisson(omega)
        u, v = self.velocity(psi)
        S = self.advect(S, u, v)
        S = self.diffuse(S)
        if self.cfg.mode == "dynamic":
            omega = self.advect(omega, u, v)
            B = omega.shape[0]
            lap = (omega.reshape(B, -1) @ self.L.T.to(omega.dtype)).reshape(omega.shape)
            omega = omega - self.cfg.nu * lap
        return S, omega, psi, (u, v)
