"""
Oráculo NUMÉRICO del SMA: optimiza las acciones directamente con los latentes
VERDADEROS (información completa, sin restricción de observación). Cota
superior común para el compute-regret (§20). Determinista; cacheable.
"""
from __future__ import annotations
import torch

from .env import episode_objective


ORACLE_VERSION = 3      # v3: 12000 its (v2 con 8000 dejaba budget_lo a resid
                        # 1.06%, justo sobre el umbral del 1%). v2: convergido +
                        # MISMO espacio de acciones que los actuadores (v1 con
                        # 300 its: una política podía batirlo → regret negativo)


def _project(raw: torch.Tensor) -> dict:
    """Proyección al MISMO espacio de acciones que ActuatorHead (rangos MVP).
    depth continua (cota ligeramente holgada respecto al STE; documentado)."""
    return {
        "halt_bias": -3.0 + 6.0 * torch.sigmoid(raw[..., 0]),
        "depth_max": 2.0 + 22.0 * torch.sigmoid(raw[..., 1]),
        "tool_gate": torch.sigmoid(raw[..., 2]),
        "budget_scale": 0.25 + 3.75 * torch.sigmoid(raw[..., 3]),
    }


def oracle_actions(lat: dict, iters: int = 12000, lr: float = 0.05,
                   seed: int = 0) -> dict:
    """Optimiza (B,T,4) acciones sobre J con Adam + decay de lr (cosine a lr/10).
    Registra el residuo de convergencia ΔJ del último 12.5% de iteraciones.
    Determinista (init ceros + semilla fija)."""
    B, T = lat["d"].shape
    dev = lat["d"].device
    torch.manual_seed(seed)
    raw = torch.zeros(B, T, 4, device=dev, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr / 10)
    J_checkpoint = None
    for it in range(iters):
        out = episode_objective(lat, _project(raw))
        loss = out["J"].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it == int(iters * 0.875):
            J_checkpoint = float(loss)
    with torch.no_grad():
        A = _project(raw)
        out = episode_objective(lat, A)
    J_final = float(out["J"].mean())
    conv_resid = abs(J_checkpoint - J_final) / max(abs(J_final), 1e-9) \
        if J_checkpoint is not None else float("nan")
    return {"J": out["J"].detach(), "A": {k: v.detach() for k, v in A.items()},
            "convergence_residual": conv_resid,
            "terms": {k: out[k].detach() for k in ("err", "hard", "plan", "hold", "viol_rate")}}
