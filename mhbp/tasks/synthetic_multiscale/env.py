"""
SMA — Synthetic Multiscale Allocation (Fase 2, pre-registrado en PREREG_PHASE2.md).

Entorno DIFERENCIABLE de asignación de recursos con TRES escalas latentes:
  d_t  dificultad   (rápida: AR(1)+saltos)
  ρ_t  coste error  (media: Markov, permanencia ~32)
  B_w  presupuesto  (lenta: sinusoide ×[0.6,1.4] por ventana)

Acciones por tick (los 4 actuadores del MVP): halt_bias h, depth n, tool g,
budget_scale s (planificación de reserva de la ventana siguiente).

Las OBSERVACIONES (interocepción ruidosa/retrasada) van DETACHED: el gradiente
fluye solo por las acciones → sin leakage por el canal de observación.
El objetivo J es una función batched (B,T)-vectorizada de las acciones.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class EnvConfig:
    T: int = 256
    W: int = 32                 # ticks por ventana
    # dificultad (rápida)
    d_phi: float = 0.7
    d_sigma: float = 0.12
    d_jump_p: float = 0.05
    # riesgo (media) — dwell 14 ≠ W (varios cambios por ventana: reasignación
    # INTRA-ventana; con dwell==W la escala media apenas era accionable, panel)
    risk_hi: float = 4.0
    risk_dwell: int = 14
    # presupuesto (lenta) — DOS frecuencias no conmensurables con la ventana
    # (con periodo exacto 4, B_{w+2}=2μ−B_w era una recurrencia determinista y
    # "planificar" degeneraba en una línea de retardo; hallazgo del panel)
    B_base: float = 280.0
    B_period_windows: float = 3.7
    B_period2_windows: float = 6.3
    B_amp: float = 0.28
    B_amp2: float = 0.18
    # física de la tarea
    D_max: float = 19.2
    beta_err: float = 0.5
    gamma_tool: float = 0.5
    cost_tool: float = 6.0
    alpha_halt_q: float = 0.3   # daño de calidad del halting en difíciles
    alpha_halt_c: float = 0.5   # ahorro de coste del halting en fáciles
    # pesos del objetivo
    lam_hard: float = 0.02
    lam_plan: float = 0.005
    kap_hold: float = 0.002
    lam_cost: float = 0.0008    # coste intrínseco del gasto: ahorrar cuando el
                                # riesgo es bajo PAGA (hace accionable la escala
                                # media; +7.2% de J era su valor sin esto, panel)
    # ruido de observación — calibrado tras el fallo del gate G3 v1: con ruido
    # bajo el entorno era demasiado reactivo-resoluble y el término de
    # presupuesto (observable sin memoria) dominaba el hueco. Ahora el filtrado
    # TEMPORAL paga: entropy muy ruidosa (d tiene correlación ~3 ticks),
    # risk con 35% de flips (hay que integrar ~10 ticks), presupuesto restante
    # ruidoso y CUANTIZADO a cuartos de ventana (planificar exige integrar).
    # v3b: 0.45/0.35 hacían fallar G2 (ni el privilegiado aprendía en 250 pasos);
    # punto medio: el filtrado temporal sigue pagando (d se reconstruye mal por
    # tick) pero la tarea es aprendible al presupuesto del confirmatorio
    obs_noise: float = 0.30
    risk_delay: int = 2
    risk_flip_p: float = 0.30
    budget_obs_noise: float = 0.25
    budget_obs_quant: int = 8   # gpu_memory se actualiza cada W/quant... ticks


def sample_latents(cfg: EnvConfig, B: int, T: int, seed: int, device,
                   ood: str | None = None) -> dict:
    """Latentes del episodio (sin grad). ood ∈ {None, 'budget_lo', 'budget_hi',
    'riskfreq', 'long', 'step', 'risk_only'}. Determinista en seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dwell = cfg.risk_dwell
    B_scale = 1.0
    step_drop = None
    d_const = None
    if ood == "budget_lo":
        B_scale = 0.6
    elif ood == "budget_hi":
        B_scale = 1.5
    elif ood == "riskfreq":
        dwell = 8
    elif ood == "step":
        step_drop = (T // 2, 0.4)          # B ×0.4 a partir de t=T/2
    elif ood == "risk_only":
        d_const = 0.55

    # dificultad AR(1) + saltos
    d = torch.empty(B, T)
    x = torch.rand(B, generator=g)
    for t in range(T):
        eps = torch.randn(B, generator=g) * cfg.d_sigma
        x = 0.5 + cfg.d_phi * (x - 0.5) + eps
        jump = torch.rand(B, generator=g) < cfg.d_jump_p
        x = torch.where(jump, torch.rand(B, generator=g), x)
        x = x.clamp(0.02, 0.98)
        d[:, t] = x
    if d_const is not None:
        d = torch.full_like(d, d_const)

    # riesgo Markov 2 estados
    rho = torch.empty(B, T)
    state = (torch.rand(B, generator=g) < 0.5).double()
    p_sw = 1.0 / dwell
    for t in range(T):
        sw = torch.rand(B, generator=g) < p_sw
        state = torch.where(sw, 1.0 - state, state)
        rho[:, t] = 1.0 + (cfg.risk_hi - 1.0) * state

    # presupuesto por ventana: dos sinusoides NO conmensurables + fases propias
    nW = T // cfg.W
    ph1 = torch.rand(B, 1, generator=g) * 2 * math.pi
    ph2 = torch.rand(B, 1, generator=g) * 2 * math.pi
    wi = torch.arange(nW).unsqueeze(0).double()
    Bw = cfg.B_base * B_scale * (1.0
        + cfg.B_amp * torch.sin(2 * math.pi * wi / cfg.B_period_windows + ph1)
        + cfg.B_amp2 * torch.sin(2 * math.pi * wi / cfg.B_period2_windows + ph2))
    Bw = Bw + cfg.B_base * 0.05 * torch.randn(B, nW, generator=g)
    if step_drop is not None:
        t0, fac = step_drop
        w0 = t0 // cfg.W
        Bw[:, w0:] = Bw[:, w0:] * fac

    # ruidos de observación pre-generados (determinismo y pareado exacto)
    noise = {
        "d": torch.randn(B, T, generator=g) * cfg.obs_noise,
        "u": torch.randn(B, T, generator=g),
        "risk_flip": torch.rand(B, T, generator=g) < cfg.risk_flip_p,
        "budget": torch.randn(B, T, generator=g) * 0.1,
        "meter": torch.randn(B, T, generator=g) * 0.08,   # ruido de medición del
                                                          # coste propio (evita
                                                          # integrar el gasto exacto)
    }
    return {"d": d.to(device), "rho": rho.to(device), "Bw": Bw.to(device),
            "noise": {k: v.to(device) for k, v in noise.items()},
            "T": T, "nW": nW, "cfg": cfg}


# --------------------------------------------------------------------------- #
def signals_at(lat: dict, t: int, prev_cost: torch.Tensor,
               spent_in_window: torch.Tensor) -> dict:
    """Canales interoceptivos (B,) del tick t. TODO detached y ruidoso.
    prev_cost: coste del tick anterior; spent_in_window: gasto acumulado."""
    cfg: EnvConfig = lat["cfg"]
    d, rho, Bw, nz = lat["d"], lat["rho"], lat["Bw"], lat["noise"]
    B = d.shape[0]
    w = min(t // cfg.W, lat["nW"] - 1)
    td = max(t - cfg.risk_delay, 0)
    risk_ind = (rho[:, td] > 1.0).double()
    risk_obs = torch.where(nz["risk_flip"][:, t], 1.0 - risk_ind, risk_ind)
    # presupuesto restante: el llamador pasa el gasto OBSERVADO (rancio, se
    # refresca cada W//budget_obs_quant ticks en el rollout) — sin memoria no
    # se interpola bien entre refrescos
    remaining = (Bw[:, w] - spent_in_window) / cfg.B_base
    return {
        "entropy": (d[:, t] + nz["d"][:, t]).clamp(0, 1.5).detach(),
        "uncertainty": (nz["u"][:, t].abs() * (0.5 + d[:, t])).detach(),
        "risk": risk_obs.detach(),
        "token_cost": (prev_cost * (1.0 + nz["meter"][:, t]) / 24.0).detach(),
        "gpu_memory": (remaining + cfg.budget_obs_noise * nz["budget"][:, t] / 0.1).detach(),
        "elapsed_time": torch.full((B,), (t % cfg.W) / cfg.W, device=d.device).detach(),
        "queue_load": torch.full((B,), w / max(lat["nW"] - 1, 1), device=d.device).detach(),
    }


def tick_cost(lat: dict, t: int, acts: dict) -> torch.Tensor:
    """Coste del tick (B,), diferenciable respecto a las acciones."""
    cfg: EnvConfig = lat["cfg"]
    d = lat["d"][:, t]
    n = acts["depth_max"]
    g = acts["tool_gate"]
    sh = torch.sigmoid(acts["halt_bias"])
    return n * (1.0 - cfg.alpha_halt_c * sh * (1.0 - d)) + cfg.cost_tool * g


def episode_objective(lat: dict, A: dict) -> dict:
    """Objetivo del episodio, vectorizado. A: dict de acciones apiladas (B,T).
    Devuelve términos y J (B,). Diferenciable respecto a A."""
    cfg: EnvConfig = lat["cfg"]
    d, rho, Bw = lat["d"], lat["rho"], lat["Bw"]
    n, g = A["depth_max"], A["tool_gate"]
    sh = torch.sigmoid(A["halt_bias"])
    s = A["budget_scale"]
    q_eff = n * (1.0 + cfg.gamma_tool * g) * (1.0 - cfg.alpha_halt_q * sh * d)
    err = torch.sigmoid(cfg.beta_err * (d * cfg.D_max - q_eff))
    err_term = (rho * err).mean(dim=1)                                    # (B,)
    cost = n * (1.0 - cfg.alpha_halt_c * sh * (1.0 - d)) + cfg.cost_tool * g
    T, W, nW = lat["T"], cfg.W, lat["nW"]
    spend_w = cost.view(-1, nW, W).sum(-1)                                # (B,nW)
    # reserva de la ventana w = B_nom · mean(s en la ventana w−1); w=0 → 1.0
    s_w = s.view(-1, nW, W).mean(-1)                                      # (B,nW)
    reserve = torch.cat([torch.ones_like(s_w[:, :1]),
                         s_w[:, :-1]], dim=1) * cfg.B_base
    hard = torch.relu(spend_w - Bw).pow(2).mean(1)
    plan = torch.relu(spend_w - reserve).pow(2).mean(1)
    hold = torch.relu(reserve - spend_w).mean(1)
    cost_term = spend_w.mean(1)                          # coste intrínseco (λ_c)
    J = (err_term + cfg.lam_hard * hard + cfg.lam_plan * plan
         + cfg.kap_hold * hold + cfg.lam_cost * cost_term)
    viol_rate = (spend_w > Bw).double().mean(1)
    viol_mag = (torch.relu(spend_w - Bw) / Bw.clamp(min=1e-6))            # (B,nW)
    return {"J": J, "err": err_term, "hard": hard, "plan": plan, "hold": hold,
            "cost_term": cost_term, "viol_rate": viol_rate,
            "viol_mag_mean": viol_mag.mean(1),
            "viol_mag_p95": viol_mag.quantile(0.95, dim=1),
            "spend_w": spend_w, "err_t": err, "cost_t": cost}


# --------------------------------------------------------------------------- #
def rollout(controller, lat: dict, device) -> dict:
    """Ejecuta el controlador tick a tick sobre el episodio (observaciones
    detached) y evalúa el objetivo sobre las acciones apiladas. El modo
    train/eval lo fija el llamador en el controlador (no aquí)."""
    cfg: EnvConfig = lat["cfg"]
    B, T = lat["d"].shape
    controller.reset(B, device)
    prev_cost = torch.zeros(B, device=device)
    spent = torch.zeros(B, device=device)
    spent_stale = torch.zeros(B, device=device)     # observación cuantizada
    q_refresh = max(cfg.W // cfg.budget_obs_quant, 1)
    acts_traj = {k: [] for k in ("halt_bias", "depth_max", "tool_gate", "budget_scale")}
    for t in range(T):
        if t % cfg.W == 0:
            spent = torch.zeros(B, device=device)
            spent_stale = torch.zeros(B, device=device)
        if t % q_refresh == 0:
            spent_stale = spent.clone()
        sig = signals_at(lat, t, prev_cost, spent_stale)
        acts = controller.step(sig)
        for k in acts_traj:
            acts_traj[k].append(acts[k])
        with torch.no_grad():
            c = tick_cost(lat, t, {k: v.detach() for k, v in acts.items()})
            prev_cost = c
            spent = spent + c
    A = {k: torch.stack(v, dim=1) for k, v in acts_traj.items()}          # (B,T)
    out = episode_objective(lat, A)
    out["A"] = A
    return out


def tracking_metrics(lat: dict, A: dict) -> dict:
    """Correlaciones de seguimiento (¿cada acción sigue a su factor?)."""
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = a.norm() * b.norm() + 1e-12
        return float((a * b).sum() / den)
    d = lat["d"].flatten()
    rho = lat["rho"].flatten()
    n = A["depth_max"].detach().flatten()
    g = A["tool_gate"].detach().flatten()
    h = torch.sigmoid(A["halt_bias"].detach()).flatten()
    cfg: EnvConfig = lat["cfg"]
    spend_w = episode_objective(lat, {k: v.detach() for k, v in A.items()})["spend_w"]
    smooth = sum(float((A[k][:, 1:] - A[k][:, :-1]).abs().mean()) for k in A)
    # la política óptima hace TRIAJE (U invertida: abandona los imposibles) →
    # corr global d↔depth es negativa en el óptimo (panel). El seguimiento se
    # mide en la región NO saturada (d<0.6); la global se reporta descriptiva.
    lo = d < 0.6
    return {
        "corr_d_depth_low": corr(d[lo], n[lo]),          # métrica de seguimiento
        "corr_d_depth_global": corr(d, n),               # descriptiva (triaje)
        "corr_rho_tool": corr(rho, g),
        "corr_d_halt": corr(d, h),                       # esperado NEGATIVO
        "corr_B_spend": corr(lat["Bw"].flatten(), spend_w.flatten()),
        "action_smoothness": smooth,
    }
