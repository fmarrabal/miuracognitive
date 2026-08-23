"""Fase 4 v2: automodelo APRENDIDO con daño parcial (reglas AGENCY_V2.md).

Contra los fatales de v1 (matriz verdadera inyectada, planta conocida, daño
total, cero aprendizaje):
  R5: la matriz de actuadores B es ALEATORIA POR EPISODIO y el agente no la
      recibe: la identifica online desde transiciones (regresión genérica con
      init aleatoria). La planta (cuerpo de onda amortiguada) tampoco entrega
      sus constantes al agente: el modelo online ajusta el mapa completo
      (estado,acción)->Δestado.
  Daño PARCIAL y sin flag: a mitad de episodio, la columna de un actuador se
      escala ×severity∈U(0.2,0.6); hay ruido de observación.
  R1: baselines = frozen (identifica y congela: sin adaptación post-daño),
      reinit-tuned (detector scripted de cambio con umbral afinado + reajuste)
      y skyline con la B verdadera en todo momento (cota superior, no rival).
  R7: métrica GRADUADA = regret de regulación frente al skyline; dose-response
      sobre la severidad del daño.

El entorno es data-side aquí mismo (planta pequeña autocontenida) porque la
planta ES el experimento; la verdad (B, daño) vive en el escenario solo para
el skyline y las métricas.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SelfModel2Config:
    n_nodes: int = 4               # estado = (posición, velocidad) por nodo
    n_actuators: int = 4
    horizon: int = 120
    damage_tick: int = 60
    omega0: float = 0.4            # constantes de la PLANTA (el agente NO las ve)
    zeta: float = 0.35
    coupling: float = 0.15
    action_scale: float = 0.25
    obs_noise: float = 0.02
    drive_scale: float = 0.35      # perturbación exógena que hay que regular
    severity_min: float = 0.2      # daño parcial: efectividad residual
    severity_max: float = 0.6
    fit_lr: float = 0.5            # lr del ajuste online del automodelo
    fit_l2: float = 1e-3


def chain_laplacian(n: int) -> torch.Tensor:
    L = torch.zeros(n, n)
    for i in range(n - 1):
        L[i, i] += 1; L[i + 1, i + 1] += 1
        L[i, i + 1] -= 1; L[i + 1, i] -= 1
    return L


class WavePlant:
    """Planta: cadena de osciladores amortiguados acoplados + actuadores B.
    La VERDAD (B, daño) vive aquí; el agente solo ve estados ruidosos."""

    def __init__(self, cfg: SelfModel2Config, batch: int, gen: torch.Generator,
                 severity: float | None = None, device="cpu"):
        self.cfg = cfg
        self.B = batch
        n, a = cfg.n_nodes, cfg.n_actuators
        self.L = chain_laplacian(n).to(device)
        # B aleatoria por episodio (columnas normalizadas): NADA que memorizar
        Bm = torch.randn(batch, n, a, generator=gen).to(device)
        self.B_true = cfg.action_scale * Bm / Bm.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.damaged_col = torch.randint(0, a, (batch,), generator=gen).to(device)
        if severity is None:
            sev = cfg.severity_min + (cfg.severity_max - cfg.severity_min) \
                * torch.rand(batch, generator=gen)
        else:
            sev = torch.full((batch,), float(severity))
        self.severity = sev.to(device)
        self.pos = torch.zeros(batch, n, device=device)
        self.vel = torch.zeros(batch, n, device=device)
        self.gen = gen
        self.device = device
        self.t = 0

    def B_eff(self) -> torch.Tensor:
        """Matriz efectiva (con daño a partir de damage_tick, sin flag)."""
        Bm = self.B_true.clone()
        if self.t >= self.cfg.damage_tick:
            idx = self.damaged_col.view(self.B, 1, 1).expand(-1, Bm.shape[1], 1)
            col = Bm.gather(2, idx)
            Bm = Bm.scatter(2, idx, col * self.severity.view(self.B, 1, 1))
        return Bm

    def observe(self) -> torch.Tensor:
        noise = self.cfg.obs_noise * torch.randn(
            self.B, 2 * self.cfg.n_nodes, generator=self.gen).to(self.device)
        return torch.cat([self.pos, self.vel], dim=-1) + noise

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B,A) continua en [-1,1]. Devuelve Δestado observado (B,2n)."""
        c = self.cfg
        drive = c.drive_scale * torch.randn(self.B, c.n_nodes,
                                            generator=self.gen).to(self.device) \
            * (torch.rand(self.B, 1, generator=self.gen).to(self.device) < 0.15)
        force = torch.bmm(self.B_eff(), action.unsqueeze(-1)).squeeze(-1)
        acc = (-(c.omega0 ** 2) * self.pos
               - 2 * c.zeta * c.omega0 * self.vel
               - c.coupling * self.pos @ self.L.T
               + force + drive)
        before = torch.cat([self.pos, self.vel], dim=-1)
        self.vel = self.vel + acc
        self.pos = self.pos + self.vel
        self.t += 1
        return torch.cat([self.pos, self.vel], dim=-1) - before

    def cost(self) -> torch.Tensor:
        return (self.pos ** 2).sum(-1) + 0.1 * (self.vel ** 2).sum(-1)


class OnlineSelfModel:
    """Automodelo GENÉRICO aprendido online: ajusta W en Δestado ≈ W·[obs;acción]
    por gradiente, INIT ALEATORIA (R5: sin verdad inyectada)."""

    def __init__(self, cfg: SelfModel2Config, batch: int, device="cpu",
                 gen: torch.Generator | None = None):
        n, a = 2 * cfg.n_nodes, cfg.n_actuators
        self.cfg = cfg
        self.W = 0.01 * torch.randn(batch, n, n + a, generator=gen).to(device)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = torch.cat([obs, action], dim=-1)
        return torch.bmm(self.W, z.unsqueeze(-1)).squeeze(-1)

    def update(self, obs, action, delta):
        z = torch.cat([obs, action], dim=-1)            # (B,D)
        err = self.predict(obs, action) - delta         # (B,n)
        grad = torch.bmm(err.unsqueeze(-1), z.unsqueeze(1))  # (B,n,D)
        denom = (z ** 2).sum(-1).view(-1, 1, 1) + 1.0   # normalized LMS
        self.W = self.W - self.cfg.fit_lr * grad / denom - self.cfg.fit_l2 * self.W
        return float(err.square().mean())


def plan_action(model: OnlineSelfModel, obs: torch.Tensor,
                n_candidates: int = 64, gen: torch.Generator | None = None) -> torch.Tensor:
    """Control por muestreo: elige la acción candidata cuyo estado PREDICHO por
    el AUTOMODELO minimiza el coste (el plan usa el modelo aprendido, no la
    planta). Idéntico para todos los brazos: solo cambia CÓMO se mantiene W."""
    B, device = obs.shape[0], obs.device
    cand = 2 * torch.rand(n_candidates, model.W.shape[-1] - obs.shape[-1],
                          generator=gen).to(device) - 1
    n = obs.shape[-1] // 2
    best_a, best_c = None, None
    for i in range(cand.shape[0]):
        a = cand[i].unsqueeze(0).expand(B, -1)
        nxt = obs + model.predict(obs, a)
        cost = (nxt[:, :n] ** 2).sum(-1) + 0.1 * (nxt[:, n:] ** 2).sum(-1)
        if best_c is None:
            best_a, best_c = a.clone(), cost
        else:
            better = cost < best_c
            best_a = torch.where(better.unsqueeze(-1), a, best_a)
            best_c = torch.where(better, cost, best_c)
    return best_a


def run_episode(cfg: SelfModel2Config, batch: int, seed: int, arm: str,
                severity: float | None = None, reinit_threshold: float = 0.0,
                device="cpu") -> dict:
    """arm ∈ {adaptive, frozen, reinit, oracle, random}.
    adaptive: automodelo online SIEMPRE aprendiendo.
    frozen:   aprende hasta damage_tick/2 y congela (sin adaptación post-daño).
    reinit:   frozen + detector scripted: si el error de predicción medio en
              ventana supera reinit_threshold (AFINADO), re-lanza el ajuste.
    oracle:   skyline con B_eff verdadera en cada tick (cota superior).
    random:   acciones aleatorias (suelo).
    """
    gen = torch.Generator().manual_seed(seed)
    plant = WavePlant(cfg, batch, gen, severity=severity, device=device)
    model = OnlineSelfModel(cfg, batch, device=device, gen=gen)
    plan_gen = torch.Generator().manual_seed(seed + 999)
    costs, pred_errs = [], []
    err_window = []
    frozen = False
    for t in range(cfg.horizon):
        obs = plant.observe()
        if arm == "random":
            a = 2 * torch.rand(batch, cfg.n_actuators, generator=plan_gen).to(device) - 1
        elif arm == "oracle":
            # Skyline: mapa lineal VERDADERO completo Δs = A_true·s + [B;B]·a
            #   Δvel = -K·pos - D·vel + B·a ;  Δpos = vel + Δvel
            om = OnlineSelfModel(cfg, batch, device=device)
            n = cfg.n_nodes
            n2, na = 2 * n, cfg.n_actuators
            K = (cfg.omega0 ** 2) * torch.eye(n, device=device) \
                + cfg.coupling * plant.L
            D = 2 * cfg.zeta * cfg.omega0 * torch.eye(n, device=device)
            I = torch.eye(n, device=device)
            A_true = torch.zeros(n2, n2, device=device)
            A_true[:n, :n], A_true[:n, n:] = -K, I - D
            A_true[n:, :n], A_true[n:, n:] = -K, -D
            W = torch.zeros(batch, n2, n2 + na, device=device)
            W[:, :, :n2] = A_true.unsqueeze(0)
            Beff = plant.B_eff()
            W[:, :, n2:] = torch.cat([Beff, Beff], dim=1)
            om.W = W
            a = plan_action(om, obs, gen=plan_gen)
        else:
            a = plan_action(model, obs, gen=plan_gen)
        delta = plant.step(a)
        err = float((model.predict(obs, a) - delta).square().mean())
        pred_errs.append(err)
        err_window.append(err)
        if len(err_window) > 5:
            err_window.pop(0)
        do_update = arm == "adaptive"
        if arm in ("frozen", "reinit"):
            if t < cfg.damage_tick // 2:
                do_update = True                       # fase de identificación
            elif arm == "reinit" and sum(err_window) / len(err_window) > reinit_threshold:
                do_update = True                       # detector scripted disparado
        if do_update:
            model.update(obs, a, delta)
        costs.append(plant.cost())
    costs = torch.stack(costs, dim=1)                  # (B,T)
    pre = costs[:, :cfg.damage_tick].mean().item()
    post = costs[:, cfg.damage_tick:].mean().item()
    return {"cost_pre": pre, "cost_post": post,
            "pred_err_final": pred_errs[-1],
            "cost_total": float(costs.mean())}
