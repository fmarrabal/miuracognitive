"""F2X: agentes del campo ejecutivo (reglas AGENCY_V2).

- LearnedExecPolicy: GRU end-to-end por REINFORCE sobre el retorno ambiental
  (R4). Observa progreso, urgencia, valores, plazos restantes, energía y
  acción previa; el triaje/compromiso/automantenimiento deben EMERGER.
- Scripted afinables (R1), familia de scheduling seria:
    ValueDensityPolicy(e_min): greedy por densidad de valor entre proyectos
      FACTIBLES (plazo alcanzable), con recarga por umbral de energía.
    EDFPolicy(e_min): earliest-deadline-first factible (óptimo clásico en
      scheduling monoprocesador) + recarga por umbral.
    DesignerPolicy(e_min, margin): densidad de valor + histéresis + override
      de crisis POR PRECURSOR (conoce la regla generativa) + recarga.
- Ablación: memoryless. HBP exploratorio con ganancia aprendible (R8).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data.exec_v2 import (ExecV2Config, ExecV2Dataset, ExecV2Scenario,
                          ExecV2World, urgency_signal)


@dataclass
class LearnedExecConfig:
    hidden: int = 128
    memoryless: bool = False
    use_hbp: bool = False
    hbp_alpha_const: float | None = None


class LearnedExecPolicy(nn.Module):
    def __init__(self, env_cfg: ExecV2Config, cfg: LearnedExecConfig | None = None):
        super().__init__()
        self.env_cfg = env_cfg
        self.cfg = cfg or LearnedExecConfig()
        k, a = env_cfg.n_projects, env_cfg.n_actions
        d_in = 4 * k + a + 2                    # p,u,v,plazo_rest + prev_a + energía + t/T
        if self.cfg.memoryless:
            self.net = nn.Sequential(nn.Linear(d_in, self.cfg.hidden), nn.Tanh(),
                                     nn.Linear(self.cfg.hidden, a))
            self.vhead = nn.Sequential(nn.Linear(d_in, self.cfg.hidden), nn.Tanh(),
                                       nn.Linear(self.cfg.hidden, 1))
        else:
            self.cell = nn.GRUCell(d_in, self.cfg.hidden)
            self.head = nn.Linear(self.cfg.hidden, a)
            self.vhead = nn.Linear(self.cfg.hidden, 1)   # crítico (baseline A2C)
        self._state = None
        self._last_obs = None
        if self.cfg.use_hbp:
            from model.hbp import HBPConfig, HomeostaticBackgroundProcessor
            hcfg = HBPConfig(n_nodes=3, d_f=8, d_e=8, d_u=16, d_intero=4,
                             alpha_const=self.cfg.hbp_alpha_const,
                             diff_solver="implicit")
            self.hbp = HomeostaticBackgroundProcessor(hcfg)
            self.intero_proj = nn.Linear(2 * k + 1, 3 * hcfg.d_intero)
            self.hbp_priority = nn.Linear(hcfg.d_h, a)
            nn.init.zeros_(self.hbp_priority.weight)
            nn.init.zeros_(self.hbp_priority.bias)
            self.hbp_gain = nn.Parameter(torch.tensor(0.1))

    def reset(self, batch_size: int, device):
        if not self.cfg.memoryless:
            self._state = torch.zeros(batch_size, self.cfg.hidden, device=device)
        if self.cfg.use_hbp:
            self.hbp.reset_state(batch_size, device=device, dtype=torch.float32)

    def logits(self, obs: torch.Tensor, intero_raw: torch.Tensor | None = None):
        self._last_obs = obs
        if self.cfg.memoryless:
            out = self.net(obs)
        else:
            self._state = self.cell(obs, self._state)
            out = self.head(self._state)
        if self.cfg.use_hbp and intero_raw is not None:
            b = obs.shape[0]
            vei = self.hbp.step(self.intero_proj(intero_raw).view(b, 3, -1))
            out = out + self.hbp_gain * self.hbp_priority(vei.mean(dim=1))
        return out

    def value(self) -> torch.Tensor:
        """V(s) del último estado visto (crítico A2C)."""
        if self.cfg.memoryless:
            return self.vhead(self._last_obs).squeeze(-1)
        return self.vhead(self._state).squeeze(-1)


def _obs(cfg, sc, world, u_t, prev_a, t, device):
    T = cfg.horizon
    rest = ((sc.deadlines.to(device) - t).clamp_min(0).float()
            / T)                                            # plazo restante norm.
    frac = torch.full((world.p.shape[0], 1), t / T, device=device)
    return torch.cat([world.p * (~world.done).float(), u_t,
                      sc.values.to(device), rest, prev_a,
                      world.energy.unsqueeze(-1), frac], dim=-1)


def rollout_learned(cfg: ExecV2Config, sc: ExecV2Scenario,
                    policy: LearnedExecPolicy, sample: bool, device):
    B = sc.batch_size
    u = urgency_signal(cfg, sc).to(device)
    world = ExecV2World(cfg, sc, device)
    prev_a = torch.zeros(B, cfg.n_actions, device=device)
    policy.reset(B, device)
    logps, ents, values = [], [], []
    for t in range(cfg.horizon):
        intero = torch.cat([world.p, u[:, t],
                            world.energy.unsqueeze(-1)], dim=-1) \
            if policy.cfg.use_hbp else None
        lg = policy.logits(_obs(cfg, sc, world, u[:, t], prev_a, t, device),
                           intero)
        dist = torch.distributions.Categorical(logits=lg)
        a = dist.sample() if sample else lg.argmax(dim=-1)
        if sample:
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
            values.append(policy.value())
        world.step(a)
        prev_a = torch.nn.functional.one_hot(a, cfg.n_actions).float()
    out = {"world": world, "return": world.final_return()}
    if sample:
        out["logp_t"] = torch.stack(logps, dim=1)              # (B,T)
        out["rewards_t"] = torch.stack(world.tick_rewards, dim=1)  # (B,T)
        out["values_t"] = torch.stack(values, dim=1)           # (B,T)
        out["entropy"] = torch.stack(ents, dim=1).mean()
    return out


def train_learned_exec(cfg: ExecV2Config, policy: LearnedExecPolicy, seed: int,
                       steps: int = 1500, batch: int = 256, lr: float = 1e-3,
                       device="cpu"):
    """REINFORCE con REWARD-TO-GO sobre la recompensa incremental (el mismo
    retorno ambiental entregado en su tick; invariante verificado en smoke) +
    baseline por paso (media de batch) + bono de entropía. La versión a nivel
    de trayectoria no asignaba crédito en horizonte 64 (piloto: 2.50 vs
    designer 3.68) — iteración de RECETA, no de entorno (G1)."""
    import math
    from dataclasses import replace as _replace
    torch.manual_seed(seed)
    policy = policy.to(device)
    if policy.cfg.use_hbp:
        policy.hbp.pin_fp32()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    policy.train()
    # CURRICULUM de ENTRENAMIENTO (el entorno de EVAL no se toca): proyectos
    # cortos al principio (se descubre que completar paga) y reales después.
    stages = [(0.4, _replace(cfg, work_rate=0.12)),
              (0.7, _replace(cfg, work_rate=0.09)),
              (1.01, cfg)]
    datasets = {id(c): ExecV2Dataset(c, seed=seed + i)
                for i, (_, c) in enumerate(stages)}
    for s in range(1, steps + 1):
        frac = s / steps
        cur_cfg = next(c for cut, c in stages if frac <= cut)
        cur = lr * min(1.0, s / 50) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = cur
        sc = datasets[id(cur_cfg)].batch(batch).to(device)
        out = rollout_learned(cur_cfg, sc, policy, sample=True, device=device)
        g2go = torch.flip(torch.cumsum(torch.flip(out["rewards_t"], [1]), 1), [1])
        adv = g2go - out["values_t"]                      # ventaja con crítico
        loss = (-(adv.detach() * out["logp_t"]).mean()
                + 0.5 * adv.pow(2).mean()                 # MSE del crítico
                - 0.01 * out["entropy"])
        if policy.cfg.use_hbp:
            loss = loss + 0.1 * policy.hbp.stability_penalty()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return policy


# ------------------------- scripted afinables (R1) -------------------------- #

class _ScriptedBase:
    def __init__(self, cfg: ExecV2Config, e_min: float):
        self.cfg, self.e_min = cfg, e_min
        self._u_hist = []
        self._current = None

    def reset(self, batch_size: int, device):
        self._u_hist = []
        self._current = torch.zeros(batch_size, dtype=torch.long, device=device)

    def _feasible(self, sc, world, t, device):
        """Proyectos aún completables antes de su plazo (con la verdad
        observable: plazo y progreso; sin usar verdad oculta)."""
        need = (1.0 - world.p).clamp_min(0) / self.cfg.work_rate
        left = (sc.deadlines.to(device) - t).clamp_min(0).float()
        return (~world.done) & (need <= left)

    def _recharge(self, world):
        return world.energy < self.e_min


class ValueDensityPolicy(_ScriptedBase):
    """Greedy por densidad de valor entre factibles + recarga por umbral."""

    def act(self, sc, world, u_t, t, device):
        feas = self._feasible(sc, world, t, device)
        need = ((1.0 - world.p).clamp_min(1e-6) / self.cfg.work_rate)
        density = sc.values.to(device) / need
        density = torch.where(feas, density, torch.full_like(density, -1e9))
        choice = density.argmax(-1) + 1
        none = ~feas.any(-1)
        a = torch.where(self._recharge(world) | none,
                        torch.zeros_like(choice), choice)
        return a


class EDFPolicy(_ScriptedBase):
    """Earliest-deadline-first entre factibles + recarga por umbral."""

    def act(self, sc, world, u_t, t, device):
        feas = self._feasible(sc, world, t, device)
        dl = sc.deadlines.to(device).float()
        dl = torch.where(feas, dl, torch.full_like(dl, 1e9))
        choice = dl.argmin(-1) + 1
        none = ~feas.any(-1)
        return torch.where(self._recharge(world) | none,
                           torch.zeros_like(choice), choice)


class DesignerPolicy(_ScriptedBase):
    """Densidad de valor + histéresis + override de crisis POR PRECURSOR
    (conoce la regla generativa) + recarga. El listón del diseñador."""

    def __init__(self, cfg, e_min: float, margin: float):
        super().__init__(cfg, e_min)
        self.margin = margin

    def act(self, sc, world, u_t, t, device):
        feas = self._feasible(sc, world, t, device)
        need = ((1.0 - world.p).clamp_min(1e-6) / self.cfg.work_rate)
        score = sc.values.to(device) / need
        score = torch.where(feas, score, torch.full_like(score, -1e9))
        greedy = score.argmax(-1) + 1
        cur_idx = (self._current - 1).clamp_min(0)
        cur_ok = self._current > 0
        cur_score = score.gather(1, cur_idx.unsqueeze(1)).squeeze(1)
        best = score.max(-1).values
        switch = (best > cur_score + self.margin) | ~cur_ok
        choice = torch.where(switch, greedy, self._current)
        if len(self._u_hist) >= 2:
            u1, u2 = self._u_hist[-1], self._u_hist[-2]
            rising = (u_t >= 0.99) & (u1 > u2 + 0.05) & (u1 < 0.99)
            has = rising.any(-1)
            choice = torch.where(has, rising.float().argmax(-1) + 1, choice)
        self._u_hist.append(u_t)
        none = ~feas.any(-1)
        a = torch.where(self._recharge(world) | none,
                        torch.zeros_like(choice), choice)
        self._current = a
        return a


@torch.no_grad()
def eval_exec_policy(cfg: ExecV2Config, sc: ExecV2Scenario, policy,
                     device="cpu") -> dict:
    from data.exec_v2 import portfolio_skyline
    B = sc.batch_size
    u = urgency_signal(cfg, sc).to(device)
    if isinstance(policy, LearnedExecPolicy):
        policy.eval()
        out = rollout_learned(cfg, sc, policy, sample=False, device=device)
        world = out["world"]
    else:
        world = ExecV2World(cfg, sc, device)
        policy.reset(B, device)
        for t in range(cfg.horizon):
            world.step(policy.act(sc, world, u[:, t], t, device))
    ret = world.final_return()
    sky = portfolio_skyline(cfg, sc).to(device)
    on_time = world.done & (world.done_at <= sc.deadlines.to(device))
    return {
        "return": float(ret.mean()),
        "regret_vs_skyline": float((sky - ret).mean() / sky.mean().clamp_min(1e-6)),
        "completed_ontime": float(on_time.float().sum(-1).mean()),
        "collapses": float(world.collapses.mean()),
        "energy_final": float(world.energy.mean()),
    }
