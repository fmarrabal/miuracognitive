"""Fase 2 v2: agentes de compromiso endógeno (reglas AGENCY_V2.md).

- LearnedGoalPolicy: GRU entrenada por REINFORCE sobre el RETORNO ambiental
  (R4: cero supervisión hacia reglas; el compromiso debe emerger del coste de
  cambiar y del valor de atender crisis reales). La ablación memoryless es la
  misma política sin recurrencia (reportada como ablación, R1).
- Scripted afinables (R1): greedy(w); histéresis(w,m); smart(w,m) = histéresis
  + detector del PRECURSOR (conoce la regla generativa: baseline de
  conocimiento-del-diseñador, el listón más alto).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data.goals_v2 import (Goals2Config, Goals2Dataset, Goals2Scenario,
                           episode_return, urgency_signal)


@dataclass
class LearnedGoalConfig:
    hidden: int = 64
    memoryless: bool = False       # ablación: sin recurrencia


class LearnedGoalPolicy(nn.Module):
    def __init__(self, env_cfg: Goals2Config, cfg: LearnedGoalConfig | None = None):
        super().__init__()
        self.env_cfg = env_cfg
        self.cfg = cfg or LearnedGoalConfig()
        k, a = env_cfg.n_projects, env_cfg.n_actions
        d_in = 3 * k + a + 1
        if self.cfg.memoryless:
            self.net = nn.Sequential(nn.Linear(d_in, self.cfg.hidden), nn.Tanh(),
                                     nn.Linear(self.cfg.hidden, a))
        else:
            self.cell = nn.GRUCell(d_in, self.cfg.hidden)
            self.head = nn.Linear(self.cfg.hidden, a)
        self._state = None

    def reset(self, batch_size: int, device):
        if not self.cfg.memoryless:
            self._state = torch.zeros(batch_size, self.cfg.hidden, device=device)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        if self.cfg.memoryless:
            return self.net(obs)
        self._state = self.cell(obs, self._state)
        return self.head(self._state)


def _make_obs(cfg, sc, u_t, p, done, prev_a, t):
    frac = torch.full((p.shape[0], 1), t / cfg.horizon, device=p.device)
    return torch.cat([p * (~done).float(), u_t, sc.values,
                      prev_a, frac], dim=-1)


def rollout_learned(cfg: Goals2Config, sc: Goals2Scenario,
                    policy: LearnedGoalPolicy, sample: bool):
    """Rollout con la MISMA dinámica que episode_return (progreso/decay/done).
    sample=True: muestreo categórico + logprobs (REINFORCE); False: argmax."""
    B = sc.batch_size
    device = sc.values.device
    u = urgency_signal(cfg, sc).to(device)
    K = cfg.n_projects
    p = torch.zeros(B, K, device=device)
    done = torch.zeros(B, K, dtype=torch.bool, device=device)
    prev_a = torch.zeros(B, cfg.n_actions, device=device)
    policy.reset(B, device)
    actions, logps, ents = [], [], []
    for t in range(cfg.horizon):
        logits = policy.logits(_make_obs(cfg, sc, u[:, t], p, done, prev_a, t))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample() if sample else logits.argmax(dim=-1)
        if sample:
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
        worked = torch.nn.functional.one_hot(a, cfg.n_actions)[:, 1:].float()
        p = p + cfg.work_rate * worked
        p = torch.where(done | worked.bool(), p, p * (1.0 - cfg.decay))
        done = done | (p >= 1.0)
        prev_a = torch.nn.functional.one_hot(a, cfg.n_actions).float()
        actions.append(a)
    acts = torch.stack(actions, dim=1)                 # (B,T)
    out = {"actions": acts, "return": episode_return(cfg, sc, acts)}
    if sample:
        out["logp"] = torch.stack(logps, dim=1).sum(-1)
        out["entropy"] = torch.stack(ents, dim=1).mean()
    return out


def train_learned_goals(env_cfg: Goals2Config, policy: LearnedGoalPolicy,
                        seed: int, steps: int = 400, batch: int = 256,
                        lr: float = 1e-3, device="cpu"):
    """REINFORCE con baseline de media de batch + bono de entropía."""
    torch.manual_seed(seed)
    ds = Goals2Dataset(env_cfg, seed=seed)
    policy = policy.to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    policy.train()
    for _ in range(steps):
        sc = ds.batch(batch).to(device)
        out = rollout_learned(env_cfg, sc, policy, sample=True)
        adv = out["return"] - out["return"].mean()
        loss = -(adv.detach() * out["logp"]).mean() - 0.01 * out["entropy"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return policy


# ------------------------- scripted afinables (R1) -------------------------- #

class ScriptedGoalPolicy:
    """greedy / histéresis / smart(+detector de precursor) sobre el score
    s_k = w·(v_k·(1−p_k)·no_completado) + (1−w)·u_k."""

    def __init__(self, cfg: Goals2Config, mode: str, w: float,
                 margin: float = 0.0):
        assert mode in ("greedy", "hysteresis", "smart")
        self.cfg, self.mode, self.w, self.m = cfg, mode, w, margin
        self._current = None
        self._u_hist = []

    def reset(self, batch_size: int, device):
        self._current = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._u_hist = []

    def act(self, sc: Goals2Scenario, u_t, p, done) -> torch.Tensor:
        score = (self.w * sc.values * (1 - p).clamp(0, 1) * (~done).float()
                 + (1 - self.w) * u_t)
        greedy = score.argmax(-1) + 1
        if self.mode == "greedy":
            self._current = greedy
            return greedy
        cur_idx = (self._current - 1).clamp_min(0)
        cur_score = score.gather(1, cur_idx.unsqueeze(1)).squeeze(1)
        never = self._current == 0
        best = score.max(-1).values
        switch = (best > cur_score + self.m) | never
        choice = torch.where(switch, greedy, self._current)
        if self.mode == "smart" and len(self._u_hist) >= 2:
            # Detector del precursor (regla generativa conocida): pico ~1.0
            # precedido de rampa creciente en 2 ticks -> crisis REAL.
            u1, u2 = self._u_hist[-1], self._u_hist[-2]
            rising = (u_t >= 0.99) & (u1 > u2 + 0.05) & (u1 < 0.99)
            has = rising.any(-1)
            target = rising.float().argmax(-1) + 1
            choice = torch.where(has, target, choice)
        self._u_hist.append(u_t)
        self._current = choice
        return choice


@torch.no_grad()
def eval_goal_policy(cfg: Goals2Config, sc: Goals2Scenario, policy) -> dict:
    """Evalúa retorno + métricas de compromiso; scripted y aprendidos."""
    B = sc.batch_size
    device = sc.values.device
    if isinstance(policy, LearnedGoalPolicy):
        policy.eval()
        out = rollout_learned(cfg, sc, policy, sample=False)
        acts = out["actions"]
    else:
        u = urgency_signal(cfg, sc).to(device)
        K = cfg.n_projects
        p = torch.zeros(B, K, device=device)
        done = torch.zeros(B, K, dtype=torch.bool, device=device)
        policy.reset(B, device)
        seq = []
        for t in range(cfg.horizon):
            a = policy.act(sc, u[:, t], p, done)
            worked = torch.nn.functional.one_hot(a, cfg.n_actions)[:, 1:].float()
            p = p + cfg.work_rate * worked
            p = torch.where(done | worked.bool(), p, p * (1.0 - cfg.decay))
            done = done | (p >= 1.0)
            seq.append(a)
        acts = torch.stack(seq, dim=1)
    ret = episode_return(cfg, sc, acts)
    # métricas de compromiso: atención a crisis vs distractores (verdad solo aquí)
    S = sc.spike_t.shape[1]
    att = {True: [0, 0], False: [0, 0]}
    for s in range(S):
        t_s, proj = sc.spike_t[:, s], sc.spike_proj[:, s]
        crisis = sc.spike_is_crisis[:, s]
        attended = torch.zeros(B, dtype=torch.bool, device=device)
        for dt in range(cfg.crisis_window):
            tt = (t_s + dt).clamp_max(cfg.horizon - 1)
            attended |= acts.gather(1, tt.unsqueeze(1)).squeeze(1) == proj + 1
        for flag in (True, False):
            m = crisis == flag
            att[flag][0] += int((attended & m.to(device)).sum())
            att[flag][1] += int(m.sum())
    switches = (acts[:, 1:] != acts[:, :-1]).float().mean()
    return {
        "return": float(ret.mean()),
        "crisis_attended": att[True][0] / max(1, att[True][1]),
        "distractor_attended": att[False][0] / max(1, att[False][1]),
        "switch_rate": float(switches),
    }
