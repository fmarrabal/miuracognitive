"""AHA-2: agentes (rediseño post-auditoría, reglas AGENCY_V2.md).

- LearnedPolicy: GRU end-to-end. NO recibe predictor separado ni oráculos: si
  quiere anticipar debe integrar los cues imperfectos en su memoria (R3, R6).
- La ABLACIÓN cue-blind es la misma red con el canal de cues a cero, y se
  reporta como ablación (R1).
- Baselines EXTERNOS scripted afinables por grid-search (R1):
    ThresholdPolicy(θ): reactivo puro bien afinado.
    CueFollowerPolicy(c, τ): programa la acción τ ticks tras cada cue.
    ComboPolicy(θ, c, τ): unión de ambas reglas.
- HBPPolicy: ejecutivo GRU + campo homeostático con ganancia de acoplamiento
  APRENDIBLE (init pequeña; diagnóstico de interferencia de la auditoría) y
  contraste de integrador puro vía alpha_const∈{1,0} con rangos IDÉNTICOS (R8).

La pérdida de entrenamiento es el objetivo homeostático AMBIENTAL, idéntico
para todos los brazos aprendidos: déficit bajo setpoint + castigo por
inviabilidad. El coste de actuar lo cobra el entorno (R4).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data.aha_v2 import AHA2Config, AHA2Scenario, rollout_world
from model.hbp import HBPConfig, HomeostaticBackgroundProcessor


# --------------------------------------------------------------------------- #
#  Política aprendida (GRU end-to-end)
# --------------------------------------------------------------------------- #

@dataclass
class LearnedPolicyConfig:
    hidden: int = 64
    cue_blind: bool = False        # ablación: canal de cues a cero
    use_hbp: bool = False
    hbp_alpha_const: float | None = None   # 1.0=onda, 0.0=difusión (rangos idénticos)


class LearnedAHA2Policy(nn.Module):
    """GRU que consume [levels, error, velocity, cue, prev_action] por tick."""

    def __init__(self, env_cfg: AHA2Config, cfg: LearnedPolicyConfig | None = None):
        super().__init__()
        self.env_cfg = env_cfg
        self.cfg = cfg or LearnedPolicyConfig()
        n, a = env_cfg.n_needs, env_cfg.n_actions
        d_in = 4 * n + a + 1                      # +1: batería (interocepción)
        self.cell = nn.GRUCell(d_in, self.cfg.hidden)
        self.head = nn.Linear(self.cfg.hidden, a)
        self._state: torch.Tensor | None = None
        if self.cfg.use_hbp:
            # Campo homeostático pequeño (3 nodos: percepción-ejecutivo-actuador)
            # con el MISMO rango de parámetros en ambos brazos; el integrador se
            # fija con alpha_const (onda pura vs difusión pura, solver implícito).
            hcfg = HBPConfig(n_nodes=3, d_f=8, d_e=8, d_u=16, d_intero=4,
                             alpha_const=self.cfg.hbp_alpha_const,
                             diff_solver="implicit")
            self.hbp = HomeostaticBackgroundProcessor(hcfg)
            self.intero_proj = nn.Linear(2 * n, 3 * hcfg.d_intero)
            self.hbp_priority = nn.Linear(hcfg.d_h, a)
            nn.init.zeros_(self.hbp_priority.weight)
            nn.init.zeros_(self.hbp_priority.bias)
            # Ganancia de acoplamiento APRENDIBLE, init pequeña: el campo debe
            # ganarse su influencia (la ganancia fija 2.0 de v1 interfería).
            self.hbp_gain = nn.Parameter(torch.tensor(0.1))

    def reset(self, batch_size: int, device):
        self._state = torch.zeros(batch_size, self.cfg.hidden, device=device)
        if self.cfg.use_hbp:
            self.hbp.reset_state(batch_size, device=device, dtype=torch.float32)

    tau: float = 1.0    # temperatura (recocida en entrenamiento; 1.0 en eval)

    def __call__(self, obs: dict, t: int) -> torch.Tensor:
        cue = torch.zeros_like(obs["cue"]) if self.cfg.cue_blind else obs["cue"]
        x = torch.cat([obs["levels"], obs["error"], obs["velocity"], cue,
                       obs["prev_action"], obs["battery"]], dim=-1)
        self._state = self.cell(x, self._state)
        logits = self.head(self._state)
        if self.cfg.use_hbp:
            b = obs["levels"].shape[0]
            intero = self.intero_proj(
                torch.cat([obs["error"], obs["velocity"]], dim=-1)
            ).view(b, 3, -1)
            vei = self.hbp.step(intero)
            logits = logits + self.hbp_gain * self.hbp_priority(vei.mean(dim=1))
        return torch.softmax(logits / max(self.tau, 1e-3), dim=-1)


def homeostatic_loss(cfg: AHA2Config, levels_hist: torch.Tensor,
                     viol_weight: float = 4.0) -> torch.Tensor:
    """Objetivo AMBIENTAL idéntico para todos los brazos aprendidos (R4):
    déficit medio bajo el umbral de confort + castigo por acercarse a la
    inviabilidad. Sin términos anti-estrategia: actuar de más ya lo cobra el
    entorno vía action_cost."""
    comfort = cfg.setpoint_min
    deficit = torch.relu(comfort - levels_hist).mean()
    danger = torch.relu(cfg.viability_threshold + 0.05 - levels_hist).mean()
    return deficit + viol_weight * danger


def trajectory_reward(cfg: AHA2Config, out: dict) -> torch.Tensor:
    """Recompensa AMBIENTAL por trayectoria (B,): supervivencia + confort.
    Idéntica para todos los brazos aprendidos; sin términos anti-estrategia
    (el coste de actuar ya lo cobra el mundo vía action_cost)."""
    lh = out["levels_hist"]
    deficit = torch.relu(cfg.setpoint_min - lh).mean(dim=(1, 2))
    danger = torch.relu(cfg.viability_threshold + 0.05 - lh).mean(dim=(1, 2))
    survived = (~out["violated"]).float()
    return survived - deficit - 4.0 * danger


def train_learned(env_cfg: AHA2Config, policy: LearnedAHA2Policy, seed: int,
                  steps: int = 800, batch: int = 256, lr: float = 1e-3,
                  device: torch.device | str = "cpu"):
    """REINFORCE con baseline de media de batch + bono de entropía (la receta
    que el piloto de fase 2 validó; el ST-Gumbel entrenaba políticas que
    dependían del ruido de muestreo y colapsaban bajo argmax — documentado)."""
    import math as _math
    from data.aha_v2 import AHA2Dataset
    torch.manual_seed(seed)
    ds = AHA2Dataset(env_cfg, seed=seed)
    policy = policy.to(device)
    if policy.cfg.use_hbp:
        policy.hbp.pin_fp32()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    policy.train()
    for s in range(1, steps + 1):
        frac = s / max(steps, 1)
        cur_lr = lr * min(1.0, s / 50) * 0.5 * (1 + _math.cos(_math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        sc = ds.batch(batch).to(device)
        policy.reset(batch, device)
        out = rollout_world(env_cfg, sc, policy, mode="sample")
        reward = trajectory_reward(env_cfg, out)
        adv = reward - reward.mean()
        loss = -(adv.detach() * out["logp"]).mean() - 0.01 * out["entropy"]
        if policy.cfg.use_hbp:
            loss = loss + 0.1 * policy.hbp.stability_penalty()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return policy


# --------------------------------------------------------------------------- #
#  Baselines scripted externos (afinables; R1)
# --------------------------------------------------------------------------- #

class ThresholdPolicy:
    """Reactivo hábil: restaura la necesidad mínima cuando cae bajo θ."""

    def __init__(self, env_cfg: AHA2Config, theta: float):
        self.c, self.theta = env_cfg, theta

    def reset(self, batch_size: int, device):
        pass

    def __call__(self, obs: dict, t: int) -> torch.Tensor:
        levels = obs["levels"]
        need = levels.argmin(dim=-1)
        act = (levels.min(dim=-1).values < self.theta).long() * (need + 1)
        return torch.nn.functional.one_hot(act, self.c.n_actions).float()


class CueFollowerPolicy:
    """Anticipador scripted: al ver cue>c en la necesidad i, programa una
    acción para i dentro de τ ticks (τ afinado ≈ adelanto medio − delay)."""

    def __init__(self, env_cfg: AHA2Config, c_min: float, tau: int):
        self.c, self.c_min, self.tau = env_cfg, c_min, tau
        self._sched = None

    def reset(self, batch_size: int, device):
        self._sched = torch.zeros(batch_size, self.tau + 1, self.c.n_needs,
                                  device=device)

    def __call__(self, obs: dict, t: int) -> torch.Tensor:
        cue = obs["cue"]
        self._sched[:, -1] = self._sched[:, -1] + (cue > self.c_min).float()
        due = self._sched[:, 0]                                  # (B,N)
        self._sched = torch.cat(
            [self._sched[:, 1:], torch.zeros_like(self._sched[:, :1])], dim=1)
        act = torch.where(due.sum(-1) > 0, due.argmax(-1) + 1,
                          torch.zeros_like(due.sum(-1), dtype=torch.long))
        return torch.nn.functional.one_hot(act, self.c.n_actions).float()


class ComboPolicy:
    """Unión afinada: acción programada por cue si vence; si no, reactivo θ."""

    def __init__(self, env_cfg: AHA2Config, theta: float, c_min: float, tau: int):
        self.c = env_cfg
        self.th = ThresholdPolicy(env_cfg, theta)
        self.cf = CueFollowerPolicy(env_cfg, c_min, tau)

    def reset(self, batch_size: int, device):
        self.cf.reset(batch_size, device)

    def __call__(self, obs: dict, t: int) -> torch.Tensor:
        a_cf = self.cf(obs, t)
        a_th = self.th(obs, t)
        use_cf = (a_cf[:, 0] < 0.5).unsqueeze(-1)                # cue-acción vence
        return torch.where(use_cf, a_cf, a_th)


# --------------------------------------------------------------------------- #
#  Métricas de evaluación (usan la verdad de eventos SOLO aquí)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_policy(env_cfg: AHA2Config, scenario: AHA2Scenario, policy) -> dict:
    if hasattr(policy, "eval"):
        policy.eval()
    policy.reset(scenario.batch_size, scenario.initial_levels.device)
    out = rollout_world(env_cfg, scenario, policy, mode="greedy")
    B, T, _ = scenario.cues.shape
    acts = out["actions"][:, :, 1:]                              # (B,T,N)
    survival = 1.0 - out["violated"].float().mean().item()
    time_viable = (out["levels_hist"].min(-1).values
                   >= env_cfg.viability_threshold).float().mean().item()
    act_rate = float(acts.sum(-1).mean())
    # Adelanto de anticipación: para cada evento, última acción previa en su
    # necesidad dentro de una ventana de 12 ticks (verdad SOLO para medir).
    leads, hits = [], 0
    for row in range(min(B, 256)):
        for k in range(scenario.events_t.shape[1]):
            t_e = int(scenario.events_t[row, k])
            need = int(scenario.events_need[row, k])
            w0 = max(0, t_e - 12)
            window = acts[row, w0:t_e + 1, need]
            idx = window.nonzero()
            if len(idx) > 0:
                hits += 1
                leads.append(t_e - (w0 + int(idx[-1])))
    return {
        "survival": survival,
        "time_viable": time_viable,
        "action_rate": act_rate,
        "anticipation_lead": (sum(leads) / len(leads)) if leads else 0.0,
        "event_coverage": hits / max(1, min(B, 256) * scenario.events_t.shape[1]),
    }
