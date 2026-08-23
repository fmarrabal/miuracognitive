"""Fase 3 v2: agentes de descubrimiento sobre campos desconocidos (AGENCY_V2.md).

- LearnedProber: GRU in-context que consume (x_t, y_t, t/T) y propone la
  siguiente sonda y, al final, el objetivo. Meta-entrenada entre episodios por
  BPTT a través del campo diferenciable (legítimo: el agente en eval solo ve
  observaciones; R5: sin forma funcional ni parámetros verdaderos dentro).
- Scripted afinables (R1): RandomProbe (aleatorio + argmax observado),
  GridProbe (rejilla + argmax), FDAscent (ascenso por diferencias finitas con
  paso afinado, arranque en la mejor de k sondas aleatorias).
- G-untrained (R6): el prober sin entrenar debe tener regret ~scripted-random.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data.discovery_v2 import Discovery2Config, FieldBatch, normalized_regret


@dataclass
class ProberConfig:
    hidden: int = 96


class LearnedProber(nn.Module):
    def __init__(self, cfg: Discovery2Config, pc: ProberConfig | None = None):
        super().__init__()
        self.cfg = cfg
        self.pc = pc or ProberConfig()
        self.cell = nn.GRUCell(4, self.pc.hidden)     # (x,y de la sonda, resp, t/T)
        self.probe_head = nn.Linear(self.pc.hidden, 2)
        self.target_head = nn.Linear(self.pc.hidden, 2)

    def run_episode(self, fields: FieldBatch, gen: torch.Generator | None = None):
        """Devuelve (target (B,2), probes (B,T,2)). Diferenciable end-to-end."""
        B = fields.B
        device = fields.c1.device
        h = torch.zeros(B, self.pc.hidden, device=device)
        x = torch.full((B, 2), 0.5, device=device)     # primera sonda: centro
        probes = []
        for t in range(self.cfg.n_probes):
            y = fields.probe(x, gen=gen)               # ruidosa, reparametrizada
            frac = torch.full((B, 1), t / self.cfg.n_probes, device=device)
            h = self.cell(torch.cat([x, y.unsqueeze(-1), frac], dim=-1), h)
            probes.append(x)
            x = torch.sigmoid(self.probe_head(h))      # siguiente sonda en [0,1]²
        target = torch.sigmoid(self.target_head(h))
        return target, torch.stack(probes, dim=1)


def train_prober(cfg: Discovery2Config, prober: LearnedProber, seed: int,
                 steps: int = 600, batch: int = 256, lr: float = 1e-3,
                 device="cpu"):
    """Meta-entrenamiento: minimiza E[regret] por BPTT a través del campo."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    prober = prober.to(device)
    opt = torch.optim.AdamW(prober.parameters(), lr=lr, weight_decay=0.01)
    prober.train()
    for _ in range(steps):
        fields = FieldBatch(cfg, batch, gen, device=device)
        target, _ = prober.run_episode(fields)
        # pérdida = 1 − f(objetivo)/f_max, versión diferenciable
        loss = (1.0 - fields.f(target) / fields.f_max()).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prober.parameters(), 1.0)
        opt.step()
    return prober


# ------------------------- scripted afinables (R1) -------------------------- #

class RandomProbe:
    def __init__(self, cfg: Discovery2Config):
        self.cfg = cfg

    def run_episode(self, fields: FieldBatch, gen: torch.Generator):
        B = fields.B
        device = fields.c1.device
        xs = torch.rand(B, self.cfg.n_probes, 2, generator=gen).to(device)
        ys = torch.stack([fields.probe(xs[:, t], gen=gen)
                          for t in range(self.cfg.n_probes)], dim=1)
        best = ys.argmax(dim=1)
        return xs.gather(1, best.view(B, 1, 1).expand(B, 1, 2)).squeeze(1), xs


class GridProbe:
    def __init__(self, cfg: Discovery2Config, jitter: float = 0.0):
        self.cfg, self.jitter = cfg, jitter

    def run_episode(self, fields: FieldBatch, gen: torch.Generator):
        B = fields.B
        device = fields.c1.device
        n = self.cfg.n_probes
        rows = int(n ** 0.5)
        cols = (n + rows - 1) // rows
        lin_r = torch.linspace(0.15, 0.85, rows)
        lin_c = torch.linspace(0.15, 0.85, cols)
        pts = torch.stack(torch.meshgrid(lin_r, lin_c, indexing="ij"),
                          dim=-1).reshape(-1, 2)[:n].to(device)
        xs = pts.unsqueeze(0).expand(B, -1, 2).clone()
        if self.jitter > 0:
            xs = (xs + self.jitter * (torch.rand(B, n, 2, generator=gen).to(device)
                                      - 0.5)).clamp(0, 1)
        ys = torch.stack([fields.probe(xs[:, t], gen=gen) for t in range(n)], dim=1)
        best = ys.argmax(dim=1)
        return xs.gather(1, best.view(B, 1, 1).expand(B, 1, 2)).squeeze(1), xs


class FDAscent:
    """Mejor-de-k aleatorias + ascenso por diferencias finitas (paso afinado)."""

    def __init__(self, cfg: Discovery2Config, k_init: int = 4, step: float = 0.1):
        self.cfg, self.k, self.step = cfg, k_init, step

    def run_episode(self, fields: FieldBatch, gen: torch.Generator):
        B = fields.B
        device = fields.c1.device
        xs_all = []
        x0 = torch.rand(B, self.k, 2, generator=gen).to(device)
        y0 = torch.stack([fields.probe(x0[:, i], gen=gen) for i in range(self.k)],
                         dim=1)
        xs_all.append(x0)
        best = y0.argmax(dim=1)
        x = x0.gather(1, best.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)
        budget = self.cfg.n_probes - self.k
        eps = 0.04
        # cada iteración FD gasta 3 sondas (x, x+eps_x, x+eps_y) y da un paso
        for _ in range(budget // 3):
            ya = fields.probe((x + torch.tensor([eps, 0.0], device=device)).clamp(0, 1), gen=gen)
            yb = fields.probe((x + torch.tensor([0.0, eps], device=device)).clamp(0, 1), gen=gen)
            yc = fields.probe(x, gen=gen)
            grad = torch.stack([(ya - yc) / eps, (yb - yc) / eps], dim=-1)
            x = (x + self.step * grad).clamp(0.0, 1.0)
            xs_all.append(x.unsqueeze(1))
        return x, torch.cat(xs_all, dim=1)


@torch.no_grad()
def eval_prober(cfg: Discovery2Config, agent, seed: int, batch: int = 512,
                device="cpu") -> dict:
    gen = torch.Generator().manual_seed(seed + 50_000)
    fields = FieldBatch(cfg, batch, gen, device=device)
    if isinstance(agent, LearnedProber):
        agent.eval()
        target, _ = agent.run_episode(fields, gen=None)
    else:
        target, _ = agent.run_episode(fields, gen)
    reg = normalized_regret(fields, target)
    per_fam = {}
    for i, name in enumerate(cfg.families):
        m = fields.family == i
        if m.any():
            per_fam[name] = float(reg[m].mean())
    return {"regret": float(reg.mean()), **{f"regret_{k}": v
                                            for k, v in per_fam.items()}}
