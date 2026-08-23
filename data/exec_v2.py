"""Fase 2-ESCALADA (F2X): campo ejecutivo — triaje, plazos y automantenimiento.

Escala goals_v2 (compromiso entre 3 proyectos homogéneos) al conflicto
ejecutivo completo, manteniendo las reglas AGENCY_V2 (R1-R8):

  · 6 proyectos con VALORES y PLAZOS heterogéneos: completarlos todos es
    IMPOSIBLE (capacidad de trabajo ≈ 2-3 proyectos) -> el triaje endógeno
    (qué cartera perseguir y qué abandonar deliberadamente) es obligatorio
    POR EL MUNDO, no por la pérdida.
  · ENERGÍA finita (acoplamiento con fase 1): trabajar drena; descansar
    recarga; llegar a 0 = colapso con inactividad forzada. Automantenerse
    cuesta tiempo de proyecto: el conflicto necesidades↔metas es físico.
  · Crisis reales (con precursor de 2 ticks) y distractores ALIASADOS
    (pico instantáneamente idéntico) heredados de goals_v2.

La verdad (qué picos son crisis, plazos alcanzables óptimos) vive en el
escenario SOLO para recompensa/métricas/skyline; nunca es input del agente.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class ExecV2Config:
    n_projects: int = 6
    horizon: int = 64
    work_rate: float = 0.06        # ~17 ticks de trabajo por proyecto
    decay: float = 0.012           # decaimiento si NO se trabaja (no completado)
    # --- energía (automantenimiento) ---
    energy_drain: float = 0.025    # por tick trabajado (crisis incluidas)
    energy_recharge: float = 0.10  # por tick de descanso (tope 1.0)
    collapse_ticks: int = 4        # inactividad forzada al llegar a 0
    # --- plazos ---
    deadline_frac_min: float = 0.45
    deadline_frac_max: float = 1.0
    # --- crisis / distractores (aliasing de goals_v2) ---
    n_crises: int = 2
    n_distractors: int = 3
    precursor_len: int = 2
    crisis_window: int = 4
    crisis_bonus: float = 0.8
    crisis_penalty: float = 0.8
    urgency_noise: float = 0.05
    value_min: float = 0.5
    value_max: float = 1.5

    @property
    def n_actions(self) -> int:
        return self.n_projects + 1   # 0 = descansar/recargar; k = trabajar k

    def ood_interrupts(self) -> "ExecV2Config":
        """OOD-A: estadística de interrupciones distinta (regla intacta)."""
        return replace(self, n_distractors=6, n_crises=3)

    def ood_energy(self) -> "ExecV2Config":
        """OOD-B: metabolismo más caro (drena +50%)."""
        return replace(self, energy_drain=0.0375)


@dataclass
class ExecV2Scenario:
    values: torch.Tensor          # (B,K) observable
    deadlines: torch.Tensor       # (B,K) tick límite (long, observable)
    urgency_base: torch.Tensor    # (B,T,K)
    spike_t: torch.Tensor         # (B,S) long
    spike_proj: torch.Tensor      # (B,S) long
    spike_is_crisis: torch.Tensor  # (B,S) bool (verdad: solo recompensa/métrica)

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    def to(self, device) -> "ExecV2Scenario":
        return ExecV2Scenario(*(getattr(self, f).to(device) for f in (
            "values", "deadlines", "urgency_base", "spike_t", "spike_proj",
            "spike_is_crisis")))


class ExecV2Dataset:
    def __init__(self, cfg: ExecV2Config | None = None, seed: int = 0):
        self.cfg = cfg or ExecV2Config()
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)

    def batch(self, batch_size: int) -> ExecV2Scenario:
        c = self.cfg
        K, T = c.n_projects, c.horizon
        S = c.n_crises + c.n_distractors
        values = c.value_min + (c.value_max - c.value_min) * torch.rand(
            batch_size, K, generator=self.gen)
        dl_lo = int(c.deadline_frac_min * T)
        dl_hi = int(c.deadline_frac_max * T)
        deadlines = torch.randint(dl_lo, dl_hi, (batch_size, K),
                                  generator=self.gen)
        base = c.urgency_noise * torch.rand(batch_size, T, K, generator=self.gen)
        first = c.precursor_len + 2
        last = T - c.crisis_window - 1
        spike_t = torch.randint(first, last, (batch_size, S), generator=self.gen)
        spike_proj = torch.randint(0, K, (batch_size, S), generator=self.gen)
        is_crisis = torch.zeros(batch_size, S, dtype=torch.bool)
        is_crisis[:, :c.n_crises] = True
        perm = torch.argsort(torch.rand(batch_size, S, generator=self.gen), dim=1)
        spike_t = torch.gather(spike_t, 1, perm)
        spike_proj = torch.gather(spike_proj, 1, perm)
        is_crisis = torch.gather(is_crisis, 1, perm)
        return ExecV2Scenario(values, deadlines, base, spike_t, spike_proj,
                              is_crisis)


def urgency_signal(cfg: ExecV2Config, sc: ExecV2Scenario) -> torch.Tensor:
    """u(t) observable (B,T,K): base + rampa (solo crisis) + pico idéntico.
    VECTORIZADA (el bucle B×S era el cuello de botella del entrenamiento);
    equivalencia con la versión de referencia verificada en el smoke."""
    c = cfg
    B, T, K = sc.urgency_base.shape
    u = sc.urgency_base.clone()
    S = sc.spike_t.shape[1]
    dev = u.device
    rows = torch.arange(B, device=dev).unsqueeze(1).expand(B, S)   # (B,S)

    # Picos: u[row, t0+w, proj] = 1.0 para w en [0, window). Duplicados escriben
    # el mismo valor (1.0) -> last-wins es seguro.
    for w in range(c.crisis_window):
        tt = (sc.spike_t + w).clamp_max(T - 1)                     # (B,S)
        idx = (rows.reshape(-1), tt.reshape(-1), sc.spike_proj.reshape(-1))
        u.index_put_(idx, torch.maximum(u[idx], torch.ones(B * S, device=dev)))

    # Rampa del precursor: SOLO entradas de crisis (filtrado evita que un
    # duplicado no-crisis pise la rampa con last-wins).
    cmask = sc.spike_is_crisis.reshape(-1)
    for j in range(c.precursor_len):
        tt = (sc.spike_t - c.precursor_len + j).clamp_min(0)
        idx = (rows.reshape(-1)[cmask], tt.reshape(-1)[cmask],
               sc.spike_proj.reshape(-1)[cmask])
        ramp = torch.full((int(cmask.sum()),), 0.3 * (j + 1), device=dev)
        u.index_put_(idx, torch.maximum(u[idx], ramp))
    return u.clamp(0.0, 1.0)


class ExecV2World:
    """Dinámica por ticks compartida por TODOS los agentes (única fuente de
    verdad de progreso/energía/colapso; la recompensa se acumula aquí)."""

    def __init__(self, cfg: ExecV2Config, sc: ExecV2Scenario, device):
        B, K = sc.batch_size, cfg.n_projects
        self.cfg, self.sc, self.device = cfg, sc, device
        self.p = torch.zeros(B, K, device=device)
        self.done = torch.zeros(B, K, dtype=torch.bool, device=device)
        self.done_at = torch.full((B, K), 10_000, device=device,
                                  dtype=torch.long)
        self.energy = torch.ones(B, device=device)
        self.forced_rest = torch.zeros(B, dtype=torch.long, device=device)
        self.actions = []
        self.collapses = torch.zeros(B, device=device)
        self.t = 0
        # Recompensa INCREMENTAL: el MISMO retorno ambiental, entregado en el
        # tick en que ocurre (completación a tiempo; crisis al resolverse).
        # Invariante verificado en smoke: Σ_t tick_rewards == final_return.
        self._attended = torch.zeros(B, sc.spike_t.shape[1],
                                     dtype=torch.bool, device=device)
        self.tick_rewards = []

    def step(self, a: torch.Tensor):
        """a: (B,) entera (0=descanso). Aplica colapso, energía, progreso."""
        c = self.cfg
        B, K = self.p.shape
        # colapso: fuerza descanso
        forced = self.forced_rest > 0
        a = torch.where(forced, torch.zeros_like(a), a)
        self.forced_rest = (self.forced_rest - 1).clamp_min(0)
        worked = torch.nn.functional.one_hot(a, K + 1)[:, 1:].float()
        working = worked.sum(-1)
        # energía
        self.energy = self.energy - c.energy_drain * working \
            + c.energy_recharge * (1 - working)
        self.energy = self.energy.clamp(0.0, 1.0)
        newly_collapsed = (self.energy <= 0) & ~forced
        self.forced_rest = torch.where(
            newly_collapsed,
            torch.full_like(self.forced_rest, c.collapse_ticks),
            self.forced_rest)
        self.collapses = self.collapses + newly_collapsed.float()
        # progreso (solo cuenta hacia el valor si llega ANTES del plazo)
        self.p = self.p + c.work_rate * worked
        self.p = torch.where(self.done | worked.bool(), self.p,
                             self.p * (1.0 - c.decay))
        newly_done = (~self.done) & (self.p >= 1.0)
        self.done_at = torch.where(
            newly_done, torch.full_like(self.done_at, self.t), self.done_at)
        self.done = self.done | newly_done
        self.actions.append(a)
        # --- recompensa incremental de este tick (mismo retorno, en su momento) ---
        sc, B = self.sc, self.p.shape[0]
        r = (sc.values.to(self.device)
             * (newly_done & (self.t <= sc.deadlines.to(self.device))).float()
             ).sum(-1)
        for s in range(sc.spike_t.shape[1]):
            t0 = sc.spike_t[:, s].to(self.device)
            proj = sc.spike_proj[:, s].to(self.device)
            crisis = sc.spike_is_crisis[:, s].to(self.device)
            in_win = (self.t >= t0) & (self.t < t0 + c.crisis_window)
            att_now = in_win & (a == proj + 1) & ~self._attended[:, s]
            self._attended[:, s] |= att_now
            r = r + torch.where(crisis & att_now,
                                torch.full_like(r, c.crisis_bonus),
                                torch.zeros_like(r))
            at_end = self.t == (t0 + c.crisis_window - 1)
            r = r - torch.where(crisis & at_end & ~self._attended[:, s],
                                torch.full_like(r, c.crisis_penalty),
                                torch.zeros_like(r))
        self.tick_rewards.append(r)
        self.t += 1

    def final_return(self) -> torch.Tensor:
        c = self.cfg
        sc = self.sc
        B = self.p.shape[0]
        on_time = self.done & (self.done_at <= sc.deadlines.to(self.device))
        ret = (sc.values.to(self.device) * on_time.float()).sum(-1)
        acts = torch.stack(self.actions, dim=1)          # (B,T)
        for s in range(sc.spike_t.shape[1]):
            t_s = sc.spike_t[:, s].to(self.device)
            proj = sc.spike_proj[:, s].to(self.device)
            crisis = sc.spike_is_crisis[:, s].to(self.device)
            attended = torch.zeros(B, dtype=torch.bool, device=self.device)
            for dt in range(c.crisis_window):
                tt = (t_s + dt).clamp_max(c.horizon - 1)
                attended |= acts.gather(1, tt.unsqueeze(1)).squeeze(1) == proj + 1
            ret = ret + torch.where(
                crisis,
                torch.where(attended, torch.full_like(ret, c.crisis_bonus),
                            torch.full_like(ret, -c.crisis_penalty)),
                torch.zeros_like(ret))
        return ret


@torch.no_grad()
def portfolio_skyline(cfg: ExecV2Config, sc: ExecV2Scenario) -> torch.Tensor:
    """Cota superior GREEDY de cartera con la verdad (solo métrica): capacidad
    de trabajo efectiva ≈ horizon·recarga/(drena+recarga); llena por densidad
    de valor respetando plazos. NO es un óptimo exacto (es cota heurística
    superior razonable para normalizar el regret)."""
    c = cfg
    B, K = sc.values.shape
    ticks_per_project = int(1.0 / c.work_rate) + 1
    duty = c.energy_recharge / (c.energy_drain + c.energy_recharge)
    capacity = int(c.horizon * duty)
    sky = torch.zeros(B)
    order = torch.argsort(sc.values, dim=1, descending=True)
    for row in range(B):
        used = 0
        for k in order[row].tolist():
            need = ticks_per_project
            if used + need <= min(capacity, int(sc.deadlines[row, k])):
                used += need
                sky[row] += float(sc.values[row, k])
    return sky + c.n_crises * c.crisis_bonus
