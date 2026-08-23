"""Fase 2 v2: entorno de COMPROMISO endógeno (rediseño post-auditoría).

El agente gestiona K proyectos con progreso que DECAE al abandonarse
(cambiar de meta cuesta), crisis reales que hay que atender e interrupciones
DISTRACTORAS calibradas para aliasear el estado instantáneo: en el tick del
pico, una crisis real y un distractor son INDISTINGUIBLES sin memoria (la
crisis real viene precedida de una rampa de 2 ticks; el distractor no).

Reglas del rediseño aplicadas:
  R4: ninguna supervisión hacia reglas programadas; el episodio se valora por
      RETORNO ambiental (valor de proyectos completados + crisis atendidas −
      crisis perdidas) y los agentes aprendidos entrenan por policy gradient.
  R7: el retorno es GRADUADO (maximización de valor, no supervivencia binaria).
La verdad de qué picos son crisis y cuáles distractores vive en el escenario
SOLO para la recompensa y las métricas; nunca es input del agente.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class Goals2Config:
    n_projects: int = 3
    horizon: int = 48
    work_rate: float = 0.06        # progreso por tick trabajado
    decay: float = 0.015           # decaimiento multiplicativo si NO se trabaja
    n_crises: int = 2
    n_distractors: int = 3
    precursor_len: int = 2         # rampa previa SOLO en crisis reales
    crisis_window: int = 4         # ticks para atender tras el pico
    crisis_bonus: float = 0.8
    crisis_penalty: float = 0.8
    urgency_noise: float = 0.05
    value_min: float = 0.5
    value_max: float = 1.5

    @property
    def n_actions(self) -> int:
        return self.n_projects + 1   # 0 = descansar; k = trabajar proyecto k

    def ood_variant(self) -> "Goals2Config":
        """OOD confirmatorio: más distractores y crisis desplazadas en el tiempo.
        La REGLA de desambiguación (rampa=real) no cambia; cambian frecuencias."""
        return replace(self, n_distractors=6, n_crises=3)


@dataclass
class Goals2Scenario:
    """Agenda exógena del episodio (verdad solo para recompensa/métricas)."""

    values: torch.Tensor          # (B,K) valor de cada proyecto (observable)
    urgency_base: torch.Tensor    # (B,T,K) ruido base de urgencia (observable vía u_t)
    spike_t: torch.Tensor         # (B,S) tick del pico (crisis+distractores)
    spike_proj: torch.Tensor      # (B,S) proyecto
    spike_is_crisis: torch.Tensor  # (B,S) bool: True=crisis real (con rampa)

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    def to(self, device) -> "Goals2Scenario":
        return Goals2Scenario(*(getattr(self, f).to(device) for f in (
            "values", "urgency_base", "spike_t", "spike_proj", "spike_is_crisis")))


class Goals2Dataset:
    def __init__(self, cfg: Goals2Config | None = None, seed: int = 0):
        self.cfg = cfg or Goals2Config()
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)

    def batch(self, batch_size: int) -> Goals2Scenario:
        c = self.cfg
        S = c.n_crises + c.n_distractors
        values = c.value_min + (c.value_max - c.value_min) * torch.rand(
            batch_size, c.n_projects, generator=self.gen)
        base = c.urgency_noise * torch.rand(
            batch_size, c.horizon, c.n_projects, generator=self.gen)
        first = c.precursor_len + 2
        last = c.horizon - c.crisis_window - 1
        spike_t = torch.randint(first, last, (batch_size, S), generator=self.gen)
        spike_proj = torch.randint(0, c.n_projects, (batch_size, S),
                                   generator=self.gen)
        is_crisis = torch.zeros(batch_size, S, dtype=torch.bool)
        is_crisis[:, :c.n_crises] = True
        # baraja el orden para que el índice no delate el tipo
        perm = torch.argsort(torch.rand(batch_size, S, generator=self.gen), dim=1)
        spike_t = torch.gather(spike_t, 1, perm)
        spike_proj = torch.gather(spike_proj, 1, perm)
        is_crisis = torch.gather(is_crisis, 1, perm)
        return Goals2Scenario(values, base, spike_t, spike_proj, is_crisis)


def urgency_signal(cfg: Goals2Config, sc: Goals2Scenario) -> torch.Tensor:
    """Construye u(t) OBSERVABLE (B,T,K): base + rampas (solo crisis) + picos
    (crisis Y distractores, misma altura 1.0 -> aliasing instantáneo)."""
    c = cfg
    B, T, K = sc.urgency_base.shape
    u = sc.urgency_base.clone()
    S = sc.spike_t.shape[1]
    for s in range(S):
        t_s = sc.spike_t[:, s]                       # (B,)
        proj = sc.spike_proj[:, s]
        crisis = sc.spike_is_crisis[:, s]
        for row in range(B):
            t0, p = int(t_s[row]), int(proj[row])
            if bool(crisis[row]):
                for j in range(c.precursor_len):     # rampa 0.3, 0.6 antes del pico
                    u[row, t0 - c.precursor_len + j, p] = max(
                        float(u[row, t0 - c.precursor_len + j, p]),
                        0.3 * (j + 1))
            hold = range(t0, min(T, t0 + c.crisis_window))
            for tt in hold:                          # pico idéntico en ambos tipos
                u[row, tt, p] = 1.0
    return u.clamp(0.0, 1.0)


def episode_return(cfg: Goals2Config, sc: Goals2Scenario,
                   actions: torch.Tensor) -> torch.Tensor:
    """Retorno ambiental (B,) para acciones (B,T) enteras (0=descanso).

    progreso: +work_rate al trabajado, ×(1-decay) al resto; completar p>=1
    consolida el valor (no decae tras completar). Crisis atendida = trabajar
    ese proyecto en [pico, pico+window); perdida = no hacerlo. Los
    distractores no dan ni quitan: atenderlos solo cuesta el tiempo.
    """
    c = cfg
    B, T = actions.shape
    K = c.n_projects
    device = actions.device
    p = torch.zeros(B, K, device=device)
    done = torch.zeros(B, K, dtype=torch.bool, device=device)
    for t in range(T):
        worked = torch.nn.functional.one_hot(
            actions[:, t].clamp_min(0), K + 1)[:, 1:].float()
        p = p + c.work_rate * worked
        p = torch.where(done | worked.bool(), p, p * (1.0 - c.decay))
        done = done | (p >= 1.0)
    ret = (sc.values * done.float()).sum(-1)
    S = sc.spike_t.shape[1]
    for s in range(S):
        t_s, proj = sc.spike_t[:, s], sc.spike_proj[:, s]
        crisis = sc.spike_is_crisis[:, s]
        attended = torch.zeros(B, dtype=torch.bool, device=device)
        for dt in range(c.crisis_window):
            tt = (t_s + dt).clamp_max(T - 1)
            attended = attended | (actions.gather(1, tt.unsqueeze(1)).squeeze(1)
                                   == proj + 1)
        ret = ret + torch.where(
            crisis.to(device),
            torch.where(attended, torch.full_like(ret, c.crisis_bonus),
                        torch.full_like(ret, -c.crisis_penalty)),
            torch.zeros_like(ret))
    return ret
