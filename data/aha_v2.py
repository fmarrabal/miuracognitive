"""AHA-2: entorno de regulación anticipatoria SIN oráculos (rediseño post-auditoría).

Diferencias vinculantes frente a data/aha.py (v1), según AGENCY_V2.md:
  R2: `action_delay` es un parámetro BARRIDO (0..3). Con delay 0-1 un reactivo
      hábil sobrevive; la ventaja anticipatoria debe crecer con el retardo
      (dose-response), no estar garantizada por aritmética.
  R3: los cues son RUIDOSOS Y PARCIALES: fiabilidad<1 (eventos sin cue),
      adelanto con jitter U{lead_min..lead_max}, magnitud ×U(0.7,1.3), canal
      equivocado con prob. channel_noise, y FALSAS ALARMAS sin evento.
      Anticipar exige inferencia estadística, no copiar un oráculo.
  El coste de actuar es AMBIENTAL (drena todas las necesidades): la estrategia
  "actuar siempre" la castiga el mundo, no una pérdida moldeada (R4 de v1).

El escenario conserva la verdad de los eventos (events_*) SOLO para métricas
(adelanto de anticipación, falsas acciones); jamás como input de ningún agente.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class AHA2Config:
    n_needs: int = 3
    horizon: int = 56
    n_events: int = 5
    storm_len: int = 3             # cada evento drena su magnitud repartida en
                                   # storm_len ticks: con delay bajo se puede
                                   # reaccionar A MITAD de tormenta (R2); con
                                   # delay alto solo la anticipación llega a tiempo
    action_delay: int = 2          # BARRIDO en el benchmark: {0,1,2,3}
    action_effect: float = 0.22
    action_cost: float = 0.02      # metabólico: drena TODAS las necesidades al actuar.
                                   # Junto con el tope level_cap=1.0 hace que la
                                   # estrategia degenerada "actuar siempre" se
                                   # DESANGRE (restauros clampados a nada + coste
                                   # acumulado): el timing debe aprenderse. Fix
                                   # AMBIENTAL, no de pérdida (R4).
    level_cap: float = 1.0         # una necesidad no puede estar más que llena
    action_budget: int = 14        # BATERÍA física: nº máximo de restauros por
                                   # episodio (con la batería vacía, actuar no
                                   # produce efecto). 5 tormentas ≈ 5-8 acciones
                                   # bien puestas -> "actuar siempre" es
                                   # físicamente imposible; la selectividad es
                                   # obligatoria por el MUNDO, no por la pérdida.
    basal_drain: float = 0.0015
    hazard_min: float = 0.30      # tormenta SIN defender mata (0.78-0.36<0.45);
    hazard_max: float = 0.42      # defendida a tope sobrevive (1.0-0.42=0.58).
                                   # Con batería 14 < ~18 cargas que costaría
                                   # mantener las 3 necesidades siempre llenas,
                                   # la profilaxis indiscriminada está
                                   # INFRAFINANCIADA: saber QUÉ canal viene
                                   # (cue, 85% fiable) es la única estrategia
                                   # asequible -> el valor de la señal es
                                   # económico, no impuesto por la pérdida.
    viability_threshold: float = 0.45
    initial_min: float = 0.75
    initial_max: float = 0.88
    setpoint_min: float = 0.70
    setpoint_max: float = 0.80
    # --- Precursores imperfectos (R3) ---
    cue_reliability: float = 0.75  # prob. de que un evento emita cue
    cue_lead_min: int = 4
    cue_lead_max: int = 9
    cue_mag_noise: float = 0.30    # magnitud del cue ×U(1-x, 1+x)
    cue_channel_noise: float = 0.15  # prob. de canal equivocado
    false_alarm_prob: float = 0.35   # prob. de UNA falsa alarma por (fila, necesidad)

    @property
    def n_actions(self) -> int:
        return self.n_needs + 1    # 0 = no-op; 1..N = restaurar necesidad

    def with_delay(self, delay: int) -> "AHA2Config":
        return replace(self, action_delay=delay)

    def validate(self) -> None:
        if self.n_needs < 2:
            raise ValueError("AHA-2 requiere al menos dos necesidades")
        if not (0 <= self.action_delay <= self.cue_lead_min - 1):
            raise ValueError("se requiere 0 <= action_delay < cue_lead_min")
        if self.cue_lead_min < 1 or self.cue_lead_max < self.cue_lead_min:
            raise ValueError("rango de adelanto de cue inválido")
        if self.horizon < 2 * self.cue_lead_max + 6:
            raise ValueError("horizon demasiado corto")
        if not (0.0 <= self.cue_reliability <= 1.0):
            raise ValueError("cue_reliability en [0,1]")


@dataclass
class AHA2Scenario:
    """Batch de mundos. events_* son VERDAD para métricas, nunca input."""

    initial_levels: torch.Tensor   # (B,N)
    setpoints: torch.Tensor        # (B,N)
    cues: torch.Tensor             # (B,T,N) señal imperfecta observable
    disturbances: torch.Tensor     # (B,T,N) pérdida exógena real
    events_t: torch.Tensor         # (B,E) tick del evento (long)
    events_need: torch.Tensor      # (B,E) necesidad golpeada (long)

    @property
    def batch_size(self) -> int:
        return int(self.initial_levels.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.cues.shape[1])

    def to(self, device) -> "AHA2Scenario":
        return AHA2Scenario(*(getattr(self, f).to(device) for f in (
            "initial_levels", "setpoints", "cues", "disturbances",
            "events_t", "events_need")))

    def validate(self, cfg: AHA2Config) -> None:
        b, t, n = self.cues.shape
        assert (b, n) == tuple(self.initial_levels.shape)
        assert t == cfg.horizon and n == cfg.n_needs
        for x in (self.initial_levels, self.setpoints, self.cues, self.disturbances):
            if not torch.isfinite(x).all():
                raise ValueError("escenario con NaN/Inf")
        if (self.cues < 0).any() or (self.disturbances < 0).any():
            raise ValueError("cues/disturbances deben ser no negativos")


class AHA2Dataset:
    """Generador reproducible. Eventos estratificados en ventanas; tipos de
    necesidad permutados (amenazas competidoras sin orden fijo)."""

    def __init__(self, cfg: AHA2Config | None = None, seed: int = 0):
        self.cfg = cfg or AHA2Config()
        self.cfg.validate()
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen)

    def batch(self, batch_size: int) -> AHA2Scenario:
        c = self.cfg
        initial = c.initial_min + (c.initial_max - c.initial_min) * self._rand(batch_size, c.n_needs)
        setp = c.setpoint_min + (c.setpoint_max - c.setpoint_min) * self._rand(batch_size, c.n_needs)
        cues = torch.zeros(batch_size, c.horizon, c.n_needs)
        dist = torch.zeros_like(cues)
        ev_t = torch.zeros(batch_size, c.n_events, dtype=torch.long)
        ev_need = torch.zeros(batch_size, c.n_events, dtype=torch.long)

        first = c.cue_lead_max + 1
        last = c.horizon - c.storm_len - 1
        edges = torch.linspace(first, last + 1, c.n_events + 1)
        for row in range(batch_size):
            types = torch.randperm(c.n_needs, generator=self.gen).tolist()
            while len(types) < c.n_events:
                types.extend(torch.randperm(c.n_needs, generator=self.gen).tolist())
            for k in range(c.n_events):
                lo = int(edges[k].item())
                hi = max(lo + 1, int(edges[k + 1].item()))
                t_e = int(torch.randint(lo, min(hi, last + 1), (1,), generator=self.gen).item())
                need = types[k]
                mag = float(c.hazard_min + (c.hazard_max - c.hazard_min)
                            * self._rand(()).item())
                # Tormenta: la magnitud se reparte en storm_len ticks (R2)
                dist[row, t_e:t_e + c.storm_len, need] += mag / c.storm_len
                ev_t[row, k], ev_need[row, k] = t_e, need
                # --- Precursor IMPERFECTO (R3) ---
                if self._rand(()).item() < c.cue_reliability:
                    lead = int(torch.randint(c.cue_lead_min, c.cue_lead_max + 1,
                                             (1,), generator=self.gen).item())
                    t_c = max(0, t_e - lead)
                    noisy = mag * (1.0 - c.cue_mag_noise
                                   + 2.0 * c.cue_mag_noise * self._rand(()).item())
                    ch = need
                    if self._rand(()).item() < c.cue_channel_noise:
                        ch = int(torch.randint(0, c.n_needs, (1,), generator=self.gen).item())
                    cues[row, t_c, ch] += noisy
            # --- Falsas alarmas: cue sin evento (R3) ---
            for need in range(c.n_needs):
                if self._rand(()).item() < c.false_alarm_prob:
                    t_f = int(torch.randint(first, last, (1,), generator=self.gen).item())
                    fake = c.hazard_min + (c.hazard_max - c.hazard_min) * self._rand(()).item()
                    cues[row, t_f, need] += fake

        sc = AHA2Scenario(initial, setp, cues, dist, ev_t, ev_need)
        sc.validate(c)
        return sc


def rollout_world(cfg: AHA2Config, scenario: AHA2Scenario, policy,
                  mode: str = "greedy"):
    """Simulador ÚNICO para todos los agentes (aprendidos y scripted).

    policy(obs, t) -> (B, n_actions) probabilidades. Modos:
      greedy : one-hot argmax (evaluación y scripted).
      sample : muestreo categórico + logp/entropía (REINFORCE; las acciones
               ejecutadas son duras -> misma aritmética del mundo que en eval).
      st     : straight-through Gumbel (legado; sesgo train/eval documentado
               en el piloto: no usar para entrenar).
    obs = {levels, error, velocity, cue, prev_action}; nunca contiene eventos.

    Devuelve dict con levels_hist (B,T+1,N), actions (B,T,A), violated (B,),
    min_level (B,) y, en sample, logp (B,) y entropy (escalar). Orden por
    tick: observar -> actuar (coste metabólico inmediato, efecto encolado
    delay ticks) -> llegadas -> drenaje basal + perturbación -> clamp.
    """
    c = cfg
    B, T, N = scenario.cues.shape
    device = scenario.initial_levels.device
    levels = scenario.initial_levels.clone()
    prev_a = torch.zeros(B, c.n_actions, device=device)
    prev_levels = levels.clone()
    queue = torch.zeros(B, max(c.action_delay, 1), N, device=device)
    store = torch.full((B,), float(c.action_budget), device=device)  # batería
    levels_hist = [levels.clone()]
    actions_hist = []
    violated = torch.zeros(B, dtype=torch.bool, device=device)
    min_level = levels.min(dim=-1).values

    logps, ents = [], []
    for t in range(T):
        obs = {
            "levels": levels,
            "error": scenario.setpoints - levels,
            "velocity": levels - prev_levels,
            "cue": scenario.cues[:, t],
            "prev_action": prev_a,
            # interocepción de la batería (el agente SIENTE su energía restante)
            "battery": (store / max(c.action_budget, 1)).unsqueeze(-1),
        }
        a = policy(obs, t)                                   # (B, A) probs
        if mode == "sample":
            dist = torch.distributions.Categorical(probs=a.clamp_min(1e-9))
            idx = dist.sample()
            logps.append(dist.log_prob(idx))
            ents.append(dist.entropy())
            a = torch.nn.functional.one_hot(idx, c.n_actions).float()
        elif mode == "st":
            gumbel = -torch.log(-torch.log(
                torch.rand_like(a).clamp_min(1e-9)).clamp_min(1e-9))
            idx = (a.clamp_min(1e-9).log() + gumbel).argmax(dim=-1)
            hard = torch.nn.functional.one_hot(idx, c.n_actions).float()
            a = hard + a - a.detach()
        else:
            a = torch.nn.functional.one_hot(a.argmax(dim=-1), c.n_actions).float()
        act_mass = 1.0 - a[:, 0]                             # prob/indicador de actuar
        has_charge = (store > 0).float()                     # batería disponible
        restore = a[:, 1:] * c.action_effect * has_charge.unsqueeze(-1)
        store = store - act_mass.detach() * has_charge       # gasta carga al actuar
        prev_levels = levels
        # coste metabólico ambiental: actuar drena TODAS las necesidades
        levels = levels - c.action_cost * act_mass.unsqueeze(-1)
        if c.action_delay == 0:
            arriving = restore
        else:
            arriving = queue[:, 0]
            queue = torch.cat([queue[:, 1:], torch.zeros_like(queue[:, :1])], dim=1)
            queue = queue.clone()
            queue[:, -1] = queue[:, -1] + restore
        levels = levels + arriving - c.basal_drain - scenario.disturbances[:, t]
        levels = levels.clamp(0.0, c.level_cap)
        violated = violated | (levels.min(dim=-1).values < c.viability_threshold)
        min_level = torch.minimum(min_level, levels.min(dim=-1).values)
        prev_a = a
        levels_hist.append(levels.clone())
        actions_hist.append(a)

    out = {
        "levels_hist": torch.stack(levels_hist, dim=1),      # (B,T+1,N)
        "actions": torch.stack(actions_hist, dim=1),         # (B,T,A)
        "violated": violated,
        "min_level": min_level,
    }
    if mode == "sample":
        out["logp"] = torch.stack(logps, dim=1).sum(dim=1)   # (B,)
        out["entropy"] = torch.stack(ents, dim=1).mean()
    return out
