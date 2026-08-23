"""Entorno sintético para Anticipatory Homeostatic Agency (AHA).

El entorno no entrega una tarea ni un ``goal_id``. Cada trayectoria contiene
variables internas continuas, perturbaciones exógenas y señales predictivas
puntuales. Las acciones restauradoras tienen latencia: esperar a observar el
déficit es demasiado tarde para evitar una salida de la región viable.

La separación entre ``cues`` y ``disturbances`` es deliberada. Un predictor
causal sólo observa el prefijo de señales; los eventos futuros se usan como
objetivos auto-supervisados durante entrenamiento, nunca como input en test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AHAEnvironmentConfig:
    """Dinámica del cuerpo mínimo simulado por AHA."""

    n_needs: int = 3
    horizon: int = 48
    n_events: int = 5
    cue_lead: int = 6
    action_delay: int = 3
    prediction_horizon: int = 4
    action_effect: float = 0.20
    action_cost: float = 0.040
    basal_drain: float = 0.0015
    hazard_min: float = 0.30
    hazard_max: float = 0.38
    viability_threshold: float = 0.45
    initial_min: float = 0.68
    initial_max: float = 0.82
    setpoint_min: float = 0.70
    setpoint_max: float = 0.80

    @property
    def n_actions(self) -> int:
        # 0 = no-op; 1..n_needs = restaurar una necesidad.
        return self.n_needs + 1

    def validate(self) -> None:
        if self.n_needs < 2:
            raise ValueError("AHA requiere al menos dos necesidades")
        if self.horizon < 2 * self.cue_lead + 4:
            raise ValueError("horizon demasiado corto para anticipación")
        if not (0 < self.action_delay < self.cue_lead):
            raise ValueError("se requiere 0 < action_delay < cue_lead")
        if self.prediction_horizon < self.action_delay:
            raise ValueError("prediction_horizon debe cubrir action_delay")
        if self.n_events < 1:
            raise ValueError("n_events debe ser positivo")
        if not (0 < self.viability_threshold < 1):
            raise ValueError("viability_threshold debe estar en (0,1)")


@dataclass
class AHAScenario:
    """Batch pareado de mundos exógenos y estados internos iniciales."""

    initial_levels: torch.Tensor       # (B,N), 1 = necesidad satisfecha
    setpoints: torch.Tensor            # (B,N), nivel preferido
    cues: torch.Tensor                 # (B,T,N), señal anterior al evento
    disturbances: torch.Tensor         # (B,T,N), pérdida exógena real

    @property
    def batch_size(self) -> int:
        return int(self.initial_levels.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.cues.shape[1])

    @property
    def n_needs(self) -> int:
        return int(self.initial_levels.shape[1])

    def to(self, device: torch.device | str) -> "AHAScenario":
        return AHAScenario(
            initial_levels=self.initial_levels.to(device),
            setpoints=self.setpoints.to(device),
            cues=self.cues.to(device),
            disturbances=self.disturbances.to(device),
        )

    def clone(self) -> "AHAScenario":
        return AHAScenario(
            initial_levels=self.initial_levels.clone(),
            setpoints=self.setpoints.clone(),
            cues=self.cues.clone(),
            disturbances=self.disturbances.clone(),
        )

    def validate(self, cfg: AHAEnvironmentConfig) -> None:
        shape_bn = (self.batch_size, cfg.n_needs)
        shape_btn = (self.batch_size, cfg.horizon, cfg.n_needs)
        if tuple(self.initial_levels.shape) != shape_bn:
            raise ValueError("forma incorrecta de initial_levels")
        if tuple(self.setpoints.shape) != shape_bn:
            raise ValueError("forma incorrecta de setpoints")
        if tuple(self.cues.shape) != shape_btn:
            raise ValueError("forma incorrecta de cues")
        if tuple(self.disturbances.shape) != shape_btn:
            raise ValueError("forma incorrecta de disturbances")
        tensors = (self.initial_levels, self.setpoints,
                   self.cues, self.disturbances)
        if not all(torch.isfinite(x).all() for x in tensors):
            raise ValueError("el escenario contiene NaN/Inf")
        if (self.cues < 0).any() or (self.disturbances < 0).any():
            raise ValueError("cues/disturbances deben ser no negativos")


class AnticipatoryHomeostasisDataset:
    """Generador reproducible de trayectorias AHA.

    Los eventos se muestrean dentro de ventanas estratificadas para cubrir toda
    la trayectoria. El tipo de necesidad se permuta y después se completa al
    azar; así cada batch contiene amenazas competidoras sin codificar un orden
    fijo energía→temperatura→integridad.
    """

    def __init__(self, cfg: AHAEnvironmentConfig | None = None,
                 seed: int = 0):
        self.cfg = cfg or AHAEnvironmentConfig()
        self.cfg.validate()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def batch(self, batch_size: int, hazards: bool = True) -> AHAScenario:
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        c = self.cfg
        initial = (c.initial_min + (c.initial_max - c.initial_min)
                   * torch.rand(batch_size, c.n_needs,
                                generator=self.generator))
        setpoints = (c.setpoint_min + (c.setpoint_max - c.setpoint_min)
                     * torch.rand(batch_size, c.n_needs,
                                  generator=self.generator))
        cues = torch.zeros(batch_size, c.horizon, c.n_needs)
        disturbances = torch.zeros_like(cues)

        if hazards:
            first = c.cue_lead + 2
            last = c.horizon - 2
            edges = torch.linspace(first, last + 1, c.n_events + 1)
            for row in range(batch_size):
                base_types = torch.randperm(
                    c.n_needs, generator=self.generator).tolist()
                while len(base_types) < c.n_events:
                    base_types.extend(torch.randperm(
                        c.n_needs, generator=self.generator).tolist())
                for event_idx in range(c.n_events):
                    lo = int(edges[event_idx].item())
                    hi = max(lo + 1, int(edges[event_idx + 1].item()))
                    event_t = int(torch.randint(
                        lo, min(hi, last + 1), (1,),
                        generator=self.generator).item())
                    need = base_types[event_idx]
                    magnitude = float(
                        c.hazard_min + (c.hazard_max - c.hazard_min)
                        * torch.rand((), generator=self.generator).item())
                    disturbances[row, event_t, need] += magnitude
                    cues[row, event_t - c.cue_lead, need] += magnitude

        scenario = AHAScenario(initial, setpoints, cues, disturbances)
        scenario.validate(c)
        return scenario


def future_disturbance_targets(
        disturbances: torch.Tensor, prediction_horizon: int) -> torch.Tensor:
    """Perturbación acumulada causalmente relevante en los próximos H ticks.

    Para la decisión tomada en ``t`` se suman los eventos ``t+1..t+H``. El
    target sólo se usa para entrenar/calibrar el predictor; durante un rollout
    la política recibe exclusivamente su predicción desde el prefijo observado.
    """

    if disturbances.ndim != 3:
        raise ValueError("disturbances debe tener forma (B,T,N)")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon debe ser positivo")
    _, horizon, _ = disturbances.shape
    target = torch.zeros_like(disturbances)
    for tick in range(horizon):
        stop = min(horizon, tick + prediction_horizon + 1)
        if tick + 1 < stop:
            target[:, tick] = disturbances[:, tick + 1:stop].sum(dim=1)
    return target
