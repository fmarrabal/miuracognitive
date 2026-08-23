"""
mHBP — Interocepción: señal universal + encoder normalizado y enmascarado.

`InteroceptiveSignal`: 26 canales canónicos (§11 del plan). El encoder mantiene
estadísticas running (EMA), enmascara señales ausentes y produce el forzamiento
ACOTADO por campo: F_q = g_q · tanh(U_q s) con g_q ≤ f_max (entra en la cota
ISS de la Prop. 2 de MATH_SPEC.md — la interocepción no puede desestabilizar).

Regla anti-leakage del proyecto: en tareas donde se mida asignación no trivial
de cómputo, la longitud/dificultad literal NO debe entrar como canal (se
controla desde la tarea, no aquí; `leak_mask` permite vetar canales).
"""
from __future__ import annotations
from dataclasses import dataclass, fields as dc_fields
import torch
import torch.nn as nn

# Canales canónicos, en orden fijo (índice = posición en el tensor de señales)
CHANNELS = [
    "entropy", "confidence", "logit_margin", "state_delta", "residual",
    "progress", "memory_load", "memory_conflict", "attention_entropy",
    "novelty", "uncertainty", "risk", "elapsed_time", "token_cost",
    "energy_cost", "gpu_utilization", "gpu_memory", "tool_failures",
    "retrieval_gain", "verifier_disagreement", "agent_disagreement",
    "task_criticality", "ood_score", "queue_load", "latency", "cost_monetary",
]
N_CHANNELS = len(CHANNELS)


@dataclass
class InteroceptiveSignal:
    """Contenedor de señales de UN tick. Los campos no observados quedan a None
    (el encoder los enmascara). Valores: tensores (B,) o floats."""
    module_id: str = "global"
    tick: int = 0
    values: dict | None = None            # {canal: (B,) tensor | float}

    def to_tensor(self, batch: int, device, dtype=torch.float32):
        """(B, N_CHANNELS) señales + (B, N_CHANNELS) máscara de validez."""
        s = torch.zeros(batch, N_CHANNELS, device=device, dtype=dtype)
        m = torch.zeros(batch, N_CHANNELS, device=device, dtype=dtype)
        vals = self.values or {}
        for j, name in enumerate(CHANNELS):
            v = vals.get(name)
            if v is None:
                continue
            if not torch.is_tensor(v):
                v = torch.full((batch,), float(v), device=device, dtype=dtype)
            s[:, j] = v.to(device=device, dtype=dtype)
            m[:, j] = 1.0
        return s, m


class RunningNorm(nn.Module):
    """Normalización running por canal (EMA de media y varianza), consciente de
    la máscara: sólo actualiza con señales válidas. No aprende (buffers)."""

    def __init__(self, n: int = N_CHANNELS, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.zeros(n))

    def forward(self, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                nvalid = mask.sum(0).clamp(min=1.0)
                m_batch = (s * mask).sum(0) / nvalid
                v_batch = ((s - m_batch) ** 2 * mask).sum(0) / nvalid
                upd = (mask.sum(0) > 0).float() * self.momentum
                self.mean.mul_(1 - upd).add_(upd * m_batch)
                self.var.mul_(1 - upd).add_(upd * v_batch)
                self.count.add_(mask.sum(0))
        # suelo de varianza: con B=1 y señal ~constante la EMA colapsa a ~0 y
        # amplificaría ×1/√eps cualquier desviación (hallazgo del code-review)
        x = (s - self.mean) / torch.sqrt(self.var.clamp(min=1e-3) + self.eps)
        return x * mask                                  # ausentes -> 0 exacto


class InteroceptionEncoder(nn.Module):
    """señales (B, n_ch) -> forzamiento acotado por campo [(B, N_q, d)].
    `leak_mask`: canales vetados (anti-leakage), a 0 antes del encoder.
    `head_vetoes`: veto POR-CABEZA {índice_de_campo: (canales,)} — N2 §3:
    el canal de valor entra SOLO al campo risk («valorar sin tocar
    contenido» por construcción; test: ∂wm_gates/∂canal = 0)."""

    def __init__(self, field_cfgs, f_max: float = 0.5,
                 leak_mask: list[str] | None = None,
                 head_vetoes: dict | None = None):
        super().__init__()
        self.norm = RunningNorm()
        self.f_max = float(f_max)
        vet = torch.ones(N_CHANNELS)
        for name in (leak_mask or []):
            vet[CHANNELS.index(name)] = 0.0
        self.register_buffer("veto", vet)
        hv = torch.ones(len(field_cfgs), N_CHANNELS)
        for q, names in (head_vetoes or {}).items():
            for name in names:
                hv[q, CHANNELS.index(name)] = 0.0
        self.register_buffer("head_veto", hv)
        self.heads = nn.ModuleList([
            nn.Linear(N_CHANNELS, c.n_nodes * c.d) for c in field_cfgs])
        self._cfgs = list(field_cfgs)
        for h in self.heads:                              # arranque suave
            nn.init.normal_(h.weight, std=0.05)
            nn.init.zeros_(h.bias)

    def forward(self, sig: InteroceptiveSignal, batch: int, device,
                dtype=torch.float64) -> list[torch.Tensor]:
        # dtype de las estadísticas (FP64 si el modelo es .double()): sin
        # truncados silenciosos de señales FP64 → FP32
        s, m = sig.to_tensor(batch, device, dtype=self.norm.mean.dtype)
        x = self.norm(s, m) * self.veto
        out = []
        for q, (head, c) in enumerate(zip(self.heads, self._cfgs)):
            f = self.f_max * torch.tanh(head(x * self.head_veto[q]))
            out.append(f.view(batch, c.n_nodes, c.d).to(dtype))
        return out
