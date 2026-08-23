"""
mHBP F3b — Controladores de sesión con PARIDAD total (PREREG_F3B v2 §3-§4).

Tres controladores que compiten con el plano mHBP bajo EXACTAMENTE las mismas
observaciones (la cola §3 de s_raw + completitud + drive del input) y las
mismas FORMAS de actuador (sesgo de halting tanh ±1; presión de presupuesto
softplus·señal·(1+γ·tanh(V·estado)); gates de WM sigmoides; block_gate tanh
acotado). Lo ÚNICO que cambia entre brazos es la fuente del estado:

  · hbp_sess  → campo único de 2º orden (HomeostaticBackgroundProcessor):
                el incumbente, con su física Verlet intacta, PERSISTENTE.
  · gru_sess  → GRUCell sobre las señales normalizadas: ¿basta cualquier
                estado recurrente? (el contraste que F2b exigía y que el
                panel repuso como crítico #1 de baselines).
  · react_sess→ sin estado: features del tick → actuadores. El suelo de pura
                observabilidad (cuánto pacing se logra solo mirando).

Los tres duck-typean el contrato de miura en session_mode: step_session,
on_instance_kick, reset_state, modulation, pérdidas, pin_fp32, swap_halves.
Los actuadores duplican las ECUACIONES del MhbpReasonerAdapter (no la clase:
el adaptador mhbp conserva sus nombres de parámetros por compatibilidad con
los checkpoints F3a); el test de paridad de f3b_session verifica las formas.

Convención de s_raw (session_mode): (B, N, 11) — cols 0-2 señales por nodo;
cola: [3]=esfuerzo [4]=frac_valid [5]=gastado [6]=restante [7]=stake
[8]=inst_left [9]=margen decode [10]=entropía decode.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..interoception import InteroceptiveSignal, RunningNorm, N_CHANNELS, CHANNELS
from model.hbp import HBPConfig, HomeostaticBackgroundProcessor


# ------------------------------------------------------------------------- #
#  Cabezas de actuación compartidas (las MISMAS formas que el adaptador mhbp)
# ------------------------------------------------------------------------- #
class SessionActuatorHeads(nn.Module):
    """z_halt (B,dz) → sesgo halting; z_res (B,dz) → ganancia de presión;
    z_wm (B,dz) → gates WM; z_gate (B,dz) → block_gate. Init ≈ neutra
    (idéntica disciplina F2b: el brazo arranca comportándose como gating)."""

    def __init__(self, dz: int, d_h: int,
                 gate_gain: float = 0.3, res_gain: float = 0.8):
        super().__init__()
        self.gate_gain, self.res_gain = gate_gain, res_gain
        self.V_halt = nn.Linear(dz, 1)
        self.V_res = nn.Linear(dz, 1)
        self.raw_pressure = nn.Parameter(torch.tensor(-5.0))   # neutro (F3a)
        self.V_wm_w = nn.Linear(dz, d_h)
        self.V_wm_f = nn.Linear(dz, d_h)
        self.b_wm_w = nn.Parameter(torch.zeros(1))
        self.b_wm_f = nn.Parameter(torch.zeros(1))
        self.V_gate = nn.Linear(dz, d_h)
        for m in (self.V_halt, self.V_res, self.V_wm_w, self.V_wm_f, self.V_gate):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def halt_and_pressure(self, z: torch.Tensor, effort: torch.Tensor,
                          spent: torch.Tensor) -> torch.Tensor:
        wdt = self.V_halt.weight.dtype
        halt_mod = torch.tanh(self.V_halt(z.to(wdt))).squeeze(-1)
        gain_res = 1.0 + self.res_gain * torch.tanh(
            self.V_res(z.to(wdt))).squeeze(-1)
        press_sig = 0.5 * (effort.to(wdt) + spent.to(wdt))
        pressure = torch.nn.functional.softplus(self.raw_pressure) \
            * press_sig * gain_res
        return (halt_mod + pressure).clamp(-1.0, 1.0)

    def content_gates(self, z: torch.Tensor):
        wdt = self.V_wm_w.weight.dtype
        wm_w = torch.sigmoid(self.b_wm_w + self.V_wm_w(z.to(wdt)))
        wm_f = torch.sigmoid(self.b_wm_f + self.V_wm_f(z.to(wdt)))
        gate = self.gate_gain * torch.tanh(self.V_gate(z.to(wdt)))
        return wm_w, wm_f, gate


class _CfgShim:
    def __init__(self, n_nodes, d_h, d_intero):
        self.n_nodes, self.d_h, self.d_intero = n_nodes, d_h, d_intero


def _sig_from_sraw(s_raw, effort, dmargin, dentropy, hit_cap, dch, L):
    """Mapa de canales IDÉNTICO al del adaptador mhbp (paridad §3)."""
    spent = s_raw[:, 0, 5].float()
    return InteroceptiveSignal(values={
        "attention_entropy": s_raw[:, :L, 2].float().mean(1),
        "entropy": s_raw[:, L, 2].float(),
        "state_delta": s_raw[:, L, 0].float(),
        "novelty": s_raw[:, L, 1].float(),
        "memory_load": s_raw[:, L + 1, 0].float(),
        "elapsed_time": effort.float(),
        "ood_score": s_raw[:, 0, 4].float(),
        "confidence": dmargin.float(),
        "uncertainty": dentropy.float(),
        "risk": dentropy.float(),
        "energy_cost": spent,
        "latency": s_raw[:, 0, 6].float(),
        "cost_monetary": s_raw[:, 0, 7].float(),
        "task_criticality": s_raw[:, 0, 7].float(),
        "queue_load": s_raw[:, 0, 8].float(),
        "gpu_utilization": dch[:, 0].float(),
        "gpu_memory": dch[:, 1].float(),
        "token_cost": hit_cap.float(),
    }), spent


class _SessionControllerBase(nn.Module):
    """Contrato común: drive del input, mod-dict completo, fusión active_idx,
    frontera de instancia (pre-bucle neutro, drive sobrescrito)."""

    def __init__(self, n_nodes: int, d_h: int, d_intero: int, dz: int,
                 gate_gain: float = 0.3, res_gain: float = 0.8):
        super().__init__()
        self.cfg = _CfgShim(n_nodes, d_h, d_intero)
        self.heads = SessionActuatorHeads(dz, d_h, gate_gain, res_gain)
        self.drive_proj = nn.Linear(d_h, 2)
        nn.init.normal_(self.drive_proj.weight, std=0.02)
        nn.init.zeros_(self.drive_proj.bias)
        self.register_buffer("h_t", torch.zeros(1, n_nodes, d_h),
                             persistent=False)
        self._B = None
        self._last_mod = None
        self.lesion = set()

    # --- estado propio del núcleo: implementan las subclases --- #
    def _core_reset(self, batch, device):
        raise NotImplementedError

    def _core_step(self, sig: InteroceptiveSignal, B_tick, device, active_idx):
        """Avanza el núcleo (solo filas activas) y devuelve z (B_tick, dz)."""
        raise NotImplementedError

    def _core_swap(self, half: int):
        pass

    # ------------------------------------------------------------------ #
    def reset_state(self, batch: int = 1, device=None, dtype=None):
        device = device or self.heads.V_halt.weight.device
        self.h_t = torch.zeros(batch, self.cfg.n_nodes, self.cfg.d_h,
                               device=device,
                               dtype=dtype or self.heads.V_halt.weight.dtype)
        self._B = batch
        self._last_mod = None
        self._core_reset(batch, device)

    def on_instance_kick(self, kick: torch.Tensor):
        """Frontera de instancia (§6): drive sobrescrito, pre-bucle neutro.
        El estado del NÚCLEO persiste (lo resetea el runner por sesión)."""
        self.h_t = kick.to(self.h_t.dtype)
        self._last_mod = None

    def step_session(self, s_raw, ext, effort, decode_margin, decode_entropy,
                     hit_cap, active_idx=None) -> dict:
        B_tick = s_raw.shape[0]
        L = self.cfg.n_nodes - 2
        h_drive = (self.h_t if active_idx is None
                   else self.h_t.index_select(0, active_idx))
        drive = (h_drive.mean(dim=1) + ext).float()
        dch = torch.tanh(self.drive_proj(drive.to(self.drive_proj.weight.dtype)))
        sig, spent = _sig_from_sraw(s_raw, effort, decode_margin,
                                    decode_entropy, hit_cap, dch, L)
        z = self._core_step(sig, B_tick, s_raw.device, active_idx)

        bias_total = self.heads.halt_and_pressure(z, effort, spent)
        if "halt" in self.lesion:
            bias_total = torch.zeros_like(bias_total)
        halt_th = 0.5 + 0.5 * bias_total
        wm_w, wm_f, gate = self.heads.content_gates(z)
        if "wm" in self.lesion:
            wm_w = torch.full_like(wm_w, 0.5)
            wm_f = torch.full_like(wm_f, 0.5)
        if "gate" in self.lesion:
            gate = torch.zeros_like(gate)

        N = self.cfg.n_nodes
        wdt = halt_th.dtype
        mod_tick = {
            "halt_threshold": halt_th.unsqueeze(1).expand(B_tick, N),
            "wm_write": wm_w.unsqueeze(1).expand(B_tick, N, -1),
            "wm_forget": wm_f.unsqueeze(1).expand(B_tick, N, -1),
            "router_bias": torch.zeros(B_tick, N, device=s_raw.device, dtype=wdt),
            "block_gate": gate.unsqueeze(1).expand(B_tick, N, -1),
        }
        if active_idx is None:
            self._last_mod = mod_tick
            return mod_tick
        if self._last_mod is None:
            self._last_mod = {k: v for k, v in self.modulation().items()}
        merged = {k: self._last_mod[k].index_copy(
            0, active_idx, v.to(self._last_mod[k].dtype))
            for k, v in mod_tick.items()}
        self._last_mod = merged
        return merged

    def modulation(self) -> dict:
        if self._last_mod is not None:
            return self._last_mod
        B = self._B or 1
        N, d_h = self.cfg.n_nodes, self.cfg.d_h
        dev, dt = self.h_t.device, self.h_t.dtype
        return {
            "halt_threshold": torch.full((B, N), 0.5, device=dev, dtype=dt),
            "wm_write": torch.full((B, N, d_h), 0.5, device=dev, dtype=dt),
            "wm_forget": torch.full((B, N, d_h), 0.5, device=dev, dtype=dt),
            "router_bias": torch.zeros(B, N, device=dev, dtype=dt),
            "block_gate": torch.zeros(B, N, d_h, device=dev, dtype=dt),
        }

    # ---- contrato de pérdidas/diagnóstico (neutro salvo subclase) ---- #
    def interoception_loss(self, s_target) -> torch.Tensor:
        return torch.zeros((), device=self.h_t.device)

    def homeostatic_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.h_t.device, dtype=self.h_t.dtype)

    def stability_penalty(self) -> torch.Tensor:
        return torch.zeros((), device=self.h_t.device)

    def state_summary(self) -> dict:
        return {"deviation_from_rest": 0.0, "vei_variance": 0.0}

    def pin_fp32(self):
        return self

    @torch.no_grad()
    def swap_halves(self, half: int):
        self.h_t = torch.cat([self.h_t[half:], self.h_t[:half]], 0)
        self._core_swap(half)


# ------------------------------------------------------------------------- #
#  hbp_sess — el incumbente (campo único 2º orden) con paridad de interfaz
# ------------------------------------------------------------------------- #
class SingleFieldSessionAdapter(_SessionControllerBase):
    """El campo único de MiuraCognitive (Verlet, 2º orden) como controlador
    de sesión: su física es EXACTAMENTE la del incumbente F3a (hbp.py); las
    cabezas de modulación nativas quedan sin uso — la actuación pasa por las
    formas compartidas (paridad §3). Su VEI h_t PERSISTE entre instancias
    (h_t ES el campo: el kick se SUMA, no sobrescribe — §6)."""

    def __init__(self, n_nodes: int = 6, d_h: int = 64, d_intero: int = 4,
                 **kw):
        super().__init__(n_nodes, d_h, d_intero, dz=d_h, **kw)
        # HBPConfig deriva d_h = d_f + d_e + d_u; se reparte manteniendo la
        # proporción del incumbente (16/16/32 para d_h=64)
        d_f = d_e = d_h // 4
        hc = HBPConfig(n_nodes=n_nodes, d_f=d_f, d_e=d_e,
                       d_u=d_h - 2 * d_f, d_intero=d_intero)
        self.core = HomeostaticBackgroundProcessor(hc)
        self.core.init_physics()
        # proyección propia de la cola extendida (11 → d_intero); miura ya no
        # proyecta en session_mode
        self.intero_proj = nn.Linear(11, d_intero)
        nn.init.normal_(self.intero_proj.weight, std=0.02)
        nn.init.zeros_(self.intero_proj.bias)

    def on_instance_kick(self, kick: torch.Tensor):
        # h_t del NÚCLEO es el campo persistente; el drive-buffer del
        # contrato (self.h_t) refleja el estado del campo para el pooling.
        self.core.h_t = self.core.h_t + kick.to(self.core.h_t.dtype)
        self.h_t = self.core.h_t
        self._last_mod = None

    def _core_reset(self, batch, device):
        self.core.reset_state(batch, device=device, dtype=self.h_t.dtype)
        self.h_t = self.core.h_t

    def _core_step(self, sig, B_tick, device, active_idx):
        # la física del incumbente consume s_n proyectado por nodo — se
        # reconstruye (B, N, 11) → (B, N, d_intero) con la proyección propia
        s_raw = self._s_raw_cache
        s_n = self.intero_proj(s_raw.to(self.intero_proj.weight.dtype))
        ext = self._ext_cache.unsqueeze(1).expand(B_tick, self.cfg.n_nodes, -1)
        self.core.step(s_n, ext_force=ext, active_idx=active_idx)
        self.h_t = self.core.h_t
        z = (self.core.h_t if active_idx is None
             else self.core.h_t.index_select(0, active_idx)).mean(dim=1)
        return z.float()

    def step_session(self, s_raw, ext, effort, decode_margin, decode_entropy,
                     hit_cap, active_idx=None) -> dict:
        # cachés para _core_step (la base no pasa s_raw/ext al núcleo)
        self._s_raw_cache, self._ext_cache = s_raw, ext
        try:
            return super().step_session(s_raw, ext, effort, decode_margin,
                                        decode_entropy, hit_cap, active_idx)
        finally:
            self._s_raw_cache = self._ext_cache = None

    # pérdidas nativas del incumbente (asimetría documentada, §6.5)
    def interoception_loss(self, s_target) -> torch.Tensor:
        # el objetivo nativo es s_n (d_intero); aquí llega s_raw crudo — se
        # proyecta con la misma cabeza (sin gradiente al objetivo)
        with torch.no_grad():
            tgt = self.intero_proj(s_target.to(self.intero_proj.weight.dtype))
        return self.core.interoception_loss(tgt.detach())

    def homeostatic_loss(self) -> torch.Tensor:
        return self.core.homeostatic_loss()

    def stability_penalty(self) -> torch.Tensor:
        return self.core.stability_penalty()

    def state_summary(self) -> dict:
        return self.core.state_summary()

    def pin_fp32(self):
        if hasattr(self.core, "pin_fp32"):
            self.core.pin_fp32()
        return self

    def _core_swap(self, half: int):
        self.core.h_t = torch.cat(
            [self.core.h_t[half:], self.core.h_t[:half]], 0)
        self.core.h_tm1 = torch.cat(
            [self.core.h_tm1[half:], self.core.h_tm1[:half]], 0)
        self.h_t = self.core.h_t


# ------------------------------------------------------------------------- #
#  gru_sess — estado recurrente genérico con las mismas señales
# ------------------------------------------------------------------------- #
class GruSessionAdapter(_SessionControllerBase):
    """GRUCell sobre las señales interoceptivas normalizadas. hidden
    PERSISTE entre instancias (reset por sesión). H se iguala a la dimensión
    de ESTADO PERSISTENTE del plano (§6.5: paridad de capacidad de memoria,
    no de parámetros — declarado)."""

    def __init__(self, n_nodes: int = 6, d_h: int = 64, d_intero: int = 4,
                 hidden: int = 128, **kw):
        super().__init__(n_nodes, d_h, d_intero, dz=hidden, **kw)
        self.norm = RunningNorm()
        self.gru = nn.GRUCell(N_CHANNELS, hidden)
        self.hidden = hidden
        self.register_buffer("h_gru", torch.zeros(1, hidden), persistent=False)

    def _core_reset(self, batch, device):
        self.h_gru = torch.zeros(batch, self.hidden, device=device,
                                 dtype=torch.float32)

    def _core_step(self, sig, B_tick, device, active_idx):
        s, m = sig.to_tensor(B_tick, device, dtype=torch.float32)
        x = self.norm(s, m)
        h_prev = (self.h_gru if active_idx is None
                  else self.h_gru.index_select(0, active_idx))
        h_new = self.gru(x, h_prev)
        if active_idx is None:
            self.h_gru = h_new
        else:
            self.h_gru = self.h_gru.index_copy(0, active_idx, h_new)
        return h_new

    def pin_fp32(self):
        # el núcleo recurrente vive en FP32 (mismo trato que el plano: evita
        # la congelación BF16 de parámetros lentos)
        self.norm = self.norm.float()
        self.gru = self.gru.float()
        return self

    def _core_swap(self, half: int):
        self.h_gru = torch.cat([self.h_gru[half:], self.h_gru[:half]], 0)


# ------------------------------------------------------------------------- #
#  react_sess — el suelo: observaciones del tick → actuadores, sin estado
# ------------------------------------------------------------------------- #
class ReactSessionAdapter(_SessionControllerBase):
    """Política reactiva sin estado entre ticks NI entre instancias: cuánto
    gobierno se consigue con pura observabilidad (§4). Embedding pequeño de
    las señales normalizadas del tick → cabezas compartidas."""

    def __init__(self, n_nodes: int = 6, d_h: int = 64, d_intero: int = 4,
                 dz: int = 32, **kw):
        super().__init__(n_nodes, d_h, d_intero, dz=dz, **kw)
        self.norm = RunningNorm()
        self.embed = nn.Linear(N_CHANNELS, dz)
        nn.init.normal_(self.embed.weight, std=0.05)
        nn.init.zeros_(self.embed.bias)

    def _core_reset(self, batch, device):
        pass

    def _core_step(self, sig, B_tick, device, active_idx):
        s, m = sig.to_tensor(B_tick, device, dtype=torch.float32)
        x = self.norm(s, m)
        return torch.tanh(self.embed(x))

    def pin_fp32(self):
        self.norm = self.norm.float()
        self.embed = self.embed.float()
        return self
