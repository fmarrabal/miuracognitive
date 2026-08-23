"""
mHBP — Adaptador al reasoner de MiuraCognitive (Fase 3a, PREREG_F3A.md).

Sustituye al HomeostaticBackgroundProcessor (campo único) por el plano
certificado de 4 campos, con TODA la disciplina aprendida en el arco:

  · MODULADOR, no fuente (F2/F2b): toda vía de salida es una modulación
    acotada de un camino guiado por señal, con init ≈ neutro (arranca
    comportándose como gating_wm). Sin vía del plano a los logits de tarea.
  · El enchufe es λ PRE-TECHO (G0.1): el sesgo de halting modula la señal de
    completitud; la presión de PRESUPUESTO es una vía separada guiada por el
    ESFUERZO (señal) cuya ganancia modula el campo resource.
  · Interocepción de completitud (G0.1): margen y entropía del decode en la
    posición de respuesta, por tick — auto-observación, sin K ni target.
  · Escalas τ=(1,3,6,12) dentro del horizonte de pensamiento (N_max=24).
  · Estabilidad POR CONSTRUCCIÓN del plano (F1): stability_penalty() = 0
    aquí; los certificados se verifican offline con la batería de mhbp.

Duck-typing del contrato que miura.py espera: reset_state, h_t (buffer de
DRIVE para el kick del input), modulation-dict con las mismas formas,
interoception_loss/homeostatic_loss/stability_penalty/state_summary/pin_fp32.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..coupled_fields import MHBPConfig, CoupledMultiscaleHBP
from ..interoception import InteroceptiveSignal


class _CfgShim:
    """Lo mínimo que miura.py lee de cfg.hbp."""
    def __init__(self, n_nodes, d_h, d_intero):
        self.n_nodes = n_nodes
        self.d_h = d_h
        self.d_intero = d_intero


class MhbpReasonerAdapter(nn.Module):
    is_mhbp = True

    def __init__(self, n_nodes: int = 6, d_h: int = 64, d_intero: int = 4,
                 gate_gain: float = 0.3, res_gain: float = 0.8,
                 mask_completeness: bool = False,
                 taus: tuple = (1.0, 3.0, 6.0, 12.0),
                 per_instance_gates: bool = False,
                 value_channel: bool = False,
                 coupling_topology: str = "chain"):
        super().__init__()
        self.cfg = _CfgShim(n_nodes, d_h, d_intero)
        # leak_mask opcional (brazo de descomposición pre-declarado en PREREG
        # v2: ¿cuánto del efecto es la interocepción de completitud?)
        lm = ("confidence", "uncertainty", "risk") if mask_completeness else ()
        # taus: F3a = (1,3,6,12) dentro de N_max=24; F3b (§6.4 PREREG_F3B v2)
        # = (1,3,10,32) — la escala lenta integra el régimen/presupuesto de la
        # sesión (~30-60 ticks), la 3ª la instancia (~10 ticks).
        # value_channel (N2 §3): el canal de valor entra SOLO a la cabeza del
        # campo risk (índice 1) — veto por-cabeza en fast/deliberative/resource
        # («valorar sin tocar contenido» por construcción).
        hv = ((0, ("task_criticality", "cost_monetary")),
              (2, ("task_criticality", "cost_monetary")),
              (3, ("task_criticality", "cost_monetary"))) if value_channel else ()
        pc = MHBPConfig(taus=tuple(float(t) for t in taus), dt=1.0, theta=1.0,
                        coupling_topology=coupling_topology, allostasis=True,
                        dtype="float32", leak_mask=lm, head_leak_masks=hv)
        self.value_channel = value_channel
        self.plane = CoupledMultiscaleHBP(pc)
        d = pc.d                                       # dim latente de los campos
        self.gate_gain = gate_gain
        self.res_gain = res_gain
        # cabezas de MODULACIÓN (init pequeño ⇒ arranque neutro)
        self.V_halt = nn.Linear(4 * d, 1)              # [ū,w̄]_fast ⊕ [ū,w̄]_risk
        self.V_res = nn.Linear(2 * d, 1)               # ganancia de la presión
        # presión de presupuesto: init −5 ⇒ softplus≈0.0067 (arranque NEUTRO;
        # con −2 la rampa sesgaba +0.13·esfuerzo desde el paso 0 — panel F3a)
        self.raw_pressure = nn.Parameter(torch.tensor(-5.0))
        self.V_wm_w = nn.Linear(2 * d, d_h)            # deliberative → gates WM
        self.V_wm_f = nn.Linear(2 * d, d_h)
        self.b_wm_w = nn.Parameter(torch.zeros(1))
        self.b_wm_f = nn.Parameter(torch.zeros(1))
        self.V_gate = nn.Linear(2 * d, d_h)            # fast → gate de bloques
        self.drive_proj = nn.Linear(d_h, 2)            # drive del input → 2 canales
        for m in (self.V_halt, self.V_res, self.V_wm_w, self.V_wm_f,
                  self.V_gate, self.drive_proj):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        # buffer de DRIVE: miura hace `hbp.h_t = hbp.h_t + kick` (contrato)
        self.register_buffer("h_t", torch.zeros(1, n_nodes, d_h), persistent=False)
        self._B = None
        self._last_mod = None
        # lesión por vías EN EVAL (diagnóstico F3a, rama 3): subconjunto de
        # {"halt", "wm", "gate"} — neutraliza esa vía de modulación
        self.lesion = set()
        # --- F3b (PREREG_F3B v2 §5): gates de CONTENIDO por-instancia ---
        # Si True, wm_write/wm_forget/block_gate se computan del estado
        # PERSISTIDO en la frontera de instancia (antes de tick alguno) y se
        # mantienen constantes durante la instancia. Halting y presión siguen
        # por-tick (gobierno del cómputo; la hipótesis M3 es sobre contenido).
        self.per_instance_gates = per_instance_gates
        self._frozen_gates = None          # (wm_w, wm_f, gate) cacheados
        # --- F3a-R (PREREG_F3A_R): vías de CONTENIDO congeladas tras el
        # tick 1 DENTRO de cada forward — la operacionalización F3a de la
        # hipótesis de mecanismo (sin fluctuación por-tick, con modulación
        # dependiente del estado). reset_state limpia el caché por forward.
        self.freeze_content_after_tick1 = False

    # ------------------------------------------------------------------ #
    def reset_state(self, batch: int = 1, device=None, dtype=None):
        device = device or self.V_halt.weight.device
        self.plane.to(device)
        self.plane.reset_state(batch, device=device)
        self.h_t = torch.zeros(batch, self.cfg.n_nodes, self.cfg.d_h,
                               device=device,
                               dtype=dtype or self.V_halt.weight.dtype)
        self._B = batch
        self._last_mod = None
        self._frozen_gates = None

    # ------------------------------------------------------------------ #
    def _pool(self, q: int):
        u, w = self.plane.fields[q].node_mean()        # (B, d) fp32
        return torch.cat([u, w], dim=-1).float()

    def step_mhbp(self, s_raw: torch.Tensor, ext: torch.Tensor,
                  effort: torch.Tensor, decode_margin: torch.Tensor,
                  decode_entropy: torch.Tensor, hit_cap: torch.Tensor,
                  active_idx=None, value_signal=None) -> dict:
        """Un tick del plano + modulaciones. s_raw: (B, N, 5) señales por nodo
        [norma, activación, entropía, esfuerzo, frac_valid]; ext: (B, d_h)
        drive del input; decode_*: (B,) auto-observación de completitud;
        hit_cap: (B,) 1.0 si el tick es el techo (evento de presupuesto).
        value_signal (N2 §3): (B_tick,) valor DETACHED (stakê endógeno o
        stake del oráculo) → canales task_criticality/cost_monetary, que el
        veto por-cabeza restringe al campo risk. None = canal ausente
        (máscara → 0 exacto: bit-igual con F3a, test de cableado).
        active_idx: tick enmascarado por filas (ruta forced de la batería —
        el NotImplementedError de F3a queda retirado)."""
        B = s_raw.shape[0]
        L = self.cfg.n_nodes - 2
        h_drive = (self.h_t if active_idx is None
                   else self.h_t.index_select(0, active_idx))
        drive = (h_drive.mean(dim=1) + ext).float()    # (B_tick, d_h)
        dch = torch.tanh(self.drive_proj(drive.to(self.drive_proj.weight.dtype)))
        vals = {
            "attention_entropy": s_raw[:, :L, 2].float().mean(1),
            "entropy": s_raw[:, L, 2].float(),               # reasoner
            "state_delta": s_raw[:, L, 0].float(),           # progreso
            "novelty": s_raw[:, L, 1].float(),               # activación
            "memory_load": s_raw[:, L + 1, 0].float(),       # ocupación WM
            "elapsed_time": effort.float(),
            "queue_load": s_raw[:, 0, 4].float(),            # frac_valid
            "confidence": decode_margin.float(),             # G0.1: completitud
            "uncertainty": decode_entropy.float(),
            "risk": decode_entropy.float(),                  # el campo risk la ve
            "gpu_utilization": dch[:, 0].float(),            # drive del input (2 ch)
            "gpu_memory": dch[:, 1].float(),
            "token_cost": hit_cap.float(),                   # evento de techo → resource
        }
        if value_signal is not None:
            v = value_signal.detach().float()
            vals["task_criticality"] = v
            vals["cost_monetary"] = v
        sig = InteroceptiveSignal(values=vals)
        self._tick_plane(sig, active_idx)               # avanza los 4 campos
        z_f, z_r = self._pool(0), self._pool(1)         # fast, risk
        z_d, z_m = self._pool(2), self._pool(3)         # deliberative, resource
        if active_idx is not None:
            z_f = z_f.index_select(0, active_idx)
            z_r = z_r.index_select(0, active_idx)
            z_d = z_d.index_select(0, active_idx)
            z_m = z_m.index_select(0, active_idx)
        wdt = self.V_halt.weight.dtype

        # 1) sesgo de halting sobre λ pre-techo: autoridad SIMÉTRICA ±1,
        #    equiparada al incumbente ((σ−0.5)·2 ∈ (−1,1)) — panel F3a
        halt_mod = torch.tanh(
            self.V_halt(torch.cat([z_f, z_r], -1).to(wdt))).squeeze(-1)
        # 2) presión de PRESUPUESTO: señal = esfuerzo; el resource escala.
        #    ADITIVA sobre el logit (forma declarada), no fundida en otra tanh
        gain_res = 1.0 + self.res_gain * torch.tanh(
            self.V_res(z_m.to(wdt))).squeeze(-1)
        pressure = torch.nn.functional.softplus(self.raw_pressure) \
            * effort.to(wdt) * gain_res
        bias_total = (halt_mod + pressure).clamp(-1.0, 1.0)
        if "halt" in self.lesion:
            bias_total = torch.zeros_like(bias_total)
        halt_th = 0.5 + 0.5 * bias_total                        # (B,)
        # 3-4) gates de CONTENIDO (WM desde deliberative; block_gate desde
        # fast). Con freeze_content_after_tick1 (F3a-R): se computan en el
        # tick 1 y se REUTILIZAN el resto del forward (el gradiente a las
        # cabezas fluye solo por el tick 1 — sin detach, declarado).
        if self.freeze_content_after_tick1 and self._frozen_gates is not None:
            wm_w, wm_f, gate = self._frozen_gates
            if active_idx is not None:
                wm_w = wm_w.index_select(0, active_idx)
                wm_f = wm_f.index_select(0, active_idx)
                gate = gate.index_select(0, active_idx)
        else:
            wm_w = torch.sigmoid(self.b_wm_w + self.V_wm_w(z_d.to(wdt)))
            wm_f = torch.sigmoid(self.b_wm_f + self.V_wm_f(z_d.to(wdt)))
            gate = self.gate_gain * torch.tanh(self.V_gate(z_f.to(wdt)))
            if self.freeze_content_after_tick1 and active_idx is None:
                self._frozen_gates = (wm_w, wm_f, gate)
        if "wm" in self.lesion:
            wm_w = torch.full_like(wm_w, 0.5)
            wm_f = torch.full_like(wm_f, 0.5)
        if "gate" in self.lesion:
            gate = torch.zeros_like(gate)

        N = self.cfg.n_nodes
        mod = {
            "halt_threshold": halt_th.unsqueeze(1).expand(B, N),
            "wm_write": wm_w.unsqueeze(1).expand(B, N, -1),
            "wm_forget": wm_f.unsqueeze(1).expand(B, N, -1),
            "router_bias": torch.zeros(B, N, device=s_raw.device, dtype=wdt),
            "block_gate": gate.unsqueeze(1).expand(B, N, -1),
        }
        if active_idx is None:
            self._last_mod = mod
            return mod
        # batch parcial (ruta forced): fusionar sobre la última modulación
        if self._last_mod is None:
            self._last_mod = {k: v for k, v in self.modulation().items()}
        merged = {k: self._last_mod[k].index_copy(0, active_idx, v.to(
            self._last_mod[k].dtype)) for k, v in mod.items()}
        self._last_mod = merged
        return merged

    # ================= F3b: interfaz de sesión (PREREG_F3B v2) ================= #
    def on_instance_kick(self, kick: torch.Tensor):
        """Frontera de instancia (§6): el drive se SOBRESCRIBE (h_t := kick;
        el acumulador monótono era un contador de posición — crítico del
        panel), el pre-bucle queda neutro (_last_mod = None) y, con gates
        por-instancia, se congelan AHORA del estado persistido (antes de tick
        alguno de la instancia). El estado de los CAMPOS no se toca aquí: la
        persistencia la gobierna el runner (reset solo en frontera de sesión)."""
        self.h_t = kick.to(self.h_t.dtype)
        self._last_mod = None
        if self.per_instance_gates:
            self._frozen_gates = self._content_gates()
        else:
            self._frozen_gates = None

    def _content_gates(self):
        """Gates de CONTENIDO (wm_write, wm_forget, block_gate) del estado
        actual de los campos. En la frontera post-reset de sesión los campos
        son cero → gates neutros (σ(0)=0.5, gate=0): instancia 1 corre sin
        modulación de contenido (documentado en el prereg §5)."""
        z_f, z_d = self._pool(0), self._pool(2)
        wdt = self.V_halt.weight.dtype
        wm_w = torch.sigmoid(self.b_wm_w + self.V_wm_w(z_d.to(wdt)))
        wm_f = torch.sigmoid(self.b_wm_f + self.V_wm_f(z_d.to(wdt)))
        gate = self.gate_gain * torch.tanh(self.V_gate(z_f.to(wdt)))
        return wm_w, wm_f, gate

    def _tick_plane(self, sig, active_idx):
        """Tick del plano con máscara por filas (§6.1): las filas inactivas
        quedan bit-idénticas (gather → tick del sub-batch → scatter). Sin
        active_idx, tick normal del batch completo."""
        if active_idx is None:
            self.plane.tick(sig)
            return
        allo = self.plane.allo
        B_full = self.plane._B
        full_uw = []
        for f in self.plane.fields:
            full_uw.append((f.u, f.w))
            f.u = f.u.index_select(0, active_idx)
            f.w = f.w.index_select(0, active_idx)
        shift_full = allo._last_shift
        if shift_full is not None:
            allo._last_shift = shift_full.index_select(0, active_idx)
        self.plane._B = int(active_idx.numel())
        try:
            self.plane.tick(sig)
        finally:
            self.plane._B = B_full
        for f, (u_full, w_full) in zip(self.plane.fields, full_uw):
            f.u = u_full.index_copy(0, active_idx, f.u)
            f.w = w_full.index_copy(0, active_idx, f.w)
        if shift_full is not None:
            allo._last_shift = shift_full.index_copy(
                0, active_idx, allo._last_shift)
        elif allo._last_shift is not None:
            # el shift nació en este tick: expandir al batch completo (filas
            # inactivas a cero = sin desplazamiento alostático)
            base = torch.zeros(B_full, *allo._last_shift.shape[1:],
                               device=allo._last_shift.device,
                               dtype=allo._last_shift.dtype)
            allo._last_shift = base.index_copy(0, active_idx, allo._last_shift)

    def step_session(self, s_raw: torch.Tensor, ext: torch.Tensor,
                     effort: torch.Tensor, decode_margin: torch.Tensor,
                     decode_entropy: torch.Tensor, hit_cap: torch.Tensor,
                     active_idx=None) -> dict:
        """Un tick en modo SESIÓN (F3b). s_raw: (B_tick, N, 11) — cols 0-2
        señales del nodo; cola §3: [3]=esfuerzo, [4]=frac_valid, [5]=gastado,
        [6]=restante, [7]=stake, [8]=inst_left, [9]=margen, [10]=entropía.
        Devuelve el mod-dict del BATCH COMPLETO (con active_idx, las filas
        inactivas conservan su última modulación)."""
        B_tick = s_raw.shape[0]
        L = self.cfg.n_nodes - 2
        h_drive = (self.h_t if active_idx is None
                   else self.h_t.index_select(0, active_idx))
        drive = (h_drive.mean(dim=1) + ext).float()
        dch = torch.tanh(self.drive_proj(drive.to(self.drive_proj.weight.dtype)))
        spent = s_raw[:, 0, 5].float()
        sig = InteroceptiveSignal(values={
            "attention_entropy": s_raw[:, :L, 2].float().mean(1),
            "entropy": s_raw[:, L, 2].float(),               # reasoner
            "state_delta": s_raw[:, L, 0].float(),           # progreso
            "novelty": s_raw[:, L, 1].float(),               # activación
            "memory_load": s_raw[:, L + 1, 0].float(),       # ocupación WM
            "elapsed_time": effort.float(),
            "ood_score": s_raw[:, 0, 4].float(),             # frac_valid (tamaño)
            "confidence": decode_margin.float(),             # completitud (§3)
            "uncertainty": decode_entropy.float(),
            "risk": decode_entropy.float(),
            "energy_cost": spent,                            # §6.3: fracción gastada
            "latency": s_raw[:, 0, 6].float(),               # fracción restante
            "cost_monetary": s_raw[:, 0, 7].float(),         # stake (ex ante)
            "task_criticality": s_raw[:, 0, 7].float(),
            "queue_load": s_raw[:, 0, 8].float(),            # instancias restantes
            "gpu_utilization": dch[:, 0].float(),            # drive del input
            "gpu_memory": dch[:, 1].float(),
            "token_cost": hit_cap.float(),                   # techo (incl. forzados)
        })
        self._tick_plane(sig, active_idx)
        z_f, z_r = self._pool(0), self._pool(1)              # fast, risk
        z_d, z_m = self._pool(2), self._pool(3)              # deliberative, resource
        if active_idx is not None:
            z_f = z_f.index_select(0, active_idx)
            z_r = z_r.index_select(0, active_idx)
            z_d = z_d.index_select(0, active_idx)
            z_m = z_m.index_select(0, active_idx)
        wdt = self.V_halt.weight.dtype

        # 1) sesgo de halting simétrico ±1 (forma F3a)
        halt_mod = torch.tanh(
            self.V_halt(torch.cat([z_f, z_r], -1).to(wdt))).squeeze(-1)
        # 2) presión de PRESUPUESTO (§6.2): señal = ½(esfuerzo + gastado) —
        #    diferenciable dentro del tick; el resource escala la ganancia.
        gain_res = 1.0 + self.res_gain * torch.tanh(
            self.V_res(z_m.to(wdt))).squeeze(-1)
        press_sig = 0.5 * (effort.to(wdt) + spent.to(wdt))
        pressure = torch.nn.functional.softplus(self.raw_pressure) \
            * press_sig * gain_res
        bias_total = (halt_mod + pressure).clamp(-1.0, 1.0)
        if "halt" in self.lesion:
            bias_total = torch.zeros_like(bias_total)
        halt_th = 0.5 + 0.5 * bias_total
        # 3-4) gates de CONTENIDO: por-tick o congelados por-instancia (§5)
        if self.per_instance_gates and self._frozen_gates is not None:
            wm_w, wm_f, gate = self._frozen_gates
            if active_idx is not None:
                wm_w = wm_w.index_select(0, active_idx)
                wm_f = wm_f.index_select(0, active_idx)
                gate = gate.index_select(0, active_idx)
        else:
            wm_w = torch.sigmoid(self.b_wm_w + self.V_wm_w(z_d.to(wdt)))
            wm_f = torch.sigmoid(self.b_wm_f + self.V_wm_f(z_d.to(wdt)))
            gate = self.gate_gain * torch.tanh(self.V_gate(z_f.to(wdt)))
        if "wm" in self.lesion:
            wm_w = torch.full_like(wm_w, 0.5)
            wm_f = torch.full_like(wm_f, 0.5)
        if "gate" in self.lesion:
            gate = torch.zeros_like(gate)

        N, d_h = self.cfg.n_nodes, self.cfg.d_h
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
        # batch parcial: fusionar sobre la última modulación completa
        if self._last_mod is None:
            self._last_mod = {k: v for k, v in self.modulation().items()}
        merged = {k: self._last_mod[k].index_copy(0, active_idx, v.to(
            self._last_mod[k].dtype)) for k, v in mod_tick.items()}
        self._last_mod = merged
        return merged

    def modulation(self) -> dict:
        """Contrato de miura para el estado PRE-bucle (kick inicial): neutro
        salvo el drive ya acumulado en h_t (el primer tick lo consumirá)."""
        if self._last_mod is not None:
            return self._last_mod
        B = self._B or 1
        N, d_h = self.cfg.n_nodes, self.cfg.d_h
        dev = self.h_t.device
        dt = self.h_t.dtype
        return {
            "halt_threshold": torch.full((B, N), 0.5, device=dev, dtype=dt),
            "wm_write": torch.full((B, N, d_h), 0.5, device=dev, dtype=dt),
            "wm_forget": torch.full((B, N, d_h), 0.5, device=dev, dtype=dt),
            "router_bias": torch.zeros(B, N, device=dev, dtype=dt),
            "block_gate": torch.zeros(B, N, d_h, device=dev, dtype=dt),
        }

    # ---------------- contrato de pérdidas/diagnóstico ---------------- #
    def interoception_loss(self, s_target) -> torch.Tensor:
        return torch.zeros((), device=self.h_t.device)

    def homeostatic_loss(self) -> torch.Tensor:
        pen = 0.0
        for f in self.plane.fields:
            pen = pen + f.u.pow(2).mean()
        return (pen / len(self.plane.fields)).to(self.h_t.dtype)

    def stability_penalty(self) -> torch.Tensor:
        # estabilidad POR CONSTRUCCIÓN (F1); certificados offline (batería mhbp)
        return torch.zeros((), device=self.h_t.device)

    def state_summary(self) -> dict:
        with torch.no_grad():
            dev = sum(float(f.u.norm()) for f in self.plane.fields)
            var = sum(float(f.u.var()) for f in self.plane.fields)
        return {"deviation_from_rest": dev, "vei_variance": var}

    def pin_fp32(self):
        # el PLANO vive entero en FP32 (params Y buffers: es el régimen
        # certificado); las cabezas del adaptador siguen el dtype del modelo
        self.plane = self.plane.float()
        self.plane.pin_fp32()
        for p in (self.raw_pressure,):
            p.data = p.data.float()
        return self

    # ---------------- instrumentación (M2 swap de la batería) ---------------- #
    @torch.no_grad()
    def swap_halves(self, half: int):
        """Trasplanta el estado COMPLETO del plano entre mitades del batch."""
        self.h_t = torch.cat([self.h_t[half:], self.h_t[:half]], 0)
        for f in self.plane.fields:
            f.u = torch.cat([f.u[half:], f.u[:half]], 0)
            f.w = torch.cat([f.w[half:], f.w[:half]], 0)
        a = self.plane.allo
        if a._last_shift is not None and a._last_shift.shape[0] == 2 * half:
            a._last_shift = torch.cat([a._last_shift[half:], a._last_shift[:half]], 0)
