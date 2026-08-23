"""Executor secuencial y controladores para el benchmark causal de cómputo."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.budgeted_stream import BudgetedStreamBatch
from .hbp import HBPConfig, HomeostaticBackgroundProcessor


@dataclass
class BudgetedStreamConfig:
    n_states: int = 12
    n_ops: int = 4
    max_steps: int = 9
    d_model: int = 128
    variant: str = "hbp_full"       # gating_wm | hbp_first | hbp_full
    observe_length: bool = True      # False: el scheduler nunca recibe K
    halt_mod_gain: float = 2.0
    step_cost: float = 0.01
    auxiliary_weight: float = 0.5
    completion_weight: float = 1.0
    # Las CE antes de K son deliberadamente irreducibles (faltan operaciones).
    # Sirven como señal de crédito para p(n|x), pero no deben enseñar al executor
    # a adivinar el futuro y contradecir la supervisión del estado corriente.
    detach_depth_loss_for_policy: bool = True
    beta_intero: float = 0.02
    beta_homeo: float = 0.001
    beta_stab: float = 0.01


def _hbp_config(variant: str) -> HBPConfig:
    cfg = HBPConfig(n_nodes=3)
    if variant == "hbp_first":
        cfg.order = 1
        cfg.zeta_init = 2.5
        cfg.zeta_min = 2.0
        cfg.c_init = 0.3
        cfg.c_max = 0.4
        cfg.omega0_init = 0.3
        cfg.omega0_max = 0.45
    elif variant == "hbp_full":
        cfg.order = 2
        cfg.zeta_init = 0.5
        cfg.zeta_min = 0.05
        cfg.c_init = 0.4
        cfg.c_max = 0.7
    else:
        raise ValueError(f"variante HBP desconocida: {variant}")
    return cfg


class BudgetedStreamReasoner(nn.Module):
    """Consume exactamente una operación nueva por tick.

    El GRU representa la memoria de trabajo/executor. El HBP, cuando existe,
    solo modula la decisión de halting: así el contraste causal de cuotas no se
    confunde con cambiar el algoritmo que ejecuta las transiciones.
    """

    def __init__(self, cfg: BudgetedStreamConfig):
        super().__init__()
        if cfg.variant not in ("gating_wm", "hbp_first", "hbp_full"):
            raise ValueError("variant debe ser gating_wm, hbp_first o hbp_full")
        self.cfg = cfg
        d = cfg.d_model
        self.state_emb = nn.Embedding(cfg.n_states, d)
        self.op_emb = nn.Embedding(cfg.n_ops + 1, d, padding_idx=cfg.n_ops)
        self.executor = nn.GRUCell(d, d)
        self.state_norm = nn.LayerNorm(d)
        self.readout = nn.Linear(d, cfg.n_states)
        if cfg.observe_length:
            # Benchmark de mecanismo: conoce tamaño del trabajo K y tiempo t.
            self.halt_proj = nn.Linear(2, 1)
        else:
            # Régimen endógeno: sólo estado causal del executor, cambio,
            # incertidumbre y esfuerzo consumido. K no entra en este camino.
            hidden = max(16, d // 2)
            self.halt_proj = nn.Sequential(
                nn.Linear(d + 3, hidden), nn.Tanh(), nn.Linear(hidden, 1))

        # Inicializar el camino compartido *antes* de construir el HBP. Los
        # módulos internos del HBP consumen RNG al crearse; si se llamara a
        # self.apply(...) al final, embeddings/readout partirían de pesos
        # distintos entre variantes aun usando la misma seed. El GRU conserva
        # su reset_parameters de construcción, que también ocurre antes del HBP.
        for module in (self.state_emb, self.op_emb, self.state_norm,
                       self.readout, self.halt_proj):
            module.apply(self._init_weights)

        self.use_hbp = cfg.variant != "gating_wm"
        if self.use_hbp:
            hcfg = _hbp_config(cfg.variant)
            self.hbp = HomeostaticBackgroundProcessor(hcfg)
            self.intero_proj = nn.Linear(4, hcfg.d_intero)
            self.input_to_hbp = nn.Linear(d, hcfg.d_h)
            for module in (self.hbp, self.intero_proj, self.input_to_hbp):
                module.apply(self._init_weights)

        self._last_halt_probs = None
        self._last_n_expected = None
        self._last_reasoner_step_units = None
        self._last_active_batch_sizes = None
        if self.use_hbp:
            self.hbp.init_physics()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def pin_fp32(self):
        if self.use_hbp:
            self.hbp.pin_fp32()
        return self

    def _reset_hbp(self, state: torch.Tensor) -> None:
        if not self.use_hbp:
            return
        B = state.shape[0]
        self.hbp.reset_state(B, device=state.device, dtype=state.dtype)
        kick = self.input_to_hbp(state).unsqueeze(1).expand(B, 3, -1)
        self.hbp.h_t = self.hbp.h_t + kick

    def _hbp_tick(self, state: torch.Tensor, previous: torch.Tensor,
                  logits: torch.Tensor, lengths: torch.Tensor,
                  tick: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve (halt_bias, interoception_loss) para el batch completo."""
        B = state.shape[0]
        dtype, device = state.dtype, state.device
        effort = torch.full((B,), tick / self.cfg.max_steps, device=device, dtype=dtype)
        progress = (state - previous).norm(dim=-1) / (self.cfg.d_model ** 0.5)
        magnitude = state.abs().mean(dim=-1)
        entropy = -(logits.float().softmax(dim=-1)
                    * logits.float().log_softmax(dim=-1)).sum(dim=-1).to(dtype)
        if self.cfg.observe_length:
            context = lengths.to(dtype) / self.cfg.max_steps
        else:
            # Sustituye la longitud privilegiada por confianza interna.
            context = 1.0 - entropy / torch.log(torch.tensor(
                float(self.cfg.n_states), device=device, dtype=dtype))
        # Tres nodos con vistas complementarias del mismo proceso: executor,
        # memoria/estado y controlador de confianza.
        raw = torch.stack([
            torch.stack([progress, magnitude, effort, context], dim=-1),
            torch.stack([magnitude, progress, effort, context], dim=-1),
            torch.stack([entropy, progress, effort, context], dim=-1),
        ], dim=1)                                                       # (B,3,4)
        intero = self.intero_proj(raw)
        ext = self.input_to_hbp(state).unsqueeze(1).expand(B, 3, -1)
        self.hbp.step(intero, ext_force=ext)
        mod = self.hbp.modulation()
        halt_bias = ((mod["halt_threshold"][:, 2] - 0.5)
                     * self.cfg.halt_mod_gain)
        return halt_bias, self.hbp.interoception_loss(intero.detach()).float()

    def executor_auxiliary_loss(self, batch: BudgetedStreamBatch) -> torch.Tensor:
        """Supervisa el estado corriente sin ejecutar el controller ni el HBP.

        Durante el preentrenamiento sólo se optimiza el executor. Evitar el
        rollout homeostático aquí conserva exactamente su objetivo auxiliar y
        elimina cómputo que no puede contribuir gradiente a esos parámetros.
        """
        state = self.state_emb(batch.initial_state)
        auxiliary_sum = torch.zeros(
            (), device=state.device, dtype=torch.float32)
        auxiliary_count = 0
        for tick in range(1, self.cfg.max_steps + 1):
            op = batch.operations[:, tick - 1]
            candidate = self.executor(self.op_emb(op), state)
            valid = batch.lengths >= tick
            state = torch.where(valid.unsqueeze(-1), candidate, state)
            if valid.any():
                logits = self.readout(self.state_norm(state[valid]))
                auxiliary_sum = auxiliary_sum + F.cross_entropy(
                    logits, batch.intermediate_states[valid, tick - 1],
                    reduction="sum").float()
                auxiliary_count += int(valid.sum())
        return auxiliary_sum / max(1, auxiliary_count)

    def forward_soft(self, batch: BudgetedStreamBatch,
                     compute_loss: bool = True) -> dict[str, torch.Tensor]:
        """Rollout PonderNet completo; todas las operaciones futuras permanecen ocultas."""
        B = batch.initial_state.shape[0]
        device = batch.initial_state.device
        state = self.state_emb(batch.initial_state)
        self._reset_hbp(state)
        remainder = torch.ones(B, device=device, dtype=torch.float32)
        n_expected = torch.zeros(B, device=device, dtype=torch.float32)
        halt_probs, halt_logits, logits_steps, final_losses = [], [], [], []
        auxiliary_sum = torch.zeros((), device=device, dtype=torch.float32)
        auxiliary_count = 0
        intero_sum = torch.zeros((), device=device, dtype=torch.float32)

        for tick in range(1, self.cfg.max_steps + 1):
            previous = state
            op = batch.operations[:, tick - 1]
            candidate = self.executor(self.op_emb(op), state)
            consumes_real_op = (batch.lengths >= tick).unsqueeze(-1)
            state = torch.where(consumes_real_op, candidate, state)
            logits = self.readout(self.state_norm(state))
            logits_steps.append(logits)

            halt_bias = None
            if self.use_hbp:
                halt_bias, intero_loss = self._hbp_tick(
                    state, previous, logits, batch.lengths, tick)
                intero_sum = intero_sum + intero_loss

            effort_feature = torch.full(
                (B, 1), float(tick), device=device, dtype=state.dtype)
            if self.cfg.observe_length:
                length_feature = batch.lengths.to(state.dtype).unsqueeze(-1)
                halt_features = torch.cat(
                    [length_feature, effort_feature], dim=-1)
            else:
                normalized = self.state_norm(state)
                progress = ((state - previous).norm(dim=-1, keepdim=True)
                            / (self.cfg.d_model ** 0.5))
                entropy = -(logits.float().softmax(dim=-1)
                            * logits.float().log_softmax(dim=-1)).sum(
                                dim=-1, keepdim=True).to(state.dtype)
                entropy = entropy / torch.log(torch.tensor(
                    float(self.cfg.n_states), device=device, dtype=state.dtype))
                effort_feature = effort_feature / self.cfg.max_steps
                halt_features = torch.cat(
                    [normalized, progress, entropy, effort_feature], dim=-1)
            halt_logit = self.halt_proj(halt_features).squeeze(-1).float()
            if halt_bias is not None:
                halt_logit = halt_logit + halt_bias.float()
            halt_logits.append(halt_logit)
            lam = torch.sigmoid(halt_logit)
            if tick == self.cfg.max_steps:
                lam = torch.ones_like(lam)
            p_tick = remainder * lam
            halt_probs.append(p_tick)
            n_expected = n_expected + p_tick * tick
            remainder = remainder * (1.0 - lam)

            if compute_loss:
                final_losses.append(F.cross_entropy(
                    logits, batch.final_state, reduction="none"))
                valid = batch.lengths >= tick
                if valid.any():
                    aux = F.cross_entropy(
                        logits[valid], batch.intermediate_states[valid, tick - 1],
                        reduction="sum")
                    auxiliary_sum = auxiliary_sum + aux.float()
                    auxiliary_count += int(valid.sum())

        p = torch.stack(halt_probs, dim=1)                              # (B,N)
        raw_halt_logits = torch.stack(halt_logits, dim=1)               # (B,N)
        logits_by_step = torch.stack(logits_steps, dim=1)               # (B,N,C)
        mixed_prob = (p.unsqueeze(-1) * logits_by_step.float().softmax(dim=-1)).sum(dim=1)
        self._last_halt_probs = p.detach()
        self._last_n_expected = n_expected.detach()
        self._last_reasoner_step_units = None
        self._last_active_batch_sizes = None
        out = {"probabilities": mixed_prob, "halt_probs": p,
               "n_expected": n_expected, "logits_by_step": logits_by_step}
        out["halt_logits_by_step"] = raw_halt_logits

        if compute_loss:
            per_depth_loss = torch.stack(final_losses, dim=1)
            policy_depth_loss = (per_depth_loss.detach()
                                 if self.cfg.detach_depth_loss_for_policy
                                 else per_depth_loss)
            expected_task = (p * policy_depth_loss).sum(dim=1).mean()
            auxiliary = auxiliary_sum / max(1, auxiliary_count)
            step_cost = self.cfg.step_cost * n_expected.mean()
            rows = torch.arange(B, device=device)
            completion = -torch.log(
                p[rows, batch.lengths - 1].clamp_min(1e-8)).mean()
            total = (expected_task + self.cfg.auxiliary_weight * auxiliary
                     + self.cfg.completion_weight * completion + step_cost)
            losses = {"expected_task": expected_task, "auxiliary": auxiliary,
                      "completion": completion, "step_cost": step_cost}
            if self.use_hbp:
                intero = intero_sum / self.cfg.max_steps
                homeo = self.hbp.homeostatic_loss()
                stab = self.hbp.stability_penalty()
                total = (total + self.cfg.beta_intero * intero
                         + self.cfg.beta_homeo * homeo
                         + self.cfg.beta_stab * stab)
                losses.update({"intero": intero, "homeo": homeo, "stab": stab})
            losses["total"] = total
            out["losses"] = losses
        return out

    @torch.no_grad()
    def forward_forced(self, batch: BudgetedStreamBatch,
                       forced_steps: torch.Tensor) -> torch.Tensor:
        """Ejecuta cuotas enteras usando solo el sub-batch todavía activo."""
        B = batch.initial_state.shape[0]
        if forced_steps.shape != (B,):
            raise ValueError("forced_steps debe tener forma (B,)")
        forced_steps = forced_steps.to(batch.initial_state.device, dtype=torch.long)
        if (forced_steps < 1).any() or (forced_steps > self.cfg.max_steps).any():
            raise ValueError(f"forced_steps debe estar en [1,{self.cfg.max_steps}]")
        current = self.state_emb(batch.initial_state)
        final = torch.empty_like(current)
        active_sizes = []
        for tick in range(1, int(forced_steps.max()) + 1):
            active = (forced_steps >= tick).nonzero(as_tuple=True)[0]
            active_sizes.append(int(active.numel()))
            state = current.index_select(0, active)
            op = batch.operations.index_select(0, active)[:, tick - 1]
            candidate = self.executor(self.op_emb(op), state)
            valid = (batch.lengths.index_select(0, active) >= tick).unsqueeze(-1)
            state_next = torch.where(valid, candidate, state)
            current = current.index_copy(0, active, state_next)
            stopping = (forced_steps == tick).nonzero(as_tuple=True)[0]
            if stopping.numel():
                final = final.index_copy(0, stopping, current.index_select(0, stopping))
        self._last_reasoner_step_units = int(forced_steps.sum())
        self._last_active_batch_sizes = active_sizes
        return self.readout(self.state_norm(final))
