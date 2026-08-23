"""
MiuraCognitive - Recurrencia Adaptativa (Adaptive Depth)
=========================================================
Mecanismo de halting inspirado en PonderNet (Banino et al. 2021).

Un bloque de procesamiento se aplica EN BUCLE hasta que una señal de
parada (aprendida) decide detenerse. La cantidad de cómputo se adapta a
la dificultad del input.

CLAVE: el umbral de parada está MODULADO por el HBP (si el estado interno
indica alta incertidumbre, el modelo itera más).

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
import torch
import torch.nn as nn


class AdaptiveHalting(nn.Module):
    """Controlador de halting. En cada iteración produce una probabilidad
    de parada λ_n. La probabilidad de detenerse exactamente en el paso n
    sigue una distribución tipo geométrica: p_n = λ_n · Π_{k<n}(1-λ_k)."""

    def __init__(self, d_model: int, max_steps: int = 8, n_obs: int = 0):
        super().__init__()
        self.max_steps = max_steps
        # Red de parada: mira el estado pooled + la LONGITUD VÁLIDA normalizada.
        # La longitud es una señal legítima y observable para decidir cuánto
        # iterar; antes llegaba de forma implícita (fracción de PAD en el mean
        # sin máscara) — aquí se da explícita y el pooled queda limpio de PAD.
        # n_obs (F3b, §3 del prereg): nº de features de OBSERVACIÓN DEL ENTORNO
        # (fracción de presupuesto gastada/restante, stake, etc.) que entran
        # como input directo del halting — el conjunto de observación es
        # idéntico para todos los brazos; n_obs=0 reproduce el halting F3a.
        self.n_obs = n_obs
        self.halt_proj = nn.Linear(d_model + 1 + n_obs, 1)
        # Contabilidad F3b (§6.1): ticks CARGADOS por fila en el último forward
        # (un tick carga mientras la fila conserve masa PonderNet > umbral).
        self._last_charged_ticks = None
        # Instrumento VG-B0 (N3 fase B): offset POR FILA sobre el logit de
        # parada, fijado en EVAL (None = inerte; negativo = pensar más).
        self.logit_offset = None

    def forward(self, step_fn, x, hbp_tick=None, pool_mask=None,
                return_step_states: bool = False, intervene_fn=None,
                row_caps: torch.Tensor | None = None,
                obs_feats: torch.Tensor | None = None):
        """
        Args:
          step_fn: aplica UNA iteración del reasoner. Firma: (x, mod) -> x,
                   donde mod es la modulación top-down del HBP (o None).
          x:       (B, T, d_model) estado inicial.
          hbp_tick: callback opcional (x_n, x_prev, n) -> (halt_bias, reasoner_mod).
                   Es el ACOPLAMIENTO con el HBP: un tick del VEI por iteración.
          pool_mask: (B, T) bool opcional — posiciones VÁLIDAS (no-PAD). Se usa
                   SOLO para derivar la señal de longitud explícita frac_valid;
                   el pooling del halting es sobre la secuencia completa (las
                   posiciones PAD son scratchpad computacional del reasoner).
          intervene_fn: callback opcional (x, n) -> x aplicado TRAS cada paso
                   (instrumentación G0: lesiones/swaps de estado; None = inerte).
          row_caps: (B,) long opcional — presupuesto de ticks POR FILA (F3b,
                   §6.1): cuando n >= cap_i la fila i recibe λ:=1 (clamp,
                   misma semántica que el override de N_max). El corte es
                   por-fila, no por batch.
          obs_feats: (B, n_obs) opcional — observaciones del entorno (§3)
                   concatenadas al input del halting.

        Con return_step_states=True el bucle NO corta temprano (corre los
        N_max ticks): los instrumentos G0 necesitan la trayectoria completa.

        Devuelve (x_final, n_expected, halt_dist).
        """
        B = x.shape[0]
        device = x.device
        if (obs_feats is None) != (self.n_obs == 0):
            raise ValueError(f"obs_feats debe casar con n_obs={self.n_obs}")
        if row_caps is not None:
            row_caps = row_caps.to(device=device, dtype=torch.long).clamp(
                1, self.max_steps)
        # Acumuladores de PonderNet en FP32: el producto telescópico de (1-λ)
        # sobre ~10-24 pasos acumula 2-4% de error relativo en BF16 y cuantiza
        # n_expected (ruido en corr(K, n_iter)). Solo weighted_state necesita
        # el dtype de x.
        remainders = torch.ones(B, device=device, dtype=torch.float32)
        n_expected = torch.zeros(B, device=device, dtype=torch.float32)
        weighted_state = torch.zeros_like(x)
        halt_dist = []
        step_states = []
        halt_probs_live = []
        if pool_mask is not None:
            frac_valid = (pool_mask.to(x.dtype).sum(dim=1, keepdim=True)
                          / x.shape[1])                              # (B,1) longitud norm.
        else:
            frac_valid = torch.ones(B, 1, device=device, dtype=x.dtype)

        reasoner_mod = None     # modulación del HBP para el reasoner (None en t=0)
        x_prev = x
        charged = torch.zeros(B, device=device, dtype=torch.float32)
        for n in range(1, self.max_steps + 1):
            # Contabilidad §6.1: el tick carga a las filas que aún tienen masa
            # (se computa ANTES de actualizar remainders con el λ de este tick).
            charged = charged + (remainders > 1e-4).float()

            x = step_fn(x, reasoner_mod)             # una iteración de pensamiento
            if intervene_fn is not None:
                x = intervene_fn(x, n)               # instrumentación G0 (swap/lesión)

            # Tick del HBP con la interocepción del estado actual (si hay HBP).
            halt_bias = None
            if hbp_tick is not None:
                halt_bias, reasoner_mod = hbp_tick(x, x_prev, n)

            pooled = x.mean(dim=1)      # secuencia completa (incluye scratchpad)
            halt_in = [pooled, frac_valid]
            if obs_feats is not None:
                halt_in.append(obs_feats.to(pooled.dtype))
            logit = self.halt_proj(torch.cat(halt_in, dim=-1)).squeeze(-1).float()
            if halt_bias is not None:
                # Sesgo de parada del HBP (ya escalado por halt_mod_gain en miura).
                logit = logit + halt_bias.float()

            if self.logit_offset is not None:
                logit = logit + self.logit_offset.to(logit.dtype)
            lam = torch.sigmoid(logit)
            if n == self.max_steps:
                lam = torch.ones_like(lam)
            elif row_caps is not None:
                # λ-clamp POR FILA al agotar su presupuesto (§6.1): la fila
                # deja de acumular masa (y de cargar ticks) desde su cap.
                lam = torch.where(torch.as_tensor(n, device=device) >= row_caps,
                                  torch.ones_like(lam), lam)

            p_n = remainders * lam
            halt_dist.append(p_n.detach())
            if return_step_states:
                step_states.append(x)
                halt_probs_live.append(p_n)
            weighted_state = weighted_state + p_n.to(x.dtype).view(B, 1, 1) * x
            n_expected = n_expected + p_n * n
            remainders = remainders * (1.0 - lam)
            x_prev = x

            if remainders.max().item() < 1e-4 and not return_step_states:
                break                    # con estados grabados: rollout completo

        self._last_charged_ticks = charged.detach()
        if return_step_states:
            return (weighted_state, n_expected, halt_dist,
                    step_states, halt_probs_live)
        return weighted_state, n_expected, halt_dist

    def forward_forced(self, step_fn, x, forced_steps, hbp_tick=None):
        """Ejecuta una cuota ENTERA de pasos por muestra.

        A diferencia de :meth:`forward`, esta ruta no forma una mezcla PonderNet:
        cada ejemplo devuelve el estado alcanzado exactamente tras ``forced_steps``
        iteraciones. En cada tick solo se envían a ``step_fn`` las filas activas,
        de modo que ``forced_steps.sum()`` es el número efectivo de ejecuciones
        muestra-reasoner (el backbone fijo no forma parte de ese presupuesto).

        ``step_fn`` y ``hbp_tick`` reciben un argumento adicional ``active_idx``
        con los índices de las filas activas dentro del batch original. Esta ruta
        se usa para intervenciones de evaluación; el entrenamiento soft existente
        permanece idéntico.
        """
        if forced_steps.ndim != 1 or forced_steps.shape[0] != x.shape[0]:
            raise ValueError("forced_steps debe tener forma (B,)")
        if forced_steps.is_floating_point():
            rounded = forced_steps.round()
            if not torch.equal(forced_steps, rounded):
                raise ValueError("forced_steps debe contener enteros")
            forced_steps = rounded
        forced_steps = forced_steps.to(device=x.device, dtype=torch.long)
        if (forced_steps < 1).any() or (forced_steps > self.max_steps).any():
            raise ValueError(f"forced_steps debe estar en [1, {self.max_steps}]")

        B = x.shape[0]
        current = x
        final_state = torch.empty_like(x)
        reasoner_mod = None
        halt_dist = []

        for n in range(1, int(forced_steps.max().item()) + 1):
            active_idx = (forced_steps >= n).nonzero(as_tuple=True)[0]
            x_prev = current.index_select(0, active_idx)
            mod_active = (None if reasoner_mod is None
                          else reasoner_mod.index_select(0, active_idx))
            x_next = step_fn(x_prev, mod_active, active_idx)
            current = current.index_copy(0, active_idx, x_next)

            if hbp_tick is not None:
                _, mod_next = hbp_tick(x_next, x_prev, n, active_idx)
                if reasoner_mod is None:
                    reasoner_mod = torch.zeros(
                        (B, mod_next.shape[-1]), device=mod_next.device,
                        dtype=mod_next.dtype)
                reasoner_mod = reasoner_mod.index_copy(0, active_idx, mod_next)

            stopping_idx = (forced_steps == n).nonzero(as_tuple=True)[0]
            if stopping_idx.numel():
                final_state = final_state.index_copy(
                    0, stopping_idx, current.index_select(0, stopping_idx))
            halt_dist.append((forced_steps == n).to(torch.float32).detach())

        return final_state, forced_steps.to(torch.float32), halt_dist


def halting_regularization(n_expected: torch.Tensor, lambda_p: float = 0.2) -> torch.Tensor:
    """Penalización por exceso de cómputo. Empuja E[n_iter] a ser bajo,
    equilibrado contra la mejora en la tarea principal.
    Versión simple: penaliza el número esperado de iteraciones."""
    return n_expected.mean() * lambda_p
