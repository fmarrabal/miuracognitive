"""Tarea causal de cómputo: transiciones reveladas una por tick.

Cada ejemplo define un estado inicial y K operaciones no conmutativas. El
executor solo recibe la operación t en el tick t; por construcción, antes de K
no ha observado las operaciones restantes y no puede determinar el estado final.
Después de K el estado se congela. Así, el valor marginal del cómputo es positivo
hasta completar la cadena y cero después, sin el artefacto de "overthinking" de
permcomp.

Los batches balanceados contienen el mismo número de ejemplos de cada longitud
K=min_steps..max_steps. Con 1..9 y batch=36, sum(K)=36*5: la asignación oráculo
q_i=K_i y la uniforme q_i=5 tienen exactamente el mismo presupuesto total.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BudgetedStreamBatch:
    initial_state: torch.Tensor       # (B,)
    operations: torch.Tensor          # (B,N), PAD_OP después de K
    lengths: torch.Tensor             # (B,)
    final_state: torch.Tensor         # (B,)
    intermediate_states: torch.Tensor # (B,N), estado tras cada tick (congelado después de K)

    def to(self, device: torch.device) -> "BudgetedStreamBatch":
        return BudgetedStreamBatch(*(
            value.to(device) for value in (
                self.initial_state, self.operations, self.lengths,
                self.final_state, self.intermediate_states)))


class BudgetedTransitionDataset:
    """Generador de cadenas sobre un conjunto fijo de permutaciones de estados."""

    def __init__(self, n_states: int = 12, min_steps: int = 1,
                 max_steps: int = 9, seed: int = 0,
                 terminal_markers: bool = False):
        if n_states < 6 or n_states % 2:
            raise ValueError("n_states debe ser par y >=6")
        if min_steps < 1 or max_steps < min_steps:
            raise ValueError("rango de pasos inválido")
        self.n_states = int(n_states)
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.terminal_markers = bool(terminal_markers)
        self.gen = torch.Generator().manual_seed(int(seed))
        self.permutations = self._build_permutations(self.n_states)
        self.n_base_ops = int(self.permutations.shape[0])
        # En el régimen de K oculto cada operación tiene una versión LAST. El
        # marcador sólo se revela junto con la transición final, nunca antes.
        self.n_ops = self.n_base_ops * (2 if self.terminal_markers else 1)
        self.PAD_OP = self.n_ops

    @staticmethod
    def _build_permutations(n: int) -> torch.Tensor:
        """Cuatro generadores no conmutativos, biyectivos y sin información perdida."""
        identity = torch.arange(n)
        cycle_fwd = (identity + 1) % n
        cycle_back = (identity - 1) % n
        pair_swap = identity.clone()
        pair_swap[0::2], pair_swap[1::2] = identity[1::2], identity[0::2]
        # Interleave: primera mitad -> pares, segunda mitad -> impares.
        interleave = torch.empty(n, dtype=torch.long)
        half = n // 2
        interleave[:half] = torch.arange(0, n, 2)
        interleave[half:] = torch.arange(1, n, 2)
        perms = torch.stack([cycle_fwd, cycle_back, pair_swap, interleave])
        for p in perms:
            if not torch.equal(p.sort().values, identity):
                raise RuntimeError("se construyó una operación no biyectiva")
        return perms

    @property
    def mean_required_steps(self) -> float:
        return 0.5 * (self.min_steps + self.max_steps)

    def _balanced_lengths(self, batch_size: int) -> torch.Tensor:
        n_lengths = self.max_steps - self.min_steps + 1
        if batch_size % n_lengths:
            raise ValueError(
                f"batch_size={batch_size} debe ser múltiplo de {n_lengths} "
                "para conservar el presupuesto exacto")
        repeats = batch_size // n_lengths
        lengths = torch.arange(self.min_steps, self.max_steps + 1).repeat_interleave(repeats)
        return lengths[torch.randperm(batch_size, generator=self.gen)]

    def batch(self, batch_size: int = 36, balanced: bool = True) -> BudgetedStreamBatch:
        if balanced:
            lengths = self._balanced_lengths(batch_size)
        else:
            lengths = torch.randint(
                self.min_steps, self.max_steps + 1, (batch_size,), generator=self.gen)
        initial = torch.randint(0, self.n_states, (batch_size,), generator=self.gen)
        operations = torch.full(
            (batch_size, self.max_steps), self.PAD_OP, dtype=torch.long)
        intermediate = torch.empty(
            batch_size, self.max_steps, dtype=torch.long)
        state = initial.clone()
        rows = torch.arange(batch_size)
        for tick in range(self.max_steps):
            active = lengths > tick
            op = torch.randint(
                0, self.n_base_ops, (batch_size,), generator=self.gen)
            token = op.clone()
            if self.terminal_markers:
                last = lengths == tick + 1
                token[last] = token[last] + self.n_base_ops
            operations[active, tick] = token[active]
            if active.any():
                state[active] = self.permutations[
                    op[active], state[active]]
            intermediate[:, tick] = state
        final = state.clone()
        batch = BudgetedStreamBatch(
            initial_state=initial, operations=operations, lengths=lengths,
            final_state=final, intermediate_states=intermediate)
        self.validate(batch)
        return batch

    def validate(self, batch: BudgetedStreamBatch) -> None:
        B, N = batch.operations.shape
        if N != self.max_steps or batch.lengths.shape != (B,):
            raise ValueError("formas inválidas en BudgetedStreamBatch")
        rows = torch.arange(B)
        expected = batch.intermediate_states[rows, batch.lengths - 1]
        if not torch.equal(expected, batch.final_state):
            raise ValueError("final_state no coincide con el estado en K")
        for tick in range(N):
            after_end = batch.lengths <= tick
            if after_end.any() and not (batch.operations[after_end, tick] == self.PAD_OP).all():
                raise ValueError("hay operaciones reveladas después de K")
            if tick > 0:
                frozen = batch.lengths <= tick
                if frozen.any() and not torch.equal(
                        batch.intermediate_states[frozen, tick],
                    batch.intermediate_states[frozen, tick - 1]):
                    raise ValueError("el estado no se congeló después de K")
        if self.terminal_markers:
            terminal = ((batch.operations >= self.n_base_ops)
                        & (batch.operations < self.PAD_OP))
            if not torch.equal(terminal.sum(dim=1), torch.ones(B, dtype=torch.long)):
                raise ValueError("cada muestra debe contener un único marcador terminal")
            if not terminal[rows, batch.lengths - 1].all():
                raise ValueError("el marcador terminal no está en la operación K")
