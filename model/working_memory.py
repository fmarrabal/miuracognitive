"""
MiuraCognitive - Memoria de Trabajo Explícita (Working Memory)
===============================================================
Tensor de memoria externo de M slots × d_model que el modelo lee y escribe
ACTIVAMENTE (a diferencia del KV-cache, que es pasivo).

Los gates de escritura/olvido están MODULADOS por el HBP: el estado
emocional sesga qué se recuerda (lo emocionalmente cargado se retiene más,
imitando el sesgo de memoria emocional humano).

Inspiración: Neural Turing Machines (Graves 2014) con estabilización moderna.

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkingMemory(nn.Module):
    """Memoria de M slots. Operaciones diferenciables de lectura y escritura
    mediante atención por contenido."""

    def __init__(self, d_model: int, n_slots: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots

        # Proyecciones para lectura (query desde el estado actual)
        self.read_query = nn.Linear(d_model, d_model)
        # Proyecciones para escritura
        self.write_key = nn.Linear(d_model, d_model)
        self.write_value = nn.Linear(d_model, d_model)
        # Gates base (se combinan con la modulación del HBP)
        self.gate_write = nn.Linear(d_model, 1)
        self.gate_forget = nn.Linear(d_model, 1)

    def init_memory(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Memoria inicial vacía: (B, n_slots, d_model)."""
        return torch.zeros(batch_size, self.n_slots, self.d_model,
                           device=device, dtype=dtype)

    def read(self, memory: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Lee de memoria por atención de contenido.
        memory: (B, M, d), state: (B, d) -> read: (B, d)."""
        q = self.read_query(state)                         # (B, d)
        # Scores de atención sobre slots
        scores = torch.einsum("bd,bmd->bm", q, memory)     # (B, M)
        scores = scores / (self.d_model ** 0.5)
        attn = F.softmax(scores, dim=-1)                   # (B, M)
        read = torch.einsum("bm,bmd->bd", attn, memory)    # (B, d)
        return read

    def write(self, memory: torch.Tensor, state: torch.Tensor,
              hbp_write: torch.Tensor | None = None,
              hbp_forget: torch.Tensor | None = None) -> torch.Tensor:
        """Escribe en memoria con gates modulados por el HBP.
        memory: (B, M, d), state: (B, d).
        hbp_write/forget: (d,) o (B, d) modulación opcional del HBP."""
        B = state.shape[0]
        key = self.write_key(state)                        # (B, d)
        value = self.write_value(state)                    # (B, d)

        # Dónde escribir: atención por contenido entre key y slots actuales
        scores = torch.einsum("bd,bmd->bm", key, memory)   # (B, M)
        scores = scores / (self.d_model ** 0.5)
        write_addr = F.softmax(scores, dim=-1)             # (B, M)

        # Gates base (escalares por batch)
        g_w = torch.sigmoid(self.gate_write(state))        # (B, 1)
        g_f = torch.sigmoid(self.gate_forget(state))       # (B, 1)

        # Modulación del HBP: combina con los gates base.
        # Reducimos los gates vectoriales del HBP a escalar por media.
        if hbp_write is not None:
            mod_w = hbp_write.mean(dim=-1, keepdim=True)   # (1,) o (B,1)
            if mod_w.dim() == 1:
                mod_w = mod_w.view(1, 1)
            g_w = g_w * mod_w
        if hbp_forget is not None:
            mod_f = hbp_forget.mean(dim=-1, keepdim=True)
            if mod_f.dim() == 1:
                mod_f = mod_f.view(1, 1)
            g_f = g_f * mod_f

        # Actualización por slot:
        #   mem' = mem·(1 - addr·g_f) + (addr·g_w)·value
        addr = write_addr.unsqueeze(-1)                    # (B, M, 1)
        forget_term = memory * (1.0 - addr * g_f.unsqueeze(1))
        write_term = addr * g_w.unsqueeze(1) * value.unsqueeze(1)
        new_memory = forget_term + write_term
        return new_memory

    def occupancy(self, memory: torch.Tensor) -> torch.Tensor:
        """Mide cuán 'llena' está la memoria (norma media de slots).
        Útil como señal de interocepción para el HBP."""
        return memory.norm(dim=-1).mean()
