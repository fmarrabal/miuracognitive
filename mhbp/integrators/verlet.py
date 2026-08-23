"""
mHBP — Integrador de REFERENCIA: Verlet amortiguado (compatibilidad con el HBP
original). Acoplamiento y giroscópico EXPLÍCITOS por diferencia atrasada —
condicionalmente estable (hereda la condición "genuinamente discreta" (i) de la
Prop. 2 del paper del HBP: el giroscópico atrasado inyecta energía √(1+μ̃²) por
paso). Sólo para comparación A/B con Cayley-IMEX; no usar en entrenamiento.
"""
from __future__ import annotations
import torch

from .cayley_imex import field_offsets, pack, unpack, _place_blockdiag


class VerletRef:
    """Misma API que CayleyIMEX (prepare/step/energy) sobre estado (U, W).
    Internamente: u⁻ = u − h·w;  u⁺ = 2u − u⁻ + h²(−𝓚u + F) − h(C+G)(u−u⁻);
    w⁺ = (u⁺ − u)/h."""

    def __init__(self):
        self._ready = False

    def prepare(self, fields, taus, coupling, dt, dtype=torch.float64, device=None):
        device = device or fields[0].L.device
        if len(taus) != len(fields):
            raise ValueError(f"prepare: len(taus)={len(taus)} != nº campos={len(fields)}")
        offs, n = field_offsets(fields)
        self.offsets, self.n_tot = offs, n
        C_blk = torch.zeros(n, n, dtype=dtype, device=device)
        K_blk = torch.zeros(n, n, dtype=dtype, device=device)
        G_blk = torch.zeros(n, n, dtype=dtype, device=device)
        hvec = torch.zeros(n, dtype=dtype, device=device)
        for f, off, tau in zip(fields, offs, taus):
            K, C, G = f.operators(dtype=dtype)
            N, d = f.cfg.n_nodes, f.cfg.d
            _place_blockdiag(C_blk, C, off, N, d)
            _place_blockdiag(K_blk, K, off, N, d)
            _place_blockdiag(G_blk, G, off, N, d)
            hvec[off:off + N * d] = dt / tau.to(dtype)
        Kglob = K_blk
        if coupling is not None:
            Kglob = Kglob + coupling.assemble_B(offs, n, [f.cfg for f in fields],
                                                dtype=dtype, device=device)
        self._K, self._CG = Kglob, C_blk + G_blk
        self._hvec = hvec
        self._fields = fields
        self._ready = True
        return self

    def step(self, U, W, F=None):
        assert self._ready
        h = self._hvec
        u_prev = U - h * W
        acc = -(U @ self._K.T)
        if F is not None:
            acc = acc + F
        Up = 2.0 * U - u_prev + (h ** 2) * acc - h * ((U - u_prev) @ self._CG.T)
        Wp = (Up - U) / h
        return Up, Wp

    def energy(self, U, W):
        pot = 0.5 * torch.einsum("bi,ij,bj->b", U, self._K, U)
        return 0.5 * (W ** 2).sum(-1) + pot
