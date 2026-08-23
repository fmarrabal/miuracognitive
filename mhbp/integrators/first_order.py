"""
mHBP — Integrador de PRIMER ORDEN (ablation C3 de la Fase 2: ¿importa el 2º orden?).

Dinámica sobreamortiguada:  C u̇ = −𝓚 u + F   (sin inercia, sin giroscópico:
G no tiene colocación giroscópica posible en 1er orden; entra posicional en 𝓚
si se quisiera — aquí se OMITE para el contraste limpio orden-puro).

Discretización backward-Euler (incondicionalmente estable):
    (C + h𝓚) u⁺ = C u + h F        [SPD: C≻0, 𝓚≻0]

Misma API que CayleyIMEX (prepare/step/energy); el estado W se acepta y se
devuelve a cero (los controladores no cambian).
"""
from __future__ import annotations
import torch

from .cayley_imex import field_offsets, _place_blockdiag


class FirstOrderIMEX:
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
        hvec = torch.zeros(n, dtype=dtype, device=device)
        for f, off, tau in zip(fields, offs, taus):
            K, C, G = f.operators(dtype=dtype)          # G se omite (orden puro)
            N, d = f.cfg.n_nodes, f.cfg.d
            _place_blockdiag(C_blk, C, off, N, d)
            _place_blockdiag(K_blk, K, off, N, d)
            hvec[off:off + N * d] = dt / tau.to(dtype)
        Kglob = K_blk
        if coupling is not None:
            Kglob = Kglob + coupling.assemble_B(offs, n, [f.cfg for f in fields],
                                                dtype=dtype, device=device)
        # (C + H𝓚) u⁺ = C u + H F   — H𝓚 no es simétrica con τ distintas; el
        # sistema equivalente simétrico: premultiplicar por nada — resolvemos el
        # sistema general con LU (matriz fija por prepare, factorizable una vez).
        A = C_blk + hvec.unsqueeze(1) * Kglob
        self._A_lu = torch.linalg.lu_factor(A)
        self._C = C_blk
        self._hvec = hvec
        self._Kglob = Kglob
        self._fields = fields
        self._ready = True
        return self

    def step(self, U, W, F=None):
        assert self._ready
        rhs = U @ self._C.T
        if F is not None:
            rhs = rhs + F * self._hvec
        Up = torch.linalg.lu_solve(*self._A_lu, rhs.unsqueeze(-1)).squeeze(-1)
        return Up, torch.zeros_like(W)

    def energy(self, U, W):
        pot = 0.5 * torch.einsum("bi,ij,bj->b", U, self._Kglob, U)
        return pot                                      # sin término cinético
