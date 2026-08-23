"""
mHBP — Integrador principal Cayley-IMEX (§7 de MATH_SPEC.md).

Splitting de Strang por tick global Δt (paso local h_q = Δt/τ_q):

  (1) w_q ← Q_q(h_q/2) · w_q          [Cayley: rotación giroscópica EXACTAMENTE
                                       ortogonal — no inyecta energía discreta]
  (2) θ-implícito GLOBAL (θ=1 BE / θ=½ CN) sobre la parte simétrica CON el
      acoplamiento dentro del núcleo:
        A_θ W⁺ = R_w W − H·𝓚·U + H·F,   A_θ = I + θ·H·C + θ²·H·𝓚·H  (SPD)
        U⁺ = U + H·(θ W⁺ + (1−θ) W)
  (3) w_q ← Q_q(h_q/2) · w_q

𝓚 = blockdiag(K_qℓ) + 𝓑_acoplamiento ≻ 0; H = diag(h_q I). A_θ es SPD por
construcción (congruencias) ⇒ Cholesky. Todo diferenciable (autograd a través
de cholesky_solve y de las solves de Cayley).

Layout global: índice plano (campo q, nodo i, dim ℓ) = off_q + i·d + ℓ.
"""
from __future__ import annotations
import torch


def field_offsets(fields) -> tuple[list[int], int]:
    """Offsets planos por campo y tamaño total n_tot = Σ N_q·d."""
    offs, n = [], 0
    for f in fields:
        offs.append(n)
        n += f.cfg.n_nodes * f.cfg.d
    return offs, n


def pack(states: list[torch.Tensor]) -> torch.Tensor:
    """[(B,N_q,d)] -> (B, n_tot) respetando el layout (nodo, dim)."""
    return torch.cat([s.reshape(s.shape[0], -1) for s in states], dim=1)


def unpack(flat: torch.Tensor, fields) -> list[torch.Tensor]:
    """(B, n_tot) -> [(B,N_q,d)]."""
    out, k = [], 0
    for f in fields:
        n = f.cfg.n_nodes * f.cfg.d
        out.append(flat[:, k:k + n].reshape(-1, f.cfg.n_nodes, f.cfg.d))
        k += n
    return out


def _place_blockdiag(M: torch.Tensor, block: torch.Tensor, off: int, N: int, d: int):
    """Coloca block (d,N,N) por-dimensión en la matriz global M con el layout
    idx = off + i·d + ℓ (in-place)."""
    for l in range(d):
        idx = off + torch.arange(N, device=M.device) * d + l
        M[idx.unsqueeze(1), idx.unsqueeze(0)] += block[l]


class CayleyIMEX:
    """Integrador principal. `prepare()` una vez por forward (ensambla y factoriza);
    `step()` por tick. theta=1.0 (BE, disipativo, por defecto) o 0.5 (CN, orden 2)."""

    def __init__(self, theta: float = 1.0):
        assert 0.5 <= theta <= 1.0
        self.theta = float(theta)
        self._ready = False

    # ------------------------------------------------------------------ #
    def prepare(self, fields, taus: torch.Tensor, coupling, dt: float,
                dtype=torch.float64, device=None):
        """Ensambla C_blk, 𝓚 = K_blk + 𝓑, H y factoriza A_θ. `coupling` puede ser
        None o un módulo con assemble_B(offsets, n_tot, dtype, device) -> (n,n) PSD."""
        device = device or fields[0].L.device
        if len(taus) != len(fields):
            raise ValueError(f"prepare: len(taus)={len(taus)} != nº campos={len(fields)} "
                             "(un zip truncado congelaría campos en silencio)")
        offs, n = field_offsets(fields)
        self.offsets, self.n_tot = offs, n
        th = self.theta

        C_blk = torch.zeros(n, n, dtype=dtype, device=device)
        K_blk = torch.zeros(n, n, dtype=dtype, device=device)
        hvec = torch.zeros(n, dtype=dtype, device=device)
        self._cay = []                      # matrices de Cayley (d,N,N) por campo
        self.antiresonance_margin = float("inf")   # min_q (1 − h_q·σ_max(G_q)/4)
        for f, off, tau in zip(fields, offs, taus):
            K, C, G = f.operators(dtype=dtype)
            N, d = f.cfg.n_nodes, f.cfg.d
            _place_blockdiag(C_blk, C, off, N, d)
            _place_blockdiag(K_blk, K, off, N, d)
            h_q = dt / tau.to(dtype)
            hvec[off:off + N * d] = h_q
            # --- condición ANTI-RESONANCIA (hallazgo crítico del ataque) ---
            # La media rotación de Cayley gira 2·atan(h_q·μ/4) por modo; si llega
            # a π/2 (h_q·σ_max(G)/4 = 1), Q² tiene autovalor −1 que compuesto con
            # el modo flip de Crank–Nicolson (θ=½, ciego al punto medio) da un
            # modo NEUTRO exacto (ρ=1) DENTRO de las cajas. Con θ=1 el BE lo
            # amortigua (margen medido ≥2e−4 en 840 configs adversarias). Para
            # θ<1 se EXIGE rotación estrictamente < π/2 con holgura.
            with torch.no_grad():
                sig_max = float(torch.linalg.svdvals(G.detach()).max()) if (G != 0).any() else 0.0
            marg = 1.0 - float(h_q) * sig_max / 4.0
            self.antiresonance_margin = min(self.antiresonance_margin, marg)
            if th < 1.0 and marg <= 0.05:
                raise ValueError(
                    f"CayleyIMEX θ={th}: condición anti-resonancia violada en el campo "
                    f"'{f.cfg.name}' (h_q·σ_max(G)/4 = {1 - marg:.3f} ≥ 0.95). Con θ<1 la "
                    f"rotación de Cayley cerca de π/2 produce un modo neutro (ρ=1). "
                    f"Usa θ=1, reduce dt, o baja b_max/beta_max (MATH_SPEC §7).")
            # Cayley de media rotación: Q = (I + (h/4)G)⁻¹(I − (h/4)G)
            a = h_q / 2.0
            I = torch.eye(N, dtype=dtype, device=device).expand(d, N, N)
            self._cay.append(torch.linalg.solve(I + (a / 2.0) * G, I - (a / 2.0) * G))

        Kglob = K_blk
        if coupling is not None:
            Kglob = Kglob + coupling.assemble_B(offs, n, [f.cfg for f in fields],
                                                dtype=dtype, device=device)
        HC = hvec.unsqueeze(1) * C_blk                       # = diag(h_q C_q), simétrica
        HKH = hvec.unsqueeze(1) * Kglob * hvec.unsqueeze(0)  # congruencia, simétrica PSD
        self._HK = hvec.unsqueeze(1) * Kglob                 # para el RHS (H·𝓚·U)
        self._HKH = HKH
        self._HC = HC
        self._hvec = hvec
        I_n = torch.eye(n, dtype=dtype, device=device)
        A_th = I_n + th * HC + (th ** 2) * HKH               # SPD
        self._A_chol = torch.linalg.cholesky(A_th)
        # matriz del RHS para θ<1: R_w = I − (1−θ)HC − θ(1−θ)H𝓚H
        if th < 1.0:
            self._Rw = I_n - (1.0 - th) * HC - th * (1.0 - th) * HKH
        else:
            self._Rw = None
        self._Kglob = Kglob                                  # para energía/certificados
        self._fields = fields
        self._ready = True
        return self

    # ------------------------------------------------------------------ #
    def _apply_cayley(self, W_flat: torch.Tensor) -> torch.Tensor:
        """Media rotación giroscópica por campo/dimensión sobre W (B, n_tot)."""
        Ws = unpack(W_flat, self._fields)
        out = []
        for w, Q in zip(Ws, self._cay):
            out.append(torch.einsum("dij,bjd->bid", Q, w))
        return pack(out)

    def step(self, U: torch.Tensor, W: torch.Tensor,
             F: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Un tick global. U, W, F: (B, n_tot). Devuelve (U⁺, W⁺)."""
        assert self._ready, "llama a prepare() antes de step()"
        th = self.theta
        W1 = self._apply_cayley(W)                                    # (1)
        rhs = (W1 if self._Rw is None else W1 @ self._Rw.T)           # R_w · W
        rhs = rhs - U @ self._HK.T                                    # − H𝓚U
        if F is not None:
            rhs = rhs + F * self._hvec                                # + H F
        Wp = torch.cholesky_solve(rhs.unsqueeze(-1), self._A_chol).squeeze(-1)  # (2)
        Up = U + (th * Wp + (1.0 - th) * W1) * self._hvec
        Wp = self._apply_cayley(Wp)                                   # (3)
        return Up, Wp

    # ------------------------------------------------------------------ #
    def energy(self, U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Energía global E = ½‖W‖² + ½ Uᵀ𝓚U (incluye acoplamiento). (B,)"""
        pot = 0.5 * torch.einsum("bi,ij,bj->b", U, self._Kglob, U)
        return 0.5 * (W ** 2).sum(-1) + pot

    def cayley_orthogonality_error(self) -> float:
        """max_q ‖QᵀQ − I‖_∞ — debe ser ~1e−15 (test de aceptación)."""
        err = 0.0
        for Q in self._cay:
            I = torch.eye(Q.shape[-1], dtype=Q.dtype, device=Q.device)
            err = max(err, float((Q.transpose(-1, -2) @ Q - I).abs().max()))
        return err
