"""
MiuraCognitive — Simulador del campo homeostático a ESCALA GIGANTE
==================================================================
Reproduce la MISMA física de `model/hbp.py` (familia PDE: onda amortiguada,
difusión, advección b·A, dispersión β·A³, no linealidad KdV ν) pero con
operadores por `roll`/shift en O(N) para topologías estructuradas (cadena,
anillo, malla 2D toroidal) y sparse CSR para grafos aleatorios. Esto permite
N de 6 → 10⁷ en la GPU Blackwell.

En el ANILLO, los operadores L, A, A³ son circulantes → diagonalizan por DFT →
(i) relación de dispersión ANALÍTICA y (ii) integrador pseudo-espectral (split-
step Fourier, gold-standard para KdV) que integra la parte lineal EXACTA. Ahí
viven los solitones que a N=6 no caben.

Convención: el campo es (..., N), eje de nodos = -1. Física en FP64 por defecto.

Autor: Francisco M. Arrabal (Curro) + asistencia Claude.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import torch


# ----------------------------------------------------------------------------- #
# Utilidades de dispositivo / dtype
# ----------------------------------------------------------------------------- #
def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


F64 = torch.float64
F32 = torch.float32


# ============================================================================= #
# OPERADORES DEL GRAFO — matvec L(x), A(x), A³(x) en O(N) por roll/slicing.
# Reproducen build_chain_laplacian / build_chain_advection de model/hbp.py.
# ============================================================================= #
class Ops:
    """Interfaz común. x tiene forma (..., N); el eje de nodos es -1."""
    kind: str
    N: int
    # radios espectrales (para certificados)
    rho_L: float
    rho_A: float
    rho_A3: float
    fourier: bool = False   # True si L,A diagonalizan por DFT (anillo/toro)

    def L(self, x): raise NotImplementedError
    def A(self, x): raise NotImplementedError
    def A3(self, x):
        return self.A(self.A(self.A(x)))


class RingOps(Ops):
    """Ciclo C_N (periódico). L y A circulantes → DFT las diagonaliza.
    Autovalores: λ_L(k)=4sin²(θ/2)  con θ=2πk/N ; A: 2i·sin(θ) ; A³: -8i·sin³(θ).
    ρ(L)=4, ρ(A)=2, ρ(A³)=8 (independientes de N)."""
    kind = "ring"

    def __init__(self, N: int):
        self.N = N
        self.fourier = True
        # radios analíticos (exactos para N par; ~exactos para impar)
        self.rho_L = 4.0
        self.rho_A = 2.0 * math.sin(math.pi * (N // 2) / N) if N % 2 else 2.0
        self.rho_A3 = self.rho_A ** 3

    def L(self, x):
        return 2.0 * x - torch.roll(x, 1, -1) - torch.roll(x, -1, -1)

    def A(self, x):
        # (A x)[i] = x[i+1] - x[i-1]
        return torch.roll(x, -1, -1) - torch.roll(x, 1, -1)

    def fourier_symbols(self, device, dtype=F64):
        """Símbolos de Fourier por modo k=0..N-1 (tensores complejos)."""
        k = torch.arange(self.N, device=device, dtype=dtype)
        th = 2.0 * math.pi * k / self.N
        lamL = 4.0 * torch.sin(th / 2.0) ** 2                 # real >=0
        muA = 2.0 * torch.sin(th)                             # A = i*muA
        muA3 = -(muA ** 3)                                    # A³ = i*muA3  (=(i muA)³ = -i muA³)
        return th, lamL, muA, muA3


class ChainOps(Ops):
    """Camino P_N (bordes). Reproduce EXACTAMENTE model/hbp.py a cualquier N."""
    kind = "chain"

    def __init__(self, N: int):
        self.N = N
        self.fourier = False
        # ρ(L) del path: λ_max = 2 - 2cos((N-1)π/N)  (autovalores 2-2cos(kπ/N))
        self.rho_L = 2.0 - 2.0 * math.cos((N - 1) * math.pi / N) if N > 1 else 0.0
        # A del path: autovalores ±i·2cos(kπ/(N+1))? -> radio ~ 2cos(π/(N+1))
        self.rho_A = 2.0 * math.cos(math.pi / (N + 1)) if N > 1 else 0.0
        self.rho_A3 = self.rho_A ** 3

    def L(self, x):
        Lx = torch.empty_like(x)
        Lx[..., 1:-1] = 2.0 * x[..., 1:-1] - x[..., :-2] - x[..., 2:]
        Lx[..., 0] = x[..., 0] - x[..., 1]
        Lx[..., -1] = x[..., -1] - x[..., -2]
        return Lx

    def A(self, x):
        Ax = torch.empty_like(x)
        Ax[..., 1:-1] = x[..., 2:] - x[..., :-2]
        Ax[..., 0] = x[..., 1]
        Ax[..., -1] = -x[..., -2]
        return Ax


class Grid2DOps(Ops):
    """Malla 2D TOROIDAL m×m (periódica en ambos ejes). L = L_x + L_y (5 puntos);
    A = advección en x. Diagonaliza por DFT-2D. N = m*m."""
    kind = "grid2d"

    def __init__(self, m: int):
        self.m = m
        self.N = m * m
        self.fourier = True
        self.rho_L = 8.0        # 4 por eje
        self.rho_A = 2.0
        self.rho_A3 = 8.0

    def _r(self, x):
        return x.reshape(*x.shape[:-1], self.m, self.m)

    def L(self, x):
        g = self._r(x)
        out = (4.0 * g
               - torch.roll(g, 1, -1) - torch.roll(g, -1, -1)
               - torch.roll(g, 1, -2) - torch.roll(g, -1, -2))
        return out.reshape(*x.shape)

    def A(self, x):
        g = self._r(x)
        out = torch.roll(g, -1, -1) - torch.roll(g, 1, -1)     # transporte en x
        return out.reshape(*x.shape)


class SparseOps(Ops):
    """Grafo arbitrario (regular-aleatorio, small-world). L y A sparse CSR.
    A antisimétrica: cada arista (i<j) orientada +1 (i->j) / -1 (j->i)."""

    def __init__(self, kind, N, edges, device, dtype=F64):
        self.kind = kind
        self.N = N
        self.fourier = False
        rows = torch.tensor([e[0] for e in edges] + [e[1] for e in edges], device=device)
        cols = torch.tensor([e[1] for e in edges] + [e[0] for e in edges], device=device)
        # Laplaciano combinatorio L = D - Adj
        deg = torch.zeros(N, device=device, dtype=dtype).index_add_(
            0, rows, torch.ones(rows.numel(), device=device, dtype=dtype))
        adj_vals = torch.ones(rows.numel(), device=device, dtype=dtype)
        Adj = torch.sparse_coo_tensor(torch.stack([rows, cols]), -adj_vals, (N, N))
        Dg = torch.sparse_coo_tensor(
            torch.stack([torch.arange(N, device=device), torch.arange(N, device=device)]),
            deg, (N, N))
        self._L = (Dg + Adj).coalesce().to_sparse_csr()
        # Advección antisimétrica orientada
        e_i = torch.tensor([e[0] for e in edges], device=device)
        e_j = torch.tensor([e[1] for e in edges], device=device)
        a_rows = torch.cat([e_i, e_j]); a_cols = torch.cat([e_j, e_i])
        a_vals = torch.cat([torch.ones(e_i.numel(), device=device, dtype=dtype),
                            -torch.ones(e_i.numel(), device=device, dtype=dtype)])
        self._A = torch.sparse_coo_tensor(torch.stack([a_rows, a_cols]), a_vals, (N, N)).coalesce().to_sparse_csr()
        # radios por power-iteration (matvec)
        self.rho_L = float(_power_iter_sym(self._L, N, device, dtype))
        self.rho_A = float(_power_iter_norm(self._A, N, device, dtype))
        self.rho_A3 = self.rho_A ** 3

    def L(self, x):
        sh = x.shape
        flat = x.reshape(-1, self.N).t()            # (N, B)
        out = torch.sparse.mm(self._L, flat).t()
        return out.reshape(sh)

    def A(self, x):
        sh = x.shape
        flat = x.reshape(-1, self.N).t()
        out = torch.sparse.mm(self._A, flat).t()
        return out.reshape(sh)


def _power_iter_sym(M_csr, N, device, dtype, iters=200):
    v = torch.randn(N, 1, device=device, dtype=dtype); v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        w = torch.sparse.mm(M_csr, v)
        lam = float(w.norm())
        v = w / (w.norm() + 1e-30)
    return lam


def _power_iter_norm(A_csr, N, device, dtype, iters=200):
    # ρ de A antisimétrica = mayor |valor singular| = sqrt(ρ(AᵀA)) = sqrt(ρ(-A²))
    v = torch.randn(N, 1, device=device, dtype=dtype); v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        w = torch.sparse.mm(A_csr, v)
        w = torch.sparse.mm(A_csr, w)      # A²v  (=-AᵀA v)
        lam = float(w.norm())
        v = w / (w.norm() + 1e-30)
    return math.sqrt(lam)


def build_random_regular(N, d, seed):
    """Grafo d-regular aleatorio (modelo de emparejamiento / configuración)."""
    g = torch.Generator().manual_seed(seed)
    stubs = torch.arange(N).repeat_interleave(d)
    for _try in range(200):
        perm = stubs[torch.randperm(stubs.numel(), generator=g)]
        a, b = perm[0::2], perm[1::2]
        edges = set()
        ok = True
        for u, v in zip(a.tolist(), b.tolist()):
            if u == v or (u, v) in edges or (v, u) in edges:
                ok = False; break
            edges.add((min(u, v), max(u, v)))
        if ok:
            return list(edges)
    return list(edges)   # aprox si no perfecto


def build_watts_strogatz(N, k, p, seed):
    g = torch.Generator().manual_seed(seed)
    edges = set()
    for i in range(N):
        for j in range(1, k // 2 + 1):
            edges.add((i, (i + j) % N))
    edges = set((min(a, b), max(a, b)) for a, b in edges)
    edges = list(edges)
    out = set()
    for (a, b) in edges:
        if torch.rand(1, generator=g).item() < p:
            nn = int(torch.randint(0, N, (1,), generator=g).item())
            if nn != a and (min(a, nn), max(a, nn)) not in out:
                out.add((min(a, nn), max(a, nn)))
                continue
        out.add((min(a, b), max(a, b)))
    return list(out)


def build_ops(kind, N, device, dtype=F64, seed=0, **kw):
    if kind == "ring":
        return RingOps(N)
    if kind == "chain":
        return ChainOps(N)
    if kind == "grid2d":
        m = int(round(math.sqrt(N)))
        return Grid2DOps(m)
    if kind == "random_regular":
        d = kw.get("degree", 4)
        return SparseOps(kind, N, build_random_regular(N, d, seed), device, dtype)
    if kind == "ws_smallworld":
        return SparseOps(kind, N, build_watts_strogatz(N, kw.get("k", 4), kw.get("p", 0.1), seed), device, dtype)
    if kind == "expander":
        return SparseOps(kind, N, build_random_regular(N, kw.get("degree", 6), seed), device, dtype)
    raise ValueError(kind)


# ============================================================================= #
# PARÁMETROS FÍSICOS (mismos rangos que HBPConfig)
# ============================================================================= #
@dataclass
class Phys:
    omega0: float = 0.5
    zeta: float = 0.5
    c: float = 0.4
    D: float = 0.0          # difusión estructural
    b: float = 0.0          # advección
    beta: float = 0.0       # dispersión KdV (β·A³)
    nu: float = 0.0         # no linealidad KdV
    gamma: float = 1.0      # tasa de la rama difusiva
    dt: float = 1.0
    nonlin: str = "saturated"   # "saturated" (modelo) | "genuine" (u·u_x de KdV)


# ============================================================================= #
# RAMA ONDA (Verlet, 2º orden) — reproduce hbp.py.step() order=2
# ============================================================================= #
def wave_step(ops: Ops, p: Phys, ht, htm1, hs=0.0, placement="gyro"):
    """Un paso de Verlet de la onda amortiguada + operadores antisimétricos.
    placement='gyro' (correcto, 2º orden) | 'circulatory' (posicional -> flutter)."""
    dt = p.dt
    v = ht - htm1
    u = ht - hs
    restit = -(p.omega0 ** 2) * u
    spatial = -(p.c ** 2) * ops.L(ht)
    nladv = 0.0
    if p.nu != 0.0:
        Au = ops.A(u)
        nladv = -p.nu * (torch.tanh(u) * torch.tanh(Au) if p.nonlin == "saturated" else u * Au)
    rhs = restit + spatial + nladv
    damp = -(2.0 * p.zeta * p.omega0 / dt) * v
    if p.D != 0.0:
        damp = damp - (p.D / dt) * ops.L(v)
    anti = 0.0
    if placement == "gyro":
        if p.b != 0.0:
            anti = anti - (p.b / dt) * ops.A(v)
        if p.beta != 0.0:
            anti = anti - (p.beta / dt) * ops.A3(v)
    else:  # circulatory: posicional sobre u (fuerza no conservativa -> flutter)
        if p.b != 0.0:
            anti = anti - p.b * ops.A(u)
        if p.beta != 0.0:
            anti = anti - p.beta * ops.A3(u)
    h_next = 2.0 * ht - htm1 + (dt ** 2) * (rhs + damp + anti)
    return h_next


def run_wave(ops, p, u0, ticks, hs=0.0, placement="gyro", record_every=0, clamp=None):
    """Rollout de la onda. Devuelve trazas de energía/norma y opcionalmente frames.
    u0: (..., N) estado inicial (desviación de hs). Velocidad inicial nula."""
    ht = u0.clone(); htm1 = u0.clone()
    norms, energies = [], []
    frames = []
    for t in range(ticks):
        hn = wave_step(ops, p, ht, htm1, hs=hs, placement=placement)
        if clamp is not None:
            hn = torch.clamp(hn, -clamp, clamp)
        htm1, ht = ht, hn
        n = float((ht - hs).norm())
        norms.append(n)
        # energía E = ½||v||² + ½ uᵀK u  (K = ω₀²I + c²L); v FÍSICA = (ht-htm1)/dt
        vv = (ht - htm1) / p.dt
        uu = ht - hs
        Ku = (p.omega0 ** 2) * uu + (p.c ** 2) * ops.L(uu)
        E = 0.5 * float((vv * vv).sum()) + 0.5 * float((uu * Ku).sum())
        energies.append(E)
        if not math.isfinite(n):
            break
        if record_every and (t % record_every == 0):
            frames.append((ht - hs).detach().to("cpu"))
    return {"norms": norms, "energies": energies, "frames": frames, "final": ht}


# ============================================================================= #
# SOLVER KdV PSEUDO-ESPECTRAL (ANILLO) — split-step Fourier, parte lineal EXACTA
# ============================================================================= #
def kdv_ring_dispersion(N, p: Phys, device, dtype=F64):
    """Símbolo lineal Ω(k) de la rama de 1er orden en el anillo:
       γ u̇ = -(b·A + β·A³)u  →  û̇ = -i·Ω(k)·û,  Ω(k)=(b·2sinθ - β·8sin³θ)/γ.
    Devuelve Ω(k) (real, k=0..N-1)."""
    k = torch.arange(N, device=device, dtype=dtype)
    th = 2.0 * math.pi * k / N
    s = torch.sin(th)
    muA = 2.0 * s
    muA3 = -(muA ** 3)              # A³ = i·muA3
    # -(bA+βA³)u en Fourier = -(b·i·muA + β·i·muA3)û = -i(b·muA+β·muA3)û
    Omega = (p.b * muA + p.beta * muA3) / p.gamma
    return Omega


def run_kdv_ring(N, p: Phys, u0, ticks, device, dtype=F64, record_every=0,
                 damp_omega0=0.0, damp_zeta=0.0, damp_c=0.0):
    """Integra la rama de PRIMER orden en el anillo por split-step Fourier (Strang).
      γ u̇ = -(b·A+β·A³)u + r(u) - (ω₀²+c²L)u·[si damp]  , r=no linealidad.
    Parte lineal (advección+dispersión[+restitución/difusión opcional]) EXACTA en
    Fourier; no linealidad por medio paso explícito (Strang). u0: (..., N)."""
    u = u0.clone().to(device=device, dtype=dtype)
    Omega = kdv_ring_dispersion(N, p, device, dtype)          # (N,) real
    # amortiguamiento opcional (rompe la conservación → mata solitones): decaimiento
    # lineal γ u̇ ⊃ -(ω₀² + c²λ_L)u  → factor real exp(-(ω₀²+c²λ_L)/γ · dt)
    k = torch.arange(N, device=device, dtype=dtype)
    th = 2.0 * math.pi * k / N
    lamL = 4.0 * torch.sin(th / 2.0) ** 2
    decay = (damp_omega0 ** 2 + damp_c ** 2 * lamL) / p.gamma
    lin_phase = torch.exp(torch.complex(-decay * p.dt, -Omega * p.dt))   # e^{-(decay+iΩ)dt}
    half = torch.exp(torch.complex(-decay * p.dt / 2, -Omega * p.dt / 2))

    def nonlin(uu):
        if p.nu == 0.0:
            return torch.zeros_like(uu)
        Au = torch.roll(uu, -1, -1) - torch.roll(uu, 1, -1)   # A en anillo
        r = (torch.tanh(uu) * torch.tanh(Au) if p.nonlin == "saturated" else uu * Au)
        return -(p.nu / p.gamma) * r

    norms, mass, frames = [], [], []
    for t in range(ticks):
        # Strang: ½ no lineal (explícito RK2) — ½ lineal exacto — repite
        uh = torch.fft.fft(u)
        uh = uh * half
        u = torch.fft.ifft(uh).real
        # paso no lineal completo (RK2 de dt sobre r)
        k1 = nonlin(u)
        k2 = nonlin(u + p.dt * k1)
        u = u + 0.5 * p.dt * (k1 + k2)
        uh = torch.fft.fft(u)
        uh = uh * half
        u = torch.fft.ifft(uh).real
        norms.append(float(u.norm()))
        mass.append(float(u.sum(-1).abs().mean()))
        if not math.isfinite(norms[-1]):
            break
        if record_every and (t % record_every == 0):
            frames.append(u.detach().to("cpu"))
    return {"norms": norms, "mass": mass, "frames": frames, "final": u, "Omega": Omega.to("cpu")}


# ============================================================================= #
# DETECCIÓN DE SOLITONES
# ============================================================================= #
def detect_peaks(u_row, thresh_frac=0.3):
    """Picos locales de un perfil 1D (tensor 1D en CPU) por encima de thresh_frac*max."""
    u = u_row
    N = u.numel()
    mx = float(u.max())
    if mx <= 0:
        return []
    thr = thresh_frac * mx
    left = torch.roll(u, 1); right = torch.roll(u, -1)
    ispeak = (u > left) & (u >= right) & (u > thr)
    idx = torch.nonzero(ispeak).flatten().tolist()
    return idx


def fwhm(u_row, peak_idx):
    """Ancho a media altura alrededor de un pico (en nº de nodos, con wrap)."""
    u = u_row
    N = u.numel()
    h = float(u[peak_idx]) / 2.0
    # expandir a izquierda/derecha hasta bajar de h
    l = 0
    while l < N and float(u[(peak_idx - l) % N]) > h:
        l += 1
    r = 0
    while r < N and float(u[(peak_idx + r) % N]) > h:
        r += 1
    return l + r


def track_soliton(frames, dt_frames, box):
    """Sigue el pico dominante a lo largo de frames (lista de tensores 1D CPU) y
    estima velocidad (nodos/tick), estabilidad de amplitud y de ancho.
    box: N (para wrap de posición). Devuelve métricas agregadas."""
    if len(frames) < 3:
        return None
    N = box
    positions, amps, widths = [], [], []
    for fr in frames:
        pk = detect_peaks(fr)
        if not pk:
            positions.append(None); amps.append(None); widths.append(None); continue
        # pico dominante
        best = max(pk, key=lambda i: float(fr[i]))
        positions.append(best)
        amps.append(float(fr[best]))
        widths.append(fwhm(fr, best))
    # velocidad: desenrollar posiciones (unwrap circular)
    valid = [(i, positions[i]) for i in range(len(positions)) if positions[i] is not None]
    if len(valid) < 3:
        return {"n_frames_with_peak": len(valid), "velocity": None,
                "amp_cv": None, "width_cv": None, "life_frac": len(valid) / len(frames)}
    xs = [p for _, p in valid]
    unwrapped = [xs[0]]
    for j in range(1, len(xs)):
        d = xs[j] - xs[j - 1]
        if d > N / 2: d -= N
        if d < -N / 2: d += N
        unwrapped.append(unwrapped[-1] + d)
    ts = [i for i, _ in valid]
    # regresión lineal pos vs tick-frame
    import statistics
    n = len(ts)
    mt = sum(ts) / n; mx = sum(unwrapped) / n
    num = sum((ts[i] - mt) * (unwrapped[i] - mx) for i in range(n))
    den = sum((ts[i] - mt) ** 2 for i in range(n)) + 1e-30
    slope = num / den                       # nodos por frame
    vel = slope / dt_frames                 # nodos por tick
    av = [a for a in amps if a is not None]
    wv = [w for w in widths if w is not None]
    amp_cv = (statistics.pstdev(av) / (statistics.mean(av) + 1e-30)) if len(av) > 1 else None
    width_cv = (statistics.pstdev(wv) / (statistics.mean(wv) + 1e-30)) if len(wv) > 1 else None
    return {"n_frames_with_peak": len(valid), "velocity_nodes_per_tick": vel,
            "amp_mean": statistics.mean(av) if av else None,
            "amp_cv": amp_cv, "width_mean": statistics.mean(wv) if wv else None,
            "width_cv": width_cv, "life_frac": len(valid) / len(frames)}


# ============================================================================= #
# CERTIFICADOS / ESTABILIDAD
# ============================================================================= #
def flutter_threshold(p: Phys, ops: Ops):
    """Umbral de Merkin: colocación circulatoria estable sii β·ρ(A³) < 2ζω₀².
    Devuelve (lhs, rhs, estable_por_certificado)."""
    lhs = p.beta * ops.rho_A3
    rhs = 2.0 * p.zeta * p.omega0 ** 2
    return lhs, rhs, lhs < rhs


def companion_spectral_radius(ops_small: Ops, p: Phys, device, dtype=F64, placement="gyro"):
    """Radio espectral EXACTO de la companion 2N×2N (solo N moderado). Verifica el
    certificado analítico contra las raíces reales del mapa de Verlet."""
    N = ops_small.N
    I = torch.eye(N, device=device, dtype=dtype)
    # materializa L, A, A³ aplicando los operadores a la base canónica
    Lm = ops_small.L(I); Am = ops_small.A(I); A3m = ops_small.A3(I)
    dt = p.dt
    K = (dt ** 2) * (p.omega0 ** 2 * I + p.c ** 2 * Lm)
    C = dt * (2.0 * p.zeta * p.omega0 * I + p.D * Lm)
    if placement == "gyro":
        G = dt * (p.b * Am + p.beta * A3m)
        top_left = 2 * I - K - C - G
        top_right = -(I - C - G)
    else:
        Gpos = (dt ** 2) * (p.b * Am + p.beta * A3m)
        top_left = 2 * I - K - C - Gpos
        top_right = -(I - C)
    Z = torch.zeros(N, N, device=device, dtype=dtype)
    comp = torch.cat([torch.cat([top_left, top_right], 1),
                      torch.cat([I, Z], 1)], 0)
    ev = torch.linalg.eigvals(comp.to(torch.complex128))
    return float(ev.abs().max())
