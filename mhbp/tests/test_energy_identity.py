"""Identidad de energía: continua (Prop. 1, residuo ~0) y discreta (disipación
del esquema BE+Cayley sin forzamiento; Verlet de referencia acotado)."""
import torch

from mhbp.coupled_fields import MHBPConfig, CoupledMultiscaleHBP
from mhbp.field import FieldConfig
from mhbp.integrators.verlet import VerletRef
from mhbp.stability.certificates import energy_dissipation_check

torch.manual_seed(0)


def model4(theta=1.0, dt=0.5):
    cfg = MHBPConfig(dt=dt, theta=theta, allostasis=False)
    m = CoupledMultiscaleHBP(cfg).double()
    m.reset_state(1)
    return m


def test_continuous_energy_identity():
    """Ė por la dinámica == fórmula de la Prop. 1: residuo < 1e−10 (FP64).
    Se evalúa con las matrices globales ensambladas (incluye acoplamiento)."""
    m = model4()
    vr = VerletRef()
    taus = m.timescales()
    vr.prepare(list(m.fields), taus, m.coupling, m.cfg.dt)
    n = vr.n_tot
    Tinv = vr._hvec / m.cfg.dt                                   # 1/τ por entrada
    K, CG = vr._K, vr._CG
    C = 0.5 * (CG + CG.T)                                        # parte simétrica = C
    torch.manual_seed(3)
    for _ in range(10):
        u = torch.randn(n, dtype=torch.float64)
        w = torch.randn(n, dtype=torch.float64)
        F = torch.randn(n, dtype=torch.float64)
        udot = Tinv * w
        wdot = Tinv * (-(CG @ w) - K @ u + F)
        # Ė por derivación directa de E = ½‖w‖² + ½u𝓚u
        Edot_dyn = w @ wdot + udot @ (K @ u)
        # fórmula de la Prop. 1
        Edot_formula = -(w * Tinv) @ (C @ w) + (w * Tinv) @ F
        assert abs(Edot_dyn - Edot_formula) < 1e-10 * max(1.0, abs(Edot_dyn))


def test_discrete_dissipation_BE():
    """θ=1 (BE) + Cayley: E no creciente SIEMPRE sin forzamiento."""
    m = model4(theta=1.0)
    rep = energy_dissipation_check(m, trials=20, ticks=50)
    assert rep["ok"], rep


def test_discrete_dissipation_CN():
    """θ=½ (CN): disipación (ζ>0) — tolerancia 1e−9 por redondeo."""
    m = model4(theta=0.5)
    rep = energy_dissipation_check(m, trials=10, ticks=50, tol=1e-9)
    assert rep["ok"], rep


def test_verlet_reference_bounded():
    """El Verlet de referencia con dt seguro: energía acotada (no explota)."""
    m = model4()
    vr = VerletRef()
    vr.prepare(list(m.fields), m.timescales(), m.coupling, dt=0.1)
    n = vr.n_tot
    torch.manual_seed(4)
    U = torch.randn(1, n, dtype=torch.float64)
    W = torch.randn(1, n, dtype=torch.float64)
    E0 = float(vr.energy(U, W))
    for _ in range(500):
        U, W = vr.step(U, W, None)
    assert torch.isfinite(U).all()
    assert float(vr.energy(U, W)) < 2.0 * E0
