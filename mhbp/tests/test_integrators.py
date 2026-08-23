"""Tests numéricos del integrador: Cayley ortogonal, convergencia vs expm,
oscilador analítico, equivalencia de batch, determinismo, protección extrema."""
import math
import torch
import pytest

from mhbp.coupled_fields import MHBPConfig, CoupledMultiscaleHBP, default_four_fields
from mhbp.field import FieldConfig
from mhbp.integrators.cayley_imex import CayleyIMEX, pack
from mhbp.integrators.verlet import VerletRef

torch.manual_seed(0)


def small_model(theta=1.0, topo="chain", dt=0.25, b=0.1, beta=0.03):
    fcs = [FieldConfig(name="f0", n_nodes=4, d=3, b_init=b, beta_init=beta),
           FieldConfig(name="f1", n_nodes=3, d=3, b_init=b, beta_init=beta)]
    cfg = MHBPConfig(d=3, dt=dt, theta=theta, taus=(1.0, 4.0),
                     coupling_topology=topo, allostasis=False)
    m = CoupledMultiscaleHBP(cfg, field_cfgs=fcs).double()
    m.reset_state(1)
    return m


def test_cayley_orthogonality():
    m = small_model()
    integ = m._prepare()
    assert integ.cayley_orthogonality_error() < 1e-12


def _continuous_S(m):
    """Matriz continua S del sistema global (para la referencia expm)."""
    vr = VerletRef()
    taus = m.timescales()
    vr.prepare(list(m.fields), taus, m.coupling, m.cfg.dt)
    n = vr.n_tot
    Tinv = (vr._hvec / m.cfg.dt) ** 1                      # h/dt = 1/τ
    S = torch.zeros(2 * n, 2 * n, dtype=torch.float64)
    S[:n, n:] = torch.diag(Tinv)
    S[n:, :n] = -Tinv.unsqueeze(1) * vr._K
    S[n:, n:] = -Tinv.unsqueeze(1) * vr._CG
    return S, n


@pytest.mark.parametrize("theta,order_lo,order_hi", [(1.0, 0.85, 1.4), (0.5, 1.7, 2.4)])
def test_convergence_to_expm(theta, order_lo, order_hi):
    """Error global vs exponencial de matriz: pendiente = orden del esquema."""
    torch.manual_seed(1)
    m = small_model(theta=theta)
    S, n = _continuous_S(m)
    z0 = torch.randn(1, 2 * n, dtype=torch.float64) * 0.5
    T_total = 2.0
    ref = (torch.linalg.matrix_exp(S * T_total) @ z0.squeeze(0))
    errs, hs = [], []
    for nsteps in (8, 16, 32, 64, 128):
        dt = T_total / nsteps
        integ = CayleyIMEX(theta=theta)
        integ.prepare(list(m.fields), m.timescales(), m.coupling, dt)
        U, W = z0[:, :n].clone(), z0[:, n:].clone()
        for _ in range(nsteps):
            U, W = integ.step(U, W, None)
        z = torch.cat([U, W], 1).squeeze(0)
        errs.append(float((z - ref).norm()))
        hs.append(dt)
    import numpy as np
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert order_lo < slope < order_hi, f"orden medido {slope:.2f} (θ={theta}), errs={errs}"


def test_analytic_damped_oscillator():
    """Campo 1-nodo sin G: u(t) analítico subamortiguado. CN (θ=½), h=0.01."""
    fc = FieldConfig(name="osc", n_nodes=1, d=1, omega_init=1.0, omega_min=0.99,
                     omega_max=1.01, zeta_init=0.3, c_init=0.0, c_max=1e-9,
                     b_init=0.0, b_max=1e-9, beta_init=0.0, beta_max=1e-9,
                     D_init=0.0, D_max=1e-9)
    cfg = MHBPConfig(d=1, dt=0.01, theta=0.5, taus=(1.0,), coupling_topology="none",
                     allostasis=False)
    m = CoupledMultiscaleHBP(cfg, field_cfgs=[fc]).double()
    m.reset_state(1)
    integ = m._prepare()
    om = float(m.fields[0].params.omega)
    ze = float(m.fields[0].params.zeta)
    U = torch.ones(1, 1, dtype=torch.float64)
    W = torch.zeros(1, 1, dtype=torch.float64)
    T, h = 10.0, 0.01
    for _ in range(int(T / h)):
        U, W = integ.step(U, W, None)
    om_d = om * math.sqrt(1 - ze ** 2)
    u_exact = math.exp(-ze * om * T) * (math.cos(om_d * T)
                                        + ze * om / om_d * math.sin(om_d * T))
    err = abs(float(U) - u_exact)
    assert err < 0.01 * math.exp(-ze * om * T) + 1e-4, f"err={err:.2e} vs exacto {u_exact:.4e}"


def test_batch_equivalence_and_determinism():
    m = small_model()
    integ = m._prepare()
    n = integ.n_tot
    torch.manual_seed(2)
    z = torch.randn(4, 2 * n, dtype=torch.float64)
    F = torch.randn(4, n, dtype=torch.float64) * 0.1
    Ub, Wb = integ.step(z[:, :n], z[:, n:], F)             # batch
    for i in range(4):                                     # una a una
        Ui, Wi = integ.step(z[i:i + 1, :n], z[i:i + 1, n:], F[i:i + 1])
        assert torch.allclose(Ub[i], Ui[0], atol=1e-12)
        assert torch.allclose(Wb[i], Wi[0], atol=1e-12)
    U2, W2 = integ.step(z[:, :n], z[:, n:], F)             # determinismo bitwise
    assert torch.equal(Ub, U2) and torch.equal(Wb, W2)


def test_extreme_params_finite():
    """Cajas al extremo + estados grandes: 100 ticks sin NaN (θ=1)."""
    m = small_model(dt=1.0)
    for f in m.fields:
        for p in f.params.parameters():
            p.data.fill_(40.0)                             # squash al tope
    m.reset_state(2)
    integ = m._prepare()
    n = integ.n_tot
    U = 1e3 * torch.randn(2, n, dtype=torch.float64)
    W = 1e3 * torch.randn(2, n, dtype=torch.float64)
    E0 = float(integ.energy(U, W).max())
    for _ in range(100):
        U, W = integ.step(U, W, None)
    assert torch.isfinite(U).all() and torch.isfinite(W).all()
    # sobreamortiguado (ζ≈40): decae LENTO (tasa ~ω/2ζ) pero decae — sin explotar
    assert float(integ.energy(U, W).max()) < E0, "la energía debe decrecer"
