"""Certificado espectral exacto: ρ(Φ)<1 en configs seguras aleatorias; linealidad
del sondeo; detección de inestabilidad al violar la estructura."""
import torch
import pytest

from mhbp.coupled_fields import MHBPConfig, CoupledMultiscaleHBP
from mhbp.stability.certificates import (one_step_matrix, exact_spectral_radius,
                                         structural_certificate, stability_report)

torch.manual_seed(0)


def rand_model(seed, topo="chain", theta=1.0, mode="fixed"):
    torch.manual_seed(seed)
    cfg = MHBPConfig(dt=0.5, theta=theta, coupling_topology=topo,
                     timescale_mode=mode, allostasis=False)
    m = CoupledMultiscaleHBP(cfg).double()
    # aleatoriza TODOS los crudos dentro de las cajas
    for f in m.fields:
        for p in f.params.parameters():
            p.data.uniform_(-3, 3)
    if m.coupling is not None:
        m.coupling.raw_kappa.data.uniform_(-3, 3)
        for W in m.coupling.W:
            W.data.normal_(0, 1.0)
    m.reset_state(1)
    return m


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("topo,theta", [("chain", 1.0), ("full", 1.0),
                                        ("none", 1.0), ("chain", 0.5)])
def test_rho_below_one_random_safe(seed, topo, theta):
    m = rand_model(seed, topo=topo, theta=theta)
    rho = exact_spectral_radius(m)
    assert rho < 1.0, f"ρ={rho:.6f} con params seguros (seed={seed}, {topo}, θ={theta})"


def test_rho_learnable_timescales():
    m = rand_model(99, mode="learnable")
    assert exact_spectral_radius(m) < 1.0


def test_probe_linearity():
    """Φ·z == step(z) para z aleatorio: el sondeo captura un mapa LINEAL real."""
    m = rand_model(5)
    Phi = one_step_matrix(m)
    integ = m._prepare()
    n = integ.n_tot
    z = torch.randn(3, 2 * n, dtype=torch.float64)
    U, W = integ.step(z[:, :n], z[:, n:], None)
    z_step = torch.cat([U, W], 1)
    z_mat = z @ Phi.T
    assert torch.allclose(z_step, z_mat, atol=1e-10), "no-linealidad o fuga de estado"


def test_structural_certificate_ok():
    m = rand_model(7)
    s = structural_certificate(m)
    assert s["all_ok"], s


def test_detects_injected_instability(monkeypatch):
    """Al violar la estructura (amortiguamiento NEGATIVO inyectado en TODOS los
    campos), ρ ≥ 1: el certificado detecta, no da falsos PASS."""
    from mhbp.operators import FieldParams
    m = rand_model(11, topo="none")
    monkeypatch.setattr(FieldParams, "zeta",
                        property(lambda self: -0.3 * torch.ones_like(self.raw_zeta)))
    rho = exact_spectral_radius(m)
    assert rho > 1.0, f"instabilidad inyectada NO detectada (ρ={rho:.4f})"


def test_verlet_reference_unstable_big_dt_detected():
    """El Verlet de referencia con Δt grande viola su CFL: ρ>1 — y el sondeo
    lo mide (contraste con Cayley-IMEX, incondicional)."""
    cfg_v = MHBPConfig(dt=6.0, integrator="verlet_ref", coupling_topology="none",
                       allostasis=False)
    mv = CoupledMultiscaleHBP(cfg_v).double()
    mv.reset_state(1)
    rho_v = exact_spectral_radius(mv)
    cfg_c = MHBPConfig(dt=6.0, integrator="cayley_imex", coupling_topology="none",
                       allostasis=False)
    mc = CoupledMultiscaleHBP(cfg_c).double()
    mc.reset_state(1)
    rho_c = exact_spectral_radius(mc)
    assert rho_v > 1.0, f"Verlet con Δt=6 debería violar CFL (ρ={rho_v:.3f})"
    assert rho_c < 1.0, f"Cayley-IMEX debe ser incondicional (ρ={rho_c:.6f})"


def test_stability_report():
    m = rand_model(13)
    rep = stability_report(m)
    assert rep["ok"], rep
