"""Escalas temporales: orden estricto POR CONSTRUCCIÓN en los 3 modos."""
import torch

from mhbp.coupled_fields import Timescales

torch.manual_seed(0)


def test_fixed():
    ts = Timescales((1.0, 4.0, 8.0, 32.0), mode="fixed")
    taus = ts()
    assert (torch.diff(taus) > 0).all()
    assert torch.allclose(taus, torch.tensor([1.0, 4.0, 8.0, 32.0]).double())


def test_learnable_order_for_arbitrary_raws():
    ts = Timescales(mode="learnable")
    for _ in range(200):
        ts.raw_a.data = 20.0 * torch.randn_like(ts.raw_a)   # crudos arbitrarios (Q+1)
        taus = ts()
        assert (torch.diff(taus) > 1e-6).all(), taus
        assert taus.min() >= ts.tau_min - 1e-9 and taus.max() <= ts.tau_max + 1e-9


def test_learnable_order_after_grad_steps():
    ts = Timescales(mode="learnable")
    opt = torch.optim.SGD(ts.parameters(), lr=1.0)
    for _ in range(50):
        taus = ts()
        loss = -taus[0] + taus[3]                          # empuja a colapsar el orden
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            assert (torch.diff(ts()) > 1e-6).all(), "orden roto tras SGD agresivo"


def test_context_mode():
    ts = Timescales(mode="context", ctx_dim=5)
    for _ in range(50):
        ctx = torch.randn(3, 5) * 10
        taus = ts(ctx)
        assert (torch.diff(taus) > 1e-6).all()
        assert taus.min() >= ts.tau_min - 1e-9 and taus.max() <= ts.tau_max + 1e-9
