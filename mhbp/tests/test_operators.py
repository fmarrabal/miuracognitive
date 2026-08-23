"""Tests algebraicos: grafos, cajas seguras, operadores K/C/G, acoplamiento PSD."""
import torch
import pytest

from mhbp.operators import build_graph, FieldParams, build_KCG, apply_op
from mhbp.field import FieldConfig, HomeostaticField
from mhbp.coupled_fields import InterfaceCoupling, default_four_fields
from mhbp.integrators.cayley_imex import field_offsets, pack

torch.manual_seed(0)


@pytest.mark.parametrize("kind", ["chain", "ring", "star", "complete"])
@pytest.mark.parametrize("n", [2, 4, 7])
def test_graphs(kind, n):
    L, A = build_graph(kind, n)
    assert torch.allclose(L, L.T), "L simétrico"
    ev = torch.linalg.eigvalsh(L)
    assert ev.min() > -1e-12, "L PSD"
    assert abs(ev.min()) < 1e-10, "L tiene modo constante (combinatorio)"
    assert torch.allclose(A, -A.T), "A antisimétrica"
    assert torch.allclose(L.sum(1), torch.zeros(n, dtype=L.dtype), atol=1e-12), \
        "filas de L suman 0"


def test_param_boxes():
    p = FieldParams(8)
    for raw in (p.raw_omega, p.raw_zeta):
        raw.data.uniform_(-40, 40)                       # extremos absurdos
    assert (p.omega >= p.omega_min).all() and (p.omega <= p.omega_max).all()
    assert (p.zeta >= p.zeta_min).all()
    for name in ("c", "D", "b", "beta"):
        v = getattr(p, name)
        assert v >= 0, name


def test_KCG_structure():
    p = FieldParams(4, b_init=0.1, beta_init=0.05)
    L, A = build_graph("chain", 6)
    K, C, G = build_KCG(p, L, A)
    for M, nm in ((K, "K"), (C, "C")):
        assert torch.allclose(M, M.transpose(-1, -2)), f"{nm} simétrica"
        assert torch.linalg.eigvalsh(M).min() > 0, f"{nm} definida positiva"
    assert (G + G.transpose(-1, -2)).abs().max() < 1e-12, "G antisimétrica"


def test_apply_op_equals_dense():
    p = FieldParams(3)
    L, A = build_graph("chain", 5)
    K, _, _ = build_KCG(p, L, A)
    x = torch.randn(2, 5, 3, dtype=torch.float64)
    ref = torch.stack([K[l] @ x[..., l].T for l in range(3)], -1).transpose(0, 1)
    assert torch.allclose(apply_op(K, x), ref, atol=1e-13)


def test_coupling_psd_and_energy_consistency():
    """𝓑 ⪰ 0 para W/κ arbitrarias, y ½ z𝓑z == E_cpl(medias) — cross-check clave."""
    cfgs = default_four_fields(d=4)
    fields = [HomeostaticField(c) for c in cfgs]
    cpl = InterfaceCoupling(cfgs, [(0, 1), (1, 2), (2, 3)], d_c=3).double()
    for W in cpl.W:                                       # valores arbitrarios
        W.data = torch.randn_like(W) * 2.0
    offs, n = field_offsets(fields)
    B = cpl.assemble_B(offs, n, cfgs, torch.float64, torch.device("cpu"))
    assert torch.allclose(B, B.T, atol=1e-12)
    assert torch.linalg.eigvalsh(0.5 * (B + B.T)).min() > -1e-10, "𝓑 PSD"
    us = [torch.randn(2, c.n_nodes, c.d, dtype=torch.float64) for c in cfgs]
    z = pack(us)
    quad = 0.5 * torch.einsum("bi,ij,bj->b", z, B, z)
    E = cpl.energy([u.mean(1) for u in us])
    assert torch.allclose(quad, E.double(), atol=1e-10), "forma cuadrática == potencial"


def test_pin_fp32():
    f = HomeostaticField(FieldConfig(d=4))
    f.params.raw_omega.data = f.params.raw_omega.data.to(torch.bfloat16).float()
    f.params.pin_fp32()
    assert f.params.raw_omega.dtype == torch.float32
