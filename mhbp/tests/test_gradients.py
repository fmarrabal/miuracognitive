"""Gradientes: gradcheck FP64 a través de ticks completos + flujo de gradiente
a TODOS los grupos de parámetros (campos, acoplamiento, interfaz, actuadores)."""
import torch
from torch.autograd import gradcheck

from mhbp.coupled_fields import MHBPConfig, CoupledMultiscaleHBP
from mhbp.field import FieldConfig
from mhbp.interoception import InteroceptiveSignal

torch.manual_seed(0)


def tiny_model():
    fcs = [FieldConfig(name="a", n_nodes=2, d=2, b_init=0.1, beta_init=0.02),
           FieldConfig(name="b", n_nodes=3, d=2, b_init=0.1, beta_init=0.02)]
    cfg = MHBPConfig(d=2, dt=0.3, taus=(1.0, 4.0), allostasis=False)
    return CoupledMultiscaleHBP(cfg, field_cfgs=fcs).double()


def test_gradcheck_two_ticks():
    m = tiny_model()
    m.reset_state(1)
    integ = m._prepare()
    n = integ.n_tot

    def rollout(U0, W0, F):
        U, W = U0, W0
        for _ in range(2):
            U, W = integ.step(U, W, F)
        return (U ** 2).sum() + (W ** 2).sum()

    U0 = (0.1 * torch.randn(1, n, dtype=torch.float64)).requires_grad_(True)
    W0 = (0.1 * torch.randn(1, n, dtype=torch.float64)).requires_grad_(True)
    F = (0.1 * torch.randn(1, n, dtype=torch.float64)).requires_grad_(True)
    assert gradcheck(rollout, (U0, W0, F), atol=1e-6, rtol=1e-4)


def test_grad_flows_to_all_parameter_groups():
    """Un rollout con pérdida sobre actuadores debe dar gradiente finito y no nulo
    en: params físicos, κ, W de interfaz, encoder interoceptivo y actuadores."""
    m = tiny_model()
    m.reset_state(2)
    sig = InteroceptiveSignal(values={"entropy": torch.tensor([0.5, 1.5]),
                                     "progress": torch.tensor([0.1, 0.9])})
    m.train()
    loss = 0.0
    for _ in range(4):
        acts, info = m.tick(sig)
        loss = loss + acts["halt_bias"].pow(2).mean() + acts["budget_scale"].mean()
    loss = loss + 1e-3 * m.energy().mean()
    loss.backward()
    groups = {
        "raw_omega(f0)": m.fields[0].params.raw_omega,
        "raw_zeta(f1)": m.fields[1].params.raw_zeta,
        "kappa": m.coupling.raw_kappa,
        "W_interface": m.coupling.W[0],
        "intero_head": m.intero.heads[0].weight,
        "actuator_W": m.actuators.W[0],
    }
    # h_star NO participa en Fase 1 (la dinámica corre sobre u = h − h*; h* entra
    # en Fase 3 cuando el reasoner lea h = h* + u) — no se exige gradiente.
    for name, p in groups.items():
        assert p.grad is not None, f"sin grad: {name}"
        assert torch.isfinite(p.grad).all(), f"grad no finito: {name}"
    # los que actúan en el forward de verdad deben ser además NO nulos
    for name in ("raw_omega(f0)", "kappa", "actuator_W", "intero_head"):
        assert groups[name].grad.abs().max() > 0, f"grad idénticamente nulo: {name}"
