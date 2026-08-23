"""Actuadores: acotación por construcción bajo estados extremos, STE discreto,
intervención causal (override / freeze_field) y registro de saturación."""
import torch

from mhbp.actuators import ActuatorHead, MVP_ACTUATORS
from mhbp.field import FieldConfig

torch.manual_seed(0)
CFGS = [FieldConfig(name=f"f{i}", n_nodes=4, d=6) for i in range(3)]


def rand_pooled(scale, B=5):
    return [(scale * torch.randn(B, 6, dtype=torch.float64),
             scale * torch.randn(B, 6, dtype=torch.float64)) for _ in CFGS]


def test_bounds_under_extreme_states():
    head = ActuatorHead(CFGS).double()
    for scale in (0.0, 1.0, 1e3):
        acts, info = head(rand_pooled(scale))
        for sp in MVP_ACTUATORS:
            a = acts[sp.name]
            assert (a >= sp.lo - 1e-9).all() and (a <= sp.hi + 1e-9).all(), \
                f"{sp.name} fuera de [{sp.lo},{sp.hi}] con escala {scale}"
            assert torch.isfinite(a).all()


def test_depth_is_integer_valued():
    head = ActuatorHead(CFGS).double()
    acts, _ = head(rand_pooled(10.0))
    d = acts["depth_max"]
    assert torch.allclose(d, d.round(), atol=1e-9), "STE debe devolver enteros"


def test_rest_state_gives_a0():
    head = ActuatorHead(CFGS).double()
    acts, _ = head(rand_pooled(0.0))
    for sp in MVP_ACTUATORS:
        ref = round(sp.a0) if sp.discrete else sp.a0
        assert torch.allclose(acts[sp.name], torch.full_like(acts[sp.name], ref),
                              atol=1e-6), f"{sp.name}: reposo != a0"


def test_override_and_freeze():
    head = ActuatorHead(CFGS).double()
    pooled = rand_pooled(2.0)
    head.override("tool_gate", 0.77)
    acts, _ = head(pooled)
    assert torch.allclose(acts["tool_gate"], torch.full_like(acts["tool_gate"], 0.77))
    head.override("tool_gate", None)

    a_ref, info_ref = head(pooled)
    head.freeze_field(0)
    a_frz, info_frz = head(pooled)
    head.freeze_field(0, frozen=False)
    assert float(info_frz["influence"][0].abs().sum()) == 0.0, "campo congelado influye"
    assert not torch.allclose(a_ref["halt_bias"], a_frz["halt_bias"]), \
        "congelar un campo con estado no nulo debe cambiar la acción"


def test_saturation_logged():
    head = ActuatorHead(CFGS).double()
    _, info = head(rand_pooled(1e4))
    for name, frac in info["saturation"].items():
        assert 0.0 <= frac <= 1.0
    assert max(info["saturation"].values()) > 0.5, "estados enormes deben saturar"
