"""End-to-end Fase 1: rollout largo finito, energía acotada, alostasis acotada,
checkpoint bitwise, protección NaN, determinismo de trayectoria."""
import torch
import pytest

from mhbp.coupled_fields import MHBPConfig, CoupledMultiscaleHBP
from mhbp.interoception import InteroceptiveSignal

torch.manual_seed(0)


def signal(B, t):
    g = torch.Generator().manual_seed(1000 + t)
    return InteroceptiveSignal(tick=t, values={
        "entropy": torch.rand(B, generator=g) * 2,
        "uncertainty": torch.rand(B, generator=g),
        "progress": torch.rand(B, generator=g),
        "gpu_utilization": torch.rand(B, generator=g),
        "token_cost": torch.rand(B, generator=g) * 5,
    })


def test_long_rollout_finite_and_bounded():
    m = CoupledMultiscaleHBP(MHBPConfig()).double()
    m.reset_state(3)
    energies = []
    with torch.no_grad():
        for t in range(200):
            acts, info = m.tick(signal(3, t))
            energies.append(float(info["energy"].max()))
    assert all(torch.isfinite(torch.tensor(energies))), "energía no finita"
    # forzamiento acotado => ISS: la energía no puede crecer sin límite
    assert max(energies[100:]) < 10.0 * max(energies[:20]) + 10.0, \
        f"energía creciendo sin cota: {energies[:3]} -> {energies[-3:]}"
    d = m.diagnostics()
    assert all(v["finite"] for k, v in d.items() if isinstance(v, dict) and "finite" in v)


def test_allostasis_bounded_and_regularized():
    m = CoupledMultiscaleHBP(MHBPConfig(allostasis=True, eps_allostasis=0.1)).double()
    m.reset_state(2)
    with torch.no_grad():
        for t in range(20):
            m.tick(signal(2, t))
        shift = m.allo._last_shift
        assert shift is not None and float(shift.abs().max()) <= 0.1 + 1e-9, \
            "desplazamiento alostático fuera de la cota ε_a"
    reg = m.regularizers()["allostasis"]
    assert torch.isfinite(reg) and float(reg) >= 0


def test_checkpoint_bitwise():
    """state_dict + save/load_dynamic_state (contrato explícito: u/w son buffers
    NO persistentes) -> trayectoria posterior IDÉNTICA."""
    m1 = CoupledMultiscaleHBP(MHBPConfig()).double()
    m1.reset_state(2)
    with torch.no_grad():
        for t in range(10):
            m1.tick(signal(2, t))
    ckpt = {"sd": m1.state_dict(), "dyn": m1.save_dynamic_state()}

    m2 = CoupledMultiscaleHBP(MHBPConfig()).double()
    m2.load_state_dict(ckpt["sd"])
    m2.load_dynamic_state(ckpt["dyn"])
    assert m2._tick == m1._tick, "el tick no sobrevivió al checkpoint"
    with torch.no_grad():
        for t in range(10, 15):
            a1, _ = m1.tick(signal(2, t))
            a2, _ = m2.tick(signal(2, t))
            for k in a1:
                assert torch.equal(a1[k], a2[k]), f"checkpoint no bitwise en {k} (t={t})"


def test_nan_protection_raises():
    m = CoupledMultiscaleHBP(MHBPConfig()).double()
    m.reset_state(1)
    m.fields[0].u[0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        m.tick(None)


def test_requires_reset():
    m = CoupledMultiscaleHBP(MHBPConfig()).double()
    with pytest.raises(RuntimeError):
        m.tick(None)


def test_trajectory_determinism():
    outs = []
    for _rep in range(2):
        torch.manual_seed(42)
        m = CoupledMultiscaleHBP(MHBPConfig()).double()
        m.reset_state(2)
        with torch.no_grad():
            for t in range(30):
                acts, _ = m.tick(signal(2, t))
        outs.append({k: v.clone() for k, v in acts.items()})
    for k in outs[0]:
        assert torch.equal(outs[0][k], outs[1][k]), f"no determinista: {k}"
