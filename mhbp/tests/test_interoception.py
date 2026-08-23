"""Interocepción (contrato §8): máscara, leak_mask, cota f_max, EMA, dtype."""
import torch

from mhbp.field import FieldConfig
from mhbp.interoception import (InteroceptiveSignal, InteroceptionEncoder,
                                RunningNorm, CHANNELS, N_CHANNELS)

torch.manual_seed(0)
CFGS = [FieldConfig(name="a", n_nodes=3, d=4), FieldConfig(name="b", n_nodes=2, d=4)]


def test_missing_channel_zero_contribution():
    """Canal ausente ⇒ F idéntico a no pasarlo (contribución 0 exacta)."""
    enc = InteroceptionEncoder(CFGS).double().eval()
    s1 = InteroceptiveSignal(values={"entropy": 1.5})
    s2 = InteroceptiveSignal(values={"entropy": 1.5, "risk": None})
    F1 = enc(s1, 2, torch.device("cpu"))
    F2 = enc(s2, 2, torch.device("cpu"))
    for a, b in zip(F1, F2):
        assert torch.equal(a, b)


def test_leak_mask_vetoes_channel():
    """leak_mask: el valor del canal vetado NO afecta al forzamiento."""
    enc = InteroceptionEncoder(CFGS, leak_mask=["entropy"]).double().eval()
    Fa = enc(InteroceptiveSignal(values={"entropy": 0.0, "risk": 1.0}), 2, torch.device("cpu"))
    Fb = enc(InteroceptiveSignal(values={"entropy": 1e6, "risk": 1.0}), 2, torch.device("cpu"))
    for a, b in zip(Fa, Fb):
        assert torch.allclose(a, b, atol=1e-12), "canal vetado se está filtrando"


def test_f_max_bound_under_extreme_signals():
    enc = InteroceptionEncoder(CFGS, f_max=0.5).double().eval()
    vals = {k: 1e6 for k in CHANNELS}
    F = enc(InteroceptiveSignal(values=vals), 3, torch.device("cpu"))
    for f in F:
        assert float(f.abs().max()) <= 0.5 + 1e-12


def test_ema_frozen_in_eval_and_var_floor():
    rn = RunningNorm()
    s = torch.full((1, N_CHANNELS), 3.7)
    m = torch.ones(1, N_CHANNELS)
    rn.train()
    for _ in range(2000):
        rn(s, m)
    # suelo de varianza: señal constante no amplifica ×1/√eps (hallazgo review)
    x = rn(s + 0.01, m)
    assert float(x.abs().max()) < 5.0, "amplificación por colapso de varianza"
    mean_before = rn.mean.clone()
    rn.eval()
    rn(s * 100, m)
    assert torch.equal(rn.mean, mean_before), "EMA se actualizó en eval()"


def test_fp64_no_silent_truncation():
    """Señales FP64 llegan al encoder FP64 sin pasar por FP32 (minor review)."""
    enc = InteroceptionEncoder(CFGS).double().eval()
    v = 0.1234567890123456789
    sig = InteroceptiveSignal(values={"entropy": torch.tensor([v], dtype=torch.float64)})
    s, m = sig.to_tensor(1, torch.device("cpu"), dtype=enc.norm.mean.dtype)
    assert s.dtype == torch.float64
    assert abs(float(s[0, CHANNELS.index("entropy")]) - v) < 1e-16
