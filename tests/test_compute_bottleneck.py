import torch

from eval.compute_bottleneck import (
    optimal_halt_allocation,
    rank_yoked_allocation,
    shuffled_allocation,
    uniform_allocation,
)
from model.adaptive_depth import AdaptiveHalting
from model.hbp import HBPConfig
from model.miura import MiuraCognitiveFull, MiuraFullConfig
from model.transformer import MiuraConfig


def test_optimal_allocation_obeys_exact_budget_and_preferences():
    probs = torch.tensor([
        [0.90, 0.08, 0.02, 0.00],
        [0.05, 0.10, 0.25, 0.60],
        [0.10, 0.20, 0.60, 0.10],
    ])
    quotas = optimal_halt_allocation(probs, total_budget=8)
    assert quotas.tolist() == [1, 4, 3]
    assert int(quotas.sum()) == 8


def test_yoked_controls_preserve_multiset_and_budget():
    quotas = torch.tensor([1, 2, 4, 4, 6])
    difficulty = torch.tensor([9, 2, 7, 3, 12])
    g = torch.Generator().manual_seed(7)
    shuffled = shuffled_allocation(quotas, g)
    oracle = rank_yoked_allocation(quotas, difficulty, hardest_gets_most=True)
    anti = rank_yoked_allocation(quotas, difficulty, hardest_gets_most=False)
    expected = quotas.sort().values
    for control in (shuffled, oracle, anti):
        assert torch.equal(control.sort().values, expected)
        assert int(control.sum()) == int(quotas.sum())
    assert int(oracle[difficulty.argmax()]) == int(quotas.max())
    assert int(anti[difficulty.argmax()]) == int(quotas.min())


def test_uniform_allocation_is_exact():
    quotas = uniform_allocation(5, total_budget=17, max_steps=6)
    assert quotas.tolist() == [4, 4, 3, 3, 3]
    assert int(quotas.sum()) == 17


def test_forced_halting_only_executes_active_rows():
    halting = AdaptiveHalting(d_model=2, max_steps=4)
    x = torch.zeros(3, 1, 2)
    quotas = torch.tensor([1, 2, 4])
    batch_sizes = []

    def step_fn(h, mod, active_idx):
        batch_sizes.append(int(h.shape[0]))
        return h + 1

    out, exact_steps, _ = halting.forward_forced(step_fn, x, quotas)
    assert batch_sizes == [3, 2, 1, 1]
    assert out[:, 0, 0].tolist() == [1.0, 2.0, 4.0]
    assert exact_steps.tolist() == [1.0, 2.0, 4.0]


def test_full_hbp_model_accepts_sparse_forced_steps():
    torch.manual_seed(3)
    tcfg = MiuraConfig(vocab_size=20, d_model=32, n_layers=1,
                       n_heads=4, d_ff=64, max_seq_len=8)
    cfg = MiuraFullConfig(transformer=tcfg, hbp=HBPConfig(),
                          use_hbp=True, use_adaptive_depth=True,
                          use_working_memory=True, max_halt_steps=4)
    model = MiuraCognitiveFull(cfg).eval()
    idx = torch.randint(1, 20, (3, 8))
    quotas = torch.tensor([1, 2, 4])
    logits, losses = model(idx, None, forced_steps=quotas)
    assert logits.shape == (3, 8, 20)
    assert losses == {}
    assert model._last_effective_steps.tolist() == [1.0, 2.0, 4.0]
    assert model._last_reasoner_step_units == 7
    assert model.hbp.h_t.shape[0] == 3


def test_sparse_batch_matches_individual_forced_execution():
    torch.manual_seed(11)
    tcfg = MiuraConfig(vocab_size=20, d_model=32, n_layers=1,
                       n_heads=4, d_ff=64, max_seq_len=8)
    cfg = MiuraFullConfig(transformer=tcfg, hbp=HBPConfig(),
                          use_hbp=True, use_adaptive_depth=True,
                          use_working_memory=True, max_halt_steps=4)
    model = MiuraCognitiveFull(cfg).eval()
    idx = torch.randint(1, 20, (3, 8))
    quotas = torch.tensor([1, 2, 4])
    batched, _ = model(idx, None, forced_steps=quotas)
    singles = []
    for i in range(idx.shape[0]):
        logits, _ = model(idx[i:i + 1], None, forced_steps=quotas[i:i + 1])
        singles.append(logits)
    individual = torch.cat(singles, dim=0)
    torch.testing.assert_close(batched, individual, rtol=2e-5, atol=2e-5)


def test_expected_ponder_loss_credits_halting_controller():
    torch.manual_seed(19)
    tcfg = MiuraConfig(vocab_size=20, d_model=32, n_layers=1,
                       n_heads=4, d_ff=64, max_seq_len=8)
    cfg = MiuraFullConfig(transformer=tcfg, hbp=HBPConfig(),
                          use_hbp=False, use_adaptive_depth=True,
                          use_working_memory=False, max_halt_steps=4,
                          ponder_expected_loss=True)
    model = MiuraCognitiveFull(cfg).train()
    idx = torch.randint(1, 20, (3, 8))
    targets = torch.full_like(idx, -1)
    targets[:, 6] = torch.tensor([3, 7, 11])
    _, losses = model(idx, targets)
    assert torch.isfinite(losses["total"])
    assert "ponder_expected_task" in losses
    losses["total"].backward()
    grad = model.halting.halt_proj.weight.grad
    assert grad is not None and float(grad.abs().max()) > 0.0
