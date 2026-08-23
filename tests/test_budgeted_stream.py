import torch

from data.budgeted_stream import BudgetedStreamBatch, BudgetedTransitionDataset
from eval.budgeted_stream import evaluate_budgeted_stream
from eval.hidden_need import evaluate_hidden_need, online_prefix_allocation
from model.budgeted_stream import BudgetedStreamConfig, BudgetedStreamReasoner


def _small_model(variant="gating_wm", max_steps=3):
    return BudgetedStreamReasoner(BudgetedStreamConfig(
        n_states=12, n_ops=4, max_steps=max_steps,
        d_model=32, variant=variant))


def test_shared_executor_initialization_is_paired_across_variants():
    models = []
    for variant in ("gating_wm", "hbp_first", "hbp_full"):
        torch.manual_seed(2026)
        models.append(_small_model(variant=variant))

    shared = ("state_emb", "op_emb", "executor", "state_norm", "readout")
    reference = dict(models[0].named_parameters())
    for model in models[1:]:
        candidate = dict(model.named_parameters())
        for prefix in shared:
            names = [name for name in reference
                     if name == prefix or name.startswith(prefix + ".")]
            assert names
            for name in names:
                torch.testing.assert_close(
                    reference[name], candidate[name], rtol=0, atol=0)


def test_balanced_batch_has_exact_required_budget_and_frozen_suffix():
    ds = BudgetedTransitionDataset(min_steps=1, max_steps=3, seed=4)
    batch = ds.batch(6, balanced=True)
    assert sorted(batch.lengths.tolist()) == [1, 1, 2, 2, 3, 3]
    assert int(batch.lengths.sum()) == 6 * 2
    ds.validate(batch)
    for row, k in enumerate(batch.lengths.tolist()):
        if k < ds.max_steps:
            assert (batch.operations[row, k:] == ds.PAD_OP).all()
            assert (batch.intermediate_states[row, k:] == batch.final_state[row]).all()


def test_hidden_length_stream_reveals_one_terminal_marker_only_at_K():
    ds = BudgetedTransitionDataset(
        min_steps=1, max_steps=3, seed=40, terminal_markers=True)
    batch = ds.batch(6, balanced=True)
    terminal = ((batch.operations >= ds.n_base_ops)
                & (batch.operations < ds.PAD_OP))
    assert terminal.sum().item() == 6
    rows = torch.arange(6)
    assert terminal[rows, batch.lengths - 1].all()
    for row, k in enumerate(batch.lengths.tolist()):
        assert not terminal[row, :k - 1].any()


def test_implicit_stream_has_no_terminal_token_and_cannot_anticipate_K():
    ds = BudgetedTransitionDataset(
        min_steps=1, max_steps=3, seed=42, terminal_markers=False)
    generated = ds.batch(6, balanced=True)
    assert not ((generated.operations >= ds.n_base_ops)
                & (generated.operations < ds.PAD_OP)).any()

    torch.manual_seed(42)
    model = BudgetedStreamReasoner(BudgetedStreamConfig(
        n_states=12, n_ops=4, max_steps=5, d_model=32,
        variant="hbp_full", observe_length=False)).eval()
    batch = BudgetedStreamBatch(
        initial_state=torch.tensor([4, 4]),
        operations=torch.tensor([
            [0, 1, 2, 4, 4],
            [0, 1, 2, 3, 0],
        ]),
        lengths=torch.tensor([3, 5]),
        final_state=torch.tensor([0, 1]),
        intermediate_states=torch.zeros(2, 5, dtype=torch.long))
    result = model.forward_soft(batch, compute_loss=False)
    # Incluso en el tick K de la primera muestra, los dos prefijos son iguales:
    # el final sólo se hace observable al financiar el tick siguiente sin cambio.
    torch.testing.assert_close(
        result["halt_logits_by_step"][0, :3],
        result["halt_logits_by_step"][1, :3], rtol=0, atol=0)


def test_hidden_scheduler_prefix_is_independent_of_future_length():
    torch.manual_seed(41)
    model = BudgetedStreamReasoner(BudgetedStreamConfig(
        n_states=12, n_ops=8, max_steps=5, d_model=32,
        variant="hbp_full", observe_length=False)).eval()
    batch = BudgetedStreamBatch(
        initial_state=torch.tensor([4, 4]),
        operations=torch.tensor([
            [0, 1, 6, 8, 8],       # K=3; LAST(2)=6
            [0, 1, 2, 3, 4],       # K=5; LAST(0)=4
        ]),
        lengths=torch.tensor([3, 5]),
        final_state=torch.tensor([0, 1]),
        intermediate_states=torch.zeros(2, 5, dtype=torch.long))
    result = model.forward_soft(batch, compute_loss=False)
    torch.testing.assert_close(
        result["halt_logits_by_step"][0, :2],
        result["halt_logits_by_step"][1, :2], rtol=0, atol=0)


def test_online_allocator_reads_only_each_current_prefix_and_is_exact():
    # Empieza leyendo sólo la columna 0: elige muestra 0. Después puede leer
    # columna 1 de esa muestra, ve que ya debe parar y elige muestra 1.
    halt_logits = torch.tensor([
        [-2.0, 5.0, -999.0],
        [-1.0, 4.0, 999.0],
    ])
    quotas, choices = online_prefix_allocation(halt_logits, total_budget=4)
    assert choices == [0, 1]
    assert quotas.tolist() == [2, 2]
    assert int(quotas.sum()) == 4


def test_forced_prefix_cannot_see_future_operations():
    torch.manual_seed(2)
    model = _small_model(max_steps=5).eval()
    ops = torch.tensor([
        [0, 1, 2, 0, 3],
        [0, 1, 2, 3, 0],
    ])
    batch = BudgetedStreamBatch(
        initial_state=torch.tensor([4, 4]), operations=ops,
        lengths=torch.tensor([5, 5]), final_state=torch.tensor([0, 1]),
        intermediate_states=torch.zeros(2, 5, dtype=torch.long))
    logits = model.forward_forced(batch, torch.tensor([3, 3]))
    torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)


def test_ticks_after_K_cost_budget_but_do_not_change_task_state():
    torch.manual_seed(3)
    model = _small_model(max_steps=5).eval()
    ops = torch.tensor([
        [0, 2, 4, 4, 4],
        [0, 2, 4, 4, 4],
    ])
    batch = BudgetedStreamBatch(
        initial_state=torch.tensor([7, 7]), operations=ops,
        lengths=torch.tensor([2, 2]), final_state=torch.tensor([0, 0]),
        intermediate_states=torch.zeros(2, 5, dtype=torch.long))
    logits = model.forward_forced(batch, torch.tensor([2, 5]))
    torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
    assert model._last_reasoner_step_units == 7
    assert model._last_active_batch_sizes == [2, 2, 1, 1, 1]


def test_soft_objective_credits_executor_and_halting():
    torch.manual_seed(5)
    ds = BudgetedTransitionDataset(min_steps=1, max_steps=3, seed=5)
    model = _small_model(max_steps=3).train()
    result = model.forward_soft(ds.batch(6), compute_loss=True)
    assert torch.isfinite(result["losses"]["total"])
    result["losses"]["total"].backward()
    assert float(model.executor.weight_hh.grad.abs().max()) > 0
    assert float(model.halt_proj.weight.grad.abs().max()) > 0


def test_fast_executor_pretraining_matches_full_auxiliary_objective():
    torch.manual_seed(51)
    ds = BudgetedTransitionDataset(
        min_steps=1, max_steps=3, seed=51, terminal_markers=False)
    model = _small_model("hbp_full", max_steps=3).train()
    batch = ds.batch(6)
    fast = model.executor_auxiliary_loss(batch)
    full = model.forward_soft(batch, compute_loss=True)["losses"]["auxiliary"]
    torch.testing.assert_close(fast, full, rtol=1e-6, atol=1e-7)


def test_hbp_full_soft_rollout_is_finite_and_differentiable():
    torch.manual_seed(6)
    ds = BudgetedTransitionDataset(min_steps=1, max_steps=3, seed=6)
    model = _small_model("hbp_full", max_steps=3).train()
    result = model.forward_soft(ds.batch(6), compute_loss=True)
    result["losses"]["total"].backward()
    assert torch.isfinite(result["probabilities"]).all()
    assert model.hbp.raw_zeta.grad is not None
    assert float(model.hbp.raw_zeta.grad.abs().max()) > 0


def test_causal_evaluator_enforces_equal_total_budget():
    torch.manual_seed(7)
    ds = BudgetedTransitionDataset(min_steps=1, max_steps=3, seed=7)
    model = _small_model(max_steps=3).eval()
    result = evaluate_budgeted_stream(
        model, ds, torch.device("cpu"), mean_budget=2,
        n_batches=1, batch_size=6, n_shuffles=1, seed=7)
    for policy in result["policies"].values():
        assert policy["reasoner_step_units"] == 12


def test_implicit_online_evaluator_enforces_equal_total_budget():
    torch.manual_seed(71)
    ds = BudgetedTransitionDataset(
        min_steps=1, max_steps=3, seed=71, terminal_markers=False)
    model = BudgetedStreamReasoner(BudgetedStreamConfig(
        n_states=12, n_ops=4, max_steps=3, d_model=32,
        variant="gating_wm", observe_length=False)).eval()
    result = evaluate_hidden_need(
        model, ds, torch.device("cpu"), mean_budget=2,
        n_batches=1, batch_size=6, n_shuffles=1, seed=71)
    assert result["design"]["terminal_marker_present"] is False
    for policy in result["policies"].values():
        assert policy["reasoner_step_units"] == 12
