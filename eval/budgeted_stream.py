"""Evaluación causal para la tarea de transiciones reveladas por tick."""

from __future__ import annotations

import time
from collections import defaultdict

import torch

from data.budgeted_stream import BudgetedTransitionDataset
from eval.compute_bottleneck import (
    _cluster_signflip_exact,
    _mcnemar_exact,
    _pearson,
    optimal_halt_allocation,
    rank_yoked_allocation,
    shuffled_allocation,
    uniform_allocation,
)


def _run_forced(model, batch, quotas: torch.Tensor) -> tuple[list[bool], float, int]:
    q = quotas.to(batch.initial_state.device)
    if batch.initial_state.device.type == "cuda":
        torch.cuda.synchronize(batch.initial_state.device)
    t0 = time.perf_counter()
    logits = model.forward_forced(batch, q)
    if batch.initial_state.device.type == "cuda":
        torch.cuda.synchronize(batch.initial_state.device)
    elapsed = time.perf_counter() - t0
    units = int(model._last_reasoner_step_units)
    if units != int(quotas.sum()):
        raise RuntimeError(f"unidades ejecutadas {units} != presupuesto {int(quotas.sum())}")
    correct = logits.argmax(dim=-1).eq(batch.final_state)
    return [bool(x) for x in correct.cpu().tolist()], elapsed, units


@torch.no_grad()
def evaluate_budgeted_stream(model, dataset: BudgetedTransitionDataset,
                             device: torch.device, mean_budget: int = 5,
                             n_batches: int = 20, batch_size: int = 36,
                             n_shuffles: int = 4,
                             seed: int = 20260719) -> dict:
    """Evalúa políticas bajo una suma exacta e idéntica de ticks."""
    if batch_size % (dataset.max_steps - dataset.min_steps + 1):
        raise ValueError("batch_size no permite un batch de longitudes balanceado")
    if float(mean_budget) != dataset.mean_required_steps:
        raise ValueError(
            "mean_budget debe coincidir con la media exacta de K para incluir oracle_exact")
    if n_shuffles < 1:
        raise ValueError("n_shuffles debe ser >=1")
    model.eval()
    total_per_batch = int(mean_budget * batch_size)
    correct_all: dict[str, list[bool]] = defaultdict(list)
    units_all: dict[str, int] = defaultdict(int)
    time_all: dict[str, float] = defaultdict(float)
    by_k: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    records = []
    soft_correct = []
    all_k, all_q, all_ne = [], [], []

    for batch_idx in range(n_batches):
        batch = dataset.batch(batch_size, balanced=True).to(device)
        soft = model.forward_soft(batch, compute_loss=False)
        halt_probs = soft["halt_probs"].detach().float().cpu()
        n_expected = soft["n_expected"].detach().float().cpu()
        soft_ok = soft["probabilities"].argmax(dim=-1).eq(batch.final_state)
        soft_correct.extend(bool(x) for x in soft_ok.cpu().tolist())

        learned = optimal_halt_allocation(halt_probs, total_per_batch)
        uniform = uniform_allocation(
            batch_size, total_per_batch, max_steps=dataset.max_steps)
        lengths_cpu = batch.lengths.cpu()
        oracle_exact = lengths_cpu.clone()
        if int(oracle_exact.sum()) != total_per_batch:
            raise RuntimeError("el batch balanceado no conserva sum(K)=presupuesto")
        allocations = {
            "learned": learned,
            "uniform": uniform,
            "oracle_exact": oracle_exact,
            "oracle_yoked": rank_yoked_allocation(
                learned, lengths_cpu, hardest_gets_most=True),
            "anti_oracle": rank_yoked_allocation(
                learned, lengths_cpu, hardest_gets_most=False),
        }
        for j in range(n_shuffles):
            gen = torch.Generator().manual_seed(
                int(seed) + batch_idx * 10_007 + j * 1_000_003)
            allocations[f"shuffle_{j}"] = shuffled_allocation(learned, gen)

        learned_multiset = learned.sort().values
        batch_correct = {}
        batch_units = {}
        for name, quotas in allocations.items():
            if int(quotas.sum()) != total_per_batch:
                raise RuntimeError(f"{name}: presupuesto incorrecto")
            if name in ("oracle_yoked", "anti_oracle") or name.startswith("shuffle_"):
                if not torch.equal(quotas.sort().values, learned_multiset):
                    raise RuntimeError(f"{name}: cambió el multiconjunto yoked")
            ok, elapsed, units = _run_forced(model, batch, quotas)
            batch_correct[name] = ok
            batch_units[name] = units
            correct_all[name].extend(ok)
            units_all[name] += units
            time_all[name] += elapsed
            for is_correct, k in zip(ok, lengths_cpu.tolist()):
                by_k[name][int(k)][0] += int(is_correct)
                by_k[name][int(k)][1] += 1

        all_k.extend(int(x) for x in lengths_cpu.tolist())
        all_q.extend(int(x) for x in learned.tolist())
        all_ne.extend(float(x) for x in n_expected.tolist())
        records.append({
            "batch": batch_idx,
            "lengths": [int(x) for x in lengths_cpu.tolist()],
            "n_expected": [float(x) for x in n_expected.tolist()],
            "allocations": {name: [int(x) for x in q.tolist()]
                            for name, q in allocations.items()},
            "correct": batch_correct,
            "reasoner_step_units": batch_units,
        })

    policies = {}
    for name, values in correct_all.items():
        n = len(values)
        curve = {str(k): {"accuracy": c / kn, "correct": c, "n": kn}
                 for k, (c, kn) in sorted(by_k[name].items())}
        policies[name] = {
            "accuracy": sum(values) / n,
            "correct": int(sum(values)),
            "n": n,
            "mean_steps": units_all[name] / n,
            "reasoner_step_units": int(units_all[name]),
            "wall_time_s": float(time_all[name]),
            "accuracy_by_K": curve,
        }

    comparisons = {}
    learned_values = correct_all["learned"]
    for control in ("uniform", "shuffle_0", "oracle_exact",
                    "oracle_yoked", "anti_oracle"):
        control_values = correct_all[control]
        stat = _mcnemar_exact(learned_values, control_values)
        learned_batches = [sum(rec["correct"]["learned"]) for rec in records]
        control_batches = [sum(rec["correct"][control]) for rec in records]
        stat["cluster_randomization"] = _cluster_signflip_exact(
            learned_batches, control_batches)
        stat["accuracy_diff_learned_minus_control"] = (
            policies["learned"]["accuracy"] - policies[control]["accuracy"])
        comparisons[f"learned_vs_{control}"] = stat

    k_tensor, q_tensor = torch.tensor(all_k), torch.tensor(all_q)
    shuffle_acc = [policies[f"shuffle_{j}"]["accuracy"]
                   for j in range(n_shuffles)]
    return {
        "design": {
            "task": "progressive_transition_stream",
            "information_constraint": "operation_t_visible_only_at_reasoner_tick_t",
            "state_after_K": "frozen",
            "unit": "sample_executor_tick",
            "exact_total_budget": True,
            "mean_budget": int(mean_budget),
            "total_budget_per_batch": total_per_batch,
            "balanced_lengths": [dataset.min_steps, dataset.max_steps],
            "batch_size": int(batch_size),
            "n_batches": int(n_batches),
            "n_shuffles": int(n_shuffles),
            "primary_contrast": "learned_vs_uniform",
            "primary_test": "exact_batch_cluster_sign_flip_two_sided",
            "seed": int(seed),
        },
        "variant": model.cfg.variant,
        "soft_reference": {
            "accuracy": sum(soft_correct) / len(soft_correct),
            "mean_expected_steps": sum(all_ne) / len(all_ne),
            "corr_K_expected_steps": _pearson(
                torch.tensor(all_k), torch.tensor(all_ne)),
            "n": len(soft_correct),
        },
        "allocation": {
            "corr_K_learned_quota": _pearson(k_tensor, q_tensor),
            "mean_absolute_error_to_K": float((q_tensor - k_tensor).abs().float().mean()),
            "exact_match_rate": float(q_tensor.eq(k_tensor).float().mean()),
            "quota_min": int(q_tensor.min()),
            "quota_max": int(q_tensor.max()),
        },
        "policies": policies,
        "comparisons": comparisons,
        "shuffle_accuracy_mean": sum(shuffle_acc) / len(shuffle_acc),
        "shuffle_accuracy_min": min(shuffle_acc),
        "shuffle_accuracy_max": max(shuffle_acc),
        "records": records,
    }

