"""Evaluación online del cuello de botella cuando K está oculto."""

from __future__ import annotations

import time
from collections import defaultdict

import torch

from data.budgeted_stream import BudgetedStreamBatch, BudgetedTransitionDataset
from eval.budgeted_stream import _run_forced
from eval.compute_bottleneck import (
    _cluster_signflip_exact,
    _mcnemar_exact,
    _pearson,
    rank_yoked_allocation,
    shuffled_allocation,
    uniform_allocation,
)


def online_prefix_allocation(halt_logits_by_step: torch.Tensor,
                             total_budget: int) -> tuple[torch.Tensor, list[int]]:
    """Asigna unidades una a una consultando sólo el prefijo actual.

    `halt_logits_by_step[i, q_i-1]` es una función del estado de la muestra i
    después de exactamente q_i ticks. Las columnas futuras nunca participan en
    la elección hasta que la muestra haya recibido los ticks correspondientes.
    """
    scores = halt_logits_by_step.detach().float().cpu()
    if scores.ndim != 2:
        raise ValueError("halt_logits_by_step debe tener forma (B,N)")
    batch_size, max_steps = scores.shape
    if not batch_size <= total_budget <= batch_size * max_steps:
        raise ValueError("presupuesto fuera del rango factible")
    quotas = torch.ones(batch_size, dtype=torch.long)
    rows = torch.arange(batch_size)
    choices: list[int] = []
    for _ in range(total_budget - batch_size):
        # Logit de parada bajo => alta necesidad de continuar.
        current_need = -scores[rows, quotas - 1]
        current_need[quotas >= max_steps] = -torch.inf
        chosen = int(torch.argmax(current_need))
        if not torch.isfinite(current_need[chosen]):
            raise RuntimeError("no quedan muestras ampliables para el presupuesto")
        quotas[chosen] += 1
        choices.append(chosen)
    if int(quotas.sum()) != total_budget:
        raise RuntimeError("el asignador online violó el presupuesto")
    return quotas, choices


@torch.no_grad()
def evaluate_hidden_need(model, dataset: BudgetedTransitionDataset,
                         device: torch.device, mean_budget: int = 5,
                         n_batches: int = 20, batch_size: int = 36,
                         n_shuffles: int = 4,
                         seed: int = 20260720,
                         ablate_terminal_marker: bool = False) -> dict:
    if model.cfg.observe_length:
        raise ValueError("hidden_need exige cfg.observe_length=False")
    if ablate_terminal_marker and not dataset.terminal_markers:
        raise ValueError("no hay marcador terminal que ablacionar")
    if batch_size % (dataset.max_steps - dataset.min_steps + 1):
        raise ValueError("batch_size no permite longitudes balanceadas")
    if float(mean_budget) != dataset.mean_required_steps:
        raise ValueError("mean_budget debe igualar la media exacta de K")
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
    all_k, all_q, all_ne = [], [], []

    for batch_idx in range(n_batches):
        batch = dataset.batch(batch_size, balanced=True)
        if ablate_terminal_marker:
            operations = batch.operations.clone()
            terminal = ((operations >= dataset.n_base_ops)
                        & (operations < dataset.PAD_OP))
            operations[terminal] -= dataset.n_base_ops
            batch = BudgetedStreamBatch(
                batch.initial_state, operations, batch.lengths,
                batch.final_state, batch.intermediate_states)
        batch = batch.to(device)
        soft = model.forward_soft(batch, compute_loss=False)
        halt_logits = soft["halt_logits_by_step"].detach().cpu()
        learned, choices = online_prefix_allocation(halt_logits, total_per_batch)
        uniform = uniform_allocation(
            batch_size, total_per_batch, max_steps=dataset.max_steps)
        lengths = batch.lengths.cpu()
        oracle = lengths.clone()
        if int(oracle.sum()) != total_per_batch:
            raise RuntimeError("sum(K) no coincide con el presupuesto")

        allocations = {
            "learned_online": learned,
            "uniform": uniform,
            "oracle_exact": oracle,
            "oracle_yoked": rank_yoked_allocation(
                learned, lengths, hardest_gets_most=True),
            "anti_oracle": rank_yoked_allocation(
                learned, lengths, hardest_gets_most=False),
        }
        for j in range(n_shuffles):
            gen = torch.Generator().manual_seed(
                int(seed) + batch_idx * 10_007 + j * 1_000_003)
            allocations[f"shuffle_{j}"] = shuffled_allocation(learned, gen)

        learned_multiset = learned.sort().values
        batch_correct, batch_units = {}, {}
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
            for is_correct, k in zip(ok, lengths.tolist()):
                by_k[name][int(k)][0] += int(is_correct)
                by_k[name][int(k)][1] += 1

        all_k.extend(int(x) for x in lengths.tolist())
        all_q.extend(int(x) for x in learned.tolist())
        all_ne.extend(float(x) for x in soft["n_expected"].detach().cpu().tolist())
        records.append({
            "batch": batch_idx,
            "lengths": [int(x) for x in lengths.tolist()],
            "n_expected": [float(x) for x in soft["n_expected"].detach().cpu()],
            "allocations": {name: [int(x) for x in q.tolist()]
                            for name, q in allocations.items()},
            "online_choice_order": choices,
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
    learned_values = correct_all["learned_online"]
    learned_batches = [sum(rec["correct"]["learned_online"]) for rec in records]
    for control in ("uniform", "shuffle_0", "oracle_exact",
                    "oracle_yoked", "anti_oracle"):
        stat = _mcnemar_exact(learned_values, correct_all[control])
        control_batches = [sum(rec["correct"][control]) for rec in records]
        stat["cluster_randomization"] = _cluster_signflip_exact(
            learned_batches, control_batches)
        stat["accuracy_diff_learned_minus_control"] = (
            policies["learned_online"]["accuracy"] - policies[control]["accuracy"])
        comparisons[f"learned_online_vs_{control}"] = stat

    k_tensor = torch.tensor(all_k)
    q_tensor = torch.tensor(all_q)
    shuffle_acc = [policies[f"shuffle_{j}"]["accuracy"]
                   for j in range(n_shuffles)]
    return {
        "design": {
            "task": "hidden_length_transition_stream",
            "length_visible_to_scheduler": False,
            "completion_supervision": False,
            "terminal_information": (
                "LAST marker ablated at evaluation"
                if ablate_terminal_marker else
                ("LAST marker visible only with operation K"
                 if dataset.terminal_markers else
                 "no terminal marker; completion observable only after zero progress")),
            "terminal_marker_ablation": bool(ablate_terminal_marker),
            "terminal_marker_present": bool(
                dataset.terminal_markers and not ablate_terminal_marker),
            "allocation": "online one-unit priority; current prefix score only",
            "unit": "sample_executor_tick",
            "exact_total_budget": True,
            "mean_budget": int(mean_budget),
            "total_budget_per_batch": total_per_batch,
            "batch_size": int(batch_size),
            "n_batches": int(n_batches),
            "n_shuffles": int(n_shuffles),
            "primary_contrast": "learned_online_vs_uniform",
            "primary_test": "exact_batch_cluster_sign_flip_two_sided",
            "seed": int(seed),
        },
        "variant": model.cfg.variant,
        "soft_reference": {
            "mean_expected_steps": sum(all_ne) / len(all_ne),
            "corr_K_expected_steps": _pearson(
                torch.tensor(all_k), torch.tensor(all_ne)),
        },
        "allocation": {
            "corr_K_learned_quota": _pearson(k_tensor, q_tensor),
            "mean_absolute_error_to_K": float(
                (q_tensor - k_tensor).abs().float().mean()),
            "exact_match_rate": float(q_tensor.eq(k_tensor).float().mean()),
            "underallocation_rate": float((q_tensor < k_tensor).float().mean()),
            "overallocation_rate": float((q_tensor > k_tensor).float().mean()),
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
