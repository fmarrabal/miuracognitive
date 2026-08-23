"""Evaluación causal de asignación bajo un presupuesto compartido de reasoner.

La política se planifica sin etiquetas a partir de p(parar en n) del rollout
PonderNet. Después se impone una cuota entera por muestra y se ejecuta solo el
sub-batch activo. El contraste principal, ``learned`` frente a ``shuffle_0``,
mantiene exactamente constantes:

  * modelo, pesos, ejemplos y orden del batch;
  * suma total de pasos muestra-reasoner;
  * multiconjunto de cuotas individuales.

Solo cambia qué ejemplo recibe cada cuota. Por ello una diferencia de accuracy
es un efecto causal de la asignación dentro del batch evaluado, no de disponer de
más cómputo. El rollout de planificación es común a todos los brazos y se reporta
por separado: este experimento prueba el valor contrafactual de la política, no
todavía la eficiencia end-to-end de un scheduler desplegado.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

import torch


def optimal_halt_allocation(halt_probs: torch.Tensor, total_budget: int,
                            min_steps: int = 1) -> torch.Tensor:
    """MAP conjunto de cuotas bajo ``sum(cuotas) == total_budget``.

    Resuelve exactamente mediante programación dinámica

        argmax_q  Σ_i log p_i(q_i),  s.a. Σ_i q_i = total_budget.

    ``halt_probs`` tiene forma (B, N), donde la columna j representa parar tras
    j+1 pasos. El resultado vive en CPU y contiene enteros en [min_steps, N].
    """
    if halt_probs.ndim != 2:
        raise ValueError("halt_probs debe tener forma (B, N)")
    probs = halt_probs.detach().float().cpu()
    B, N = probs.shape
    if not 1 <= min_steps <= N:
        raise ValueError(f"min_steps debe estar en [1, {N}]")
    lo, hi = B * min_steps, B * N
    if not lo <= int(total_budget) <= hi:
        raise ValueError(f"presupuesto {total_budget} fuera de [{lo}, {hi}]")

    # Normaliza por robustez ante un early break con remanente <1e-4.
    probs = probs.clamp_min(1e-12)
    probs = probs / probs.sum(dim=1, keepdim=True)
    logp = probs.log()
    neg_inf = float("-inf")
    dp = torch.full((B + 1, total_budget + 1), neg_inf, dtype=torch.float64)
    back = torch.full((B + 1, total_budget + 1), -1, dtype=torch.int16)
    dp[0, 0] = 0.0

    for i in range(1, B + 1):
        min_used = i * min_steps
        max_used = min(total_budget, i * N)
        prev_min = (i - 1) * min_steps
        prev_max = min(total_budget, (i - 1) * N)
        for used in range(min_used, max_used + 1):
            best_score, best_n = neg_inf, -1
            n_lo = max(min_steps, used - prev_max)
            n_hi = min(N, used - prev_min)
            for n in range(n_lo, n_hi + 1):
                prev = float(dp[i - 1, used - n])
                if math.isinf(prev) and prev < 0:
                    continue
                score = prev + float(logp[i - 1, n - 1])
                if score > best_score:
                    best_score, best_n = score, n
            if best_n >= 0:
                dp[i, used] = best_score
                back[i, used] = best_n

    if int(back[B, total_budget]) < 0:
        raise RuntimeError("no se encontró una asignación factible")
    out = torch.empty(B, dtype=torch.long)
    used = int(total_budget)
    for i in range(B, 0, -1):
        n = int(back[i, used])
        out[i - 1] = n
        used -= n
    assert used == 0 and int(out.sum()) == int(total_budget)
    return out


def uniform_allocation(batch_size: int, total_budget: int,
                       max_steps: int, min_steps: int = 1) -> torch.Tensor:
    """Cuotas lo más iguales posible, con suma exacta."""
    lo, hi = batch_size * min_steps, batch_size * max_steps
    if not lo <= int(total_budget) <= hi:
        raise ValueError(f"presupuesto {total_budget} fuera de [{lo}, {hi}]")
    base, rem = divmod(int(total_budget), int(batch_size))
    out = torch.full((batch_size,), base, dtype=torch.long)
    if rem:
        out[:rem] += 1
    if int(out.min()) < min_steps or int(out.max()) > max_steps:
        raise RuntimeError("la asignación uniforme violó los límites")
    return out


def shuffled_allocation(allocation: torch.Tensor,
                        generator: torch.Generator) -> torch.Tensor:
    """Permuta el mismo multiconjunto de cuotas."""
    perm = torch.randperm(allocation.numel(), generator=generator)
    return allocation.index_select(0, perm)


def rank_yoked_allocation(allocation: torch.Tensor, difficulty: torch.Tensor,
                          hardest_gets_most: bool = True) -> torch.Tensor:
    """Reasigna el mismo multiconjunto por ranking de dificultad observada."""
    q_sorted = allocation.sort().values
    order = difficulty.detach().cpu().argsort()
    if not hardest_gets_most:
        q_sorted = q_sorted.flip(0)
    out = torch.empty_like(q_sorted)
    out[order] = q_sorted
    return out


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    if a.numel() < 2 or a.std(unbiased=False) < 1e-8 or b.std(unbiased=False) < 1e-8:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean()
                 / (a.std(unbiased=False) * b.std(unbiased=False)))


def _mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """McNemar exacto bilateral para resultados binarios pareados."""
    a_only = sum(bool(x) and not bool(y) for x, y in zip(a, b))
    b_only = sum(bool(y) and not bool(x) for x, y in zip(a, b))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(a_only, b_only) + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only,
            "discordant": n, "p_exact_two_sided": p}


def _cluster_signflip_exact(a: list[int], b: list[int]) -> dict:
    """Randomización exacta por batch para respetar la asignación acoplada.

    Bajo H0, intercambiar los nombres de los dos brazos dentro de cada batch
    cambia el signo de su diferencia de aciertos. La convolución discreta evita
    enumerar 2**n_batches combinaciones explícitamente.
    """
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0]
    observed = abs(sum(diffs))
    if not nonzero:
        return {"batch_differences": diffs, "observed_correct_diff": 0,
                "nonzero_batches": 0, "p_exact_two_sided": 1.0}
    distribution = {0: 1}
    for d in nonzero:
        nxt = defaultdict(int)
        for current, count in distribution.items():
            nxt[current + d] += count
            nxt[current - d] += count
        distribution = dict(nxt)
    total = 2 ** len(nonzero)
    extreme = sum(count for value, count in distribution.items()
                  if abs(value) >= observed)
    return {"batch_differences": diffs,
            "observed_correct_diff": sum(diffs),
            "nonzero_batches": len(nonzero),
            "p_exact_two_sided": extreme / total}


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Ajuste step-down de Holm, monótono y limitado a 1."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running = len(ordered), 0.0
    adjusted = {}
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * float(p)))
        adjusted[name] = running
    return adjusted


def _last_answer_correct(logits: torch.Tensor, targets: torch.Tensor,
                         answer_offset: int, answer_size: int) -> torch.Tensor:
    valid = targets != -1
    if not valid.any(dim=1).all():
        raise ValueError("cada muestra debe tener al menos un target supervisado")
    last_pos = valid.long().cumsum(dim=1).argmax(dim=1)
    rows = torch.arange(targets.shape[0], device=targets.device)
    pred = answer_offset + logits[..., answer_offset:answer_offset + answer_size].argmax(dim=-1)
    return pred[rows, last_pos].eq(targets[rows, last_pos])


def _run_forced(model, inp: torch.Tensor, tgt: torch.Tensor,
                quotas: torch.Tensor, answer_offset: int,
                answer_size: int) -> tuple[list[bool], float, int]:
    quotas_dev = quotas.to(device=inp.device)
    if inp.device.type == "cuda":
        torch.cuda.synchronize(inp.device)
    t0 = time.perf_counter()
    logits, _ = model(inp, None, forced_steps=quotas_dev)
    if inp.device.type == "cuda":
        torch.cuda.synchronize(inp.device)
    elapsed = time.perf_counter() - t0
    units = model._last_reasoner_step_units
    if units != int(quotas.sum()):
        raise RuntimeError(f"presupuesto ejecutado {units} != asignado {int(quotas.sum())}")
    correct = _last_answer_correct(logits, tgt, answer_offset, answer_size)
    return [bool(x) for x in correct.cpu().tolist()], elapsed, int(units)


@torch.no_grad()
def evaluate_compute_bottleneck(model, ds, device: torch.device,
                                mean_budgets: list[int] | tuple[int, ...],
                                n_batches: int = 20, batch_size: int = 32,
                                n_shuffles: int = 4, seed: int = 20260714,
                                min_steps: int = 1,
                                primary_mean_budget: int = 6,
                                primary_control: str = "shuffle_0") -> dict:
    """Ejecuta el experimento causal completo sobre un modelo ya entrenado."""
    if not model.cfg.use_adaptive_depth:
        raise ValueError("el modelo no tiene reasoner adaptativo")
    if n_shuffles < 1:
        raise ValueError("n_shuffles debe ser >=1")
    max_steps = int(model.cfg.max_halt_steps)
    budgets = [int(x) for x in mean_budgets]
    if any(x < min_steps or x > max_steps for x in budgets):
        raise ValueError(f"mean_budgets debe estar en [{min_steps}, {max_steps}]")
    if int(primary_mean_budget) not in budgets:
        raise ValueError("primary_mean_budget debe estar incluido en mean_budgets")
    if primary_control not in ("shuffle_0", "uniform"):
        raise ValueError("primary_control debe ser shuffle_0 o uniform")

    model.eval()
    a0, a_size = int(ds.answer_offset), int(ds.answer_size)
    out = {
        "design": {
            "unit": "sample_reasoner_step",
            "backbone_compute": "fixed_and_excluded",
            "planner": "soft_halt_distribution_shared_across_conditions",
            "primary_contrast": f"learned_vs_{primary_control}",
            "primary_mean_budget": int(primary_mean_budget),
            "primary_test": "exact_batch_cluster_sign_flip_two_sided",
            "paired": True,
            "exact_total_budget": True,
            "yoked_quota_multiset_for_shuffles_oracle_and_anti_oracle": True,
            "mean_budgets": budgets,
            "batch_size": int(batch_size),
            "n_batches": int(n_batches),
            "n_shuffles": int(n_shuffles),
            "seed": int(seed),
            "min_steps": int(min_steps),
            "max_steps": max_steps,
        },
        "soft_reference": {"correct": 0, "n": 0},
        "budgets": {},
    }

    # Cada presupuesto ve exactamente los mismos batches: reconstruimos el
    # estado RNG del dataset antes de cada barrido.
    initial_ds_state = ds.gen.get_state().clone()
    for mean_budget in budgets:
        ds.gen.set_state(initial_ds_state.clone())
        total_budget = int(mean_budget * batch_size)
        policies_correct: dict[str, list[bool]] = defaultdict(list)
        policies_units: dict[str, int] = defaultdict(int)
        policies_time: dict[str, float] = defaultdict(float)
        bucket_counts: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0]))
        records = []
        all_k, all_learned_q = [], []
        soft_correct_budget = []

        for batch_idx in range(n_batches):
            inp, tgt, difficulty = ds.batch(batch_size)
            inp, tgt = inp.to(device), tgt.to(device)

            # Planificación común, sin targets. Copiamos antes de que las rutas
            # forzadas sobrescriban los diagnósticos del último forward.
            soft_logits, _ = model(inp, None)
            halt_probs = model._last_halt_dist.detach().float().cpu()
            if halt_probs.shape[1] < max_steps:
                halt_probs = torch.nn.functional.pad(
                    halt_probs, (0, max_steps - halt_probs.shape[1]))
            soft_correct = _last_answer_correct(soft_logits, tgt, a0, a_size)
            soft_correct_list = [bool(x) for x in soft_correct.cpu().tolist()]
            soft_correct_budget.extend(soft_correct_list)

            learned = optimal_halt_allocation(
                halt_probs, total_budget=total_budget, min_steps=min_steps)
            uniform = uniform_allocation(
                batch_size, total_budget, max_steps=max_steps, min_steps=min_steps)
            oracle = rank_yoked_allocation(learned, difficulty, hardest_gets_most=True)
            anti = rank_yoked_allocation(learned, difficulty, hardest_gets_most=False)
            allocations = {
                "learned": learned,
                "uniform": uniform,
                "oracle": oracle,
                "anti_oracle": anti,
            }
            for j in range(n_shuffles):
                g = torch.Generator().manual_seed(
                    int(seed) + mean_budget * 1_000_003 + batch_idx * 10_007 + j)
                allocations[f"shuffle_{j}"] = shuffled_allocation(learned, g)

            expected_multiset = learned.sort().values
            for name, q in allocations.items():
                if int(q.sum()) != total_budget:
                    raise RuntimeError(f"{name}: suma de cuotas incorrecta")
                if name not in ("uniform", "learned") and not torch.equal(
                        q.sort().values, expected_multiset):
                    raise RuntimeError(f"{name}: el multiconjunto de cuotas cambió")

            batch_correct = {}
            batch_units = {}
            for name, q in allocations.items():
                correct, elapsed, units = _run_forced(
                    model, inp, tgt, q, a0, a_size)
                batch_correct[name] = correct
                batch_units[name] = units
                policies_correct[name].extend(correct)
                policies_units[name] += units
                policies_time[name] += elapsed
                for ok, k in zip(correct, difficulty.tolist()):
                    bname = ds.bucket(int(k))
                    bucket_counts[name][bname][0] += int(ok)
                    bucket_counts[name][bname][1] += 1

            all_k.extend(int(x) for x in difficulty.tolist())
            all_learned_q.extend(int(x) for x in learned.tolist())
            records.append({
                "batch": batch_idx,
                "difficulty": [int(x) for x in difficulty.tolist()],
                "halt_expected": [float(x) for x in
                                  (halt_probs * torch.arange(1, max_steps + 1)).sum(dim=1).tolist()],
                "halt_probs": [[float(y) for y in row] for row in halt_probs.tolist()],
                "allocations": {k: [int(x) for x in v.tolist()]
                                for k, v in allocations.items()},
                "correct": batch_correct,
                "reasoner_step_units": batch_units,
            })

        summaries = {}
        for name, correct in policies_correct.items():
            n = len(correct)
            bc = {}
            for bname, (c, bn) in bucket_counts[name].items():
                bc[bname] = {"accuracy": c / bn if bn else 0.0,
                             "correct": c, "n": bn}
            summaries[name] = {
                "accuracy": sum(correct) / n if n else 0.0,
                "correct": int(sum(correct)),
                "n": n,
                "reasoner_step_units": int(policies_units[name]),
                "mean_steps": policies_units[name] / n if n else 0.0,
                "wall_time_s": float(policies_time[name]),
                "buckets": bc,
            }

        learned_correct = policies_correct["learned"]
        comparisons = {}
        for control in ["shuffle_0", "uniform", "oracle", "anti_oracle"]:
            control_correct = policies_correct[control]
            stat = _mcnemar_exact(learned_correct, control_correct)
            learned_by_batch = [sum(r["correct"]["learned"]) for r in records]
            control_by_batch = [sum(r["correct"][control]) for r in records]
            stat["cluster_randomization"] = _cluster_signflip_exact(
                learned_by_batch, control_by_batch)
            stat["accuracy_diff_learned_minus_control"] = (
                summaries["learned"]["accuracy"] - summaries[control]["accuracy"])
            comparisons[f"learned_vs_{control}"] = stat
        shuffle_accs = [summaries[f"shuffle_{j}"]["accuracy"]
                        for j in range(n_shuffles)]

        out["budgets"][str(mean_budget)] = {
            "total_budget_per_batch": total_budget,
            "policies": summaries,
            "comparisons": comparisons,
            "shuffle_accuracy_mean": sum(shuffle_accs) / len(shuffle_accs),
            "shuffle_accuracy_min": min(shuffle_accs),
            "shuffle_accuracy_max": max(shuffle_accs),
            "allocation": {
                "corr_difficulty_learned_quota": _pearson(
                    torch.tensor(all_k), torch.tensor(all_learned_q)),
                "quota_min": min(all_learned_q),
                "quota_max": max(all_learned_q),
                "n_unique_quotas": len(set(all_learned_q)),
            },
            "soft_reference_accuracy": sum(soft_correct_budget) / len(soft_correct_budget),
            "records": records,
        }

    family = {
        str(b): (out["budgets"][str(b)]["comparisons"]
                 [f"learned_vs_{primary_control}"]
                 ["cluster_randomization"]["p_exact_two_sided"])
        for b in budgets
    }
    adjusted = _holm_adjust(family)
    for b in budgets:
        out["budgets"][str(b)]["comparisons"][f"learned_vs_{primary_control}"][
            "p_holm_across_budgets"] = adjusted[str(b)]
    out["multiplicity"] = {
        "family": f"learned_vs_{primary_control}_across_mean_budgets",
        "method": "Holm",
        "raw_cluster_p": family,
        "adjusted_p": adjusted,
        "confirmatory_budget": int(primary_mean_budget),
        "other_budgets": "dose_response_exploratory",
    }

    # El soft reference es idéntico entre presupuestos por restaurar el RNG.
    first = out["budgets"][str(budgets[0])]
    soft_n = n_batches * batch_size
    soft_acc = first["soft_reference_accuracy"]
    out["soft_reference"] = {
        "accuracy": soft_acc,
        "correct": int(round(soft_acc * soft_n)),
        "n": soft_n,
        "note": "mezcla PonderNet; no es cómputo duro ni brazo causal",
    }
    return out
