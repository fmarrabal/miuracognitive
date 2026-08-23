"""Benchmark endógeno: K oculto, sin etiqueta de finalización.

El scheduler aprende sólo de la pérdida de tarea esperada y de una restricción
de presupuesto medio. En evaluación, las unidades se asignan online consultando
únicamente el score del prefijo ejecutado de cada muestra.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import platform
import statistics
import time

import torch

from data.budgeted_stream import BudgetedTransitionDataset
from eval.hidden_need import evaluate_hidden_need
from eval.compute_bottleneck import _cluster_signflip_exact
from model.budgeted_stream import BudgetedStreamConfig, BudgetedStreamReasoner


VARIANTS = ("gating_wm", "hbp_first", "hbp_full")


def _csv_strings(value: str) -> list[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("lista vacía")
    return out


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(x) for x in _csv_strings(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("usa enteros separados por comas") from exc


def _atomic_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def train_one(variant: str, seed: int, args,
              device: torch.device) -> tuple[dict, str]:
    torch.manual_seed(seed)
    train_ds = BudgetedTransitionDataset(
        n_states=args.n_states, min_steps=1, max_steps=args.max_steps,
        seed=seed, terminal_markers=args.terminal_marker)
    cfg = BudgetedStreamConfig(
        n_states=train_ds.n_states, n_ops=train_ds.n_ops,
        max_steps=train_ds.max_steps, d_model=args.d_model,
        variant=variant, observe_length=False, step_cost=0.0,
        auxiliary_weight=args.auxiliary_weight, completion_weight=0.0)
    model = BudgetedStreamReasoner(cfg).to(device=device, dtype=torch.float32)
    model.pin_fp32()

    executor_modules = (model.state_emb, model.op_emb, model.executor,
                        model.state_norm, model.readout)
    executor_params = [p for module in executor_modules for p in module.parameters()]
    executor_ids = {id(p) for p in executor_params}
    controller_params = [p for p in model.parameters()
                         if p.requires_grad and id(p) not in executor_ids]
    reuse_path = os.path.join(
        args.out_dir, "checkpoints", f"gating_wm_seed{seed}.pt")
    reuse_executor = bool(
        args.reuse_paired_executor and variant != "gating_wm"
        and os.path.exists(reuse_path))
    if reuse_executor:
        source = torch.load(reuse_path, map_location="cpu", weights_only=True)
        source_cfg = source["config"]
        for key in ("n_states", "n_ops", "max_steps", "d_model", "observe_length"):
            if source_cfg[key] != asdict(cfg)[key]:
                raise RuntimeError(f"executor fuente incompatible en {key}")
        state = model.state_dict()
        shared_prefixes = (
            "state_emb.", "op_emb.", "executor.", "state_norm.", "readout.")
        for name, value in source["state_dict"].items():
            if name.startswith(shared_prefixes):
                state[name] = value
        model.load_state_dict(state)
        # El generador de datos es local. Consumir los batches de pretraining
        # alinea exactamente la fase de controller con el control pareado.
        for _ in range(args.pretrain_steps):
            train_ds.batch(args.batch_size, balanced=True)
        optimizer = torch.optim.AdamW(
            controller_params, lr=args.policy_lr, weight_decay=1e-4,
            betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(
            executor_params, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    target_budget = train_ds.mean_required_steps
    trace = {key: [] for key in (
        "step", "phase", "loss", "expected_task", "auxiliary",
        "completion_diagnostic", "budget_penalty", "mean_expected_steps",
        "corr_K_expected_steps", "soft_accuracy")}
    started = time.time()
    total_updates = args.pretrain_steps + args.steps
    first_update = args.pretrain_steps + 1 if reuse_executor else 1
    model.train()

    for step in range(first_update, total_updates + 1):
        pretraining = step <= args.pretrain_steps
        if step == args.pretrain_steps + 1 and not reuse_executor:
            optimizer = torch.optim.AdamW(
                controller_params, lr=args.policy_lr, weight_decay=1e-4,
                betas=(0.9, 0.95))
        phase_step = step if pretraining else step - args.pretrain_steps
        phase_total = args.pretrain_steps if pretraining else args.steps
        warm = min(1.0, phase_step / max(1, args.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(
            math.pi * phase_step / max(1, phase_total)))
        base_lr = args.lr if pretraining else args.policy_lr
        lr = base_lr * warm * cosine
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = train_ds.batch(args.batch_size, balanced=True).to(device)
        if pretraining:
            result = None
            loss = model.executor_auxiliary_loss(batch)
        else:
            result = model.forward_soft(batch, compute_loss=True)
            budget_penalty = (
                result["n_expected"].mean() - target_budget).square()
            # No completion loss y ningún acceso a K desde el controller.
            loss = (result["losses"]["expected_task"]
                    + args.budget_weight * budget_penalty)
            if model.use_hbp:
                loss = (loss + cfg.beta_intero * result["losses"]["intero"]
                        + cfg.beta_homeo * result["losses"]["homeo"]
                        + cfg.beta_stab * result["losses"]["stab"])

        model.zero_grad(set_to_none=True)
        loss.backward()
        active_params = executor_params if pretraining else controller_params
        torch.nn.utils.clip_grad_norm_(active_params, 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == total_updates:
            if result is None:
                with torch.no_grad():
                    result = model.forward_soft(batch, compute_loss=True)
            budget_penalty = (
                result["n_expected"].mean() - target_budget).square()
            with torch.no_grad():
                pred = result["probabilities"].argmax(dim=-1)
                acc = float(pred.eq(batch.final_state).float().mean())
                ne = result["n_expected"].detach().float().cpu()
                k = batch.lengths.detach().float().cpu()
                corr = (0.0 if ne.std(unbiased=False) < 1e-8 else
                        float(((ne-ne.mean())*(k-k.mean())).mean()
                              /(ne.std(unbiased=False)*k.std(unbiased=False))))
            values = {
                "step": step,
                "phase": "executor" if pretraining else "policy",
                "loss": float(loss.detach()),
                "expected_task": float(result["losses"]["expected_task"].detach()),
                "auxiliary": float(result["losses"]["auxiliary"].detach()),
                "completion_diagnostic": float(
                    result["losses"]["completion"].detach()),
                "budget_penalty": float(budget_penalty.detach()),
                "mean_expected_steps": float(ne.mean()),
                "corr_K_expected_steps": corr,
                "soft_accuracy": acc,
            }
            for key, value in values.items():
                trace[key].append(value)
            print(f"  [{variant} s{seed} {values['phase']}] "
                  f"{step:4d}/{total_updates} loss={values['loss']:.3f} "
                  f"task={values['expected_task']:.3f} acc={acc:.3f} "
                  f"E[n]={values['mean_expected_steps']:.2f} "
                  f"corr(K,E[n])={corr:+.2f}", flush=True)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{variant}_seed{seed}.pt")
    torch.save({
        "state_dict": model.state_dict(), "config": asdict(cfg),
        "meta": {
            "variant": variant, "seed": seed,
            "steps": args.steps, "pretrain_steps": args.pretrain_steps,
            "task": ("hidden_length_transition_stream" if args.terminal_marker
                     else "implicit_termination_transition_stream"),
            "terminal_marker_used": bool(args.terminal_marker),
            "executor_reused_from": (os.path.abspath(reuse_path)
                                     if reuse_executor else None),
            "length_supervision_used": False,
            "completion_supervision_used": False,
        },
    }, ckpt_path)

    eval_ds = BudgetedTransitionDataset(
        n_states=args.n_states, min_steps=1, max_steps=args.max_steps,
        seed=seed + 100_000, terminal_markers=args.terminal_marker)
    evaluation = evaluate_hidden_need(
        model, eval_ds, device=device, mean_budget=int(target_budget),
        n_batches=args.eval_batches, batch_size=args.batch_size,
        n_shuffles=args.n_shuffles, seed=args.eval_seed + seed)
    evaluation["training"] = {
        "config": asdict(cfg), "seed": seed,
        "steps": args.steps, "pretrain_steps": args.pretrain_steps,
        "batch_size": args.batch_size, "lr": args.lr,
        "policy_lr": args.policy_lr, "budget_weight": args.budget_weight,
        "target_mean_budget": target_budget,
        "terminal_marker_used": bool(args.terminal_marker),
        "executor_reused_from": (os.path.abspath(reuse_path)
                                 if reuse_executor else None),
        "executor_pretrain_updates_this_run": (
            0 if reuse_executor else args.pretrain_steps),
        "length_supervision_used": False,
        "completion_supervision_used": False,
        "elapsed_s": time.time() - started, "trace": trace,
    }
    evaluation["provenance"] = {
        "checkpoint": os.path.abspath(ckpt_path),
        "torch": torch.__version__, "python": platform.python_version(),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device)
                        if device.type == "cuda" else "cpu"),
    }
    return evaluation, ckpt_path


def _aggregate(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    for result in results:
        grouped[result["variant"]].append(result)
    summary = {}
    for variant, cells in grouped.items():
        if not cells:
            continue
        deltas = [c["comparisons"]["learned_online_vs_uniform"]
                  ["accuracy_diff_learned_minus_control"] for c in cells]
        learned = [c["policies"]["learned_online"]["accuracy"] for c in cells]
        uniform = [c["policies"]["uniform"]["accuracy"] for c in cells]
        summary[variant] = {
            "n_seeds": len(cells),
            "seeds": [c["training"]["seed"] for c in cells],
            "mean_learned_accuracy": statistics.mean(learned),
            "mean_uniform_accuracy": statistics.mean(uniform),
            "mean_delta_accuracy": statistics.mean(deltas),
            "sd_delta_accuracy": (statistics.stdev(deltas)
                                  if len(deltas) > 1 else 0.0),
            "mean_corr_K_quota": statistics.mean(
                c["allocation"]["corr_K_learned_quota"] for c in cells),
            "mean_exact_match_rate": statistics.mean(
                c["allocation"]["exact_match_rate"] for c in cells),
            "per_seed_delta": deltas,
        }
    return summary


def _paired_architecture_contrasts(results: list[dict]) -> dict:
    by_cell = {(r["variant"], r["training"]["seed"]): r for r in results}
    seeds = sorted({r["training"]["seed"] for r in results})
    pairs = (("hbp_first", "gating_wm"),
             ("hbp_full", "gating_wm"),
             ("hbp_full", "hbp_first"))
    out = {}
    for numerator, denominator in pairs:
        if any((numerator, s) not in by_cell or
               (denominator, s) not in by_cell for s in seeds):
            continue
        correct_diffs, quota_change, quota_mae = [], [], []
        for seed in seeds:
            a, b = by_cell[(numerator, seed)], by_cell[(denominator, seed)]
            a_correct = [x for rec in a["records"]
                         for x in rec["correct"]["learned_online"]]
            b_correct = [x for rec in b["records"]
                         for x in rec["correct"]["learned_online"]]
            qa = [x for rec in a["records"]
                  for x in rec["allocations"]["learned_online"]]
            qb = [x for rec in b["records"]
                  for x in rec["allocations"]["learned_online"]]
            if len(a_correct) != len(b_correct) or len(qa) != len(qb):
                raise RuntimeError("celdas pareadas con distinto tamaño")
            correct_diffs.append(int(sum(a_correct) - sum(b_correct)))
            quota_change.append(sum(x != y for x, y in zip(qa, qb)) / len(qa))
            quota_mae.append(sum(abs(x-y) for x, y in zip(qa, qb)) / len(qa))
        n_per_seed = len(a_correct)
        max_abs_index = max(
            range(len(correct_diffs)), key=lambda i: abs(correct_diffs[i]))
        sensitivity_diffs = [x for i, x in enumerate(correct_diffs)
                             if i != max_abs_index]
        out[f"{numerator}_minus_{denominator}"] = {
            "per_seed_correct_difference": correct_diffs,
            "mean_accuracy_difference": statistics.mean(correct_diffs) / n_per_seed,
            "median_accuracy_difference": statistics.median(
                correct_diffs) / n_per_seed,
            "positive_negative_zero_seeds": {
                "positive": sum(x > 0 for x in correct_diffs),
                "negative": sum(x < 0 for x in correct_diffs),
                "zero": sum(x == 0 for x in correct_diffs),
            },
            "seed_level_exact_signflip": _cluster_signflip_exact(
                correct_diffs, [0] * len(correct_diffs)),
            "leave_largest_absolute_seed_out": {
                "removed_seed": seeds[max_abs_index],
                "removed_correct_difference": correct_diffs[max_abs_index],
                "mean_accuracy_difference": (
                    statistics.mean(sensitivity_diffs) / n_per_seed),
                "seed_level_exact_signflip": _cluster_signflip_exact(
                    sensitivity_diffs, [0] * len(sensitivity_diffs)),
            },
            "per_seed_quota_change_rate": quota_change,
            "per_seed_mean_absolute_quota_change": quota_mae,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=_csv_strings, default=list(VARIANTS))
    ap.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--pretrain-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=36)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--n-shuffles", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=9)
    ap.add_argument("--n-states", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--policy-lr", type=float, default=3e-3)
    ap.add_argument("--budget-weight", type=float, default=2.0)
    ap.add_argument("--auxiliary-weight", type=float, default=0.5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--eval-seed", type=int, default=20260720)
    ap.add_argument("--out-dir", default="results_hidden_need")
    ap.add_argument("--device", default=None)
    ap.add_argument("--terminal-marker", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="usa LAST en K; --no-terminal-marker fuerza terminación implícita")
    ap.add_argument("--reuse-paired-executor",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="para HBP carga el executor congelado de gating_wm de la misma seed")
    ap.add_argument("--resume-existing",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="reutiliza JSON/checkpoint completos ya presentes en out-dir")
    args = ap.parse_args()
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        ap.error(f"variantes desconocidas: {unknown}")
    if args.batch_size % args.max_steps:
        ap.error("batch-size debe ser múltiplo de max-steps")
    if min(args.steps, args.pretrain_steps, args.eval_batches,
           args.n_shuffles) < 1:
        ap.error("steps/pretrain/eval-batches/n-shuffles deben ser >=1")

    device = torch.device(args.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    print("Benchmark endógeno de necesidad de cómputo")
    print(f"  device={device} variantes={args.variants} seeds={args.seeds}")
    terminal_desc = "LAST causal en K" if args.terminal_marker else "sin LAST"
    print(f"  K oculto; {terminal_desc}; sin completion loss; "
          "asignación online; presupuesto medio=5")
    results = []
    for variant in args.variants:
        for seed in args.seeds:
            path = os.path.join(args.out_dir, f"{variant}_seed{seed}.json")
            ckpt = os.path.join(
                args.out_dir, "checkpoints", f"{variant}_seed{seed}.pt")
            if (args.resume_existing and os.path.exists(path)
                    and os.path.exists(ckpt)):
                with open(path, encoding="utf-8") as f:
                    result = json.load(f)
                print(f"  REUSE {variant} s{seed}: {path} | {ckpt}")
            else:
                result, ckpt = train_one(variant, seed, args, device)
                _atomic_json(result, path)
            results.append(result)
            cmp = result["comparisons"]["learned_online_vs_uniform"]
            p = cmp["cluster_randomization"]["p_exact_two_sided"]
            print(f"  RESULT {variant} s{seed}: "
                  f"learned={result['policies']['learned_online']['accuracy']:.3f} "
                  f"uniform={result['policies']['uniform']['accuracy']:.3f} "
                  f"Δ={cmp['accuracy_diff_learned_minus_control']:+.3f} "
                  f"p={p:.3g} corr(K,q)="
                  f"{result['allocation']['corr_K_learned_quota']:+.3f}")
            print(f"    {path} | {ckpt}")

    aggregate = {
        "design": {
            "variants": args.variants, "seeds": args.seeds,
            "steps": args.steps, "pretrain_steps": args.pretrain_steps,
            "max_steps": args.max_steps, "mean_budget": 5.0,
            "length_supervision_used": False,
            "completion_supervision_used": False,
            "terminal_marker_used": bool(args.terminal_marker),
            "reuse_paired_executor": bool(args.reuse_paired_executor),
            "resume_existing": bool(args.resume_existing),
            "allocation": "online_prefix_only",
            "primary_contrast": "learned_online_vs_uniform",
        },
        "summary": _aggregate(results),
        "paired_architecture_contrasts": _paired_architecture_contrasts(results),
    }
    _atomic_json(aggregate, os.path.join(args.out_dir, "summary.json"))
    print(f"Resumen: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
