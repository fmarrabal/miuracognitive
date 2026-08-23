"""Benchmark causal donde cada tick revela y ejecuta una transición nueva.

Uso piloto:

  python budgeted_stream_benchmark.py --variants gating_wm,hbp_full --seeds 0 --steps 1000

El protocolo por defecto entrena tres variantes × tres semillas y evalúa con
presupuesto medio 5 sobre batches balanceados K=1..9.
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
from eval.budgeted_stream import evaluate_budgeted_stream
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


def train_one(variant: str, seed: int, args, device: torch.device) -> tuple[dict, str]:
    torch.manual_seed(seed)
    train_ds = BudgetedTransitionDataset(
        n_states=args.n_states, min_steps=1, max_steps=args.max_steps, seed=seed)
    cfg = BudgetedStreamConfig(
        n_states=train_ds.n_states, n_ops=train_ds.n_ops,
        max_steps=train_ds.max_steps, d_model=args.d_model,
        variant=variant, step_cost=args.step_cost,
        auxiliary_weight=args.auxiliary_weight,
        completion_weight=args.completion_weight)
    model = BudgetedStreamReasoner(cfg).to(device=device, dtype=torch.float32)
    model.pin_fp32()
    executor_modules = (model.state_emb, model.op_emb, model.executor,
                        model.state_norm, model.readout)
    executor_params = [p for module in executor_modules for p in module.parameters()]
    executor_ids = {id(p) for p in executor_params}
    controller_params = [p for p in model.parameters()
                         if p.requires_grad and id(p) not in executor_ids]
    optimizer = torch.optim.AdamW(
        executor_params, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    trace = {"step": [], "loss": [], "soft_acc": [],
             "mean_expected_steps": [], "corr_K_expected_steps": [],
             "phase": [], "expected_task": [], "auxiliary": [],
             "completion": []}
    started = time.time()
    model.train()
    total_updates = args.pretrain_steps + args.steps

    for step in range(1, total_updates + 1):
        pretraining = step <= args.pretrain_steps
        if step == args.pretrain_steps + 1:
            optimizer = torch.optim.AdamW(
                controller_params, lr=args.policy_lr, weight_decay=1e-4,
                betas=(0.9, 0.95))
        phase_step = step if pretraining else step - args.pretrain_steps
        phase_total = args.pretrain_steps if pretraining else args.steps
        warm = min(1.0, phase_step / max(1, args.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * phase_step / max(1, phase_total)))
        base_lr = args.lr if pretraining else args.policy_lr
        lr = base_lr * warm * cosine
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = train_ds.batch(args.batch_size, balanced=True).to(device)
        result = model.forward_soft(batch, compute_loss=True)
        if pretraining:
            loss = result["losses"]["auxiliary"]
        elif args.policy_objective == "completion":
            loss = result["losses"]["completion"]
        else:
            loss = result["losses"]["total"]
        model.zero_grad(set_to_none=True)
        loss.backward()
        active_params = executor_params if pretraining else controller_params
        torch.nn.utils.clip_grad_norm_(active_params, 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == total_updates:
            with torch.no_grad():
                pred = result["probabilities"].argmax(dim=-1)
                acc = float(pred.eq(batch.final_state).float().mean())
                ne = result["n_expected"].detach().float().cpu()
                k = batch.lengths.detach().float().cpu()
                corr = (0.0 if ne.std(unbiased=False) < 1e-8 else
                        float(((ne-ne.mean())*(k-k.mean())).mean()
                              /(ne.std(unbiased=False)*k.std(unbiased=False))))
            trace["step"].append(step)
            trace["loss"].append(float(loss.detach()))
            trace["soft_acc"].append(acc)
            trace["mean_expected_steps"].append(float(ne.mean()))
            trace["corr_K_expected_steps"].append(corr)
            phase = "executor" if pretraining else "policy"
            trace["phase"].append(phase)
            trace["expected_task"].append(float(
                result["losses"]["expected_task"].detach()))
            trace["auxiliary"].append(float(
                result["losses"]["auxiliary"].detach()))
            trace["completion"].append(float(
                result["losses"]["completion"].detach()))
            print(f"  [{variant} s{seed} {phase}] {step:4d}/{total_updates} "
                  f"loss={float(loss.detach()):.3f} soft_acc={acc:.3f} "
                  f"E[n]={float(ne.mean()):.2f} corr={corr:+.2f} "
                  f"Ldone={trace['completion'][-1]:.2f}", flush=True)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{variant}_seed{seed}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "meta": {"variant": variant, "seed": seed, "steps": args.steps,
                 "pretrain_steps": args.pretrain_steps,
                 "policy_objective": args.policy_objective,
                 "task": "progressive_transition_stream"},
    }, ckpt_path)

    eval_ds = BudgetedTransitionDataset(
        n_states=args.n_states, min_steps=1, max_steps=args.max_steps,
        seed=seed + 100_000)
    evaluation = evaluate_budgeted_stream(
        model, eval_ds, device=device,
        mean_budget=int(eval_ds.mean_required_steps),
        n_batches=args.eval_batches, batch_size=args.batch_size,
        n_shuffles=args.n_shuffles, seed=args.eval_seed + seed)
    evaluation["training"] = {
        "config": asdict(cfg), "seed": seed, "steps": args.steps,
        "pretrain_steps": args.pretrain_steps,
        "policy_objective": args.policy_objective,
        "batch_size": args.batch_size, "lr": args.lr,
        "policy_lr": args.policy_lr,
        "warmup_steps": args.warmup_steps,
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
        deltas = [c["comparisons"]["learned_vs_uniform"]
                  ["accuracy_diff_learned_minus_control"] for c in cells]
        corrs = [c["allocation"]["corr_K_learned_quota"] for c in cells]
        learned = [c["policies"]["learned"]["accuracy"] for c in cells]
        uniform = [c["policies"]["uniform"]["accuracy"] for c in cells]
        summary[variant] = {
            "n_seeds": len(cells),
            "seeds": [c["training"]["seed"] for c in cells],
            "mean_learned_accuracy": statistics.mean(learned),
            "mean_uniform_accuracy": statistics.mean(uniform),
            "mean_delta_accuracy": statistics.mean(deltas),
            "sd_delta_accuracy": (statistics.stdev(deltas) if len(deltas) > 1 else 0.0),
            "mean_corr_K_quota": statistics.mean(corrs),
            "per_seed_delta": deltas,
            "per_seed_corr_K_quota": corrs,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=_csv_strings,
                    default=list(VARIANTS))
    ap.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--pretrain-steps", type=int, default=1500,
                    help="actualizaciones auxiliares del executor antes de entrenar halting")
    ap.add_argument("--batch-size", type=int, default=36)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--n-shuffles", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=9)
    ap.add_argument("--n-states", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--policy-lr", type=float, default=5e-3)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--step-cost", type=float, default=0.01)
    ap.add_argument("--auxiliary-weight", type=float, default=0.5)
    ap.add_argument("--completion-weight", type=float, default=1.0,
                    help="peso de -log p(parar exactamente al consumir K)")
    ap.add_argument("--policy-objective", choices=["completion", "joint"],
                    default="completion",
                    help="completion aísla el scheduler sobre executor congelado")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval-seed", type=int, default=20260719)
    ap.add_argument("--out-dir", default="results_budgeted_stream")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        ap.error(f"variantes desconocidas: {unknown}")
    if args.steps < 1 or args.pretrain_steps < 0 or args.eval_batches < 1 or args.n_shuffles < 1:
        ap.error("steps/eval-batches/n-shuffles deben ser >=1 y pretrain-steps >=0")
    n_lengths = args.max_steps
    if args.batch_size % n_lengths:
        ap.error(f"batch-size debe ser múltiplo de {n_lengths}")
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("Benchmark causal de transiciones progresivas")
    print(f"  device={device} variantes={args.variants} seeds={args.seeds}")
    print(f"  K=1..{args.max_steps}, presupuesto medio={(1+args.max_steps)/2:g}")

    results = []
    for variant in args.variants:
        for seed in args.seeds:
            result, ckpt = train_one(variant, seed, args, device)
            path = os.path.join(args.out_dir, f"{variant}_seed{seed}.json")
            _atomic_json(result, path)
            results.append(result)
            cmp = result["comparisons"]["learned_vs_uniform"]
            p = cmp["cluster_randomization"]["p_exact_two_sided"]
            print(f"  RESULT {variant} s{seed}: "
                  f"learned={result['policies']['learned']['accuracy']:.3f} "
                  f"uniform={result['policies']['uniform']['accuracy']:.3f} "
                  f"Δ={cmp['accuracy_diff_learned_minus_control']:+.3f} "
                  f"p={p:.3g} corr(K,q)={result['allocation']['corr_K_learned_quota']:+.3f}")
            print(f"    {path} | {ckpt}")

    aggregate = {
        "design": {"variants": args.variants, "seeds": args.seeds,
                   "steps": args.steps, "pretrain_steps": args.pretrain_steps,
                   "policy_objective": args.policy_objective,
                   "max_steps": args.max_steps,
                   "mean_budget": (1 + args.max_steps) / 2,
                   "primary_contrast": "learned_vs_uniform"},
        "summary": _aggregate(results),
    }
    _atomic_json(aggregate, os.path.join(args.out_dir, "summary.json"))
    print(f"Resumen: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
