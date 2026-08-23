"""Control negativo: evalúa checkpoints endógenos borrando el marcador LAST."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from data.budgeted_stream import BudgetedTransitionDataset
from eval.hidden_need import evaluate_hidden_need
from model.budgeted_stream import BudgetedStreamConfig, BudgetedStreamReasoner


VARIANTS = ("gating_wm", "hbp_first", "hbp_full")


def _atomic_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results_hidden_need")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=36)
    ap.add_argument("--n-shuffles", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=20260720)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    device = torch.device(args.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    out_dir = os.path.join(args.results_dir, "terminal_ablation")
    results = []

    for variant in variants:
        for seed in seeds:
            ckpt_path = os.path.join(
                args.results_dir, "checkpoints", f"{variant}_seed{seed}.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            cfg = BudgetedStreamConfig(**ckpt["config"])
            model = BudgetedStreamReasoner(cfg).to(device=device, dtype=torch.float32)
            model.load_state_dict(ckpt["state_dict"])
            model.pin_fp32().eval()
            dataset = BudgetedTransitionDataset(
                n_states=cfg.n_states, min_steps=1, max_steps=cfg.max_steps,
                seed=seed + 100_000, terminal_markers=True)
            result = evaluate_hidden_need(
                model, dataset, device=device,
                mean_budget=int(dataset.mean_required_steps),
                n_batches=args.eval_batches, batch_size=args.batch_size,
                n_shuffles=args.n_shuffles, seed=args.eval_seed + seed,
                ablate_terminal_marker=True)
            result["checkpoint"] = os.path.abspath(ckpt_path)
            result["seed"] = seed
            path = os.path.join(out_dir, f"{variant}_seed{seed}.json")
            _atomic_json(result, path)
            results.append(result)
            delta = result["comparisons"]["learned_online_vs_uniform"][
                "accuracy_diff_learned_minus_control"]
            print(f"{variant} s{seed}: learned="
                  f"{result['policies']['learned_online']['accuracy']:.3f} "
                  f"uniform={result['policies']['uniform']['accuracy']:.3f} "
                  f"Δ={delta:+.3f} corr="
                  f"{result['allocation']['corr_K_learned_quota']:+.3f}")

    summary = {"design": {
        "terminal_marker_ablation": True,
        "same_checkpoints_as_primary": True,
        "variants": variants, "seeds": seeds,
        "eval_batches": args.eval_batches,
    }, "summary": {}}
    for variant in variants:
        cells = [r for r in results if r["variant"] == variant]
        summary["summary"][variant] = {
            "mean_learned_accuracy": statistics.mean(
                r["policies"]["learned_online"]["accuracy"] for r in cells),
            "mean_uniform_accuracy": statistics.mean(
                r["policies"]["uniform"]["accuracy"] for r in cells),
            "mean_delta_accuracy": statistics.mean(
                r["comparisons"]["learned_online_vs_uniform"]
                ["accuracy_diff_learned_minus_control"] for r in cells),
            "mean_corr_K_quota": statistics.mean(
                r["allocation"]["corr_K_learned_quota"] for r in cells),
            "mean_exact_match_rate": statistics.mean(
                r["allocation"]["exact_match_rate"] for r in cells),
        }
    _atomic_json(summary, os.path.join(out_dir, "summary.json"))
    print(f"Resumen: {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()

