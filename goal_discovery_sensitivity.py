"""Auditoría post-confirmatoria de sensibilidad del decodificador de fase 3."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from data.goal_discovery import (
    GoalDiscoveryDataset, GoalDiscoveryEnvironmentConfig)
from eval.goal_discovery import evaluate_decoder_sensitivity
from model.goal_discovery import (
    ContinuousGoalDiscoverer, GoalDiscoveryControllerConfig)


def _atomic_json(data: dict, path: str) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",
                        default="results_goal_discovery/confirmatory_v1/checkpoints")
    parser.add_argument("--out-file",
                        default="results_goal_discovery/confirmatory_v1/sensitivity_audit.json")
    parser.add_argument("--seeds", default=",".join(
        str(seed) for seed in range(200, 220)))
    parser.add_argument("--sigma-scales", default="0.85,1.0,1.15")
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",")]
    scales = tuple(float(item) for item in args.sigma_scales.split(","))
    device = torch.device(args.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"))

    records = []
    for seed in seeds:
        path = os.path.join(
            args.checkpoint_dir, f"discoverer_seed{seed}.pt")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        env_cfg = GoalDiscoveryEnvironmentConfig(
            **checkpoint["environment_config"])
        model_cfg = GoalDiscoveryControllerConfig(
            **checkpoint["model_config"])
        model = ContinuousGoalDiscoverer(model_cfg).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        dataset = GoalDiscoveryDataset(env_cfg, seed=seed + 200_000)
        conditions = evaluate_decoder_sensitivity(
            model, dataset, device, sigma_scales=scales,
            n_batches=args.eval_batches,
            batch_size=args.eval_batch_size)
        records.append({"seed": seed, "conditions": conditions})
        values = " ".join(
            f"{key}={cell['goal_success_rate']:.3f}"
            for key, cell in conditions.items())
        print(f"s{seed}: {values}", flush=True)

    condition_keys = list(records[0]["conditions"])
    aggregate = {
        key: {
            "mean_goal_success_rate": statistics.mean(
                record["conditions"][key]["goal_success_rate"]
                for record in records),
            "min_seed_goal_success_rate": min(
                record["conditions"][key]["goal_success_rate"]
                for record in records),
            "mean_target_response": statistics.mean(
                record["conditions"][key]["mean_target_response"]
                for record in records),
        }
        for key in condition_keys}
    result = {
        "status": ("pass" if min(
            cell["min_seed_goal_success_rate"]
            for cell in aggregate.values()) >= 0.70 else "fail"),
        "criterion": (
            "goal success >= 0.70 in every seed at decoder sigma scales "
            "0.85, 1.00 and 1.15, without retraining"),
        "post_confirmatory": True,
        "n_seeds": len(seeds),
        "aggregate": aggregate,
        "records": records,
    }
    _atomic_json(result, args.out_file)
    print(f"status={result['status']} -> {args.out_file}")


if __name__ == "__main__":
    main()
