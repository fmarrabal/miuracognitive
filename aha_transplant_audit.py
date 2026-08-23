"""Auditoría post hoc de la latencia del trasplante de necesidad de AHA-1."""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
import re
import statistics

import torch

from data.aha import AHAEnvironmentConfig, AnticipatoryHomeostasisDataset
from model.aha import AHAControllerConfig, AnticipatoryHomeostaticAgent


def _seed_from_path(path: str) -> int:
    match = re.search(r"_seed(\d+)\.pt$", path)
    if match is None:
        raise ValueError(f"checkpoint sin seed reconocible: {path}")
    return int(match.group(1))


@torch.no_grad()
def audit_checkpoint(path: str, device: torch.device, *,
                     window: int, batch_size: int) -> list[dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    env_cfg = AHAEnvironmentConfig(**checkpoint["environment_config"])
    model_cfg = AHAControllerConfig(**checkpoint["model_config"])
    if not 1 <= window <= env_cfg.horizon:
        raise ValueError("window debe estar dentro del horizonte")
    agent = AnticipatoryHomeostaticAgent(model_cfg).to(device).eval()
    agent.load_state_dict(checkpoint["state_dict"])

    seed = _seed_from_path(path)
    base = AnticipatoryHomeostasisDataset(
        env_cfg, seed=seed + 900_000).batch(
            batch_size, hazards=False).to(device)
    base.cues.zero_()
    base.disturbances.zero_()

    rows = []
    for need in range(env_cfg.n_needs):
        deficit, satiated = base.clone(), base.clone()
        deficit.setpoints.fill_(0.75)
        satiated.setpoints.fill_(0.75)
        deficit.initial_levels.fill_(0.78)
        satiated.initial_levels.fill_(0.78)
        deficit.initial_levels[:, need] = 0.55
        satiated.initial_levels[:, need] = 0.95

        low = agent.rollout(deficit, env_cfg, hard_actions=True)
        high = agent.rollout(satiated, env_cfg, hard_actions=True)
        correct = need + 1
        low_hit = low["action_ids"][:, :window] == correct
        high_hit = high["action_ids"][:, :window] == correct
        first_tick = next(
            (tick for tick in range(window)
             if bool(low_hit[:, tick].any())), None)
        rows.append({
            "seed": seed,
            "need": need,
            "deficit_any_correct": float(
                low_hit.any(dim=1).float().mean()),
            "satiated_any_correct": float(
                high_hit.any(dim=1).float().mean()),
            "mean_target_probability_effect": float((
                low["action_probs"][:, :window, correct]
                - high["action_probs"][:, :window, correct]).mean()),
            "first_deficit_correct_tick": first_tick,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_aha/confirmatory_v1")
    parser.add_argument("--variant", default="gating_wm")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="transplant_latency_audit.json")
    args = parser.parse_args()
    device = torch.device(args.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    pattern = os.path.join(
        args.results_dir, "checkpoints", f"{args.variant}_seed*.pt")
    paths = sorted(glob.glob(pattern), key=_seed_from_path)
    if not paths:
        parser.error(f"no hay checkpoints para {pattern}")

    rows = []
    for path in paths:
        rows.extend(audit_checkpoint(
            path, device, window=args.window, batch_size=args.batch_size))
    ticks = Counter(
        row["first_deficit_correct_tick"]
        if row["first_deficit_correct_tick"] is not None else "none"
        for row in rows)
    report = {
        "design": {
            "analysis": "post_hoc_transplant_latency",
            "confirmatory_decision_uses_this": False,
            "variant": args.variant,
            "window_ticks": args.window,
            "batch_size": args.batch_size,
            "external_world": "stationary; cues and disturbances zero",
            "deficit_level": 0.55,
            "satiated_level": 0.95,
            "other_levels": 0.78,
            "setpoints": 0.75,
        },
        "summary": {
            "n_seeds": len(paths),
            "n_seed_need_cells": len(rows),
            "deficit_any_correct_within_window": statistics.mean(
                row["deficit_any_correct"] for row in rows),
            "satiated_any_correct_within_window": statistics.mean(
                row["satiated_any_correct"] for row in rows),
            "mean_target_probability_effect_within_window": statistics.mean(
                row["mean_target_probability_effect"] for row in rows),
            "first_deficit_correct_tick_counts": {
                str(key): value for key, value in sorted(
                    ticks.items(), key=lambda item: str(item[0]))},
        },
        "records": rows,
    }
    output = (args.output if os.path.isabs(args.output)
              else os.path.join(args.results_dir, args.output))
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, output)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Guardado: {output}")


if __name__ == "__main__":
    main()
