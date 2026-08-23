"""Fase 2: selección y mantenimiento causal de metas endógenas."""

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

from aha_benchmark import _exact_signflip
from data.endogenous_goals import (
    EndogenousGoalDataset, EndogenousGoalEnvironmentConfig)
from eval.compute_bottleneck import _holm_adjust
from eval.endogenous_goals import evaluate_endogenous_goals
from model.endogenous_goals import (
    EndogenousGoalAgent, EndogenousGoalControllerConfig)


VARIANTS = EndogenousGoalAgent.VARIANTS


def _csv_strings(value: str) -> list[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("lista vacía")
    return out


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(item) for item in _csv_strings(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "usa enteros separados por comas") from exc


def _atomic_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def train_one(variant: str, seed: int, args,
              env_cfg: EndogenousGoalEnvironmentConfig,
              device: torch.device) -> tuple[dict, str]:
    torch.manual_seed(seed)
    dataset = EndogenousGoalDataset(env_cfg, seed=seed)
    model_cfg = EndogenousGoalControllerConfig(
        n_needs=env_cfg.n_needs, d_model=args.d_model,
        variant=variant, goal_action_gain=args.goal_action_gain,
        viability_weight=args.violation_weight)
    model = EndogenousGoalAgent(model_cfg).to(
        device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
        betas=(0.9, 0.95))
    trace = {key: [] for key in (
        "step", "total", "homeostatic", "viability", "effort",
        "proposal", "alignment", "gate", "goal_change", "completion")}
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        warm = min(1.0, step / max(1, args.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(
            math.pi * step / max(1, args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * warm * cosine
        scenario = dataset.batch(
            args.batch_size, critical_mode="mixed").to(device)
        result = model.rollout(scenario, env_cfg, hard_actions=False)
        loss = result["losses"]["total"]
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            values = {"step": step} | {
                key: float(value.detach())
                for key, value in result["losses"].items()}
            for key in trace:
                trace[key].append(values[key])
            print(
                f"  [{variant} s{seed}] {step:4d}/{args.steps} "
                f"loss={values['total']:.4f} "
                f"homeo={values['homeostatic']:.4f} "
                f"viol={values['viability']:.4f} "
                f"complete={values['completion']:.3f} "
                f"change={values['goal_change']:.3f}", flush=True)

    checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{variant}_seed{seed}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": asdict(model_cfg),
        "environment_config": asdict(env_cfg),
        "meta": {
            "phase": "phase_2_endogenous_goals",
            "variant": variant,
            "seed": seed,
            "external_goal_supplied": False,
        },
    }, checkpoint_path)

    evaluation_dataset = EndogenousGoalDataset(
        env_cfg, seed=seed + 100_000)
    evaluation = evaluate_endogenous_goals(
        model, evaluation_dataset, device,
        n_batches=args.eval_batches, batch_size=args.eval_batch_size)
    evaluation["training"] = {
        "seed": seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "trace": trace,
        "elapsed_s": time.time() - started,
    }
    evaluation["model_config"] = asdict(model_cfg)
    evaluation["environment_config"] = asdict(env_cfg)
    evaluation["provenance"] = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device)
                        if device.type == "cuda" else "cpu"),
    }
    return evaluation, checkpoint_path


def _aggregate(results: list[dict]) -> dict:
    grouped = {variant: [] for variant in VARIANTS}
    for result in results:
        grouped[result["variant"]].append(result)
    normal_keys = (
        "survival_rate", "violation_rate",
        "mean_absolute_homeostatic_error", "mean_project_completions",
        "initial_goal_selection_rate", "initial_action_selection_rate",
        "mild_commitment_rate", "mild_goal_persistence_rate",
        "mild_completion_rate", "critical_switch_rate",
        "critical_goal_switch_rate", "critical_rescue_rate",
        "work_switch_rate", "goal_action_alignment_rate",
        "minor_alias_max_abs_gap", "minor_update_gate_at_conflict",
        "critical_update_gate_at_conflict")
    causal_keys = (
        "memory_on_mild_completion", "memory_on_mild_commitment",
        "memory_on_survival", "goal_content_on_minor_completion",
        "goal_rotation_follow_rate", "goal_rotation_changed_action_rate",
        "body_transplant_goal_selection_rate",
        "body_transplant_action_selection_rate")
    summary = {}
    for variant, cells in grouped.items():
        if not cells:
            continue
        summary[variant] = {
            "n_seeds": len(cells),
            "seeds": [cell["training"]["seed"] for cell in cells],
            "normal": {key: statistics.mean(
                cell["conditions"]["normal"][key] for cell in cells)
                for key in normal_keys},
            "causal": {key: statistics.mean(
                cell["causal_effects"][key] for cell in cells)
                for key in causal_keys},
        }

    by_cell = {(cell["variant"], cell["training"]["seed"]): cell
               for cell in results}
    seeds = sorted({cell["training"]["seed"] for cell in results})
    contrast = None
    confirmatory = None
    if seeds and all((variant, seed) in by_cell
                     for variant in VARIANTS for seed in seeds):
        metrics = (
            "mild_completion_rate", "critical_rescue_rate",
            "survival_rate")
        differences = {
            metric: [
                by_cell[("goal_memory", seed)]["conditions"]["normal"][metric]
                - by_cell[("reactive", seed)]["conditions"]["normal"][metric]
                for seed in seeds]
            for metric in metrics}
        tests = {metric: _exact_signflip(values)
                 for metric, values in differences.items()}
        holm = _holm_adjust({
            metric: test["p_exact_two_sided"]
            for metric, test in tests.items()})
        contrast = {
            "seeds": seeds,
            "mean_differences": {
                metric: statistics.mean(values)
                for metric, values in differences.items()},
            "seed_level_exact_signflip": tests,
            "holm_adjusted_p": holm,
        }
        # Familia fijada antes de confirmatory_v1. Los guardrails impiden
        # llamar "meta" a la mera perseveración ciega o a una prioridad que no
        # siga al cuerpo.
        primary_differences = {
            "memory_minus_reactive_mild_completion": differences[
                "mild_completion_rate"],
            "memory_lesion_on_mild_completion": [
                by_cell[("goal_memory", seed)]["causal_effects"]
                ["memory_on_mild_completion"] for seed in seeds],
            "goal_content_rotation_on_minor_completion": [
                by_cell[("goal_memory", seed)]["causal_effects"]
                ["goal_content_on_minor_completion"] for seed in seeds],
        }
        primary_tests = {
            name: _exact_signflip(values)
            for name, values in primary_differences.items()}
        primary_holm = _holm_adjust({
            name: test["p_exact_two_sided"]
            for name, test in primary_tests.items()})
        criteria = {
            name: (statistics.mean(values) > 0.0
                   and primary_holm[name] < 0.05)
            for name, values in primary_differences.items()}
        memory_cells = [by_cell[("goal_memory", seed)] for seed in seeds]
        guardrails = {
            "critical_rescue_each_seed_at_least_0.95": min(
                cell["conditions"]["normal"]["critical_rescue_rate"]
                for cell in memory_cells) >= 0.95,
            "body_transplant_goal_each_seed_at_least_0.95": min(
                cell["causal_effects"]
                ["body_transplant_goal_selection_rate"]
                for cell in memory_cells) >= 0.95,
            "body_transplant_action_each_seed_at_least_0.95": min(
                cell["causal_effects"]
                ["body_transplant_action_selection_rate"]
                for cell in memory_cells) >= 0.95,
            "initial_goal_each_seed_at_least_0.95": min(
                cell["conditions"]["normal"]
                ["initial_goal_selection_rate"]
                for cell in memory_cells) >= 0.95,
            "minor_state_alias_max_below_1e-6": max(
                cell["conditions"]["normal"]["minor_alias_max_abs_gap"]
                for cell in memory_cells) < 1e-6,
        }
        confirmatory = {
            "status": ("pass" if (all(criteria.values())
                                    and all(guardrails.values())) else "fail"),
            "decision_rule": (
                "all three effects positive with Holm-adjusted exact "
                "sign-flip p < 0.05, and all five guardrails true"),
            "alpha_familywise": 0.05,
            "n_paired_seeds": len(seeds),
            "seeds": seeds,
            "mean_effects": {
                name: statistics.mean(values)
                for name, values in primary_differences.items()},
            "seed_level_exact_signflip": primary_tests,
            "holm_adjusted_p": primary_holm,
            "criterion_passed": criteria,
            "guardrails": guardrails,
        }
    return {"summary": summary,
            "goal_memory_minus_reactive": contrast,
            "confirmatory_primary_family": confirmatory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=_csv_strings,
                        default=["goal_memory", "reactive"])
    parser.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--goal-action-gain", type=float, default=4.0)
    parser.add_argument("--violation-weight", type=float, default=12.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-dir", default="results_endogenous_goals")
    parser.add_argument("--summary-file", default="summary.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-existing",
                        action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        parser.error(f"variantes desconocidas: {unknown}")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds debe contener enteros distintos")
    if min(args.steps, args.batch_size,
           args.eval_batches, args.eval_batch_size) < 1:
        parser.error("steps y batches deben ser positivos")

    env_cfg = EndogenousGoalEnvironmentConfig()
    env_cfg.validate()
    device = torch.device(args.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    print("Fase 2: metas endógenas sostenidas")
    print(f"  device={device} variants={args.variants} seeds={args.seeds}")
    print("  sin goal_id; gaps obligatorios; conflicto leve aliasado; crisis")
    results = []
    for variant in args.variants:
        for seed in args.seeds:
            result_path = os.path.join(
                args.out_dir, f"{variant}_seed{seed}.json")
            checkpoint_path = os.path.join(
                args.out_dir, "checkpoints", f"{variant}_seed{seed}.pt")
            if (args.resume_existing and os.path.exists(result_path)
                    and os.path.exists(checkpoint_path)):
                with open(result_path, encoding="utf-8") as stream:
                    result = json.load(stream)
                print(f"  REUSE {variant} s{seed}")
            else:
                result, checkpoint_path = train_one(
                    variant, seed, args, env_cfg, device)
                _atomic_json(result, result_path)
            results.append(result)
            normal = result["conditions"]["normal"]
            causal = result["causal_effects"]
            print(
                f"  RESULT {variant} s{seed}: "
                f"mild_complete={normal['mild_completion_rate']:.3f} "
                f"critical_rescue={normal['critical_rescue_rate']:.3f} "
                f"survival={normal['survival_rate']:.3f} "
                f"Δlesion={causal['memory_on_mild_completion']:+.3f}")
            print(f"    {result_path} | {checkpoint_path}")

    aggregate = {
        "design": {
            "phase": "phase_2_endogenous_goals",
            "roadmap": ["self_regulation", "anticipation",
                        "endogenous_goals", "discovered_goals",
                        "self_model", "metacognition"],
            "variants": args.variants,
            "seeds": args.seeds,
            "steps": args.steps,
            "environment": asdict(env_cfg),
            "external_goal_supplied": False,
            "claim_if_positive": (
                "causal endogenous goal selection and maintenance; "
                "not open-ended goal discovery or consciousness"),
        },
        **_aggregate(results),
    }
    summary_path = os.path.join(args.out_dir, args.summary_file)
    _atomic_json(aggregate, summary_path)
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
