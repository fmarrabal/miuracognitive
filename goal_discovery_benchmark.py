"""Fase 3: descubrimiento causal de metas continuas no enumeradas."""

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
from data.goal_discovery import (
    GoalDiscoveryDataset, GoalDiscoveryEnvironmentConfig)
from eval.compute_bottleneck import _holm_adjust
from eval.goal_discovery import evaluate_goal_discovery
from model.goal_discovery import (
    ContinuousGoalDiscoverer, GoalDiscoveryControllerConfig)


VARIANTS = ContinuousGoalDiscoverer.VARIANTS


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
              env_cfg: GoalDiscoveryEnvironmentConfig,
              device: torch.device) -> tuple[dict, str]:
    # Reiniciar aquí hace emparejados tanto los pesos iniciales como los
    # paisajes de entrenamiento para las dos arquitecturas.
    torch.manual_seed(seed)
    dataset = GoalDiscoveryDataset(env_cfg, seed=seed)
    model_cfg = GoalDiscoveryControllerConfig(
        n_needs=env_cfg.n_needs, goal_dim=env_cfg.goal_dim,
        n_probes=env_cfg.n_probes, d_model=args.d_model,
        variant=variant,
        beta_response_prediction=args.beta_response_prediction,
        beta_probe_diversity=args.beta_probe_diversity,
        min_probe_distance=args.min_probe_distance,
        urgency_temperature=args.urgency_temperature,
        probe_residual_scale=args.probe_residual_scale,
        goal_residual_scale=args.goal_residual_scale)
    model = ContinuousGoalDiscoverer(model_cfg).to(
        device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
        betas=(0.9, 0.95))
    trace = {key: [] for key in (
        "step", "total", "homeostatic", "response_prediction",
        "probe_diversity")}
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        warm = min(1.0, step / max(1, args.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(
            math.pi * step / max(1, args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * warm * cosine
        scenario = dataset.batch(args.batch_size).to(device)
        result = model.rollout(scenario, env_cfg)
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
                f"pred={values['response_prediction']:.4f} "
                f"div={values['probe_diversity']:.4f}", flush=True)

    checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{variant}_seed{seed}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": asdict(model_cfg),
        "environment_config": asdict(env_cfg),
        "meta": {
            "phase": "phase_3_discovered_goals",
            "variant": variant,
            "seed": seed,
            "external_goal_supplied": False,
            "goal_catalog_exists": False,
            "center_supervision": False,
        },
    }, checkpoint_path)

    evaluation_dataset = GoalDiscoveryDataset(
        env_cfg, seed=seed + 100_000)
    evaluation = evaluate_goal_discovery(
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
        "goal_success_rate", "mean_target_center_distance",
        "mean_target_response", "dominant_restored_rate",
        "mean_homeostatic_improvement", "mean_final_absolute_error",
        "feasible_goal_rate", "continuous_unique_goal_fraction",
        "mean_normalized_probe_distance")
    causal_keys = (
        "feedback_on_goal_success", "valid_outcomes_on_goal_success",
        "goal_content_on_success", "feedback_on_homeostatic_improvement",
        "body_transplant_switch_rate", "body_transplant_new_goal_success",
        "body_transplant_goal_shift")
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
        architecture_differences = [
            by_cell[("discoverer", seed)]["conditions"]["normal"]
            ["goal_success_rate"]
            - by_cell[("no_feedback", seed)]["conditions"]["normal"]
            ["goal_success_rate"]
            for seed in seeds]
        contrast = {
            "seeds": seeds,
            "mean_goal_success_difference": statistics.mean(
                architecture_differences),
            "seed_level_exact_signflip": _exact_signflip(
                architecture_differences),
        }

        # Familia y umbrales fijados antes de observar ningún piloto de fase 3.
        # Exigir las cuatro piezas evita llamar descubrimiento a una salida
        # continua decorativa, a exploración sin uso causal o a mero déficit.
        discoverer_cells = [by_cell[("discoverer", seed)] for seed in seeds]
        primary_differences = {
            "discoverer_minus_no_feedback_goal_success":
                architecture_differences,
            "feedback_lesion_on_goal_success": [
                cell["causal_effects"]["feedback_on_goal_success"]
                for cell in discoverer_cells],
            "outcome_validity_on_goal_success": [
                cell["causal_effects"]["valid_outcomes_on_goal_success"]
                for cell in discoverer_cells],
            "goal_content_reflection_on_success": [
                cell["causal_effects"]["goal_content_on_success"]
                for cell in discoverer_cells],
        }
        tests = {name: _exact_signflip(values)
                 for name, values in primary_differences.items()}
        holm = _holm_adjust({
            name: test["p_exact_two_sided"]
            for name, test in tests.items()})
        criteria = {
            name: (statistics.mean(values) > 0.0 and holm[name] < 0.05)
            for name, values in primary_differences.items()}
        guardrails = {
            "normal_success_each_seed_at_least_0.70": min(
                cell["conditions"]["normal"]["goal_success_rate"]
                for cell in discoverer_cells) >= 0.70,
            "feasible_each_seed_equals_1": min(
                cell["conditions"]["normal"]["feasible_goal_rate"]
                for cell in discoverer_cells) == 1.0,
            "continuous_unique_each_seed_at_least_0.95": min(
                cell["conditions"]["normal"]
                ["continuous_unique_goal_fraction"]
                for cell in discoverer_cells) >= 0.95,
            "probe_distance_each_seed_at_least_0.05": min(
                cell["conditions"]["normal"]
                ["mean_normalized_probe_distance"]
                for cell in discoverer_cells) >= 0.05,
            "body_transplant_switch_each_seed_at_least_0.70": min(
                cell["causal_effects"]["body_transplant_switch_rate"]
                for cell in discoverer_cells) >= 0.70,
            "body_transplant_success_each_seed_at_least_0.50": min(
                cell["causal_effects"]
                ["body_transplant_new_goal_success"]
                for cell in discoverer_cells) >= 0.50,
        }
        confirmatory = {
            "status": ("pass" if (all(criteria.values())
                                    and all(guardrails.values())) else "fail"),
            "decision_rule": (
                "all four effects positive with Holm-adjusted exact "
                "sign-flip p < 0.05, and all six guardrails true"),
            "alpha_familywise": 0.05,
            "n_paired_seeds": len(seeds),
            "seeds": seeds,
            "mean_effects": {
                name: statistics.mean(values)
                for name, values in primary_differences.items()},
            "seed_level_exact_signflip": tests,
            "holm_adjusted_p": holm,
            "criterion_passed": criteria,
            "guardrails": guardrails,
        }
    return {
        "summary": summary,
        "discoverer_minus_no_feedback": contrast,
        "confirmatory_primary_family": confirmatory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=_csv_strings,
                        default=["discoverer", "no_feedback"])
    parser.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--beta-response-prediction", type=float,
                        default=0.05)
    parser.add_argument("--beta-probe-diversity", type=float, default=0.01)
    parser.add_argument("--min-probe-distance", type=float, default=0.22)
    parser.add_argument("--urgency-temperature", type=float, default=12.0)
    parser.add_argument("--probe-residual-scale", type=float, default=0.25)
    parser.add_argument("--goal-residual-scale", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-dir", default="results_goal_discovery")
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

    env_cfg = GoalDiscoveryEnvironmentConfig()
    env_cfg.validate()
    device = torch.device(args.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    print("Fase 3: descubrimiento continuo de metas")
    print(f"  device={device} variants={args.variants} seeds={args.seeds}")
    print("  sin goal_id, catálogo, centro visible ni supervisión del centro")
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
                f"success={normal['goal_success_rate']:.3f} "
                f"response={normal['mean_target_response']:.3f} "
                f"unique={normal['continuous_unique_goal_fraction']:.3f} "
                f"dLesion={causal['feedback_on_goal_success']:+.3f}")
            print(f"    {result_path} | {checkpoint_path}")

    aggregate = {
        "design": {
            "phase": "phase_3_discovered_goals",
            "roadmap": ["self_regulation", "anticipation",
                        "endogenous_goals", "discovered_goals",
                        "self_model", "metacognition"],
            "variants": args.variants,
            "seeds": args.seeds,
            "steps": args.steps,
            "environment": asdict(env_cfg),
            "external_goal_supplied": False,
            "goal_catalog_exists": False,
            "center_visible_to_policy": False,
            "training_center_supervision": False,
            "decoder": "differentiable_rbf_trilateration_plus_learned_residual",
            "equal_compute_and_probe_budget": True,
            "claim_if_positive": (
                "causal discovery of continuous, compositional goals in "
                "the tested affordance grammar; not arbitrary purposes, "
                "consciousness or a human soul"),
        },
        **_aggregate(results),
    }
    summary_path = os.path.join(args.out_dir, args.summary_file)
    _atomic_json(aggregate, summary_path)
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
