"""Fase 4: automodelo causal y adaptación a daño corporal no anunciado."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import platform
import statistics

import torch

from aha_benchmark import _exact_signflip
from data.self_model import SelfModelDataset, SelfModelEnvironmentConfig
from eval.compute_bottleneck import _holm_adjust
from eval.self_model import evaluate_self_model
from model.self_model import SelfModelControllerConfig, WaveBodySelfModel


VARIANTS = WaveBodySelfModel.VARIANTS


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


def evaluate_one(variant: str, seed: int, args,
                 env_cfg: SelfModelEnvironmentConfig,
                 device: torch.device) -> tuple[dict, str]:
    torch.manual_seed(seed)
    model_cfg = SelfModelControllerConfig(
        n_nodes=env_cfg.n_nodes, variant=variant,
        model_update_gain=args.model_update_gain,
        action_cost=args.action_cost)
    model = WaveBodySelfModel(model_cfg, env_cfg).to(device)
    dataset = SelfModelDataset(env_cfg, seed=seed + 300_000)
    evaluation = evaluate_self_model(
        model, dataset, device,
        n_batches=args.eval_batches, batch_size=args.eval_batch_size)
    evaluation["evaluation"] = {
        "seed": seed,
        "n_batches": args.eval_batches,
        "batch_size": args.eval_batch_size,
        "n_episodes": args.eval_batches * args.eval_batch_size,
    }
    evaluation["model_config"] = asdict(model_cfg)
    evaluation["environment_config"] = asdict(env_cfg)
    checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{variant}_seed{seed}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": asdict(model_cfg),
        "environment_config": asdict(env_cfg),
        "meta": {
            "phase": "phase_4_self_model",
            "variant": variant,
            "seed": seed,
            "damage_flag_visible": False,
            "true_actuator_matrix_visible": False,
        },
    }, checkpoint_path)
    evaluation["provenance"] = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": str(device),
    }
    return evaluation, checkpoint_path


def _aggregate(results: list[dict]) -> dict:
    grouped = {variant: [] for variant in VARIANTS}
    for result in results:
        grouped[result["variant"]].append(result)
    normal_keys = (
        "pre_damage_recovery_rate", "post_damage_recovery_rate",
        "mean_recovery_latency", "backup_switch_rate",
        "failed_primary_repeat_rate", "mean_post_damage_target_error",
        "post_damage_viability_rate", "post_adaptation_prediction_mse",
        "post_adaptation_oracle_match_rate",
        "final_damaged_actuator_model_mse", "passive_spectral_radius")
    causal_keys = (
        "model_update_on_recovery", "valid_efference_on_recovery",
        "model_content_on_recovery", "model_update_on_target_error",
        "model_update_on_prediction_mse",
        "damage_transplant_recovery_rate",
        "damage_transplant_backup_switch_rate")
    summary = {}
    for variant, cells in grouped.items():
        if not cells:
            continue
        summary[variant] = {
            "n_seeds": len(cells),
            "seeds": [cell["evaluation"]["seed"] for cell in cells],
            "normal": {key: statistics.mean(
                cell["conditions"]["normal"][key] for cell in cells)
                for key in normal_keys},
            "causal": {key: statistics.mean(
                cell["causal_effects"][key] for cell in cells)
                for key in causal_keys},
        }

    by_cell = {(cell["variant"], cell["evaluation"]["seed"]): cell
               for cell in results}
    seeds = sorted({cell["evaluation"]["seed"] for cell in results})
    contrast = None
    confirmatory = None
    if seeds and all((variant, seed) in by_cell
                     for variant in VARIANTS for seed in seeds):
        architecture_difference = [
            by_cell[("self_model", seed)]["conditions"]["normal"]
            ["post_damage_recovery_rate"]
            - by_cell[("stale_model", seed)]["conditions"]["normal"]
            ["post_damage_recovery_rate"]
            for seed in seeds]
        contrast = {
            "seeds": seeds,
            "mean_post_damage_recovery_difference": statistics.mean(
                architecture_difference),
            "seed_level_exact_signflip": _exact_signflip(
                architecture_difference),
        }
        adaptive = [by_cell[("self_model", seed)] for seed in seeds]
        # Familia y umbrales congelados antes del primer run confirmatorio.
        primary_differences = {
            "self_model_minus_stale_recovery": architecture_difference,
            "online_update_on_recovery": [
                cell["causal_effects"]["model_update_on_recovery"]
                for cell in adaptive],
            "valid_efference_on_recovery": [
                cell["causal_effects"]["valid_efference_on_recovery"]
                for cell in adaptive],
            "model_content_on_recovery": [
                cell["causal_effects"]["model_content_on_recovery"]
                for cell in adaptive],
        }
        tests = {name: _exact_signflip(values)
                 for name, values in primary_differences.items()}
        holm = _holm_adjust({
            name: test["p_exact_two_sided"]
            for name, test in tests.items()})
        criteria = {
            name: (statistics.mean(values) > 0.0 and holm[name] < 0.05)
            for name, values in primary_differences.items()}
        paired_gaps = [abs(
            by_cell[("self_model", seed)]["conditions"]["update_lesion"]
            ["post_damage_recovery_rate"]
            - by_cell[("stale_model", seed)]["conditions"]["normal"]
            ["post_damage_recovery_rate"])
            for seed in seeds]
        guardrails = {
            "pre_damage_recovery_each_seed_at_least_0.95": min(
                cell["conditions"]["normal"]["pre_damage_recovery_rate"]
                for cell in adaptive) >= 0.95,
            "post_damage_recovery_each_seed_at_least_0.95": min(
                cell["conditions"]["normal"]["post_damage_recovery_rate"]
                for cell in adaptive) >= 0.95,
            "backup_switch_each_seed_at_least_0.95": min(
                cell["conditions"]["normal"]["backup_switch_rate"]
                for cell in adaptive) >= 0.95,
            "prediction_mse_each_seed_below_1e-3": max(
                cell["conditions"]["normal"]
                ["post_adaptation_prediction_mse"]
                for cell in adaptive) < 1e-3,
            "passive_wave_radius_each_seed_below_1": max(
                cell["conditions"]["normal"]["passive_spectral_radius"]
                for cell in adaptive) < 1.0,
            "damage_transplant_recovery_each_seed_at_least_0.95": min(
                cell["causal_effects"]["damage_transplant_recovery_rate"]
                for cell in adaptive) >= 0.95,
            "damage_transplant_switch_each_seed_at_least_0.95": min(
                cell["causal_effects"]
                ["damage_transplant_backup_switch_rate"]
                for cell in adaptive) >= 0.95,
            "update_lesion_equals_stale_control": max(paired_gaps) < 1e-9,
        }
        confirmatory = {
            "status": ("pass" if (all(criteria.values())
                                    and all(guardrails.values())) else "fail"),
            "decision_rule": (
                "all four effects positive with Holm-adjusted exact "
                "sign-flip p < 0.05, and all eight guardrails true"),
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
            "max_lesion_stale_gap": max(paired_gaps),
        }
    return {
        "summary": summary,
        "self_model_minus_stale": contrast,
        "confirmatory_primary_family": confirmatory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=_csv_strings,
                        default=["self_model", "stale_model"])
    parser.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--model-update-gain", type=float, default=1.0)
    parser.add_argument("--action-cost", type=float, default=0.002)
    parser.add_argument("--out-dir", default="results_self_model")
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
    if min(args.eval_batches, args.eval_batch_size) < 1:
        parser.error("los batches deben ser positivos")

    env_cfg = SelfModelEnvironmentConfig()
    env_cfg.validate()
    device = torch.device(args.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    print("Fase 4: automodelo de un cuerpo de onda dañado")
    print(f"  device={device} variants={args.variants} seeds={args.seeds}")
    print("  sin damage flag, matriz real, target_id ni presupuesto extra")
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
                result, checkpoint_path = evaluate_one(
                    variant, seed, args, env_cfg, device)
                _atomic_json(result, result_path)
            results.append(result)
            normal = result["conditions"]["normal"]
            causal = result["causal_effects"]
            print(
                f"  RESULT {variant} s{seed}: "
                f"pre={normal['pre_damage_recovery_rate']:.3f} "
                f"post={normal['post_damage_recovery_rate']:.3f} "
                f"backup={normal['backup_switch_rate']:.3f} "
                f"dLesion={causal['model_update_on_recovery']:+.3f}")
            print(f"    {result_path} | {checkpoint_path}")

    aggregate = {
        "design": {
            "phase": "phase_4_self_model",
            "roadmap": ["self_regulation", "anticipation",
                        "endogenous_goals", "discovered_goals",
                        "self_model", "metacognition"],
            "variants": args.variants,
            "seeds": args.seeds,
            "environment": asdict(env_cfg),
            "passive_body": "stable_damped_wave",
            "damage_flag_visible_to_policy": False,
            "true_actuator_matrix_visible_to_policy": False,
            "equal_action_and_compute_budget": True,
            "claim_if_positive": (
                "causal online self-model of action consequences and "
                "adaptation to actuator damage in the tested wave body; "
                "not a narrative self, consciousness or human selfhood"),
        },
        **_aggregate(results),
    }
    summary_path = os.path.join(args.out_dir, args.summary_file)
    _atomic_json(aggregate, summary_path)
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
