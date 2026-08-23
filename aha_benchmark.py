"""Benchmark AHA: agencia homeostática anticipatoria causal.

Fase 1 de la hoja de ruta:
    autorregulación -> ANTICIPACIÓN -> metas endógenas -> ...

No se suministra una meta por episodio. Un predictor se preentrena de manera
auto-supervisada para anticipar perturbaciones desde señales causales, se
congela, y después el agente aprende a mantener tres necesidades dentro de su
región viable. La evaluación lesiona la predicción, baraja los cues y trasplanta
estados internos bajo mundos externos idénticos.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import math
import os
import platform
import statistics
import time

import torch

from data.aha import AHAEnvironmentConfig, AnticipatoryHomeostasisDataset
from eval.aha import evaluate_aha
from eval.compute_bottleneck import _holm_adjust
from model.aha import AHAControllerConfig, AnticipatoryHomeostaticAgent


VARIANTS = AnticipatoryHomeostaticAgent.VARIANTS


def _csv_strings(value: str) -> list[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("lista vacía")
    return out


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(x) for x in _csv_strings(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "usa enteros separados por comas") from exc


def _atomic_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _exact_signflip(values: list[float]) -> dict:
    """Test exacto bilateral por seed para una diferencia continua."""

    nonzero = [float(x) for x in values if abs(float(x)) > 1e-12]
    observed = abs(sum(nonzero))
    if not nonzero:
        return {"differences": values, "nonzero_seeds": 0,
                "observed_sum": 0.0, "p_exact_two_sided": 1.0}
    if len(nonzero) > 20:
        raise ValueError("el test exacto está limitado a 20 seeds")
    extreme, total = 0, 2 ** len(nonzero)
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        value = abs(sum(sign * delta
                        for sign, delta in zip(signs, nonzero)))
        extreme += value >= observed - 1e-12
    return {"differences": values, "nonzero_seeds": len(nonzero),
            "observed_sum": sum(values),
            "p_exact_two_sided": extreme / total}


def _set_lr(optimizer: torch.optim.Optimizer, base_lr: float,
            step: int, total: int, warmup: int) -> None:
    warm = min(1.0, step / max(1, warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * step / max(1, total)))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * warm * cosine


def _set_predictor_lr(optimizer: torch.optim.Optimizer, base_lr: float,
                      step: int, warmup: int) -> None:
    """Warm-up sin decaimiento: aprender la latencia exige updates tardíos."""

    warm = min(1.0, step / max(1, warmup))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * warm


def train_one(variant: str, seed: int, args,
              env_cfg: AHAEnvironmentConfig,
              device: torch.device,
              predictor_source: str | None = None) -> tuple[dict, str]:
    torch.manual_seed(seed)
    dataset = AnticipatoryHomeostasisDataset(env_cfg, seed=seed)
    model_cfg = AHAControllerConfig(
        n_needs=env_cfg.n_needs, d_model=args.d_model,
        d_predictor=args.d_predictor, variant=variant,
        action_temperature=args.temperature,
        violation_weight=args.violation_weight)
    model = AnticipatoryHomeostaticAgent(model_cfg).to(
        device=device, dtype=torch.float32).pin_fp32()
    started = time.time()

    trace = {key: [] for key in (
        "phase", "step", "loss", "homeostatic", "viability",
        "effort", "prediction", "survival_soft")}
    model.train()
    predictor_reused = predictor_source is not None
    if predictor_reused:
        source = torch.load(
            predictor_source, map_location="cpu", weights_only=True)
        source_env = source["environment_config"]
        source_model = source["model_config"]
        for key in ("n_needs", "horizon", "cue_lead", "action_delay",
                    "prediction_horizon"):
            expected = asdict(env_cfg)[key]
            if source_env[key] != expected:
                raise RuntimeError(f"predictor fuente incompatible en {key}")
        for key in ("n_needs", "d_predictor"):
            if source_model[key] != asdict(model_cfg)[key]:
                raise RuntimeError(f"predictor fuente incompatible en {key}")
        state = model.state_dict()
        for name, value in source["state_dict"].items():
            if name.startswith(("predictor.", "predictor_head.")):
                state[name] = value
        model.load_state_dict(state)
        # Mantiene idéntica la secuencia de mundos de la fase controller.
        for _ in range(args.predictor_steps):
            dataset.batch(args.batch_size, hazards=True)
        trace["phase"].append("predictor_reused")
        trace["step"].append(args.predictor_steps)
        trace["loss"].append(None)
        for key in ("homeostatic", "viability", "effort", "prediction",
                    "survival_soft"):
            trace[key].append(None)
        print(f"  [{variant} s{seed} predictor] REUSE {predictor_source}",
              flush=True)
    else:
        predictor_optimizer = torch.optim.AdamW(
            model.predictor_parameters, lr=args.predictor_lr,
            weight_decay=1e-4, betas=(0.9, 0.95))
        for step in range(1, args.predictor_steps + 1):
            _set_predictor_lr(predictor_optimizer, args.predictor_lr, step,
                              args.warmup_steps)
            scenario = dataset.batch(args.batch_size, hazards=True).to(device)
            loss = model.prediction_loss(scenario, env_cfg)
            predictor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor_parameters, 1.0)
            predictor_optimizer.step()
            if (step == 1 or step % args.log_every == 0
                    or step == args.predictor_steps):
                value = float(loss.detach())
                trace["phase"].append("predictor")
                trace["step"].append(step)
                trace["loss"].append(value)
                for key in ("homeostatic", "viability", "effort"):
                    trace[key].append(None)
                trace["prediction"].append(value)
                trace["survival_soft"].append(None)
                print(f"  [{variant} s{seed} predictor] "
                      f"{step:4d}/{args.predictor_steps} mse={value:.5f}",
                      flush=True)

    # El modelo del mundo se congela: las diferencias posteriores sólo pueden
    # proceder del controlador/HBP, no de predictores divergentes.
    for parameter in model.predictor_parameters:
        parameter.requires_grad_(False)
    controller_optimizer = torch.optim.AdamW(
        model.controller_parameters, lr=args.controller_lr,
        weight_decay=1e-4, betas=(0.9, 0.95))

    for step in range(1, args.steps + 1):
        _set_lr(controller_optimizer, args.controller_lr, step,
                args.steps, args.warmup_steps)
        scenario = dataset.batch(args.batch_size, hazards=True).to(device)
        result = model.rollout(
            scenario, env_cfg, hard_actions=False, straight_through=True)
        # prediction es constante en esta fase. Usamos sólo control + términos
        # HBP para que el target futuro no supervise directamente las acciones.
        loss = result["losses"]["control"]
        if model.use_hbp:
            loss = (loss + model_cfg.beta_intero * result["losses"]["intero"]
                    + model_cfg.beta_homeo * result["losses"]["hbp_homeo"]
                    + model_cfg.beta_stab * result["losses"]["stability"])
        controller_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.controller_parameters, 1.0)
        controller_optimizer.step()

        if (step == 1 or step % args.log_every == 0
                or step == args.steps):
            with torch.no_grad():
                levels = result["levels"][:, 1:]
                survival = float((levels >= env_cfg.viability_threshold)
                                 .flatten(1).all(dim=1).float().mean())
            values = {key: float(result["losses"][key].detach())
                      for key in ("homeostatic", "viability",
                                  "effort", "prediction")}
            trace["phase"].append("controller")
            trace["step"].append(step)
            trace["loss"].append(float(loss.detach()))
            for key in values:
                trace[key].append(values[key])
            trace["survival_soft"].append(survival)
            print(f"  [{variant} s{seed} controller] "
                  f"{step:4d}/{args.steps} loss={float(loss.detach()):.4f} "
                  f"homeo={values['homeostatic']:.4f} "
                  f"viol={values['viability']:.4f} "
                  f"survival={survival:.3f} effort={values['effort']:.3f}",
                  flush=True)

    checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{variant}_seed{seed}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": asdict(model_cfg),
        "environment_config": asdict(env_cfg),
        "meta": {
            "variant": variant, "seed": seed,
            "external_goal_supplied": False,
            "predictor_frozen_during_control": True,
            "predictor_reused_from": (
                os.path.abspath(predictor_source)
                if predictor_source is not None else None),
            "phase": "AHA-1 anticipation",
        },
    }, checkpoint_path)

    evaluation_dataset = AnticipatoryHomeostasisDataset(
        env_cfg, seed=seed + 100_000)
    evaluation = evaluate_aha(
        model, evaluation_dataset, device, n_batches=args.eval_batches,
        batch_size=args.eval_batch_size, seed=args.eval_seed + seed)
    evaluation["training"] = {
        "seed": seed, "predictor_steps": args.predictor_steps,
        "controller_steps": args.steps, "batch_size": args.batch_size,
        "predictor_lr": args.predictor_lr,
        "controller_lr": args.controller_lr,
        "predictor_frozen_during_control": True,
        "predictor_reused_from": (
            os.path.abspath(predictor_source)
            if predictor_source is not None else None),
        "trace": trace, "elapsed_s": time.time() - started,
    }
    evaluation["model_config"] = asdict(model_cfg)
    evaluation["environment_config"] = asdict(env_cfg)
    evaluation["provenance"] = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "torch": torch.__version__, "python": platform.python_version(),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device)
                        if device.type == "cuda" else "cpu"),
    }
    return evaluation, checkpoint_path


def _aggregate(results: list[dict]) -> dict:
    grouped = {variant: [] for variant in VARIANTS}
    for result in results:
        grouped[result["variant"]].append(result)
    summary = {}
    for variant, cells in grouped.items():
        if not cells:
            continue
        summary[variant] = {
            "n_seeds": len(cells),
            "seeds": [cell["training"]["seed"] for cell in cells],
            "normal": {
                key: statistics.mean(
                    cell["conditions"]["normal"][key] for cell in cells)
                for key in ("survival_rate", "violation_rate",
                            "mean_homeostatic_deficit",
                            "mean_absolute_homeostatic_error",
                            "anticipatory_event_rate",
                            "blank_initiative_rate", "action_rate",
                            "false_action_fraction", "prediction_mse",
                            "prediction_event_contrast",
                            "prediction_target_correlation")
            },
            "causal": {
                key: statistics.mean(
                    cell["causal_effects"][key] for cell in cells)
                for key in ("prediction_on_survival",
                            "prediction_on_violation_rate",
                            "prediction_on_blank_initiative",
                            "valid_cue_on_survival",
                            "valid_cue_on_blank_initiative")
            },
            "need_transplant": {
                key: statistics.mean(
                    cell["causal_effects"]["need_transplant"][key]
                    for cell in cells)
                for key in ("mean_target_probability_effect",
                            "mean_goal_flip_rate")
            },
        }

    by_cell = {(result["variant"], result["training"]["seed"]): result
               for result in results}
    seeds = sorted({result["training"]["seed"] for result in results})
    comparisons = {}
    pvalues = {}
    for numerator, denominator in (
            ("gating_wm", "reactive"),
            ("hbp_first", "gating_wm"),
            ("hbp_full", "gating_wm"),
            ("hbp_full", "hbp_first")):
        if any((numerator, seed) not in by_cell or
               (denominator, seed) not in by_cell for seed in seeds):
            continue
        key = f"{numerator}_minus_{denominator}"
        metric_differences = {}
        tests = {}
        for metric in ("survival_rate", "blank_initiative_rate"):
            differences = [
                by_cell[(numerator, seed)]["conditions"]["normal"][metric]
                - by_cell[(denominator, seed)]["conditions"]["normal"][metric]
                for seed in seeds]
            metric_differences[metric] = differences
            tests[metric] = _exact_signflip(differences)
            pvalues[f"{key}:{metric}"] = tests[metric]["p_exact_two_sided"]
        # Menos violaciones es mejor; se orienta positivo a favor del numerador.
        violation_differences = [
            by_cell[(denominator, seed)]["conditions"]["normal"]
            ["violation_rate"]
            - by_cell[(numerator, seed)]["conditions"]["normal"]
            ["violation_rate"] for seed in seeds]
        metric_differences["violation_reduction"] = violation_differences
        tests["violation_reduction"] = _exact_signflip(violation_differences)
        pvalues[f"{key}:violation_reduction"] = tests[
            "violation_reduction"]["p_exact_two_sided"]
        comparisons[key] = {
            "per_seed_differences": metric_differences,
            "mean_differences": {metric: statistics.mean(values)
                                 for metric, values
                                 in metric_differences.items()},
            "seed_level_exact_signflip": tests,
        }
    holm = _holm_adjust(pvalues) if pvalues else {}
    for key, comparison in comparisons.items():
        comparison["holm_adjusted_p"] = {
            metric: holm[f"{key}:{metric}"]
            for metric in comparison["seed_level_exact_signflip"]}

    # Familia confirmatoria fijada antes de ejecutar confirmatory_v1. Los dos
    # primeros tests exigen conducta anticipatoria útil frente al control; los
    # otros dos exigen que esa conducta dependa tanto del mecanismo predictivo
    # como del contenido causal de los cues. Todos están orientados de modo que
    # un valor positivo favorece la hipótesis AHA.
    confirmatory = None
    required = (("gating_wm", seed) in by_cell
                and ("reactive", seed) in by_cell for seed in seeds)
    if seeds and all(required):
        primary_differences = {
            "gating_minus_reactive_survival": [
                by_cell[("gating_wm", seed)]["conditions"]["normal"]
                ["survival_rate"]
                - by_cell[("reactive", seed)]["conditions"]["normal"]
                ["survival_rate"] for seed in seeds],
            "gating_minus_reactive_blank_initiative": [
                by_cell[("gating_wm", seed)]["conditions"]["normal"]
                ["blank_initiative_rate"]
                - by_cell[("reactive", seed)]["conditions"]["normal"]
                ["blank_initiative_rate"] for seed in seeds],
            "prediction_lesion_on_gating_survival": [
                by_cell[("gating_wm", seed)]["causal_effects"]
                ["prediction_on_survival"] for seed in seeds],
            "valid_cue_on_gating_survival": [
                by_cell[("gating_wm", seed)]["causal_effects"]
                ["valid_cue_on_survival"] for seed in seeds],
        }
        primary_tests = {
            name: _exact_signflip(differences)
            for name, differences in primary_differences.items()}
        primary_holm = _holm_adjust({
            name: test["p_exact_two_sided"]
            for name, test in primary_tests.items()})
        decisions = {
            name: (statistics.mean(primary_differences[name]) > 0.0
                   and primary_holm[name] < 0.05)
            for name in primary_differences}
        confirmatory = {
            "status": ("pass" if all(decisions.values()) else "fail"),
            "decision_rule": (
                "all four effects must be positive and Holm-adjusted "
                "two-sided exact sign-flip p < 0.05"),
            "alpha_familywise": 0.05,
            "n_paired_seeds": len(seeds),
            "seeds": seeds,
            "mean_effects": {
                name: statistics.mean(differences)
                for name, differences in primary_differences.items()},
            "seed_level_exact_signflip": primary_tests,
            "holm_adjusted_p": primary_holm,
            "criterion_passed": decisions,
        }
    return {"summary": summary,
            "paired_architecture_contrasts": comparisons,
            "confirmatory_primary_family": confirmatory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=_csv_strings,
                        default=list(VARIANTS))
    parser.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    parser.add_argument("--predictor-steps", type=int, default=600)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--n-events", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-predictor", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--violation-weight", type=float, default=12.0)
    parser.add_argument("--predictor-lr", type=float, default=2e-3)
    parser.add_argument("--controller-lr", type=float, default=2e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=20260721)
    parser.add_argument("--out-dir", default="results_aha")
    parser.add_argument(
        "--summary-file", default="summary.json",
        help="nombre del resumen dentro de out-dir; permite ejecución por shards")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-existing",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reuse-paired-predictor",
                        action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        parser.error(f"variantes desconocidas: {unknown}")
    if not 1 <= len(set(args.seeds)) <= 20:
        parser.error("se requieren 1..20 seeds distintas")
    if min(args.predictor_steps, args.steps, args.batch_size,
           args.eval_batches, args.eval_batch_size) < 1:
        parser.error("steps/batches deben ser positivos")

    env_cfg = AHAEnvironmentConfig(
        horizon=args.horizon, n_events=args.n_events)
    env_cfg.validate()
    device = torch.device(args.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    print("AHA-1: agencia homeostática anticipatoria")
    print(f"  device={device} variantes={args.variants} seeds={args.seeds}")
    print("  sin goal_id; predictor causal congelado; acciones demoradas; "
          "lesión + cue shuffle + trasplante de necesidad")

    results = []
    predictor_sources: dict[int, str] = {}
    for variant in args.variants:
        for seed in args.seeds:
            path = os.path.join(args.out_dir, f"{variant}_seed{seed}.json")
            checkpoint = os.path.join(
                args.out_dir, "checkpoints", f"{variant}_seed{seed}.pt")
            if (args.resume_existing and os.path.exists(path)
                    and os.path.exists(checkpoint)):
                with open(path, encoding="utf-8") as stream:
                    result = json.load(stream)
                print(f"  REUSE {variant} s{seed}: {path} | {checkpoint}")
            else:
                result, checkpoint = train_one(
                    variant, seed, args, env_cfg, device,
                    predictor_source=(predictor_sources.get(seed)
                                      if args.reuse_paired_predictor else None))
                _atomic_json(result, path)
            predictor_sources.setdefault(seed, checkpoint)
            results.append(result)
            normal = result["conditions"]["normal"]
            causal = result["causal_effects"]
            print(f"  RESULT {variant} s{seed}: "
                  f"survival={normal['survival_rate']:.3f} "
                  f"viol={normal['violation_rate']:.3f} "
                  f"anticip={normal['anticipatory_event_rate']:.3f} "
                  f"blank={normal['blank_initiative_rate']:.3f} "
                  f"Δlesion_survival={causal['prediction_on_survival']:+.3f}")
            print(f"    {path} | {checkpoint}")

    aggregate = {
        "design": {
            "phase": "AHA-1 anticipation",
            "roadmap": ["self_regulation", "anticipation",
                        "endogenous_goals", "discovered_goals",
                        "self_model", "metacognition"],
            "variants": args.variants, "seeds": args.seeds,
            "predictor_steps": args.predictor_steps,
            "controller_steps": args.steps,
            "environment": asdict(env_cfg),
            "external_goal_supplied": False,
            "reuse_paired_predictor": bool(args.reuse_paired_predictor),
            "primary_claim_if_positive": (
                "causal anticipatory self-regulation; not consciousness"),
        },
        **_aggregate(results),
    }
    summary_path = os.path.join(args.out_dir, args.summary_file)
    _atomic_json(aggregate, summary_path)
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
