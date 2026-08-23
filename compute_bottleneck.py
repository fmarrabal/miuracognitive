"""CLI del experimento causal de cuello de botella de cómputo.

Ejemplo piloto:

  python compute_bottleneck.py --n-batches 4 --budgets 4,6,8,10

El script no entrena ni cambia el checkpoint. Guarda JSON atómico con las
cuotas, resultados binarios pareados y tests de McNemar exactos.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import torch

from data.synthetic_recall import PermutationCompositionDataset
from eval.compute_bottleneck import evaluate_compute_bottleneck
from miura_infer import MiuraModel


def _parse_budgets(value: str) -> list[int]:
    try:
        out = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("usa enteros separados por comas") from exc
    if not out:
        raise argparse.ArgumentTypeError("indica al menos un presupuesto")
    if len(set(out)) != len(out):
        raise argparse.ArgumentTypeError("los presupuestos no pueden repetirse")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/hbp_full_adjacent.pt")
    ap.add_argument("--out", default="results_compute_bottleneck/pilot.json")
    ap.add_argument("--budgets", type=_parse_budgets, default=[4, 6, 8, 10],
                    help="media exacta de pasos por muestra, p.ej. 4,6,8,10")
    ap.add_argument("--primary-budget", type=int, default=6,
                    help="presupuesto confirmatorio; los demás son exploratorios")
    ap.add_argument("--primary-control", choices=["shuffle_0", "uniform"],
                    default="shuffle_0",
                    help="control del contraste confirmatorio")
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-shuffles", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument("--eval-min-ops", type=int, default=2)
    ap.add_argument("--eval-max-ops", type=int, default=24)
    ap.add_argument("--device", default=None,
                    help="por defecto CUDA si está disponible; también cpu/cuda:0")
    args = ap.parse_args()

    if args.n_batches < 1 or args.batch_size < 2:
        ap.error("n-batches debe ser >=1 y batch-size >=2")
    if args.n_shuffles < 1:
        ap.error("n-shuffles debe ser >=1")
    if args.eval_min_ops < 1 or args.eval_max_ops < args.eval_min_ops:
        ap.error("rango de dificultad inválido")
    if args.primary_budget not in args.budgets:
        ap.error("primary-budget debe estar incluido en budgets")

    started = time.time()
    wrapper = MiuraModel.from_checkpoint(args.checkpoint, device=args.device)
    model = wrapper.model
    task = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["task"]
    gen_set = wrapper.meta.get("gen_set", "adjacent")
    ds = PermutationCompositionDataset(
        n_elems=int(task["n_elems"]), min_ops=args.eval_min_ops,
        max_ops=args.eval_max_ops, seq_len=int(task["seq_len"]),
        seed=args.seed + 100_000, gen_set=gen_set)

    print("Experimento causal de cuello de botella")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {wrapper.device}")
    print(f"  presupuestos medios: {args.budgets}")
    print(f"  muestras por presupuesto: {args.n_batches * args.batch_size}")
    result = evaluate_compute_bottleneck(
        model, ds, wrapper.device, mean_budgets=args.budgets,
        n_batches=args.n_batches, batch_size=args.batch_size,
        n_shuffles=args.n_shuffles, seed=args.seed,
        primary_mean_budget=args.primary_budget,
        primary_control=args.primary_control)
    result["provenance"] = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_meta": wrapper.meta,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": str(wrapper.device),
        "device_name": (torch.cuda.get_device_name(wrapper.device)
                        if wrapper.device.type == "cuda" else "cpu"),
        "elapsed_s": time.time() - started,
        "eval_min_ops": args.eval_min_ops,
        "eval_max_ops": args.eval_max_ops,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp, args.out)

    print(f"\nResultado pareado (aprendido − {args.primary_control})")
    for budget in args.budgets:
        cell = result["budgets"][str(budget)]
        cmp = cell["comparisons"][f"learned_vs_{args.primary_control}"]
        alloc = cell["allocation"]
        p_cluster = cmp["cluster_randomization"]["p_exact_two_sided"]
        mark = " [PRIMARIO]" if budget == args.primary_budget else ""
        print(f"  B={budget:2d}: Δacc={cmp['accuracy_diff_learned_minus_control']:+.4f} "
              f"p_cluster={p_cluster:.4g} p_Holm={cmp['p_holm_across_budgets']:.4g} "
              f"corr(K,q)={alloc['corr_difficulty_learned_quota']:+.3f} "
              f"q=[{alloc['quota_min']},{alloc['quota_max']}]{mark}")
    print(f"\nGuardado: {args.out}")


if __name__ == "__main__":
    main()
