"""
MiuraCognitive - Orquestador del estudio de ablations
======================================================
Lanza las 4 variantes × N semillas de forma secuencial, guardando un JSON
de resultados por run en results/. Luego se agrega con eval/aggregate.py.

Uso:
    # Estudio completo (4 variantes × 3 semillas) a 5000 pasos
    PYTHONPATH=. python -m experiments.run_ablations --max_steps 5000 --seeds 0 1 2

    # Prueba rápida
    PYTHONPATH=. python -m experiments.run_ablations --max_steps 200 --seeds 0

Tras terminar, ejecutar:
    PYTHONPATH=. python -m eval.aggregate

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
import argparse
import time

from training.config import TrainConfig
from training.trainer import train_run

VARIANTS = ["vanilla", "gating", "gating_wm", "hbp_first", "hbp_full",
            "hbp_mix", "hbp_kdv", "hbp_elliptic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--max_distractors", type=int, default=60)
    ap.add_argument("--task", default="runsum", choices=["runsum", "permcomp", "iterfunc", "recall"])
    ap.add_argument("--max_writes", type=int, default=16)
    ap.add_argument("--train_max_writes", type=int, default=None)
    ap.add_argument("--halt_mod_gain", type=float, default=2.0)
    ap.add_argument("--max_halt_steps", type=int, default=10)
    ap.add_argument("--perm_gens", default="adjacent", choices=["adjacent", "cycle_transp"])
    args = ap.parse_args()

    total_runs = len(args.variants) * len(args.seeds)
    print("=" * 70)
    print(f"ESTUDIO DE ABLATIONS: {len(args.variants)} variantes × "
          f"{len(args.seeds)} semillas = {total_runs} runs")
    print(f"Pasos por run: {args.max_steps} | distractores máx: {args.max_distractors}")
    print("=" * 70)

    t0 = time.time()
    run_idx = 0
    summary = []
    for variant in args.variants:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n----- Run {run_idx}/{total_runs}: {variant} (seed {seed}) -----")
            cfg = TrainConfig(variant=variant, max_steps=args.max_steps,
                              seed=seed, max_distractors=args.max_distractors,
                              task=args.task, max_writes=args.max_writes,
                              train_max_writes=(args.train_max_writes if args.train_max_writes is not None else args.max_writes),
                              halt_mod_gain=args.halt_mod_gain, perm_gens=args.perm_gens,
                              max_halt_steps=args.max_halt_steps)
            res = train_run(cfg, results_dir=args.results_dir, verbose=True)
            summary.append((variant, seed, res["final_acc"]))

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"ESTUDIO COMPLETO en {elapsed/60:.1f} min")
    print("=" * 70)
    print(f"{'variante':12s} {'seed':>4s} {'corto':>8s} {'medio':>8s} {'largo':>8s}")
    for variant, seed, acc in summary:
        print(f"{variant:12s} {seed:>4d} {acc['corto']:>8.3f} "
              f"{acc['medio']:>8.3f} {acc['largo']:>8.3f}")
    print("\nResultados en JSON guardados en:", args.results_dir)
    print("Ahora ejecuta:  PYTHONPATH=. python -m eval.aggregate")


if __name__ == "__main__":
    main()
