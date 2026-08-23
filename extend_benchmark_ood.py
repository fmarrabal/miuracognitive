"""Extiende las celdas OOD del benchmark a n=10 para dar poder estadístico al
hallazgo central (adaptividad de cómputo OOD). Corre seeds 3-9 para las 3
variantes con reasoner en los 2 conjuntos de generadores (42 runs)."""
import json, os, itertools, time
from training.config import TrainConfig
from training.trainer import train_run

OUT = "results_benchmark"
VARIANTS = ["gating_wm", "hbp_first", "hbp_full"]
GENS = ["adjacent", "cycle_transp"]
SEEDS = [3, 4, 5, 6, 7, 8, 9]

cells = list(itertools.product(GENS, VARIANTS, SEEDS))
t0 = time.time()
for i, (gens, variant, seed) in enumerate(cells, 1):
    name = f"ood_{gens}_{variant}_seed{seed}.json"
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        continue
    cfg = TrainConfig(task="permcomp", perm_gens=gens, variant=variant, max_writes=24,
                      train_max_writes=12, max_steps=2500, seed=seed)
    res = train_run(cfg, results_dir=None, verbose=False)
    res["_cell"] = {"regime": "ood", "gens": gens, "variant": variant, "seed": seed}
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{i:2d}/{len(cells)}] {name:40s} corr={res['compute_diag']['corr_K_niter']:+.2f} "
          f"({(time.time()-t0)/60:.0f} min)", flush=True)
print(f"\nEXTENSION COMPLETA en {(time.time()-t0)/60:.1f} min. -> python benchmark_report.py")
