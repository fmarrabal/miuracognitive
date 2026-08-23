"""
BENCHMARK COMPLETO — MiuraCognitive
====================================
Matriz uniforme y reproducible:
  5 variantes  x  2 conjuntos de generadores (S_5)  x  2 regímenes  x  3 semillas
  = 60 runs.

Variantes:  vanilla, gating, gating_wm (WM sin HBP, control), hbp_first, hbp_full.
Generadores: adjacent (transposiciones adyacentes) | cycle_transp (5-ciclo + transp).
Regímenes:  indist (train=eval, K<=24) | ood (train K<=12, eval K<=24 = extrapolación).

Guarda un JSON por celda en results_benchmark/ (RESUMIBLE: salta las existentes).
Métricas por celda: accuracy por bucket (corto/medio/largo) y, para variantes con
reasoner, E[n_iter] y corr(K, n_iter) (adaptividad de cómputo).

Tras terminar:  python benchmark_report.py

Uso:  $env:PYTHONPATH="." ; python benchmark.py
"""
import json, os, itertools, time
from training.config import TrainConfig
from training.trainer import train_run

OUT = "results_benchmark"
os.makedirs(OUT, exist_ok=True)

VARIANTS = ["vanilla", "gating", "gating_wm", "hbp_first", "hbp_full"]
GENS = ["adjacent", "cycle_transp"]
REGIMES = [("indist", 24), ("ood", 12)]   # (nombre, train_max_writes)
SEEDS = [0, 1, 2]
MAX_STEPS = 2500

cells = list(itertools.product(GENS, REGIMES, VARIANTS, SEEDS))
print(f"BENCHMARK: {len(cells)} celdas "
      f"({len(VARIANTS)} variantes x {len(GENS)} generadores x {len(REGIMES)} regímenes x {len(SEEDS)} semillas)")
t0 = time.time()
for i, (gens, (regime, tmw), variant, seed) in enumerate(cells, 1):
    name = f"{regime}_{gens}_{variant}_seed{seed}.json"
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        print(f"[{i:2d}/{len(cells)}] skip {name}", flush=True)
        continue
    cfg = TrainConfig(task="permcomp", perm_gens=gens, variant=variant, max_writes=24,
                      train_max_writes=tmw, max_steps=MAX_STEPS, seed=seed)
    res = train_run(cfg, results_dir=None, verbose=False)
    res["_cell"] = {"regime": regime, "gens": gens, "variant": variant, "seed": seed}
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    a = res["final_acc"]
    cd = res.get("compute_diag")
    corr = f"{cd['corr_K_niter']:+.2f}" if cd else "  -  "
    print(f"[{i:2d}/{len(cells)}] {name:46s} largo={a['largo']:.3f} corr={corr} "
          f"({(time.time()-t0)/60:.0f} min)", flush=True)
print(f"\nBENCHMARK COMPLETO en {(time.time()-t0)/60:.1f} min. Ahora: python benchmark_report.py")
