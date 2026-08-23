"""
BENCHMARK v2 — MiuraCognitive (post-auditoría)
===============================================
Cambios frente a v1 (motivados por la auditoría):
  - eval con seed disjunta (sin contaminación train->eval en régimen in-dist),
  - argmax restringido al sub-vocabulario de respuesta,
  - WM dentro del bucle del reasoner; interocepción por nodo; masked pooling,
  - N_max = 24 (>= K_max: sin techo arquitectónico de cómputo),
  - beta_final (peso a la respuesta final),
  - SEEDS FRESCAS PRE-REGISTRADAS (10..): nada de extensión post-hoc,
  - escritura atómica de JSONs + validación de config al reanudar.

DISEÑO PRE-REGISTRADO (declarado ANTES de mirar ningún número v2):
  - Análisis primario: contraste PAREADO por seed (Fisher-z de corr OOD-pura
    K>=14, hbp_full - gating_wm), n=10 seeds frescas, en 2 conjuntos de
    generadores; corrección Holm sobre los 4 contrastes (2 gens x {full,first}).
  - Secundario: accuracy por bucket (in-dist n=3; OOD n=10).

Uso:  $env:PYTHONPATH="." ; python benchmark_v2.py
"""
import json, os, itertools, time
from training.config import TrainConfig
from training.trainer import train_run

OUT = "results_benchmark_v2"
os.makedirs(OUT, exist_ok=True)

MAX_STEPS = 2500
N_MAX = 24
GENS = ["adjacent", "cycle_transp"]

# Celdas: (regime, train_max_writes, variantes, seeds)
CELLS = []
for gens in GENS:
    # in-dist: 5 variantes x 3 seeds frescas
    for v in ["vanilla", "gating", "gating_wm", "hbp_first", "hbp_full"]:
        for s in [10, 11, 12]:
            CELLS.append((gens, "indist", 24, v, s))
    # OOD (confirmatorio): 3 variantes x 10 seeds frescas
    for v in ["gating_wm", "hbp_first", "hbp_full"]:
        for s in range(10, 20):
            CELLS.append((gens, "ood", 12, v, s))

print(f"BENCHMARK v2: {len(CELLS)} celdas (N_max={N_MAX}, seeds frescas >=10)")
t0 = time.time()
for i, (gens, regime, tmw, variant, seed) in enumerate(CELLS, 1):
    name = f"{regime}_{gens}_{variant}_seed{seed}.json"
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        try:
            old = json.load(open(path, encoding="utf-8"))
            if old.get("cfg", {}).get("max_steps") == MAX_STEPS and \
               old.get("cfg", {}).get("max_halt_steps") == N_MAX:
                print(f"[{i:3d}/{len(CELLS)}] skip {name}", flush=True)
                continue
            print(f"[{i:3d}/{len(CELLS)}] config distinta, re-corre {name}", flush=True)
        except Exception:
            print(f"[{i:3d}/{len(CELLS)}] JSON corrupto, re-corre {name}", flush=True)
    cfg = TrainConfig(task="permcomp", perm_gens=gens, variant=variant, max_writes=24,
                      train_max_writes=tmw, max_steps=MAX_STEPS, seed=seed,
                      max_halt_steps=N_MAX)
    res = train_run(cfg, results_dir=None, verbose=False)
    res["_cell"] = {"regime": regime, "gens": gens, "variant": variant, "seed": seed}
    tmp = path + ".tmp"                       # escritura atómica
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, path)
    a = res["final_acc"]
    cd = res.get("compute_diag") or {}
    corr = cd.get("corr_K_niter_ood", cd.get("corr_K_niter"))
    corr_s = f"{corr:+.2f}" if corr is not None else "  -  "
    print(f"[{i:3d}/{len(CELLS)}] {name:44s} largo={a['largo']:.3f} corrOOD={corr_s} "
          f"({(time.time()-t0)/60:.0f} min)", flush=True)
print(f"\nBENCHMARK v2 COMPLETO en {(time.time()-t0)/60:.1f} min. -> python benchmark_report_v2.py")
