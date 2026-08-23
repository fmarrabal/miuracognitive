"""
BENCHMARK v4 — des-confusión del ORDEN + baseline GRU de interfaz
equiparada (PREREG_V4.md; condiciones de aceptación del revisor empírico).

Protocolo idéntico a v3 (OOD, 2500 pasos, N_max=24, pin_fp32) con seeds
FRESCAS 20-39 y tres brazos: hbp_full · hbp_first_eq (1er orden, topes
equiparados, IMEX implícito) · hbp_gru (núcleo GRU, interfaz idéntica).
Bucle seed-EXTERNO: un corte parcial deja n balanceado entre brazos.
REANUDABLE por fichero.

Uso:  $env:PYTHONPATH="." ; python benchmark_v4.py [--seeds 20 40] [--smoke]
"""
import argparse
import json
import os
import time

from training.config import TrainConfig
from training.trainer import train_run

OUT = "results_benchmark_v4"
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs=2, default=[20, 40])
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

MAX_STEPS = 120 if args.smoke else 2500
N_MAX = 24
GENS = ["adjacent", "cycle_transp"]
ARMS = ["hbp_full", "hbp_first_eq", "hbp_gru"]

CELLS = []
for s in range(args.seeds[0], args.seeds[1]):      # seed EXTERNO
    for gens in GENS:
        for v in ARMS:
            CELLS.append((gens, "ood", 12, v, s))

print(f"BENCHMARK v4: {len(CELLS)} celdas (pasos={MAX_STEPS}, "
      f"seeds {args.seeds[0]}-{args.seeds[1] - 1})", flush=True)
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
        except Exception:
            pass
    cfg = TrainConfig(task="permcomp", perm_gens=gens, variant=variant,
                      max_writes=24, train_max_writes=tmw,
                      max_steps=MAX_STEPS, seed=seed, max_halt_steps=N_MAX)
    res = train_run(cfg, results_dir=None, verbose=False)
    res["_cell"] = {"regime": regime, "gens": gens, "variant": variant,
                    "seed": seed, "prereg": "PREREG_V4"}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, path)
    a = res["final_acc"]
    cd = res.get("compute_diag") or {}
    corr = cd.get("corr_K_niter_ood", cd.get("corr_K_niter"))
    corr_s = f"{corr:+.2f}" if corr is not None else "  -  "
    print(f"[{i:3d}/{len(CELLS)}] {name:48s} largo={a['largo']:.3f} "
          f"corrOOD={corr_s} ({(time.time() - t0) / 60:.0f} min)",
          flush=True)
print(f"\nBENCHMARK v4 COMPLETO en {(time.time() - t0) / 60:.1f} min.",
      flush=True)
