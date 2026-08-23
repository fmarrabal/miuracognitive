"""Probe v2d — pooling completo restaurado (scratchpad) + todos los fixes que
no dañan. 4 runs decisivos (seed 0, adjacent): {gating_wm, hbp_full} x {indist, ood}.
Criterio: recuperar E[n_iter] ~ 10+ y accuracy in-dist largo >= 0.3."""
import json, os, time
from training.config import TrainConfig
from training.trainer import train_run

os.makedirs("results_v2d", exist_ok=True)
t0 = time.time()
for regime, tmw in [("indist", 24), ("ood", 12)]:
    for variant in ["gating_wm", "hbp_full"]:
        tag = f"{regime}_{variant}"
        path = f"results_v2d/{tag}.json"
        if os.path.exists(path):
            print(f"skip {tag}", flush=True)
            continue
        cfg = TrainConfig(task="permcomp", perm_gens="adjacent", variant=variant,
                          max_writes=24, train_max_writes=tmw, max_steps=2500,
                          seed=0, max_halt_steps=24, wm_in_loop=True)
        res = train_run(cfg, results_dir=None, verbose=False)
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, indent=2)
        os.replace(path + ".tmp", path)
        a = res["final_acc"]; d = res["compute_diag"]
        print(f"{tag:22s} corto={a['corto']:.3f} medio={a['medio']:.3f} largo={a['largo']:.3f} "
              f"niter={d['mean_niter']:.1f} corr={d['corr_K_niter']:+.3f} "
              f"corrOOD={d.get('corr_K_niter_ood', float('nan')):+.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
print("PROBE v2d COMPLETO")
