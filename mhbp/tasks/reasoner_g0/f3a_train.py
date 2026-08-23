"""
F3a — Entrenamiento (PREREG_F3A.md). Reutiliza las 12 celdas de G0
(hbp_full/gating_wm, cycle, seeds 0-2) y entrena las nuevas: miura_mhbp
seeds 0-5 × {indist, ood} + incumbentes seeds 3-5 × {indist, ood}.
Mismo protocolo v3, mismos nombres de celda que g0_train (los instrumentos y
el report son compartidos). Resumible.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.f3a_train
"""
import json
import os
import time

from training.config import TrainConfig
from training.trainer import train_run
from .g0_train import RES, CKPT, MAX_STEPS, N_MAX, cell_id

GENS = "cycle_transp"


def cells():
    out = []
    for seed in range(6):
        for regime, tmw in (("indist", 24), ("ood", 12)):
            out.append(("miura_mhbp", seed, regime, tmw))
    for seed in (3, 4, 5):
        for variant in ("hbp_full", "gating_wm"):
            for regime, tmw in (("indist", 24), ("ood", 12)):
                out.append((variant, seed, regime, tmw))
    return out


def main():
    todo = cells()
    print(f"F3a-train: {len(todo)} celdas nuevas (cycle_transp, protocolo v3)")
    t0 = time.time()
    for i, (variant, seed, regime, tmw) in enumerate(todo, 1):
        cid = cell_id(variant, GENS, seed, regime)
        jpath = os.path.join(RES, f"{cid}.json")
        cfinal = os.path.join(CKPT, f"{cid}.pt")
        if os.path.exists(jpath) and os.path.exists(cfinal):
            print(f"[{i:2d}/{len(todo)}] skip {cid}", flush=True)
            continue
        cfg = TrainConfig(task="permcomp", perm_gens=GENS, variant=variant,
                          max_writes=24, train_max_writes=tmw,
                          max_steps=MAX_STEPS, seed=seed, max_halt_steps=N_MAX,
                          out_dir=CKPT)
        res = train_run(cfg, results_dir=None, save_ckpt=True, verbose=False)
        cpath = os.path.join(CKPT, f"{variant}_seed{seed}_final.pt")
        if os.path.exists(cpath):
            os.replace(cpath, cfinal)
        res["_cell"] = {"variant": variant, "gens": GENS, "seed": seed,
                        "regime": regime, "protocol": "v3+ckpt", "phase": "f3a"}
        tmp = jpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(res, f)
        os.replace(tmp, jpath)
        a = res["final_acc"]
        cd = res.get("compute_diag") or {}
        corr = cd.get("corr_K_niter_ood")
        corr_s = f"{corr:+.2f}" if corr is not None else "  -  "
        print(f"[{i:2d}/{len(todo)}] {cid:40s} largo={a['largo']:.3f} "
              f"corrOOD={corr_s} ({(time.time()-t0)/60:.0f} min)", flush=True)
    print(f"F3a-train COMPLETO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
