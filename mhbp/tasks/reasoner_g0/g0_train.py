"""
G0 — Entrenamiento instrumentable (PREREG_G0.md).

30 celdas: variantes {vanilla, gating_wm, hbp_full} × gens {adjacent,
cycle_transp} × seeds {0,1,2} × regímenes {indist (train K<=24), ood (train
K<=12)} — vanilla solo indist (es el control T, sin reasoner). Protocolo v3
(2500 pasos, N_max=24, pin_fp32). Guarda CHECKPOINT por celda (los
instrumentos M/I se corren sobre el modelo entrenado). Resumible.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.g0_train
"""
import json
import os
import time

from training.config import TrainConfig
from training.trainer import train_run

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "ckpts")
os.makedirs(RES, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

MAX_STEPS = 2500
N_MAX = 24
SEEDS = [0, 1, 2]
GENS = ["adjacent", "cycle_transp"]


def cells():
    out = []
    for gens in GENS:
        for seed in SEEDS:
            for variant in ("gating_wm", "hbp_full"):
                out.append((variant, gens, seed, "indist", 24))
                out.append((variant, gens, seed, "ood", 12))
            out.append(("vanilla", gens, seed, "indist", 24))
    return out


def cell_id(variant, gens, seed, regime):
    return f"{variant}_{gens}_{regime}_s{seed}"


def main():
    todo = cells()
    print(f"G0-train: {len(todo)} celdas (protocolo v3: {MAX_STEPS} pasos, N_max={N_MAX})")
    t0 = time.time()
    for i, (variant, gens, seed, regime, tmw) in enumerate(todo, 1):
        cid = cell_id(variant, gens, seed, regime)
        jpath = os.path.join(RES, f"{cid}.json")
        cpath = os.path.join(CKPT, f"{variant}_seed{seed}_final.pt")   # nombre del trainer
        cfinal = os.path.join(CKPT, f"{cid}.pt")
        if os.path.exists(jpath) and os.path.exists(cfinal):
            print(f"[{i:2d}/{len(todo)}] skip {cid}", flush=True)
            continue
        cfg = TrainConfig(task="permcomp", perm_gens=gens, variant=variant,
                          max_writes=24, train_max_writes=tmw,
                          max_steps=MAX_STEPS, seed=seed, max_halt_steps=N_MAX,
                          out_dir=CKPT)
        res = train_run(cfg, results_dir=None, save_ckpt=True, verbose=False)
        # el trainer guarda como {variant}_seed{seed}_final.pt: renombrar a la celda
        if os.path.exists(cpath):
            os.replace(cpath, cfinal)
        res["_cell"] = {"variant": variant, "gens": gens, "seed": seed,
                        "regime": regime, "protocol": "v3+ckpt"}
        tmp = jpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(res, f)
        os.replace(tmp, jpath)
        a = res["final_acc"]
        cd = res.get("compute_diag") or {}
        corr = cd.get("corr_K_niter_ood")
        corr_s = f"{corr:+.2f}" if corr is not None else "  -  "
        print(f"[{i:2d}/{len(todo)}] {cid:38s} largo={a['largo']:.3f} "
              f"corrOOD={corr_s} ({(time.time()-t0)/60:.0f} min)", flush=True)
    print(f"G0-train COMPLETO en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
