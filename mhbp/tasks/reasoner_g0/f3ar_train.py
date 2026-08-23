"""
F3a-R — Entrenamiento (PREREG_F3A_R.md, rama C del fork F3b).

24 celdas: {miura_mhbp_pi1, miura_mhbp_noc} × 6 seeds × {indist, ood},
protocolo v3 idéntico a f3a_train (MAX_STEPS, N_MAX, streams y seeds de F3a).
Los comparadores por-tick e incumbentes NO se re-entrenan (celdas F3a
existentes; equivalencia de código verificada por T1 de tests_f3b_wiring).

Sharding por brazo para 2 workers:
  python -m mhbp.tasks.reasoner_g0.f3ar_train --arm pi1
  python -m mhbp.tasks.reasoner_g0.f3ar_train --arm noc
Resumible (skip por celda completada). --smoke: 1 celda × 60 pasos.
"""
import argparse
import json
import os
import time

from training.config import TrainConfig
from training.trainer import train_run
from .g0_train import RES, CKPT, MAX_STEPS, N_MAX, cell_id

GENS = "cycle_transp"
VARIANTS = {"pi1": "miura_mhbp_pi1", "noc": "miura_mhbp_noc",
            "pt": "miura_mhbp_ptR",   # control fresco (robustez R1)
            "hbpr": "hbp_fullR"}      # incumbente fresco (enmienda 1)


def cells(variant):
    out = []
    for seed in range(6):
        for regime, tmw in (("indist", 24), ("ood", 12)):
            out.append((variant, seed, regime, tmw))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(VARIANTS), required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="1 celda × 60 pasos (verificación pre-noche)")
    args = ap.parse_args()
    variant = VARIANTS[args.arm]
    todo = cells(variant)
    if args.smoke:
        todo = todo[:1]
    steps = 60 if args.smoke else MAX_STEPS
    print(f"F3a-R[{args.arm}]: {len(todo)} celdas ({variant}, protocolo v3, "
          f"max_steps={steps})", flush=True)
    t0 = time.time()
    for i, (var, seed, regime, tmw) in enumerate(todo, 1):
        cid = cell_id(var, GENS, seed, regime)
        if args.smoke:
            cid += "_smoke"
        jpath = os.path.join(RES, f"{cid}.json")
        cfinal = os.path.join(CKPT, f"{cid}.pt")
        if os.path.exists(jpath) and os.path.exists(cfinal):
            print(f"[{i:2d}/{len(todo)}] skip {cid}", flush=True)
            continue
        cfg = TrainConfig(task="permcomp", perm_gens=GENS, variant=var,
                          max_writes=24, train_max_writes=tmw,
                          max_steps=steps, seed=seed, max_halt_steps=N_MAX,
                          out_dir=CKPT)
        res = train_run(cfg, results_dir=None, save_ckpt=True, verbose=False)
        cpath = os.path.join(CKPT, f"{var}_seed{seed}_final.pt")
        if os.path.exists(cpath):
            os.replace(cpath, cfinal)
        res["_cell"] = {"variant": var, "gens": GENS, "seed": seed,
                        "regime": regime, "protocol": "v3+ckpt",
                        "phase": "f3a_r"}
        tmp = jpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(res, f)
        os.replace(tmp, jpath)
        a = res["final_acc"]
        cd = res.get("compute_diag") or {}
        corr = cd.get("corr_K_niter_ood")
        corr_s = f"{corr:+.2f}" if corr is not None else "  -  "
        print(f"[{i:2d}/{len(todo)}] {cid:44s} largo={a['largo']:.3f} "
              f"corrOOD={corr_s} ({(time.time() - t0) / 60:.0f} min)",
              flush=True)
    print(f"F3a-R[{args.arm}] COMPLETO en {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
