"""
G0 — Corre los instrumentos M/I sobre los checkpoints entrenados (resumible).

Por (variant ∈ {gating_wm, hbp_full}, gens, seed):
  con el ckpt INDIST: M1 (probes F_full/F_pref), M2 (swap), I (AUC halting)
  con el ckpt OOD   : M3 (fidelidad de trayectoria vs acierto OOD)
T y P se leen de los JSON de g0_train (evaluate/compute_diag).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.g0_run
"""
import json
import os
import time

import torch

from training.config import TrainConfig
from training.trainer import build_model, build_dataset
from .instruments import (collect_trajectories, train_probes, probe_accuracy,
                          per_sample_fidelity, swap_experiment, spearman,
                          pearson, auc)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "ckpts")
N_MAX = 24
SEEDS = [0, 1, 2]
GENS = ["adjacent", "cycle_transp"]
VARIANTS = ["gating_wm", "hbp_full"]
N_PROBE_TRAIN = 2560
N_PROBE_TEST = 768
N_OOD = 768


def load_model(cid_ckpt, variant, gens, device, dtype):
    cfg = TrainConfig(task="permcomp", perm_gens=gens, variant=variant,
                      max_writes=24, max_steps=1, seed=0, max_halt_steps=N_MAX)
    ds = build_dataset(cfg)
    model = build_model(variant, ds.vocab_size, cfg.seq_len,
                        max_halt_steps=N_MAX).to(device=device, dtype=dtype)
    sd = torch.load(os.path.join(CKPT, cid_ckpt), map_location=device,
                    weights_only=True)
    model.load_state_dict(sd["model"])
    if model.cfg.use_hbp:
        model.hbp.pin_fp32()
    model.eval()
    return model


def run_cell(variant, gens, seed, device, dtype):
    t0 = time.time()
    out = {"variant": variant, "gens": gens, "seed": seed}

    # ---------- ckpt INDIST: M1, M2, I ----------
    model = load_model(f"{variant}_{gens}_indist_s{seed}.pt", variant, gens,
                       device, dtype)
    mk_ds = lambda s, mw=24, mn=2: build_dataset(
        TrainConfig(task="permcomp", perm_gens=gens, max_writes=mw,
                    min_writes=mn, seed=s))
    # probes: streams frescos disjuntos de train (seed) y eval (seed+100k)
    ds_ptr = mk_ds(seed + 200_000)
    ds_pte = mk_ds(seed + 300_000)
    tr = collect_trajectories(model, ds_ptr, N_PROBE_TRAIN, device, dtype)
    te = collect_trajectories(model, ds_pte, N_PROBE_TEST, device, dtype)
    n_cls = model.lm_head.weight.shape[0]          # sobra; probes usan answer_size
    n_ans = 120
    heads_ans = train_probes(tr["feat_ans"], tr["ans"], n_ans, device)
    heads_mid = train_probes(tr["feat_mid"], tr["mid"], n_ans, device)
    F_full = probe_accuracy(heads_ans, te["feat_ans"], te["ans"], device)
    F_pref = probe_accuracy(heads_mid, te["feat_mid"], te["mid"], device)
    out_acc = float(te["correct"].float().mean())
    ticks = torch.arange(1, len(F_full) + 1).float()
    out["M1"] = {
        "F_full": F_full, "F_pref": F_pref, "output_acc": out_acc,
        "trend_spearman": spearman(ticks, torch.tensor(F_full)),
        "F_final_over_acc": (F_full[-1] / out_acc if out_acc > 0 else 0.0),
        "F_first": F_full[0], "F_last": F_full[-1],
    }
    # I: parada informada (sobre el stream de eval fresco)
    out["I"] = {"auc_pstop_correct": auc(te["p_stop"], te["correct"]),
                "auc_nexp_correct": auc(-te["n_exp"], te["correct"])}
    # M2: swap en el rango medio-largo
    ds_swap = mk_ds(seed + 400_000, mw=20, mn=8)
    out["M2"] = swap_experiment(model, ds_swap, device, dtype)
    del model
    torch.cuda.empty_cache()

    # ---------- ckpt OOD: M3 ----------
    model = load_model(f"{variant}_{gens}_ood_s{seed}.pt", variant, gens,
                       device, dtype)
    # probes entrenadas en el RANGO DE TRAIN del ckpt ood (K<=12)...
    ds_ptr2 = mk_ds(seed + 500_000, mw=12)
    tr2 = collect_trajectories(model, ds_ptr2, N_PROBE_TRAIN, device, dtype)
    heads2 = train_probes(tr2["feat_ans"], tr2["ans"], n_ans, device)
    # ...aplicadas a trayectorias OOD (K>=14)
    ds_ood = mk_ds(seed + 600_000, mw=24)
    ood = collect_trajectories(model, ds_ood, N_OOD, device, dtype, k_min=14)
    fid = per_sample_fidelity(heads2, ood["feat_ans"], ood["ans"], device)
    out["M3"] = {"corr_fidelity_correct": pearson(fid, ood["correct"].float()),
                 "fidelity_mean": float(fid.mean()),
                 "ood_acc": float(ood["correct"].float().mean()),
                 "n_ood": int(len(fid))}
    del model
    torch.cuda.empty_cache()
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for variant in VARIANTS:
        for gens in GENS:
            for seed in SEEDS:
                cid = f"mip_{variant}_{gens}_s{seed}"
                path = os.path.join(RES, f"{cid}.json")
                if os.path.exists(path):
                    print(f"skip {cid}", flush=True)
                    continue
                r = run_cell(variant, gens, seed, device, dtype)
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(r, f)
                os.replace(tmp, path)
                m1, m3 = r["M1"], r["M3"]
                print(f"OK {cid}: F1->{m1['F_first']:.2f}..{m1['F_last']:.2f} "
                      f"(acc={m1['output_acc']:.2f}) trend={m1['trend_spearman']:+.2f} "
                      f"AUC={r['I']['auc_pstop_correct']:.2f} "
                      f"M3corr={m3['corr_fidelity_correct']:+.2f} ({r['wall_s']}s)",
                      flush=True)
    print("G0-run COMPLETO")


if __name__ == "__main__":
    main()
