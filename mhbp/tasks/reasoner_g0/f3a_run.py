"""
F3a — Instrumentos post-entrenamiento (PREREG_F3A v2).

Por seed 0-5 (cycle_transp):
  miura_mhbp (indist): M1 (+trend ON-POLICY), M2 (swap con swap_halves), I'
  miura_mhbp (ood)   : M3 (+fidelidad ON-POLICY)
  hbp_full / gating_wm (indist): I' (guardarraíl comparativo)
T/P/E se leen de los JSON de entrenamiento. Resumible.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.f3a_run
"""
import json
import math
import os
import time

import torch

from training.config import TrainConfig
from training.trainer import build_dataset
from .g0_run import load_model, N_MAX
from .instruments import (collect_trajectories, train_probes, probe_accuracy,
                          per_sample_fidelity, swap_experiment, spearman,
                          pearson, auc)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GENS = "cycle_transp"
SEEDS = [0, 1, 2, 3, 4, 5]


def mk_ds(seed, mw=24, mn=2):
    return build_dataset(TrainConfig(task="permcomp", perm_gens=GENS,
                                     max_writes=mw, min_writes=mn, seed=seed))


@torch.no_grad()
def onpolicy_fidelity(heads, feats, labels, n_exp, device):
    """Fidelidad por muestra restringida a t <= ceil(n_exp_b)."""
    hits = []
    for t, head in enumerate(heads):
        pred = head(feats[:, t, :].to(device)).argmax(-1).cpu()
        hits.append((pred == labels).float())
    H = torch.stack(hits, dim=1)                                # (B, T)
    fid = torch.zeros(H.shape[0])
    for b in range(H.shape[0]):
        n = max(1, min(int(math.ceil(float(n_exp[b]))), H.shape[1]))
        fid[b] = H[b, :n].mean()
    return fid


def run_iprime(variant, seed, device, dtype):
    model = load_model(f"{variant}_{GENS}_indist_s{seed}.pt", variant, GENS,
                       device, dtype)
    te = collect_trajectories(model, mk_ds(seed + 800_000), 512, device, dtype)
    out = {"iprime": auc(te["lam_best"], te["correct"]),
           "acc": float(te["correct"].float().mean())}
    del model
    torch.cuda.empty_cache()
    return out


def run_mhbp_battery(seed, device, dtype):
    out = {}
    model = load_model(f"miura_mhbp_{GENS}_indist_s{seed}.pt", "miura_mhbp",
                       GENS, device, dtype)
    tr = collect_trajectories(model, mk_ds(seed + 200_000), 2560, device, dtype)
    te = collect_trajectories(model, mk_ds(seed + 800_000), 768, device, dtype)
    heads = train_probes(tr["feat_ans"], tr["ans"], 120, device)
    F_full = probe_accuracy(heads, te["feat_ans"], te["ans"], device)
    out_acc = float(te["correct"].float().mean())
    ticks = torch.arange(1, len(F_full) + 1).float()
    n75 = max(3, int(math.ceil(float(te["n_exp"].quantile(0.75)))))
    out["M1"] = {
        "F_full": F_full, "output_acc": out_acc,
        "trend_full": spearman(ticks, torch.tensor(F_full)),
        "trend_onpolicy": spearman(ticks[:n75], torch.tensor(F_full[:n75])),
        "F_final_over_acc": (F_full[min(n75, len(F_full)) - 1] / out_acc
                             if out_acc > 0 else 0.0),
        "n75": n75,
    }
    out["Iprime"] = {"iprime": auc(te["lam_best"], te["correct"]), "acc": out_acc}
    out["M2"] = swap_experiment(model, mk_ds(seed + 400_000, mw=20, mn=8),
                                device, dtype)
    del model
    torch.cuda.empty_cache()

    model = load_model(f"miura_mhbp_{GENS}_ood_s{seed}.pt", "miura_mhbp",
                       GENS, device, dtype)
    tr2 = collect_trajectories(model, mk_ds(seed + 500_000, mw=12), 2560,
                               device, dtype)
    heads2 = train_probes(tr2["feat_ans"], tr2["ans"], 120, device)
    ood = collect_trajectories(model, mk_ds(seed + 600_000), 768, device,
                               dtype, k_min=14)
    fid_full = per_sample_fidelity(heads2, ood["feat_ans"], ood["ans"], device)
    fid_onp = onpolicy_fidelity(heads2, ood["feat_ans"], ood["ans"],
                                ood["n_exp"], device)
    out["M3"] = {
        "corr_full": pearson(fid_full, ood["correct"].float()),
        "corr_onpolicy": pearson(fid_onp, ood["correct"].float()),
        "ood_acc": float(ood["correct"].float().mean()),
    }
    del model
    torch.cuda.empty_cache()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for seed in SEEDS:
        # I' de incumbentes
        for variant in ("hbp_full", "gating_wm"):
            cid = f"f3a_iprime_{variant}_s{seed}"
            path = os.path.join(RES, f"{cid}.json")
            if os.path.exists(path):
                print(f"skip {cid}", flush=True)
                continue
            t0 = time.time()
            r = run_iprime(variant, seed, device, dtype)
            r["wall_s"] = round(time.time() - t0, 1)
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(r, f)
            os.replace(path + ".tmp", path)
            print(f"OK {cid}: I'={r['iprime']:.3f} ({r['wall_s']}s)", flush=True)
        # batería mhbp
        cid = f"f3a_mip_miura_mhbp_s{seed}"
        path = os.path.join(RES, f"{cid}.json")
        if os.path.exists(path):
            print(f"skip {cid}", flush=True)
            continue
        t0 = time.time()
        r = run_mhbp_battery(seed, device, dtype)
        r["wall_s"] = round(time.time() - t0, 1)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(r, f)
        os.replace(path + ".tmp", path)
        m1, m3 = r["M1"], r["M3"]
        print(f"OK {cid}: trend_onp={m1['trend_onpolicy']:+.2f} "
              f"I'={r['Iprime']['iprime']:.3f} M3onp={m3['corr_onpolicy']:+.2f} "
              f"({r['wall_s']}s)", flush=True)
    print("F3a-run COMPLETO")


if __name__ == "__main__":
    main()
