"""
F3a-R — Batería post-entrenamiento + veredicto (PREREG_F3A_R.md).

Batería IDÉNTICA a f3a_run.run_mhbp_battery para miura_mhbp_pi1 y
miura_mhbp_noc (noc se evalúa CON su lesión activa: es su condición
operativa, fijada por build_model). Después, veredicto R1/R2/R3 contra los
comparadores F3a existentes (f3a_mip_miura_mhbp_s*.json por-tick;
f3a_m3ref_hbp_full.json incumbente).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.f3ar_run
"""
import json
import math
import os
import time

import torch

from training.config import TrainConfig
from training.trainer import build_dataset
from .g0_run import load_model
from .instruments import (collect_trajectories, train_probes, probe_accuracy,
                          per_sample_fidelity, swap_experiment, spearman,
                          pearson, auc)
from .f3a_run import mk_ds, onpolicy_fidelity

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GENS = "cycle_transp"
SEEDS = [0, 1, 2, 3, 4, 5]
VARIANTS = ("miura_mhbp_pi1", "miura_mhbp_noc", "miura_mhbp_ptR")
M3_ONLY = ("hbp_fullR",)     # incumbente fresco (enmienda 1): solo M3


def run_m3_only(variant, seed, device, dtype):
    """Solo la mitad M3 de la batería (celdas ood): re-estimación del toque."""
    model = load_model(f"{variant}_{GENS}_ood_s{seed}.pt", variant,
                       GENS, device, dtype)
    tr2 = collect_trajectories(model, mk_ds(seed + 500_000, mw=12), 2560,
                               device, dtype)
    heads2 = train_probes(tr2["feat_ans"], tr2["ans"], 120, device)
    ood = collect_trajectories(model, mk_ds(seed + 600_000), 768, device,
                               dtype, k_min=14)
    fid_full = per_sample_fidelity(heads2, ood["feat_ans"], ood["ans"], device)
    fid_onp = onpolicy_fidelity(heads2, ood["feat_ans"], ood["ans"],
                                ood["n_exp"], device)
    out = {"M3": {"corr_full": pearson(fid_full, ood["correct"].float()),
                  "corr_onpolicy": pearson(fid_onp, ood["correct"].float()),
                  "ood_acc": float(ood["correct"].float().mean())}}
    del model
    torch.cuda.empty_cache()
    return out


def run_battery(variant, seed, device, dtype):
    """Calcada de f3a_run.run_mhbp_battery, parametrizada por variante."""
    out = {}
    model = load_model(f"{variant}_{GENS}_indist_s{seed}.pt", variant,
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
        "n75": n75,
    }
    out["Iprime"] = {"iprime": auc(te["lam_best"], te["correct"]),
                     "acc": out_acc}
    out["M2"] = swap_experiment(model, mk_ds(seed + 400_000, mw=20, mn=8),
                                device, dtype)
    del model
    torch.cuda.empty_cache()

    model = load_model(f"{variant}_{GENS}_ood_s{seed}.pt", variant,
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


def paired_t_onesided(d):
    """t pareada unilateral (H1: media > 0) → (t, p, dz)."""
    import statistics
    n = len(d)
    m = sum(d) / n
    sd = statistics.stdev(d)
    if sd == 0:
        return float("inf"), 0.0, float("inf")
    t = m / (sd / math.sqrt(n))
    # p unilateral por t-CDF (aprox. por integración simple de la densidad t)
    from math import lgamma, pi as MPI
    df = n - 1

    def tpdf(x):
        return math.exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / \
            math.sqrt(df * MPI) * (1 + x * x / df) ** (-(df + 1) / 2)
    # integra la cola [t, 40] con Simpson
    lo, hi, N = t, 40.0, 4000
    if lo >= hi:
        return t, 0.0, m / sd
    h = (hi - lo) / N
    s = tpdf(lo) + tpdf(hi)
    for i in range(1, N):
        s += tpdf(lo + i * h) * (4 if i % 2 else 2)
    p = s * h / 3
    return t, max(0.0, min(1.0, p)), m / sd


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for variant in VARIANTS:
        for seed in SEEDS:
            cid = f"f3ar_mip_{variant}_s{seed}"
            path = os.path.join(RES, f"{cid}.json")
            if os.path.exists(path):
                print(f"skip {cid}", flush=True)
                continue
            t0 = time.time()
            r = run_battery(variant, seed, device, dtype)
            r["wall_s"] = round(time.time() - t0, 1)
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(r, f)
            os.replace(path + ".tmp", path)
            print(f"OK {cid}: trend_onp={r['M1']['trend_onpolicy']:+.2f} "
                  f"I'={r['Iprime']['iprime']:.3f} "
                  f"M3onp={r['M3']['corr_onpolicy']:+.3f} "
                  f"({r['wall_s']}s)", flush=True)
    for variant in M3_ONLY:
        for seed in SEEDS:
            cid = f"f3ar_mip_{variant}_s{seed}"
            path = os.path.join(RES, f"{cid}.json")
            ck = os.path.join(HERE, "ckpts",
                              f"{variant}_{GENS}_ood_s{seed}.pt")
            if os.path.exists(path):
                print(f"skip {cid}", flush=True)
                continue
            if not os.path.exists(ck):
                print(f"pend {cid} (sin ckpt aún)", flush=True)
                continue
            t0 = time.time()
            r = run_m3_only(variant, seed, device, dtype)
            r["wall_s"] = round(time.time() - t0, 1)
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(r, f)
            os.replace(path + ".tmp", path)
            print(f"OK {cid}: M3onp={r['M3']['corr_onpolicy']:+.3f} "
                  f"({r['wall_s']}s)", flush=True)

    # ---------------- veredicto R1/R2/R3 (PREREG_F3A_R) ---------------- #
    def m3onp(pattern):
        vals = []
        for s in SEEDS:
            with open(os.path.join(RES, pattern.format(s=s)),
                      encoding="utf-8") as f:
                vals.append(json.load(f)["M3"]["corr_onpolicy"])
        return vals

    pt = m3onp("f3a_mip_miura_mhbp_s{s}.json")        # por-tick (F3a)
    pi1 = m3onp("f3ar_mip_miura_mhbp_pi1_s{s}.json")
    noc = m3onp("f3ar_mip_miura_mhbp_noc_s{s}.json")
    with open(os.path.join(RES, "f3a_m3ref_hbp_full.json"),
              encoding="utf-8") as f:
        ref = json.load(f)
    ref_vals = [ref[f"hbp_full_s{s}"]["corr_onpolicy"] for s in SEEDS]
    ref_mean = sum(ref_vals) / len(ref_vals)

    mean = lambda v: sum(v) / len(v)
    d_pi1 = [a - b for a, b in zip(pi1, pt)]
    t1, p1, dz1 = paired_t_onesided(d_pi1)
    verdict = {
        "por_tick_F3a": {"per_seed": pt, "mean": mean(pt)},
        "pi1": {"per_seed": pi1, "mean": mean(pi1)},
        "noc": {"per_seed": noc, "mean": mean(noc)},
        "incumbente_ref": ref_mean,
        "R1_recuperacion": {"diffs": d_pi1, "t": t1, "p_unilateral": p1,
                            "dz": dz1, "pasa": bool(p1 < 0.05 and mean(d_pi1) > 0)},
        "R2_no_inferioridad": {"margen": ref_mean - 0.04,
                               "pasa": bool(mean(pi1) >= ref_mean - 0.04)},
        "noc_diff_vs_por_tick": mean(noc) - mean(pt),
    }
    # --- robustez pre-declarada (control por-tick FRESCO, si existe) --- #
    if all(os.path.exists(os.path.join(
            RES, f"f3ar_mip_miura_mhbp_ptR_s{s}.json")) for s in SEEDS):
        ptR = m3onp("f3ar_mip_miura_mhbp_ptR_s{s}.json")
        dR = [a - b for a, b in zip(pi1, ptR)]
        tR, pR, dzR = paired_t_onesided(dR)
        dRn = [a - b for a, b in zip(noc, ptR)]
        tRn, pRn, dzRn = paired_t_onesided(dRn)
        verdict["control_fresco_ptR"] = {"per_seed": ptR, "mean": mean(ptR)}
        verdict["R1_vs_control_fresco"] = {
            "pi1": {"diffs": dR, "t": tR, "p_unilateral": pR, "dz": dzR,
                    "pasa": bool(pR < 0.05 and mean(dR) > 0)},
            "noc": {"diffs": dRn, "t": tRn, "p_unilateral": pRn, "dz": dzRn}}
        print(f"\n  [robustez] control fresco ptR: {mean(ptR):+.3f}  "
              f"{['%+.2f' % v for v in ptR]}")
        print(f"  [robustez] R1' pi1>ptR: Δ={mean(dR):+.3f} t={tR:.2f} "
              f"p={pR:.4f} dz={dzR:.2f}")
        print(f"  [robustez] noc>ptR: Δ={mean(dRn):+.3f} p={pRn:.4f}")

        # --- ENMIENDA 1: toque re-estimado con incumbente FRESCO --- #
        if all(os.path.exists(os.path.join(
                RES, f"f3ar_mip_hbp_fullR_s{s}.json")) for s in SEEDS):
            import statistics
            hbR = m3onp("f3ar_mip_hbp_fullR_s{s}.json")
            dT = [a - b for a, b in zip(ptR, hbR)]     # toque contemporáneo
            mT = mean(dT)
            sdT = statistics.stdev(dT)
            half = 2.571 * sdT / math.sqrt(6)          # t(5, 97.5%)
            ic = (mT - half, mT + half)
            replica = ic[1] < 0 and abs(mT) >= 0.03
            no_replica = ic[0] <= 0.0 <= ic[1]
            verdict["toque_contemporaneo"] = {
                "hbp_fullR": {"per_seed": hbR, "mean": mean(hbR)},
                "diffs_ptR_menos_hbpR": dT, "mean": mT, "ic95": ic,
                "rama": ("REPLICA" if replica else
                         "NO_REPLICADO" if no_replica else "AMBIGUO")}
            print(f"\n  [enmienda 1] incumbente fresco hbp_fullR: "
                  f"{mean(hbR):+.3f}  {['%+.2f' % v for v in hbR]}")
            print(f"  [enmienda 1] TOQUE contemporáneo (ptR−hbpR): "
                  f"Δ={mT:+.3f} IC95=[{ic[0]:+.3f}, {ic[1]:+.3f}] → "
                  f"{verdict['toque_contemporaneo']['rama']}")
    with open(os.path.join(RES, "f3ar_veredicto.json"), "w",
              encoding="utf-8") as f:
        json.dump(verdict, f, indent=1)
    print("\n=== F3a-R veredicto (M3 on-policy) ===")
    print(f"  por-tick (F3a): {mean(pt):+.3f}  {['%+.2f' % v for v in pt]}")
    print(f"  pi1           : {mean(pi1):+.3f}  {['%+.2f' % v for v in pi1]}")
    print(f"  noc           : {mean(noc):+.3f}  {['%+.2f' % v for v in noc]}")
    print(f"  incumbente ref: {ref_mean:+.3f}")
    print(f"  R1 pi1>por-tick: Δ={mean(d_pi1):+.3f} t={t1:.2f} "
          f"p={p1:.4f} dz={dz1:.2f} → {'PASA' if verdict['R1_recuperacion']['pasa'] else 'NO'}")
    print(f"  R2 pi1 ≥ ref−0.04={ref_mean - 0.04:.3f} → "
          f"{'PASA' if verdict['R2_no_inferioridad']['pasa'] else 'NO'}")
    print("Guardado f3ar_veredicto.json")


if __name__ == "__main__":
    main()
