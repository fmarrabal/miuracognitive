import json, glob
import numpy as np

def curve(regime, gens, variant):
    runs = [json.load(open(p, encoding="utf-8")) for p in
            sorted(glob.glob(f"results_benchmark_v2/{regime}_{gens}_{variant}_seed*.json"))]
    agg = {}
    for r in runs:
        for k, d in (r.get("compute_diag") or {}).get("niter_by_K", {}).items():
            agg.setdefault(int(k), []).append(d["niter"])
    return {k: np.mean(v) for k, v in sorted(agg.items())}, len(runs)

for regime in ["indist", "ood"]:
    print(f"\n=== E[n_iter | K] — adjacent | {regime} (N_max=24) ===")
    for v in ["gating_wm", "hbp_first", "hbp_full"]:
        c, n = curve(regime, "adjacent", v)
        if not c: continue
        ks = sorted(c)
        # muestra K = 2,4,8,12,16,20,24
        show = [k for k in [2, 4, 8, 12, 16, 20, 24] if k in c]
        line = "  ".join(f"K{k}={c[k]:.1f}" for k in show)
        print(f"  {v:10s}(n{n}): {line}")
