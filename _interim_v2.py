import json, glob, os
import numpy as np

def load(regime, gens, variant):
    out = []
    for p in sorted(glob.glob(f"results_benchmark_v2/{regime}_{gens}_{variant}_seed*.json")):
        out.append(json.load(open(p, encoding="utf-8")))
    return out

def col(runs, path):
    vals = []
    for r in runs:
        d = r.get("compute_diag") or {}
        if path == "largo": vals.append(r["final_acc"]["largo"])
        elif path == "medio": vals.append(r["final_acc"]["medio"])
        elif path == "niter": vals.append(d.get("mean_niter"))
        elif path == "corr_full": vals.append(d.get("corr_K_niter"))
        elif path == "corr_ood": vals.append(d.get("corr_K_niter_ood"))
    return np.array([v for v in vals if v is not None], dtype=float)

def s(a):
    return f"{a.mean():+.3f}±{a.std(ddof=1):.3f}(n{len(a)})" if len(a) > 1 else (f"{a[0]:+.3f}(n1)" if len(a) else "  -  ")

for gens in ["adjacent", "cycle_transp"]:
    for regime in ["indist", "ood"]:
        print(f"\n=== {gens} | {regime} ===")
        for v in ["vanilla","gating","gating_wm","hbp_first","hbp_full"]:
            runs = load(regime, gens, v)
            if not runs: continue
            print(f"  {v:11s} n={len(runs)} largo={s(col(runs,'largo'))} medio={s(col(runs,'medio'))} "
                  f"niter={s(col(runs,'niter'))} corrFULL={s(col(runs,'corr_full'))} corrOOD={s(col(runs,'corr_ood'))}")
        # contraste pareado corrOOD hbp vs gwm (K>=14)
        gwm = {r["_cell"]["seed"]: r for r in load(regime, gens, "gating_wm")}
        if regime == "ood" and gwm:
            for v in ["hbp_first","hbp_full"]:
                hv = {r["_cell"]["seed"]: r for r in load(regime, gens, v)}
                common = sorted(set(gwm) & set(hv))
                diffs = []
                for k in common:
                    a = gwm[k].get("compute_diag",{}).get("corr_K_niter_ood")
                    b = hv[k].get("compute_diag",{}).get("corr_K_niter_ood")
                    if a is not None and b is not None: diffs.append(b - a)
                if len(diffs) > 1:
                    d = np.array(diffs); t = d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
                    print(f"    PAREADO corrOOD {v}-gwm: Δ={d.mean():+.3f} t={t:.2f} (n={len(d)})")
