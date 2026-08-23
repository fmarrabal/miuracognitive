"""A/B v2 (física congelada por bug BF16) vs v3 (física aprendible, pin_fp32):
mismas seeds, mismo protocolo. ¿El fix cambió el efecto? ¿Los parámetros físicos
se movieron de verdad en v3?"""
import json, glob, os, math
import numpy as np

def load(dirname, regime, gens, variant):
    out = {}
    for p in sorted(glob.glob(os.path.join(dirname, f"{regime}_{gens}_{variant}_seed*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        out[r["_cell"]["seed"]] = r
    return out

def corr(r):
    cd = r.get("compute_diag") or {}
    return cd.get("corr_K_niter_ood")

print("=" * 78)
print("A/B por seed: corrOOD(K>=14) de hbp_full — v2 (congelada) vs v3 (aprendible)")
print("=" * 78)
for gens in ["adjacent", "cycle_transp"]:
    v2 = load("results_benchmark_v2", "ood", gens, "hbp_full")
    v3 = load("results_benchmark_v3", "ood", gens, "hbp_full")
    seeds = sorted(set(v2) & set(v3))
    d = [corr(v3[s]) - corr(v2[s]) for s in seeds]
    m2 = np.mean([corr(v2[s]) for s in seeds]); m3 = np.mean([corr(v3[s]) for s in seeds])
    dd = np.array(d)
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else float("nan")
    print(f"  {gens:13s}: v2={m2:+.3f}  v3={m3:+.3f}  Δ(v3-v2)={dd.mean():+.3f}±{dd.std(ddof=1):.3f} "
          f"t={t:+.2f} (n={len(dd)})")

print("\n¿Se movieron los parámetros físicos en v3? (hbp_diag de cada run)")
for gens in ["adjacent"]:
    for variant in ["hbp_first", "hbp_full"]:
        for ver, dirn in [("v2", "results_benchmark_v2"), ("v3", "results_benchmark_v3")]:
            runs = load(dirn, "ood", gens, variant)
            zs = []
            for s, r in runs.items():
                hd = r.get("hbp_diag") or {}
                dampd = hd.get("damping") or {}
                z = dampd.get("zeta_mean")
                if z is not None:
                    zs.append(z)
            if zs:
                init = 2.5 if variant == "hbp_first" else 0.5
                print(f"  {variant:10s} {ver}: ζ_mean={np.mean(zs):.4f}±{np.std(zs):.4f} "
                      f"(init={init}; movido={abs(np.mean(zs)-init)>1e-3}) n={len(zs)}")

# Fisher combinado de los 2 contrastes headline de v3 (con la cautela de seeds compartidas)
print("\nFisher combinado v3 (2 contrastes hbp_full-gwm; seeds compartidas entre gens -> orientativo):")
ps = [0.0253, 0.0277]
chi2 = -2 * sum(math.log(p) for p in ps)
# p-value chi2 df=4
from math import exp
# supervivencia chi2 df=4: (1 + x/2) * exp(-x/2)
x = chi2
p_comb = (1 + x / 2) * exp(-x / 2)
print(f"  chi2={chi2:.2f} (df=4) -> p_comb≈{p_comb:.4f}")
