"""A/B pareado hbp_elliptic vs hbp_full (protocolo v3, OOD): accuracy largo y
adaptividad de cómputo OOD (corr K,n_iter). ¿La no-localidad rompe el null?"""
import json, glob, os, math
import statistics as st

D = "results_benchmark_v3"


def load(gens, v):
    out = {}
    for p in sorted(glob.glob(os.path.join(D, f"ood_{gens}_{v}_seed*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        out[r["_cell"]["seed"]] = r
    return out


def corr(r):
    return (r.get("compute_diag") or {}).get("corr_K_niter_ood")


def fisher(x):
    x = max(min(x, 0.999), -0.999)
    return 0.5 * math.log((1 + x) / (1 - x))


def paired(diffs):
    n = len(diffs)
    m = sum(diffs) / n
    sd = st.stdev(diffs) if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    pos = sum(1 for d in diffs if d > 0)
    return m, sd, t, pos, n


print("=" * 78)
print("A/B hbp_elliptic (no-local) − hbp_full (local), protocolo v3 OOD")
print("=" * 78)
for gens in ["adjacent", "cycle_transp"]:
    el, fu = load(gens, "hbp_elliptic"), load(gens, "hbp_full")
    seeds = sorted(set(el) & set(fu))
    if not seeds:
        print(f"{gens}: (elíptico incompleto, {len(el)} celdas)"); continue
    # accuracy largo (OOD)
    da = [el[s]["final_acc"]["largo"] - fu[s]["final_acc"]["largo"] for s in seeds]
    m, sd, t, pos, n = paired(da)
    print(f"\n{gens} | ACCURACY largo OOD:")
    print(f"   elíptico={st.mean([el[s]['final_acc']['largo'] for s in seeds]):.3f}  "
          f"full={st.mean([fu[s]['final_acc']['largo'] for s in seeds]):.3f}  "
          f"Δ={m:+.4f}±{sd:.3f} t={t:+.2f} ({pos}/{n})")
    # corr OOD (adaptividad de cómputo) — Fisher-z pareado
    dc = [fisher(corr(el[s])) - fisher(corr(fu[s])) for s in seeds
          if corr(el[s]) is not None and corr(fu[s]) is not None]
    if dc:
        m, sd, t, pos, n = paired(dc)
        print(f"   corrOOD: elíptico={st.mean([corr(el[s]) for s in seeds]):+.3f}  "
              f"full={st.mean([corr(fu[s]) for s in seeds]):+.3f}  "
              f"Δz={m:+.3f} t={t:+.2f} ({pos}/{n})")
print("\n(H0: la no-localidad es neutra como toda física local -> Δ≈0)")
