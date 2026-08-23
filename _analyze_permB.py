import json, glob
import numpy as np

def grab(d, v, key):
    xs = []
    for p in sorted(glob.glob(f"{d}/{v}_seed*.json")):
        r = json.load(open(p, encoding="utf-8"))
        xs.append(r["final_acc"]["largo"] if key == "acc" else r["compute_diag"]["corr_K_niter"])
    return np.array(xs)

def report(tag, d):
    print(f"=== {tag} ===")
    print(f"{'variante':10s} {'n':>3s} {'LARGO_acc':>13s} {'corr(K,nit)_OOD':>18s}")
    cr = {}
    for v in ["gating_wm", "hbp_first", "hbp_full"]:
        acc = grab(d, v, "acc"); c = grab(d, v, "corr"); cr[v] = c
        sem = c.std(ddof=1)/np.sqrt(len(c))
        print(f"{v:10s} {len(c):>3d} {acc.mean():.3f}+-{acc.std():.3f}  {c.mean():.3f}+-{c.std():.3f} (SEM {sem:.3f})")
    g = cr["gating_wm"]
    for v in ["hbp_first", "hbp_full"]:
        h = cr[v]; diff = h.mean()-g.mean()
        semd = np.sqrt(h.std(ddof=1)**2/len(h) + g.std(ddof=1)**2/len(g))
        print(f"  {v} - gating_wm: delta={diff:+.3f}  t={diff/semd:.2f}")
    print()

report("GENERALIDAD: permcomp cycle_transp (5-ciclo+transp), n=10", "results_permB")
report("PRIMARIO: permcomp adjacent, n=10 (referencia)", "results_extrap")
