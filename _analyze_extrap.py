import json, glob
import numpy as np

def stats(v, key):
    xs = []
    for p in sorted(glob.glob(f"results_extrap/{v}_seed*.json")):
        r = json.load(open(p, encoding="utf-8"))
        if key == "largo_acc":
            xs.append(r["final_acc"]["largo"])
        else:
            xs.append(r["compute_diag"][key])
    return np.array(xs)

print("=== EXTRAPOLACION (train K<=12, eval K<=24) — poder estadistico ===")
print(f"{'variante':10s} {'n':>3s} {'LARGO_acc':>14s} {'corr(K,nit)_OOD':>18s}")
data = {}
for v in ["gating_wm", "hbp_first", "hbp_full"]:
    acc = stats(v, "largo_acc")
    cr = stats(v, "corr_K_niter")
    data[v] = cr
    sem = cr.std(ddof=1) / np.sqrt(len(cr)) if len(cr) > 1 else float("nan")
    print(f"{v:10s} {len(cr):>3d} {acc.mean():.3f}+-{acc.std():.3f}   "
          f"{cr.mean():.3f}+-{cr.std():.3f} (SEM {sem:.3f})")

print()
g = data["gating_wm"]
for v in ["hbp_first", "hbp_full"]:
    h = data[v]
    n = min(len(g), len(h))
    diff = h.mean() - g.mean()
    sem_diff = np.sqrt(h.std(ddof=1)**2/len(h) + g.std(ddof=1)**2/len(g))
    t = diff / sem_diff if sem_diff > 0 else float("nan")
    print(f"{v} - gating_wm (corr OOD): delta={diff:+.3f}  SEM_diff={sem_diff:.3f}  t={t:.2f}  (n_g={len(g)}, n_h={len(h)})")
print()
print("Interpretacion: t>2 ~ significativo (la adaptividad OOD del HBP supera al baseline WM).")
