import json
print("permcomp cycle_transp (train K<=12, eval K<=24), seed0:   chance=1/120=0.008")
print(f"  {'variante':10s} {'corto':>6s} {'medio':>6s} {'LARGO':>6s} {'nit_co':>7s} {'nit_la':>7s} {'corr_OOD':>9s}")
A = {}
for v in ["vanilla", "gating_wm", "hbp_full"]:
    r = json.load(open(f"results_permB_probe/{v}_seed0.json", encoding="utf-8"))
    a = r["final_acc"]; d = r.get("compute_diag"); A[v] = (a, d)
    if d:
        print(f"  {v:10s} {a['corto']:>6.3f} {a['medio']:>6.3f} {a['largo']:>6.3f} "
              f"{d['corto']['mean_niter']:>7.2f} {d['largo']['mean_niter']:>7.2f} {d['corr_K_niter']:>9.3f}")
    else:
        print(f"  {v:10s} {a['corto']:>6.3f} {a['medio']:>6.3f} {a['largo']:>6.3f} {'-':>7s} {'-':>7s} {'-':>9s}")
print()
print("Regimen valido si: corto alto (aprendible), vanilla cae con K, reasoner itera (nit>>1),")
print("y la corr_OOD del HBP supera a la de gating_wm (direccional).")
