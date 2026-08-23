import json
print("ITERFUNC (train K<=12, eval K<=24), seed0:")
print("  corto/medio = in-dist ; LARGO = extrapolacion. chance=1/8=0.125")
print(f"  {'variante':10s} {'corto':>6s} {'medio':>6s} {'LARGO':>6s} {'nit_co':>7s} {'nit_la':>7s} {'corr_OOD':>9s}")
A = {}
for v in ["vanilla", "gating_wm", "hbp_full"]:
    r = json.load(open(f"results_iterfunc/{v}_seed0.json", encoding="utf-8"))
    a = r["final_acc"]; d = r.get("compute_diag"); A[v] = (a, d)
    if d:
        print(f"  {v:10s} {a['corto']:>6.3f} {a['medio']:>6.3f} {a['largo']:>6.3f} "
              f"{d['corto']['mean_niter']:>7.2f} {d['largo']['mean_niter']:>7.2f} {d['corr_K_niter']:>9.3f}")
    else:
        print(f"  {v:10s} {a['corto']:>6.3f} {a['medio']:>6.3f} {a['largo']:>6.3f} {'-':>7s} {'-':>7s} {'-':>9s}")
print()
print("Checklist de regimen valido:")
print(f"  aprendible (vanilla corto alto)? {A['vanilla'][0]['corto']:.3f}")
print(f"  vanilla se degrada con K? corto {A['vanilla'][0]['corto']:.3f} -> largo {A['vanilla'][0]['largo']:.3f}")
print(f"  reasoner itera? gating_wm nit {A['gating_wm'][1]['mean_niter']:.2f} | hbp_full nit {A['hbp_full'][1]['mean_niter']:.2f}")
print(f"  senal OOD direccional? hbp_full corr {A['hbp_full'][1]['corr_K_niter']:.3f} vs gating_wm {A['gating_wm'][1]['corr_K_niter']:.3f}")
