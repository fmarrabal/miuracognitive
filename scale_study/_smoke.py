"""Smoke test: el plan no tiene IDs duplicados y cada tipo de celda corre sin error."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scale_study import run_night as R

cells = R.plan(full=True)
ids = [c[0] for c in cells]
dup = len(ids) - len(set(ids))
print(f"PLAN: {len(cells)} celdas, duplicados={dup}")
from collections import Counter
blocks = Counter(cid.split("_")[0] for cid in ids)
print("por bloque:", dict(blocks))
assert dup == 0, "IDS DUPLICADOS"

# corre una celda de cada tipo con ticks reducidos
tests = [
    ("B1", R.cell_B1, {"kind": "ring", "N": 1024, "seed": 0}),
    ("B1sp", R.cell_B1, {"kind": "expander", "N": 1024, "seed": 0}),
    ("B2", R.cell_B2, {"N": 1024, "beta": 0.05, "modes": [4, 16, 64, 200], "ticks": 500}),
    ("B4", R.cell_B4, {"N": 256, "zeta": 0.1, "beta": 0.1, "placement": "gyro", "dt": 0.1, "ticks": 300}),
    ("B4c", R.cell_B4, {"N": 128, "zeta": 0.1, "beta": 0.25, "placement": "circulatory", "dt": 0.1, "ticks": 300}),
    ("B3s", R.cell_B3s, {"block": "B3s", "N": 1024, "amp": 0.6, "beta": 0.1, "nu": 0.2,
                         "nonlin": "genuine", "damp": "none", "ticks": 1500, "dt": 0.05}),
    ("B3s_sat", R.cell_B3s, {"block": "B3s", "N": 1024, "amp": 1.5, "beta": 0.1, "nu": 0.1,
                             "nonlin": "saturated", "damp": "weak", "ticks": 1500, "dt": 0.05}),
    ("B3p", R.cell_B3p, {"N": 1024, "amp": 0.8, "width": 0.02, "beta": 0.1, "nu": 0.2,
                         "nonlin": "genuine", "damp": "none", "ticks": 1500, "dt": 0.05, "sign": 1.0}),
    ("B3b", R.cell_B3b, {"N": 1024, "amp1": 0.9, "amp2": 0.35, "beta": 0.1, "nu": 0.2,
                         "nonlin": "genuine", "ticks": 1500, "dt": 0.05}),
    ("B3c", R.cell_B3c, {"N": 1024, "amp": 0.4, "beta": 0.1, "nu": 0.2, "ticks": 1500, "dt": 0.05}),
    ("B5", R.cell_B5, {"N": 10000, "ticks": 300}),
]
for name, fn, cfg in tests:
    t = time.time()
    try:
        r = fn(cfg)
        import json
        s = len(json.dumps(r))
        print(f"[OK] {name:8s} keys={len(r)} json={s}B ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"[ERR] {name:8s}: {repr(e)[:150]}")
print("SMOKE OK")
