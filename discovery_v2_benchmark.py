"""Benchmark Fase 3 v2 (descubrimiento sin ley conocida; AGENCY_V2.md).

  tune    100-102: grid de scripted (FD: k_init×step; grid: jitter).
  pilot   200-202: prober aprendido vs scripted afinados + G-untrained.
  confirm 300-319: P1 = regret aprendido − mejor scripted (sign test);
                   secundarios por familia de campo; G-LOSO.

Uso: $env:PYTHONPATH="." ; python discovery_v2_benchmark.py --stage tune|pilot|confirm
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import torch

from data.discovery_v2 import Discovery2Config
from model.discovery_v2 import (FDAscent, GridProbe, LearnedProber, RandomProbe,
                                eval_prober, train_prober)

OUT = "results_discovery_v2"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_STEPS = 800
TUNE_SEEDS = [100, 101, 102]
PILOT_SEEDS = [200, 201, 202]
CONFIRM_SEEDS = list(range(300, 320))


def sign_test_p(diffs):
    n = len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    total = sum(math.comb(n, k) for k in range(min(pos, n - pos) + 1))
    return min(1.0, 2.0 * total / 2 ** n)


def tune(cfg):
    out = {}
    grids = {
        "fd": [{"k_init": k, "step": s}
               for k, s in itertools.product([3, 4, 6], [0.05, 0.1, 0.2, 0.4])],
        "grid": [{"jitter": j} for j in [0.0, 0.05, 0.1]],
    }
    build = {"fd": lambda p: FDAscent(cfg, **p),
             "grid": lambda p: GridProbe(cfg, **p)}
    for name, combos in grids.items():
        best, log = None, []
        for params in combos:
            rs = [eval_prober(cfg, build[name](params), s, device=DEV)["regret"]
                  for s in TUNE_SEEDS]
            m = sum(rs) / len(rs)
            log.append({"params": params, "regret": m})
            if best is None or m < best["regret"]:
                best = {"params": params, "regret": m}
        out[name] = {"best": best, "grid": log}
        print(f"[tune] {name:6s} -> {best['params']} regret={best['regret']:.3f}",
              flush=True)
    with open(f"{OUT}/tuned_scripted.json", "w") as f:
        json.dump(out, f, indent=2)


def scripted_from(tuned, cfg):
    return {
        "random": RandomProbe(cfg),
        "grid": GridProbe(cfg, **tuned["grid"]["best"]["params"]),
        "fd": FDAscent(cfg, **tuned["fd"]["best"]["params"]),
    }


def run_learned(cfg, seed, steps=TRAIN_STEPS):
    torch.manual_seed(seed)
    pro = LearnedProber(cfg)
    if steps > 0:
        train_prober(cfg, pro, seed=seed, steps=steps, device=DEV)
    return eval_prober(cfg, pro.to(DEV), seed, device=DEV)


def stage_pilot(cfg):
    tuned = json.load(open(f"{OUT}/tuned_scripted.json"))
    rows = []
    for s in PILOT_SEEDS:
        for name, agent in scripted_from(tuned, cfg).items():
            rows.append({"arm": name, "seed": s,
                         **eval_prober(cfg, agent, s, device=DEV)})
        r = run_learned(cfg, s)
        rows.append({"arm": "learned", "seed": s, **r})
        print(f"[pilot] s={s} learned regret={r['regret']:.3f}", flush=True)
        if s == PILOT_SEEDS[0]:
            rows.append({"arm": "learned_UNTRAINED", "seed": s,
                         **run_learned(cfg, s, steps=0)})
    with open(f"{OUT}/pilot.json", "w") as f:
        json.dump(rows, f, indent=2)
    import statistics
    print("\n===== PILOTO FASE 3 v2 =====")
    for arm in sorted({r["arm"] for r in rows}):
        xs = [r["regret"] for r in rows if r["arm"] == arm]
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(f"  {arm:18s} regret={sum(xs)/len(xs):.3f}±{sd:.3f} (n={len(xs)})")


def stage_confirm(cfg):
    tuned = json.load(open(f"{OUT}/tuned_scripted.json"))
    best_name = min(("random", "grid", "fd"),
                    key=lambda n: (tuned.get(n, {}).get("best", {}).get("regret", 0.271)
                                   if n != "random" else 0.271))
    rows = []
    t0 = time.time()
    for s in CONFIRM_SEEDS:
        for name, agent in scripted_from(tuned, cfg).items():
            rows.append({"arm": name, "seed": s,
                         **eval_prober(cfg, agent, s, device=DEV)})
        rows.append({"arm": "learned", "seed": s, **run_learned(cfg, s)})
        print(f"[confirm] seed {s} ({(time.time()-t0)/60:.0f} min)", flush=True)
    with open(f"{OUT}/confirm.json", "w") as f:
        json.dump(rows, f, indent=2)
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r["regret"]
    best_scripted = min(("random", "grid", "fd"),
                        key=lambda n: sum(by[n].values()))
    d = [by[best_scripted][s] - by["learned"][s] for s in CONFIRM_SEEDS]
    pos = sum(1 for x in d if x > 0)
    print(f"\n===== CONFIRMATORIO FASE 3 v2 =====")
    print(f"mejor scripted: {best_scripted} "
          f"(regret={sum(by[best_scripted].values())/20:.3f})")
    print(f"P1 (regret scripted − learned): media={sum(d)/len(d):+.4f} "
          f"seeds+={pos}/{len(d)} p_sign={sign_test_p(d):.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "pilot", "confirm"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = Discovery2Config()
    print(f"Fase 3 v2 [{args.stage}] en {DEV}", flush=True)
    {"tune": tune, "pilot": stage_pilot, "confirm": stage_confirm}[args.stage](cfg)
