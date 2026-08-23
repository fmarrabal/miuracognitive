"""Benchmark Fase 2 v2 (compromiso endógeno; reglas AGENCY_V2.md).

Etapas con seeds disjuntas:
  tune    100-102: grid de scripted {greedy(w), hysteresis(w,m), smart(w,m)}.
  pilot   200-202: learned vs scripted afinados vs ablación memoryless.
          Gates: G-untrained, G-ceiling, aprendibilidad.
  confirm 300-319: P1 = retorno learned − mejor scripted (in-dist);
          P2 = ídem en OOD (distractores ×2, crisis +1; misma regla de
          desambiguación). Sign test + Holm sobre {P1,P2}; G-LOSO.

Uso: $env:PYTHONPATH="." ; python goals_v2_benchmark.py --stage tune|pilot|confirm
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import torch

from data.goals_v2 import Goals2Config, Goals2Dataset
from model.goals_v2 import (LearnedGoalConfig, LearnedGoalPolicy,
                            ScriptedGoalPolicy, eval_goal_policy,
                            train_learned_goals)

OUT = "results_goals_v2"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EVAL_BATCH = 512
TRAIN_STEPS = 400

TUNE_SEEDS = [100, 101, 102]
PILOT_SEEDS = [200, 201, 202]
CONFIRM_SEEDS = list(range(300, 320))


def eval_arm(cfg, policy, seed) -> dict:
    ds = Goals2Dataset(cfg, seed=seed + 50_000)
    sc = ds.batch(EVAL_BATCH).to(DEV)
    return eval_goal_policy(cfg, sc, policy)


def tune_scripted(cfg: Goals2Config) -> dict:
    grids = {
        "greedy": {"w": [0.0, 0.25, 0.5, 0.75, 1.0]},
        "hysteresis": {"w": [0.25, 0.5, 0.75],
                       "margin": [0.05, 0.1, 0.2, 0.4]},
        "smart": {"w": [0.25, 0.5, 0.75],
                  "margin": [0.05, 0.1, 0.2, 0.4]},
    }
    out = {}
    for name, grid in grids.items():
        keys = list(grid)
        best, log = None, []
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            rs = [eval_arm(cfg, ScriptedGoalPolicy(cfg, name, **params), s)["return"]
                  for s in TUNE_SEEDS]
            m = sum(rs) / len(rs)
            log.append({"params": params, "return": m})
            if best is None or m > best["return"]:
                best = {"params": params, "return": m}
        out[name] = {"best": best, "grid": log}
        print(f"[tune] {name:10s} -> {best['params']} ret={best['return']:.3f}",
              flush=True)
    return out


def run_learned(cfg, arm: str, seed: int, steps: int = TRAIN_STEPS) -> dict:
    torch.manual_seed(seed)
    pol = LearnedGoalPolicy(cfg, LearnedGoalConfig(
        memoryless=(arm == "learned_memoryless")))
    pol = train_learned_goals(cfg, pol, seed=seed, steps=steps, device=DEV)
    return eval_arm(cfg, pol, seed)


def scripted_from(tuned, cfg):
    return {name: ScriptedGoalPolicy(cfg, name, **tuned[name]["best"]["params"])
            for name in ("greedy", "hysteresis", "smart")}


def sign_test_p(diffs) -> float:
    n = len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    total = sum(math.comb(n, k) for k in range(min(pos, n - pos) + 1))
    return min(1.0, 2.0 * total / 2 ** n)


def stage_tune(cfg):
    tuned = tune_scripted(cfg)
    with open(f"{OUT}/tuned_scripted.json", "w") as f:
        json.dump(tuned, f, indent=2)


def stage_pilot(cfg):
    tuned = json.load(open(f"{OUT}/tuned_scripted.json"))
    rows = []
    for s in PILOT_SEEDS:
        for name, pol in scripted_from(tuned, cfg).items():
            rows.append({"arm": name, "seed": s, **eval_arm(cfg, pol, s)})
        for arm in ("learned", "learned_memoryless"):
            r = run_learned(cfg, arm, s)
            rows.append({"arm": arm, "seed": s, **r})
            print(f"[pilot] s={s} {arm:20s} ret={r['return']:.3f} "
                  f"crisis={r['crisis_attended']:.2f} "
                  f"distr={r['distractor_attended']:.2f}", flush=True)
        if s == PILOT_SEEDS[0]:
            r0 = run_learned(cfg, "learned", s, steps=0)
            rows.append({"arm": "learned_UNTRAINED", "seed": s, **r0})
    with open(f"{OUT}/pilot.json", "w") as f:
        json.dump(rows, f, indent=2)
    import statistics
    print("\n===== PILOTO FASE 2 v2 =====")
    for arm in sorted({r["arm"] for r in rows}):
        xs = [r["return"] for r in rows if r["arm"] == arm]
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(f"  {arm:20s} ret={sum(xs)/len(xs):+.3f}±{sd:.3f} (n={len(xs)})")


def stage_confirm(cfg):
    tuned = json.load(open(f"{OUT}/tuned_scripted.json"))
    best_name = max(("greedy", "hysteresis", "smart"),
                    key=lambda n: tuned[n]["best"]["return"])
    print(f"[confirm] mejor scripted (por tuning): {best_name}")
    cfg_ood = cfg.ood_variant()
    rows = []
    t0 = time.time()
    for s in CONFIRM_SEEDS:
        pol = None
        torch.manual_seed(s)
        pol_learned = LearnedGoalPolicy(cfg, LearnedGoalConfig())
        pol_learned = train_learned_goals(cfg, pol_learned, seed=s,
                                          steps=TRAIN_STEPS, device=DEV)
        for tag, c in (("indist", cfg), ("ood", cfg_ood)):
            sc_pol = ScriptedGoalPolicy(c, best_name,
                                        **tuned[best_name]["best"]["params"])
            r_s = eval_arm(c, sc_pol, s)
            r_l = eval_arm(c, pol_learned, s)
            rows.append({"arm": f"scripted_{tag}", "seed": s, **r_s})
            rows.append({"arm": f"learned_{tag}", "seed": s, **r_l})
        r_m = run_learned(cfg, "learned_memoryless", s)
        rows.append({"arm": "memoryless_indist", "seed": s, **r_m})
        print(f"[confirm] seed {s} ({(time.time()-t0)/60:.0f} min)", flush=True)
    with open(f"{OUT}/confirm.json", "w") as f:
        json.dump(rows, f, indent=2)
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r["return"]
    p1 = [by["learned_indist"][s] - by["scripted_indist"][s] for s in CONFIRM_SEEDS]
    p2 = [by["learned_ood"][s] - by["scripted_ood"][s] for s in CONFIRM_SEEDS]
    print("\n===== CONFIRMATORIO FASE 2 v2 =====")
    for name, d in (("P1 in-dist", p1), ("P2 OOD", p2)):
        pos = sum(1 for x in d if x > 0)
        print(f"{name}: media={sum(d)/len(d):+.4f} seeds+={pos}/{len(d)} "
              f"p_sign={sign_test_p(d):.2e}")
    d3 = [by["learned_indist"][s] - by["memoryless_indist"][s] for s in CONFIRM_SEEDS]
    print(f"secundario learned-memoryless: media={sum(d3)/len(d3):+.4f} "
          f"p_sign={sign_test_p(d3):.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "pilot", "confirm"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = Goals2Config()
    print(f"Fase 2 v2 [{args.stage}] en {DEV}", flush=True)
    {"tune": stage_tune, "pilot": stage_pilot,
     "confirm": stage_confirm}[args.stage](cfg)
