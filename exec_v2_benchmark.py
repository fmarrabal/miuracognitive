"""Benchmark F2X — campo ejecutivo (triaje + plazos + energía; AGENCY_V2).

  tune    100-102: grid de scripted {value_density(e_min), edf(e_min),
          designer(e_min, margin)}.
  pilot   200-202: learned vs scripted afinados + memoryless + G-untrained
          (+ HBP exploratorio). Gates: aprendibilidad, no-techo, colapsos sanos.
  confirm 300-319: P1 = retorno learned − mejor scripted (in-dist);
          P2 = ídem OOD-interrupciones; P3 = ídem OOD-energía. Holm sobre
          {P1,P2,P3}. Secundarios: memoryless, HBP, regret vs skyline.

Uso: $env:PYTHONPATH="." ; python exec_v2_benchmark.py --stage tune|pilot|confirm
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import torch

from data.exec_v2 import ExecV2Config, ExecV2Dataset
from model.exec_v2 import (DesignerPolicy, EDFPolicy, LearnedExecConfig,
                           LearnedExecPolicy, ValueDensityPolicy,
                           eval_exec_policy, train_learned_exec)

OUT = "results_exec_v2"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
EVAL_BATCH = 512
# Receta congelada: A2C+curriculum+reward-to-go. Sanity: 2500->3.30, 4000->3.41
# (gap con designer 3.72 casi idéntico). 2500 = ahorro 40% sin cambiar la historia.
TRAIN_STEPS = 2500
TUNE_SEEDS = [100, 101, 102]
PILOT_SEEDS = [200, 201, 202]
CONFIRM_SEEDS = list(range(300, 320))


def sign_test_p(diffs):
    n = len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    total = sum(math.comb(n, k) for k in range(min(pos, n - pos) + 1))
    return min(1.0, 2.0 * total / 2 ** n)


def eval_arm(cfg, policy, seed):
    ds = ExecV2Dataset(cfg, seed=seed + 50_000)
    sc = ds.batch(EVAL_BATCH).to(DEV)
    return eval_exec_policy(cfg, sc, policy, device=DEV)


def tune(cfg):
    grids = {
        "value_density": [{"e_min": e} for e in (0.1, 0.2, 0.3, 0.4)],
        "edf": [{"e_min": e} for e in (0.1, 0.2, 0.3, 0.4)],
        "designer": [{"e_min": e, "margin": m}
                     for e, m in itertools.product((0.1, 0.2, 0.3),
                                                   (0.02, 0.05, 0.1))],
    }
    build = {"value_density": lambda p: ValueDensityPolicy(cfg, **p),
             "edf": lambda p: EDFPolicy(cfg, **p),
             "designer": lambda p: DesignerPolicy(cfg, **p)}
    out = {}
    for name, combos in grids.items():
        best, log = None, []
        for params in combos:
            rs = [eval_arm(cfg, build[name](params), s)["return"]
                  for s in TUNE_SEEDS]
            m = sum(rs) / len(rs)
            log.append({"params": params, "return": m})
            if best is None or m > best["return"]:
                best = {"params": params, "return": m}
        out[name] = {"best": best, "grid": log}
        print(f"[tune] {name:14s} -> {best['params']} ret={best['return']:.3f}",
              flush=True)
    with open(f"{OUT}/tuned_scripted.json", "w") as f:
        json.dump(out, f, indent=2)


def scripted_from(tuned, cfg):
    return {
        "value_density": ValueDensityPolicy(cfg, **tuned["value_density"]["best"]["params"]),
        "edf": EDFPolicy(cfg, **tuned["edf"]["best"]["params"]),
        "designer": DesignerPolicy(cfg, **tuned["designer"]["best"]["params"]),
    }


ARMS = {
    "learned": LearnedExecConfig(),
    "learned_memoryless": LearnedExecConfig(memoryless=True),
    "hbp_wave": LearnedExecConfig(use_hbp=True, hbp_alpha_const=1.0),
    "hbp_diff": LearnedExecConfig(use_hbp=True, hbp_alpha_const=0.0),
}


def run_learned(cfg, arm, seed, steps=TRAIN_STEPS, eval_cfg=None):
    torch.manual_seed(seed)
    pol = LearnedExecPolicy(cfg, ARMS[arm])
    if steps > 0:
        pol = train_learned_exec(cfg, pol, seed=seed, steps=steps, device=DEV)
    else:
        pol = pol.to(DEV)
    return eval_arm(eval_cfg or cfg, pol, seed)


def stage_pilot(cfg):
    tuned = json.load(open(f"{OUT}/tuned_scripted.json"))
    rows = []
    for s in PILOT_SEEDS:
        for name, pol in scripted_from(tuned, cfg).items():
            rows.append({"arm": name, "seed": s, **eval_arm(cfg, pol, s)})
        for arm in ARMS:
            r = run_learned(cfg, arm, s)
            rows.append({"arm": arm, "seed": s, **r})
            print(f"[pilot] s={s} {arm:18s} ret={r['return']:.3f} "
                  f"ontime={r['completed_ontime']:.2f} "
                  f"colaps={r['collapses']:.2f}", flush=True)
        if s == PILOT_SEEDS[0]:
            rows.append({"arm": "learned_UNTRAINED", "seed": s,
                         **run_learned(cfg, "learned", s, steps=0)})
    with open(f"{OUT}/pilot.json", "w") as f:
        json.dump(rows, f, indent=2)
    import statistics
    print("\n===== PILOTO F2X =====")
    for arm in sorted({r["arm"] for r in rows}):
        xs = [r["return"] for r in rows if r["arm"] == arm]
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(f"  {arm:20s} ret={sum(xs)/len(xs):+.3f}±{sd:.3f} (n={len(xs)})")


def _best_scripted(tuned):
    return max(("value_density", "edf", "designer"),
              key=lambda n: tuned[n]["best"]["return"])


def _confirm_seed(cfg, tuned, best_name, s):
    """Primario + memoryless de UNA seed. Resumible: escribe seed_{s}.json."""
    path = f"{OUT}/confirm_seed{s}.json"
    if os.path.exists(path):
        return json.load(open(path))
    cfg_oi, cfg_oe = cfg.ood_interrupts(), cfg.ood_energy()
    torch.manual_seed(s)
    pol = train_learned_exec(cfg, LearnedExecPolicy(cfg, ARMS["learned"]),
                             seed=s, steps=TRAIN_STEPS, device=DEV)
    rec = {"seed": s}
    build = {"value_density": ValueDensityPolicy, "edf": EDFPolicy,
             "designer": DesignerPolicy}[best_name]
    for tag, c in (("indist", cfg), ("ood_int", cfg_oi), ("ood_ene", cfg_oe)):
        sp = build(c, **tuned[best_name]["best"]["params"])
        rec[f"scripted_{tag}"] = eval_arm(c, sp, s)["return"]
        rec[f"learned_{tag}"] = eval_arm(c, pol, s)["return"]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, path)
    return rec


def stage_confirm(cfg):
    tuned = json.load(open(f"{OUT}/tuned_scripted.json"))
    best_name = _best_scripted(tuned)
    print(f"[confirm] mejor scripted: {best_name}", flush=True)
    t0 = time.time()
    recs = []
    for s in CONFIRM_SEEDS:
        recs.append(_confirm_seed(cfg, tuned, best_name, s))
        print(f"[confirm] seed {s} ({(time.time()-t0)/60:.0f} min)", flush=True)
    _confirm_report(recs)


def _confirm_report(recs):
    by = {r["seed"]: r for r in recs}
    seeds = sorted(by)
    print("\n===== CONFIRMATORIO F2X =====")
    for tag, label in (("indist", "P1 in-dist"),
                       ("ood_int", "P2 OOD-interrupciones"),
                       ("ood_ene", "P3 OOD-energía")):
        d = [by[s][f"learned_{tag}"] - by[s][f"scripted_{tag}"] for s in seeds]
        pos = sum(1 for x in d if x > 0)
        print(f"{label}: media={sum(d)/len(d):+.4f} seeds+={pos}/{len(d)} "
              f"p_sign={sign_test_p(d):.2e}")


def stage_report(cfg):
    import glob
    recs = [json.load(open(p)) for p in glob.glob(f"{OUT}/confirm_seed*.json")]
    print(f"[report] {len(recs)} seeds encontradas")
    _confirm_report(recs)


def stage_hbp(cfg):
    """Pase exploratorio HBP (6 seeds, aparte por su coste 2x): learned vs
    hbp_wave/hbp_diff en el escenario multi-impulso (la prueba más natural
    del campo homeostático). Resumible."""
    seeds = list(range(300, 308))       # 8 seeds (coste 2x de los HBP)
    t0 = time.time()
    for s in seeds:
        path = f"{OUT}/hbp_seed{s}.json"
        if os.path.exists(path):
            continue
        rec = {"seed": s}
        for arm in ("learned", "learned_memoryless", "hbp_wave", "hbp_diff"):
            rec[arm] = run_learned(cfg, arm, s)["return"]
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=2)
        os.replace(tmp, path)
        print(f"[hbp] seed {s} ({(time.time()-t0)/60:.0f} min)", flush=True)
    import glob
    recs = [json.load(open(p)) for p in glob.glob(f"{OUT}/hbp_seed*.json")]
    print(f"[secundarios] {len(recs)} seeds")
    for arm in ("learned_memoryless", "hbp_wave", "hbp_diff"):
        d = [r["learned"] - r[arm] for r in recs]
        print(f"  learned-{arm}: media={sum(d)/len(d):+.4f} "
              f"p_sign={sign_test_p(d):.2e} (n={len(d)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["tune", "pilot", "confirm", "report", "hbp"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = ExecV2Config()
    print(f"F2X [{args.stage}] en {DEV}", flush=True)
    {"tune": tune, "pilot": stage_pilot, "confirm": stage_confirm,
     "report": stage_report, "hbp": stage_hbp}[args.stage](cfg)
