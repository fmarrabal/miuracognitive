"""Benchmark AHA-2 (rediseño post-auditoría; reglas en AGENCY_V2.md).

Etapas (seeds DISJUNTAS por etapa):
  tune    seeds 100-102: grid-search de los scripted (θ / c,τ / combo).
          Guardrail G-tuning: se persiste el grid completo y el óptimo.
  pilot   seeds 200-202: aprendido vs scripted afinados vs ablación cue-blind
          vs HBP (onda/difusión) en delays {0,2,3}. Gates: G-untrained
          (la red sin entrenar debe rendir peor), G-ceiling (<0.98), y
          aprendibilidad. El piloto DECIDE la config y se congela.
  confirm seeds 300-319 (20): primarios pre-registrados
          P1 = supervivencia aprendido − combo afinado en delay=2 (pareado)
          P2 = pendiente dosis-respuesta de esa diferencia en delay∈{0,1,2,3}
          Sign test exacto + t pareada, Holm sobre {P1, P2}. Secundarios:
          aprendido−ablación, brazos HBP (exploratorios). G-LOSO.

Uso:  $env:PYTHONPATH="." ; python aha_v2_benchmark.py --stage tune|pilot|confirm
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import torch

from data.aha_v2 import AHA2Config, AHA2Dataset
from model.aha_v2 import (ComboPolicy, CueFollowerPolicy, LearnedAHA2Policy,
                          LearnedPolicyConfig, ThresholdPolicy,
                          evaluate_policy, train_learned)

OUT = "results_aha_v2"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EVAL_BATCH = 512
TRAIN_STEPS = 1000    # scan: 800->0.99, 1500->1.000; 1000 = margen bajo techo (R7)

TUNE_SEEDS = [100, 101, 102]
PILOT_SEEDS = [200, 201, 202]
CONFIRM_SEEDS = list(range(300, 320))
DELAYS = [0, 1, 2, 3]
PRIMARY_DELAY = 2


def eval_arm(env_cfg, policy, seed) -> dict:
    ds = AHA2Dataset(env_cfg, seed=seed + 50_000)      # eval disjunto del train
    sc = ds.batch(EVAL_BATCH).to(DEV)
    return evaluate_policy(env_cfg, sc, policy)


# ------------------------------- TUNE -------------------------------------- #

def tune_scripted(base_cfg: AHA2Config) -> dict:
    """Grid-search honesto de los scripted por delay (G-tuning)."""
    grids = {
        "threshold": {"theta": [round(0.44 + 0.02 * i, 3) for i in range(11)]},
        "cue_follower": {"c_min": [0.05, 0.10, 0.15, 0.20],
                         "tau": list(range(0, 8))},
        "combo": {"theta": [0.48, 0.52, 0.56, 0.60],
                  "c_min": [0.05, 0.10, 0.15],
                  "tau": list(range(0, 8))},
    }
    build = {
        "threshold": lambda c, p: ThresholdPolicy(c, **p),
        "cue_follower": lambda c, p: CueFollowerPolicy(c, **p),
        "combo": lambda c, p: ComboPolicy(c, **p),
    }
    result = {}
    for delay in DELAYS:
        cfg = base_cfg.with_delay(delay)
        result[str(delay)] = {}
        for name, grid in grids.items():
            keys = list(grid)
            best = None
            log = []
            for combo in itertools.product(*(grid[k] for k in keys)):
                params = dict(zip(keys, combo))
                surv = []
                for s in TUNE_SEEDS:
                    surv.append(eval_arm(cfg, build[name](cfg, params), s)["survival"])
                m = sum(surv) / len(surv)
                log.append({"params": params, "survival": m})
                if best is None or m > best["survival"]:
                    best = {"params": params, "survival": m}
            result[str(delay)][name] = {"best": best, "grid": log}
            print(f"[tune] delay={delay} {name:12s} -> {best['params']} "
                  f"surv={best['survival']:.3f}", flush=True)
    return result


# ---------------------------- brazos aprendidos ----------------------------- #

LEARNED_ARMS = {
    "learned": LearnedPolicyConfig(),
    "learned_cueblind": LearnedPolicyConfig(cue_blind=True),          # ABLACIÓN
    "hbp_wave": LearnedPolicyConfig(use_hbp=True, hbp_alpha_const=1.0),
    "hbp_diff": LearnedPolicyConfig(use_hbp=True, hbp_alpha_const=0.0),
}


def run_learned(env_cfg, arm: str, seed: int, steps: int = TRAIN_STEPS) -> dict:
    torch.manual_seed(seed)
    pol = LearnedAHA2Policy(env_cfg, LEARNED_ARMS[arm])
    pol = train_learned(env_cfg, pol, seed=seed, steps=steps, device=DEV)
    return eval_arm(env_cfg, pol, seed)


def scripted_from(tuned: dict, env_cfg: AHA2Config, delay: int) -> dict:
    t = tuned[str(delay)]
    return {
        "threshold": ThresholdPolicy(env_cfg, **t["threshold"]["best"]["params"]),
        "cue_follower": CueFollowerPolicy(env_cfg, **t["cue_follower"]["best"]["params"]),
        "combo": ComboPolicy(env_cfg, **t["combo"]["best"]["params"]),
    }


# ------------------------------- estadística -------------------------------- #

def sign_test_p(diffs) -> float:
    n = len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    total = 0.0
    for k in range(min(pos, n - pos) + 1):
        total += math.comb(n, k)
    return min(1.0, 2.0 * total / 2 ** n)


def paired_t(diffs) -> tuple[float, float]:
    import statistics
    n = len(diffs)
    m = sum(diffs) / n
    sd = statistics.stdev(diffs)
    if sd == 0:
        return float("inf"), 0.0
    t = m / (sd / math.sqrt(n))
    # p aproximada vía normal (n=20; para el registro exacto se usa scipy aparte)
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / math.sqrt(2))))
    return t, p


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, m = {}, len(items)
    for rank, (k, p) in enumerate(items):
        out[k] = min(1.0, p * (m - rank))
    return out


# --------------------------------- etapas ----------------------------------- #

def stage_tune(base_cfg):
    os.makedirs(OUT, exist_ok=True)
    tuned = tune_scripted(base_cfg)
    with open(f"{OUT}/tuned_scripted.json", "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"[tune] guardado en {OUT}/tuned_scripted.json")


def stage_pilot(base_cfg):
    tuned = json.load(open(f"{OUT}/tuned_scripted.json"))
    rows = []
    pilot_delays = [0, 2, 3]
    for delay in pilot_delays:
        cfg = base_cfg.with_delay(delay)
        for s in PILOT_SEEDS:
            for name, pol in scripted_from(tuned, cfg, delay).items():
                r = eval_arm(cfg, pol, s)
                rows.append({"arm": name, "delay": delay, "seed": s, **r})
            arms = (["learned", "learned_cueblind"] if delay != PRIMARY_DELAY
                    else list(LEARNED_ARMS))
            for arm in arms:
                r = run_learned(cfg, arm, s)
                rows.append({"arm": arm, "delay": delay, "seed": s, **r})
                print(f"[pilot] d={delay} s={s} {arm:16s} "
                      f"surv={r['survival']:.3f} lead={r['anticipation_lead']:.2f}",
                      flush=True)
            # G-untrained (R6): la red sin entrenar debe rendir peor
            if delay == PRIMARY_DELAY and s == PILOT_SEEDS[0]:
                r0 = run_learned(cfg, "learned", s, steps=0)
                rows.append({"arm": "learned_UNTRAINED", "delay": delay,
                             "seed": s, **r0})
    with open(f"{OUT}/pilot.json", "w") as f:
        json.dump(rows, f, indent=2)
    _pilot_report(rows)


def _pilot_report(rows):
    import statistics
    print("\n===== PILOTO AHA-2 =====")
    arms = sorted({r["arm"] for r in rows})
    for delay in sorted({r["delay"] for r in rows}):
        print(f"-- delay={delay}")
        for arm in arms:
            xs = [r["survival"] for r in rows
                  if r["arm"] == arm and r["delay"] == delay]
            if xs:
                sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
                print(f"   {arm:18s} surv={sum(xs)/len(xs):.3f}±{sd:.3f} (n={len(xs)})")
    print("Gates: G-untrained (UNTRAINED << learned), G-ceiling (<0.98 en "
          "brazos primarios), aprendibilidad (learned > cueblind en d>=2).")


def stage_confirm(base_cfg):
    tuned = json.load(open(f"{OUT}/tuned_scripted.json"))
    rows = []
    t0 = time.time()
    for s in CONFIRM_SEEDS:
        per_delay_diff = {}
        for delay in DELAYS:
            cfg = base_cfg.with_delay(delay)
            combo = scripted_from(tuned, cfg, delay)["combo"]
            r_combo = eval_arm(cfg, combo, s)
            r_learn = run_learned(cfg, "learned", s)
            rows.append({"arm": "combo", "delay": delay, "seed": s, **r_combo})
            rows.append({"arm": "learned", "delay": delay, "seed": s, **r_learn})
            per_delay_diff[delay] = r_learn["survival"] - r_combo["survival"]
            if delay == PRIMARY_DELAY:
                for arm in ("learned_cueblind", "hbp_wave", "hbp_diff"):
                    r = run_learned(cfg, arm, s)
                    rows.append({"arm": arm, "delay": delay, "seed": s, **r})
        # pendiente dosis-respuesta por seed (regresión sobre delay)
        xs = list(per_delay_diff)
        ys = [per_delay_diff[d] for d in xs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                 / sum((x - mx) ** 2 for x in xs))
        rows.append({"arm": "_doseresp_slope", "delay": -1, "seed": s,
                     "survival": slope})
        print(f"[confirm] seed {s} listo ({(time.time()-t0)/60:.0f} min)", flush=True)
    with open(f"{OUT}/confirm.json", "w") as f:
        json.dump(rows, f, indent=2)
    _confirm_report(rows)


def _confirm_report(rows):
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["delay"]), {})[r["seed"]] = r["survival"]
    p1 = [by[("learned", PRIMARY_DELAY)][s] - by[("combo", PRIMARY_DELAY)][s]
          for s in CONFIRM_SEEDS]
    p2 = [by[("_doseresp_slope", -1)][s] for s in CONFIRM_SEEDS]
    pvals = {"P1_learned_vs_combo": sign_test_p(p1),
             "P2_doseresp_slope": sign_test_p(p2)}
    ph = holm(pvals)
    print("\n===== CONFIRMATORIO AHA-2 (primarios pre-registrados) =====")
    for name, diffs in (("P1 aprendido-combo@d2", p1), ("P2 pendiente dosis", p2)):
        t, _ = paired_t(diffs)
        pos = sum(1 for d in diffs if d > 0)
        print(f"{name}: media={sum(diffs)/len(diffs):+.4f} "
              f"seeds+={pos}/{len(diffs)} t={t:.2f} "
              f"p_Holm={ph[list(pvals)[0 if 'P1' in name else 1]]:.2e}")
    # G-LOSO sobre P1
    worst = max(range(len(p1)),
                key=lambda i: abs(sum(p1) / len(p1)
                                  - sum(p1[:i] + p1[i+1:]) / (len(p1) - 1)))
    loso = p1[:worst] + p1[worst+1:]
    print(f"G-LOSO P1: sin seed más influyente -> media={sum(loso)/len(loso):+.4f} "
          f"p_sign={sign_test_p(loso):.2e}")
    for arm in ("learned_cueblind", "hbp_wave", "hbp_diff"):
        if (arm, PRIMARY_DELAY) in by:
            d = [by[("learned", PRIMARY_DELAY)][s] - by[(arm, PRIMARY_DELAY)][s]
                 for s in CONFIRM_SEEDS if s in by[(arm, PRIMARY_DELAY)]]
            print(f"secundario learned-{arm}: media={sum(d)/len(d):+.4f} "
                  f"p_sign={sign_test_p(d):.2e} (n={len(d)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "pilot", "confirm"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    base = AHA2Config()
    base.validate()
    print(f"AHA-2 [{args.stage}] en {DEV}", flush=True)
    {"tune": stage_tune, "pilot": stage_pilot, "confirm": stage_confirm}[args.stage](base)
