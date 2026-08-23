"""Benchmark Fase 4 v2 (automodelo aprendido con daño parcial; AGENCY_V2.md).

  tune    100-102: umbral del detector del brazo reinit.
  pilot   200-202: adaptive vs frozen vs reinit-afinado vs oracle vs random.
  confirm 300-319: P1 = coste post-daño frozen − adaptive (¿la adaptación
          online paga?); P2 = reinit-afinado − adaptive (¿aprendizaje continuo
          bate al detector scripted?); dose-response sobre severidad.

Sin entrenamiento entre episodios: todo el aprendizaje es ONLINE dentro del
episodio (identificación del cuerpo); los seeds gobiernan planta y ruido.

Uso: $env:PYTHONPATH="." ; python self_model_v2_benchmark.py --stage tune|pilot|confirm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from model.self_model_v2 import SelfModel2Config, run_episode

OUT = "results_self_model_v2"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH = 256
TUNE_SEEDS = [100, 101, 102]
PILOT_SEEDS = [200, 201, 202]
CONFIRM_SEEDS = list(range(300, 320))
SEVERITIES = [0.2, 0.4, 0.6]      # dose-response (efectividad residual)


def sign_test_p(diffs):
    n = len(diffs)
    pos = sum(1 for d in diffs if d > 0)
    total = sum(math.comb(n, k) for k in range(min(pos, n - pos) + 1))
    return min(1.0, 2.0 * total / 2 ** n)


def tune(cfg):
    best, log = None, []
    for thr in [0.01, 0.02, 0.04, 0.08, 0.16]:
        cs = [run_episode(cfg, BATCH, s, "reinit", reinit_threshold=thr,
                          device=DEV)["cost_post"] for s in TUNE_SEEDS]
        m = sum(cs) / len(cs)
        log.append({"threshold": thr, "cost_post": m})
        if best is None or m < best["cost_post"]:
            best = {"threshold": thr, "cost_post": m}
    with open(f"{OUT}/tuned_reinit.json", "w") as f:
        json.dump({"best": best, "grid": log}, f, indent=2)
    print(f"[tune] reinit -> thr={best['threshold']} "
          f"cost_post={best['cost_post']:.3f}", flush=True)


def stage_pilot(cfg):
    thr = json.load(open(f"{OUT}/tuned_reinit.json"))["best"]["threshold"]
    rows = []
    for s in PILOT_SEEDS:
        for arm in ("random", "oracle", "adaptive", "frozen", "reinit"):
            r = run_episode(cfg, BATCH, s, arm, reinit_threshold=thr, device=DEV)
            rows.append({"arm": arm, "seed": s, **r})
            print(f"[pilot] s={s} {arm:9s} post={r['cost_post']:.3f} "
                  f"err={r['pred_err_final']:.4f}", flush=True)
    with open(f"{OUT}/pilot.json", "w") as f:
        json.dump(rows, f, indent=2)


def stage_confirm(cfg):
    thr = json.load(open(f"{OUT}/tuned_reinit.json"))["best"]["threshold"]
    rows = []
    t0 = time.time()
    for s in CONFIRM_SEEDS:
        for arm in ("adaptive", "frozen", "reinit", "oracle"):
            for sev in SEVERITIES:
                r = run_episode(cfg, BATCH, s, arm, severity=sev,
                                reinit_threshold=thr, device=DEV)
                rows.append({"arm": arm, "seed": s, "severity": sev, **r})
        print(f"[confirm] seed {s} ({(time.time()-t0)/60:.0f} min)", flush=True)
    with open(f"{OUT}/confirm.json", "w") as f:
        json.dump(rows, f, indent=2)
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["severity"]), {})[r["seed"]] = r["cost_post"]
    print("\n===== CONFIRMATORIO FASE 4 v2 =====")
    for sev in SEVERITIES:
        p1 = [by[("frozen", sev)][s] - by[("adaptive", sev)][s] for s in CONFIRM_SEEDS]
        p2 = [by[("reinit", sev)][s] - by[("adaptive", sev)][s] for s in CONFIRM_SEEDS]
        ora = sum(by[("oracle", sev)].values()) / 20
        ada = sum(by[("adaptive", sev)].values()) / 20
        print(f"severidad={sev}: P1 frozen−adaptive={sum(p1)/20:+.4f} "
              f"(+{sum(1 for x in p1 if x > 0)}/20, p={sign_test_p(p1):.2e}) | "
              f"P2 reinit−adaptive={sum(p2)/20:+.4f} "
              f"(p={sign_test_p(p2):.2e}) | regret vs oracle={ada-ora:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "pilot", "confirm"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = SelfModel2Config()
    print(f"Fase 4 v2 [{args.stage}] en {DEV}", flush=True)
    {"tune": tune, "pilot": stage_pilot, "confirm": stage_confirm}[args.stage](cfg)
