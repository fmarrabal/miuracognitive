"""
Fase 2 — Orquestador del confirmatorio (PREREG_PHASE2.md).

Celdas: 5 controladores PRIMARIOS (mhbp, gru, mhbp_taueq, mhbp_first,
hbp_single) × semillas 0..19  +  3 exploratorios (mhbp_nocpl, mlp, ar2) ×
semillas 0..9. RESUMIBLE: 1 JSON atómico por celda; reclamo por fichero .claim
exclusivo (varios workers en paralelo sobre la misma GPU sin colisiones).

  # worker (lanzar K procesos; steps=500 = pre-registrado v3c):
  python -m mhbp.experiments.exp_phase2 --worker 0
  # estado:
  python -m mhbp.experiments.exp_phase2 --status
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "tasks", "synthetic_multiscale", "results_phase2")
CLAIM_TTL_S = 2 * 3600          # los runs duran ≤40 min; 2h de margen

# PREREG v3c: C1-C3 primarios; C4 (hbp_single) EXPLORATORIO. steps=500 (el
# diagnóstico de techo mostró 250 infra-entrenado: priv 0.575→0.610→0.721 a
# 250/500/750). Semillas ajustadas al presupuesto nocturno con 6 workers:
# el par C3 (mhbp/mhbp_first) con n=24; gru n=20; taueq n=16; exploratorios n=8.
SEEDS = {"mhbp": 24, "mhbp_first": 24, "gru": 20, "mhbp_taueq": 16,
         "hbp_single": 8, "mhbp_nocpl": 8, "mhbp_noallo": 8,
         "mlp": 8, "ar2": 8,
         # Fase 2b (PREREG_PHASE2B v2): campo-modulador vs gain-scheduler GRU
         # vs lazo reactivo puro (gru_gov = el control que aísla la FÍSICA)
         "mhbp_gov": 20, "react": 20, "gru_gov": 20}


def cells():
    return [(c, s) for c, n in SEEDS.items() for s in range(n)]


def cell_id(c, s):
    return f"{c}_s{s}"


def try_claim(cid: str) -> bool:
    os.makedirs(RES, exist_ok=True)
    claim = os.path.join(RES, f"{cid}.claim")
    if os.path.exists(os.path.join(RES, f"{cid}.json")):
        return False
    if os.path.exists(claim):
        try:
            if time.time() - os.path.getmtime(claim) < CLAIM_TTL_S:
                return False                          # otro worker la tiene
            os.remove(claim)                          # reclamo rancio
        except (FileNotFoundError, PermissionError):
            return False                              # carrera: otro lo limpió
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def worker(widx: int, steps: int, device: str):
    from mhbp.tasks.synthetic_multiscale.train import train_run
    todo = cells()
    rng = random.Random(widx)
    rng.shuffle(todo)                                 # orden distinto por worker
    done_here = 0
    for c, s in todo:
        cid = cell_id(c, s)
        if not try_claim(cid):
            continue
        print(f"[w{widx}] {cid} ...", flush=True)
        t0 = time.time()
        try:
            train_run(c, s, device_str=device, steps=steps,
                      out_json=os.path.join(RES, f"{cid}.json"),
                      cache_dir=os.path.join(RES, "oracle_cache"),
                      log_every=0)
            print(f"[w{widx}] OK {cid} ({time.time()-t0:.0f}s)", flush=True)
            done_here += 1
        except Exception as e:
            with open(os.path.join(RES, f"{cid}.error"), "w") as f:
                f.write(repr(e))
            print(f"[w{widx}] ERR {cid}: {repr(e)[:100]}", flush=True)
        finally:
            claim = os.path.join(RES, f"{cid}.claim")
            if os.path.exists(claim):
                os.remove(claim)
    print(f"[w{widx}] FIN — completadas {done_here}", flush=True)


def prewarm(device: str = "cuda"):
    """Calienta la caché del oráculo (7 protocolos) en UN solo proceso, antes
    de lanzar los workers (evita la carrera de caché)."""
    import torch
    from mhbp.tasks.synthetic_multiscale.train import (EVAL_PROTOCOLS,
                                                       EVAL_EPISODES,
                                                       _oracle_cached)
    from mhbp.tasks.synthetic_multiscale.env import EnvConfig, sample_latents
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg = EnvConfig()
    cache = os.path.join(RES, "oracle_cache")
    for proto, spec in EVAL_PROTOCOLS.items():
        t0 = time.time()
        lat = sample_latents(cfg, EVAL_EPISODES, spec["T"], spec["seed"], dev,
                             ood=spec["ood"])
        J = _oracle_cached(cache, proto, lat, dev, cfg, spec)
        print(f"  oráculo[{proto}]: J={float(J.mean()):.4f} ({time.time()-t0:.0f}s)",
              flush=True)
    print("prewarm completo", flush=True)


def status():
    total = len(cells())
    done = sum(1 for c, s in cells()
               if os.path.exists(os.path.join(RES, f"{cell_id(c, s)}.json")))
    errs = len([f for f in os.listdir(RES) if f.endswith(".error")]) \
        if os.path.isdir(RES) else 0
    claims = len([f for f in os.listdir(RES) if f.endswith(".claim")]) \
        if os.path.isdir(RES) else 0
    print(f"{done}/{total} celdas — errores {errs}, en curso {claims}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--prewarm", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
    elif a.prewarm:
        prewarm(a.device)
    elif a.worker is not None:
        worker(a.worker, a.steps, a.device)
    else:
        ap.print_help()
