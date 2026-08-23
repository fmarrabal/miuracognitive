"""
N3 — Fase A, paso 3: BRAZOS sobre la eval congelada (PREREG_N3 v2 §2.4, §3).

Asignaciones OFFLINE (features de input → tablas/cabezas congeladas de la
sonda; λ bisecado y lotería ajustada sobre las ASIGNACIONES de eval — sin
tocar etiquetas); ejecución forced por brazo; nativo como referencia.
e objetivo = 5 (el mayor verde de VG-N3c, < E[n̄] nativo 5.42).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n3_eval
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .n2_env import N2Dataset, N2Spec
from .n3_sonda import load_ckpt, CKPTS
from .n3_gates import load_sonda, k_table, fit_stake_head, interp_full, N_GRID

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
E_TARGET = 5.0
M_EVAL = 16384
DELTA0 = 0.02


def alloc_argmax(values_by_row, lam):
    """values_by_row: (m, 24) — n_i = argmax_n v[i,n] − λ·n."""
    n_grid = np.arange(1, 25)[None, :]
    return np.argmax(values_by_row - lam * n_grid, axis=1) + 1


def lottery_to_target(values, target):
    """λ⁻/λ⁺ + Bernoulli q para clavar E[n]=target sobre ESTAS asignaciones."""
    lo, hi = 0.0, 4.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if alloc_argmax(values, mid).mean() > target:
            lo = mid
        else:
            hi = mid
    n_hi, n_lo = alloc_argmax(values, lo), alloc_argmax(values, hi)
    e_hi, e_lo = n_hi.mean(), n_lo.mean()
    if abs(e_hi - e_lo) < 1e-9:
        return n_lo, n_lo, 0.0
    q = float(np.clip((target - e_lo) / (e_hi - e_lo), 0, 1))
    return n_lo, n_hi, q


def realize(n_lo, n_hi, q, rng):
    pick = rng.random(len(n_lo)) < q
    return np.where(pick, n_hi, n_lo).astype(int)


@torch.no_grad()
def run_forced(model, X, Y, n_alloc, device):
    """Ejecuta la asignación n_alloc (m,) por lotes → correct (m,)."""
    hits = []
    for i in range(0, len(X), 256):
        xb = X[i:i + 256].to(device)
        nb = torch.tensor(n_alloc[i:i + 256], device=device)
        model.hbp.reset_state(xb.shape[0], device=device)
        model.value_ctx = None
        logits, _ = model(xb, forced_steps=nb)
        yb = Y[i:i + 256]
        valid = (yb != -1)
        il = valid.float().cumsum(1).argmax(1)
        rb = torch.arange(xb.shape[0])
        pred = logits[rb, il.to(device)].argmax(-1).cpu()
        hits.append((pred == yb[rb, il]).float())
    return torch.cat(hits).numpy()


@torch.no_grad()
def run_native(model, X, Y, device):
    hits, ns = [], []
    for i in range(0, len(X), 256):
        xb = X[i:i + 256].to(device)
        model.hbp.reset_state(xb.shape[0], device=device)
        model.value_ctx = None
        logits, _ = model(xb)
        yb = Y[i:i + 256]
        valid = (yb != -1)
        il = valid.float().cumsum(1).argmax(1)
        rb = torch.arange(xb.shape[0])
        pred = logits[rb, il.to(device)].argmax(-1).cpu()
        hits.append((pred == yb[rb, il]).float())
        ns.append(model._last_n_expected.float().cpu())
    return torch.cat(hits).numpy(), torch.cat(ns).numpy()


def metrics(correct, stakes, ks, n_alloc):
    hi = stakes > 1
    def pcorr(a, b, z):
        zc = np.stack([z, np.ones_like(z)], 1)
        ra = a - zc @ np.linalg.lstsq(zc, a.astype(float), rcond=None)[0]
        rb = b - zc @ np.linalg.lstsq(zc, b.astype(float), rcond=None)[0]
        d = np.linalg.norm(ra) * np.linalg.norm(rb)
        return float(ra @ rb / d) if d > 0 else 0.0
    return {"payoff": float((stakes * correct).sum() / stakes.sum()),
            "acc": float(correct.mean()),
            "acc_alto": float(correct[hi].mean()),
            "acc_bajo": float(correct[~hi].mean()),
            "E_n": float(np.mean(n_alloc)),
            "corr_n_stake_K": pcorr(np.asarray(n_alloc, dtype=float),
                                    stakes, ks)}


def main():
    from training.trainer import select_device_dtype
    device, dtype = select_device_dtype()
    spec = N2Spec(p_hi=0.15)
    # eval congelada (seed 999) m=16384 — misma para todos los ckpts
    ds = N2Dataset(spec, seed=999, split="eval")
    Xs, Ys, Ks, Ss = [], [], [], []
    while sum(x.shape[0] for x in Xs) < M_EVAL:
        x, y, k, s = ds.batch(512)
        Xs.append(x)
        Ys.append(y)
        Ks.append(k)
        Ss.append(s)
    X = torch.cat(Xs)[:M_EVAL]
    Y = torch.cat(Ys)[:M_EVAL]
    KK = torch.cat(Ks)[:M_EVAL].numpy()
    SS = torch.cat(Ss)[:M_EVAL].numpy().astype(float)
    print(f"eval congelada: {M_EVAL} instancias (p_hi_emp="
          f"{(SS > 1).mean():.3f})", flush=True)

    out = {"e_target": E_TARGET, "delta0": DELTA0, "ckpts": {}}
    t0 = time.time()
    for name in CKPTS:
        sd = load_sonda(name)
        all_idx = np.arange(len(sd["K"]))
        tab = k_table(sd, all_idx)                     # p̂ (tabla congelada)
        ckpt = torch.load(os.path.join(HERE, "ckpts", f"{name}.pt"),
                          map_location="cpu")
        emb = ckpt["tok_emb.weight"].float().numpy()
        shd, _ = fit_stake_head(sd, emb, all_idx)      # stakê congelada
        acc_K = {K: interp_full(tab[K]) for K in tab}  # (24,) por K

        with torch.no_grad():
            f_eval = torch.tensor(np.concatenate(
                [emb[X[:, 0].numpy()], emb[X[:, 1].numpy()]], axis=1),
                dtype=torch.float32)
            s_hat = F.softplus(shd(f_eval)).squeeze(-1).numpy()
        acc_rows = np.stack([acc_K[int(k)] for k in KK])      # (m, 24)
        sbar = float(SS.mean())

        rng = np.random.default_rng(11)
        allocs = {}
        allocs["uniforme"] = np.full(M_EVAL, int(E_TARGET))
        allocs["dificultad"] = realize(*lottery_to_target(
            sbar * acc_rows, E_TARGET), rng)
        allocs["expl"] = realize(*lottery_to_target(
            s_hat[:, None] * acc_rows, E_TARGET), rng)
        allocs["oraculo_clase"] = realize(*lottery_to_target(
            SS[:, None] * acc_rows, E_TARGET), rng)
        # regla de dos niveles (sin p̂): DP sobre {n_lo, n_hi} por stakê>4
        best = None
        hi_m = s_hat > 4
        for n_hi_v in range(1, 25):
            rem = (E_TARGET * M_EVAL - hi_m.sum() * n_hi_v) / max(
                (~hi_m).sum(), 1)
            if not (1 <= rem <= 24):
                continue
            n_lo_v = int(rem)
            pay = (SS * np.where(hi_m, acc_rows[np.arange(M_EVAL), n_hi_v - 1],
                                 acc_rows[np.arange(M_EVAL), n_lo_v - 1])
                   ).sum() / SS.sum()
            if best is None or pay > best[0]:
                best = (pay, n_lo_v, n_hi_v)
        allocs["regla"] = np.where(hi_m, best[2], best[1]).astype(int)

        model = load_ckpt(name, device, dtype)
        res = {}
        for arm, n_alloc in allocs.items():
            c = run_forced(model, X, Y, n_alloc, device)
            res[arm] = metrics(c, SS, KK, n_alloc)
        c_nat, n_nat = run_native(model, X, Y, device)
        res["nativo"] = metrics(c_nat, SS, KK, n_nat)
        # A3/VG-N3d: dificultad y expl al E[n̄] NATIVO del ckpt
        e_nat = float(n_nat.mean())
        for arm, vals in (("dificultad_at_nat", sbar * acc_rows),
                          ("expl_at_nat", s_hat[:, None] * acc_rows)):
            n_alloc = realize(*lottery_to_target(vals, e_nat), rng)
            c = run_forced(model, X, Y, n_alloc, device)
            res[arm] = metrics(c, SS, KK, n_alloc)
        del model
        torch.cuda.empty_cache()
        out["ckpts"][name] = res
        print(f"{name}: unif={res['uniforme']['payoff']:.3f} "
              f"dif={res['dificultad']['payoff']:.3f} "
              f"expl={res['expl']['payoff']:.3f} "
              f"orac={res['oraculo_clase']['payoff']:.3f} "
              f"regla={res['regla']['payoff']:.3f} "
              f"nativo={res['nativo']['payoff']:.3f}@{e_nat:.1f} "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)

    with open(os.path.join(RES, "n3_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Guardado n3_eval.json — el veredicto formal lo hace n3_report",
          flush=True)


if __name__ == "__main__":
    main()
