"""
N4 — G1 (KILL-GATE, DISENO_N4 §3): ¿el stream posterior por tick contiene
información sobre el éxito MÁS ALLÁ del último tick?

Si el posterior del último tick (más covariables baratas K/stake) es un
estadístico casi-suficiente, el campo-acumulador no tiene nada que
integrar y N4 muere aquí, a coste ~0.

Diseño (declarado antes de mirar resultados):
- Sustrato: los 12 checkpoints blind_flat congelados de N3 (réplicas).
- Instancias/etiquetas: las MISMAS 1024 por checkpoint de la sonda N3
  (reconstruidas con el mismo stream+dedupe; wiring test: K y stake deben
  casar con el JSON guardado). Etiquetas = correct(n) de forced_steps.
- Stream por tick t=1..24 del rollout NATIVO (record_step_states; T-N3a
  garantiza que forced(n) ≡ prefijo nativo): masa PonderNet p_t, margen
  top1−top2 y entropía de la cabeza en la posición de respuesta, flip del
  argmax, velocidad ‖Δpooled‖ relativa.
- Probes logísticos (5-fold CV, ridge, FP32):
    F_last(t)   = [K, stake] + features del tick t             (7 dims)
    F_stream(t) = [K, stake] + features de los ticks 1..t      (2+5t dims)
  Targets: A = correct(24) («¿resoluble?»), B(t) = correct(t) («¿si paro
  ahora acierto?»). CELDA PRIMARIA: t=8, target A. Resto: secundarias.
- Veredicto: ΔAUC = AUC(stream) − AUC(last), pareado por instancia.
  G1 PASA si en la celda primaria la media entre checkpoints ≥ 0.03 con
  IC95 (t, n=12) que excluye 0. Si no: N4 se cierra sin GPU.
- G0-extra (informa G3): ¿hay estructura temporal que resonar? Autocorr
  media a lag≥2 y fracción de instancias con pico espectral no-DC en las
  señales detrendadas (speed, margen).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n4_g1
"""
import json
import os
import time

import numpy as np
import torch

from training.trainer import select_device_dtype
from .n2_env import N2Dataset, N2Spec
from .n3_sonda import CKPTS, M_SONDA, SEED_SONDA, content_hash, load_ckpt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
N_MAX = 24
T_CELLS = [2, 4, 8, 16, 24]
T_PRIMARIA, TGT_PRIMARIA = 8, "A"
RNG = np.random.default_rng(20260809)


# ------------------------- probes y métricas ------------------------- #
def auc_rank(y, s):
    """AUC por Mann-Whitney (empates: rango medio)."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sr = np.asarray(s)[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _fit_logistic(Xtr, ytr, l2):
    w = torch.zeros(Xtr.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=100)

    def cl():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Xtr @ w + b, ytr) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(cl)
    return w.detach(), b.detach()


L2_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0)


def probe_auc(X, y, k=5, seed=0):
    """AUC media por fold (5-fold) con l2 elegido por CV ANIDADA (3-fold
    interna, por AUC). DOS fixes post-control-positivo (2026-08-09):
    (1) con l2 fijo, el probe de más dimensiones paga un impuesto de
    sobreajuste ~0.018 AUC que sesga ΔAUC hacia el nulo (el mismo tipo de
    error del capítulo LLM) → l2 anidado, cada probe con su mejor
    regularización; (2) agrupar scores crudos de folds con l2 distintos
    distorsiona la AUC agrupada (escalas incomparables) → AUC POR FOLD
    promediada, el estimador estándar. Mismo seed ⇒ mismos folds en ambos
    probes ⇒ la diferencia queda pareada también a nivel de fold."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    idx = np.random.default_rng(seed).permutation(len(y))
    folds = np.array_split(idx, k)
    aucs_out = []
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(k) if g != f])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr_np, ytr_np = (X[tr] - mu) / sd, y[tr]
        inner = np.array_split(np.arange(len(tr)), 3)
        best_l2, best_auc = L2_GRID[0], -1.0
        for l2 in L2_GRID:
            aucs = []
            for g in range(3):
                iv = inner[g]
                it = np.concatenate([inner[h] for h in range(3) if h != g])
                w, b = _fit_logistic(torch.tensor(Xtr_np[it]),
                                     torch.tensor(ytr_np[it]), l2)
                sc = (torch.tensor(Xtr_np[iv]) @ w + b).numpy()
                a = auc_rank(ytr_np[iv] > 0.5, sc)
                if not np.isnan(a):
                    aucs.append(a)
            m = float(np.mean(aucs)) if aucs else -1.0
            if m > best_auc:
                best_auc, best_l2 = m, l2
        w, b = _fit_logistic(torch.tensor(Xtr_np),
                             torch.tensor(ytr_np), best_l2)
        sc = (torch.tensor((X[te] - mu) / sd) @ w + b).numpy()
        a = auc_rank(y[te] > 0.5, sc)
        if not np.isnan(a):
            aucs_out.append(a)
    return float(np.mean(aucs_out))


# --------------------- extracción del stream ------------------------- #
@torch.no_grad()
def stream_ckpt(model, spec, device, seen):
    """Reconstruye las 1024 instancias de la sonda y extrae el stream
    nativo por tick. Devuelve (feats (M,24,5), K, stake, idx_last)."""
    ds = N2Dataset(spec, seed=SEED_SONDA, split="train")
    rows = []
    while len(rows) < M_SONDA:
        x, y, ks, ss = ds.batch(256)
        for b in range(x.shape[0]):
            if content_hash(x[b], int(ks[b])) in seen:
                continue
            rows.append((x[b], y[b], int(ks[b]), float(ss[b])))
            if len(rows) >= M_SONDA:
                break
    X = torch.stack([r[0] for r in rows])
    Y = torch.stack([r[1] for r in rows])
    Ks = np.array([r[2] for r in rows])
    Ss = np.array([r[3] for r in rows])
    idx_last = (Y != -1).float().cumsum(1).argmax(1)

    feats = np.zeros((M_SONDA, N_MAX, 5), dtype=np.float64)
    for i in range(0, M_SONDA, 256):
        xb = X[i:i + 256].to(device)
        il = idx_last[i:i + 256].to(device)
        B = xb.shape[0]
        rb = torch.arange(B, device=device)
        model.hbp.reset_state(B, device=device)
        model.value_ctx = None
        model.record_step_states = True
        model(xb)
        model.record_step_states = False
        states = model._last_step_states                 # [24] (B,T,d)
        masses = model._last_halt_probs_live             # [24] (B,)
        prev_pool, prev_arg = None, None
        for t in range(N_MAX):
            st = states[t]
            lg = model.lm_head(model.norm_f(st))[rb, il].float()   # (B,V)
            ls = lg.log_softmax(-1)
            top2 = lg.topk(2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1])
            ent = -(ls.exp() * ls).sum(-1)
            arg = lg.argmax(-1)
            pool = st.float().mean(dim=1)                          # (B,d)
            if prev_pool is None:
                speed = torch.zeros(B, device=device)
                flip = torch.zeros(B, device=device)
            else:
                speed = ((pool - prev_pool).norm(dim=-1)
                         / (prev_pool.norm(dim=-1) + 1e-6))
                flip = (arg != prev_arg).float()
            feats[i:i + B, t, 0] = masses[t].float().cpu().numpy()
            feats[i:i + B, t, 1] = margin.cpu().numpy()
            feats[i:i + B, t, 2] = ent.cpu().numpy()
            feats[i:i + B, t, 3] = flip.cpu().numpy()
            feats[i:i + B, t, 4] = speed.cpu().numpy()
            prev_pool, prev_arg = pool, arg
    return feats, Ks, Ss


def g0_estructura(feats):
    """¿Hay estructura temporal (más allá de rampa) que un filtro
    resonante pueda explotar? Señales detrendadas (ajuste lineal)."""
    out = {}
    for name, j in (("speed", 4), ("margen", 1)):
        sig = feats[:, 1:, j]                            # t=2..24 (speed_1=0)
        t = np.arange(sig.shape[1])
        # detrend lineal por instancia
        A = np.vstack([t, np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A, sig.T, rcond=None)
        res = sig - (A @ coef).T
        # autocorrelación media a lags 2-6
        ac = []
        for lag in range(2, 7):
            a, b = res[:, :-lag], res[:, lag:]
            num = (a * b).mean(1)
            den = res.std(1) ** 2 + 1e-12
            ac.append(num / den)
        ac = np.mean(ac, axis=0)
        # pico espectral no-DC dominante
        sp = np.abs(np.fft.rfft(res, axis=1)) ** 2
        dom = sp[:, 1:].argmax(1) + 1
        frac_peak = float(((dom > 1) & (sp[np.arange(len(dom)), dom]
                                        > 2 * sp[:, 1:].mean(1))).mean())
        out[name] = {"autocorr_lag2_6": float(np.mean(ac)),
                     "frac_pico_noDC": frac_peak}
    return out


def main():
    device, dtype = select_device_dtype()
    spec = N2Spec(p_hi=0.15)
    seen = set()
    for seed, split, m in ((999, "eval", 16384), (2999, "val", 8192)):
        ds = N2Dataset(spec, seed=seed, split=split)
        got = 0
        while got < m:
            x, y, ks, ss = ds.batch(512)
            for b in range(x.shape[0]):
                seen.add(content_hash(x[b], int(ks[b])))
            got += x.shape[0]

    t0 = time.time()
    res = {"celdas": {}, "g0": [], "por_ckpt": {}}
    deltas_prim = []
    for name in CKPTS:
        with open(os.path.join(RES, f"n3_sonda_{name}.json"),
                  encoding="utf-8") as f:
            sonda = json.load(f)
        model = load_ckpt(name, device, dtype)
        feats, Ks, Ss = stream_ckpt(model, spec, device, seen)
        # wiring test: mismas instancias que la sonda
        assert list(map(int, Ks)) == list(map(int, sonda["K"])), name
        assert np.allclose(Ss, sonda["stake"]), name
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        res["g0"].append(g0_estructura(feats))
        cov = np.stack([Ks, Ss], axis=1).astype(np.float64)
        ck = {}
        for t in T_CELLS:
            for tgt in ("A", "B"):
                y = np.array(sonda["correct"]["24" if tgt == "A" else str(t)],
                             dtype=float)
                if y.std() == 0:
                    continue
                Fl = np.concatenate([cov, feats[:, t - 1, :]], axis=1)
                Fs = np.concatenate(
                    [cov, feats[:, :t, :].reshape(M_SONDA, -1)], axis=1)
                d = probe_auc(Fs, y) - probe_auc(Fl, y)
                ck[f"t{t}_{tgt}"] = float(d)
                if t == T_PRIMARIA and tgt == TGT_PRIMARIA:
                    deltas_prim.append(float(d))
        res["por_ckpt"][name] = ck
        print(f"{name}: " + " ".join(f"{k}:{v:+.3f}" for k, v in ck.items())
              + f"  ({(time.time() - t0) / 60:.0f} min)", flush=True)

    # agregado primario: media entre checkpoints, IC-t (n=12)
    d = np.array(deltas_prim)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    ic = (m - 2.201 * se, m + 2.201 * se)                # t_{0.975, 11}
    res["celdas"]["primaria"] = {
        "celda": f"t{T_PRIMARIA}_{TGT_PRIMARIA}", "delta_auc": float(m),
        "se": float(se), "ic95": [float(ic[0]), float(ic[1])],
        "n_ckpts": len(d), "positivos": int((d > 0).sum()),
        "pasa": bool(m >= 0.03 and ic[0] > 0)}
    # secundarias agregadas
    for key in next(iter(res["por_ckpt"].values())).keys():
        vals = np.array([c[key] for c in res["por_ckpt"].values()
                         if key in c])
        res["celdas"][key] = {"media": float(vals.mean()),
                              "se": float(vals.std(ddof=1)
                                          / np.sqrt(len(vals)))}
    p = res["celdas"]["primaria"]
    print(f"\nG1 PRIMARIA ({p['celda']}): ΔAUC={p['delta_auc']:+.4f} "
          f"IC95 [{p['ic95'][0]:+.4f}, {p['ic95'][1]:+.4f}] "
          f"({p['positivos']}/12 ckpts >0) → "
          f"{'PASA' if p['pasa'] else 'NO PASA'}", flush=True)
    g0s = res["g0"]
    for sig in ("speed", "margen"):
        acs = np.mean([g[sig]["autocorr_lag2_6"] for g in g0s])
        fps = np.mean([g[sig]["frac_pico_noDC"] for g in g0s])
        print(f"G0 {sig}: autocorr(lag2-6)={acs:+.3f} "
              f"frac_pico_noDC={fps:.2f}", flush=True)
    with open(os.path.join(RES, "n4_g1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("Guardado n4_g1.json", flush=True)


if __name__ == "__main__":
    main()
