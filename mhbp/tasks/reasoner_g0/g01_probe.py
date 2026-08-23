"""
G0.1a — ¿La información de COMPLETITUD ya está en las features del halting?

Sobre los ckpts INDIST de cycle_transp: una probe logística ÚNICA (el trabajo
de una cabeza de halting) entrenada sobre (features del tick n, 1[decode_n
correcto]) con dos conjuntos de features:
  F1 = pooled global + frac_valid          (lo que el halting ve HOY)
  F2 = F1 + estado en la posición ARROW    (la respuesta-en-curso)
Métricas: probe-I' = AUC(max_{n<N} score_n, corrección final) por feature-set,
y el BASELINE I' del halting actual (λ_n reconstruida de p_n/remainders).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.g01_probe
"""
import json
import os
import time

import torch
import torch.nn as nn

from training.config import TrainConfig
from training.trainer import build_dataset
from .g0_run import load_model
from .instruments import auc

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = [0, 1, 2]
GENS = "cycle_transp"
N_MAX = 24


@torch.no_grad()
def collect(model, ds, n_samples, device, dtype, batch_size=32):
    """Por muestra y tick n: pooled (d,), arrow (d,), frac_valid, decode_ok,
    y por muestra: correct final, lambda_n (reconstruida), K."""
    model.eval()
    model.record_step_states = True
    a0, a1 = ds.answer_offset, ds.answer_offset + ds.answer_size
    out = {"pooled": [], "arrow": [], "fv": [], "dec_ok": [], "correct": [],
           "lam": [], "K": []}
    got = 0
    while got < n_samples:
        inp, tgt, nd = ds.batch(batch_size)
        inp, tgt = inp.to(device), tgt.to(device)
        if model.cfg.use_hbp:
            model.hbp.reset_state(device=device, dtype=dtype)
        logits, _ = model(inp, None)
        states = model._last_step_states                       # [N] de (B,T,d)
        halt = model._last_halt_dist.float()                   # (B, N)
        preds = a0 + logits[..., a0:a1].argmax(dim=-1)
        B = inp.shape[0]
        p_ans = (nd + 1).to(device)                            # ARROW
        rows = torch.arange(B, device=device)
        fv = (inp != 0).float().mean(dim=1)                    # frac_valid
        # decode por tick en la posición de respuesta
        pooled_t, arrow_t, dec_t = [], [], []
        for s in states:
            arr = s[rows, p_ans]                               # (B, d)
            lg = model.lm_head(model.norm_f(arr))
            dec = a0 + lg[..., a0:a1].argmax(dim=-1)           # (B,)
            pooled_t.append(s.mean(dim=1).float().cpu())
            arrow_t.append(arr.float().cpu())
            dec_t.append((dec == tgt[rows, p_ans]).float().cpu())
        pooled_t = torch.stack(pooled_t, dim=1)                # (B, N, d)
        arrow_t = torch.stack(arrow_t, dim=1)
        dec_t = torch.stack(dec_t, dim=1)                      # (B, N)
        # reconstruir λ_n de p_n
        rem = torch.ones(B)
        lam = torch.zeros_like(halt.cpu())
        hc = halt.cpu()
        for n in range(hc.shape[1]):
            lam[:, n] = (hc[:, n] / rem.clamp(min=1e-8)).clamp(0, 1)
            rem = (rem - hc[:, n]).clamp(min=0)
        take = min(B, n_samples - got)
        out["pooled"].append(pooled_t[:take])
        out["arrow"].append(arrow_t[:take])
        out["fv"].append(fv[:take].float().cpu())
        out["dec_ok"].append(dec_t[:take])
        corr = (preds[rows, p_ans] == tgt[rows, p_ans]).float().cpu()
        out["correct"].append(corr[:take])
        out["lam"].append(lam[:take])
        out["K"].append(nd[:take].float())
        got += take
    model.record_step_states = False
    model._last_step_states = None
    return {k: torch.cat(v) for k, v in out.items()}


def fit_probe(X, y, device, epochs=200, lr=1e-2):
    head = nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    X, y = X.to(device), y.to(device)
    for _ in range(epochs):
        loss = nn.functional.binary_cross_entropy_with_logits(
            head(X).squeeze(-1), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def probe_iprime(head, feats, correct, device):
    """feats: (B, N, F). score por tick; I' = AUC(max_{n<N} score, correct)."""
    with torch.no_grad():
        sc = head(feats.to(device)).squeeze(-1).cpu()          # (B, N)
    best = sc[:, :-1].max(dim=1).values                        # excluye el techo
    return auc(best, correct)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    results = {}
    for variant in ("gating_wm", "hbp_full"):
        for seed in SEEDS:
            t0 = time.time()
            model = load_model(f"{variant}_{GENS}_indist_s{seed}.pt", variant,
                               GENS, device, dtype)
            mk = lambda s: build_dataset(TrainConfig(
                task="permcomp", perm_gens=GENS, max_writes=24, seed=s))
            tr = collect(model, mk(seed + 700_000), 1536, device, dtype)
            te = collect(model, mk(seed + 800_000), 512, device, dtype)

            def flatten(d, keys):
                # (B,N,·) -> (B·(N-1), F) excluyendo el tick del techo
                Xs = []
                for k in keys:
                    v = d[k]
                    if v.dim() == 3:
                        Xs.append(v[:, :-1].reshape(-1, v.shape[-1]))
                    else:                                       # fv por muestra
                        Xs.append(v.unsqueeze(1).expand(-1, N_MAX - 1)
                                  .reshape(-1, 1))
                return torch.cat(Xs, dim=1)

            y_tr = tr["dec_ok"][:, :-1].reshape(-1)
            X1_tr = flatten(tr, ["pooled", "fv"])
            X2_tr = flatten(tr, ["pooled", "arrow", "fv"])
            h1 = fit_probe(X1_tr, y_tr, device)
            h2 = fit_probe(X2_tr, y_tr, device)

            def feats_te(keys):
                Xs = []
                for k in keys:
                    v = te[k]
                    if v.dim() == 3:
                        Xs.append(v)
                    else:
                        Xs.append(v.unsqueeze(1).unsqueeze(2)
                                  .expand(-1, v_shape_n(te), 1))
                return torch.cat(Xs, dim=2)

            def v_shape_n(d):
                return d["pooled"].shape[1]

            F1 = feats_te(["pooled", "fv"])
            F2 = feats_te(["pooled", "arrow", "fv"])
            auc1 = probe_iprime(h1, F1, te["correct"], device)
            auc2 = probe_iprime(h2, F2, te["correct"], device)
            # baseline I' del halting ACTUAL (max λ pre-techo)
            lam_best = te["lam"][:, :-1].max(dim=1).values
            i_base = auc(lam_best, te["correct"])
            results[f"{variant}_s{seed}"] = {
                "probe_F1_auc": auc1, "probe_F2_auc": auc2,
                "halting_Iprime_baseline": i_base,
                "test_acc": float(te["correct"].mean()),
                "wall_s": round(time.time() - t0, 1),
            }
            print(f"{variant}_s{seed}: probe F1={auc1:.3f} F2={auc2:.3f} | "
                  f"I'(halting actual)={i_base:.3f} | acc={float(te['correct'].mean()):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    with open(os.path.join(RES, "g01a_probe.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("G0.1a COMPLETO")


if __name__ == "__main__":
    main()
