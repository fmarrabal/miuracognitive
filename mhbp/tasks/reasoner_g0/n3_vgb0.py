"""
N3 fase B — VG-B0 (gate previo a la enmienda): ¿queda headroom de VALOR
ENCIMA de la parada posterior nativa?

Composición en EVAL (sin entrenar): el halting NATIVO corre intacto y un
offset por-fila condicionado a ŝ sesga su logit (−δ_hi a los ŝ>4: pensar
más; +δ_lo a los demás, bisecado para EMPAREJAR E[n̄] al nativo por ckpt,
|Δ|≤0.05). Barrido δ_hi ∈ {0.5, 1.0, 2.0, 3.5}; headroom_B0 =
max_δ payoff(δ) − payoff(nativo). Umbral declarado ANTES de mirar:
media ≥ 0.01 Y ≥ 0 en ≥10/12 ckpts.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n3_vgb0
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .n2_env import N2Dataset, N2Spec
from .n3_sonda import load_ckpt, CKPTS
from .n3_gates import load_sonda, fit_stake_head

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
M_EVAL = 16384
DELTAS = [0.5, 1.0, 2.0, 3.5]


@torch.no_grad()
def native_pass(model, X, Y, device, offsets=None):
    hits, ns = [], []
    for i in range(0, len(X), 256):
        xb = X[i:i + 256].to(device)
        model.halting.logit_offset = (None if offsets is None
                                      else offsets[i:i + 256].to(device))
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
    model.halting.logit_offset = None
    return torch.cat(hits).numpy(), torch.cat(ns).numpy()


def main():
    from training.trainer import select_device_dtype
    device, dtype = select_device_dtype()
    spec = N2Spec(p_hi=0.15)
    ds = N2Dataset(spec, seed=999, split="eval")
    Xs, Ys, Ks, Ss = [], [], [], []
    while sum(x.shape[0] for x in Xs) < M_EVAL:
        x, y, k, s = ds.batch(512)
        Xs.append(x); Ys.append(y); Ks.append(k); Ss.append(s)
    X = torch.cat(Xs)[:M_EVAL]
    Y = torch.cat(Ys)[:M_EVAL]
    SS = torch.cat(Ss)[:M_EVAL].numpy().astype(float)

    out = {"deltas": DELTAS, "ckpts": {}}
    t0 = time.time()
    for name in CKPTS:
        sd = load_sonda(name)
        ckpt = torch.load(os.path.join(HERE, "ckpts", f"{name}.pt"),
                          map_location="cpu")
        emb = ckpt["tok_emb.weight"].float().numpy()
        shd, _ = fit_stake_head(sd, emb, np.arange(len(sd["K"])))
        with torch.no_grad():
            f_eval = torch.tensor(np.concatenate(
                [emb[X[:, 0].numpy()], emb[X[:, 1].numpy()]], axis=1),
                dtype=torch.float32)
            s_hat = F.softplus(shd(f_eval)).squeeze(-1).numpy()
        hi = torch.tensor(s_hat > 4)

        model = load_ckpt(name, device, dtype)
        c0, n0 = native_pass(model, X, Y, device)
        base_pay = float((SS * c0).sum() / SS.sum())
        e_nat = float(n0.mean())
        best = {"delta": 0.0, "payoff": base_pay, "d_lo": 0.0, "E_n": e_nat}
        for d_hi in DELTAS:
            # bisecar δ_lo para emparejar E[n̄] al nativo
            lo, hic = 0.0, 4.0
            for _ in range(7):
                d_lo = 0.5 * (lo + hic)
                off = torch.where(hi, torch.tensor(-d_hi),
                                  torch.tensor(d_lo)).float()
                c1, n1 = native_pass(model, X, Y, device, off)
                if n1.mean() > e_nat:
                    lo = d_lo
                else:
                    hic = d_lo
                if abs(n1.mean() - e_nat) <= 0.05:
                    break
            pay = float((SS * c1).sum() / SS.sum())
            if abs(n1.mean() - e_nat) <= 0.05 and pay > best["payoff"]:
                best = {"delta": d_hi, "payoff": pay, "d_lo": d_lo,
                        "E_n": float(n1.mean())}
        hr = best["payoff"] - base_pay
        out["ckpts"][name] = {"nativo": base_pay, "E_n_nativo": e_nat,
                              "mejor": best, "headroom": hr}
        print(f"{name}: nativo={base_pay:.3f}@{e_nat:.2f} "
              f"mejor(δ={best['delta']})={best['payoff']:.3f}@{best['E_n']:.2f} "
              f"→ headroom={hr:+.4f} ({(time.time() - t0) / 60:.0f} min)",
              flush=True)
        del model
        torch.cuda.empty_cache()

    hrs = np.array([v["headroom"] for v in out["ckpts"].values()])
    ok = hrs.mean() >= 0.01 and (hrs >= 0).sum() >= 10
    out["veredicto"] = {"media": float(hrs.mean()), "sd": float(hrs.std(ddof=1)),
                        "positivos": int((hrs >= 0).sum()), "verde": bool(ok)}
    print(f"\nVG-B0: headroom sobre el posterior = {hrs.mean():+.4f} ± "
          f"{hrs.std(ddof=1):.4f} ({(hrs >= 0).sum()}/12 ≥ 0) → "
          f"{'VERDE — la fase B integra' if ok else 'ROJO — el posterior ya lo capturó (cierre con jerarquía completa)'}")
    with open(os.path.join(RES, "n3_vgb0.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Guardado n3_vgb0.json")


if __name__ == "__main__":
    main()
