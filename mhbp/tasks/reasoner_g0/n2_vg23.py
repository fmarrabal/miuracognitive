"""
N2 — VG2b y VG3-GPU (PREREG_N2 v2 §6). Ckpts F3a existentes, ~minutos.

VG2b (percepción desde el ESTADO): probe logístico del stake desde los
estados pooled de los ticks 1 y 2 (record_step_states) — umbral AUC ≥ 0.9.
Si el motivo no sobrevive en el estado temprano, V̂ nacería ciego y el
negativo se atribuiría mal (crítico #ramas del panel).

VG3-GPU:
  (a) confusores: en un ckpt F3a (naive al motivo), corr(stake, señal) para
      margen/entropía del decode en t1-t2 y E[n] del halting — deben ser ≈0
      (si el motivo YA mueve el cómputo de un modelo stake-naive, C3 nace
      confundido). Umbral |r| < 0.1.
  (b) dificultad EMPAREJADA: mismas instancias base con slots motivo vs
      no-motivo → Δacc por pares con forward_forced n ∈ {6, 12, 24}.
      Umbral |Δacc| < max(0.02, 2·SE).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n2_vg23
"""
import json
import math
import os

import numpy as np
import torch

from data.synthetic_recall import PermutationCompositionDataset
from .g0_run import load_model
from .n2_env import N2Dataset, N2Spec, SLOT_ALPHABET

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GENS = "cycle_transp"
CKPTS = [("hbp_full", s) for s in (0, 1, 2)] + \
        [("gating_wm", s) for s in (0, 1, 2)]


def probe_auc(feats, labels, iters=400):
    """Probe MLP (1 capa oculta, 32) con split 50/50 → AUC en test.
    RONDA 2 (2026-08-04): con el motivo-RELACIÓN (igualdad de slots), la
    lectura es XOR-like sobre la suma de embeddings — inaccesible a un probe
    lineal (AUC 0.75-0.93); el multiconjunto de 2 slots SÍ es decodificable
    con una capa oculta. V̂ usa la MISMA familia (enmienda del prereg §3)."""
    n = feats.shape[0]
    half = n // 2
    mu, sd = feats[:half].mean(0), feats[:half].std(0).clamp(min=1e-6)
    ftr = (feats - mu) / sd
    net = torch.nn.Sequential(
        torch.nn.Linear(feats.shape[1], 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    y = labels.float()
    # balanceo por pesos (p_hi=0.10 desbalanceado)
    pw = ((1 - y[:half].mean()) / y[:half].mean().clamp(min=1e-6)).clamp(max=50)
    for _ in range(iters):
        opt.zero_grad()
        z = net(ftr[:half]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            z, y[:half], pos_weight=pw)
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = net(ftr[half:]).squeeze(-1)
        yt = y[half:]
        pos, neg = s[yt > 0.5], s[yt < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        auc = (pos[:, None] > neg[None, :]).float().mean() \
            + 0.5 * (pos[:, None] == neg[None, :]).float().mean()
        return float(auc)


@torch.no_grad()
def collect_states(model, ds, m, device, dtype):
    xs, ts, ks, ss = ds.batch(m)
    model.record_step_states = True
    outs = {"t1": [], "t2": [], "margin1": [], "entropy2": [], "n_exp": []}
    B = 256
    for i in range(0, m, B):
        x = xs[i:i + B].to(device)
        logits, _ = model(x)
        st = model._last_step_states

        def feat(tick):
            # la MISMA entrada que V̂ (ronda 2): pooled ⊕ slot0 ⊕ slot1
            h = st[tick].float()
            return torch.cat([h.mean(1), h[:, 0, :], h[:, 1, :]], -1).cpu()
        outs["t1"].append(feat(0))
        outs["t2"].append(feat(1))
        outs["n_exp"].append(model._last_n_expected.float().cpu())
        # margen/entropía del decode en ARROW para t1 y t2
        arrow = (x == 2).float().argmax(dim=1).clamp(max=st[0].shape[1] - 1)
        rws = torch.arange(x.shape[0], device=device)
        for key, tick in (("margin1", 0), ("entropy2", 1)):
            lg = model.lm_head(model.norm_f(st[tick][rws, arrow])).float()
            pr = torch.softmax(lg, dim=-1)
            top2 = pr.topk(2, dim=-1).values
            if key == "margin1":
                outs[key].append((top2[:, 0] - top2[:, 1]).cpu())
            else:
                ent = -(pr * (pr + 1e-9).log()).sum(-1) / math.log(pr.shape[-1])
                outs[key].append(ent.cpu())
    model.record_step_states = False
    return ({k: torch.cat(v) for k, v in outs.items()},
            ks, ss)


@torch.no_grad()
def paired_difficulty(model, spec, seed, device, dtype, m=4096):
    """Mismas instancias base, slots motivo vs no-motivo → Δacc pareada."""
    base = PermutationCompositionDataset(
        min_ops=spec.k_lo, max_ops=spec.k_hi, seq_len=spec.seq_len - 2,
        seed=seed + 55_000, gen_set="cycle_transp")
    rng = np.random.default_rng(99)
    toks, labs = [], []
    for _ in range(m):
        t, l, k = base.sample()
        toks.append(t)
        labs.append(l)
    out = {}
    for cond in ("motivo", "no_motivo"):
        accs = {}
        for n_f in (6, 12, 24):
            hits = []
            for i in range(0, m, 256):
                bt, bl = toks[i:i + 256], labs[i:i + 256]
                rows = []
                for t in bt:
                    if cond == "motivo":               # par IGUAL
                        s0 = s1 = SLOT_ALPHABET[int(rng.integers(0, 4))]
                    else:                              # par desigual
                        while True:
                            a = SLOT_ALPHABET[int(rng.integers(0, 4))]
                            b = SLOT_ALPHABET[int(rng.integers(0, 4))]
                            if a != b:
                                s0, s1 = a, b
                                break
                    rows.append(torch.cat([torch.tensor([s0, s1]), t]))
                x = torch.stack(rows)[:, :-1].to(device)
                y = torch.stack([torch.cat([torch.tensor([-1, -1]), l])
                                 for l in bl])[:, :-1]
                forced = torch.full((x.shape[0],), n_f, device=device)
                logits, _ = model(x, forced_steps=forced)
                valid = (y != -1)
                idx_last = valid.float().cumsum(1).argmax(1)
                rws = torch.arange(x.shape[0])
                pred = logits[rws, idx_last.to(device)].argmax(-1).cpu()
                hits.append((pred == y[rws, idx_last]).float())
            accs[n_f] = torch.cat(hits)
        out[cond] = accs
    res = {}
    for n_f in (6, 12, 24):
        d = out["motivo"][n_f] - out["no_motivo"][n_f]
        se = float(d.std() / math.sqrt(len(d)))
        res[n_f] = {"delta": float(d.mean()), "se": se,
                    "ok": bool(abs(float(d.mean())) < max(0.02, 2 * se))}
    return res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    spec = N2Spec()
    out = {"VG2b": {}, "VG3a": {}, "VG3b": {}}
    import dataclasses
    spec_probe = dataclasses.replace(spec, p_hi=0.5)   # probe BALANCEADO:
    # VG2b certifica PERCEPCIÓN de la relación; el prior de clase es
    # irrelevante para la legibilidad y p_hi=0.10 mataba la eficiencia
    # muestral del probe (77 positivos)
    for variant, seed in CKPTS:
        model = load_model(f"{variant}_{GENS}_indist_s{seed}.pt", variant,
                           GENS, device, dtype)
        ds = N2Dataset(spec_probe, seed=seed + 70_000, split="eval")
        feats, ks, ss = collect_states(model, ds, 3072, device, dtype)
        y = (ss > 1)
        auc1 = probe_auc(feats["t1"], y)
        auc2 = probe_auc(feats["t2"], y)
        r_m = float(np.corrcoef(feats["margin1"].numpy(), ss.numpy())[0, 1])
        r_e = float(np.corrcoef(feats["entropy2"].numpy(), ss.numpy())[0, 1])
        r_n = float(np.corrcoef(feats["n_exp"].numpy(), ss.numpy())[0, 1])
        key = f"{variant}_s{seed}"
        out["VG2b"][key] = {"auc_t1": auc1, "auc_t2": auc2}
        out["VG3a"][key] = {"r_margen_t1": r_m, "r_entropia_t2": r_e,
                            "r_En": r_n}
        print(f"{key:22s} VG2b: AUC t1={auc1:.3f} t2={auc2:.3f} | "
              f"VG3a: r_margen={r_m:+.3f} r_entropia={r_e:+.3f} "
              f"r_E[n]={r_n:+.3f}", flush=True)
        if seed == 0:                       # VG3b con un ckpt por variante
            res = paired_difficulty(model, spec, seed, device, dtype)
            out["VG3b"][key] = res
            msg = " ".join(f"n{n}: Δ={v['delta']:+.4f}±{v['se']:.4f}"
                           f"{'✓' if v['ok'] else '✗'}"
                           for n, v in res.items())
            print(f"{key:22s} VG3b (dificultad pareada): {msg}", flush=True)
        del model
        torch.cuda.empty_cache()

    aucs2 = [v["auc_t2"] for v in out["VG2b"].values()]
    vg2b_ok = min(aucs2) >= 0.9
    vg3a_ok = all(abs(v[k]) < 0.1 for v in out["VG3a"].values() for k in v)
    vg3b_ok = all(v["ok"] for res in out["VG3b"].values()
                  for v in res.values())
    out["veredicto"] = {"VG2b": bool(vg2b_ok), "VG3a": bool(vg3a_ok),
                        "VG3b": bool(vg3b_ok)}
    with open(os.path.join(RES, "n2_vg23.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nVG2b (AUC t2 mín {min(aucs2):.3f} ≥ 0.9): "
          f"{'PASA' if vg2b_ok else 'FALLA'}")
    print(f"VG3a (confusores |r|<0.1): {'PASA' if vg3a_ok else 'FALLA'}")
    print(f"VG3b (dificultad pareada): {'PASA' if vg3b_ok else 'FALLA'}")
    print("Guardado results/n2_vg23.json")


if __name__ == "__main__":
    main()
