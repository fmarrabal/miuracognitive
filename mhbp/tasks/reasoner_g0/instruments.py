"""
G0 — Instrumentos M (mecanismo), I (interfaz) sobre modelos entrenados.

M1 probes    : ¿los estados por tick decodifican resultados parciales, con
               fidelidad creciente? (F_full(t): respuesta final desde la
               posición de lectura; F_pref(t): permutación del prefijo medio
               desde su posición — la supervisión densa da los targets gratis)
M2 swap      : trasplante COMPLETO de estado (x + WM + HBP) entre pares del
               mismo K en el tick t*: ¿la salida sigue al donante (estado
               portador), re-deriva con coste (ticks extra), o revierte gratis
               (estado no load-bearing)?
M3 ligadura  : fidelidad de trayectoria por instancia (probes) vs acierto OOD.
I  halting   : AUC(concentración de la parada, corrección).

Todo en eval, sin gradiente (salvo el entrenamiento de las probes lineales).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from training.trainer import build_dataset
from training.config import TrainConfig


# --------------------------------------------------------------------------- #
def _forward_eval(model, inp, device, dtype):
    if model.cfg.use_hbp:
        model.hbp.reset_state(device=device, dtype=dtype)
    logits, _ = model(inp, None)
    return logits


@torch.no_grad()
def collect_trajectories(model, ds, n_samples, device, dtype, batch_size=32,
                         k_min=None, k_max=None):
    """Corre el modelo con record_step_states y extrae por muestra:
    feat_ans[t]  (d,)  estado del tick t en la posición de LECTURA (K+1)
    feat_mid[t]  (d,)  estado del tick t en la posición del prefijo MEDIO
    + metadatos (K, ans_token, mid_token, pred, correct, halt_dist, n_exp)."""
    model.eval()
    model.record_step_states = True
    a0, a1 = ds.answer_offset, ds.answer_offset + ds.answer_size
    out = {"feat_ans": [], "feat_mid": [], "K": [], "ans": [], "mid": [],
           "pred": [], "correct": [], "p_stop": [], "lam_best": [], "n_exp": []}
    got = 0
    while got < n_samples:
        inp, tgt, nd = ds.batch(batch_size)
        inp, tgt = inp.to(device), tgt.to(device)
        keep = torch.ones(inp.shape[0], dtype=torch.bool)
        if k_min is not None:
            keep &= (nd >= k_min)
        if k_max is not None:
            keep &= (nd <= k_max)
        if keep.sum() == 0:
            continue
        logits = _forward_eval(model, inp, device, dtype)
        states = model._last_step_states                     # [T_ticks] de (B,T,d)
        preds = a0 + logits[..., a0:a1].argmax(dim=-1)
        for b in range(inp.shape[0]):
            if not keep[b] or got >= n_samples:
                continue
            K = int(nd[b])
            p_ans = K + 1                                    # posición ARROW
            p_mid = max((K // 2) - 1, 0)                     # op del prefijo medio
            if tgt[b, p_ans] == -1 or tgt[b, p_mid] == -1:
                continue
            fa = torch.stack([s[b, p_ans].float() for s in states])   # (T_ticks, d)
            fm = torch.stack([s[b, p_mid].float() for s in states])
            out["feat_ans"].append(fa.cpu())
            out["feat_mid"].append(fm.cpu())
            out["K"].append(K)
            out["ans"].append(int(tgt[b, p_ans]) - a0)
            out["mid"].append(int(tgt[b, p_mid]) - a0)
            out["pred"].append(int(preds[b, p_ans]) - a0)
            out["correct"].append(int(preds[b, p_ans] == tgt[b, p_ans]))
            hd = model._last_halt_dist[b]                    # (N_ticks,)
            out["p_stop"].append(float(hd.max()))
            # I' (G0.1): mejor λ PRE-TECHO, reconstruida de p_n/remainders
            hdf = hd.float()
            rem = 1.0
            lam_best = 0.0
            for t_i in range(hdf.shape[0] - 1):              # excluye el techo
                lam = float(hdf[t_i]) / max(rem, 1e-8)
                lam_best = max(lam_best, min(lam, 1.0))
                rem -= float(hdf[t_i])
            out["lam_best"].append(lam_best)
            out["n_exp"].append(float(model._last_n_expected[b]))
            got += 1
    model.record_step_states = False
    model._last_step_states = None
    return {k: (torch.stack(v) if k in ("feat_ans", "feat_mid") else torch.tensor(v))
            for k, v in out.items()}


# --------------------------------------------------------------------------- #
def train_probes(feats_tr, labels_tr, n_classes, device, epochs=300, lr=1e-2):
    """Una probe lineal POR TICK. feats: (N, T_ticks, d). Devuelve lista de heads."""
    N, T, d = feats_tr.shape
    heads = []
    for t in range(T):
        X = feats_tr[:, t, :].to(device)
        y = labels_tr.to(device)
        head = nn.Linear(d, n_classes).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            loss = nn.functional.cross_entropy(head(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
        heads.append(head.eval())
    return heads


@torch.no_grad()
def probe_accuracy(heads, feats_te, labels_te, device):
    """Accuracy por tick. (T_ticks,)"""
    accs = []
    for t, head in enumerate(heads):
        pred = head(feats_te[:, t, :].to(device)).argmax(-1).cpu()
        accs.append(float((pred == labels_te).float().mean()))
    return accs


@torch.no_grad()
def per_sample_fidelity(heads, feats, labels, device):
    """Fidelidad de trayectoria por muestra: fracción de ticks cuya probe
    decodifica la respuesta correcta. (N,)"""
    hits = []
    for t, head in enumerate(heads):
        pred = head(feats[:, t, :].to(device)).argmax(-1).cpu()
        hits.append((pred == labels).float())
    return torch.stack(hits).mean(0)


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean()
                 / (ra.std(unbiased=False) * rb.std(unbiased=False)))


def pearson(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean()
                 / (a.std(unbiased=False) * b.std(unbiased=False)))


def auc(scores, labels):
    """AUC por rangos (Mann-Whitney). labels binarios."""
    scores, labels = torch.as_tensor(scores).float(), torch.as_tensor(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = scores.argsort().argsort().float() + 1
    r_pos = ranks[labels == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((r_pos - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- #
@torch.no_grad()
def swap_experiment(model, ds, device, dtype, t_stars=(3, 6, 9), n_pairs=192,
                    k_min=8, k_max=20, batch_pairs=16):
    """M2: pares del mismo K; en el tick t* se intercambian x, WM y HBP entre
    las dos mitades del batch. Desenlaces por muestra:
      follow_donor    : pred == respuesta del donante
      rederive_cost   : pred == respuesta propia Y n_exp sube > 0.5 ticks
      free_revert     : pred == respuesta propia Y sin coste
      neither         : ninguna de las dos respuestas
    Baseline sin swap para el coste. Devuelve tasas por t*."""
    model.eval()
    a0, a1 = ds.answer_offset, ds.answer_offset + ds.answer_size

    # --- recolectar pares del mismo K ---
    buckets = {}
    pairs_inp, pairs_tgt, pairs_K = [], [], []
    while len(pairs_inp) < n_pairs:
        inp, tgt, nd = ds.batch(64)
        for b in range(64):
            K = int(nd[b])
            if not (k_min <= K <= k_max):
                continue
            if K in buckets:
                (i2, t2) = buckets.pop(K)
                pairs_inp.append((i2, inp[b])); pairs_tgt.append((t2, tgt[b]))
                pairs_K.append(K)
                if len(pairs_inp) >= n_pairs:
                    break
            else:
                buckets[K] = (inp[b], tgt[b])

    results = {t: {"follow_donor": 0, "rederive_cost": 0, "free_revert": 0,
                   "neither": 0, "n": 0, "dn_mean": 0.0} for t in t_stars}
    for start in range(0, n_pairs, batch_pairs):
        chunk = list(range(start, min(start + batch_pairs, n_pairs)))
        A_inp = torch.stack([pairs_inp[i][0] for i in chunk]).to(device)
        B_inp = torch.stack([pairs_inp[i][1] for i in chunk]).to(device)
        A_tgt = torch.stack([pairs_tgt[i][0] for i in chunk]).to(device)
        B_tgt = torch.stack([pairs_tgt[i][1] for i in chunk]).to(device)
        inp = torch.cat([A_inp, B_inp], 0)
        tgt = torch.cat([A_tgt, B_tgt], 0)
        Bh = len(chunk)
        Ks = torch.tensor([pairs_K[i] for i in chunk])
        p_ans = (Ks + 1).to(device)
        rows = torch.arange(2 * Bh, device=device)
        p_all = torch.cat([p_ans, p_ans])
        own_ans = tgt[rows, p_all]                       # respuesta propia
        donor_ans = torch.cat([own_ans[Bh:], own_ans[:Bh]])

        # baseline sin swap
        model.tick_intervene = None
        logits = _forward_eval(model, inp, device, dtype)
        n_base = model._last_n_expected.clone()
        # con swap en t*
        for t_star in t_stars:
            def swap_fn(x, n, _B=Bh, _t=t_star):
                if n == _t:
                    x = torch.cat([x[_B:], x[:_B]], 0)
                    wb = model._wm_box
                    if wb is not None:
                        for key in ("mem", "w", "f"):
                            if wb.get(key) is not None:
                                v = wb[key]
                                wb[key] = torch.cat([v[_B:], v[:_B]], 0)
                    if hasattr(model.hbp, "swap_halves"):      # adaptador mHBP
                        model.hbp.swap_halves(_B)
                    elif model.cfg.use_hbp:
                        model.hbp.h_t = torch.cat(
                            [model.hbp.h_t[_B:], model.hbp.h_t[:_B]], 0)
                        model.hbp.h_tm1 = torch.cat(
                            [model.hbp.h_tm1[_B:], model.hbp.h_tm1[:_B]], 0)
                return x
            model.tick_intervene = swap_fn
            logits_s = _forward_eval(model, inp, device, dtype)
            model.tick_intervene = None
            n_swap = model._last_n_expected
            preds = a0 + logits_s[..., a0:a1].argmax(dim=-1)
            pred_at = preds[rows, p_all]
            dn = (n_swap - n_base).float()
            r = results[t_star]
            for i in range(2 * Bh):
                # el swap ocurre en el tick t*: muestras que ya habían parado
                # efectivamente (n_base < t*) no fueron intervenidas de verdad
                if float(n_base[i]) < t_star - 0.5:
                    continue
                # pares con la MISMA respuesta: no informativos para follow/revert
                if donor_ans[i] == own_ans[i]:
                    continue
                r["n"] += 1
                r["dn_mean"] += float(dn[i])
                if pred_at[i] == donor_ans[i] and donor_ans[i] != own_ans[i]:
                    r["follow_donor"] += 1
                elif pred_at[i] == own_ans[i]:
                    if dn[i] > 0.5:
                        r["rederive_cost"] += 1
                    else:
                        r["free_revert"] += 1
                else:
                    r["neither"] += 1
    for t in t_stars:
        r = results[t]
        n = max(r["n"], 1)
        r["dn_mean"] /= n
        for k in ("follow_donor", "rederive_cost", "free_revert", "neither"):
            r[k] = r[k] / n
    return results
