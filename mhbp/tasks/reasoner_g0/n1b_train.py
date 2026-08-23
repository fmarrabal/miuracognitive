"""
N1b — Runner de entrenamiento (PREREG_N1B §3-§4).

Brazo ÚNICO blind_flat (CE plana; stakes solo en metadatos): como en N3,
los brazos de valor/dificultad/posterior son de EVAL sobre checkpoints
congelados. Variante n2_endo, N_MAX=24, protocolo de n2_train (AdamW,
clip, cosine), sin replay de V̂ (beta_val=0 → cabezas sin gradiente,
excluidas del optimizador).

Eval final integrada para los gates:
  A-N1b: acc nativa global y por L (umbral 0.60 / 0.40 en L=18).
  D-N1b: curva forzada acc(n) n∈{2,4,8,12,16,24} global y por L
         (pendiente ≥ 0.15; separación de n₀.₉ entre L=18 y L=6 ≥ 2).

  python -m mhbp.tasks.reasoner_g0.n1b_train --seeds 0 --runs 1 --steps 2500
  (--smoke: 60 pasos, solo cableado)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from training.config import TrainConfig
from training.trainer import build_model, get_lr, select_device_dtype
from .n1b_env import (ARROW_IDX, SEQ_LEN_D, L_SET, N1bDatasetDenso,
                      N1bSpec)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "ckpts")
N_MAX = 24
EVAL_SEED = 999
EVAL_M = 4096
FORCED_GRID = [2, 4, 8, 12, 16, 24]


def build_n1b_model(vocab, device, dtype, seed, steps):
    torch.manual_seed(seed)
    # FIX panel (bug LR): sin max_steps el coseno anela al default (5000)
    # y el run de 2500/4000 pasos termina sin fase de anelado.
    cfg = TrainConfig(task="permcomp", variant="n2_endo", seed=seed,
                      max_halt_steps=N_MAX, max_steps=steps)
    # RONDA 3: batch 128 (el retrieval grokkea con batch 128 y NO con 32
    # — probe denso: silla hasta ~9k, acc 1.000 en 15k)
    cfg.batch_size = 128
    model = build_model("n2_endo", vocab, SEQ_LEN_D,
                        halt_mod_gain=cfg.halt_mod_gain,
                        max_halt_steps=N_MAX, wm_in_loop=cfg.wm_in_loop,
                        ponder_expected_loss=cfg.ponder_expected_loss
                        ).to(device=device, dtype=dtype)
    model.cfg.beta_val = 0.0
    # PREREG v2 §9.5: con bias 0, la masa PonderNet del tick 17 es 8·10⁻⁶
    # y L=18 no recibe gradiente NUNCA en una familia todo-o-nada (sin
    # mejora parcial que empuje λ abajo, como sí había en S₅). Init a −3
    # → n̄ inicial cerca de N_MAX: todos los ticks ven gradiente.
    with torch.no_grad():
        model.halting.halt_proj.bias.fill_(-3.0)
    if model.cfg.use_hbp:
        model.hbp.pin_fp32()
    return model, cfg


@torch.no_grad()
def eval_gates(model, spec, device, dtype):
    """Gates v2 (PREREG §9.4): A sobre acc FORZADA(24); C acantilado-
    muerde; D 3 patas; LL llegada-legible (Spearman(n̄,d|L))."""
    ds = N1bDatasetDenso(spec, seed=EVAL_SEED, split="eval")
    model.eval()
    acc_nat, n_exp_all, Ls_all, d_all = [], [], [], []
    forced = {n: {L: [] for L in L_SET} for n in FORCED_GRID}
    for i in range(0, EVAL_M, 256):
        x, y, Ls, ss = ds.batch(min(256, EVAL_M - i))
        d_b = ds.last_d.clone()
        x, y = x.to(device), y.to(device)
        il = torch.full((x.shape[0],), ARROW_IDX,
                        device=device, dtype=torch.long)
        rows = torch.arange(x.shape[0], device=device)
        model.value_ctx = {"stake_true": ss.to(device)}
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        logits, _ = model(x)
        corr = (logits[rows, il].argmax(-1) == y[rows, il]).float().cpu()
        acc_nat.append(corr)
        n_exp_all.append(model._last_n_expected.float().cpu())
        Ls_all.append(Ls)
        d_all.append(d_b)
        for n in FORCED_GRID:
            model.value_ctx = None
            if model.cfg.use_hbp:
                model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
            lg, _ = model(x, forced_steps=torch.full(
                (x.shape[0],), n, device=device))
            cf = (lg[rows, il].argmax(-1) == y[rows, il]).float().cpu()
            for L in L_SET:
                forced[n][L].append(cf[Ls == L])
    corr_nat = torch.cat(acc_nat)
    n_exp = torch.cat(n_exp_all)
    Ls_v = torch.cat(Ls_all)
    d_v = torch.cat(d_all)
    curvaL = {L: [float(torch.cat(forced[n][L]).mean())
                  for n in FORCED_GRID] for L in L_SET}
    curva = [float(np.mean([curvaL[L][j] for L in L_SET]))
             for j in range(len(FORCED_GRID))]
    i2, i8, i24 = (FORCED_GRID.index(n) for n in (2, 8, 24))

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        s = np.corrcoef(ra, rb)[0, 1]
        return float(s) if np.isfinite(s) else 0.0
    # LL: Spearman(n̄, d | L) en instancias correctas nativas, media por L
    sp_L = {}
    for L in L_SET:
        m = (Ls_v == L) & (corr_nat > 0)
        sp_L[L] = (spearman(n_exp[m].numpy(), d_v[m].numpy())
                   if int(m.sum()) > 30 else 0.0)
    sp_med = float(np.mean(list(sp_L.values())))
    E_nL = {L: float(n_exp[Ls_v == L].mean()) for L in L_SET}
    res = {"acc_nat": float(corr_nat.mean()),
           "acc_nat_por_L": {str(L): float(corr_nat[Ls_v == L].mean())
                             for L in L_SET},
           "E_n": float(n_exp.mean()),
           "E_n_por_L": {str(L): E_nL[L] for L in L_SET},
           "acc_f24": curva[i24],
           "acc_f24_L18": curvaL[18][i24],
           "curva_forzada": dict(zip(map(str, FORCED_GRID), curva)),
           "curva_por_L": {str(L): curvaL[L] for L in L_SET},
           "pendiente": curva[i24] - curva[i2],
           "pendiente_L18": curvaL[18][i24] - curvaL[18][i2],
           "profunda_L18": curvaL[18][i24] - curvaL[18][i8],
           "spearman_n_d_por_L": {str(L): sp_L[L] for L in L_SET},
           "spearman_n_d": sp_med}
    res["gate_A"] = bool(res["acc_f24"] >= 0.60
                         and res["acc_f24_L18"] >= 0.40)
    res["gate_C"] = bool(res["acc_f24"] - res["acc_nat"] >= 0.05
                         and res["E_n"] <= 20)
    res["gate_D"] = bool(res["pendiente"] >= 0.15
                         and res["pendiente_L18"] >= 0.15
                         and E_nL[18] - E_nL[6] >= 2
                         and res["profunda_L18"] >= 0.05)
    res["gate_LL"] = bool(sp_med >= 0.3)
    return res


def train_cell(seed, run, steps, spec, device, dtype):
    ds = N1bDatasetDenso(spec, seed=seed * 10 + run)
    model, cfg = build_n1b_model(ds.vocab_size, device, dtype, seed, steps)
    # PREREG v2 §9.5: warmup sin presión de frugalidad (β_halt=0 el
    # primer 25% de pasos; luego el valor de config)
    beta_halt_final = float(model.cfg.beta_halt)
    model.cfg.beta_halt = 0.0
    warmup_halt = steps // 4
    # probe de parámetros activos (CE plana, sin V̂)
    x, y, _, ss = ds.batch(2)
    x, y = x.to(device), y.to(device)
    w = torch.full_like(ss, float(ss.mean())).to(device)
    model.value_ctx = {"stake_true": w}
    if model.cfg.use_hbp:
        model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
    _, pl = model(x, y, sample_weights=w)
    pl["total"].backward()
    act = [p for p in model.parameters()
           if p.requires_grad and p.grad is not None]
    model.zero_grad()
    opt = torch.optim.AdamW(act, lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))
    model.train()
    for step in range(1, steps + 1):
        if step == warmup_halt + 1:
            model.cfg.beta_halt = beta_halt_final
        opt.param_groups[0]["lr"] = get_lr(step, cfg)
        x, y, _, ss = ds.batch(cfg.batch_size)
        x, y = x.to(device), y.to(device)
        w = torch.full_like(ss, float(ss.mean())).to(device)
        model.value_ctx = {"stake_true": w}
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        _, loss_dict = model(x, y, sample_weights=w)
        opt.zero_grad()
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(act, cfg.grad_clip)
        opt.step()
        if step % 500 == 0:
            print(f"  s{seed}r{run} paso {step}: "
                  f"loss={float(loss_dict['total']):.3f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    spec = N1bSpec()
    device, dtype = select_device_dtype()
    steps = 60 if args.smoke else args.steps
    t0 = time.time()
    for seed in args.seeds:
        for run in range(args.runs):
            name = f"n1b_blind_flat{args.tag}_s{seed}_r{run}"
            model = train_cell(seed, run, steps, spec, device, dtype)
            res = eval_gates(model, spec, device, dtype)
            res["seed"], res["run"], res["steps"] = seed, run, steps
            torch.save(model.state_dict(), os.path.join(CKPT, f"{name}.pt"))
            with open(os.path.join(RES, f"{name}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            print(f"{name}: f24={res['acc_f24']:.3f} "
                  f"(L18 {res['acc_f24_L18']:.3f}) nat={res['acc_nat']:.3f} "
                  f"E[n]={res['E_n']:.1f} "
                  f"E[n]L6/L18={res['E_n_por_L']['6']:.1f}/"
                  f"{res['E_n_por_L']['18']:.1f} "
                  f"pend={res['pendiente']:+.3f}/{res['pendiente_L18']:+.3f} "
                  f"prof={res['profunda_L18']:+.3f} "
                  f"ρ(n,d)={res['spearman_n_d']:+.2f} "
                  f"A={'✓' if res['gate_A'] else '✗'}"
                  f"C={'✓' if res['gate_C'] else '✗'}"
                  f"D={'✓' if res['gate_D'] else '✗'}"
                  f"LL={'✓' if res['gate_LL'] else '✗'} "
                  f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
