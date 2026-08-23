"""
N2 — Runner de entrenamiento (PREREG_N2 v2 §4/§6-VG4).

Protocolo v3 (2500 pasos, N_MAX=24, pin_fp32) sobre el entorno N2 (motivo de
igualdad, K∈[13,24], p_hi/s_alto del spec). La CE va PONDERADA por el peso
del brazo (weights_for_arm: la estructura de pago del entorno); value_ctx
lleva el stake VERDADERO (solo consecuencia realizada para L_val — y canal
ex-ante en oracle). Eval final sobre SET COMÚN CONGELADO (m=4096, seed 999):
payoff_norm, acc, acc_alto/bajo, E[n̄], corr parcial(E[n], stake | K), AUC(V̂).

Réplicas (F3a-R): seed = init del modelo; run ∈ {0,1} = stream de datos
propio (N2Dataset seed*10+run) + no-determinismo de kernels.

  python -m mhbp.tasks.reasoner_g0.n2_train --arm endo --seeds 0 1 2 --runs 2
  (--smoke: 1 celda × 60 pasos)
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from training.config import TrainConfig
from training.trainer import build_model, get_lr, select_device_dtype
from .n2_env import N2Dataset, N2Spec, weights_for_arm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "ckpts")
MAX_STEPS = 2500
N_MAX = 24
EVAL_SEED = 999
EVAL_M = 4096

VARIANT = {"endo": "n2_endo", "endo_noval": "n2_endo_noval",
           "blind_flat": "n2_endo", "blind_shuffle": "n2_endo",
           "oracle": "n2_oracle", "gating_endo": "n2_gating_endo"}


def partial_corr(a, b, z):
    """corr(a, b | z) residualizando linealmente sobre z."""
    def resid(v):
        zc = np.stack([z, np.ones_like(z)], 1)
        beta, *_ = np.linalg.lstsq(zc, v, rcond=None)
        return v - zc @ beta
    ra, rb = resid(a.astype(float)), resid(b.astype(float))
    d = (np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / d) if d > 0 else 0.0


@torch.no_grad()
def eval_cell(model, spec, arm, device, dtype):
    ds = N2Dataset(spec, seed=EVAL_SEED, split="eval")
    model.eval()
    outs = {"correct": [], "stake": [], "K": [], "n_exp": [], "s_hat": []}
    for i in range(0, EVAL_M, 256):
        x, y, ks, ss = ds.batch(min(256, EVAL_M - i))
        x, y = x.to(device), y.to(device)
        model.value_ctx = {"stake_true": ss.to(device)}
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        logits, _ = model(x)
        valid = (y != -1)
        idx_last = valid.float().cumsum(1).argmax(1)
        rows = torch.arange(x.shape[0], device=device)
        pred = logits[rows, idx_last].argmax(-1)
        outs["correct"].append((pred == y[rows, idx_last]).float().cpu())
        outs["stake"].append(ss)
        outs["K"].append(ks.float())
        outs["n_exp"].append(model._last_n_expected.float().cpu())
        if model._val_cache is not None:
            outs["s_hat"].append(model._val_cache["s"].cpu())
    c = torch.cat(outs["correct"]).numpy()
    s = torch.cat(outs["stake"]).numpy()
    k = torch.cat(outs["K"]).numpy()
    n = torch.cat(outs["n_exp"]).numpy()
    hi = s > 1
    res = {
        "payoff_norm": float((s * c).sum() / s.sum()),
        "acc": float(c.mean()),
        "acc_alto": float(c[hi].mean()) if hi.any() else None,
        "acc_bajo": float(c[~hi].mean()),
        "E_n": float(n.mean()),
        "E_n_alto": float(n[hi].mean()) if hi.any() else None,
        "E_n_bajo": float(n[~hi].mean()),
        "corr_n_stake_dado_K": partial_corr(n, s, k),
    }
    if outs["s_hat"]:
        sh = torch.cat(outs["s_hat"]).numpy()
        pos, neg = sh[hi], sh[~hi]
        auc = float((pos[:, None] > neg[None, :]).mean()
                    + 0.5 * (pos[:, None] == neg[None, :]).mean())
        res["auc_vhat_stake"] = auc
    return res


def train_cell(arm, seed, run, spec, steps, device, dtype,
               beta_halt=None):
    torch.manual_seed(seed)                    # init del modelo = seed
    ds = N2Dataset(spec, seed=seed * 10 + run)  # stream propio = run
    cfg = TrainConfig(task="permcomp", perm_gens="cycle_transp",
                      variant=VARIANT[arm], max_steps=steps, seed=seed,
                      max_halt_steps=N_MAX)
    model = build_model(cfg.variant, ds.vocab_size, cfg.seq_len,
                        halt_mod_gain=cfg.halt_mod_gain,
                        max_halt_steps=N_MAX, wm_in_loop=cfg.wm_in_loop,
                        ponder_expected_loss=cfg.ponder_expected_loss
                        ).to(device=device, dtype=dtype)
    if beta_halt is not None:
        # dial de ESCASEZ (VG4: E[n̄] con β=0.01 aterriza en ~11.5; la
        # ventana VG1 exige 8-10 → β se sube y VG4b verifica el aterrizaje)
        model.cfg.beta_halt = float(beta_halt)
    if model.cfg.use_hbp:
        model.hbp.pin_fp32()
    rng_w = np.random.default_rng(seed * 100 + run + 7)   # barajado (shuffle)
    # probe de parámetros activos (incluye V̂ vía L_val con value_ctx)
    x, y, ks, ss = ds.batch(2)
    x, y = x.to(device), y.to(device)
    w = weights_for_arm(arm, ss, rng_w).to(device)
    # las CONSECUENCIAS del brazo son SUS pagos: L_val (y el canal del
    # oracle) ven w, no el stake verdadero — en blind_shuffle el entorno
    # paga barajado (si V̂ aprendiera el stake real, sería FUGA: el chequeo)
    model.value_ctx = {"stake_true": w}
    if model.cfg.use_hbp:
        model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
    _, pl = model(x, y, sample_weights=w)
    pl["total"].backward()
    # solo las CABEZAS V̂ van al replay; value_halt_a (gating_endo) es un
    # actuador entrenado por la CE y se queda en el optimizador principal
    val_ids = {id(p) for name, p in model.named_parameters()
               if name.startswith(("value_p", "value_s"))}
    active = [p for p in model.parameters()
              if p.requires_grad and p.grad is not None]
    act_main = [p for p in active if id(p) not in val_ids]
    act_val = [p for p in active if id(p) in val_ids]
    model.zero_grad()
    # VG4-fix 3 (2026-08-04): V̂ se entrena por REPLAY ESTRATIFICADO fuera
    # del grafo principal («rumiar las consecuencias»: buffer episódico,
    # re-muestreo 50/50 por clase de payoff, k updates/paso, optimizador y
    # lr propios). El online single-pass con 10% de positivos no engancha la
    # XOR (piloto: AUC 0.5-0.66); con replay el diagnóstico da 1.000 en 600
    # pasos. L_val en-grafo queda desactivada (beta_val=0) en N2.
    model.cfg.beta_val = 0.0
    opt = torch.optim.AdamW(act_main, lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    opt_val = (torch.optim.Adam(act_val, lr=1e-3) if act_val else None)
    buf_f, buf_p, buf_c = [], [], []
    model.train()
    for step in range(1, steps + 1):
        lr = get_lr(step, cfg)
        opt.param_groups[0]["lr"] = lr           # el grupo de V̂ mantiene 1e-3
        x, y, ks, ss = ds.batch(cfg.batch_size)
        x, y = x.to(device), y.to(device)
        w = weights_for_arm(arm, ss, rng_w).to(device)
        model.value_ctx = {"stake_true": w}      # consecuencias = pagos del brazo
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        logits, loss_dict = model(x, y, sample_weights=w)
        opt.zero_grad()
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(act_main, cfg.grad_clip)
        opt.step()

        # --- replay de V̂ («rumiar las consecuencias», VG4-fix 3) --- #
        feat = getattr(model, "_val_feat", None)
        if opt_val is not None and feat is not None:
            with torch.no_grad():
                valid = (y != -1)
                il = valid.float().cumsum(1).argmax(1)
                rows = torch.arange(x.shape[0], device=device)
                pred = logits[rows, il].argmax(-1)
                corr_b = (pred == y[rows, il]).float()
                pay_b = w.float() * corr_b       # el pago DEL BRAZO
            buf_f.append(feat.float())
            buf_p.append(pay_b)
            buf_c.append(corr_b)
            if len(buf_f) > 256:
                buf_f.pop(0)
                buf_p.pop(0)
                buf_c.pop(0)
            F_all = torch.cat(buf_f)
            P_all = torch.cat(buf_p)
            C_all = torch.cat(buf_c)
            idx_hi = (P_all > 1.5).nonzero().squeeze(-1)
            idx_lo = (P_all <= 1.5).nonzero().squeeze(-1)
            if len(idx_hi) >= 8 and len(idx_lo) >= 8:
                for _ in range(4):
                    sel = torch.cat([
                        idx_hi[torch.randint(0, len(idx_hi), (16,),
                                             device=device)],
                        idx_lo[torch.randint(0, len(idx_lo), (16,),
                                             device=device)]])
                    f_s = F_all[sel]
                    wdt_v = model.value_p[0].weight.dtype
                    p_hat = torch.sigmoid(
                        model.value_p(f_s.to(wdt_v))).squeeze(-1).float()
                    pay_hat = torch.nn.functional.softplus(
                        model.value_s(f_s.to(wdt_v))).squeeze(-1).float()
                    l_v = (torch.nn.functional.binary_cross_entropy(
                        p_hat.clamp(1e-4, 1 - 1e-4), C_all[sel])
                        + torch.nn.functional.mse_loss(pay_hat, P_all[sel]))
                    opt_val.zero_grad()
                    l_v.backward()
                    opt_val.step()
            model._val_feat = None
    res = eval_cell(model, spec, arm, device, dtype)
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(VARIANT))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="vg4")
    ap.add_argument("--beta_halt", type=float, default=None)
    ap.add_argument("--p_hi", type=float, default=None)
    ap.add_argument("--regime", choices=("indist", "ood"), default="indist")
    args = ap.parse_args()
    import dataclasses
    spec = N2Spec()
    if args.p_hi is not None:
        spec = dataclasses.replace(spec, p_hi=args.p_hi)
    if args.regime == "ood":
        # celdas para la batería M3 (protocolo F3a: entrenar corto, la
        # ligadura se mide extrapolando en longitud); stakes/slots activos
        spec = dataclasses.replace(spec, k_lo=2, k_hi=12)
    device, dtype = select_device_dtype()
    seeds = args.seeds[:1] if args.smoke else args.seeds
    runs = 1 if args.smoke else args.runs
    steps = 60 if args.smoke else args.steps
    t0 = time.time()
    for seed in seeds:
        for run in range(runs):
            cid = f"n2_{args.tag}_{args.arm}_s{seed}_r{run}" + \
                ("_smoke" if args.smoke else "")
            jpath = os.path.join(RES, f"{cid}.json")
            cpath = os.path.join(CKPT, f"{cid}.pt")
            if os.path.exists(jpath) and os.path.exists(cpath):
                print(f"skip {cid}", flush=True)
                continue
            model, res = train_cell(args.arm, seed, run, spec, steps,
                                    device, dtype, beta_halt=args.beta_halt)
            torch.save(model.state_dict(), cpath + ".tmp")
            os.replace(cpath + ".tmp", cpath)
            res["_cell"] = {"arm": args.arm, "seed": seed, "run": run,
                            "steps": steps, "spec": spec.__dict__}
            with open(jpath + ".tmp", "w", encoding="utf-8") as f:
                json.dump(res, f)
            os.replace(jpath + ".tmp", jpath)
            auc = res.get("auc_vhat_stake")
            print(f"{cid}: payoff={res['payoff_norm']:.3f} "
                  f"acc={res['acc']:.3f} E[n]={res['E_n']:.1f} "
                  f"corr|K={res['corr_n_stake_dado_K']:+.3f} "
                  f"AUC_V={auc if auc is None else round(auc, 3)} "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    print(f"n2_train[{args.arm}] COMPLETO en {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
