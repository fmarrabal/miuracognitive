"""
N1b — PROBE DENSO: ¿el retrieval en contexto se aprende con supervisión
multi-query? (tras: 1-hop y 1-hop-adyacente en azar con 1 posición
supervisada y 48k ejemplos; arquitectura sobrada — 4 capas, d=256).

Secuencia extendida: [... pares σ ... ARROW, q₁..q₈] con label σ(q_j) EN
la posición de cada query (patrón «sucesor de mi token» × 8 por
secuencia = 8× señal). Si esto aprende, la ronda 3 de la familia es
supervisión densa (queries auxiliares + curriculum de saltos); si ni
esto aprende a 6k pasos, el primitivo no se forma a esta escala y la
familia se cierra por la rama declarada.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n1b_diag_dense
"""
import time

import numpy as np
import torch

from training.config import TrainConfig
from training.trainer import build_model, get_lr, select_device_dtype
from .n1b_env import ARROW_IDX, ELEM_OFF, N_ELEM, N1bDataset, N1bSpec, \
    sigma_de_pares

STEPS = 6000
N_Q = 8
SEQ_P = 56          # 46 originales + 8 queries + margen


class DsDense(N1bDataset):
    def batch(self, batch_size):
        X0, Y0, Ls, Ss = super().batch(batch_size)
        B = X0.shape[0]
        X = torch.zeros(B, SEQ_P - 1, dtype=torch.long)
        Y = torch.full((B, SEQ_P - 1), -1, dtype=torch.long)
        X[:, :X0.shape[1]] = X0
        rng = self.rng_task
        for b in range(B):
            sigma = sigma_de_pares(X0[b])
            qs = rng.permutation(N_ELEM)[:N_Q]
            for j, q in enumerate(qs):
                pos = ARROW_IDX + 1 + j
                X[b, pos] = ELEM_OFF + int(q)
                Y[b, pos] = ELEM_OFF + int(sigma[q])
            Y[b, ARROW_IDX] = -1
        return X, Y, Ls, Ss


def run_cell(variant, device, dtype, steps=STEPS):
    spec = N1bSpec()
    ds = DsDense(spec, seed=0)
    torch.manual_seed(0)
    cfg = TrainConfig(task="permcomp", variant=variant, seed=0,
                      max_halt_steps=8, max_steps=steps)
    if steps > 10000:
        cfg.batch_size = 128            # grokking patience: 16× cómputo
    kw = {}
    if variant != "vanilla":
        kw = dict(halt_mod_gain=cfg.halt_mod_gain, max_halt_steps=8,
                  wm_in_loop=cfg.wm_in_loop,
                  ponder_expected_loss=cfg.ponder_expected_loss)
    model = build_model(variant, ds.vocab_size, SEQ_P,
                        **kw).to(device=device, dtype=dtype)
    model.cfg.beta_val = 0.0
    if model.cfg.use_hbp:
        model.hbp.pin_fp32()
    x, y, _, ss = ds.batch(2)
    x, y = x.to(device), y.to(device)
    w = torch.full_like(ss, 1.0).to(device)
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
        opt.param_groups[0]["lr"] = get_lr(step, cfg)
        x, y, _, ss = ds.batch(cfg.batch_size)
        x, y = x.to(device), y.to(device)
        w = torch.full_like(ss, 1.0).to(device)
        model.value_ctx = {"stake_true": w}
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        _, ld = model(x, y, sample_weights=w)
        opt.zero_grad()
        ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(act, cfg.grad_clip)
        opt.step()
        if step % 1500 == 0:
            print(f"  {variant} paso {step}: "
                  f"loss={float(ld['total'].detach()):.3f}", flush=True)
    model.eval()
    ds_e = DsDense(spec, seed=999, split="eval")
    hits = []
    with torch.no_grad():
        for _ in range(4):
            x, y, _, ss = ds_e.batch(256)
            x, y = x.to(device), y.to(device)
            model.value_ctx = {"stake_true": ss.to(device)}
            if model.cfg.use_hbp:
                model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
            lg, _ = model(x)
            m = (y != -1)
            hits.append((lg.argmax(-1)[m] == y[m]).float().cpu())
    return float(torch.cat(hits).mean())


def main():
    import sys
    device, dtype = select_device_dtype()
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "grok":
        # última bala: 30k pasos × batch 128 (grokking patience)
        spec_steps = 30000
        import training.config as tc
        acc = run_cell("vanilla", device, dtype, steps=spec_steps)
        print(f"denso8-grok / vanilla: acc={acc:.3f} (azar 0.05) "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
        return
    for variant in ("vanilla", "n2_endo"):
        acc = run_cell(variant, device, dtype)
        print(f"denso8 / {variant:<8}: acc={acc:.3f} (azar 0.05) "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)


if __name__ == "__main__":
    main()
