"""
N1b — DIAGNÓSTICO 1-HOP (no es ronda de diales: separa mecanismos antes
de gastar la ronda 3 del firewall).

Pregunta: ¿el sustrato n2_endo aprende el RETRIEVAL asociativo puro
(responder σ(s) desde los pares barajados), sin caminar ni contar?
- Si NO: hay un problema básico de cableado/config — caza de bugs, no dial.
- Si SÍ: el bloqueo del gate A es la ASIGNACIÓN DE CRÉDITO del programa
  compuesto (la CE sobre la clase d no premia el retrieval correcto) —
  y la ronda 3 debe dar crédito denso (traza supervisada), no tocar
  N_ELEM/L.

Mismo entorno n1b (pares barajados), respuesta = token de σ(s).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n1b_diag1hop
"""
import time

import numpy as np
import torch

from training.config import TrainConfig
from training.trainer import build_model, get_lr, select_device_dtype
from .n1b_env import (ARROW_IDX, ELEM_OFF, SEQ_LEN, N1bDataset, N1bSpec,
                      sigma_de_pares)

STEPS = 2500
N_MAX = 8          # el retrieval no necesita profundidad


class Ds1Hop(N1bDataset):
    """Igual que N1bDataset pero label = σ(s) (retrieval de 1 salto)."""

    def batch(self, batch_size):
        X, Y, Ls, Ss = super().batch(batch_size)
        for b in range(X.shape[0]):
            sigma = sigma_de_pares(X[b])
            s = int(X[b, 3]) - ELEM_OFF
            Y[b, ARROW_IDX] = ELEM_OFF + int(sigma[s])
        return X, Y, Ls, Ss


def main():
    device, dtype = select_device_dtype()
    spec = N1bSpec()
    ds = Ds1Hop(spec, seed=0)
    torch.manual_seed(0)
    cfg = TrainConfig(task="permcomp", variant="n2_endo", seed=0,
                      max_halt_steps=N_MAX, max_steps=STEPS)
    model = build_model("n2_endo", ds.vocab_size, SEQ_LEN,
                        halt_mod_gain=cfg.halt_mod_gain,
                        max_halt_steps=N_MAX, wm_in_loop=cfg.wm_in_loop,
                        ponder_expected_loss=cfg.ponder_expected_loss
                        ).to(device=device, dtype=dtype)
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
    t0 = time.time()
    for step in range(1, STEPS + 1):
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
        if step % 500 == 0:
            print(f"paso {step}: loss={float(ld['total'].detach()):.3f} "
                  f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
    # eval
    model.eval()
    ds_e = Ds1Hop(spec, seed=999, split="eval")
    hits = []
    with torch.no_grad():
        for _ in range(8):
            x, y, _, ss = ds_e.batch(256)
            x, y = x.to(device), y.to(device)
            model.value_ctx = {"stake_true": ss.to(device)}
            if model.cfg.use_hbp:
                model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
            lg, _ = model(x)
            il = (y != -1).float().cumsum(1).argmax(1)
            r = torch.arange(x.shape[0], device=device)
            hits.append((lg[r, il].argmax(-1) == y[r, il]).float().cpu())
    acc = float(torch.cat(hits).mean())
    print(f"\nDIAG 1-HOP: acc={acc:.3f} (azar=1/20=0.05) → "
          f"{'RETRIEVAL APRENDIDO — el bloqueo es la asignación de crédito'
             if acc >= 0.9 else
             'RETRIEVAL NO APRENDIDO — caza de bugs de cableado/config'}",
          flush=True)


if __name__ == "__main__":
    main()
