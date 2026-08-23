"""
N1b — CAZA DE BUGS (tras el diag 1-hop: loss congelada EXACTA en
ln(20)+0.006 = logits independientes del input; el modelo emite la
uniforme sobre elementos ignorando la entrada).

Tres controles que localizan el fallo:
  copy/n2_endo   : label = el propio token s (posición 3). Si NI COPIAR
                   se aprende, el bug está en el flujo input→readout del
                   pipeline con esta secuencia/vocab.
  1hop/vanilla   : retrieval sin reasoner (backbone puro). Si vanilla SÍ
                   aprende, el bucle recurrente lava la información en
                   esta config (secuencia corta: solo 2 posiciones de
                   scratchpad vs ~100 en S₅).
  copy/vanilla   : el control más básico de todos.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n1b_diag_bugs
"""
import time

import torch

from training.config import TrainConfig
from training.trainer import build_model, get_lr, select_device_dtype
from .n1b_env import (ARROW_IDX, ELEM_OFF, SEQ_LEN, N1bDataset, N1bSpec,
                      sigma_de_pares)

STEPS = 1500


class DsProbe(N1bDataset):
    def __init__(self, spec, seed, modo, split="train"):
        super().__init__(spec, seed, split)
        self.modo = modo

    def batch(self, batch_size):
        X, Y, Ls, Ss = super().batch(batch_size)
        for b in range(X.shape[0]):
            if self.modo == "copy":
                Y[b, ARROW_IDX] = X[b, 3]                  # copiar s
            elif self.modo == "1hop":
                sigma = sigma_de_pares(X[b])
                s = int(X[b, 3]) - ELEM_OFF
                Y[b, ARROW_IDX] = ELEM_OFF + int(sigma[s])
            else:                                          # 1hop_ady
                # query ADYACENTE: s repetido tras ARROW; label en esa
                # posición = σ(s) — el patrón inductivo clásico
                # («predice el sucesor de MI token»), sin indirección.
                sigma = sigma_de_pares(X[b])
                s = int(X[b, 3]) - ELEM_OFF
                Y[b, ARROW_IDX] = -1
                X[b, ARROW_IDX + 1] = ELEM_OFF + s
                Y[b, ARROW_IDX + 1] = ELEM_OFF + int(sigma[s])
        return X, Y, Ls, Ss


def run_cell(modo, variant, device, dtype):
    spec = N1bSpec()
    ds = DsProbe(spec, seed=0, modo=modo)
    torch.manual_seed(0)
    cfg = TrainConfig(task="permcomp", variant=variant, seed=0,
                      max_halt_steps=8, max_steps=STEPS)
    kw = {}
    if variant != "vanilla":
        kw = dict(halt_mod_gain=cfg.halt_mod_gain, max_halt_steps=8,
                  wm_in_loop=cfg.wm_in_loop,
                  ponder_expected_loss=cfg.ponder_expected_loss)
    model = build_model(variant, ds.vocab_size, SEQ_LEN,
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
    model.eval()
    ds_e = DsProbe(spec, seed=999, modo=modo, split="eval")
    hits = []
    with torch.no_grad():
        for _ in range(4):
            x, y, _, ss = ds_e.batch(256)
            x, y = x.to(device), y.to(device)
            model.value_ctx = {"stake_true": ss.to(device)}
            if model.cfg.use_hbp:
                model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
            lg, _ = model(x)
            il = (y != -1).float().cumsum(1).argmax(1)
            r = torch.arange(x.shape[0], device=device)
            hits.append((lg[r, il].argmax(-1) == y[r, il]).float().cpu())
    return float(torch.cat(hits).mean())


def main():
    import sys
    device, dtype = select_device_dtype()
    t0 = time.time()
    celdas = (("copy", "vanilla"), ("copy", "n2_endo"),
              ("1hop", "vanilla"), ("1hop", "n2_endo"))
    if len(sys.argv) > 1 and sys.argv[1] == "ady":
        celdas = (("1hop_ady", "vanilla"), ("1hop_ady", "n2_endo"))
    for modo, variant in celdas:
        acc = run_cell(modo, variant, device, dtype)
        print(f"{modo:>8} / {variant:<8}: acc={acc:.3f} "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)


if __name__ == "__main__":
    main()
