"""
N1b — CUERNO 1 del dilema: σ FIJA (tabla en pesos, como la Cayley de S₅).

Predicción a verificar: con σ compartida por todas las instancias, la
familia se APRENDE (la tabla entra en pesos) pero el atajo espectral
(embedding = posición-en-ciclo) resuelve d en UNA pasada → acc(n) PLANA
en n → el acantilado de cómputo muere. Si en cambio acc(n) crece con n
(el modelo camina de verdad), la familia tiene rescate con σ fijas.

σ fija con ciclos de longitud 6 y 12 (18 de los 20 elementos; 2 puntos
fijos); varían s, t (⇒ d) y los stakes. Los pares se presentan en
contexto igualmente (formato idéntico; con σ fija, los pesos ganan).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n1b_diag_fixed
"""
import time

import numpy as np
import torch

from training.config import TrainConfig
from training.trainer import build_model, get_lr, select_device_dtype
from .n1b_env import (ARROW_IDX, DIST_OFF, ELEM_OFF, LCLASS_OFF, L_SET,
                      N_ELEM, PAD, ARROW, SEQ_LEN, SLOT_OFF, N1bDataset,
                      N1bSpec)

STEPS = 5000
FORCED = [2, 8, 24]
N_MAX = 24

# σ fija (seed 424242): ciclos de 6 y 12 sobre 18 elementos + 2 fijos
_rng = np.random.default_rng(424242)
_els = _rng.permutation(N_ELEM)
CICLOS = {6: _els[:6], 12: _els[6:18]}
SIGMA_FIJA = np.arange(N_ELEM)
for _L, _c in CICLOS.items():
    for _j in range(_L):
        SIGMA_FIJA[_c[_j]] = _c[(_j + 1) % _L]


class DsFija(N1bDataset):
    def _instancia(self):
        rt = self.rng_task
        L = (6, 12)[int(rt.integers(0, 2))]
        d = int(rt.integers(1, L))
        ciclo = CICLOS[L]
        i0 = int(rt.integers(0, L))
        s = int(ciclo[i0])
        t = int(ciclo[(i0 + d) % L])
        return SIGMA_FIJA.copy(), s, t, L, d


def main():
    device, dtype = select_device_dtype()
    spec = N1bSpec()
    ds = DsFija(spec, seed=0)
    torch.manual_seed(0)
    cfg = TrainConfig(task="permcomp", variant="n2_endo", seed=0,
                      max_halt_steps=N_MAX, max_steps=STEPS)
    model = build_model("n2_endo", ds.vocab_size, SEQ_LEN,
                        halt_mod_gain=cfg.halt_mod_gain,
                        max_halt_steps=N_MAX, wm_in_loop=cfg.wm_in_loop,
                        ponder_expected_loss=cfg.ponder_expected_loss
                        ).to(device=device, dtype=dtype)
    model.cfg.beta_val = 0.0
    with torch.no_grad():
        model.halting.halt_proj.bias.fill_(-3.0)
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
        if step % 1000 == 0:
            print(f"paso {step}: loss={float(ld['total'].detach()):.3f} "
                  f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
    # eval: nativa + curva forzada por L
    model.eval()
    ds_e = DsFija(spec, seed=999, split="eval")
    res = {("nat", L): [] for L in (6, 12)}
    for n in FORCED:
        for L in (6, 12):
            res[(n, L)] = []
    n_exp = []
    with torch.no_grad():
        for _ in range(8):
            x, y, Ls, ss = ds_e.batch(256)
            x, y = x.to(device), y.to(device)
            il = (y != -1).float().cumsum(1).argmax(1)
            r = torch.arange(x.shape[0], device=device)
            model.value_ctx = {"stake_true": ss.to(device)}
            if model.cfg.use_hbp:
                model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
            lg, _ = model(x)
            c = (lg[r, il].argmax(-1) == y[r, il]).float().cpu()
            n_exp.append(model._last_n_expected.float().cpu())
            for L in (6, 12):
                res[("nat", L)].append(c[Ls == L])
            for n in FORCED:
                model.value_ctx = None
                if model.cfg.use_hbp:
                    model.hbp.reset_state(x.shape[0], device=device,
                                          dtype=dtype)
                lgf, _ = model(x, forced_steps=torch.full(
                    (x.shape[0],), n, device=device))
                cf = (lgf[r, il].argmax(-1) == y[r, il]).float().cpu()
                for L in (6, 12):
                    res[(n, L)].append(cf[Ls == L])
    acc = {k: float(torch.cat(v).mean()) for k, v in res.items()}
    print(f"\nσ-FIJA: acc nativa L6={acc[('nat', 6)]:.3f} "
          f"L12={acc[('nat', 12)]:.3f} E[n]={float(torch.cat(n_exp).mean()):.1f}")
    for L in (6, 12):
        curva = " ".join(f"n{n}:{acc[(n, L)]:.3f}" for n in FORCED)
        print(f"  curva forzada L{L}: {curva}", flush=True)
    plana = all(acc[(24, L)] - acc[(2, L)] < 0.10 for L in (6, 12))
    aprendida = acc[("nat", 6)] > 0.7
    print(f"VEREDICTO cuerno 1: "
          f"{'APRENDIDA' if aprendida else 'NO aprendida'} + curva "
          f"{'PLANA (atajo en pesos: el acantilado muere)' if plana else 'CRECIENTE (camina: posible rescate)'}",
          flush=True)


if __name__ == "__main__":
    main()
