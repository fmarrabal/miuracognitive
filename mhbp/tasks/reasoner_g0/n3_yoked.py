"""
Revisión TMLR — CONTROL YOKED del +0.223 (petición central del revisor
de arquitecturas y del estadístico): descompone la ventaja del brazo
posterior nativo en INFORMACIÓN vs RÉGIMEN DE EJECUCIÓN.

Diseño: sobre los 12 solvers congelados de N3 y el eval común,
  1. rollout NATIVO por instancia → payoff nativo y n̂_i = round(E[n_i])
     (la profundidad que el halting eligió para ESA instancia);
  2. brazo YOKED: ejecución FORZADA con forced_steps = n̂_i — misma
     información por-instancia, régimen forzado. nativo − yoked =
     coste puro de régimen a profundidades emparejadas;
  3. yoked − exante(e=5 uniforme al mismo presupuesto medio) = valor de
     la INFORMACIÓN posterior bajo régimen forzado (cota inferior del
     incremento informacional).
Pareado por instancia y por solver; IC-t sobre las 12 réplicas.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n3_yoked
"""
import json
import os
import time

import numpy as np
import torch

from training.trainer import select_device_dtype
from .n2_env import N2Dataset, N2Spec
from .n3_sonda import CKPTS, load_ckpt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
M_EVAL = 4096
SEED_EVAL = 999
BATCH = 256


@torch.no_grad()
def eval_ckpt(model, spec, device, dtype):
    ds = N2Dataset(spec, seed=SEED_EVAL, split="eval")
    pagos = {"nativo": [], "yoked": [], "exante5": []}
    n_nat = []
    for i in range(0, M_EVAL, BATCH):
        x, y, ks, ss = ds.batch(min(BATCH, M_EVAL - i))
        x, y = x.to(device), y.to(device)
        valid = (y != -1)
        il = valid.float().cumsum(1).argmax(1)
        rows = torch.arange(x.shape[0], device=device)
        w = ss.numpy()

        def pago(logits):
            pred = logits[rows, il].argmax(-1)
            c = (pred == y[rows, il]).float().cpu().numpy()
            return w * c

        # nativo
        model.value_ctx = {"stake_true": ss.to(device)}
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        lg, _ = model(x)
        pagos["nativo"].append(pago(lg))
        n_i = model._last_n_expected.float().cpu().numpy()
        n_nat.append(n_i)
        n_forced = np.clip(np.round(n_i), 1, 24).astype(int)

        # yoked: forzado a las profundidades nativas por instancia
        model.value_ctx = None
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        lg2, _ = model(x, forced_steps=torch.tensor(n_forced, device=device))
        pagos["yoked"].append(pago(lg2))

        # exante uniforme e=5 (el presupuesto de los brazos ex-ante de N3)
        model.value_ctx = None
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        lg3, _ = model(x, forced_steps=torch.full((x.shape[0],), 5,
                                                  device=device))
        pagos["exante5"].append(pago(lg3))
    tot_s = float(np.sum(np.concatenate([w for w in
                   [ds_w for ds_w in []]])) if False else 0.0)
    # payoff normalizado: Σ s·c / Σ s (re-derivamos Σs de los pagos y c)
    res = {}
    for k, v in pagos.items():
        res[k] = float(np.concatenate(v).sum())
    # Σ stakes del eval (igual para los tres brazos)
    ds2 = N2Dataset(spec, seed=SEED_EVAL, split="eval")
    s_tot = 0.0
    for i in range(0, M_EVAL, BATCH):
        _, _, _, ss = ds2.batch(min(BATCH, M_EVAL - i))
        s_tot += float(ss.sum())
    for k in res:
        res[k] /= s_tot
    res["n_nativo_medio"] = float(np.concatenate(n_nat).mean())
    return res


def main():
    device, dtype = select_device_dtype()
    spec = N2Spec(p_hi=0.15)
    t0 = time.time()
    filas = []
    for name in CKPTS:
        model = load_ckpt(name, device, dtype)
        r = eval_ckpt(model, spec, device, dtype)
        filas.append(r)
        print(f"{name}: nativo={r['nativo']:.3f} yoked={r['yoked']:.3f} "
              f"exante5={r['exante5']:.3f} n̄={r['n_nativo_medio']:.1f} "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    out = {"por_ckpt": filas, "agg": {}}
    for a, b, nombre in (("nativo", "yoked", "regimen"),
                         ("yoked", "exante5", "informacion_forzada"),
                         ("nativo", "exante5", "total")):
        d = np.array([f[a] - f[b] for f in filas])
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        out["agg"][nombre] = {"delta": float(m), "se": float(se),
                              "ic95": [float(m - 2.201 * se),
                                       float(m + 2.201 * se)]}
        print(f"Δ({a}−{b}) [{nombre}]: {m:+.4f} "
              f"IC95 [{m - 2.201 * se:+.4f}, {m + 2.201 * se:+.4f}]",
              flush=True)
    with open(os.path.join(RES, "n3_yoked.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Guardado n3_yoked.json", flush=True)


if __name__ == "__main__":
    main()
