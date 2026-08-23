"""
SEPARADOR DE LECTURA (mixture-readout) del control yoked.

Motivo: `AdaptiveHalting.forward` devuelve weighted_state = Σ_n p_n·x_n
(MEZCLA PonderNet de estados ocultos), mientras `forward_forced` devuelve
el estado ÚNICO tras forced_steps ("esta ruta no forma una mezcla
PonderNet", su propio docstring). Por tanto el brazo nativo de n3_yoked
lee de una mezcla y los brazos yoked/exante de un estado único: el
+0.383 atribuido a "régimen" incluye una cantidad desconocida de
ENSEMBLING IMPLÍCITO DE PROFUNDIDADES más un desajuste de lectura
train/test. Este script lo descompone.

Diseño (todo post-hoc sobre LA MISMA trayectoria nativa, sin reentrenar):
  con record_step_states=True guardamos {x_n} y {p_n} del rollout nativo y
  reconstruimos varias lecturas sobre los MISMOS estados:
    nativo_mix    Σ_n p_n·x_n            (= el brazo nativo publicado)
    nativo_recon  idéntico, recalculado  → CONTROL POSITIVO del instrumento
    nativo_nhat   x_{n̂}, n̂=round(E[n])   → misma ejecución, lectura única
    nativo_moda   x_{argmax_n p_n}       → variante de lectura única
  y re-ejecutamos los brazos originales yoked (forzado a n̂) y exante5.

Descomposición del total (nativo_mix − exante5):
    lectura   = nativo_mix  − nativo_nhat   (artefacto puro de lectura)
    regimen   = nativo_nhat − yoked         (régimen residual, ambos únicos)
    informac. = yoked       − exante5       (información posterior portada)
Los tres suman el total por construcción; se reportan con IC-t sobre los
12 checkpoints congelados.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n3_readout
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
N_MAX = 24

# Brazos que produce el script (orden de reporte)
BRAZOS = ["nativo_norec", "nativo_mix", "nativo_recon", "nativo_nhat",
          "nativo_moda", "yoked", "exante5"]


@torch.no_grad()
def leer(model, estado):
    """Aplica la cabeza de lenguaje tal cual la aplica miura.forward."""
    return model.lm_head(model.norm_f(estado))


@torch.no_grad()
def eval_ckpt(model, spec, device, dtype):
    ds = N2Dataset(spec, seed=SEED_EVAL, split="eval")
    pagos = {k: [] for k in BRAZOS}
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

        # --- (0) nativo SIN grabar estados: el brazo publicado, intacto.
        # Control de que grabar estados (que desactiva el corte temprano del
        # bucle) no cambia el payoff nativo.
        model.record_step_states = False
        model.value_ctx = {"stake_true": ss.to(device)}
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        lg, _ = model(x)
        pagos["nativo_norec"].append(pago(lg))

        # --- (1) nativo GRABANDO estados: misma configuración, y de aquí
        # salen todas las lecturas alternativas.
        model.record_step_states = True
        model.value_ctx = {"stake_true": ss.to(device)}
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        lg_mix, _ = model(x)
        pagos["nativo_mix"].append(pago(lg_mix))
        model.record_step_states = False

        estados = model._last_step_states          # lista N de (B,T,D)
        probs = model._last_halt_probs_live        # lista N de (B,)
        n_i = model._last_n_expected.float().cpu().numpy()
        n_nat.append(n_i)

        P = torch.stack(probs, dim=1).float()      # (B,N)
        # (2) reconstrucción de la mezcla — control positivo del instrumento
        recon = torch.zeros_like(estados[0])
        for j, s in enumerate(estados):
            recon = recon + P[:, j].to(s.dtype).view(-1, 1, 1) * s
        pagos["nativo_recon"].append(pago(leer(model, recon)))

        # (3) lectura de estado ÚNICO a la profundidad elegida n̂
        nhat = torch.from_numpy(
            np.clip(np.round(n_i), 1, N_MAX).astype(np.int64)).to(device)
        sel = torch.empty_like(estados[0])
        for n in range(1, len(estados) + 1):
            m = (nhat == n)
            if m.any():
                sel[m] = estados[n - 1][m]
        pagos["nativo_nhat"].append(pago(leer(model, sel)))

        # (4) lectura de estado único en la MODA de p_n
        moda = P.argmax(dim=1)                      # (B,) índice 0-based
        sel2 = torch.empty_like(estados[0])
        for n in range(len(estados)):
            m = (moda == n)
            if m.any():
                sel2[m] = estados[n][m]
        pagos["nativo_moda"].append(pago(leer(model, sel2)))

        model._last_step_states = None
        model._last_halt_probs_live = None
        del estados, probs, recon, sel, sel2

        # --- (5) yoked: forzado a n̂ por instancia (brazo publicado)
        n_forced = np.clip(np.round(n_i), 1, N_MAX).astype(int)
        model.value_ctx = None
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        lg2, _ = model(x, forced_steps=torch.tensor(n_forced, device=device))
        pagos["yoked"].append(pago(lg2))

        # --- (6) exante uniforme e=5 (brazo publicado)
        model.value_ctx = None
        if model.cfg.use_hbp:
            model.hbp.reset_state(x.shape[0], device=device, dtype=dtype)
        lg3, _ = model(x, forced_steps=torch.full((x.shape[0],), 5,
                                                  device=device))
        pagos["exante5"].append(pago(lg3))

    res = {k: float(np.concatenate(v).sum()) for k, v in pagos.items()}
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
        print(f"{name}: mix={r['nativo_mix']:.3f} "
              f"(norec={r['nativo_norec']:.3f} recon={r['nativo_recon']:.3f}) "
              f"nhat={r['nativo_nhat']:.3f} moda={r['nativo_moda']:.3f} "
              f"yoked={r['yoked']:.3f} ex5={r['exante5']:.3f} "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = {"por_ckpt": filas, "agg": {}}

    def ic(a, b, nombre):
        d = np.array([f[a] - f[b] for f in filas])
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = m - 2.201 * se, m + 2.201 * se
        out["agg"][nombre] = {"a": a, "b": b, "delta": float(m),
                              "se": float(se), "ic95": [float(lo), float(hi)],
                              "excluye_cero": bool(lo > 0 or hi < 0)}
        print(f"  {nombre:22s} {a:12s}−{b:12s} {m:+.4f} "
              f"IC95 [{lo:+.4f},{hi:+.4f}]"
              f"{'  *' if (lo > 0 or hi < 0) else '   n.s.'}", flush=True)

    print("\n=== CONTROLES DEL INSTRUMENTO (deben ser ~0) ===")
    ic("nativo_mix", "nativo_norec", "grabar_estados")
    ic("nativo_recon", "nativo_mix", "reconstruccion")

    print("\n=== DESCOMPOSICION DEL TOTAL ===")
    ic("nativo_mix", "exante5", "total")
    ic("nativo_mix", "nativo_nhat", "A_lectura")
    ic("nativo_nhat", "yoked", "B_regimen_resid")
    ic("yoked", "exante5", "C_informacion")
    ic("nativo_mix", "nativo_moda", "A2_lectura_moda")

    a = out["agg"]["A_lectura"]["delta"]
    b = out["agg"]["B_regimen_resid"]["delta"]
    c = out["agg"]["C_informacion"]["delta"]
    tot = out["agg"]["total"]["delta"]
    print(f"\n  suma A+B+C = {a + b + c:+.4f}  vs total = {tot:+.4f} "
          f"(residuo {a + b + c - tot:+.1e})")
    out["suma_ABC"] = float(a + b + c)

    with open(os.path.join(RES, "n3_readout.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Guardado n3_readout.json", flush=True)


if __name__ == "__main__":
    main()
