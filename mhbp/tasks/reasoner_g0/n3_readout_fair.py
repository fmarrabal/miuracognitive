"""
INSTRUMENTO CORREGIDO del yoked: ¿sabe el posterior DÓNDE poner el cómputo,
con la LECTURA MANTENIDA FIJA?

n3_readout.py demostró que el +0.383 de "régimen" es enteramente el término
de lectura (mezcla PonderNet vs estado único; residual exactamente 0). Eso
invalida la comparación nativo-vs-forzado, pero NO responde la pregunta
científica de fondo: una vez igualada la lectura, ¿la asignación posterior
por-instancia vale algo?

Diseño: todas las lecturas son MEZCLAS sobre los MISMOS estados grabados
{x_n} del rollout nativo; solo cambian los PESOS q_n.
    mix_propio  q = p_i            asignación posterior propia (= brazo nativo)
    mix_perm    q = p_{σ(i)}       distribución de OTRA instancia (derangement)
                                   → misma familia de lectura, mismo presupuesto
                                     medio por construcción, cero información
                                     sobre la instancia i
    mix_unif5   q = U{1..9}        sin información y sin forma (media 5, el
                                   presupuesto del brazo exante5)
    mix_perm2   q = p_{σ2(i)}      segunda permutación (estabilidad)

Contrastes:
    mix_propio − mix_perm   = VALOR DE LA ASIGNACIÓN POR-INSTANCIA, lectura fija
    mix_perm   − mix_unif5  = valor de la forma/marginal del presupuesto
    mix_propio − mix_unif5  = total dentro de la familia de mezclas

Nota de alcance: bajo permutación una instancia puede recibir masa en ticks
que su rollout nativo no habría "usado"; los estados existen (el rollout con
record_step_states corre los N_max ticks), así que la mezcla está bien
definida. El emparejamiento de presupuesto es EN MEDIA, igual que en el
protocolo del paper.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n3_readout_fair
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
SEED_PERM = 4242

BRAZOS = ["mix_propio", "mix_perm", "mix_perm2", "mix_unif5", "punto_nhat"]


def derangement(n, rng):
    """Permutación sin puntos fijos (ninguna instancia se recibe a sí misma)."""
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


@torch.no_grad()
def eval_ckpt(model, spec, device, dtype, rng):
    ds = N2Dataset(spec, seed=SEED_EVAL, split="eval")
    pagos = {k: [] for k in BRAZOS}
    presup = {k: [] for k in ["mix_propio", "mix_perm", "mix_perm2"]}
    diag = {k: [] for k in ["std_En", "mad_En_perm", "tv_perm"]}
    for i in range(0, M_EVAL, BATCH):
        x, y, ks, ss = ds.batch(min(BATCH, M_EVAL - i))
        x, y = x.to(device), y.to(device)
        valid = (y != -1)
        il = valid.float().cumsum(1).argmax(1)
        rows = torch.arange(x.shape[0], device=device)
        w = ss.numpy()
        B = x.shape[0]

        def pago(logits):
            pred = logits[rows, il].argmax(-1)
            c = (pred == y[rows, il]).float().cpu().numpy()
            return w * c

        def leer_mezcla(P):
            """P: (B,N) pesos. Devuelve el payoff de leer Σ_n P[:,n]·x_n."""
            m = torch.zeros_like(estados[0])
            for j, s in enumerate(estados):
                m = m + P[:, j].to(s.dtype).view(-1, 1, 1) * s
            return pago(model.lm_head(model.norm_f(m)))

        # rollout nativo grabando estados (idéntico al brazo publicado)
        model.record_step_states = True
        model.value_ctx = {"stake_true": ss.to(device)}
        if model.cfg.use_hbp:
            model.hbp.reset_state(B, device=device, dtype=dtype)
        model(x)
        model.record_step_states = False
        estados = model._last_step_states
        P = torch.stack(model._last_halt_probs_live, dim=1).float()   # (B,N)
        N = P.shape[1]
        ns = torch.arange(1, N + 1, device=device, dtype=torch.float32)

        # --- propio
        pagos["mix_propio"].append(leer_mezcla(P))
        presup["mix_propio"].append((P * ns).sum(1).cpu().numpy())

        # --- permutado (dos derangements independientes)
        for k, nombre in enumerate(("mix_perm", "mix_perm2")):
            idx = torch.from_numpy(derangement(B, rng)).to(device)
            Pp = P.index_select(0, idx)
            pagos[nombre].append(leer_mezcla(Pp))
            presup[nombre].append((Pp * ns).sum(1).cpu().numpy())
            if k == 0:
                # CONTROL POSITIVO del nulo: la permutación tiene que MOVER
                # algo. Si p_i apenas varía entre instancias, "permutar no
                # cuesta nada" sería trivial y no informativo.
                En = (P * ns).sum(1)
                Enp = (Pp * ns).sum(1)
                diag["std_En"].append(En.std(unbiased=True).item())
                diag["mad_En_perm"].append((En - Enp).abs().mean().item())
                diag["tv_perm"].append(
                    (0.5 * (P - Pp).abs().sum(1)).mean().item())

        # --- uniforme U{1..9} (media 5 = presupuesto del brazo exante5)
        Pu = torch.zeros_like(P)
        Pu[:, :9] = 1.0 / 9.0
        pagos["mix_unif5"].append(leer_mezcla(Pu))

        # --- referencia de masa puntual: estado único a n̂ (= yoked)
        nhat = torch.round((P * ns).sum(1)).long().clamp(1, N) - 1  # 0-based
        sel = torch.empty_like(estados[0])
        for n in range(N):
            m = (nhat == n)
            if m.any():
                sel[m] = estados[n][m]
        pagos["punto_nhat"].append(pago(model.lm_head(model.norm_f(sel))))

        model._last_step_states = None
        model._last_halt_probs_live = None
        del estados, P

    res = {k: float(np.concatenate(v).sum()) for k, v in pagos.items()}
    ds2 = N2Dataset(spec, seed=SEED_EVAL, split="eval")
    s_tot = 0.0
    for i in range(0, M_EVAL, BATCH):
        _, _, _, ss = ds2.batch(min(BATCH, M_EVAL - i))
        s_tot += float(ss.sum())
    for k in res:
        res[k] /= s_tot
    for k, v in presup.items():
        res["n_" + k] = float(np.concatenate(v).mean())
    for k, v in diag.items():
        res[k] = float(np.mean(v))
    return res


def main():
    device, dtype = select_device_dtype()
    spec = N2Spec(p_hi=0.15)
    rng = np.random.default_rng(SEED_PERM)
    t0 = time.time()
    filas = []
    for name in CKPTS:
        model = load_ckpt(name, device, dtype)
        r = eval_ckpt(model, spec, device, dtype, rng)
        filas.append(r)
        print(f"{name}: propio={r['mix_propio']:.3f} "
              f"perm={r['mix_perm']:.3f}/{r['mix_perm2']:.3f} "
              f"unif5={r['mix_unif5']:.3f} punto={r['punto_nhat']:.3f} "
              f"| n̄ propio={r['n_mix_propio']:.2f} perm={r['n_mix_perm']:.2f} "
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
        print(f"  {nombre:26s} {m:+.4f} IC95 [{lo:+.4f},{hi:+.4f}]"
              f"{'  *' if (lo > 0 or hi < 0) else '   n.s.'}", flush=True)

    print("\n=== CONTROL DE PRESUPUESTO (debe ser ~0) ===")
    d = np.array([f["n_mix_propio"] - f["n_mix_perm"] for f in filas])
    print(f"  n̄ propio − n̄ permutado: {d.mean():+.4f} ticks")
    out["delta_presupuesto"] = float(d.mean())

    print("\n=== CONTROL POSITIVO DEL NULO (la permutación DEBE mover algo) ===")
    for k, etiq in (("std_En", "std de E[n] entre instancias"),
                    ("mad_En_perm", "|E[n]_i − E[n]_perm(i)| medio"),
                    ("tv_perm", "distancia TV media p_i vs p_perm(i)")):
        v = float(np.mean([f[k] for f in filas]))
        out[k] = v
        print(f"  {etiq:38s} {v:.4f}")

    print("\n=== LECTURA FIJA (todas mezclas sobre los mismos estados) ===")
    ic("mix_propio", "mix_perm", "asignacion_instancia")
    ic("mix_propio", "mix_perm2", "asignacion_instancia_p2")
    ic("mix_perm", "mix_unif5", "forma_marginal")
    ic("mix_propio", "mix_unif5", "total_en_familia_mezcla")

    print("\n=== REFERENCIA: coste de colapsar a masa puntual ===")
    ic("mix_propio", "punto_nhat", "lectura_puntual")

    with open(os.path.join(RES, "n3_readout_fair.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Guardado n3_readout_fair.json", flush=True)


if __name__ == "__main__":
    main()
