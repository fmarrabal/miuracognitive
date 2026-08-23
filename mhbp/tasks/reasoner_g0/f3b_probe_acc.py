"""
F3b — Perfil de competencia acc(n,K) desde los checkpoints F3a (sin entrenar).

Referencia INDEPENDIENTE DE BRAZO del prereg (§2): sobre los ckpts
convergidos de cycle_transp (hbp_full y gating_wm, régimen indist K≤24), se
mide con forward_forced (model/adaptive_depth.py) la accuracy del decode en
la posición de respuesta con EXACTAMENTE n ticks del reasoner, sobre una
rejilla K∈[6..24] × n∈[1..24] con ≥256 muestras por K (las MISMAS muestras
para todos los n y todos los ckpts: comparación pareada).

De la tabla se deriva
    d_ref(K) = mín n tal que acc(n,K) ≥ 0.9·acc(24,K)
(ticks-hasta-competencia), el insumo de la calibración GS1/GS2 (f3b_gates) y
de la operacionalización del presupuesto B_total.

Además se reporta acc(n=1|K): el prereg exige que el suelo K_min del régimen
fácil cumpla acc(n=1)≤0.2 (si un K se resuelve en 1 tick, no exige gobierno);
se imprimen los K que lo violan.

Salida: results/f3b_acc_profile.json
  {"acc": {K: [acc n=1..24]}, "d_ref": {K: n}, "acc_n1": {K: acc},
   "checkpoints_usados": [...], "n_muestras": N,
   "acc_por_variante": {...}, "d_ref_por_variante": {...}}

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.f3b_probe_acc
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from training.config import TrainConfig
from training.trainer import build_dataset
from .g0_run import load_model, N_MAX

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "ckpts")
OUT_PATH = os.path.join(RES, "f3b_acc_profile.json")

GENS = "cycle_transp"
K_GRID = list(range(6, 25))          # rejilla de dificultad
N_GRID = list(range(1, N_MAX + 1))   # ticks forzados


def lista_ckpts():
    """Ckpts de referencia: los INDIST de cycle_transp (entrenados con K≤24)
    de ambas variantes con reasoner — la referencia promedia brazos para ser
    independiente de brazo (miura_mhbp se excluye: es el brazo bajo test en
    el programa F3, no una referencia neutral)."""
    out = []
    for variant in ("hbp_full", "gating_wm"):
        for p in sorted(glob.glob(os.path.join(
                CKPT, f"{variant}_{GENS}_indist_s*.pt"))):
            out.append((os.path.basename(p), variant))
    return out


def instancias_por_K(n_samples: int):
    """Genera las muestras de la rejilla UNA vez (pareadas entre n y ckpts).
    Reutiliza el builder de G0 con K exacto (min_writes=max_writes=K) y un
    stream propio (seed 910000+K, disjunto de los de train/eval de G0)."""
    datos = {}
    for K in K_GRID:
        ds = build_dataset(TrainConfig(task="permcomp", perm_gens=GENS,
                                       min_writes=K, max_writes=K,
                                       seed=910_000 + K))
        inp, tgt, nd = ds.batch(n_samples)
        assert (nd == K).all()
        p_ans = K + 1                              # posición de ARROW
        assert (tgt[:, p_ans] != -1).all(), f"label ausente en ARROW (K={K})"
        datos[K] = (inp, tgt[:, p_ans].clone(),
                    ds.answer_offset, ds.answer_size)
    return datos


@torch.no_grad()
def perfil_ckpt(model, datos, device, sub_batch=128):
    """acc[n-1, K-idx] para un checkpoint (forward_forced, decode en ARROW)."""
    acc = np.zeros((len(N_GRID), len(K_GRID)))
    for ki, K in enumerate(K_GRID):
        inp, ans, a0, a_sz = datos[K]
        p_ans = K + 1
        B = inp.shape[0]
        hits = np.zeros(len(N_GRID))
        for s in range(0, B, sub_batch):
            inp_s = inp[s:s + sub_batch].to(device)
            ans_s = ans[s:s + sub_batch].to(device)
            for niv, n in enumerate(N_GRID):
                forced = torch.full((inp_s.shape[0],), n, dtype=torch.long,
                                    device=device)
                logits, _ = model(inp_s, None, forced_steps=forced)
                pred = a0 + logits[:, p_ans, a0:a0 + a_sz].argmax(dim=-1)
                hits[niv] += float((pred == ans_s).sum())
        acc[:, ki] = hits / B
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=None,
                    help="muestras por K (default: 256 GPU / 64 CPU)")
    ap.add_argument("--ckpt_limit", type=int, default=None,
                    help="límite de ckpts (default: todos en GPU, 2 en CPU)")
    ap.add_argument("--sub_batch", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    n_samples = args.n_samples or (256 if device.type == "cuda" else 64)
    ckpts = lista_ckpts()
    limit = args.ckpt_limit or (len(ckpts) if device.type == "cuda" else 2)
    if limit < len(ckpts):
        # recorte balanceado entre variantes (referencia independiente de brazo)
        hb = [c for c in ckpts if c[1] == "hbp_full"]
        gw = [c for c in ckpts if c[1] == "gating_wm"]
        mix = [c for par in zip(hb, gw) for c in par]
        ckpts = mix[:limit]
    if not ckpts:
        raise SystemExit(f"No hay ckpts {GENS} indist en {CKPT}")
    print(f"Dispositivo: {device} ({dtype}); {n_samples} muestras/K; "
          f"{len(ckpts)} ckpts: {[c[0] for c in ckpts]}")

    datos = instancias_por_K(n_samples)
    perfiles = {}                      # ckpt → acc (24,19)
    t0 = time.time()
    for i, (fname, variant) in enumerate(ckpts, 1):
        model = load_model(fname, variant, GENS, device, dtype)
        perfiles[fname] = perfil_ckpt(model, datos, device, args.sub_batch)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        el = (time.time() - t0) / 60
        print(f"[{i}/{len(ckpts)}] {fname}: acc(n=24) media "
              f"{perfiles[fname][-1].mean():.3f} ({el:.1f} min)", flush=True)

    # --- agregación: media entre ckpts (cada ckpt pesa igual) ---
    A = np.stack(list(perfiles.values()))            # (n_ckpt, 24, 19)
    acc_mean = A.mean(axis=0)
    por_var = {}
    for variant in ("hbp_full", "gating_wm"):
        sel = [perfiles[f] for f, v in ckpts if v == variant]
        if sel:
            por_var[variant] = np.stack(sel).mean(axis=0)

    def d_ref_de(acc):
        out = {}
        for ki, K in enumerate(K_GRID):
            col = acc[:, ki]
            out[K] = int(np.argmax(col >= 0.9 * col[-1])) + 1
        return out

    d_ref = d_ref_de(acc_mean)
    d_ref_var = {v: d_ref_de(a) for v, a in por_var.items()}
    acc_n1 = {K: float(acc_mean[0, ki]) for ki, K in enumerate(K_GRID)}

    out = {
        "acc": {str(K): acc_mean[:, ki].tolist()
                for ki, K in enumerate(K_GRID)},
        "d_ref": {str(K): d_ref[K] for K in K_GRID},
        "acc_n1": {str(K): acc_n1[K] for K in K_GRID},
        "checkpoints_usados": [c[0] for c in ckpts],
        "n_muestras": n_samples,
        "device": str(device),
        "n_grid": N_GRID, "k_grid": K_GRID,
        "acc_por_variante": {v: {str(K): a[:, ki].tolist()
                                 for ki, K in enumerate(K_GRID)}
                             for v, a in por_var.items()},
        "d_ref_por_variante": {v: {str(K): d[K] for K in K_GRID}
                               for v, d in d_ref_var.items()},
    }
    os.makedirs(RES, exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT_PATH)

    # --- resumen legible ---
    print(f"\n{'K':>3s} {'acc(1)':>7s} {'acc(6)':>7s} {'acc(12)':>8s} "
          f"{'acc(24)':>8s} {'d_ref':>6s}"
          + ("" if not d_ref_var else "".join(f" {('d_' + v[:4]):>7s}"
                                              for v in d_ref_var)))
    for ki, K in enumerate(K_GRID):
        col = acc_mean[:, ki]
        print(f"{K:3d} {col[0]:7.3f} {col[5]:7.3f} {col[11]:8.3f} "
              f"{col[23]:8.3f} {d_ref[K]:6d}"
              + "".join(f" {d_ref_var[v][K]:7d}" for v in d_ref_var))
    print(f"\nd_ref medio sobre la rejilla K∈[6..24]: "
          f"{np.mean([d_ref[K] for K in K_GRID]):.1f} ticks/instancia")

    # --- suelo del prereg: acc(n=1|K) ≤ 0.2 (régimen fácil arranca en K_min) ---
    viol = [K for K in K_GRID if acc_n1[K] > 0.2]
    print(f"\nacc(n=1|K) por K: { {K: round(acc_n1[K], 3) for K in K_GRID} }")
    if viol:
        print(f"VIOLAN el suelo acc(n=1)≤0.2: K={viol} → el K_min del régimen "
              f"fácil debe quedar POR ENCIMA de max(viol)={max(viol)} "
              f"(dial de calibración; hoy k_easy={SessionSpecKEasy()})")
    else:
        print("Ningún K de la rejilla viola acc(n=1)≤0.2: el suelo actual vale.")
    print(f"Guardado: {OUT_PATH}")


def SessionSpecKEasy():
    from .f3b_env import SessionSpec
    return SessionSpec().k_easy


if __name__ == "__main__":
    main()
