"""
N3 — Fase A, paso 1: T-N3a (cableado) + SONDA de los 12 checkpoints
blind_flat congelados (PREREG_N3 v2 §2.1, §4).

T-N3a: sobre 1 checkpoint real, forced_steps homogéneo n debe reproducir el
PREFIJO del rollout nativo (record_step_states) tick a tick: estado del tick
n con tolerancia BF16 y canal de valor en el MISMO régimen. Si falla, la
sonda mediría otro sistema — no se sonda nada.

Sonda: por checkpoint, stream 'train' seed 1999 (dedupe por hash de
contenido contra val seed 2999 y eval seed 999), 1024 instancias × malla
n∈{1,2,3,4,5,6,8,10,12,16,20,24} con forced_steps → JSON por checkpoint:
(slot0, slot1, K, stake, correcto por n). Features del asignador = input
puro (los embeddings se consultan al ajustar; aquí basta el id).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n3_sonda
"""
import hashlib
import json
import os
import time

import torch

from training.trainer import build_model, select_device_dtype
from .n2_env import N2Dataset, N2Spec

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "ckpts")
N_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
M_SONDA = 1024
SEED_SONDA = 1999
CKPTS = [f"n2_e1_blind_flat_s{s}_r{r}" for s in range(3, 9) for r in (0, 1)]


def content_hash(row_tokens, k):
    """Hash del CONTENIDO de la tarea (K + secuencia de generadores),
    ignorando slots — la colisión que importa es la de la composición."""
    gens = row_tokens[2:2 + k].tolist()
    return hashlib.sha1(bytes([k] + gens)).hexdigest()


def load_ckpt(name, device, dtype):
    m = build_model("n2_endo", 125, 128,
                    max_halt_steps=24).to(device=device, dtype=dtype)
    sd = torch.load(os.path.join(CKPT, f"{name}.pt"), map_location=device)
    m.load_state_dict(sd, strict=True)
    m.hbp.pin_fp32()
    m.eval()
    return m


@torch.no_grad()
def t_n3a(model, spec, device):
    """Bit-igualdad (tolerancia BF16) forced(n) vs prefijo nativo."""
    ds = N2Dataset(spec, seed=12345, split="val")
    x, y, ks, ss = ds.batch(64)
    x = x.to(device)
    # rollout nativo completo con estados grabados
    model.record_step_states = True
    model.hbp.reset_state(64, device=device)
    model.value_ctx = None
    model(x)
    nat_states = [s.clone() for s in model._last_step_states]
    nat_cache = {k: v.clone() for k, v in model._val_cache.items()} \
        if model._val_cache else None
    model.record_step_states = False
    for n in (2, 3, 8, 24):
        model.hbp.reset_state(64, device=device)
        model.record_step_states = True
        model(x, forced_steps=torch.full((64,), n, device=device))
        st = model._last_step_states
        model.record_step_states = False
        d = (st[n - 1].float() - nat_states[n - 1].float()).abs().max()
        assert float(d) < 5e-2, f"T-N3a: forced({n}) diverge del prefijo " \
            f"nativo (Δmax={float(d):.4f})"
        if n >= 2:
            assert model._val_cache is not None, \
                f"T-N3a: caché de valor vacío en forced({n})"
            dv = (model._val_cache["s"] - nat_cache["s"]).abs().max()
            assert float(dv) < 5e-2, \
                f"T-N3a: stakê difiere forced vs nativo (Δ={float(dv):.4f})"
    return True


@torch.no_grad()
def sonda_ckpt(model, spec, device, seen_hashes):
    ds = N2Dataset(spec, seed=SEED_SONDA, split="train")
    rows = []
    n_col = 0
    while len(rows) < M_SONDA:
        x, y, ks, ss = ds.batch(256)
        for b in range(x.shape[0]):
            h = content_hash(x[b], int(ks[b]))
            if h in seen_hashes:
                n_col += 1
                continue
            rows.append((x[b], y[b], int(ks[b]), float(ss[b])))
            if len(rows) >= M_SONDA:
                break
    X = torch.stack([r[0] for r in rows]).to(device)
    Y = torch.stack([r[1] for r in rows])
    out = {"slot0": X[:, 0].tolist(), "slot1": X[:, 1].tolist(),
           "K": [r[2] for r in rows], "stake": [r[3] for r in rows],
           "colisiones_dedupe": n_col, "n_grid": N_GRID, "correct": {}}
    valid = (Y != -1)
    idx_last = valid.float().cumsum(1).argmax(1)
    rows_i = torch.arange(X.shape[0])
    for n in N_GRID:
        hits = []
        for i in range(0, M_SONDA, 256):
            xb = X[i:i + 256]
            model.hbp.reset_state(xb.shape[0], device=device)
            model.value_ctx = None
            logits, _ = model(xb, forced_steps=torch.full(
                (xb.shape[0],), n, device=device))
            il = idx_last[i:i + 256]
            rb = torch.arange(xb.shape[0])
            pred = logits[rb, il.to(device)].argmax(-1).cpu()
            hits.append((pred == Y[i:i + 256][rb, il]).float())
        out["correct"][str(n)] = torch.cat(hits).int().tolist()
    return out


def main():
    device, dtype = select_device_dtype()
    spec = N2Spec(p_hi=0.15)
    # hashes de eval (999) y val (2999) para el dedupe declarado (§5)
    seen = set()
    for seed, split, m in ((999, "eval", 16384), (2999, "val", 8192)):
        ds = N2Dataset(spec, seed=seed, split=split)
        got = 0
        while got < m:
            x, y, ks, ss = ds.batch(512)
            for b in range(x.shape[0]):
                seen.add(content_hash(x[b], int(ks[b])))
            got += x.shape[0]
    print(f"dedupe: {len(seen)} hashes de eval+val", flush=True)

    t0 = time.time()
    m0 = load_ckpt(CKPTS[0], device, dtype)
    t_n3a(m0, spec, device)
    print("T-N3a OK — forced ≡ prefijo nativo (estados y stakê, tol. BF16)",
          flush=True)
    del m0
    torch.cuda.empty_cache()

    for name in CKPTS:
        path = os.path.join(RES, f"n3_sonda_{name}.json")
        if os.path.exists(path):
            print(f"skip {name}", flush=True)
            continue
        model = load_ckpt(name, device, dtype)
        out = sonda_ckpt(model, spec, device, seen)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(out, f)
        os.replace(path + ".tmp", path)
        accs = {n: sum(out["correct"][str(n)]) / M_SONDA for n in (1, 3, 5, 8, 24)}
        print(f"{name}: acc(n)= " +
              " ".join(f"{n}:{a:.2f}" for n, a in accs.items()) +
              f"  colis={out['colisiones_dedupe']} "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"SONDA COMPLETA en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
