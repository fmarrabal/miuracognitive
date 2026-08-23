"""Entrena hbp_mix y hbp_full (permcomp adjacent, seed0) y comprueba si hbp_mix
APRENDE A ELEGIR SU FÍSICA: reporta α (onda↔difusión), D (difusión) y b (advección)
en función de la dificultad K, y compara accuracy con hbp_full."""
import math, json, torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model, evaluate

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
STEPS = 2500

def train(variant):
    torch.manual_seed(0)
    ds = PermutationCompositionDataset(min_ops=2, max_ops=16, seq_len=128, seed=0, gen_set="adjacent")
    ds_eval = PermutationCompositionDataset(min_ops=2, max_ops=24, seq_len=128, seed=100000, gen_set="adjacent")
    m = build_model(variant, ds.vocab_size, 128, max_halt_steps=24).to(dev, dt)
    act = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(act, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    m.train()
    for s in range(1, STEPS + 1):
        lr = 3e-4 * min(1, s/200) * 0.5*(1+math.cos(math.pi*min(1, s/STEPS)))
        for g in opt.param_groups: g["lr"] = lr
        inp, tgt, _ = ds.batch(32)
        _, ld = m(inp.to(dev), tgt.to(dev))
        opt.zero_grad(); ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(act, 1.0); opt.step()
    acc = evaluate(m, ds_eval, dev, dt, n_batches=40)
    return m, ds, acc

print("Entrenando hbp_full...", flush=True)
_, _, acc_full = train("hbp_full")
print("Entrenando hbp_mix...", flush=True)
m, ds, acc_mix = train("hbp_mix")

print("\n=== ACCURACY (permcomp adjacent, seed0, N_max=24) ===")
print(f"  hbp_full: corto={acc_full['corto']:.3f} medio={acc_full['medio']:.3f} largo={acc_full['largo']:.3f}")
print(f"  hbp_mix : corto={acc_mix['corto']:.3f} medio={acc_mix['medio']:.3f} largo={acc_mix['largo']:.3f}")

print("\n=== FÍSICA ELEGIDA por hbp_mix según dificultad K (medias) ===")
print(f"  {'K':>3s} {'α (onda↔dif)':>13s} {'D (difusión)':>13s} {'b (advección)':>14s}")
m.eval()
gen = torch.Generator().manual_seed(5)
for K in [2, 4, 6, 9, 12, 16, 20, 24]:
    seqs = []
    for _ in range(24):
        ops = [int(torch.randint(0, ds.n_gen, (1,), generator=gen).item()) for _ in range(K)]
        seq, run = [], ds.identity
        for j in ops:
            seq.append(ds.gen_offset + j); run = ds._compose(ds.gens[j], run)
        seq += [ds.Q, ds.ARROW]; seq += [ds.PAD]*(ds.seq_len-len(seq))
        seqs.append(torch.tensor(seq))
    with torch.no_grad():
        m(torch.stack(seqs).to(dev), None)
    ps = m.hbp.physics_state()
    print(f"  {K:>3d} {ps['alpha']:>13.3f} {ps['D']:>13.3f} {ps['b']:>14.3f}")
print("\n(α cerca de 1 = régimen onda; cerca de 0 = régimen difusión)")
