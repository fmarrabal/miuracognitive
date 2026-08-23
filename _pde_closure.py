"""Blindaje del null (cierra measurement-coarse y stiffness-clamp de forma hermética):
  (1) dispersión de α por-dim: std de α sobre las 384 entradas (N×d_h) DENTRO de un input,
      y rango por-(nodo,dim) ENTRE inputs. ¿Alguna dim llega al régimen difusivo (α<0.4)?
  (2) norma POR FILA de W_alpha_head (¿alguna fila > 0.3 capaz de conmutar?).
  (3) deviation_from_rest del VEI a lo largo de los ticks vs h_clamp=5 (¿satura el clamp?).
"""
import math, statistics, torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model, evaluate

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
torch.manual_seed(0)
ds = PermutationCompositionDataset(min_ops=2, max_ops=16, seq_len=128, seed=0, gen_set="adjacent")
m = build_model("hbp_mix", ds.vocab_size, 128, max_halt_steps=24).to(dev, dt)
act = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(act, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
m.train()
STEPS = 2500
for s in range(1, STEPS+1):
    lr = 3e-4 * min(1, s/200) * 0.5*(1+math.cos(math.pi*min(1, s/STEPS)))
    for g in opt.param_groups: g["lr"] = lr
    inp, tgt, _ = ds.batch(32)
    _, ld = m(inp.to(dev), tgt.to(dev))
    opt.zero_grad(); ld["total"].backward()
    torch.nn.utils.clip_grad_norm_(act, 1.0); opt.step()

m.eval()
h = m.hbp
# (2) normas por fila de W_alpha_head (d_h filas de dim d_intero)
W = h.alpha_head.weight.detach().float()   # (d_h, d_intero)
row = W.norm(dim=1)                         # (d_h,)
print(f"(2) W_alpha_head filas: max={row.max():.3f} mean={row.mean():.3f} (>0.3 = fila capaz de conmutar)")

# (1) dispersión de α por-dim
gen = torch.Generator().manual_seed(3)
def make(K):
    ops = [int(torch.randint(0, ds.n_gen, (1,), generator=gen).item()) for _ in range(K)]
    seq = [ds.gen_offset + j for j in ops] + [ds.Q, ds.ARROW]; seq += [ds.PAD]*(ds.seq_len-len(seq))
    return torch.tensor(seq)

alphas = []  # cada uno (N,d_h) del último tick
with torch.no_grad():
    for _ in range(32):
        m(make(12).unsqueeze(0).to(dev), None)
        alphas.append(h._last_alpha_full[0].float().cpu())   # (N,d_h)
A = torch.stack(alphas)   # (32, N, d_h)
within = A.std(dim=(1, 2)).mean()          # dispersión por-dim DENTRO de un input (media sobre inputs)
per_dim_range = (A.max(0).values - A.min(0).values)  # (N,d_h) rango ENTRE inputs por dim
print(f"(1) α por-dim: min={A.min():.3f} max={A.max():.3f}  std-intra-input(media)={within:.4f}")
print(f"    ¿alguna dim en régimen difusivo (α<0.4)?: {(A < 0.4).any().item()}  (nº dims={int((A<0.4).any(0).sum())}/{A.shape[1]*A.shape[2]})")
print(f"    rango por-dim ENTRE inputs: max={per_dim_range.max():.4f} mean={per_dim_range.mean():.4f} (conmutación real necesitaría ~O(0.5))")

# (3) deviation del VEI vs clamp
m.record_trace = True
with torch.no_grad():
    m(make(20).unsqueeze(0).to(dev), None)
m.record_trace = False
devs = [t["deviation"] for t in m._trace]
print(f"(3) deviation_from_rest del VEI por tick: max={max(devs):.3f}  vs h_clamp={h.cfg.h_clamp}  "
      f"-> {'SATURA' if max(devs) > 0.9*h.cfg.h_clamp else 'NO satura (sin artefacto de clamp)'}")
acc = evaluate(m, PermutationCompositionDataset(min_ops=2, max_ops=24, seq_len=128, seed=100000, gen_set="adjacent"),
               dev, dt, n_batches=40)
print(f"accuracy L/M/S = {acc['largo']:.3f}/{acc['medio']:.3f}/{acc['corto']:.3f}")
