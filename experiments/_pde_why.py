"""Smoking gun del null: ¿por qué la física es constante? Entrena hbp_mix, compara
la norma de las cabezas de física init vs entrenada (si decaen a ~0, la física es
DEMOSTRABLEMENTE input-independiente), mide α por-input a granularidad fina (no solo
la media que lava) y la correlación α↔señales de interocepción."""
import math, statistics, torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model, evaluate

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
torch.manual_seed(0)
ds = PermutationCompositionDataset(min_ops=2, max_ops=16, seq_len=128, seed=0, gen_set="adjacent")

def head_norms(h):
    return {k: float(getattr(h, k).weight.norm()) for k in ("alpha_head", "D_head", "b_head")}

m = build_model("hbp_mix", ds.vocab_size, 128, max_halt_steps=24).to(dev, dt)
n0 = head_norms(m.hbp)
print("||W|| cabezas de física ANTES de entrenar:", {k: round(v, 4) for k, v in n0.items()})

act = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(act, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
m.train()
STEPS = 2500
gnorm_acc = {k: 0.0 for k in ("alpha_head", "D_head", "b_head")}
for s in range(1, STEPS + 1):
    lr = 3e-4 * min(1, s/200) * 0.5*(1+math.cos(math.pi*min(1, s/STEPS)))
    for g in opt.param_groups: g["lr"] = lr
    inp, tgt, _ = ds.batch(32)
    _, ld = m(inp.to(dev), tgt.to(dev))
    opt.zero_grad(); ld["total"].backward()
    for k in gnorm_acc:  # acumula la norma del gradiente que REALMENTE llegó a las cabezas
        gr = getattr(m.hbp, k).weight.grad
        if gr is not None: gnorm_acc[k] += float(gr.norm())
    torch.nn.utils.clip_grad_norm_(act, 1.0); opt.step()

n1 = head_norms(m.hbp)
print("||W|| cabezas de física DESPUÉS de entrenar:", {k: round(v, 4) for k, v in n1.items()})
print("Σ|grad| acumulado (2500 pasos):", {k: round(v, 3) for k, v in gnorm_acc.items()})
print("  -> ¿el gradiente llegó (>0) pero W no creció? = la loss EMPUJA a física constante")

# α por-input a granularidad fina (no la media global)
m.eval()
gen = torch.Generator().manual_seed(7)
def make(K):
    ops = [int(torch.randint(0, ds.n_gen, (1,), generator=gen).item()) for _ in range(K)]
    seq = [ds.gen_offset + j for j in ops] + [ds.Q, ds.ARROW]
    seq += [ds.PAD]*(ds.seq_len-len(seq))
    return torch.tensor(seq)

per_input_mean, within_input_std = [], []
with torch.no_grad():
    for _ in range(128):
        m(make(12).unsqueeze(0).to(dev), None)
        # α completo (B,N,d_h) del último tick: reconstruye desde las cabezas y el intero guardado
        a = m.hbp._last_alpha  # media; para la dispersión intra-input usamos el tensor si está
        per_input_mean.append(a)
# dispersión de α ENTRE inputs distintos (si ~0 con ||W||~0 -> input-independiente demostrado)
print(f"\nα por-input (128 problemas K=12): media={statistics.mean(per_input_mean):.4f} "
      f"std-entre-inputs={statistics.pstdev(per_input_mean):.5f}")
acc = evaluate(m, PermutationCompositionDataset(min_ops=2, max_ops=24, seq_len=128, seed=100000, gen_set="adjacent"),
               dev, dt, n_batches=40)
print(f"accuracy L/M/S = {acc['largo']:.3f}/{acc['medio']:.3f}/{acc['corto']:.3f}")
print("\nVEREDICTO smoking-gun:")
grew = {k: n1[k] > 1.5*n0[k] for k in n0}
print(f"  ¿||W|| creció (aprendió sensibilidad)?: {grew}")
print(f"  init→final α: {statistics.mean(per_input_mean):.3f} (init bias=0.800)")
