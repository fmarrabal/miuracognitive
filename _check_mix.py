"""Verifica la familia PDE: (1) las variantes existentes NO cambian (retrocompat),
(2) hbp_mix construye, forward+backward OK, gradiente a las cabezas de física,
(3) el modelo mezcla física de verdad (α, D, b no triviales y dependientes del input)."""
import torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model

torch.manual_seed(0)
dev = torch.device("cpu")
ds = PermutationCompositionDataset(seq_len=128, seed=0)

print("=" * 62)
print("(1) RETROCOMPAT: hbp_full con D=b=0, gate=False -> debe ser onda pura")
m = build_model("hbp_full", ds.vocab_size, 128).to(dev).float()
print(f"    D_max={m.hbp.cfg.D_max} b_max={m.hbp.cfg.b_adv_max} gate={m.hbp.cfg.gate_physics}"
      f"  _has_D={m.hbp._has_D} _has_b={m.hbp._has_b}")
inp, tgt, _ = ds.batch(4)
_, ld = m(inp, tgt); ld["total"].backward()
print(f"    forward+backward OK, loss={ld['total'].item():.3f}")

print("=" * 62)
print("(2) hbp_mix: construye, entrena un paso, gradiente a las cabezas de física")
m = build_model("hbp_mix", ds.vocab_size, 128).to(dev).float()
h = m.hbp
print(f"    D_max={h.cfg.D_max} b_max={h.cfg.b_adv_max} gate={h.cfg.gate_physics}")
print(f"    tiene alpha_head={hasattr(h,'alpha_head')} D_head={hasattr(h,'D_head')} b_head={hasattr(h,'b_head')}")
print(f"    ρ(L)={float(h.rho_L):.3f}  ρ(A_dir)={float(h.rho_A):.3f}  A_dir antisimétrica={torch.allclose(h.adv, -h.adv.T)}")
inp, tgt, _ = ds.batch(4)
_, ld = m(inp, tgt); m.zero_grad(); ld["total"].backward()
g = lambda p: 0.0 if p.grad is None else p.grad.abs().max().item()
print(f"    |grad| alpha_head={g(h.alpha_head.weight):.2e} D_head={g(h.D_head.weight):.2e} b_head={g(h.b_head.weight):.2e}")

print("=" * 62)
print("(3) ¿MEZCLA física? Régimen físico tras un forward (medias):")
m.eval()
with torch.no_grad():
    m(inp, None); ps = h.physics_state()
print(f"    α (1=onda,0=difusión)={ps['alpha']:.3f}  D (difusión)={ps['D']:.3f}  b (advección)={ps['b']:.3f}")
# ¿depende del input?
with torch.no_grad():
    inpA,_,_ = ds.batch(4); m(inpA.to(dev), None); a1 = h.physics_state()['alpha']
    inpB,_,_ = ds.batch(4); m(inpB.to(dev), None); a2 = h.physics_state()['alpha']
print(f"    α(inputA)={a1:.4f} vs α(inputB)={a2:.4f}  -> depende del input: {abs(a1-a2)>1e-5}")
print("=" * 62)
print("estabilidad: penalty hbp_mix =", float(h.stability_penalty()))
