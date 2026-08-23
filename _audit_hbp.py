"""
Test de regresión del HBP (instrumentación, CPU/FP32).
Tras el rediseño (tick por iteración del reasoner), comprueba que los fallos
estructurales del diagnóstico null están CORREGIDOS:

  TEST 1: la modulación del HBP DEPENDE del input.
  TEST 2: ω₀, ζ, c, f_gain reciben gradiente NO nulo desde L_total.
  TEST 3: ζ CAMBIA tras unos pasos de optimización.
  TEST 4: hbp_first (1er orden) y hbp_full (2º orden) producen logits DISTINTOS.

Corre en CPU/FP32. Sale con código !=0 si alguna comprobación falla.
"""
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from data.synthetic_recall import RunningSumModDataset
from training.trainer import build_model

torch.manual_seed(0)
dev = torch.device("cpu")
ds = RunningSumModDataset(seq_len=128, seed=0)
ok = True


def make(variant, seed=0):
    torch.manual_seed(seed)
    return build_model(variant, ds.vocab_size, 128).to(dev).float()


print("=" * 64)
print("TEST 1: ¿la modulación del HBP DEPENDE del input?")
m = make("hbp_full")
inpA, _, _ = ds.batch(8)
inpB, _, _ = ds.batch(8)
m(inpA, None); modA = {k: v.detach().clone() for k, v in m.hbp.modulation().items()}
m(inpB, None); modB = {k: v.detach().clone() for k, v in m.hbp.modulation().items()}
dep = max((modA[k] - modB[k]).abs().max().item() for k in modA)
print(f"  inputs distintos: {not torch.equal(inpA, inpB)} | max|mod(A)-mod(B)| = {dep:.3e}")
t1 = dep > 1e-5
print("  ->", "PASS (depende del input)" if t1 else "FAIL (independiente del input)")
ok &= t1

print("=" * 64)
print("TEST 2: ¿reciben gradiente ω₀, ζ, c, f_gain? (un backward de L_total)")
for variant in ["hbp_first", "hbp_full"]:
    m = make(variant)
    inp, tgt, _ = ds.batch(8)
    _, ld = m(inp, tgt)
    m.zero_grad(); ld["total"].backward()
    g = lambda p: 0.0 if p.grad is None else p.grad.abs().max().item()
    gw, gz, gc, gf = g(m.hbp.raw_omega0), g(m.hbp.raw_zeta), g(m.hbp.raw_c), g(m.hbp.raw_f_gain)
    print(f"  [{variant}] order={m.hbp.cfg.order} | |grad| ω₀={gw:.2e} ζ={gz:.2e} c={gc:.2e} f_gain={gf:.2e}")
    t2 = min(gw, gz, gc, gf) > 0.0
    print("  ->", "PASS (todos != 0)" if t2 else "FAIL (algún gradiente nulo)")
    ok &= t2

print("=" * 64)
print("TEST 3: ¿cambia ζ/ω₀ tras 30 pasos (AdamW, wd=0)?")
m = make("hbp_full")
z0, w0 = m.hbp.zeta.detach().clone(), m.hbp.omega0.detach().clone()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3, weight_decay=0.0)
for _ in range(30):
    inp, tgt, _ = ds.batch(8)
    _, ld = m(inp, tgt)
    opt.zero_grad(); ld["total"].backward(); opt.step()
dz, dw = (m.hbp.zeta - z0).abs().max().item(), (m.hbp.omega0 - w0).abs().max().item()
print(f"  max|Δζ|={dz:.3e} | max|Δω₀|={dw:.3e}")
t3 = dz > 1e-4 and dw > 1e-4
print("  ->", "PASS (ζ y ω₀ se mueven)" if t3 else "FAIL (congelados)")
ok &= t3

print("=" * 64)
print("TEST 4: ¿hbp_first (1er orden) != hbp_full (2º orden) en el mismo input?")
mf = make("hbp_first", seed=123)
mh = make("hbp_full", seed=123)
inp, _, _ = ds.batch(8)
with torch.no_grad():
    lf, _ = mf(inp, None)
    lh, _ = mh(inp, None)
diff = (lf - lh).abs().max().item()
print(f"  max|logits_first - logits_full| = {diff:.3e}")
t4 = diff > 1e-4
print("  ->", "PASS (los modelos difieren)" if t4 else "FAIL (idénticos)")
ok &= t4

print("=" * 64)
print("RESULTADO:", "TODO PASS — el HBP ya es load-bearing" if ok else "HAY FALLOS")
raise SystemExit(0 if ok else 1)
