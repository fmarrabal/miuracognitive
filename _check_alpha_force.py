"""Valida alpha_const/alpha_force: (1) retrocompat bitwise (alpha_const=None ≡ antes;
alpha_const=1.0 ≡ order2 puro), (2) alpha_const=0 corre estable y difiere de onda,
(3) alpha_force por instancia produce salidas distintas por fila, (4) certificado:
m_euler activo con alpha_const<1."""
import torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model

torch.manual_seed(0)
ds = PermutationCompositionDataset(seq_len=128, seed=0)
inp, tgt, _ = ds.batch(4)

def fresh(variant, ov=None, seed=0):
    torch.manual_seed(seed)
    return build_model(variant, ds.vocab_size, 128, max_halt_steps=24, hbp_overrides=ov).float().eval()

# (1) retrocompat bitwise: hbp_full sin overrides vs alpha_const=1.0
m_a = fresh("hbp_full"); m_b = fresh("hbp_full", {"alpha_const": 1.0})
with torch.no_grad():
    la, _ = m_a(inp, None); lb, _ = m_b(inp, None)
print(f"(1) full vs alpha_const=1.0: bitwise idéntico = {torch.equal(la, lb)}")

# (2) alpha_const=0 (difusión pura sobre params de 2º orden): estable y distinto
m_c = fresh("hbp_full", {"alpha_const": 0.0})
with torch.no_grad():
    lc, _ = m_c(inp, None)
print(f"(2) alpha_const=0: finito={torch.isfinite(lc).all().item()}  distinto de onda={not torch.allclose(la, lc)}")
print(f"    certificado con rama difusiva: m_euler activo -> penalty={float(m_c.hbp.stability_penalty()):.4f} "
      f"(vs onda pura: {float(m_a.hbp.stability_penalty()):.4f})")

# (3) alpha_force por instancia: filas con α=1 deben igualar a onda, filas α=0 a difusión
m_d = fresh("hbp_full", {"mix_certificate": True})
af_ones = torch.ones(4); af_zeros = torch.zeros(4)
af_mixed = torch.tensor([1.0, 0.0, 1.0, 0.0])
with torch.no_grad():
    l_ones, _ = m_d(inp, None, alpha_force=af_ones)
    l_zeros, _ = m_d(inp, None, alpha_force=af_zeros)
    l_mix, _ = m_d(inp, None, alpha_force=af_mixed)
ok_w = torch.allclose(l_mix[0], l_ones[0]) and torch.allclose(l_mix[2], l_ones[2])
ok_d = torch.allclose(l_mix[1], l_zeros[1]) and torch.allclose(l_mix[3], l_zeros[3])
print(f"(3) alpha_force por instancia: filas α=1 ≡ onda: {ok_w} | filas α=0 ≡ difusión: {ok_d}")
diff_rows = not torch.allclose(l_mix[0], l_mix[1])
print(f"    física distinta produce salida distinta entre filas: {diff_rows}")

# (4) backward con alpha_force (el oráculo se usa EN entrenamiento)
m_e = fresh("hbp_full", {"mix_certificate": True}).train()
_, ld = m_e(inp, tgt, alpha_force=af_mixed)
ld["total"].backward()
g = [p.grad for p in m_e.parameters() if p.requires_grad and p.grad is not None]
print(f"(4) backward con alpha_force: OK, grads finitos={all(torch.isfinite(x).all() for x in g)}")
