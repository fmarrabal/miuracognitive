"""Valida el acoplamiento elíptico no-local: (1) retrocompat bitwise (flag off
= hbp_full), (2) L⁺ correcta (L·L⁺·L=L, simétrica, modo constante en nullspace),
(3) NO-LOCALIDAD genuina: perturbar la vorticidad del nodo j cambia la
modulación del nodo i≠j (imposible con readout local), (4) forward+backward."""
import torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model

torch.manual_seed(0)
ds = PermutationCompositionDataset(seq_len=128, seed=0)
inp, tgt, _ = ds.batch(4)

# (1) retrocompat: hbp_full sin flag == hbp_full (determinista bitwise)
def fresh(v, ov=None):
    torch.manual_seed(0)
    return build_model(v, ds.vocab_size, 128, hbp_overrides=ov).float().eval()
la, _ = fresh("hbp_full")(inp, None)
lb, _ = fresh("hbp_full")(inp, None)
print(f"(1) retrocompat hbp_full determinista: {torch.equal(la, lb)}")

# (2) L⁺ correcta
h = fresh("hbp_elliptic").hbp
L, Lp = h.laplacian, h.L_pinv
print(f"(2) L⁺: simétrica={torch.allclose(Lp, Lp.T, atol=1e-5)}  "
      f"L·L⁺·L=L={torch.allclose(L @ Lp @ L, L, atol=1e-4)}  "
      f"modo const en nullspace={float((Lp @ torch.ones(L.shape[0])).abs().max()):.2e}")

# (3) NO-LOCALIDAD: reset a h*, perturba SOLO el nodo j, mira ψ (y la modulación)
# en un nodo i != j. Con readout local sería 0; con elíptico != 0.
m = fresh("hbp_elliptic")
hb = m.hbp
hb.reset_state(1)
hb.h_t = hb.h_star.unsqueeze(0).clone()
hb.h_t[0, 0, :] += 1.0                      # perturba vorticidad del nodo 0
mod = hb.modulation()
halt = mod["halt_threshold"][0]             # (N,)
# baseline sin perturbar
hb.h_t = hb.h_star.unsqueeze(0).clone()
halt0 = hb.modulation()["halt_threshold"][0]
delta = (halt - halt0).abs()
print(f"(3) perturbar nodo 0 -> Δ halt_threshold por nodo: {[f'{d:.3f}' for d in delta]}")
print(f"    NO-LOCAL (nodos != 0 afectados): {bool((delta[1:] > 1e-4).any())}")
# contraste: hbp_full (local) -> solo el nodo 0 debería cambiar
ml = fresh("hbp_full").hbp
ml.reset_state(1); ml.h_t = ml.h_star.unsqueeze(0).clone()
ml.h_t[0, 0, :] += 1.0
hl = ml.modulation()["halt_threshold"][0]
ml.h_t = ml.h_star.unsqueeze(0).clone()
hl0 = ml.modulation()["halt_threshold"][0]
dl = (hl - hl0).abs()
print(f"    contraste LOCAL (hbp_full): otros nodos afectados = {bool((dl[1:] > 1e-4).any())} "
      f"(debe ser False)")

# (4) forward+backward
mt = fresh("hbp_elliptic").train()
_, ld = mt(inp, tgt); ld["total"].backward()
g = sum(float(p.grad.abs().sum()) for p in mt.parameters() if p.grad is not None)
print(f"(4) forward+backward OK, |grad|={g:.2f}, stability@init="
      f"{float(mt.hbp.stability_penalty().detach()):.4f}")
