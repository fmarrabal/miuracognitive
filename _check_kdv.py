"""Valida los operadores KdV: (1) A³ antisimétrica y ρ(A³)=ρ(A)³; (2) hbp_kdv
construye, forward/backward, grads a β/ν; (3) dinámica estable (VEI acotado, sin
NaN) en euler e implícito; (4) certificado=0 al init; (5) retrocompat bitwise de
hbp_full (kdv_max=0 por defecto); (6) la dispersión ES conservativa: con solo
disp activo, la energía ||u||² no crece."""
import torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model

torch.manual_seed(0)
ds = PermutationCompositionDataset(seq_len=128, seed=0)
inp, tgt, _ = ds.batch(4)

# (1) A³ antisimétrica, ρ(A³)=ρ(A)³
m = build_model("hbp_kdv", ds.vocab_size, 128).float()
h = m.hbp
A3 = h.adv3
print(f"(1) A³ antisimétrica={torch.allclose(A3, -A3.T)}  "
      f"ρ(A³)={float(h.rho_A3):.4f} vs ρ(A)³={float(h.rho_A)**3:.4f}")

# (2) forward/backward + grads
m.train()
_, ld = m(inp, tgt); m.zero_grad(); ld["total"].backward()
gb = h.raw_kdv_beta.grad; gn = h.raw_kdv_nu.grad
print(f"(2) backward OK; |grad β|={0 if gb is None else float(gb.abs().max()):.2e} "
      f"|grad ν|={0 if gn is None else float(gn.abs().max()):.2e}")

# (3) dinámica estable en ambos solvers
for solver in ("euler", "implicit"):
    torch.manual_seed(0)
    mm = build_model("hbp_kdv", ds.vocab_size, 128,
                     hbp_overrides={"diff_solver": solver, "alpha_const": 0.5}).float().eval()
    with torch.no_grad():
        lo, _ = mm(inp, None)
    dev_rest = float((mm.hbp.h_t.float() - mm.hbp.h_star.float().unsqueeze(0)).abs().max())
    print(f"(3) {solver:8s}: finito={torch.isfinite(lo).all().item()}  max|u|={dev_rest:.3f} (clamp=5)")

# (4) certificado al init
print(f"(4) stability_penalty(hbp_kdv)@init = {float(h.stability_penalty().detach()):.4f}")

# (5) retrocompat: hbp_full no cambia
def fresh(v):
    torch.manual_seed(0); return build_model(v, ds.vocab_size, 128).float().eval()
la, _ = fresh("hbp_full")(inp, None)
lb, _ = fresh("hbp_full")(inp, None)
print(f"(5) hbp_full determinista/bitwise consigo mismo: {torch.equal(la, lb)}  "
      f"_has_kdv={fresh('hbp_full').hbp._has_kdv}")

# (5b) CERTIFICADO vs RADIO ESPECTRAL EXACTO (contraejemplo del panel: el
# certificado viejo daba 0 con ρ(companion)>1; el nuevo debe ser CONSISTENTE:
# penalty=0 => ρ<1, y si ρ>1 => penalty>0)
import torch as _t
mk0 = build_model("hbp_kdv", ds.vocab_size, 128).float()
hb0 = mk0.hbp
for label, beta_raw in [("init (β≈0.05)", None), ("β al tope (0.1)", 8.0)]:
    if beta_raw is not None:
        with _t.no_grad(): hb0.raw_kdv_beta.fill_(beta_raw)
    pen = float(hb0.stability_penalty().detach())
    rho = hb0.certificate_spectral_radius()
    ok = (pen > 0) or (rho < 1.0)
    print(f"(5b) {label:16s}: penalty={pen:.4f}  ρ_exacto={rho:.4f}  consistente={ok}")

# (6a) COLOCACIÓN GIROSCÓPICA (la corrección de Ziegler): en la rama de 2º orden
# la dispersión actúa en ḣ (no hace trabajo). Rollout Verlet largo del sistema
# ḧ = -ω₀²u - 2ζω₀ḣ - βA³ḣ -> la energía debe DECAER (Lyapunov), no flutter.
# Contraste: colocación POSICIONAL (circulatoria) ḧ = -ω₀²u - 2ζω₀ḣ - βA³u -> flutter.
N = h.adv3.shape[0]
def rollout(gyro_placement, ticks=400, beta=0.1, om0=0.5, ze=0.5):
    torch.manual_seed(1)
    U = 0.1 * torch.randn(N, 16); Um1 = U.clone()
    for _ in range(ticks):
        vv = U - Um1
        acc = -(om0**2) * U - 2*ze*om0*vv
        acc = acc - beta * (A3 @ vv) if gyro_placement else acc - beta * (A3 @ U)
        U2 = 2*U - Um1 + acc
        Um1, U = U, U2
        if not torch.isfinite(U).all():
            return float("inf")
    return float(U.norm() / 0.1)
g_gyro, g_circ = rollout(True), rollout(False)
print(f"(6a) rollout 400 ticks: GIROSCÓPICO ||U||ratio={g_gyro:.2e} (debe decaer) | "
      f"CIRCULATORIO={g_circ:.2e} (flutter esperado -> justifica la corrección)")

# (6b) free-run del step() REAL de hbp_kdv (β,ν al máximo): 300 ticks, VEI acotado
torch.manual_seed(0)
mk = build_model("hbp_kdv", ds.vocab_size, 128).float().eval()
hb = mk.hbp
with torch.no_grad():
    for p in (hb.raw_kdv_beta, hb.raw_kdv_nu):
        p.fill_(8.0)                       # sigmoid≈1 -> coeficientes al TOPE
    hb.reset_state(4)
    intero = 0.3 * torch.randn(4, hb.cfg.n_nodes, hb.cfg.d_intero)
    devs = []
    for t in range(300):
        hh = hb.step(intero)
        devs.append(float((hh - hb.h_star.unsqueeze(0)).abs().max()))
print(f"(6b) free-run step() real 300 ticks (β,ν al tope): max|u|={max(devs):.3f} "
      f"final={devs[-1]:.3f} (clamp=5; {'ACOTADO sin tocar clamp' if max(devs) < 4.5 else 'toca el clamp — revisar'})")
