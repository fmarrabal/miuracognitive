"""Valida los fixes FATAL-2/FATAL-3:
(A) [GPU/BF16] DEMOSTRACIÓN del bug de congelación: entrena 300 pasos con y sin
    pin_fp32 -> sin pin ζ/ω₀ quedan EXACTOS en init; con pin se mueven.
(B) [CPU] retrocompat: diff_solver='euler' (default) -> regresión intacta.
(C) [CPU] solver implícito: estable con ζ=ζ_min (donde euler explota), gradiente
    fluye a gamma_diff, y con γ_init=0.5 se comporta ~igual que euler al init.
"""
import math, torch
from data.synthetic_recall import PermutationCompositionDataset
from training.trainer import build_model

def short_train(m, ds, dev, dt, steps=300, lr=3e-4):
    act = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(act, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    m.train()
    for s in range(steps):
        inp, tgt, _ = ds.batch(32)
        _, ld = m(inp.to(dev), tgt.to(dev))
        opt.zero_grad(); ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(act, 1.0); opt.step()
    return m

if torch.cuda.is_available():
    dev, dt = torch.device("cuda:0"), torch.bfloat16
    print("(A) DEMOSTRACIÓN congelación BF16 (300 pasos, GPU):")
    for pin in (False, True):
        torch.manual_seed(0)
        ds = PermutationCompositionDataset(min_ops=2, max_ops=16, seq_len=128, seed=0, gen_set="adjacent")
        m = build_model("hbp_full", ds.vocab_size, 128, max_halt_steps=24).to(dev, dt)
        if pin:
            m.hbp.pin_fp32()
        z0, w0 = float(m.hbp.zeta.float().mean()), float(m.hbp.omega0.float().mean())
        short_train(m, ds, dev, dt)
        z1, w1 = float(m.hbp.zeta.float().mean()), float(m.hbp.omega0.float().mean())
        print(f"    pin_fp32={str(pin):5s}: ζ {z0:.6f}->{z1:.6f} (Δ={z1-z0:+.6f})  "
              f"ω₀ {w0:.6f}->{w1:.6f} (Δ={w1-w0:+.6f})  raw dtype={m.hbp.raw_zeta.dtype}")
        del m; torch.cuda.empty_cache()
else:
    print("(A) SALTADO: sin GPU")

print("\n(B) retrocompat euler (CPU): ver _audit_hbp aparte")

print("\n(C) solver implícito (CPU, FP32):")
torch.manual_seed(0)
ds = PermutationCompositionDataset(seq_len=128, seed=0)
# ζ clavado al suelo (0.05): euler explícito tendría r = Ω²/(2ζω₀) >> 2 (explosivo);
# el implícito debe ser estable.
ov = {"alpha_const": 0.0, "diff_solver": "implicit",
      "zeta_init": 0.06, "zeta_min": 0.05}
m = build_model("hbp_full", ds.vocab_size, 128, max_halt_steps=24, hbp_overrides=ov).float()
r_euler = float((1.0 * (m.hbp.omega0**2 + m.hbp.c**2 * m.hbp.rho_L) / (2*m.hbp.zeta*m.hbp.omega0)).max())
print(f"    ζ={float(m.hbp.zeta.mean()):.3f}: multiplicador de euler r_max={r_euler:.2f} (>2 = explosivo)")
inp, tgt, _ = ds.batch(8)
with torch.no_grad():
    logits, _ = m.eval()(inp, None)
dev_rest = float((m.hbp.h_t - m.hbp.h_star.unsqueeze(0)).abs().max())
print(f"    forward implícito con ζ=0.05: finito={torch.isfinite(logits).all().item()}  "
      f"max|h-h*|={dev_rest:.3f} (clamp=5.0; sin explosión)")
m.train()
_, ld = m(inp, tgt); ld["total"].backward()
g = m.hbp.raw_gamma_diff.grad
print(f"    grad(γ_diff): no-nulo={g is not None and float(g.abs().max())>0}  "
      f"|max|={0.0 if g is None else float(g.abs().max()):.2e}")
# equivalencia aproximada al init con params por defecto (γ=0.5 = 2ζ₀ω₀₀)
torch.manual_seed(0)
m_e = build_model("hbp_full", ds.vocab_size, 128, max_halt_steps=24,
                  hbp_overrides={"alpha_const": 0.0}).float().eval()
torch.manual_seed(0)
m_i = build_model("hbp_full", ds.vocab_size, 128, max_halt_steps=24,
                  hbp_overrides={"alpha_const": 0.0, "diff_solver": "implicit"}).float().eval()
with torch.no_grad():
    le, _ = m_e(inp, None); li, _ = m_i(inp, None)
print(f"    implícito(γ=0.5) vs euler al init (params default): "
      f"max|Δlogits|={float((le-li).abs().max()):.4f} (≈ pequeño: mismo régimen, distinto esquema)")
