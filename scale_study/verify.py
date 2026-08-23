"""
Verificación del simulador de campo a escala ANTES de lanzar la noche.
Cada bloque imprime PASS/FAIL con números. Si algo falla, NO se lanza el barrido.
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from scale_study.field_sim import (
    ChainOps, RingOps, Grid2DOps, build_ops, Phys, wave_step, run_wave,
    run_kdv_ring, kdv_ring_dispersion, flutter_threshold,
    companion_spectral_radius, pick_device, F64,
)
from model.hbp import build_chain_laplacian, build_chain_advection

torch.manual_seed(0)
dev = pick_device()
print(f"device={dev}  dtype=float64\n")
results = {}


def check(name, ok, detail=""):
    results[name] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ---- V1: operadores == hbp.py denso (chain), bitwise ----
for N in (6, 7, 13, 64):
    L = build_chain_laplacian(N).double()
    A = build_chain_advection(N).double()
    A3 = A @ A @ A
    ops = ChainOps(N)
    x = torch.randn(3, N, dtype=F64)
    dL = (ops.L(x) - x @ L.T).abs().max().item()
    dA = (ops.A(x) - x @ A.T).abs().max().item()
    dA3 = (ops.A3(x) - x @ A3.T).abs().max().item()
    check(f"V1 chain matvec N={N}", max(dL, dA, dA3) < 1e-10,
          f"|ΔL|={dL:.1e} |ΔA|={dA:.1e} |ΔA³|={dA3:.1e}")

# ---- V1b: wave_step == fórmula densa de hbp (order=2, sin f_θ/forcing) ----
N = 6
L = build_chain_laplacian(N).double(); A = build_chain_advection(N).double(); A3 = A@A@A
ops = ChainOps(N)
p = Phys(omega0=0.6, zeta=0.3, c=0.5, D=0.2, b=0.15, beta=0.1, nu=0.0, dt=1.0)
ht = torch.randn(4, N, dtype=F64); htm1 = torch.randn(4, N, dtype=F64)
hs = 0.0
v = ht - htm1; u = ht - hs
restit = -(p.omega0**2)*u
spatial = -(p.c**2)*(ht @ L.T)
damp = -(2*p.zeta*p.omega0/p.dt)*v - (p.D/p.dt)*(v @ L.T)
gyro = -(p.b/p.dt)*(v @ A.T) - (p.beta/p.dt)*(v @ A3.T)
h_ref = 2*ht - htm1 + p.dt**2*(restit + spatial + damp + gyro)
h_sim = wave_step(ops, p, ht, htm1, hs=hs, placement="gyro")
check("V1b wave_step vs densa", (h_ref - h_sim).abs().max().item() < 1e-10,
      f"|Δ|={(h_ref-h_sim).abs().max().item():.1e}")

# ---- V2: energía sin DERIVA SECULAR (ζ=0; Verlet conserva la sombra, la real
#         oscila O(dt²)) vs decaimiento sin crecimiento (ζ>0). dt pequeño. ----
ops = RingOps(512)
u0 = torch.randn(1, 512, dtype=F64) * 0.1
cons = run_wave(ops, Phys(omega0=0.5, zeta=0.0, c=0.4, dt=0.05), u0, ticks=8000)
E = cons["energies"]; half = len(E) // 2
drift = abs(sum(E[half:]) / (len(E) - half) - sum(E[:half]) / half) / abs(E[0])
osc = (max(E) - min(E)) / abs(E[0])
check("V2a energía sin deriva secular (ζ=0)", drift < 1e-3 and osc < 0.1,
      f"deriva secular={drift:.2e}  oscilación acotada={osc:.2e}")
diss = run_wave(ops, Phys(omega0=0.5, zeta=0.2, c=0.4, dt=0.05), u0, ticks=8000)
Ed = diss["energies"]
grows = max(Ed) > Ed[0] * 1.02
check("V2b energía decae sin crecer (ζ>0)", (not grows) and Ed[-1] < 1e-3 * Ed[0],
      f"E0={Ed[0]:.3e} E_fin={Ed[-1]:.3e} máx/E0={max(Ed)/Ed[0]:.3f}")

# ---- V3: giroscópico decae / circulatorio flutter, en régimen donde el gyro
#         DISCRETO es estable (Prop. 2: μ̃² < g(2-g)). dt pequeño, ζ moderado. ----
ops = RingOps(256)
u0 = torch.randn(8, 256, dtype=F64) * 0.1
pg = Phys(omega0=0.5, zeta=0.3, c=0.0, beta=0.1, dt=0.1)     # gyro estable, circ viola Merkin
gy = run_wave(ops, pg, u0, ticks=4000, placement="gyro")
ci = run_wave(ops, pg, u0, ticks=4000, placement="circulatory")
rg = gy["norms"][-1] / gy["norms"][0]
rc = ci["norms"][-1] / ci["norms"][0] if math.isfinite(ci["norms"][-1]) else float("inf")
# condición discreta (i) para el gyro
mu_t = pg.dt * pg.beta * ops.rho_A3; g_lo = 2 * pg.zeta * pg.omega0 * pg.dt
disc_i = mu_t ** 2 < g_lo * (2 - g_lo)
lhs, rhs, stable = flutter_threshold(pg, ops)
check("V3 gyro decae vs circ flutter", rg < 1.0 and rc > 3.0,
      f"gyro ratio={rg:.2e} (disc(i) ok={disc_i}) | circ ratio={rc:.2e} | "
      f"Merkin: β·ρ(A³)={lhs:.3f} vs 2ζω₀²={rhs:.3f} (circ estable={stable})")

# ---- V4: relación de dispersión analítica en el anillo (rama onda, sin amort.) ----
N = 1024; ops = RingOps(N)
p = Phys(omega0=0.5, zeta=0.0, c=0.4, dt=0.1)
errs = []
for kmode in (8, 32, 128, 400):
    idx = torch.arange(N, dtype=F64)
    mode = torch.cos(2*math.pi*kmode*idx/N).unsqueeze(0)
    res = run_wave(ops, p, mode*0.01, ticks=6000)
    # proyecta el estado sobre el modo cada tick -> traza; FFT temporal -> frecuencia
    # (usamos norms como proxy no sirve; re-corre proyectando)
    ht = (mode*0.01).clone(); htm1 = ht.clone()
    proj = []
    for t in range(6000):
        hn = wave_step(ops, p, ht, htm1, placement="gyro")
        htm1, ht = ht, hn
        proj.append(float((ht*mode).sum()))
    tr = torch.tensor(proj, dtype=F64); tr = tr - tr.mean()
    sp = torch.fft.rfft(tr).abs()
    kpk = int(sp[1:].argmax()) + 1
    freq_meas = 2*math.pi*kpk/len(tr)/p.dt          # rad/tiempo
    lamL = 4*math.sin(math.pi*kmode/N)**2
    K = p.omega0**2 + p.c**2*lamL
    Omega_verlet = math.acos(max(-1.0, 1 - p.dt**2*K/2)) / p.dt   # frecuencia discreta exacta
    rel = abs(freq_meas - Omega_verlet)/Omega_verlet
    errs.append(rel)
check("V4 dispersión anillo (medida vs Verlet analítico)", max(errs) < 0.02,
      f"máx err relativo={max(errs):.2e} sobre k∈{{8,32,128,400}}")

# ---- V5: solver KdV conserva masa (exacto) y momento (cuasi) ----
N = 2048
p = Phys(b=1.0, beta=0.05, nu=0.5, gamma=1.0, dt=0.02, nonlin="genuine")
idx = torch.arange(N, dtype=F64)
u0 = (0.5*torch.exp(-((idx-N/2)/(N*0.03))**2)).unsqueeze(0).to(dev)
kd = run_kdv_ring(N, p, u0, ticks=3000, device=dev, record_every=0)
uf = kd["final"]
mass0 = float(u0.sum()); massf = float(uf.sum())
mom0 = float((u0**2).sum()); momf = float((uf**2).sum())
check("V5a KdV conserva masa (genuine)", abs(massf-mass0)/abs(mass0) < 1e-6,
      f"Δmasa/masa={abs(massf-mass0)/abs(mass0):.2e}")
check("V5b KdV momento cuasi-conservado", abs(momf-mom0)/mom0 < 0.05 and math.isfinite(momf),
      f"Δmom/mom={abs(momf-mom0)/mom0:.2e}")

# ---- V6: companion exacta vs certificado analítico (N moderado) ----
ops = RingOps(24)
p_stab = Phys(omega0=0.5, zeta=0.5, c=0.4, beta=0.05, dt=1.0)
rho_gyro = companion_spectral_radius(ops, p_stab, torch.device("cpu"), placement="gyro")
lhs, rhs, stable_cert = flutter_threshold(p_stab, ops)
# circulatorio con β alto: debe cruzar |ρ|>1 si viola Merkin
p_flut = Phys(omega0=0.5, zeta=0.02, c=0.0, beta=0.3, dt=1.0)
rho_circ = companion_spectral_radius(ops, p_flut, torch.device("cpu"), placement="circulatory")
l2, r2, s2 = flutter_threshold(p_flut, ops)
check("V6a gyro estable (ρ_companion<1)", rho_gyro < 1.0 + 1e-6, f"ρ_gyro={rho_gyro:.4f}")
check("V6b circ viola Merkin -> ρ>1", (not s2) and rho_circ > 1.0,
      f"ρ_circ={rho_circ:.4f}  Merkin viol.: β·ρ(A³)={l2:.3f}>2ζω₀²={r2:.3f}")

# ---- V7: ESCALA — un paso a N grande en GPU (calibra el presupuesto nocturno) ----
print("\n--- V7 escala (tiempo por 100 ticks de onda) ---")
if dev.type == "cuda":
    for N in (10**4, 10**5, 10**6, 4*10**6):
        try:
            ops = RingOps(N)
            u0 = (torch.randn(1, N, dtype=F64, device=dev) * 0.01)
            torch.cuda.synchronize(); t0 = time.time()
            _ = run_wave(ops, Phys(omega0=0.5, zeta=0.1, c=0.4, beta=0.05, dt=0.5), u0, ticks=100)
            torch.cuda.synchronize()
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated()/1e9
            torch.cuda.reset_peak_memory_stats()
            print(f"   N={N:>9,}: {dt*1000:7.1f} ms / 100 ticks   pico VRAM={mem:.2f} GB")
        except RuntimeError as e:
            print(f"   N={N:>9,}: OOM/err -> {str(e)[:60]}")
            break
    # FFT KdV a escala
    print("   --- KdV FFT (100 ticks) ---")
    for N in (10**4, 10**5, 10**6):
        try:
            u0 = (torch.randn(1, N, dtype=F64, device=dev) * 0.01)
            torch.cuda.synchronize(); t0 = time.time()
            _ = run_kdv_ring(N, Phys(b=1.0, beta=0.05, nu=0.5, dt=0.02, nonlin="genuine"),
                             u0, ticks=100, device=dev)
            torch.cuda.synchronize(); dt = time.time()-t0
            print(f"   N={N:>9,}: {dt*1000:7.1f} ms / 100 ticks")
        except RuntimeError as e:
            print(f"   N={N:>9,}: OOM/err -> {str(e)[:60]}"); break
else:
    print("   (sin CUDA — se omite el test de escala GPU)")

print("\n=================  RESUMEN  =================")
npass = sum(1 for v in results.values() if v); ntot = len(results)
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"\n{npass}/{ntot} verificaciones PASADAS")
print("VERIFICACIÓN COMPLETA — OK PARA LANZAR" if npass == ntot else "*** HAY FALLOS — NO LANZAR ***")
