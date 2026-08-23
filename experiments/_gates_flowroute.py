"""Kill-gates de FlowRoute (pre-registrados; cada uno puede matar el claim).
GATE 1: ω artesanal / red sin entrenar -> debe rutear ~azar (1/K). Si una ω
        artesanal ya rutea, la tarea mide el PRIOR, no capacidad aprendida.
GATE 2: SKYLINE ω-estática (entrenada) vs DINÁMICA. Si la estática resuelve
        (incl. cruces), la dinámica es inerte = colapso a set-point. La señal
        buscada: dinámica >> estática EN LOS CRUCES (irrealizables por flujo fijo).
Chance = 1/K. Malla y K pequeños para el gate barato.
"""
import torch
from data.flowroute import FlowRouteConfig, FlowRouteDataset, route_accuracy
from model.flow2d import Flow2DConfig, Flow2DField
from model.flowroute import (FlowRouteModel, handset_omega, train_flowroute)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = FlowRouteConfig(H=12, W=12, K=3)
fcfg = Flow2DConfig(H=12, W=12, dt=0.6, vel_scale=1.2, diffusion=0.01,
                    conserve_mass=True, nonneg=True, max_disp=1.5)
T = 18
chance = 1.0 / cfg.K
print(f"chance = {chance:.3f}")

# datos de evaluación fijos
ds_ev = FlowRouteDataset(cfg, seed=99)
sc = ds_ev.batch(256).to(dev)
frac_cross = float((sc.crossings > 0).float().mean())
print(f"fracción de instancias con cruces: {frac_cross:.2f}")

# ---- GATE 1: ω artesanal (sin entrenar) ----
flow = Flow2DField(fcfg).to(dev)
S = sc.S0
om = handset_omega(cfg, sc, flow)
for t in range(T):
    psi = flow.poisson(om)
    ux, uy = flow.velocity(psi)
    S = torch.stack([flow.diffuse(flow.advect(S[:, k], ux, uy)) for k in range(cfg.K)], 1)
r1 = route_accuracy(cfg, sc, S)
# red dinámica SIN entrenar
m0 = FlowRouteModel(cfg, fcfg, mode="dynamic", T=T).to(dev)
r0 = route_accuracy(cfg, sc, m0(sc))
print(f"\nGATE 1 (sin entrenar): ω-artesanal acc={r1['acc']:.3f} "
      f"(cruces {r1['acc_cross']:.3f}/no {r1['acc_nocross']:.3f}) | "
      f"red-aleatoria acc={r0['acc']:.3f}  [chance {chance:.3f}]")
print(f"   PASA gate1 si ambos ~chance (la ω artesanal NO resuelve sola): "
      f"{r1['acc'] < chance + 0.15}")

# ---- GATE 2: estática vs dinámica (entrenadas 3 seeds) ----
print("\nGATE 2: entrenando skyline estática vs dinámica...")
import statistics as st
res = {"static": [], "dynamic": []}
crossmet = {"static": [], "dynamic": []}
for seed in (0, 1, 2):
    for mode in ("static", "dynamic"):
        m = train_flowroute(cfg, fcfg, mode, seed=seed, steps=1500, T=T, device=dev)
        r = route_accuracy(cfg, sc, m(sc))
        res[mode].append(r["acc"])
        crossmet[mode].append(r["acc_cross"])
        print(f"   seed {seed} {mode:8s}: acc={r['acc']:.3f} "
              f"cruces={r['acc_cross']:.3f} no-cruces={r['acc_nocross']:.3f}", flush=True)
print(f"\nGATE 2 resumen (3 seeds):")
print(f"   estática: acc={st.mean(res['static']):.3f}  cruces={st.mean(crossmet['static']):.3f}")
print(f"   dinámica: acc={st.mean(res['dynamic']):.3f}  cruces={st.mean(crossmet['dynamic']):.3f}")
gap_cross = st.mean(crossmet['dynamic']) - st.mean(crossmet['static'])
print(f"   Δ(dyn−static) EN CRUCES = {gap_cross:+.3f}")
print(f"   -> DINÁMICA LOAD-BEARING si Δcruces>0 claro Y estática no resuelve cruces")
