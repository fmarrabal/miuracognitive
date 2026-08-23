"""Diagnóstico de aprendibilidad: ¿el flujo puede rutear CUANDO es factible?
K=2, difusión baja, loss de solapamiento. Mide accuracy Y masa entregada, en
el subconjunto NO-CRUCES (factible para un flujo) vs cruces. Si NO-cruces sube
claramente sobre chance -> el sustrato SÍ transporta; el fallo de multi-símbolo
es el límite de incompresibilidad (informativo). Si ni no-cruces sube -> sigue
roto."""
import torch, statistics as st
from data.flowroute import FlowRouteConfig, FlowRouteDataset, route_accuracy
from model.flow2d import Flow2DConfig
from model.flowroute import FlowRouteModel, train_flowroute

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = FlowRouteConfig(H=12, W=12, K=2)
fcfg = Flow2DConfig(H=12, W=12, dt=0.6, vel_scale=1.4, diffusion=0.004,
                    conserve_mass=True, nonneg=True, max_disp=1.5)
T = 20
ds_ev = FlowRouteDataset(cfg, seed=99)
sc = ds_ev.batch(256).to(dev)


@torch.no_grad()
def delivered(m):
    S = m(sc)
    B = sc.S0.shape[0]
    tot = 0.0
    for k in range(cfg.K):
        r = sc.dst_rc[:, k, 0]; c = sc.dst_rc[:, k, 1]
        # masa del canal k en ventana 3x3 de SU destino / masa total del canal k
        frac = []
        for b in range(B):
            patch = S[b, k, max(0, int(r[b]) - 1):int(r[b]) + 2,
                      max(0, int(c[b]) - 1):int(c[b]) + 2].sum()
            frac.append(float(patch / S[b, k].sum().clamp_min(1e-6)))
        tot += sum(frac) / len(frac)
    return tot / cfg.K


print(f"K={cfg.K} chance={1/cfg.K:.2f}, cruces en eval={float((sc.crossings>0).float().mean()):.2f}")
for mode in ("static", "dynamic"):
    accs, dels, accnc = [], [], []
    for seed in (0, 1):
        m = train_flowroute(cfg, fcfg, mode, seed=seed, steps=2000, batch=128, T=T, device=dev)
        r = route_accuracy(cfg, sc, m(sc))
        accs.append(r["acc"]); accnc.append(r["acc_nocross"]); dels.append(delivered(m))
    print(f"{mode:8s}: acc={st.mean(accs):.3f} acc_no-cruces={st.mean(accnc):.3f} "
          f"masa_entregada={st.mean(dels):.3f}", flush=True)
print("PASA aprendibilidad si el flujo entrega masa a su destino (>~0.4) y "
      "acc_no-cruces >> chance en al menos un modo")
