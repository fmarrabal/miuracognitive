"""Piloto de calibración (NO confirmatorio): coste por paso a escala real y
curva de convergencia corta de mhbp y gru → fija steps del confirmatorio."""
import time
import torch

from .env import EnvConfig, sample_latents, rollout
from .controllers import build_controller, param_table

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = EnvConfig()
print("--- parámetros (tras equiparación) ---")
for k, v in param_table().items():
    print(f"  {k:12s} core={v['core']:6d}  total={v['total']:6d}")

for name in ("mhbp", "gru"):
    torch.manual_seed(0)
    c = build_controller(name).to(DEV)
    opt = torch.optim.Adam(c.parameters(), lr=3e-3)
    times, Js = [], []
    for step in range(80):
        t0 = time.time()
        lat = sample_latents(cfg, 32, cfg.T, 999000 + step, DEV)
        out = rollout(c, lat, DEV)
        loss = out["J"].mean() + 1e-3 * c.regularizer()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(c.parameters(), 1.0)
        opt.step()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        Js.append(float(loss))
    q = lambda xs: f"{xs[0]:.3f} {xs[19]:.3f} {xs[39]:.3f} {xs[59]:.3f} {xs[79]:.3f}"
    print(f"[{name}] s/paso={sum(times[5:])/len(times[5:]):.2f}  "
          f"J@(0,20,40,60,80)=({q(Js)})")
