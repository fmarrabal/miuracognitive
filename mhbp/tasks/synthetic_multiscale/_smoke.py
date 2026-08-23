"""Smoke Fase 2: cada controlador entrena 2 pasos (T corto) sin NaN y con grads."""
import time
import torch

from .env import EnvConfig, sample_latents, rollout
from .controllers import CONTROLLERS, build_controller, param_table
from .oracle import oracle_actions

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEV}")
print("--- parámetros ---")
for k, v in param_table().items():
    print(f"  {k:12s} core={v['core']:6d}  total={v['total']:6d}")

cfg = EnvConfig()
lat = sample_latents(cfg, 4, 64, 1, DEV)
J_or = oracle_actions(lat, iters=50)["J"]
print(f"oráculo(50it) J={float(J_or.mean()):.4f}")

for name in CONTROLLERS:
    t = time.time()
    torch.manual_seed(0)
    c = build_controller(name).to(DEV)
    opt = torch.optim.Adam(c.parameters(), lr=3e-3)
    Js = []
    for step in range(2):
        lat = sample_latents(cfg, 4, 64, 10 + step, DEV)
        out = rollout(c, lat, DEV)
        loss = out["J"].mean() + 1e-3 * c.regularizer()
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(c.parameters(), 1.0)
        opt.step()
        Js.append(float(loss))
    ok = all(torch.isfinite(torch.tensor(Js)))
    print(f"  [{'OK' if ok else 'ERR'}] {name:12s} J={Js[0]:.4f}->{Js[-1]:.4f} "
          f"|grad|={float(gn):.3f} ({time.time()-t:.1f}s)")
print("SMOKE DONE")
