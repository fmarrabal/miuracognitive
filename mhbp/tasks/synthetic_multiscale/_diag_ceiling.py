"""Diagnóstico (calibración de gates, sin datos confirmatorios): ¿dónde satura
la política PRIVILEGIADA? Estima el techo de la clase de política frente al
oráculo por-instancia. Recovery @250/@500/@750 pasos."""
import torch

from .env import EnvConfig, sample_latents, episode_objective
from .gates import _rollout_sig, _train_gru, DEV
from .oracle import oracle_actions
from .controllers import BaselineController

cfg = EnvConfig()
lat_eval = sample_latents(cfg, 128, cfg.T, 7777, DEV)
J_oracle = float(oracle_actions(lat_eval)["J"].mean())

# política constante (referencia superior del hueco)
torch.manual_seed(0)
raw = torch.zeros(4, device=DEV, dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([raw], lr=0.05)
B, T = lat_eval["d"].shape
for _ in range(400):
    A = {"halt_bias": (-3 + 6 * torch.sigmoid(raw[0])).expand(B, T),
         "depth_max": (2 + 22 * torch.sigmoid(raw[1])).expand(B, T),
         "tool_gate": torch.sigmoid(raw[2]).expand(B, T),
         "budget_scale": (0.25 + 3.75 * torch.sigmoid(raw[3])).expand(B, T)}
    loss = episode_objective(lat_eval, A)["J"].mean()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    J_const = float(episode_objective(lat_eval, {
        "halt_bias": (-3 + 6 * torch.sigmoid(raw[0])).expand(B, T),
        "depth_max": (2 + 22 * torch.sigmoid(raw[1])).expand(B, T),
        "tool_gate": torch.sigmoid(raw[2]).expand(B, T),
        "budget_scale": (0.25 + 3.75 * torch.sigmoid(raw[3])).expand(B, T)})["J"].mean())
gap = J_const - J_oracle
print(f"J_oracle={J_oracle:.4f} J_const={J_const:.4f} gap={gap:.4f}")

# GRU privilegiado con evaluación en hitos
torch.manual_seed(0)
ctrl = BaselineController("gru").double().to(DEV)
opt = torch.optim.Adam(ctrl.parameters(), lr=3e-3)
for step in range(750):
    lat = sample_latents(cfg, 16, cfg.T, 555000 + step, DEV)
    out = _rollout_sig(ctrl, lat, lat, DEV, privileged=True)
    loss = out["J"].mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0)
    opt.step()
    if step + 1 in (250, 500, 750):
        ctrl.eval()
        with torch.no_grad():
            J = float(_rollout_sig(ctrl, lat_eval, lat_eval, DEV, privileged=True)["J"].mean())
        ctrl.train()
        print(f"  priv@{step+1}: J={J:.4f}  recovery={(J_const-J)/gap:.3f}", flush=True)
