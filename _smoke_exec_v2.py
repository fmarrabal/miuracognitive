"""Smoke CPU de F2X: mundo, triaje imposible-completo, energía/colapso,
scripted, aprendido (REINFORCE)."""
import torch
from data.exec_v2 import ExecV2Config, ExecV2Dataset, portfolio_skyline
from model.exec_v2 import (DesignerPolicy, EDFPolicy, LearnedExecConfig,
                           LearnedExecPolicy, ValueDensityPolicy,
                           eval_exec_policy, train_learned_exec)

cfg = ExecV2Config()
ds = ExecV2Dataset(cfg, seed=0)
sc = ds.batch(64)
sky = portfolio_skyline(cfg, sc)
cap = int(cfg.horizon * cfg.energy_recharge
          / (cfg.energy_drain + cfg.energy_recharge))
per = int(1 / cfg.work_rate) + 1
print(f"(1) mundo: capacidad≈{cap} ticks ({cap // per} proyectos de "
      f"{cfg.n_projects}) -> triaje FORZADO; skyline={float(sky.mean()):.2f}")

for name, pol in (("value_density", ValueDensityPolicy(cfg, 0.2)),
                  ("edf", EDFPolicy(cfg, 0.2)),
                  ("designer", DesignerPolicy(cfg, 0.2, 0.05))):
    r = eval_exec_policy(cfg, sc, pol)
    print(f"(2) {name:14s} ret={r['return']:+.3f} ontime={r['completed_ontime']:.2f} "
          f"colaps={r['collapses']:.2f} regret={r['regret_vs_skyline']:.2f}")

pol = LearnedExecPolicy(cfg, LearnedExecConfig())
r0 = eval_exec_policy(cfg, sc, pol)
train_learned_exec(cfg, pol, seed=0, steps=8, batch=64)
r1 = eval_exec_policy(cfg, sc, pol)
print(f"(3) learned: sin entrenar ret={r0['return']:+.3f}; tras 8 pasos "
      f"ret={r1['return']:+.3f} (corre, gradientes fluyen)")

# (3b) INVARIANTE: Σ_t recompensa incremental == retorno final (mismo objetivo)
from model.exec_v2 import rollout_learned
torch.manual_seed(1)
pol_i = LearnedExecPolicy(cfg, LearnedExecConfig())
pol_i.reset(sc.batch_size, "cpu")
out = rollout_learned(cfg, sc, pol_i, sample=True, device="cpu")
gap = float((out["rewards_t"].sum(1) - out["return"]).abs().max())
print(f"(3b) invariante Σr_t == retorno_final: gap_max={gap:.2e} "
      f"({'OK' if gap < 1e-4 else 'VIOLADO — revisar'})")

ph = LearnedExecPolicy(cfg, LearnedExecConfig(use_hbp=True, hbp_alpha_const=1.0))
train_learned_exec(cfg, ph, seed=0, steps=3, batch=32)
print("(4) brazo HBP: construye y entrena (pin_fp32, ganancia aprendible)")
print("SMOKE COMPLETO")
