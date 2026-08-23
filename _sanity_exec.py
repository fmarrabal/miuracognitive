"""Sanity F2X: ¿reward-to-go aprende a batir al designer? (1500 pasos)"""
import torch
from data.exec_v2 import ExecV2Config, ExecV2Dataset
from model.exec_v2 import (DesignerPolicy, LearnedExecConfig, LearnedExecPolicy,
                           eval_exec_policy, train_learned_exec)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = ExecV2Config()
ds = ExecV2Dataset(cfg, seed=50_300)
sc = ds.batch(512).to(dev)
r_des = eval_exec_policy(cfg, sc, DesignerPolicy(cfg, 0.1, 0.02), device=dev)
for steps in (4000,):
    torch.manual_seed(0)
    pol = LearnedExecPolicy(cfg, LearnedExecConfig())
    train_learned_exec(cfg, pol, seed=0, steps=steps, batch=256, device=dev)
    r = eval_exec_policy(cfg, sc, pol, device=dev)
    print(f"reward-to-go {steps} pasos: ret={r['return']:.3f} "
          f"ontime={r['completed_ontime']:.2f} colaps={r['collapses']:.2f} "
          f"regret={r['regret_vs_skyline']:.2f} | designer={r_des['return']:.3f}",
          flush=True)
