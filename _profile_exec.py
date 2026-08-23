"""Perfil: coste por entrenamiento de cada brazo F2X (100 pasos -> extrapolar)."""
import time, torch
from data.exec_v2 import ExecV2Config
from model.exec_v2 import LearnedExecConfig, LearnedExecPolicy, train_learned_exec

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = ExecV2Config()
for arm, pc in (("learned", LearnedExecConfig()),
                ("memoryless", LearnedExecConfig(memoryless=True)),
                ("hbp_wave", LearnedExecConfig(use_hbp=True, hbp_alpha_const=1.0))):
    torch.manual_seed(0)
    pol = LearnedExecPolicy(cfg, pc)
    t0 = time.time()
    train_learned_exec(cfg, pol, seed=0, steps=100, batch=256, device=dev)
    if dev != "cpu":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"{arm:12s}: {dt:.1f}s/100pasos -> {dt*25/60:.1f} min a 2500 pasos",
          flush=True)
