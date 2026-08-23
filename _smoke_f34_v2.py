"""Smoke CPU de fases 3 y 4 v2."""
import torch
from data.discovery_v2 import Discovery2Config, FieldBatch, normalized_regret
from model.discovery_v2 import (FDAscent, GridProbe, LearnedProber, RandomProbe,
                                eval_prober, train_prober)
from model.self_model_v2 import SelfModel2Config, run_episode

print("=== FASE 3 ===")
cfg = Discovery2Config()
gen = torch.Generator().manual_seed(0)
fields = FieldBatch(cfg, 64, gen)
fm = fields.f_max()
print(f"(1) campos: familias={fields.family.bincount(minlength=4).tolist()} "
      f"f_max en [{float(fm.min()):.2f},{float(fm.max()):.2f}]")
for name, agent in (("random", RandomProbe(cfg)), ("grid", GridProbe(cfg)),
                    ("fd", FDAscent(cfg, k_init=4, step=0.1))):
    r = eval_prober(cfg, agent, seed=1, batch=256)
    print(f"(2) {name:8s} regret={r['regret']:.3f}")
pro = LearnedProber(cfg)
r0 = eval_prober(cfg, pro, seed=1, batch=256)
train_prober(cfg, pro, seed=0, steps=10, batch=64)
r1 = eval_prober(cfg, pro, seed=1, batch=256)
print(f"(3) prober: sin entrenar={r0['regret']:.3f} tras 10 pasos={r1['regret']:.3f} "
      f"(gradiente meta fluye)")

print("=== FASE 4 ===")
c4 = SelfModel2Config()
for arm in ("random", "oracle", "adaptive", "frozen"):
    r = run_episode(c4, batch=64, seed=3, arm=arm, reinit_threshold=0.05)
    print(f"(4) {arm:9s} cost_pre={r['cost_pre']:.3f} cost_post={r['cost_post']:.3f} "
          f"err_fin={r['pred_err_final']:.4f}")
print("SMOKE COMPLETO")
