"""Sanity: ¿REINFORCE aprende algo evaluable por argmax en AHA-2? (300 pasos)"""
import torch
from data.aha_v2 import AHA2Config, AHA2Dataset
from model.aha_v2 import (LearnedAHA2Policy, LearnedPolicyConfig,
                          evaluate_policy, train_learned)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = AHA2Config().with_delay(2)
ds = AHA2Dataset(cfg, seed=50_200)
sc = ds.batch(512).to(dev)
for name, pc in (("learned", LearnedPolicyConfig()),
                 ("cueblind", LearnedPolicyConfig(cue_blind=True))):
    torch.manual_seed(0)
    pol = LearnedAHA2Policy(cfg, pc)
    train_learned(cfg, pol, seed=0, steps=1000, batch=256, device=dev)
    r = evaluate_policy(cfg, sc, pol)
    print(f"{name:9s} 1000 pasos @d2: surv={r['survival']:.3f} "
          f"lead={r['anticipation_lead']:.2f} act={r['action_rate']:.3f}",
          flush=True)
