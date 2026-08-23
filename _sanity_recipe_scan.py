"""Scan de receta REINFORCE para AHA-2 @delay=2: pasos × entropía."""
import torch
from data.aha_v2 import AHA2Config, AHA2Dataset, rollout_world
from model.aha_v2 import (LearnedAHA2Policy, LearnedPolicyConfig,
                          evaluate_policy, trajectory_reward)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = AHA2Config().with_delay(2)


def train(seed, steps, ent_w, lr=1e-3, batch=256):
    import math
    torch.manual_seed(seed)
    ds = AHA2Dataset(cfg, seed=seed)
    pol = LearnedAHA2Policy(cfg, LearnedPolicyConfig()).to(dev)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=0.01)
    pol.train()
    for s in range(1, steps + 1):
        frac = s / steps
        cur = lr * min(1.0, s / 50) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = cur
        sc = ds.batch(batch).to(dev)
        pol.reset(batch, dev)
        out = rollout_world(cfg, sc, pol, mode="sample")
        r = trajectory_reward(cfg, out)
        adv = r - r.mean()
        loss = -(adv.detach() * out["logp"]).mean() - ent_w * out["entropy"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
    return pol


ds_ev = AHA2Dataset(cfg, seed=50_200)
sc_ev = ds_ev.batch(512).to(dev)
for steps in (800, 1500):
    for ent in (0.01, 0.03):
        pol = train(0, steps, ent)
        r = evaluate_policy(cfg, sc_ev, pol)
        print(f"steps={steps} ent={ent}: surv={r['survival']:.3f} "
              f"act={r['action_rate']:.3f} lead={r['anticipation_lead']:.2f}",
              flush=True)
print("(listones @d2: threshold 0.335, combo 0.561)")
