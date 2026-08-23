"""Smoke CPU de AHA-2: escenario, scripted, aprendido (grad), brazos HBP."""
import torch
from data.aha_v2 import AHA2Config, AHA2Dataset, rollout_world
from model.aha_v2 import (ComboPolicy, CueFollowerPolicy, LearnedAHA2Policy,
                          LearnedPolicyConfig, ThresholdPolicy,
                          evaluate_policy, homeostatic_loss, train_learned)

cfg = AHA2Config()
cfg.validate()
ds = AHA2Dataset(cfg, seed=0)
sc = ds.batch(32)
n_ev = sc.events_t.numel()
print(f"(1) escenario OK: {n_ev} eventos; horizon={sc.horizon}; "
      f"masa de cue total={float(sc.cues.sum()):.2f} vs masa de hazard="
      f"{float(sc.disturbances.sum()):.2f} (cues!=hazards por ruido/fiabilidad)")

for name, pol in [("threshold", ThresholdPolicy(cfg, 0.55)),
                  ("cue_follower", CueFollowerPolicy(cfg, 0.1, 3)),
                  ("combo", ComboPolicy(cfg, 0.55, 0.1, 3))]:
    r = evaluate_policy(cfg, sc, pol)
    print(f"(2) {name:12s} surv={r['survival']:.3f} "
          f"lead={r['anticipation_lead']:.2f} act={r['action_rate']:.3f}")

pol = LearnedAHA2Policy(cfg, LearnedPolicyConfig())
pol.reset(32, "cpu")
out = rollout_world(cfg, sc, pol, mode="st")
loss = homeostatic_loss(cfg, out["levels_hist"])
loss.backward()
g = sum(float(p.grad.abs().sum()) for p in pol.parameters() if p.grad is not None)
print(f"(3) learned soft: loss={float(loss):.4f} |grad|={g:.4f} (fluye={g > 0})")
pol2 = LearnedAHA2Policy(cfg, LearnedPolicyConfig())
train_learned(cfg, pol2, seed=0, steps=5, batch=32)
print("(3) 5 pasos de entrenamiento OK")

for a in (1.0, 0.0):
    p = LearnedAHA2Policy(cfg, LearnedPolicyConfig(use_hbp=True, hbp_alpha_const=a))
    p.reset(8, "cpu")
    sc8 = ds.batch(8)
    o = rollout_world(cfg, sc8, p, mode="st")
    hl = homeostatic_loss(cfg, o["levels_hist"]) + 0.1 * p.hbp.stability_penalty()
    hl.backward()
    print(f"(4) hbp alpha={a}: OK, grad a la ganancia={p.hbp_gain.grad is not None}")

# (5) barrido de delay: el reactivo debe DEGRADARSE con el delay (no morir en 0)
for d in (0, 1, 2, 3):
    c2 = cfg.with_delay(d)
    ds2 = AHA2Dataset(c2, seed=7)
    s2 = ds2.batch(256)
    r = evaluate_policy(c2, s2, ThresholdPolicy(c2, 0.55))
    print(f"(5) threshold@delay={d}: surv={r['survival']:.3f}")
print("SMOKE COMPLETO")
