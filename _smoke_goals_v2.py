"""Smoke CPU de fase 2 v2: escenario, retorno, scripted, REINFORCE."""
import torch
from data.goals_v2 import Goals2Config, Goals2Dataset, episode_return, urgency_signal
from model.goals_v2 import (LearnedGoalConfig, LearnedGoalPolicy,
                            ScriptedGoalPolicy, eval_goal_policy,
                            train_learned_goals)

cfg = Goals2Config()
ds = Goals2Dataset(cfg, seed=0)
sc = ds.batch(64)
u = urgency_signal(cfg, sc)
print(f"(1) escenario: u en [{float(u.min()):.2f},{float(u.max()):.2f}]; "
      f"crisis={int(sc.spike_is_crisis.sum())} distractores="
      f"{int((~sc.spike_is_crisis).sum())}")

# aliasing: en el tick del pico, crisis y distractor deben verse IGUAL (=1.0)
vals = []
for row in range(8):
    for s in range(sc.spike_t.shape[1]):
        vals.append(float(u[row, int(sc.spike_t[row, s]), int(sc.spike_proj[row, s])]))
print(f"(2) aliasing: valor de u en TODOS los picos = {set(round(v,2) for v in vals)}")

for mode, kw in (("greedy", {"w": 0.5}), ("hysteresis", {"w": 0.5, "margin": 0.1}),
                 ("smart", {"w": 0.5, "margin": 0.1})):
    r = eval_goal_policy(cfg, sc, ScriptedGoalPolicy(cfg, mode, **kw))
    print(f"(3) {mode:10s} ret={r['return']:+.3f} crisis={r['crisis_attended']:.2f} "
          f"distr={r['distractor_attended']:.2f} switch={r['switch_rate']:.2f}")

pol = LearnedGoalPolicy(cfg, LearnedGoalConfig())
train_learned_goals(cfg, pol, seed=0, steps=8, batch=64)
r = eval_goal_policy(cfg, sc, pol)
print(f"(4) learned (8 pasos REINFORCE): ret={r['return']:+.3f} (corre y aprende algo)")
print("SMOKE COMPLETO")
