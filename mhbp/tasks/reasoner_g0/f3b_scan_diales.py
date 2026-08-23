"""Escaneo del dial-space (B_total × stake_high) con las políticas de oráculo
de f3b_gates — INSTRUMENTACIÓN para elegir la ronda 2 de calibración (§10):
solo se commitea el spec ganador; ninguna política aprendida interviene."""
import dataclasses
import numpy as np

from mhbp.tasks.reasoner_g0.f3b_env import SessionSpec, gen_session, seed_env_train
from mhbp.tasks.reasoner_g0.f3b_gates import (
    load_profile, POLITICAS, eval_politica, _stationary_demand)

acc, d_ref, origen = load_profile(False)
print(f"Perfil: {origen}")
base = SessionSpec()
dem = _stationary_demand(base, d_ref) * base.E
print(f"Demanda de sesión (rangos ronda 1): {dem:.1f}\n")
N_MC = 1500

print(f"{'B':>4} {'sh':>3} | {'uni':>6} {'stakeg':>6} {'contad':>6} "
      f"{'bayes':>6} {'orac':>6} | {'headroom':>8} {'bay-ctd':>8} {'mordida':>7}")
GRID = ([(B, 4, 1 / 3) for B in (35, 40, 45, 50, 56)]
        + [(B, 2, 1 / 3) for B in (35, 40, 45)]
        + [(45, 4, 0.2), (45, 4, 0.45), (40, 4, 0.2), (40, 4, 0.45),
           (35, 4, 0.2), (35, 4, 0.45)])
for B, sh, psw in GRID:
        spec = dataclasses.replace(base, B_total=B, stake_high=sh,
                                   p_switch=psw,
                                   n_high=0 if sh == 1 else base.n_high)
        se = seed_env_train(0)
        sessions = [gen_session(spec, se, i, materialize=False)
                    for i in range(N_MC)]
        res = {}
        for name in ("uniforme", "stake_greedy", "contador", "bayes",
                     "oraculo"):
            sc, fz = eval_politica(spec, sessions, POLITICAS[name], acc, d_ref)
            res[name] = (sc.mean(), fz.mean())
        blind = max(res["uniforme"][0], res["stake_greedy"][0],
                    res["contador"][0])
        head = res["oraculo"][0] - blind
        bayctd = res["bayes"][0] - res["contador"][0]
        print(f"{B:>4} {sh:>3} p{psw:.2f} | " +
              " ".join(f"{res[n][0]:.4f}" for n in
                       ("uniforme", "stake_greedy", "contador", "bayes",
                        "oraculo")) +
              f" | {head:+.4f} {bayctd:+.4f} {res['uniforme'][1]*100:6.1f}%")
