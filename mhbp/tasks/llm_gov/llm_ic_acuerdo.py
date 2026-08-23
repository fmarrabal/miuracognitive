"""
Revisión TMLR — ICs bootstrap PAREADOS para los deltas de VG-L3
(posterior−exante y valor|posterior) que el paper citaba como
estimaciones puntuales (−0.005/−0.002/−0.003). Regla de la casa aplicada
al propio §6: ningún veredicto de signo sin intervalo.

Bootstrap por instancia (B=200): remuestrea las 768 instancias del caché
T13; por réplica reconstruye tablas ex-ante y DP posterior con la MISMA
maquinaria de llm_gates y evalúa los deltas en los tres presupuestos.
walks y curves se computan UNA vez y se re-indexan (válido: son
por-instancia).

  PYTHONPATH=. python -m mhbp.tasks.llm_gov.llm_ic_acuerdo
"""
import json
import os

import numpy as np

from .llm_gates import (N_GRID, class_tables, dp_posterior, frontera_exante,
                        interp_frontera, load_cache, posterior_walks,
                        vote_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
B = 200
RNG = np.random.default_rng(20260817)
LAMS = np.logspace(-3.5, 0.5, 25)
LAMS_P = np.concatenate([[0.0], np.logspace(-6, -1.5, 25)])


def deltas(data, curves, walks, idx):
    d_b = [data[i] for i in idx]
    c_b = [curves[i] for i in idx]
    w_b = [walks[i] for i in idx]
    tab = class_tables(d_b, c_b)
    f_val = frontera_exante(tab, True, LAMS)
    fit = np.arange(len(idx))[::2]
    f_post = dp_posterior(w_b, fit, False, LAMS_P)
    f_pv = dp_posterior(w_b, fit, True, LAMS_P)
    t_med = float(np.mean([d["t_med"] for d in d_b]))
    out = []
    for mult in (3, 5, 7):
        b = mult * t_med
        p_post = interp_frontera(f_post, b)
        out.append((p_post - interp_frontera(f_val, b),
                    interp_frontera(f_pv, b) - p_post))
    return out


def main():
    data = load_cache("sonda", 0, "T13")
    print(f"{len(data)} instancias; curvas+walks…", flush=True)
    curves = [vote_curve(d) for d in data]
    walks = posterior_walks(data, curves)
    punto = deltas(data, curves, walks, np.arange(len(data)))
    print("puntuales Δ(post−exante) / Δ(valor|post): " +
          " ".join(f"n̄={m}: {a:+.4f}/{b:+.4f}"
                   for m, (a, b) in zip((3, 5, 7), punto)), flush=True)
    boots = []
    for r in range(B):
        idx = RNG.integers(0, len(data), len(data))
        boots.append(deltas(data, curves, walks, idx))
        if (r + 1) % 25 == 0:
            print(f"[{r + 1}/{B}]", flush=True)
    boots = np.array(boots)                       # (B, 3, 2)
    res = {"puntuales": punto, "B": B, "ic": {}}
    for j, m in enumerate((3, 5, 7)):
        for k, nombre in enumerate(("post_menos_exante", "valor_dado_post")):
            v = boots[:, j, k]
            res["ic"][f"nbar{m}_{nombre}"] = {
                "punto": float(punto[j][k]),
                "ic95": [float(np.percentile(v, 2.5)),
                         float(np.percentile(v, 97.5))],
                "se": float(v.std(ddof=1))}
            ic = res["ic"][f"nbar{m}_{nombre}"]["ic95"]
            print(f"n̄={m} {nombre}: {punto[j][k]:+.4f} "
                  f"IC95 [{ic[0]:+.4f}, {ic[1]:+.4f}]", flush=True)
    with open(os.path.join(RES, "llm_ic_acuerdo.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("Guardado llm_ic_acuerdo.json", flush=True)


if __name__ == "__main__":
    main()
