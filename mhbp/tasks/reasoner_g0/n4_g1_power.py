"""
N4 — control POSITIVO del instrumento G1 (lección del capítulo LLM: un
nulo solo vale si el instrumento tenía potencia para ver el efecto).

Simula el escenario exacto de la celda primaria (n=1024, 7 dims last vs
42 dims stream, mismas rutinas logistic_cv/auc_rank) con una señal
temporal DISTRIBUIDA inyectada: un latente u que no aparece en el tick 8
pero sí repartido con cargas pequeñas en los ticks 1..7 — el caso que el
campo-acumulador explotaría y el probe de último tick no puede ver.
Barrido de la carga → mapa carga↦ΔAUC recuperado. Si el pipeline recupera
ΔAUC ≥ 0.03 con potencia, el nulo real es del sustrato, no del probe.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n4_g1_power
"""
import numpy as np

from .n4_g1 import probe_auc

N, REPS = 1024, 12
RNG = np.random.default_rng(20260810)


def sim(carga, rng):
    u = rng.normal(size=N)                       # latente temporal
    base = rng.normal(size=N)                    # señal del último tick
    logit = 1.2 * base + carga * 3.0 * u - 0.6
    y = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(float)
    # last(t=8): 7 dims — ven base con ruido, NO ven u
    Fl = np.stack([base + rng.normal(size=N) * 0.8 for _ in range(7)], 1)
    # stream: los mismos 7 dims del tick 8 + 35 dims de ticks 1..7 con u
    # repartido en cargas pequeñas (solo agregando ticks se recupera)
    Fs = np.concatenate(
        [Fl] + [np.stack([0.3 * u + rng.normal(size=N) for _ in range(5)], 1)
                for _ in range(7)], axis=1)
    return probe_auc(Fs, y) - probe_auc(Fl, y)


def main():
    for carga in (0.0, 0.05, 0.10, 0.20):
        d = np.array([sim(carga, np.random.default_rng(1000 + r))
                      for r in range(REPS)])
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(REPS)
        print(f"carga={carga:.2f}: ΔAUC recuperado = {m:+.4f} ± {se:.4f} "
              f"(IC-t95 [{m - 2.201 * se:+.4f}, {m + 2.201 * se:+.4f}])",
              flush=True)


if __name__ == "__main__":
    main()
