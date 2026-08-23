"""
N2 — Análisis pre-declarado de la Etapa 1 (PREREG_N2 v2 §5, jerarquía
gatekeeping). Genera results/n2_e1_veredicto.json y la tabla.

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n2_e1_report
"""
import glob
import itertools
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = [3, 4, 5, 6, 7, 8]


def load(arm):
    out = {}
    for s in SEEDS:
        runs = []
        for r in (0, 1):
            p = os.path.join(RES, f"n2_e1_{arm}_s{s}_r{r}.json")
            runs.append(json.load(open(p, encoding="utf-8")))
        out[s] = runs
    return out


def seed_means(cells, key):
    return np.array([np.mean([r[key] for r in cells[s]]) for s in SEEDS])


def paired_one_sided(d):
    n = len(d)
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    # p unilateral por integración de la densidad t (df = n−1)
    df = n - 1

    def tpdf(x):
        return math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / \
            math.sqrt(df * math.pi) * (1 + x * x / df) ** (-(df + 1) / 2)
    lo, hi, N = t, 40.0, 4000
    if lo >= hi:
        return t, 0.0, m, sd
    h = (hi - lo) / N
    ssum = tpdf(lo) + tpdf(hi)
    for i in range(1, N):
        ssum += tpdf(lo + i * h) * (4 if i % 2 else 2)
    return t, max(0.0, min(1.0, ssum * h / 3)), m, sd


def wilcoxon_one_sided(d):
    """Wilcoxon signed-rank exacto (n=6) unilateral (mediana > 0)."""
    d = [x for x in d if x != 0]
    n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    w_pos = sum(r for r, x in zip(ranks, d) if x > 0)
    tot = 0
    ge = 0
    for signs in itertools.product([0, 1], repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        tot += 1
        if w >= w_pos:
            ge += 1
    return float(w_pos), ge / tot


def main():
    arms = {a: load(a) for a in ("endo", "endo_noval", "blind_flat")}
    pay = {a: seed_means(c, "payoff_norm") for a, c in arms.items()}
    verdict = {"seeds": SEEDS,
               "payoff_por_seed": {a: v.tolist() for a, v in pay.items()}}

    # C1' (primario): endo > endo_noval, pareado por seed, unilateral
    d = pay["endo"] - pay["endo_noval"]
    t, p, m, sd = paired_one_sided(d)
    w, pw = wilcoxon_one_sided(d.tolist())
    verdict["C1p"] = {"diffs": d.tolist(), "delta": m, "sd": sd, "t": t,
                      "p_unilateral": p, "wilcoxon_p": pw,
                      "pasa": bool(p < 0.05 and m > 0)}
    print(f"C1' endo>endo_noval: Δ={m:+.4f} sd={sd:.4f} t={t:.2f} "
          f"p={p:.3f} (Wilcoxon p={pw:.3f}) → "
          f"{'PASA' if verdict['C1p']['pasa'] else 'NO'}")

    # C3: condicionado a C1' (gatekeeping) — no se testea si C1' no pasa
    if verdict["C1p"]["pasa"]:
        print("C3 se testearía (C1' pasó)")
    else:
        verdict["C3"] = "NO_TESTEADO (gatekeeping: C1' no pasó)"
        print("C3: no se testea (gatekeeping)")

    # Suelo declarado (descriptivo): endo vs blind_flat — RAMA NO DECLARADA
    # si blind domina (el prereg asumió lo contrario)
    d2 = pay["endo"] - pay["blind_flat"]
    t2, p2, m2, sd2 = paired_one_sided(d2)
    verdict["suelo_blind"] = {"delta_endo_menos_blind": m2, "sd": sd2,
                              "nota": "NEGATIVO GRANDE = rama no declarada"}
    print(f"endo − blind_flat: Δ={m2:+.4f} (sd {sd2:.4f}) — "
          f"{'RAMA NO DECLARADA (blind domina)' if m2 < 0 else 'ok'}")

    # Descriptivos pre-declarados
    for key in ("acc", "acc_alto", "acc_bajo", "E_n", "E_n_alto", "E_n_bajo",
                "corr_n_stake_dado_K"):
        verdict.setdefault("descriptivos", {})[key] = {
            a: float(seed_means(c, key).mean()) for a, c in arms.items()}
    ds = verdict["descriptivos"]
    print(f"\nrouting (corr|K): " +
          " ".join(f"{a}={ds['corr_n_stake_dado_K'][a]:+.3f}"
                   for a in arms))
    print(f"E[n] alto/bajo endo: {ds['E_n_alto']['endo']:.2f}/"
          f"{ds['E_n_bajo']['endo']:.2f} — acc_alto: "
          f"endo {ds['acc_alto']['endo']:.3f} vs blind "
          f"{ds['acc_alto']['blind_flat']:.3f}")

    # AUC de V̂ (enganche del sensor; blind_flat: nota descriptiva)
    auc = {a: seed_means(c, "auc_vhat_stake") for a, c in arms.items()
           if "auc_vhat_stake" in list(arms[a].values())[0][0]}
    verdict["auc_vhat"] = {a: v.tolist() for a, v in auc.items()}
    print(f"AUC_V medio: " +
          " ".join(f"{a}={v.mean():.3f}" for a, v in auc.items()))

    verdict["rama"] = (
        "C1' NULO + rama NO DECLARADA (blind_flat domina a los brazos "
        "ponderados incluso en acc_alto): (1) el acoplamiento valor→gobierno "
        "no añade NADA sobre las consecuencias-en-la-pérdida (canal vivo "
        "AUC~0.94, gobernador sordo corr|K~0, endo≡endo_noval); (2) las "
        "consecuencias-en-la-pérdida DAÑAN (ESS 0.40): la respuesta racional "
        "al pago 8× es competencia general, no énfasis de gradiente; (3) "
        "ningún brazo enruta cómputo por valor. Etapa 2 NO corre "
        "(gatekeeping). Guardarraíl M: sin objeto (nada se adopta) — celdas "
        "ood omitidas, desviación declarada (ahorro sobre presupuesto).")
    with open(os.path.join(RES, "n2_e1_veredicto.json"), "w",
              encoding="utf-8") as f:
        json.dump(verdict, f, indent=1)
    print("\nGuardado n2_e1_veredicto.json")


if __name__ == "__main__":
    main()
