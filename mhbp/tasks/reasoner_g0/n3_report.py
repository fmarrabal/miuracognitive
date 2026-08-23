"""
N3 — Veredicto formal de la fase A (PREREG_N3 v2 §3).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.n3_report
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = range(3, 9)
DELTA0 = 0.02


def paired_t(d, mu0=0.0):
    n = len(d)
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    t = (m - mu0) / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    df = n - 1

    def tpdf(x):
        return math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / \
            math.sqrt(df * math.pi) * (1 + x * x / df) ** (-(df + 1) / 2)
    lo, hi, N = t, 40.0, 4000
    if lo >= hi:
        return m, sd, t, 0.0
    h = (hi - lo) / N
    s = tpdf(lo) + tpdf(hi)
    for i in range(1, N):
        s += tpdf(lo + i * h) * (4 if i % 2 else 2)
    return m, sd, t, max(0.0, min(1.0, s * h / 3))


def main():
    d = json.load(open(os.path.join(RES, "n3_eval.json"), encoding="utf-8"))
    ck = d["ckpts"]

    def seed_mean(arm, key="payoff"):
        out = []
        for s in SEEDS:
            v = [ck[f"n2_e1_blind_flat_s{s}_r{r}"][arm][key] for r in (0, 1)]
            out.append(np.mean(v))
        return np.array(out)

    pay = {a: seed_mean(a) for a in
           ("uniforme", "dificultad", "expl", "oraculo_clase", "regla",
            "nativo", "dificultad_at_nat", "expl_at_nat")}
    verdict = {"payoff_por_seed": {a: v.tolist() for a, v in pay.items()}}

    # A1: expl − dificultad > δ0 (t por seed) — el bootstrap por instancia se
    # omite como redundante dado el tamaño (declarado si Δ ≫ δ0 + 5·sd)
    dA1 = pay["expl"] - pay["dificultad"]
    m1, sd1, t1, p1 = paired_t(dA1, DELTA0)
    verdict["A1"] = {"delta": m1, "sd": sd1, "t_vs_delta0": t1, "p": p1,
                     "pasa": bool(p1 < 0.05 and m1 > DELTA0)}
    # A2 lineal: (expl−dif) − 0.5·(orac−dif) ≥ 0
    dA2 = (pay["expl"] - pay["dificultad"]) - 0.5 * (
        pay["oraculo_clase"] - pay["dificultad"])
    m2, sd2, t2, p2 = paired_t(dA2)
    frac = float(np.mean((pay["expl"] - pay["dificultad"])
                         / (pay["oraculo_clase"] - pay["dificultad"])))
    verdict["A2"] = {"contraste": m2, "p": p2, "fraccion_techo": frac,
                     "pasa": bool(p2 < 0.05 and m2 > 0)}
    # VG-N3d: dificultad@E[n]_nativo ≥ nativo − 2σ (σ de los diffs)
    dD = pay["dificultad_at_nat"] - pay["nativo"]
    verdict["VG_N3d"] = {"delta": float(dD.mean()),
                         "rojo": bool(dD.mean() < -2 * dD.std(ddof=1))}
    # A3 (gatekept por A1; re-etiquetado si VG-N3d rojo)
    dA3 = pay["expl_at_nat"] - pay["nativo"]
    m3, sd3, t3, p3 = paired_t(dA3)
    verdict["A3"] = {"delta": m3, "p_bilateral_aprox": p3,
                     "reetiquetado_por_N3d": verdict["VG_N3d"]["rojo"]}
    # suelo y descriptivos
    verdict["suelo_expl_vs_unif"] = float((pay["expl"] - pay["uniforme"]).mean())
    verdict["regla_vs_expl"] = float((pay["regla"] - pay["expl"]).mean())
    corr = {a: float(np.mean([ck[c][a]["corr_n_stake_K"] for c in ck]))
            for a in ("expl", "dificultad", "nativo")}
    verdict["corr_n_stake_K"] = corr

    print("=== N3 fase A — veredicto ===")
    for a in ("uniforme", "dificultad", "regla", "expl", "oraculo_clase",
              "nativo"):
        print(f"  {a:14s} payoff={pay[a].mean():.3f}")
    print(f"A1 expl−dif: Δ={m1:+.4f} (δ0={DELTA0}) t={t1:.1f} p={p1:.2e} → "
          f"{'PASA' if verdict['A1']['pasa'] else 'NO'}")
    print(f"A2: fracción del techo de clase capturada = {frac:.3f} "
          f"(contraste lineal p={p2:.2e}) → "
          f"{'PASA' if verdict['A2']['pasa'] else 'NO'}")
    print(f"VG-N3d: dif@nativo − nativo = {dD.mean():+.3f} → "
          f"{'ROJO (A3 re-etiquetado: ejecución, no decisión)' if verdict['VG_N3d']['rojo'] else 'verde'}")
    print(f"A3 expl@nativo − nativo: Δ={m3:+.3f}")
    print(f"routing corr(n,stake|K): expl={corr['expl']:+.2f} "
          f"dif={corr['dificultad']:+.2f} nativo={corr['nativo']:+.2f}")
    with open(os.path.join(RES, "n3_veredicto.json"), "w",
              encoding="utf-8") as f:
        json.dump(verdict, f, indent=1)
    print("Guardado n3_veredicto.json")


if __name__ == "__main__":
    main()
