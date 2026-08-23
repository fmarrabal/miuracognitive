"""
Informe confirmatorio del BENCHMARK v4 (PREREG_V4.md §3): H-ORDEN y
H-GRU, pareado por seed, IC-t (n=20), Holm-2 entre generadores dentro de
cada hipótesis; accuracy como no-inferioridad (margen 0.02).

Uso:  $env:PYTHONPATH="." ; python benchmark_report_v4.py
"""
import glob
import json
import math
import os

import numpy as np

OUT = "results_benchmark_v4"
GENS = ["adjacent", "cycle_transp"]
ARMS = ["hbp_full", "hbp_first_eq", "hbp_gru"]
SEEDS = list(range(20, 40))


def t_sf(t, df):
    """P(T>t) por la incompleta regularizada (sin scipy)."""
    x = df / (df + t * t)

    def betainc(a, b, x, n=400):
        # integración numérica simple de la beta incompleta regularizada
        from math import lgamma
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        s, h = 0.0, x / n
        for i in range(n):
            u = (i + 0.5) * h
            s += math.exp((a - 1) * math.log(u)
                          + (b - 1) * math.log(1 - u) - lbeta) * h
        return min(max(s, 0.0), 1.0)
    return 0.5 * betainc(df / 2.0, 0.5, x)


def carga():
    D = {}
    for p in glob.glob(os.path.join(OUT, "ood_*.json")):
        r = json.load(open(p, encoding="utf-8"))
        c = r["_cell"]
        cd = r.get("compute_diag") or {}
        corr = cd.get("corr_K_niter_ood", cd.get("corr_K_niter"))
        D[(c["gens"], c["variant"], c["seed"])] = {
            "corr": corr, "acc": r["final_acc"]["largo"]}
    return D


def pareado(D, gens, a, b, campo):
    d = np.array([D[(gens, a, s)][campo] - D[(gens, b, s)][campo]
                  for s in SEEDS])
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(len(d))
    t = m / se if se > 0 else 0.0
    p = t_sf(abs(t), len(d) - 1)          # una cola (direccional prereg)
    return m, se, t, p, d


def main():
    D = carga()
    falt = [(g, v, s) for g in GENS for v in ARMS for s in SEEDS
            if (g, v, s) not in D]
    assert not falt, f"faltan celdas: {falt[:5]}"
    print("=== medias corr_OOD por brazo ===")
    for g in GENS:
        for v in ARMS:
            vals = [D[(g, v, s)]["corr"] for s in SEEDS]
            acc = [D[(g, v, s)]["acc"] for s in SEEDS]
            print(f"  {g:13s} {v:14s} corr={np.mean(vals):+.3f}"
                  f"±{np.std(vals, ddof=1) / math.sqrt(20):.3f} "
                  f"acc_largo={np.mean(acc):.3f}")
    res = {}
    for nombre, rival in (("H-ORDEN", "hbp_first_eq"),
                          ("H-GRU", "hbp_gru")):
        ps = []
        print(f"\n=== {nombre}: hbp_full − {rival} (corr_OOD, "
              f"pareado n=20, una cola) ===")
        for g in GENS:
            m, se, t, p, d = pareado(D, g, "hbp_full", rival, "corr")
            ic = (m - 2.093 * se, m + 2.093 * se)
            ps.append((g, m, se, t, p, ic))
            print(f"  {g:13s} Δ={m:+.4f} IC95 [{ic[0]:+.4f},{ic[1]:+.4f}] "
                  f"t={t:+.2f} p(1c)={p:.4f}")
        # Holm-2: ordenar p, umbral 0.025 / 0.05
        orden = sorted(ps, key=lambda x: x[4])
        pasa = []
        umbrales = [0.025, 0.05]
        holm_ok = True
        for k, (g, m, se, t, p, ic) in enumerate(orden):
            ok = holm_ok and (p < umbrales[k]) and m > 0
            pasa.append((g, ok))
            if not ok:
                holm_ok = False
        veredicto = all(ok for _, ok in pasa)
        print(f"  Holm-2: " + " ".join(f"{g}:{'✓' if ok else '✗'}"
                                       for g, ok in pasa)
              + f" → {nombre} {'CONFIRMADA' if veredicto else 'NO confirmada'}")
        res[nombre] = {"por_gens": {g: {"delta": m, "se": se, "t": t,
                                        "p1c": p, "ic95": list(ic)}
                                    for g, m, se, t, p, ic in ps},
                       "holm": dict(pasa), "confirmada": veredicto}
    print("\n=== accuracy (no-inferioridad de hbp_full, margen 0.02) ===")
    for rival in ("hbp_first_eq", "hbp_gru"):
        for g in GENS:
            m, se, t, p, d = pareado(D, g, "hbp_full", rival, "acc")
            ic_lo = m - 2.093 * se
            ni = ic_lo > -0.02
            print(f"  vs {rival:14s} {g:13s} Δacc={m:+.4f} "
                  f"IC_lo={ic_lo:+.4f} → {'no-inferior' if ni else 'REVISAR'}")
    with open(os.path.join(OUT, "report_v4.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=lambda o: bool(o) if hasattr(o, "item") else str(o))
    print("\nGuardado report_v4.json")


if __name__ == "__main__":
    main()
