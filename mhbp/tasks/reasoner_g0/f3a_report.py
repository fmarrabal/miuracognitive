"""
F3a — Veredicto (PREREG_F3A v2). Estimación con IC pareado + márgenes;
tests como confirmación solo si el efecto es enorme (MDE dz≈1.05, n=6).

  PYTHONPATH=. python -m mhbp.tasks.reasoner_g0.f3a_report
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GENS = "cycle_transp"
SEEDS = [0, 1, 2, 3, 4, 5]

R = []
def w(s=""):
    R.append(s)


def jload(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def paired_ci(diff, n_boot=5000):
    d = np.asarray(diff, dtype=float)
    boot = np.random.default_rng(0).choice(d, size=(n_boot, len(d)),
                                           replace=True).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    from scipy import stats
    t, p = stats.ttest_rel(d, np.zeros_like(d)) if len(d) >= 3 else (np.nan, np.nan)
    dz = d.mean() / (d.std(ddof=1) + 1e-12)
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)],
            "t": float(t), "p": float(p), "dz": float(dz)}


def main():
    # --- métricas por variante y seed ---
    P, E, ACC = {}, {}, {}
    for v in ("miura_mhbp", "hbp_full", "gating_wm"):
        P[v], E[v], ACC[v] = [], [], []
        for s in SEEDS:
            o = jload(f"{v}_{GENS}_ood_s{s}.json")
            i = jload(f"{v}_{GENS}_indist_s{s}.json")
            P[v].append((o.get("compute_diag") or {}).get("corr_K_niter_ood")
                        if o else None)
            E[v].append(((i.get("compute_diag") or {}).get("largo") or {})
                        .get("acc_per_niter") if i else None)
            ACC[v].append(i["final_acc"]["largo"] if i else None)
    IP = {}
    for v in ("hbp_full", "gating_wm"):
        IP[v] = [(jload(f"f3a_iprime_{v}_s{s}.json") or {}).get("iprime")
                 for s in SEEDS]
    mips = [jload(f"f3a_mip_miura_mhbp_s{s}.json") for s in SEEDS]
    IP["miura_mhbp"] = [(m or {}).get("Iprime", {}).get("iprime") for m in mips]

    w("# F3a — Veredicto (integración mHBP↔reasoner, cycle_transp, n=6)\n")
    w("## Métricas por variante (media±std entre seeds)\n")
    w("| variante | P (corrOOD) | E (acc/n_iter largo) | acc largo | I' |")
    w("|---|---|---|---|---|")
    for v in ("miura_mhbp", "hbp_full", "gating_wm"):
        f = lambda xs: (f"{np.mean([x for x in xs if x is not None]):.3f}"
                        f"±{np.std([x for x in xs if x is not None]):.3f}"
                        if any(x is not None for x in xs) else "—")
        w(f"| {v} | {f(P[v])} | {f(E[v])} | {f(ACC[v])} | {f(IP[v])} |")
    w("")

    # --- contrastes primarios (estimación) ---
    w("## Contrastes primarios (Δ = mhbp − hbp_full; pareado por seed)\n")
    w("| contraste | Δ medio | IC95 | t | p | dz | no-inferioridad |")
    w("|---|---|---|---|---|---|---|")
    verdict_ok = {}
    for name, m_v, margin_kind in (("C1-P", P, "P"), ("C1-E", E, "E")):
        d = [a - b for a, b in zip(m_v["miura_mhbp"], m_v["hbp_full"])
             if a is not None and b is not None]
        st = paired_ci(d)
        base = np.mean([x for x in m_v["hbp_full"] if x is not None])
        if margin_kind == "P":
            noninf = st["mean"] >= -0.05
            marg = "Δ≥−0.05"
        else:
            noninf = np.mean([x for x in m_v["miura_mhbp"] if x is not None]) \
                >= 0.9 * base
            marg = "mhbp≥0.9·inc"
        verdict_ok[name] = noninf
        w(f"| {name} | {st['mean']:+.4f} | [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}] "
          f"| {st['t']:.2f} | {st['p']:.3f} | {st['dz']:.2f} "
          f"| {marg}: {'✓' if noninf else '✗'} |")
    w("")

    # --- guardarraíles ---
    w("## Guardarraíles\n")
    acc_d = [a - b for a, b in zip(ACC["miura_mhbp"], ACC["hbp_full"])]
    g_acc = np.mean(acc_d) >= -0.03
    ip_d = [a - b for a, b in zip(IP["miura_mhbp"], IP["hbp_full"])
            if a is not None and b is not None]
    g_ip = (np.mean(ip_d) >= -0.05) if ip_d else False
    m_rows, g_m = [], True
    for s, m in zip(SEEDS, mips):
        if not m:
            g_m = False
            continue
        m1 = m["M1"]; m2 = m["M2"]; m3 = m["M3"]
        fa = np.mean([m2[t]["follow_donor"] + m2[t]["rederive_cost"]
                      for t in m2])
        facc = fa / max(m1["output_acc"], 1e-9)
        ok = (m1["trend_onpolicy"] >= 0.9 and facc >= 0.95
              and m3["corr_onpolicy"] >= 0.6)
        g_m = g_m and ok
        m_rows.append((s, m1["trend_onpolicy"], facc, m3["corr_onpolicy"], ok))
    w("| seed | M1 trend(onp) | M2 follow/acc | M3 corr(onp) | M ok |")
    w("|---|---|---|---|---|")
    for s, tr, fa, m3c, ok in m_rows:
        w(f"| {s} | {tr:+.2f} | {fa:.2f} | {m3c:+.2f} | {'✓' if ok else '✗'} |")
    w("")
    w(f"- Accuracy largo: Δ medio = {np.mean(acc_d):+.4f} (umbral ≥−0.03): "
      f"{'✓' if g_acc else '✗'}")
    w(f"- I' pareado: Δ medio = {np.mean(ip_d):+.4f} (umbral ≥−0.05): "
      f"{'✓' if g_ip else '✗'}")
    w(f"- Mecanismo M (on-policy, 6 seeds): {'✓ INTACTO' if g_m else '✗ TOCADO'}")
    # anti-overclaim
    d_gw = [a - b for a, b in zip(P["miura_mhbp"], P["gating_wm"])
            if a is not None and b is not None]
    w(f"- Anti-overclaim (P vs gating_wm): Δ = {np.mean(d_gw):+.4f}")
    w("")

    # --- veredicto ---
    w("## Veredicto (ramas pre-declaradas)\n")
    guards = g_acc and g_ip and g_m
    from scipy import stats
    d_P = [a - b for a, b in zip(P["miura_mhbp"], P["hbp_full"])]
    sig_favor = paired_ci(d_P)["p"] < 0.05 and np.mean(d_P) > 0
    if not guards:
        w("**FAIL de guardarraíl** — la integración toca mecanismo/interfaz/"
          "accuracy: diagnóstico antes de nada (rama 3/4).")
    elif sig_favor and np.mean(d_gw) >= 0:
        w("**El plano aporta gobierno** (rama 1): C1 a favor + guardarraíles + "
          "anti-overclaim. → F3b.")
    elif verdict_ok.get("C1-P") and verdict_ok.get("C1-E"):
        w("**NO-INFERIORIDAD** (rama 2): el plano certificado gobierna sin "
          "dañar (P y E dentro de margen; M intacto; I' mantenida). La escala "
          "intra-instancia no separa a los gobernadores — exactamente el techo "
          "declarado en el prereg. → F3b (escala de sesión) decide.")
    else:
        w("**FAIL del aporte** (rama añadida v2): fuera de margen de "
          "no-inferioridad o significativo en contra; se reporta tal cual.")
    out = os.path.join(HERE, "REPORT_F3A.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\n".join(R))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
