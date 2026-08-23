"""
Reporte del BENCHMARK COMPLETO. Lee results_benchmark/ y produce:
  - tabla de texto por (régimen, generadores): accuracy por bucket + E[n_iter] + corr,
    con media +- desv. entre semillas;
  - el contraste clave OOD: corr(K,n_iter) de hbp_* vs el control gating_wm, con t;
  - una tabla LaTeX (benchmark_table.tex).

Uso:  $env:PYTHONPATH="." ; python benchmark_report.py
"""
import json, glob, os
import numpy as np

VARIANTS = ["vanilla", "gating", "gating_wm", "hbp_first", "hbp_full"]
LAB = {"vanilla": "Vanilla", "gating": "Gating", "gating_wm": "Gating+WM",
       "hbp_first": "HBP 1er orden", "hbp_full": "HBP 2o orden"}


def load():
    rows = []
    for p in sorted(glob.glob("results_benchmark/*.json")):
        r = json.load(open(p, encoding="utf-8"))
        c = r["_cell"]
        cd = r.get("compute_diag") or {}
        rows.append(dict(regime=c["regime"], gens=c["gens"], variant=c["variant"],
                         corto=r["final_acc"]["corto"], medio=r["final_acc"]["medio"],
                         largo=r["final_acc"]["largo"], niter=cd.get("mean_niter"),
                         corr=cd.get("corr_K_niter")))
    return rows


def agg(rows, regime, gens, variant, field):
    xs = [r[field] for r in rows if r["regime"] == regime and r["gens"] == gens
          and r["variant"] == variant and r[field] is not None]
    return np.array(xs, dtype=float)


def fmt(a):
    return f"{a.mean():.3f}±{a.std():.3f}" if len(a) else "   -   "


def main():
    rows = load()
    if not rows:
        print("No hay resultados en results_benchmark/. Ejecuta benchmark.py primero.")
        return
    latex = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Régimen & Modelo & Corto & Medio & Largo & $\mathbb{E}[n_{it}]$ & corr$(K,n)$ \\", r"\midrule"]
    for gens in ["adjacent", "cycle_transp"]:
        for regime in ["indist", "ood"]:
            n = max((len(agg(rows, regime, gens, v, "largo")) for v in VARIANTS), default=0)
            head = f"{gens} | {regime}" + (" (extrapolación)" if regime == "ood" else "")
            print(f"\n=== {head}  (n={n} semillas) ===")
            print(f"  {'Modelo':14s} {'corto':>12s} {'medio':>12s} {'largo':>12s} {'E[n_iter]':>12s} {'corr(K,n)':>12s}")
            gwm = agg(rows, regime, gens, "gating_wm", "corr")
            for v in VARIANTS:
                co, me, la = (agg(rows, regime, gens, v, k) for k in ("corto", "medio", "largo"))
                ni, cr = agg(rows, regime, gens, v, "niter"), agg(rows, regime, gens, v, "corr")
                print(f"  {LAB[v]:14s} {fmt(co):>12s} {fmt(me):>12s} {fmt(la):>12s} {fmt(ni):>12s} {fmt(cr):>12s}")
                latex.append(f"{regime if v=='vanilla' else ''} & {LAB[v]} & {fmt(co)} & {fmt(me)} & "
                             f"{fmt(la)} & {fmt(ni)} & {fmt(cr)} \\\\")
            # contraste de adaptividad OOD vs control WM
            if regime == "ood" and len(gwm) > 1:
                for v in ["hbp_first", "hbp_full"]:
                    cr = agg(rows, regime, gens, v, "corr")
                    if len(cr) > 1:
                        semd = np.sqrt(cr.std(ddof=1)**2/len(cr) + gwm.std(ddof=1)**2/len(gwm))
                        t = (cr.mean() - gwm.mean()) / semd if semd > 0 else float("nan")
                        print(f"    -> corr({LAB[v]}) - corr(Gating+WM) = {cr.mean()-gwm.mean():+.3f}  t={t:.2f}")
            latex.append(r"\midrule")
    latex += [r"\bottomrule", r"\end{tabular}"]
    with open("benchmark_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex))
    print("\n[LaTeX guardado en benchmark_table.tex]")


if __name__ == "__main__":
    main()
