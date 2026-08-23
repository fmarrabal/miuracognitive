"""
Fase 2 — Agregador estadístico pre-registrado.

Lee results_phase2/*.json → REPORT_PHASE2.md + figuras. Contrastes primarios
C1-C4 (pareados por semilla, métrica primaria = J medio de la suite OOD) con
t-test + Wilcoxon + Holm-4 + tamaño de efecto (Cohen dz) + IC bootstrap.
Robusto a celdas incompletas (se puede correr a medias).

  python -m mhbp.analysis.phase2_report
"""
from __future__ import annotations
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "tasks", "synthetic_multiscale", "results_phase2")
FIG = os.path.join(HERE, "analysis", "figures_phase2")
os.makedirs(FIG, exist_ok=True)

# PREREG v3: C1-C3 primarios (Holm-3); C4 exploratorio (núcleo 17× menor:
# estructura y capacidad confundidas — se reporta etiquetado, fuera de Holm)
CONTRASTS = [("C1_mhbp_vs_gru", "mhbp", "gru"),
             ("C2_mhbp_vs_taueq", "mhbp", "mhbp_taueq"),
             ("C3_mhbp_vs_first", "mhbp", "mhbp_first")]
EXPLORATORY_CONTRASTS = [("X4_mhbp_vs_single", "mhbp", "hbp_single"),
                         ("X5_mhbp_vs_nocpl", "mhbp", "mhbp_nocpl"),
                         ("X6_mhbp_vs_noallo", "mhbp", "mhbp_noallo")]
PRIMARY_N = {"mhbp": 24, "mhbp_first": 24, "gru": 20, "mhbp_taueq": 16}
EXPECTED_STEPS = 500

R = []
def w(s=""):
    R.append(s)


def load():
    runs, skipped = {}, []
    for fp in glob.glob(os.path.join(RES, "*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("steps") != EXPECTED_STEPS:      # homogeneidad del confirmatorio
                skipped.append((os.path.basename(fp), f"steps={d.get('steps')}"))
                continue
            runs.setdefault(d["controller"], {})[d["seed"]] = d
        except Exception as e:
            skipped.append((os.path.basename(fp), repr(e)[:60]))
    if skipped:
        print(f"AVISO: {len(skipped)} JSON saltados (heterogéneos/corruptos):")
        for name, why in skipped[:10]:
            print(f"  {name}: {why}")
    return runs


def paired(runs, a, b, key=lambda d: d["eval"]["ood_primary_J"]):
    seeds = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    xa = np.array([key(runs[a][s]) for s in seeds])
    xb = np.array([key(runs[b][s]) for s in seeds])
    return seeds, xa, xb


def contrast_stats(xa, xb):
    """diferencia (a−b) NEGATIVA = a mejor (J menor). t pareado, Wilcoxon, dz, CI."""
    from scipy import stats
    d = xa - xb
    n = len(d)
    if n < 3:
        return {"n": n}
    t, p_t = stats.ttest_rel(xa, xb)
    try:
        wst, p_w = stats.wilcoxon(xa, xb)
    except ValueError:
        p_w = 1.0
    dz = d.mean() / (d.std(ddof=1) + 1e-12)
    boot = np.random.default_rng(0).choice(d, size=(5000, n), replace=True).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": n, "mean_diff": float(d.mean()), "t": float(t), "p_t": float(p_t),
            "p_wilcoxon": float(p_w), "dz": float(dz),
            "ci95": [float(lo), float(hi)]}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    out = [False] * m
    for rank, idx in enumerate(order):
        alpha = 0.05 / (m - rank)
        if pvals[idx] < alpha:
            out[idx] = True
        else:
            break
    return out


def main():
    runs = load()
    w("# Fase 2 — Resultados del confirmatorio (SMA)\n")
    ntot = sum(len(v) for v in runs.values())
    w(f"Corridas cargadas: **{ntot}**. Generado por `phase2_report.py`.\n")

    # ---- tabla principal por controlador ----
    w("## Métrica primaria (J en suite OOD; menor = mejor) y secundarias\n")
    w("| controlador | n | J_OOD | regret_OOD | viol_rate(iid) | J_iid | params(core/total) |")
    w("|---|---|---|---|---|---|---|")
    for c in sorted(runs):
        vs = runs[c]
        j = [d["eval"]["ood_primary_J"] for d in vs.values()]
        rg = [d["eval"]["ood_primary_regret"] for d in vs.values()]
        vr = [d["eval"]["iid"]["viol_rate"] for d in vs.values()]
        ji = [d["eval"]["iid"]["J_mean"] for d in vs.values()]
        d0 = next(iter(vs.values()))
        tag = c if c in PRIMARY_N else f"{c} (explor.)"
        w(f"| {tag} | {len(j)} | {np.mean(j):.4f}±{np.std(j):.4f} "
          f"| {np.mean(rg):.4f} | {np.mean(vr):.3f} | {np.mean(ji):.4f} "
          f"| {d0['params_core']}/{d0['params_total']} |")
    w("")

    # ---- contrastes primarios (Holm-3, con DIRECCIÓN) ----
    w("## Contrastes primarios (pareados por semilla; Holm-3 sobre p_t bilateral,")
    w("Wilcoxon como sensibilidad; Δ<0 ⇒ mhbp mejor; 'Holm SÍ' EXIGE Δ<0)\n")
    stats_rows, pvals = [], []
    for name, a, b in CONTRASTS:
        _, xa, xb = paired(runs, a, b)
        st = contrast_stats(xa, xb) if len(xa) >= 3 else {"n": len(xa)}
        stats_rows.append((name, st))
        pvals.append(st.get("p_t", 1.0))
    surv = holm(pvals) if pvals else []

    def row(name, st, sv, primary=True):
        if st.get("n", 0) < 3:
            w(f"| {name} | {st.get('n', 0)} | — | — | — | — | — | — | — |")
            return
        direc_ok = st["mean_diff"] < 0
        holm_txt = ("SÍ" if (sv and direc_ok) else
                    ("dir. contraria" if (sv and not direc_ok) else "no")) \
            if primary else "(explor.)"
        w(f"| {name} | {st['n']} | {st['mean_diff']:+.4f} | {st['t']:.2f} "
          f"| {st['p_t']:.4f} | {st['p_wilcoxon']:.4f} | {st['dz']:.2f} "
          f"| [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}] | {holm_txt} |")

    w("| contraste | n | Δ(J_OOD) | t | p | p_wilcoxon | dz | IC95 | Holm |")
    w("|---|---|---|---|---|---|---|---|---|")
    for (name, st), sv in zip(stats_rows, surv):
        row(name, st, sv, primary=True)
    for name, a, b in EXPLORATORY_CONTRASTS:
        _, xa, xb = paired(runs, a, b)
        st = contrast_stats(xa, xb) if len(xa) >= 3 else {"n": len(xa)}
        row(name, st, False, primary=False)
    w("")
    w("Sensibilidad de la métrica: la variante 'media plana de 4 protocolos'"
      " (ood_flat4_J) se reporta aparte y NO entra en Holm.\n")

    # ---- por protocolo ----
    w("## J por protocolo (media entre semillas)\n")
    protos = ["iid", "ood_budget_lo", "ood_budget_hi", "ood_riskfreq",
              "ood_long", "e4_step", "e2_risk_only"]
    w("| controlador | " + " | ".join(protos) + " |")
    w("|" + "---|" * (len(protos) + 1))
    for c in sorted(runs):
        row = [c]
        for p in protos:
            js = [d["eval"][p]["J_mean"] for d in runs[c].values() if p in d["eval"]]
            row.append(f"{np.mean(js):.3f}" if js else "—")
        w("| " + " | ".join(row) + " |")
    w("")

    # ---- seguimiento y settling ----
    w("## Seguimiento (correlaciones acción↔factor, protocolo iid) y E4\n")
    w("| controlador | corr(d,depth)↑ | corr(ρ,tool)↑ | corr(d,halt)↓ | corr(B,gasto)↑ | settling(vent.) | viol. post-escalón |")
    w("|---|---|---|---|---|---|---|")
    for c in sorted(runs):
        vs = list(runs[c].values())
        def m(proto, key):
            xs = [d["eval"][proto].get(key) for d in vs if key in d["eval"].get(proto, {})]
            return f"{np.mean(xs):.3f}" if xs else "—"
        w(f"| {c} | {m('iid','corr_d_depth')} | {m('iid','corr_rho_tool')} "
          f"| {m('iid','corr_d_halt')} | {m('iid','corr_B_spend')} "
          f"| {m('e4_step','settling_windows_mean')} | {m('e4_step','post_step_viol_mean')} |")
    w("")

    # ---- figuras ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = sorted(runs)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for ax, key, ttl in [(axes[0], "ood_primary_J", "J en suite OOD (↓ mejor)"),
                             (axes[1], "ood_primary_regret", "compute-regret OOD (↓)")]:
            data = [[d["eval"][key] for d in runs[c].values()] for c in names]
            ax.boxplot(data, tick_labels=names, showmeans=True)
            ax.set_title(ttl); ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "phase2_primary.png"), dpi=140)
        plt.close(fig)
        w(f"Figuras: `analysis/figures_phase2/phase2_primary.png`\n")
    except Exception as e:
        w(f"_(figuras no generadas: {repr(e)[:80]})_\n")

    out = os.path.join(HERE, "analysis", "REPORT_PHASE2.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    print("\n".join(R))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
