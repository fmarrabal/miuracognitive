"""
Reporte del BENCHMARK v2 con la estadística corregida por la auditoría:
  - t PAREADO por seed sobre diferencias Fisher-z de la corr OOD-pura (K>=14)
    (el diseño ES pareado: misma seed => mismos datos para todas las variantes),
  - corrección Holm-Bonferroni sobre los 4 contrastes,
  - dispersión con ddof=1 y n POR CELDA en las tablas,
  - baseline de clase mayoritaria condicionada por bucket (muestreada del dataset).

Uso:  $env:PYTHONPATH="." ; python benchmark_report_v2.py
"""
import json, glob, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# BENCH_DIR permite apuntar a otro benchmark (p.ej. results_benchmark_v3 = re-run
# con el fix de congelación BF16 / física aprendible).
OUT = os.path.join(HERE, os.environ.get("BENCH_DIR", "results_benchmark_v2"))
VARIANTS = ["vanilla", "gating", "gating_wm", "hbp_first", "hbp_full"]
LAB = {"vanilla": "Vanilla", "gating": "Gating", "gating_wm": "Gating+WM",
       "hbp_first": "HBP 1er orden", "hbp_full": "HBP 2o orden"}


def load(regime, gens, variant):
    out = {}
    for p in sorted(glob.glob(os.path.join(OUT, f"{regime}_{gens}_{variant}_seed*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        out[r["_cell"]["seed"]] = r
    return out


def fisher_z(r):
    r = max(min(r, 0.999), -0.999)
    return 0.5 * math.log((1 + r) / (1 - r))


def paired_t(diffs):
    d = np.array(diffs, dtype=float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), n
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    # p-valor bilateral por aproximación de la t de Student (df=n-1)
    from math import lgamma
    df = n - 1
    x = df / (df + t * t)
    # incomplete beta via continued fraction (suficiente para el reporte)
    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, max(1.0 - qab * x / qap, 1e-30)
        d = 1.0 / d
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = max(1.0 + aa * d, 1e-30); c = max(1.0 + aa / c, 1e-30)
            d = 1.0 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = max(1.0 + aa * d, 1e-30); c = max(1.0 + aa / c, 1e-30)
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-9:
                break
        return h
    def ibeta(a, b, x):
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        lbeta = lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log(1 - x)
        front = math.exp(lbeta)
        if x < (a + 1) / (a + b + 2):
            return front * betacf(a, b, x) / a
        return 1.0 - front * betacf(b, a, 1 - x) / b
    p = ibeta(df / 2, 0.5, x)
    return float(t), float(p), n


def fmt(xs):
    a = np.array(xs, dtype=float)
    if len(a) == 0:
        return "     -     "
    if len(a) == 1:
        return f"{a[0]:.3f} (n=1)"
    return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"


def majority_baseline(gens, n_per_k=4000):
    """Clase mayoritaria condicionada al bucket, muestreada del dataset real."""
    from data.synthetic_recall import PermutationCompositionDataset
    ds = PermutationCompositionDataset(min_ops=2, max_ops=24, seq_len=128,
                                       seed=999, gen_set=gens)
    from collections import Counter, defaultdict
    counts = defaultdict(Counter)
    for _ in range(n_per_k):
        _, labels, K = ds.sample()
        pos = (labels != -1).nonzero(as_tuple=True)[0]
        ans = int(labels[pos[-1]].item())
        counts[ds.bucket(K)][ans] += 1
    return {b: (c.most_common(1)[0][1] / sum(c.values())) for b, c in counts.items()}


def main():
    print("=" * 78)
    print("BENCHMARK v2 — seeds frescas (>=10), N_max=24, eval limpia, argmax restringido")
    print("=" * 78)
    contrasts = []
    for gens in ["adjacent", "cycle_transp"]:
        base = majority_baseline(gens)
        print(f"\n  [baseline mayoritaria {gens}: " +
              " ".join(f"{b}={v:.3f}" for b, v in sorted(base.items())) + "]")
        for regime in ["indist", "ood"]:
            print(f"\n=== {gens} | {regime}{' (extrapolación)' if regime=='ood' else ''} ===")
            print(f"  {'Modelo':14s} {'n':>3s} {'corto':>12s} {'medio':>12s} {'largo':>12s} "
                  f"{'E[n_it]':>12s} {'corrOOD(K>=14)':>15s}")
            for v in VARIANTS:
                runs = load(regime, gens, v)
                if not runs:
                    continue
                acc = {b: [r["final_acc"][b] for r in runs.values()] for b in ("corto", "medio", "largo")}
                cds = [r.get("compute_diag") for r in runs.values() if r.get("compute_diag")]
                ni = [d["mean_niter"] for d in cds]
                co = [d.get("corr_K_niter_ood") for d in cds if d.get("corr_K_niter_ood") is not None]
                print(f"  {LAB[v]:14s} {len(runs):>3d} {fmt(acc['corto']):>12s} {fmt(acc['medio']):>12s} "
                      f"{fmt(acc['largo']):>12s} {fmt(ni):>12s} {fmt(co):>15s}")
            # Contrastes pareados (solo OOD)
            if regime == "ood":
                gwm = load("ood", gens, "gating_wm")
                for v in ["hbp_first", "hbp_full"]:
                    hv = load("ood", gens, v)
                    common = sorted(set(gwm) & set(hv))
                    diffs = []
                    for s in common:
                        cg = gwm[s].get("compute_diag", {}).get("corr_K_niter_ood")
                        ch = hv[s].get("compute_diag", {}).get("corr_K_niter_ood")
                        if cg is not None and ch is not None:
                            diffs.append(fisher_z(ch) - fisher_z(cg))
                    t, p, n = paired_t(diffs)
                    contrasts.append((f"{gens}:{v}-gwm", t, p, n, np.mean(diffs) if diffs else float("nan")))
                    print(f"    -> PAREADO Fisher-z corrOOD {LAB[v]} - Gating+WM: "
                          f"Δz={np.mean(diffs):+.3f}  t={t:.2f}  p={p:.4f}  (n={n} pares)")
    # --- Métricas honestas adicionales (post-auditoría) ---
    print("\n" + "=" * 78)
    print("EXTRA 1 — PENDIENTE DE CÓMPUTO EN OOD PURO: ¿sigue gastando más allá de K=12?")
    print("  slope = E[n_iter | K in 18..24] - E[n_iter | K in 12..15]  (>0 = extrapola la política)")
    for gens in ["adjacent", "cycle_transp"]:
        print(f"  --- {gens} ---")
        for v in ["gating_wm", "hbp_first", "hbp_full"]:
            runs = load("ood", gens, v)
            slopes = []
            for r in runs.values():
                curve = (r.get("compute_diag") or {}).get("niter_by_K", {})
                hi = [d["niter"] for k, d in curve.items() if 18 <= int(k) <= 24]
                lo = [d["niter"] for k, d in curve.items() if 12 <= int(k) <= 15]
                if hi and lo:
                    slopes.append(np.mean(hi) - np.mean(lo))
            print(f"    {LAB[v]:14s} slope_OOD = {fmt(slopes)}  (n={len(slopes)})")

    print("\nEXTRA 2 — ACCURACY IN-DIST (largo): contrastes PAREADOS por seed")
    for gens in ["adjacent", "cycle_transp"]:
        base = {"gating": load("indist", gens, "gating"),
                "gating_wm": load("indist", gens, "gating_wm")}
        for v in ["hbp_first", "hbp_full"]:
            hv = load("indist", gens, v)
            for bname, bruns in base.items():
                common = sorted(set(hv) & set(bruns))
                diffs = [hv[s]["final_acc"]["largo"] - bruns[s]["final_acc"]["largo"] for s in common]
                t, p, n = paired_t(diffs)
                if n >= 2:
                    print(f"  {gens:12s} {LAB[v]} - {bname:9s}: Δ={np.mean(diffs):+.3f} t={t:.2f} p={p:.3f} (n={n})")

    # Holm-Bonferroni sobre los 4 contrastes
    print("\n=== Corrección Holm-Bonferroni (4 contrastes pre-registrados de corrOOD) ===")
    valid = [(name, t, p, n, dz) for name, t, p, n, dz in contrasts if not math.isnan(p)]
    order = sorted(valid, key=lambda c: c[2])
    m = len(order)
    for rank, (name, t, p, n, dz) in enumerate(order):
        alpha_adj = 0.05 / (m - rank)
        sig = "SIGNIFICATIVO" if p < alpha_adj else "no significativo"
        print(f"  {name:28s} t={t:5.2f} p={p:.4f} vs α_Holm={alpha_adj:.4f} -> {sig}")


if __name__ == "__main__":
    main()
