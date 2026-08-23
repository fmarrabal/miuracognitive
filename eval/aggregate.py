"""
MiuraCognitive - Agregador de resultados (tablas + figuras para el paper)
==========================================================================
Lee los JSON de results/, agrega por variante (media ± desv. entre semillas)
y produce:

  1. Tabla principal de accuracy estratificada (LaTeX + texto)        -> figures/
  2. Figura 1: accuracy vs longitud de distractor por variante       -> figures/fig1_accuracy.png
  3. Figura 2: curvas de entrenamiento (loss)                         -> figures/fig2_loss.png
  4. Figura 3: dinámica del HBP (espectro FFT del VEI)                -> figures/fig3_hbp_spectrum.png
  5. Figura 4: varianza del VEI durante entrenamiento                 -> figures/fig4_vei_var.png

Uso:
    PYTHONPATH=. python -m eval.aggregate --results_dir results --fig_dir figures

Autor: Francisco M. Arrabal (Curro) + asistencia Claude
"""

from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin display (servidor/headless)
import matplotlib.pyplot as plt

# Orden y etiquetas legibles de las variantes
VARIANT_ORDER = ["vanilla", "gating", "hbp_first", "hbp_full"]
VARIANT_LABELS = {
    "vanilla": "Vanilla",
    "gating": "Gating simple",
    "hbp_first": "HBP 1er orden",
    "hbp_full": "HBP 2o orden (onda)",
}
BUCKETS = ["corto", "medio", "largo"]
BUCKET_LABELS = {"corto": "Corto (≤15)", "medio": "Medio (16-35)", "largo": "Largo (>35)"}


def load_results(results_dir: str) -> dict:
    """Carga todos los JSON y los agrupa por variante."""
    by_variant = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        by_variant[r["variant"]].append(r)
    return by_variant


def aggregate_accuracy(by_variant: dict) -> dict:
    """Calcula media y desv. típica de accuracy por variante y bucket."""
    agg = {}
    for variant, runs in by_variant.items():
        agg[variant] = {}
        for b in BUCKETS:
            vals = np.array([r["final_acc"][b] for r in runs])
            agg[variant][b] = (float(vals.mean()), float(vals.std()))
    return agg


def make_latex_table(agg: dict, n_seeds: int) -> str:
    """Genera la tabla principal en LaTeX (booktabs)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Accuracy de recall estratificada por longitud de distractor "
        r"(media $\pm$ desv. sobre " + str(n_seeds) + r" semillas). "
        r"El HBP de segundo orden mantiene mejor el recall con distractores largos.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Modelo & Corto ($\leq$15) & Medio (16--35) & Largo ($>$35) \\",
        r"\midrule",
    ]
    for variant in VARIANT_ORDER:
        if variant not in agg:
            continue
        row = VARIANT_LABELS[variant]
        for b in BUCKETS:
            mean, std = agg[variant][b]
            row += f" & {mean:.3f}$\\pm${std:.3f}"
        row += r" \\"
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_text_table(agg: dict) -> str:
    """Tabla en texto plano para inspección rápida."""
    out = [f"\n{'Modelo':22s} {'Corto':>14s} {'Medio':>14s} {'Largo':>14s}"]
    out.append("-" * 66)
    for variant in VARIANT_ORDER:
        if variant not in agg:
            continue
        row = f"{VARIANT_LABELS[variant]:22s}"
        for b in BUCKETS:
            mean, std = agg[variant][b]
            row += f" {mean:.3f}±{std:.3f} "
        out.append(row)
    return "\n".join(out)


def fig_accuracy_by_distractor(agg: dict, fig_dir: str):
    """Figura 1: barras de accuracy por bucket y variante."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(BUCKETS))
    width = 0.2
    variants = [v for v in VARIANT_ORDER if v in agg]
    for i, variant in enumerate(variants):
        means = [agg[variant][b][0] for b in BUCKETS]
        stds = [agg[variant][b][1] for b in BUCKETS]
        ax.bar(x + (i - len(variants) / 2 + 0.5) * width, means, width,
               yerr=stds, capsize=3, label=VARIANT_LABELS[variant])
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABELS[b] for b in BUCKETS])
    ax.set_ylabel("Accuracy de recall")
    ax.set_xlabel("Longitud de distractor")
    ax.set_title("Recall por dificultad: efecto del HBP")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig1_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_loss_curves(by_variant: dict, fig_dir: str):
    """Figura 2: curvas de loss (media entre semillas) por variante."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in VARIANT_ORDER:
        if variant not in by_variant:
            continue
        runs = by_variant[variant]
        # Asumimos misma rejilla de pasos entre semillas
        steps = runs[0]["trace"]["step"]
        losses = np.array([r["trace"]["loss"] for r in runs
                           if len(r["trace"]["loss"]) == len(steps)])
        if len(losses) == 0:
            continue
        mean = losses.mean(axis=0)
        std = losses.std(axis=0)
        ax.plot(steps, mean, label=VARIANT_LABELS[variant])
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Paso de entrenamiento")
    ax.set_ylabel("Pérdida total")
    ax.set_title("Curvas de entrenamiento")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig2_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_hbp_spectrum(by_variant: dict, fig_dir: str):
    """Figura 3: espectro FFT del VEI para variantes con HBP.
    Demuestra la dinámica oscilatoria (2o orden) vs su ausencia (1er orden)."""
    hbp_variants = [v for v in ("hbp_first", "hbp_full") if v in by_variant]
    if not hbp_variants:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in hbp_variants:
        run = by_variant[variant][0]  # primera semilla, representativa
        diag = run.get("hbp_diagnostics")
        if not diag or "spectrum" not in diag:
            continue
        spec = diag["spectrum"]
        freqs = np.array(spec["freqs"])
        power = np.array(spec["power_spectrum"])
        # Normalizar para comparar formas
        if power.max() > 0:
            power = power / power.max()
        ax.plot(freqs, power, label=f"{VARIANT_LABELS[variant]} "
                f"(f*={spec['dominant_freq']:.3f})")
    ax.set_xlabel("Frecuencia (1/tick)")
    ax.set_ylabel("Potencia normalizada")
    ax.set_title("Espectro del VEI: oscilación del HBP")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig3_hbp_spectrum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_vei_variance(by_variant: dict, fig_dir: str):
    """Figura 4: varianza del VEI durante entrenamiento (¿se activa el HBP?)."""
    hbp_variants = [v for v in ("hbp_first", "hbp_full") if v in by_variant]
    if not hbp_variants:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in hbp_variants:
        runs = by_variant[variant]
        steps = runs[0]["trace"]["step"]
        vv = np.array([r["trace"]["vei_var"] for r in runs
                       if len(r["trace"]["vei_var"]) == len(steps)])
        if len(vv) == 0:
            continue
        ax.plot(steps, vv.mean(axis=0), label=VARIANT_LABELS[variant])
    ax.set_xlabel("Paso de entrenamiento")
    ax.set_ylabel("Varianza del VEI")
    ax.set_title("Actividad del HBP durante el entrenamiento")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig4_vei_var.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--fig_dir", default="figures")
    args = ap.parse_args()

    by_variant = load_results(args.results_dir)
    if not by_variant:
        print(f"No hay resultados en {args.results_dir}. Ejecuta primero el estudio.")
        return

    os.makedirs(args.fig_dir, exist_ok=True)
    n_seeds = max(len(runs) for runs in by_variant.values())

    # --- Tablas ---
    agg = aggregate_accuracy(by_variant)
    text_table = make_text_table(agg)
    latex_table = make_latex_table(agg, n_seeds)
    print(text_table)
    print("\n[LaTeX guardado en figures/table_main.tex]")

    with open(os.path.join(args.fig_dir, "table_main.tex"), "w") as f:
        f.write(latex_table)
    with open(os.path.join(args.fig_dir, "table_main.txt"), "w") as f:
        f.write(text_table)

    # --- Figuras ---
    paths = []
    paths.append(fig_accuracy_by_distractor(agg, args.fig_dir))
    paths.append(fig_loss_curves(by_variant, args.fig_dir))
    p3 = fig_hbp_spectrum(by_variant, args.fig_dir)
    p4 = fig_vei_variance(by_variant, args.fig_dir)
    if p3:
        paths.append(p3)
    if p4:
        paths.append(p4)

    print("\nFiguras generadas:")
    for p in paths:
        if p:
            print(f"  {p}")

    # --- Veredicto automático sobre la hipótesis ---
    print("\n" + "=" * 60)
    print("VEREDICTO sobre la hipótesis (distractores largos):")
    if all(v in agg for v in ("vanilla", "hbp_full")):
        v_largo = agg["vanilla"]["largo"][0]
        h_largo = agg["hbp_full"]["largo"][0]
        delta = h_largo - v_largo
        print(f"  hbp_full largo = {h_largo:.3f} | vanilla largo = {v_largo:.3f}")
        print(f"  Δ (hbp_full - vanilla) = {delta:+.3f}")
        if delta >= 0.03:
            print("  ✓ El HBP mejora el recall con distractores largos (≥ +3%).")
        else:
            print("  ✗ No se observa mejora ≥ +3%. Revisar diseño o entrenar más.")
    print("=" * 60)


if __name__ == "__main__":
    main()
