"""Figuras de datos del paper del gobernador, desde los JSON del benchmark.
TODO EN INGLES (antes estaban en espanol, y con los acentos rotos) y con el
estilo comun de figstyle.py.

  fig_ood_corr.png    protocolo v3 (n=10): adaptividad de computo OOD por
                      variante y generador. Se mantiene v3 a proposito: es lo
                      que describe su \\caption, y el aviso de que el gap del
                      5-ciclo esta confundido con capacidad lo da el texto.
  fig_v4.png          NUEVO: la campana de des-confusion v4 (20 semillas
                      frescas), que es el titular del paper y no tenia figura.
  fig_niter_bucket.png E[n_iter] por dificultad.
"""
import glob
import json
import os

import numpy as np

from figstyle import (FIELD, GREY, INK, LEARN, MUTE, WAVE, aplica, guarda,
                      limpia)

aplica()
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(HERE, "..")
BM3 = os.path.join(RAIZ, os.environ.get("BENCH_DIR", "results_benchmark_v3"))
BM4 = os.path.join(RAIZ, "results_benchmark_v4")

VAR3 = ["gating_wm", "hbp_first", "hbp_full", "hbp_kdv"]
LAB3 = {"gating_wm": "working memory (no field)",
        "hbp_first": "field, 1st order (diffusive)",
        "hbp_full": "field, 2nd order (wave)",
        "hbp_kdv": "field, KdV (dispersive)"}
COL3 = {"gating_wm": MUTE, "hbp_first": GREY, "hbp_full": WAVE,
        "hbp_kdv": FIELD}

VAR4 = ["hbp_first_eq", "hbp_full", "hbp_gru"]
LAB4 = {"hbp_first_eq": "field, 1st order\n(caps equalized)",
        "hbp_full": "field, 2nd order",
        "hbp_gru": "GRU governor\n(matched interface)"}
COL4 = {"hbp_first_eq": GREY, "hbp_full": WAVE, "hbp_gru": LEARN}

TAREAS = [("adjacent", "adjacent\ntranspositions"),
          ("cycle_transp", "5-cycle $+$\ntransposition")]


def celdas(base, gens, v):
    return [json.load(open(p, encoding="utf-8"))
            for p in sorted(glob.glob(
                os.path.join(base, f"ood_{gens}_{v}_seed*.json")))]


def corr(base, gens, v):
    return np.array([
        (r.get("compute_diag") or {}).get(
            "corr_K_niter_ood", (r.get("compute_diag") or {}).get("corr_K_niter"))
        for r in celdas(base, gens, v)], dtype=float)


def _barras(ax, base, variantes, etiquetas, colores, con_error=True):
    x = np.arange(len(TAREAS))
    w = 0.8 / max(len(variantes), 1)
    for i, v in enumerate(variantes):
        m, e = [], []
        for g, _ in TAREAS:
            c = corr(base, g, v)
            m.append(c.mean())
            e.append(c.std(ddof=1) / np.sqrt(len(c)))
        pos = x + (i - (len(variantes) - 1) / 2) * w
        ax.bar(pos, m, w * 0.92, color=colores[v], label=etiquetas[v],
               yerr=(e if con_error else None), capsize=2.5,
               error_kw=dict(lw=0.9, ecolor=INK))
    ax.set_xticks(x)
    ax.set_xticklabels([t[1] for t in TAREAS])
    ax.axhline(0, color=INK, lw=0.7)
    limpia(ax, y_solo=True)
    return x


def fig_ood_corr():
    variantes = [v for v in VAR3
                 if all(len(celdas(BM3, g, v)) for g, _ in TAREAS)]
    if not variantes:
        return "fig_ood_corr.png OMITIDA (sin celdas v3)"
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    _barras(ax, BM3, variantes, LAB3, COL3)
    ax.set_ylabel(r"OOD compute adaptivity" "\n"
                  r"$\mathrm{corr}(K,\ \mathbb{E}[n_{\mathrm{iter}}])$")
    ax.legend(loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.1)
    return guarda(fig, "fig_ood_corr.png")


def fig_v4():
    """La des-confusion: mismo eje, tres brazos, 20 semillas frescas."""
    variantes = [v for v in VAR4
                 if all(len(celdas(BM4, g, v)) for g, _ in TAREAS)]
    if not variantes:
        return "fig_v4.png OMITIDA (sin celdas v4)"
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})

    _barras(a1, BM4, variantes, LAB4, COL4)
    a1.set_ylabel(r"OOD compute adaptivity" "\n"
                  r"$\mathrm{corr}(K,\ \mathbb{E}[n_{\mathrm{iter}}])$")
    a1.set_ylim(0, 0.40)          # aire para que la leyenda no pise las barras
    a1.legend(loc="upper center", ncol=3, columnspacing=0.9, handlelength=1.0,
              fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    a1.set_title("twenty fresh seeds, per arm", pad=22)

    # panel derecho: las DIFERENCIAS pareadas, que es donde vive el veredicto
    pares = [("hbp_first_eq", "order", WAVE), ("hbp_gru", "GRU", LEARN)]
    val, lo, hi, col = [], [], [], []
    for g, _ in TAREAS:
        for rival, nombre, c in pares:
            d = corr(BM4, g, "hbp_full") - corr(BM4, g, rival)
            m = d.mean()
            se = d.std(ddof=1) / np.sqrt(len(d))
            val.append(m)
            lo.append(m - 2.093 * se)
            hi.append(m + 2.093 * se)
            col.append(c)
    y = np.arange(len(val))[::-1]
    a2.errorbar(val, y, xerr=[np.array(val) - np.array(lo),
                              np.array(hi) - np.array(val)],
                fmt="o", ms=5, lw=1.1, capsize=3, ecolor=INK,
                mfc="white", mew=1.4, linestyle="none",
                markeredgecolor=INK)
    for xi, yi, ci in zip(val, y, col):
        a2.plot([xi], [yi], "o", ms=5, color=ci, zorder=5)
    a2.axvline(0, color=INK, lw=0.7, ls=(0, (3, 3)))
    a2.set_yticks(y)
    a2.set_yticklabels([
        "vs 1st order\nadjacent", "vs GRU\nadjacent",
        "vs 1st order\ncycle", "vs GRU\ncycle"], fontsize=8)
    a2.set_xlabel("paired difference vs the 2nd-order field\n"
                  "(95% CI; right of zero: the field wins)")
    a2.set_title("where the verdict lives", pad=22)
    limpia(a2)
    fig.subplots_adjust(wspace=0.42)
    return guarda(fig, "fig_v4.png")


def fig_niter_bucket():
    buckets = ["corto", "medio", "largo"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    xb = np.arange(len(buckets))
    hay = False
    for v in VAR3:
        runs = celdas(BM3, "adjacent", v)
        if not runs:
            continue
        hay = True
        ys = [np.mean([r["compute_diag"][b]["mean_niter"]
                       for r in runs if b in r["compute_diag"]])
              for b in buckets]
        ax.plot(xb, ys, "-o", ms=4, lw=1.3, color=COL3[v], label=LAB3[v])
    if not hay:
        return "fig_niter_bucket.png OMITIDA"
    ax.set_xticks(xb)
    ax.set_xticklabels([r"short ($K\leq6$)", r"medium ($7$–$13$)",
                        r"long ($\geq14$, OOD)"])
    ax.set_ylabel(r"$\mathbb{E}[n_{\mathrm{iter}}]$ of the reasoner")
    ax.legend(loc="upper left")
    limpia(ax)
    return guarda(fig, "fig_niter_bucket.png")


if __name__ == "__main__":
    for f in (fig_ood_corr, fig_v4, fig_niter_bucket):
        print(f())
