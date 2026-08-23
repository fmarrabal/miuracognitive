"""Figuras del paper de cognicion. Todo el texto en INGLES; estilo comun en
figstyle.py.

  fig_hierarchy.png     la jerarquia de informacion, CORREGIDA tras la
                        auditoria de lectura: los tres peldanos conmensurables
                        aparte de los posteriores, que se leen de otra forma.
  fig_cliff_ladder.png  acantilado vs familia suave. EJE Y COMPARTIDO: el
                        mensaje del experimento es que un rango es 5x el otro,
                        y la version anterior lo destruia usando escalas
                        distintas en cada panel, que hacia parecer las barras
                        igual de altas.
  fig_architecture.png  esquema de la arquitectura minima completa.
"""
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figstyle import (AMBER, FIELD, GREY, INK, LEARN, MUTE, WAVE, aplica,
                      etiqueta_barras, guarda, limpia, rango)

aplica()
import matplotlib.pyplot as plt  # noqa: E402  (despues de aplica())


def fig_jerarquia():
    """Dos paneles porque hay DOS lecturas distintas, y mezclarlas en una
    sola escalera fue justamente el error que el paper retracta."""
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [3, 2.2]},
        sharey=True)

    # --- conmensurable: los tres brazos que se leen de un estado unico
    lab1 = ["uniform", "difficulty\n(class $K$)", "ex-ante value\n(allocator)"]
    v1 = [0.467, 0.546, 0.698]
    b1 = a1.bar(range(3), v1, color=[GREY, MUTE, FIELD], width=0.58)
    etiqueta_barras(a1, b1, v1)
    a1.set_xticks(range(3))
    a1.set_xticklabels(lab1)
    a1.set_ylabel("normalized payoff (matched compute)")
    a1.set_ylim(0.40, 1.02)
    a1.set_title("single-state readout — commensurable", pad=8)
    a1.annotate("ex-ante ceiling\n(matches the class-stake oracle)",
                xy=(2, 0.698), xytext=(1.15, 0.86),
                fontsize=8.5, color=FIELD, ha="center",
                arrowprops=dict(arrowstyle="->", color=FIELD, lw=0.8))
    limpia(a1)

    # --- NO conmensurable: los posteriores, leidos de la mezcla PonderNet
    lab2 = ["posterior\n(native halting)", "posterior\n+ value"]
    v2 = [0.921, 0.921]
    b2 = a2.bar(range(2), v2, color="white", width=0.58, hatch="////",
                edgecolor=MUTE, linewidth=1.0)
    etiqueta_barras(a2, b2, v2)
    a2.set_xticks(range(2))
    a2.set_xticklabels(lab2)
    a2.set_title("mixture readout — not commensurable", pad=8, color=WAVE)
    a2.text(0.5, 0.545,
            "readout audit\n"
            "readout  $+0.383$\n"
            "residual regime  $+0.000$\n"
            "instance allocation  $+0.001$\n"
            "(at fixed readout and budget)",
            ha="center", va="center", fontsize=8, color=WAVE, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=WAVE, lw=0.8))
    limpia(a2)
    a2.spines["left"].set_visible(False)
    a2.tick_params(axis="y", length=0)

    fig.subplots_adjust(wspace=0.08)
    return guarda(fig, "fig_hierarchy.png")


def fig_acantilado():
    """Eje Y COMPARTIDO: el resultado es una comparacion de AMPLITUDES."""
    labels = ["uniform", "difficulty", "ex-ante\nvalue",
              "who\narrives", "exact\ncost"]
    cliff = [0.674, 0.754, 0.885, 0.885, 0.952]
    suave = [0.159, 0.182, 0.201, 0.214, 0.214]
    x = np.arange(5)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, vals, tit, col in (
            (a1, cliff, "cliff family (cycle walk)", WAVE),
            (a2, suave, "smooth family (arithmetic)", MUTE)):
        cols = [col if i == 2 else GREY for i in x]
        b = ax.bar(x, vals, color=cols, width=0.6)
        etiqueta_barras(ax, b, vals, fs=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(tit, pad=8, color=col)
        ax.set_xlim(-0.6, 5.45)
        limpia(ax)
    a1.set_ylabel("normalized payoff (matched spent tokens)")
    a1.set_ylim(0, 1.06)

    # los corchetes son el mensaje: 0.278 contra 0.055, un factor 5.1
    rango(a1, 4.75, min(cliff), max(cliff), "range\n$0.278$", color=WAVE)
    rango(a2, 4.75, min(suave), max(suave), "range\n$0.055$", color=MUTE)
    a2.text(2.0, 0.72,
            "same five information levels,\n"
            "$5.1\\times$ $[3.4,\\,8.2]$ less range to play for",
            ha="center", fontsize=8.5, color=INK, linespacing=1.5)
    fig.subplots_adjust(wspace=0.10)
    return guarda(fig, "fig_cliff_ladder.png")


def _caja(ax, x, y, w, h, fc, ec, titulo, detalle, veredicto, col):
    """Caja de tamano UNIFORME: titulo, una linea de detalle y el veredicto
    del paper para ese componente, siempre en la misma posicion."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        fc=fc, ec=ec, lw=1.1, zorder=3))
    ax.text(x, y + h * 0.30, titulo, ha="center", va="center",
            fontsize=9, weight="bold", zorder=4)
    ax.text(x, y + h * 0.02, detalle, ha="center", va="center",
            fontsize=7.5, color=INK, zorder=4, linespacing=1.35)
    ax.plot([x - w * 0.36, x + w * 0.36], [y - h * 0.10] * 2,
            color=ec, lw=0.6, alpha=0.5, zorder=4)
    ax.text(x, y - h * 0.325, veredicto, ha="center", va="center",
            fontsize=7.5, color=col, style="italic", zorder=4,
            linespacing=1.3)


def fig_arquitectura():
    """Espina horizontal (el camino de la informacion) con el campo
    DIRECTAMENTE encima del reasoner, que es a quien modula: la relacion de
    control es vertical y la de flujo horizontal, y asi se leen sin cruzarse.
    Cada caja lleva el veredicto del paper en el mismo sitio."""
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    ax.set_xlim(0.05, 10.75)
    ax.set_ylim(0.28, 5.05)
    ax.axis("off")

    W, H, y0 = 2.4, 1.42, 1.78
    xs = [1.32, 4.02, 6.72, 9.42]

    _caja(ax, xs[0], y0, W, H, "#f3f5f7", MUTE, "backbone",
          "decoder-only transformer", "trained", MUTE)
    _caja(ax, xs[1], y0, W, H, "#e8f0f5", FIELD, "reasoner",
          "adaptive-depth recurrent loop",
          "competence and\nstopping emerge", FIELD)
    _caja(ax, xs[2], y0, W, H, "#eaf1ea", LEARN, "value module",
          "self-models $\\hat{s},\\hat{p}$",
          "value does not emerge", LEARN)
    _caja(ax, xs[3], y0, W, H, "#f3f5f7", INK, "allocator",
          "$\\arg\\max_n\\ \\hat{s}\\,\\hat{p}(n)-\\lambda n$",
          "computed, not learned", MUTE)

    # el campo, encima del reasoner: plano de control
    _caja(ax, xs[1], 4.16, W + 0.5, H - 0.22, "#f8f1e6", AMBER,
          "homeostatic field", "certified PDE family",
          "modulates, does not think", AMBER)

    # flujo horizontal
    fl = dict(arrowstyle="-|>", mutation_scale=10, color=MUTE, lw=1.0)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + W / 2 + 0.04, y0),
                                     (b - W / 2 - 0.04, y0), **fl))

    # par vertical de control, separado y sin rotar etiquetas
    yb, yt = y0 + H / 2 + 0.04, 4.16 - (H - 0.22) / 2 - 0.04
    ax.add_patch(FancyArrowPatch((xs[1] - 0.42, yb), (xs[1] - 0.42, yt),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=WAVE, lw=1.0))
    ax.add_patch(FancyArrowPatch((xs[1] + 0.42, yt), (xs[1] + 0.42, yb),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=AMBER, lw=1.0))
    ax.text(xs[1] - 0.58, (yb + yt) / 2, "interoception", fontsize=7.5,
            color=WAVE, ha="right", va="center")
    ax.text(xs[1] + 0.58, (yb + yt) / 2, "modulation", fontsize=7.5,
            color=AMBER, ha="left", va="center")

    # la nota ANCLADA al reasoner, no flotando
    ax.annotate("posterior halting adds $+0.001$\n"
                "at matched readout and budget",
                xy=(xs[1], y0 - H / 2 - 0.05), xytext=(xs[1], 0.52),
                ha="center", va="center", fontsize=7.5, color=WAVE,
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=WAVE, lw=0.7,
                                shrinkA=1, shrinkB=1))
    return guarda(fig, "fig_architecture.png")


if __name__ == "__main__":
    for f in (fig_jerarquia, fig_acantilado, fig_arquitectura):
        print(f())
