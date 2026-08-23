"""Figuras conceptuales del paper del gobernador. TODO EN INGLES —las dos
estaban en espanol dentro de un paper en ingles— y con el estilo comun de
figstyle.py.

  fig_hbp_concept.png  : el HBP como campo homeostatico sobre el grafo de
                         modulos, con la interfaz interocepcion/modulacion.
  fig_vei_dynamics.png : dinamica REAL del VEI, generada corriendo hbp.py.
"""
import os
import sys

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from figstyle import (AMBER, FIELD, GREY, INK, LEARN, MUTE, VIOLET, WAVE,
                      aplica, guarda, limpia)

aplica()
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

C_BLOCK, C_BLOCK_E = "#eef2f6", MUTE
C_REAS, C_REAS_E = "#e8f0f5", FIELD
C_WM, C_WM_E = "#eaf1ea", LEARN
C_FIELD, C_FIELD_E = "#efecf6", VIOLET


def _nodo(ax, x, y, w, h, fc, ec, label, sub):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.text(x, y + h * 0.20, label, ha="center", va="center",
            fontsize=9, weight="bold", zorder=4)
    ax.text(x, y - h * 0.22, sub, ha="center", va="center", fontsize=7.5,
            color=MUTE, zorder=4, linespacing=1.3)


def _hbp_entrenado(n_nodes=6):
    """Carga el HBP de un checkpoint entrenado si lo hay; si no, uno fresco.
    Devuelve (hbp, etiqueta) para poder declarar cual se uso."""
    import glob
    import torch
    from model.hbp import HBPConfig, HomeostaticBackgroundProcessor
    for p in sorted(glob.glob(os.path.join(HERE, "..", "checkpoints", "*.pt"))):
        try:
            obj = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        if not isinstance(sd, dict) or "hbp.h_star" not in sd:
            continue
        hs = sd["hbp.h_star"]
        cfg = HBPConfig(n_nodes=int(hs.shape[0]))
        if cfg.d_h != int(hs.shape[1]):     # d_h = d_f + d_e + d_u, derivado
            continue
        hbp = HomeostaticBackgroundProcessor(cfg)
        sub = {k[len("hbp."):]: v for k, v in sd.items() if k.startswith("hbp.")}
        faltan = hbp.load_state_dict(sub, strict=False)
        if len(faltan.missing_keys) < 8:
            hbp.eval()
            return hbp, os.path.basename(p)
    from model.hbp import HBPConfig as C2, HomeostaticBackgroundProcessor as H2
    cfg = C2(n_nodes=n_nodes, omega0_init=0.5, zeta_init=0.5, c_init=0.4)
    return H2(cfg), None


def _episodio(hbp, T=14):
    """Una 'episodio de razonamiento': el campo tickea una vez por iteracion
    del reasoner y de ahi sale el liston de parada que el reasoner ve."""
    import torch
    cfg = hbp.cfg
    N = cfg.n_nodes
    hbp.reset_state(1)
    vei, tau = [], []
    with torch.no_grad():
        for t in range(T):
            intero = torch.zeros(1, N, cfg.d_intero)
            if t < 2:                       # el input siembra el campo
                intero[0, min(3, N - 1)] = 2.0
            hbp.step(intero)
            d = (hbp.h_t - hbp.h_star.unsqueeze(0))[0]
            vei.append(float(d.pow(2).mean(dim=1).sqrt().max()))
            m = hbp.modulation()["halt_threshold"][0]
            tau.append(float(m[min(3, N - 1)]))
    return np.arange(1, T + 1), np.array(vei), np.array(tau)


def fig_concept():
    """DOS PANELES, porque una sola vista no representa lo que el paper dice.

    (a) El acoplamiento: el campo sobre el grafo de modulos, con el corredor
        de interocepcion/modulacion despejado.
    (b) Lo que ninguna version anterior mostraba y es la tesis entera: el
        campo TICKEA UNA VEZ POR ITERACION DEL REASONER, y su estado mueve el
        LISTON DE PARADA. Es decir, modula CUANTO se computa, nunca QUE. La
        traza sale de correr el solver real con un checkpoint entrenado.
    """
    fig = plt.figure(figsize=(10.0, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0.1, 9.9)
    ax.set_ylim(0.10, 9.05)
    ax.axis("off")

    # (a) DOS CAPAS APILADAS. Un diagrama horizontal de cuatro cajas no cabe
    # en medio panel sin comprimirse, y la ecuacion flotando arriba no
    # pertenecia a nada. Aqui el campo es una CAPA que contiene su propia ley
    # y sus nodos, y los modulos son otra capa debajo: el par
    # interocepcion/modulacion cruza el hueco entre las dos.
    xs = [1.95, 4.05, 6.15, 8.25]

    # --- capa del campo -------------------------------------------------
    ax.add_patch(FancyBboxPatch(
        (0.55, 5.86), 8.9, 3.01,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        fc="#f4f1fa", ec=C_FIELD_E, lw=1.1, alpha=0.95, zorder=1))
    ax.text(0.95, 8.56, "the field", fontsize=8.5, weight="bold",
            color=C_FIELD_E, ha="left", va="center", zorder=3)
    ax.text(5.0, 7.98,
            r"$\ddot h + 2\zeta\omega_0\dot h - c^2\nabla^2 h"
            r" + \omega_0^2\,(h-h^\ast) \;=\; f_\theta + g_\phi$",
            ha="center", va="center", fontsize=9.5, color=INK, zorder=3)
    ax.text(5.0, 7.32, "a damped wave over the module graph,\n"
                       r"coupled by $c^2\nabla^2$ — one tick per reasoner"
                       " iteration",
            ha="center", va="center", fontsize=7.5, color=MUTE, zorder=3,
            linespacing=1.45)

    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.plot([x0 + 0.30, x1 - 0.30], [6.35, 6.35], color=C_FIELD_E,
                lw=2.0, alpha=0.75, zorder=2)
    for i, x in enumerate(xs):
        ax.add_patch(Circle((x, 6.35), 0.30, fc="white", ec=C_FIELD_E,
                            lw=1.4, zorder=3))
        ax.text(x, 6.35, rf"$h_{{{i+1}}}$", ha="center", va="center",
                fontsize=8.5, color=C_FIELD_E, zorder=4)

    # --- el hueco: interocepcion sube, modulacion baja ------------------
    for x in xs:
        ax.add_patch(FancyArrowPatch(
            (x - 0.20, 4.20), (x - 0.20, 5.96), arrowstyle="-|>",
            mutation_scale=9, color=WAVE, lw=1.0, zorder=2))
        ax.add_patch(FancyArrowPatch(
            (x + 0.20, 5.96), (x + 0.20, 4.20), arrowstyle="-|>",
            mutation_scale=9, color=FIELD, lw=1.0, zorder=2))

    # --- capa de los modulos --------------------------------------------
    ax.add_patch(FancyBboxPatch(
        (0.55, 2.05), 8.9, 2.02,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        fc="#f6f8f9", ec=MUTE, lw=1.1, zorder=1))
    ax.text(0.95, 3.78, "the model", fontsize=8.5, weight="bold",
            color=INK, ha="left", va="center", zorder=3)

    chips = [("Block 1", C_BLOCK, C_BLOCK_E),
             (r"Block $L$", C_BLOCK, C_BLOCK_E),
             ("Reasoner", C_REAS, C_REAS_E),
             ("Memory", C_WM, C_WM_E)]
    for x, (lab, fc, ec) in zip(xs, chips):
        ax.add_patch(FancyBboxPatch(
            (x - 0.88, 2.44), 1.76, 0.70,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            fc=fc, ec=ec, lw=1.2, zorder=3))
        ax.text(x, 2.79, lab, ha="center", va="center", fontsize=8.5,
                weight="bold", zorder=4)
    ax.text((xs[0] + xs[1]) / 2, 2.79, r"$\cdots$", fontsize=11,
            ha="center", va="center", color=MUTE, zorder=4)
    ax.text(xs[2], 2.22, r"iterates $\times\, n_{\mathrm{iter}}$",
            ha="center", va="center", fontsize=7, color=C_REAS_E, zorder=4)

    # --- lo que la modulacion toca --------------------------------------
    ax.text(5.0, 1.30, r"halting threshold $\cdot$ memory gates "
                       r"$\cdot$ router bias",
            ha="center", fontsize=7.5, color=FIELD)
    ax.text(5.0, 0.80, "the field never touches the content, only the budget",
            ha="center", fontsize=8, color=INK, style="italic")

    leg = [Line2D([0], [0], color=WAVE, lw=1.8,
                  label="interoception (bottom-up)"),
           Line2D([0], [0], color=FIELD, lw=1.8,
                  label="modulation (top-down)")]
    ax.legend(handles=leg, loc="lower center", ncol=2, fontsize=7.5,
              bbox_to_anchor=(0.5, -0.03))
    ax.set_title("(a)  a field layer over the model", fontsize=9.5,
                 weight="bold", loc="left", color=INK, pad=2)

    # ---------------- (b) el tick, que es lo que define al HBP ----------
    a2 = fig.add_subplot(gs[0, 1])
    hbp, ck = _hbp_entrenado()
    t, vei, tau = _episodio(hbp)

    a2.plot(t, vei, "-o", ms=3.2, lw=1.4, color=C_FIELD_E,
            label=r"field state $\|h-h^\ast\|$")
    a2.set_xlabel("reasoner iteration (one field tick each)")
    a2.set_ylabel(r"$\|h-h^\ast\|$ at the reasoner node", color=C_FIELD_E)
    a2.tick_params(axis="y", colors=C_FIELD_E)
    a2.set_ylim(0, max(vei) * 1.35)
    limpia(a2)

    a3 = a2.twinx()
    a3.axhline(0.5, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    a3.plot(t, tau, "-s", ms=3.0, lw=1.4, color=AMBER, zorder=2,
            label=r"halting bar $\tau$")
    # eje HONESTO: 0.5 es "sin modulacion" y entra en el rango, para que no
    # parezca grande un desplazamiento de milesimas
    lo = min(0.4905, float(tau.min()) - 0.002)
    a3.set_ylim(lo, 0.5 + (0.5 - lo) * 0.42)
    a3.set_ylabel(r"halting bar $\tau$", color=AMBER)
    a3.tick_params(axis="y", colors=AMBER)
    for sp in ("top", "left"):
        a3.spines[sp].set_visible(False)
    a3.spines["right"].set_color(AMBER)
    a3.text(t[-1] + 0.15, 0.5, r"neutral ($\tau{=}0.5$)",
            fontsize=7, color=MUTE, va="bottom", ha="right")

    a3.annotate("the field settles,\nand the bar with it",
                xy=(float(t[-3]), float(tau[-3])),
                xytext=(0.44, 0.30), textcoords="axes fraction",
                fontsize=7.5, color=MUTE, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.7,
                                connectionstyle="arc3,rad=-0.25"))
    a2.set_title("(b)  one tick per iteration: the field moves the bar",
                 fontsize=9.5, weight="bold", loc="left", color=INK, pad=2)

    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a3.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5)
    a2.text(0.5, -0.24, "it sets how much to compute, never what",
            transform=a2.transAxes, ha="center", fontsize=8.5,
            color=INK, style="italic")
    if ck:
        a2.text(0.99, 1.08, f"solver run on {ck[:26]}", transform=a2.transAxes,
                ha="right", fontsize=6.5, color=MUTE)
    return guarda(fig, "fig_hbp_concept.png")


def fig_dynamics():
    """Corre el HBP real: impulso interoceptivo en el nodo 1 -> oscilacion
    amortiguada que se propaga por el laplaciano."""
    import torch
    sys.path.insert(0, os.path.join(HERE, ".."))
    from model.hbp import HBPConfig, HomeostaticBackgroundProcessor
    torch.manual_seed(0)
    cfg = HBPConfig(n_nodes=6, omega0_init=1.2, omega0_max=1.8,
                    zeta_init=0.12, zeta_min=0.05, c_init=0.55, c_max=0.7,
                    dt=1.0, h_clamp=50.0)
    hbp = HomeostaticBackgroundProcessor(cfg)
    hbp.reset_state(1)
    T, N = 60, cfg.n_nodes
    dev = np.zeros((T, N))
    for t in range(T):
        intero = torch.zeros(1, N, cfg.d_intero)
        if t < 3:
            intero[0, 0] = 2.5
        hbp.step(intero)
        d = (hbp.h_t - hbp.h_star.unsqueeze(0))[0].detach().numpy()
        dev[t] = np.sqrt((d ** 2).mean(axis=1))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                 gridspec_kw={"width_ratios": [1.12, 1]})
    tin = plt.cm.viridis(np.linspace(0.12, 0.88, N))
    for i in range(N):
        a1.plot(dev[:, i], color=tin[i], lw=1.4)
        # etiquetado directo en vez de leyenda: menos ruido
        a1.text(T + 0.8, dev[-1, i], f"$h_{{{i+1}}}$", fontsize=7.5,
                color=tin[i], va="center")
    a1.set_xlabel("reasoner tick")
    a1.set_ylabel(r"$\|h_i-h^\ast\|$  (VEI per node)")
    a1.set_title("damped oscillation of the VEI", pad=8)
    a1.set_xlim(0, T + 4)
    limpia(a1)

    im = a2.imshow(dev.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, T, 0.5, N + 0.5])
    a2.set_xlabel("reasoner tick")
    a2.set_ylabel("graph node")
    a2.set_title("propagation across modules", pad=8)
    a2.set_yticks(range(1, N + 1))
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03)
    cb.set_label(r"$\|h_i-h^\ast\|$", fontsize=8.5)
    cb.outline.set_visible(False)
    fig.subplots_adjust(wspace=0.28)
    return guarda(fig, "fig_vei_dynamics.png")


if __name__ == "__main__":
    print(fig_concept())
    print(fig_dynamics())
