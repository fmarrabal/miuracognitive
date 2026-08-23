"""Identidad visual comun de las figuras de los dos papers.

Antes cada generador traia su propia paleta, sus rejillas y sus titulos —y
tres figuras estaban en ESPANOL dentro de papers en ingles. Este modulo
centraliza el estilo para que las seis parezcan del mismo documento:

  * tipografia serif con mathtext STIX, que casa con el cuerpo del paper
    (LaTeX article/Times) en vez del sans por defecto de matplotlib;
  * paleta sobria de seis tintas, una por rol semantico;
  * sin rejilla, sin marco: solo ejes izquierdo e inferior;
  * sin titulos dentro de las figuras de datos —de eso se encarga el
    \\caption del paper—; los esquemas conceptuales si los llevan porque
    orientan al lector dentro del dibujo.

Todo el texto de las figuras va en INGLES.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# --- paleta: una tinta por ROL, no por capricho ---------------------------
INK = "#1f2933"      # texto y ejes
MUTE = "#98a2ad"     # baselines, contexto, lo que no es el mensaje
GREY = "#c7ced6"     # relleno neutro
FIELD = "#2f6f8f"    # el campo / el brazo destacado (azul apagado)
WAVE = "#a3423c"     # 2o orden, y tambien lo retractado (ladrillo)
LEARN = "#4a7c59"    # baselines aprendidos (GRU) (verde apagado)
AMBER = "#b07d2b"    # modulacion top-down
VIOLET = "#6b5b95"   # el campo en los esquemas conceptuales

PALETA = [MUTE, FIELD, WAVE, LEARN]


def aplica():
    """rcParams comunes. Llamar una vez al importar el generador."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def limpia(ax, y_solo=False):
    """Quita adornos. y_solo deja solo el eje vertical (barras categoricas)."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if y_solo:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)


def etiqueta_barras(ax, barras, valores, fmt="{:.3f}", dy=0.006, fs=8.5,
                    color=None):
    """Valor encima de cada barra: sustituye a leer contra la rejilla."""
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=fs,
                color=color or INK)


def rango(ax, x, lo, hi, texto, color=None, dx=0.0):
    """Corchete vertical que marca un rango y lo anota. Es lo que hace
    visible una comparacion de AMPLITUDES entre paneles."""
    c = color or INK
    ax.annotate("", xy=(x + dx, hi), xytext=(x + dx, lo),
                arrowprops=dict(arrowstyle="<->", color=c, lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.text(x + dx + 0.12, (lo + hi) / 2, texto, ha="left", va="center",
            fontsize=8.5, color=c)


def guarda(fig, nombre):
    p = os.path.join(FIGS, nombre)
    fig.savefig(p)
    plt.close(fig)
    return p
