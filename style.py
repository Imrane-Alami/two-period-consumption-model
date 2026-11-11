"""Shared plotting style: textbook look (LaTeX font, arrow axes, no grid)."""
from pathlib import Path

import matplotlib.pyplot as plt

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Textbook style: LaTeX font, arrow axes, no grid
# ------------------------------------------------------------------
from matplotlib import font_manager

for font in Path("/usr/share/texmf/fonts/opentype/public/lm").glob("lmroman10-*.otf"):
    font_manager.fontManager.addfont(str(font))   # Latin Modern if installed (TeX Live)

BLACK, GREEN, ORANGE, BLUE, GREY = "#000000", "#0b6e4f", "#d9731f", "#1f4e9c", "#6b6b6b"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11, "axes.labelsize": 11, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.dpi": 110,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def textbook_axes(ax, xlabel, ylabel, xmax, ymax):
    """Textbook axes: two arrows from the origin, no tick marks."""
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, xmax * 1.06); ax.set_ylim(0, ymax * 1.08)
    arrow = dict(arrowstyle="-|>", lw=0.8, color=BLACK, mutation_scale=9)
    ax.annotate("", (xmax * 1.06, 0), (0, 0), arrowprops=arrow, annotation_clip=False)
    ax.annotate("", (0, ymax * 1.08), (0, 0), arrowprops=arrow, annotation_clip=False)
    ax.text(xmax * 1.08, 0, xlabel, ha="left", va="center")
    ax.text(-xmax * 0.02, ymax * 1.08, ylabel, ha="right", va="center")


def project(ax, x, y, label_x=None, label_y=None, color=BLACK):
    """Dashed lines from (x, y) to both axes, with labels on the axes."""
    ax.plot([x, x], [0, y], ls=(0, (3, 3)), lw=0.8, color=color)
    ax.plot([0, x], [y, y], ls=(0, (3, 3)), lw=0.8, color=color)
    lim_y = ax.get_ylim()[1]
    lim_x = ax.get_xlim()[1]
    if label_x:
        ax.text(x, -lim_y * 0.035, label_x, ha="center", va="top")
    if label_y:
        ax.text(-lim_x * 0.015, y, label_y, ha="right", va="center")


def label(ax, x, y, text, color=BLACK, **kw):
    """Curve name written at its end (no legend box)."""
    ax.text(x, y, text, color=color, va="center", **kw)


def caption(fig, text, y=-0.02):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=10.5)


def style_series(ax):
    """Time series / data charts: plain, thin graduated axes."""
    ax.spines["left"].set_linewidth(0.8); ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=10)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf / .png")


