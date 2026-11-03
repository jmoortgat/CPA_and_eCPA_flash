"""Canonical paper-figure style for the multi-salt eCPA project.

Implements the figure style of the papers:
- bold text, large fonts, thick black frame, no figure titles
- colorblind-safe palette + matched line styles
- post-layout tick-label bolding (matplotlib's ``tight_layout`` sometimes
  unbolds them)

Usage:
    from paper_style import apply_style, bold_ticks, WONG, LINESTYLES
    from paper_style import PAPER_FIGURES, FIGSIZE_1COL, FIGSIZE_2COL

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    ax.plot(x, y, color=WONG[0], linestyle=LINESTYLES[0])
    ax.set_xlabel(r"$T$ [K]")          # short labels; conditions go in captions
    # NO ax.set_title()
    bold_ticks(ax)
    fig.savefig(PAPER_FIGURES / "fig_name.png", bbox_inches="tight")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parents[1]


def _paper_figures_dir() -> Path:
    """Output folder for the figures: $ECPA_PAPER_FIGURES, else <package>/figures."""
    import os
    env = os.environ.get("ECPA_PAPER_FIGURES")
    out = Path(env) if env else _CODE_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


PAPER_FIGURES = _paper_figures_dir()

# ---------------------------------------------------------------------------
# Colors and line styles (colorblind-safe)
# ---------------------------------------------------------------------------

# Wong (2011) 8-color colorblind-safe palette.
WONG = [
    "#000000",  # 0  black
    "#E69F00",  # 1  orange
    "#56B4E9",  # 2  sky blue
    "#009E73",  # 3  bluish green
    "#F0E442",  # 4  yellow
    "#0072B2",  # 5  blue
    "#D55E00",  # 6  vermillion
    "#CC79A7",  # 7  reddish purple
]

# Pair with line styles so curves stay distinguishable in grayscale and to
# colorblind readers.
LINESTYLES = [
    "-",                              # solid
    "--",                             # dashed
    "-.",                             # dash-dot
    ":",                              # dotted
    (0, (3, 1, 1, 1, 1, 1)),          # dash-dot-dot
    (0, (5, 1)),                      # long dash
    (0, (1, 1)),                      # densely dotted
    (0, (3, 3, 1, 3)),                # custom
]


# ---------------------------------------------------------------------------
# Standard figure-size presets (inches; assume LaTeX letter, ~6.5" textwidth)
# ---------------------------------------------------------------------------
FIGSIZE_1COL = (3.5, 3.0)        # single column
FIGSIZE_1COL_TALL = (3.5, 4.0)
FIGSIZE_15COL = (5.5, 4.0)       # 1.5 column / boutique
FIGSIZE_2COL = (7.0, 4.5)        # double column
FIGSIZE_SQUARE = (5.0, 5.0)      # for parity plots
FIGSIZE_PANELS_5 = (12, 7.0)     # 2 rows x 3 cols (one slot blanked)


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------
_STYLE_APPLIED = False


def apply_style(force: bool = False) -> None:
    """Apply the paper-figure rcParams. Idempotent unless ``force=True``."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED and not force:
        return
    mpl.rcParams.update({
        # Font sizes
        "font.size":          14,
        "axes.labelsize":     16,
        "axes.titlesize":     16,        # we don't use titles, but if we did
        "xtick.labelsize":    13,
        "ytick.labelsize":    13,
        "legend.fontsize":    12,
        # Bold everything
        "font.weight":        "bold",
        "axes.labelweight":   "bold",
        "axes.titleweight":   "bold",
        # Frame thickness + color
        "axes.linewidth":     1.6,
        "axes.edgecolor":     "black",
        # Tick thickness + length + color + direction
        "xtick.major.width":  1.6,
        "ytick.major.width":  1.6,
        "xtick.major.size":   6,
        "ytick.major.size":   6,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        # Curve linewidth + marker size defaults
        "lines.linewidth":    2.2,
        "lines.markersize":   7,
        # Legend frame
        "legend.frameon":     True,
        "legend.edgecolor":   "black",
        # Save quality
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })
    _STYLE_APPLIED = True


def bold_ticks(ax) -> None:
    """Force tick labels bold (some matplotlib versions unbold them in
    ``tight_layout`` after ``apply_style`` set them bold)."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def bold_colorbar(cbar) -> None:
    """Force colorbar tick labels and label bold."""
    for lbl in cbar.ax.get_yticklabels() + cbar.ax.get_xticklabels():
        lbl.set_fontweight("bold")
    # Label fontweight is normally set when ``cbar.set_label`` is called with
    # ``fontweight='bold'``; this helper handles the case where ``set_label``
    # was called without that argument.
    cbar.ax.yaxis.label.set_fontweight("bold")
    cbar.ax.xaxis.label.set_fontweight("bold")


def panel_label(ax, letter: str, x: float = 0.06, y: float = 0.94,
                fontsize: int = 14) -> None:
    """Place an ``(a)``-style panel label inside the upper-left of an axes,
    with a translucent white box behind it so it stays readable over data."""
    ax.text(
        x, y, f"({letter})",
        transform=ax.transAxes,
        fontweight="bold", fontsize=fontsize, va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
    )


def save_figure(fig, path: Path, **kw) -> None:
    """Save with the standard bbox / dpi settings (override via kw)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = dict(bbox_inches="tight")
    defaults.update(kw)
    fig.savefig(path, **defaults)
