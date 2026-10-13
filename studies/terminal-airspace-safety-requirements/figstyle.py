"""
Shared plotting style for the paper figures.

Figures are drawn at their final printed width in a two-column layout
(column = 3.5 in, page = 7.16 in), so text prints at 7-8 pt without
rescaling in LaTeX. Colors carry the same meaning in every figure.
"""

import matplotlib.pyplot as plt

COL_W = 3.5      # column width (in)
PAGE_W = 7.16    # full text width (in)

# Neutrals
INK = "#000000"
DARK = "#404040"
MID = "#808080"
LIGHT = "#BFBFBF"
FILL = "#EDEDED"

# Muted palette with a fixed meaning across figures
OWN = "#2B5C8A"        # ownship / uncrewed aircraft / with F8 (blue)
OWN_TINT = "#DCE7F2"
CREW = "#C9692C"       # crewed traffic / intruder (orange)
ALERT = "#B03A3A"      # en-route threshold, loss of well clear, unsafe region (red)
ALERT_TINT = "#F6E1E0"
TERM = "#3B8A78"       # DO-365 terminal-area volume (teal)
TERM_TINT = "#DDEFEA"
BLUES = ["#9CBCDB", "#5E8FC0", "#2B5C8A", "#173A5E"]  # ordered series, light to dark


def apply():
    """Set matplotlib rcParams for all paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.4,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
    })


def save(fig, out_dir, name):
    """Save a figure as vector PDF (fonts embedded)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
