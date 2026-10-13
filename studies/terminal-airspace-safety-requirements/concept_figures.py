"""
Draw the two conceptual figures of the paper.

    python studies/terminal-airspace-safety-requirements/concept_figures.py

Outputs
  figures/wellclear_pattern.pdf           Fig. 2  well-clear volumes against the traffic pattern
  figures/monitor_encounter.pdf           Fig. 3  timing of SR-6 (t_det, t_mon, t_wc)

The pattern geometry is the one simulated by TERAS (teras/traffic_pattern.py):
5000-ft runway 09, left-hand pattern, downwind leg 0.75 NM from the runway.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, PathPatch
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repository root
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from teras.traffic_pattern import AirfieldConfig, NM_TO_FT  # noqa: E402
import figstyle as fs  # noqa: E402


def aircraft(ax, x, y, heading_deg, size, filled, z=6, color=None):
    """Isosceles triangle pointing along the heading (0 = north, 90 = east)."""
    h = np.radians(heading_deg)
    fwd = np.array([np.sin(h), np.cos(h)])
    side = np.array([fwd[1], -fwd[0]])
    p0 = np.array([x, y])
    pts = [p0 + fwd * size, p0 - fwd * size * 0.6 + side * size * 0.6,
           p0 - fwd * size * 0.6 - side * size * 0.6]
    c = color or fs.INK
    ax.add_patch(Polygon(pts, closed=True, fc=c if filled else "white",
                         ec=c, lw=0.8, zorder=z))


def dim_arrow(ax, p, q, text, offset=(0, 0), ha="center", va="center", rot=0):
    """Two-headed dimension arrow with a label at its midpoint."""
    ax.annotate("", xy=p, xytext=q,
                arrowprops=dict(arrowstyle="<->", lw=0.5, color=fs.INK, shrinkA=0, shrinkB=0))
    mx, my = (p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1]
    ax.text(mx, my, text, ha=ha, va=va, fontsize=7, rotation=rot)


# ---------------------------------------------------------------------------
def wellclear_pattern(out_dir):
    """Fig. 2: standard well-clear volumes around an ownship following traffic on downwind."""
    cfg = AirfieldConfig()
    L = cfg.runway_length / NM_TO_FT               # 0.82 NM
    w = cfg.pattern_width / NM_TO_FT               # 0.75 NM
    x_up = (cfg.runway_length + 2500) / NM_TO_FT   # crosswind leg
    x_base = cfg.base_leg_x / NM_TO_FT             # base leg (1 NM before the threshold)
    r = 0.12                                       # corner radius (NM)

    fig, ax = plt.subplots(figsize=(0.85 * fs.COL_W, 1.9), layout="constrained")

    # runway
    ax.plot([0, L], [0, 0], color=fs.MID, lw=3.2, solid_capstyle="butt", zorder=2)
    ax.text(L / 2, -0.07, "runway 09", ha="center", va="top", fontsize=7, color=fs.DARK)

    # left-hand pattern (rectangle with rounded corners) and direction arrows
    verts = [(x_base + r, 0), (x_up - r, 0), (x_up, 0), (x_up, r), (x_up, w - r), (x_up, w),
             (x_up - r, w), (x_base + r, w), (x_base, w), (x_base, w - r), (x_base, r),
             (x_base, 0), (x_base + r, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3,
             Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3,
             Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec=fs.LIGHT, lw=0.8, zorder=1))
    for (x0, y0, dx, dy) in [(x_up, w * 0.45, 0, 0.001), (-0.15, w, -0.001, 0),
                             (x_base, w * 0.55, 0, -0.001)]:
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=fs.LIGHT, lw=0.8,
                                    mutation_scale=7))
    ax.text(-0.15, w + 0.04, "downwind", ha="center", va="bottom", fontsize=7, color=fs.DARK)

    # two aircraft in trail on the downwind leg (heading 270)
    own = np.array([0.95, w])
    spacing = 0.6
    lead = own - np.array([spacing, 0])
    aircraft(ax, *own, 270, 0.045, filled=True, color=fs.OWN)
    aircraft(ax, *lead, 270, 0.045, filled=False, color=fs.CREW)
    ax.text(own[0], own[1] - 0.07, "ownship", ha="center", va="top", fontsize=7, color=fs.OWN)
    ax.annotate("crewed traffic", xy=(lead[0] - 0.05, lead[1] + 0.03), xytext=(-0.4, 1.12),
                ha="center", va="bottom", fontsize=7, color=fs.CREW,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=fs.CREW, shrinkA=1, shrinkB=1))

    # standard well-clear volumes (horizontal radius) centered on the ownship
    r_en = 4000 / NM_TO_FT     # DO-365 en route
    r_su = 2000 / NM_TO_FT     # ASTM F3442 small UAS
    r_te = 1500 / NM_TO_FT     # DO-365 terminal area
    ax.add_patch(Circle(own, r_te, fc=fs.TERM_TINT, ec=fs.TERM, lw=0.7, zorder=1))
    ax.add_patch(Circle(own, r_su, fc="none", ec=fs.MID, lw=0.6, ls=(0, (1, 1.2)), zorder=3))
    ax.add_patch(Circle(own, r_en, fc="none", ec=fs.ALERT, lw=0.9, ls=(0, (4, 2)), zorder=3))

    def label(r, ang, text, xy_text, color):
        a = np.radians(ang)
        ax.annotate(text, xy=(own[0] + r * np.cos(a), own[1] + r * np.sin(a)), xytext=xy_text,
                    ha="left", va="center", fontsize=7, color=color,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=color))

    label(r_en, 35, "DO-365 en route\n4000 ft", (1.6, 1.6), fs.ALERT)
    label(r_su, -20, "ASTM F3442\n2000 ft", (1.62, 0.62), fs.DARK)
    label(r_te, -70, "DO-365 terminal\n1500 ft", (1.62, 0.12), fs.TERM)

    # dimensions
    yd = 1.6
    dim_arrow(ax, (lead[0], yd), (own[0], yd), "0.6 NM", offset=(0, 0.05), va="bottom")
    ax.plot([lead[0], lead[0]], [w + 0.07, yd + 0.03], color=fs.LIGHT, lw=0.4, zorder=0)
    ax.plot([own[0], own[0]], [w + 0.07, yd + 0.03], color=fs.LIGHT, lw=0.4, zorder=0)
    dim_arrow(ax, (x_base - 0.2, 0), (x_base - 0.2, w), "0.75 NM", offset=(-0.04, 0), ha="right", rot=90)

    ax.set_aspect("equal")
    ax.set_xlim(x_base - 0.45, 2.35)
    ax.set_ylim(-0.25, 1.9)
    ax.set_xlabel("East (NM)")
    ax.set_ylabel("North (NM)")
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0, 1.5])
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.spines["bottom"].set_bounds(x_base - 0.45, 1.6)
    fs.save(fig, out_dir, "wellclear_pattern")


# ---------------------------------------------------------------------------
def monitor_encounter(out_dir):
    """Fig. 3: SR-6 timing in the ownship frame.

    The ownship is fixed on the right and the intruder approaches from the
    left; the relative trajectory doubles as the time axis.
    """
    R_wc, R_det = 0.7, 3.0  # illustrative radii
    fig, ax = plt.subplots(figsize=(0.9 * fs.COL_W, 1.2), layout="constrained")

    # well-clear volume (circle) and detection range (arc)
    ax.add_patch(Circle((0, 0), R_wc, fc=fs.ALERT_TINT, ec=fs.ALERT, lw=0.8))
    th = np.radians(np.linspace(160, 200, 60))
    ax.plot(R_det * np.cos(th), R_det * np.sin(th), color=fs.OWN, lw=0.8, ls=(0, (4, 2)))
    ax.text(R_det * np.cos(np.radians(160)) + 0.1, R_det * np.sin(np.radians(160)) + 0.08,
            "detection range", ha="center", va="bottom", fontsize=7, color=fs.OWN)
    ax.text(0.55, 0.62, "well clear", ha="left", va="bottom", fontsize=7, color=fs.ALERT)
    aircraft(ax, 0, 0, 90, 0.13, filled=True, color=fs.OWN)
    ax.text(0, -0.2, "ownship", ha="center", va="top", fontsize=7, color=fs.OWN)

    x_det, x_wc = -R_det, -R_wc
    x_mon = x_wc + 0.42 * (x_det - x_wc)
    x0 = x_det - 1.0
    ax.annotate("", xy=(-0.25, 0), xytext=(x0 + 0.15, 0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=fs.CREW, mutation_scale=7,
                                linestyle=(0, (1.5, 1.5))))
    aircraft(ax, x0, 0, 90, 0.13, filled=False, color=fs.CREW)
    ax.text(x0, 0.2, "intruder", ha="center", va="bottom", fontsize=7, color=fs.CREW)
    for xv, lab in ((x_det, r"$t_{det}$"), (x_mon, r"$t_{mon}$"), (x_wc, r"$t_{wc}$")):
        ax.plot(xv, 0, "o", ms=3.2, mfc="white", mec=fs.INK, mew=0.7, zorder=5)
        ax.text(xv - (0.12 if xv == x_wc else 0), 0.13, lab, ha="center", va="bottom",
                fontsize=8, bbox=dict(fc="white", ec="none", pad=0.3))

    dim_arrow(ax, (x_mon, -0.42), (x_wc, -0.42), r"$\geq T_{rec}$", offset=(0, -0.08), va="top")
    dim_arrow(ax, (x_det, -0.95), (x_wc, -0.95), r"bounded above by $R_{det}$",
              offset=(0, -0.08), va="top")
    for xv, y1 in ((x_det, -0.95), (x_mon, -0.42), (x_wc, -0.95)):
        ax.plot([xv, xv], [-0.06, y1 - 0.04], color=fs.LIGHT, lw=0.4, zorder=0)

    ax.set_aspect("equal")
    ax.set_xlim(x0 - 0.3, 1.5)
    ax.set_ylim(-1.45, 1.15)
    ax.axis("off")
    fs.save(fig, out_dir, "monitor_encounter")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"),
                    help="directory for the figures (default: figures/)")
    args = ap.parse_args()
    fs.apply()
    wellclear_pattern(args.out_dir)
    monitor_encounter(args.out_dir)
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
