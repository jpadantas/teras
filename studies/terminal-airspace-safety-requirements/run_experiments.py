"""
Reproduce the simulation study of the paper (Section VII, Figs. 4 and 5).

    python studies/terminal-airspace-safety-requirements/run_experiments.py
    python studies/terminal-airspace-safety-requirements/run_experiments.py --skip-sensitivity

Outputs
  figures/fig_nuisance_alert_rates.pdf   Fig. 4  (Experiment 1)
  figures/fig_lead_time_recovery.pdf     Fig. 5  (Experiments 2 and 3, panels a-c)
  results/teras_results.txt              every number quoted in Section VII (incl. Experiment 4)

Well-clear volumes (standards, not assumptions)
  DO-365 en route  : HMD* 4000 ft, ZTHR 450 ft, tau*_mod 35 s
  DO-365 terminal  : HMD* 1500 ft, ZTHR 450 ft, no time threshold   <- used in Exps. 2-4
  ASTM F3442 (sUAS): HMD* 2000 ft, ZTHR 250 ft, no time threshold

Notes on interpreting the results
  * All runs are deterministic; counts are counts of geometries, not samples
    from a distribution of encounters.
  * Experiment 1: spacing is the initial along-track spacing on downwind; the
    run lasts 120 s, until the ownship is established on final. The report
    gives where the closest approach of each pair happens.
  * F6 triggers when the projected time to loss of well clear falls to
    T_rec + MonitorConfig.buffer_margin_s (1.0 s). With early detection the
    reported margin therefore equals that buffer by construction.
  * T_rec is an input to the monitor; the time the breakout really needs comes
    out of the vehicle model and the geometry. Experiment 2 shows the two
    disagreeing.
  * Speeds, T_rec and the breakout manoeuvre are assumed, not measured.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repository root
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from teras.wellclear import WellClearThresholds  # noqa: E402
from teras.traffic_pattern import NM_TO_FT  # noqa: E402
from teras.monitors import MonitorConfig  # noqa: E402
from teras.simulator import EncounterSimulator  # noqa: E402
from teras.scenarios import (  # noqa: E402
    create_nominal_pattern_encounter,
    create_base_final_conflict,
)
import figstyle as fs  # noqa: E402

DT = 0.1                  # single time step shared by trajectories and simulator (s)
TERMINAL = WellClearThresholds.do365_terminal()
ENROUTE = WellClearThresholds.do365_phase1()
SUAS = WellClearThresholds.astm_f3442()

# Experiment 2/3 encounter: ownship 80 kt turning base to final, intruder 95 kt
# straight-in, starting 1.8 NM before the threshold.
CONFLICT = dict(speed_own_kt=80.0, speed_intr_kt=95.0, intr_start_nm=1.8)
T_REC_ASSUMED = 6.0
T_REC_SUFFICIENT = 8.0
EXP1_WINDOW_S = 120.0


def _conflict(**kw):
    p = dict(CONFLICT)
    p.update(kw)
    return create_base_final_conflict(duration_s=80.0, dt=DT, **p)


def _run(own, intr, th=TERMINAL, recovery=True, **cfg):
    return EncounterSimulator(thresholds=th, monitor_config=MonitorConfig(**cfg)).run(
        own, intr, dt=DT, enable_recovery=recovery)


def needed_t_rec(own, intr, th=TERMINAL, grid=np.arange(0.5, 25.01, 0.5)):
    """Smallest T_rec on the grid for which the breakout keeps well clear (None if none)."""
    for t_rec in grid:
        rec = _run(own, intr, th, t_rec=t_rec)
        if rec.t_mon is not None and not rec.experienced_lowc:
            return float(t_rec)
    return None


def min_detection_range(own, intr, t_rec, th=TERMINAL, lo=0.1, hi=3.0, iters=12):
    """Detection range (NM) at which the lead-time margin reaches zero, by bisection."""
    def margin(r):
        return _run(own, intr, th, t_rec=t_rec, sensor_range_ft=r * NM_TO_FT).lead_time_margin_s

    m_hi = margin(hi)
    if m_hi is None or m_hi < 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = margin(mid)
        if m is not None and m >= 0:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# Experiment 1: alert burden against correctly flown in-trail traffic
# ---------------------------------------------------------------------------
def exp1_alert_burden(spacings_nm=None, hmd_values_ft=None):
    """Spacings (of 25) whose correctly flown in-trail pair enters the volume, per HMD*."""
    if spacings_nm is None:
        spacings_nm = np.linspace(0.4, 1.4, 25)
    if hmd_values_ft is None:
        hmd_values_ft = np.arange(1000.0, 4500.0 + 1, 125.0)

    trajs = [
        create_nominal_pattern_encounter(
            spacing_nm=sp, speed_own_kt=90.0, speed_intr_kt=85.0, duration_s=EXP1_WINDOW_S, dt=DT)
        for sp in spacings_nm
    ]

    def violating(th):
        sim = EncounterSimulator(thresholds=th)
        return [
            float(sp) for sp, (own, intr) in zip(spacings_nm, trajs)
            if sim.run(own, intr, dt=DT, enable_recovery=False,
                       enable_lookahead=False, early_stop_on_lowc=True).experienced_lowc
        ]

    variants = {"tau0_z450": (0.0, 450.0), "tau35_z450": (35.0, 450.0), "tau0_z250": (0.0, 250.0)}
    counts = {k: {} for k in variants}
    for h in hmd_values_ft:
        for k, (tau, z) in variants.items():
            counts[k][float(h)] = violating(WellClearThresholds(k, h, h, tau, z))

    standards = {"DO-365 terminal": violating(TERMINAL), "ASTM F3442": violating(SUAS),
                 "DO-365 en route": violating(ENROUTE)}

    closest = {}
    for sp, (own, intr) in zip(spacings_nm, trajs):
        r = EncounterSimulator().run(own, intr, dt=DT, enable_recovery=False, enable_lookahead=False)
        k = int(np.argmin(r.horizontal_ranges))
        closest[float(sp)] = (r.min_horizontal_separation, r.timestamps[k],
                              r.own_states[k].heading_deg, r.intr_states[k].heading_deg,
                              abs(r.own_states[k].z - r.intr_states[k].z))
    return dict(spacings=np.asarray(spacings_nm), hmd_values=np.asarray(hmd_values_ft),
                counts=counts, standards=standards, closest=closest)


# ---------------------------------------------------------------------------
# Experiment 2: lead-time budget and recovery in a base-to-final conflict
# ---------------------------------------------------------------------------
def exp2_lead_time():
    own, intr = _conflict()
    unmit = _run(own, intr, recovery=False, t_rec=T_REC_ASSUMED)
    rec_assumed = _run(own, intr, t_rec=T_REC_ASSUMED)
    t_need = needed_t_rec(own, intr)
    rec_ok = _run(own, intr, t_rec=T_REC_SUFFICIENT)
    neighbours = {}
    for s0 in (1.6, 1.7, 1.8, 1.9, 2.0):
        o, i = _conflict(intr_start_nm=s0)
        neighbours[s0] = needed_t_rec(o, i)
    return dict(own=own, intr=intr, unmit=unmit, rec_assumed=rec_assumed, rec_ok=rec_ok,
                t_need=t_need, neighbours=neighbours)


# ---------------------------------------------------------------------------
# Experiment 3: coupling between detection range (SR-1) and lead time (SR-6)
# ---------------------------------------------------------------------------
def exp3_coupling(ranges_nm=None, t_recs=(6.0, 8.0, 10.0, 12.0)):
    if ranges_nm is None:
        ranges_nm = np.linspace(0.3, 1.2, 91)
    own, intr = _conflict()
    margins, r_star = {}, {}
    for t_rec in t_recs:
        m = np.array([
            (lambda r: r.lead_time_margin_s if r.lead_time_margin_s is not None else np.nan)(
                _run(own, intr, t_rec=t_rec, sensor_range_ft=R * NM_TO_FT))
            for R in ranges_nm])
        margins[t_rec] = m
        pos = np.where(m >= 0)[0]
        if len(pos) and pos[0] > 0:
            k = pos[0]
            r_star[t_rec] = ranges_nm[k - 1] + (0 - m[k - 1]) * (
                ranges_nm[k] - ranges_nm[k - 1]) / (m[k] - m[k - 1])
        else:
            r_star[t_rec] = None
    return dict(ranges=np.asarray(ranges_nm), t_recs=list(t_recs), margins=margins, r_star=r_star)


# ---------------------------------------------------------------------------
# Experiment 4: sensitivity to speeds, worst case over intruder start positions
# ---------------------------------------------------------------------------
def exp4_speed_sensitivity(own_speeds=(70.0, 80.0, 90.0), intr_speeds=(80.0, 95.0, 110.0, 125.0),
                           starts_nm=np.arange(1.2, 3.01, 0.1), t_rec=10.0, verbose=True):
    rows = []
    for vo in own_speeds:
        for vi in intr_speeds:
            r_det, t_need, n_conf, n_miss = [], [], 0, 0
            for s0 in starts_nm:
                own, intr = _conflict(speed_own_kt=vo, speed_intr_kt=vi, intr_start_nm=float(s0))
                unmit = _run(own, intr, recovery=False, t_rec=t_rec)
                if unmit.t_wc_actual is None:
                    continue
                n_conf += 1
                # The monitor projects the current velocity; a conflict created by a later
                # turn may not be projected in time. Count those separately: their "needed
                # T_rec" would measure the projection, not the recovery.
                if unmit.t_mon is None or unmit.t_mon >= unmit.t_wc_actual:
                    n_miss += 1
                    continue
                r = min_detection_range(own, intr, t_rec)
                if r is not None:
                    r_det.append(r)
                n = needed_t_rec(own, intr)
                if n is not None:
                    t_need.append(n)
            rows.append(dict(vo=vo, vi=vi, n=n_conf, miss=n_miss, r_det=r_det, t_need=t_need))
            if verbose:
                print(f"  {vo:.0f}/{vi:.0f} kt: {n_conf} conflicting starts, {n_miss} missed", flush=True)
    return dict(rows=rows, t_rec=t_rec, starts=np.asarray(starts_nm))


def _dz_at_closest(rec):
    k = int(np.argmin(rec.horizontal_ranges))
    return abs(rec.own_states[k].z - rec.intr_states[k].z)


# ---------------------------------------------------------------------------
# Text report (numbers quoted in the paper)
# ---------------------------------------------------------------------------
def report(e1, e2, e3, e4=None):
    n = len(e1["spacings"])
    c = e1["counts"]
    lines = ["[Exp 1] Alert burden against correctly flown in-trail traffic",
             f"  {n} deterministic geometries, spacing {e1['spacings'][0]:.2f}-"
             f"{e1['spacings'][-1]:.2f} NM; {len(e1['hmd_values'])} values of HMD*; "
             f"{3 * n * len(e1['hmd_values'])} runs (3 time/vertical variants)"]
    for name, v in e1["standards"].items():
        largest = f"; pairs spaced up to {max(v):.2f} NM" if v else ""
        lines.append(f"  {name}: {len(v)}/{n}{largest}")
    def diffs(k):
        d = {h: len(c[k][h]) - len(c["tau0_z450"][h]) for h in c["tau0_z450"]}
        d = {h: v for h, v in d.items() if v}
        if not d:
            return "no count"
        return (f"counts at HMD* {int(min(d))}-{int(max(d))} ft "
                f"(by {min(d.values()):+d} to {max(d.values()):+d})")
    lines.append(f"  relative to tau*_mod = 0 s, ZTHR = 450 ft: tau*_mod = 35 s changes {diffs('tau35_z450')}; "
                 f"ZTHR = 250 ft changes {diffs('tau0_z250')}")
    edge = max(e1["standards"]["DO-365 en route"], default=None)
    later = [s for s in e1["spacings"] if edge is not None and s > edge + 1e-9]
    for sp in ([edge] if edge is not None else []) + later[:1]:
        k = min(e1["closest"], key=lambda s: abs(s - sp))
        d, t, ho, hi, dz = e1["closest"][k]
        lines.append(f"  spacing {k:.2f} NM: closest {d:.0f} ft (vertical {dz:.0f} ft) at t={t:.1f} s "
                     f"(ownship heading {ho:.0f}, leader {hi:.0f})")

    u, r8, ok = e2["unmit"], e2["rec_assumed"], e2["rec_ok"]
    own, intr = e2["own"], e2["intr"]
    r0 = np.hypot(own[0].x - intr[0].x, own[0].y - intr[0].y) / NM_TO_FT
    lines += ["[Exp 2] Base-to-final conflict, DO-365 terminal well clear",
              f"  ownship {own[0].speed_kt:.0f} kt, intruder {intr[0].speed_kt:.0f} kt from "
              f"{CONFLICT['intr_start_nm']} NM; initial range {r0:.2f} NM; detection range 3.0 NM",
              f"  unmitigated: t_wc = {u.t_wc_actual:.1f} s, closest {u.min_horizontal_separation:.0f} ft "
              f"horizontally with {_dz_at_closest(u):.0f} ft vertically (NMAC needs < 500 ft and < 100 ft)",
              f"  assumed T_rec = {T_REC_ASSUMED:.1f} s: t_mon = {r8.t_mon:.1f} s, t_wc projected = "
              f"{r8.t_wc_projected:.1f} s, margin = {r8.lead_time_margin_s:.2f} s (= trigger buffer)",
              f"     breakout flown: LoWC = {r8.experienced_lowc} from t = {r8.t_wc_actual:.1f} s for "
              f"{sum(r8.lowc_flags) * DT:.1f} s; closest {r8.min_horizontal_separation:.0f} ft",
              f"  T_rec the breakout needs in this geometry: {e2['t_need']} s",
              f"  T_rec = {T_REC_SUFFICIENT:.0f} s: t_mon = {ok.t_mon:.1f} s, LoWC = {ok.experienced_lowc}, "
              f"closest {ok.min_horizontal_separation:.0f} ft",
              "  needed T_rec for neighbouring starts: " +
              ", ".join(f"{k} NM -> {v} s" for k, v in e2["neighbours"].items())]

    lines += ["[Exp 3] Coupling between detection range (SR-1) and lead time (SR-6)",
              f"  {len(e3['ranges'])} detection ranges x {len(e3['t_recs'])} values of T_rec = "
              f"{len(e3['ranges']) * len(e3['t_recs'])} runs"]
    for t_rec in e3["t_recs"]:
        rs = e3["r_star"][t_rec]
        lines.append(f"  T_rec={t_rec:.0f} s: R*_det = "
                     f"{f'{rs:.2f} NM' if rs is not None else 'not reached'}; "
                     f"plateau = {np.nanmax(e3['margins'][t_rec]):.2f} s")

    if e4 is not None:
        lines.append(f"[Exp 4] Speed sensitivity, worst case over starts "
                     f"{e4['starts'][0]:.1f}-{e4['starts'][-1]:.1f} NM, T_rec = {e4['t_rec']:.0f} s")
        all_r, all_t, n_all, n_miss = [], [], 0, 0
        for row in e4["rows"]:
            n_all += row["n"]
            n_miss += row["miss"]
            if not row["r_det"]:
                lines.append(f"  {row['vo']:.0f}/{row['vi']:.0f} kt: {row['n']} conflicts, "
                             f"{row['miss']} missed by the projection")
                continue
            all_r += row["r_det"]
            all_t += row["t_need"]
            lines.append(f"  {row['vo']:.0f}/{row['vi']:.0f} kt: {row['n']} conflicts "
                         f"({row['miss']} missed by the projection); "
                         f"R*_det {min(row['r_det']):.2f}-{max(row['r_det']):.2f} NM; "
                         f"needed T_rec {min(row['t_need']):.1f}-{max(row['t_need']):.1f} s")
        lines.append(f"  overall: {n_all} conflicting geometries, {n_miss} not projected before t_wc; "
                     f"over the rest, R*_det {min(all_r):.2f}-{max(all_r):.2f} NM; "
                     f"needed T_rec {min(all_t):.1f}-{max(all_t):.1f} s")
    return lines


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_fig4(e1, out_dir):
    n = len(e1["spacings"])
    h = e1["hmd_values"]
    rate = [100.0 * len(e1["counts"]["tau0_z450"][float(x)]) / n for x in h]

    fig, ax = plt.subplots(figsize=(fs.COL_W, 1.55), layout="constrained")
    ax.plot(h, rate, color=fs.DARK, marker="o", ms=2.6, mfc=fs.DARK, zorder=3)
    for x, label, color in ((1500, "DO-365\nterminal", fs.TERM),
                            (2000, "ASTM\nF3442", fs.MID),
                            (4000, "DO-365\nen route", fs.ALERT)):
        y = 100.0 * len(e1["counts"]["tau0_z450"][float(x)]) / n
        ax.axvline(x, color=color, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.plot(x, y, "o", ms=5.5, mfc="white", mec=color, mew=1.0, zorder=4)
        ax.text(x + (40 if x < 4000 else -40), 96, label, ha="left" if x < 4000 else "right",
                va="top", fontsize=7, color=color)
    ax.set_xlabel(r"Horizontal threshold $HMD^*$ (ft)")
    ax.set_ylabel("Violating geometries")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(1000, 4500)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y")
    fs.save(fig, out_dir, "fig_nuisance_alert_rates")


def _lowc_overlay(ax, rec, color):
    t = np.array(rec.timestamps)
    r = np.array(rec.horizontal_ranges)
    f = np.array(rec.lowc_flags)
    r_masked = np.where(f, r, np.nan)
    ax.plot(t, r_masked, color=color, lw=3.2, alpha=0.35, solid_capstyle="butt", zorder=2)


def plot_fig5(e2, e3, out_dir):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(fs.PAGE_W, 2.05), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    u, r8, ok = e2["unmit"], e2["rec_assumed"], e2["rec_ok"]
    t_end = 45.0

    # (a) plan view
    def path(rec, attr):
        k = int(t_end / DT)
        st = getattr(rec, attr)[:k]
        return [s.x for s in st], [s.y for s in st]

    a1.plot([0, 5000], [0, 0], color=fs.LIGHT, lw=4, solid_capstyle="butt", zorder=0)
    a1.text(2500, -300, "runway 09", ha="center", va="top", fontsize=7, color=fs.DARK)
    a1.plot(*path(u, "own_states"), color=fs.MID, ls=(0, (1.5, 1.5)), label="ownship")
    a1.plot(*path(r8, "own_states"), color=fs.BLUES[1], lw=1.1, label=r"$F_8$, $T_{rec}=6$ s")
    a1.plot(*path(ok, "own_states"), color=fs.OWN, lw=1.3, label=r"$F_8$, $T_{rec}=8$ s")
    a1.plot(*path(u, "intr_states"), color=fs.CREW, ls=(0, (4, 2)), label="intruder")
    a1.set_xlabel("Along runway (ft)")
    a1.set_ylabel("Cross-runway (ft)")
    a1.set_aspect("equal", adjustable="datalim")
    a1.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.04), handlelength=1.8)
    a1.text(0.0, 1.02, "(a)", transform=a1.transAxes, fontsize=8, va="bottom")

    # (b) horizontal separation; shaded where well clear is lost (horizontal AND vertical)
    for rec, color, ls, lw, lab in ((u, fs.MID, (0, (1.5, 1.5)), 1.0, "unmitigated"),
                                    (r8, fs.BLUES[1], "-", 1.1, r"$T_{rec}=6$ s"),
                                    (ok, fs.OWN, "-", 1.3, r"$T_{rec}=8$ s")):
        a2.plot(rec.timestamps, rec.horizontal_ranges, color=color, ls=ls, lw=lw, label=lab, zorder=3)
        _lowc_overlay(a2, rec, fs.ALERT)
    a2.axhline(TERMINAL.hmd_star, color=fs.ALERT, lw=0.7, ls=(0, (4, 2)))
    a2.text(1.0, TERMINAL.hmd_star + 150, r"$HMD^*=1500$ ft", ha="left", va="bottom",
            fontsize=7, color=fs.ALERT)
    for tv, lab in ((r8.t_mon, r"$t_{mon}$"), (u.t_wc_actual, r"$t_{wc}$")):
        a2.axvline(tv, color=fs.MID, lw=0.6)
        a2.text(tv, 9000, lab, ha="center", va="bottom", fontsize=7)
    y = 5200
    a2.annotate("", xy=(r8.t_mon, y), xytext=(u.t_wc_actual, y),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=fs.INK, shrinkA=0, shrinkB=0))
    a2.text((r8.t_mon + u.t_wc_actual) / 2, y + 150, rf"{T_REC_ASSUMED:.0f} s + 1 s", ha="center",
            va="bottom", fontsize=7)
    a2.set_xlabel("Time (s)")
    a2.set_ylabel("Horizontal separation (ft)")
    a2.set_xlim(0, t_end)
    a2.set_ylim(0, 9000)
    a2.grid(True, axis="y")
    a2.legend(loc="upper right")
    a2.text(0.0, 1.02, "(b)", transform=a2.transAxes, fontsize=8, va="bottom")

    # (c) lead-time margin versus detection range
    r = e3["ranges"]
    a3.axhspan(-10, 0, color=fs.ALERT_TINT, lw=0, zorder=0)
    a3.axhline(0, color=fs.ALERT, lw=0.7)
    a3.axhline(MonitorConfig().buffer_margin_s, color=fs.MID, lw=0.5, ls=(0, (1, 1.5)))
    a3.text(r[-1] - 0.01, 1.15, "trigger buffer (1 s)", ha="right", va="bottom",
            fontsize=7, color=fs.DARK)
    a3.text(r[-1] - 0.01, -9.4, "SR-6 not met", ha="right", va="bottom", fontsize=7,
            color=fs.ALERT)
    styles = ["-", (0, (4, 1.5)), "-", (0, (4, 1.5))]
    for t_rec, color, ls in zip(e3["t_recs"], fs.BLUES, styles):
        a3.plot(r, e3["margins"][t_rec], color=color, ls=ls, label=f"{t_rec:.0f} s")
        if e3["r_star"][t_rec] is not None:
            a3.plot(e3["r_star"][t_rec], 0, "o", ms=3, color=color, zorder=5)
    a3.set_xlabel(r"Detection range $R_{det}$ (NM)")
    a3.set_ylabel(r"Lead-time margin $\Delta t_{margin}$ (s)")
    a3.set_xlim(r[0], r[-1])
    a3.set_ylim(-10, 3)
    a3.grid(True, axis="y")
    a3.legend(title=r"$T_{rec}$", title_fontsize=7, loc="lower right",
              bbox_to_anchor=(1.0, 0.14), ncol=2, columnspacing=1.0, handlelength=2.2)
    a3.text(0.0, 1.02, "(c)", transform=a3.transAxes, fontsize=8, va="bottom")
    fs.save(fig, out_dir, "fig_lead_time_recovery")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"),
                    help="directory for the figures (default: figures/)")
    ap.add_argument("--results", default=os.path.join(HERE, "results"),
                    help="directory for teras_results.txt (default: results/)")
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="skip Experiment 4 (speed sensitivity, the slowest part)")
    args = ap.parse_args()

    fs.apply()
    print("Running Experiment 1 ...", flush=True)
    e1 = exp1_alert_burden()
    print("Running Experiment 2 ...", flush=True)
    e2 = exp2_lead_time()
    print("Running Experiment 3 ...", flush=True)
    e3 = exp3_coupling()
    e4 = None
    if not args.skip_sensitivity:
        print("Running Experiment 4 (several minutes) ...", flush=True)
        e4 = exp4_speed_sensitivity()

    lines = report(e1, e2, e3, e4)
    print("\n".join(lines))
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, "teras_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    plot_fig4(e1, args.out_dir)
    plot_fig5(e2, e3, args.out_dir)
    print(f"\nFigures written to {args.out_dir}")


if __name__ == "__main__":
    main()
