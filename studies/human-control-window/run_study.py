"""
Human control window in terminal encounters (Monte Carlo).

    python studies/human-control-window/run_study.py                 # full study (~30 min on 20 cores)
    python studies/human-control-window/run_study.py --n 50          # quick check, prints only
    python studies/human-control-window/run_study.py --figure-only   # redraw the figure from results/

An uncrewed ownship flies the traffic pattern of an uncontrolled airport and
meets one crewed intruder with random geometry. When the run-time monitor
projects a loss of well clear it alerts the remote pilot and proposes a
recovery; the pilot can approve it or not. For each encounter that loses well
clear when nothing is done, and for each alert design, the study measures

    t_alert  onset of the alert that stays on until the loss of well clear
    t_wc     loss of well clear without any action
    t_last   latest command time such that every command from t_alert on
             still keeps well clear (best of the candidate recovery maneuvers)

and the available control window W_avail = t_last - t_alert, which is the
same as (t_wc - T_rec) - t_alert with T_rec = t_wc - t_last, the time the
recovery needs, aircraft response included. Control over the decision is
meaningful when W_avail >= W_req, the time the pilot needs.

Alert designs (same encounters)
  H45  alert horizon 45 s, both aircraft projected along their current velocity
  H30  alert horizon 30 s, same projection
  I45  alert horizon 45 s, ownship projected along its own planned trajectory
       (its intent), intruder along its current velocity

Outputs (in this folder)
  results/control_window_results.txt   every number quoted from this study
  results/encounters.csv               one row per accepted encounter
  figures/fig_control_window.pdf       (a) one encounter, (b) share with W_avail >= W_req,
                                       (c) share by closing speed
  figures/fig_control_map.pdf          where control is real or nominal (ownship at the loss
                                       of well clear), straight-line vs flight-plan alert

Assumptions (illustrative, not properties of any aircraft or sensor)
  * Well clear: DO-365 terminal-area volume (1500 ft, 450 ft, no time threshold).
  * Alert horizon: 45 s is the minimum average time of the DO-365A terminal
    warning alert and 30 s its late threshold (Jack and Kc, DASC 2020, Table 2).
    The alert must be on, uninterrupted, until the loss of well clear.
  * Detection range 3 NM, perfect tracking inside it.
  * Recovery: a climbing turn at 25 deg bank, 700 ft/min, chosen among the
    breakout to heading 060 and turns of 45 deg left or right; the monitor is
    assumed to propose the one that stays effective longest.
  * W_req: the reference value 10.5 s is the human share of the FAA AC 90-48E
    identification and reaction chart (12.5 s minus 2 s of aircraft lag,
    already inside T_rec). Command-link latency adds to it and is covered by
    the 5-25 s sensitivity range.
  * Both aircraft start on straight segments. When an encounter starts inside
    well clear or with an alert already on, the same encounter is started up
    to 90 s earlier so that every alert onset is observed. Encounters where
    either aircraft is below 100 ft at the loss of well clear are rejected.
"""

import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repository root
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from teras.monitors import MonitorConfig  # noqa: E402
from teras.simulator import EncounterSimulator  # noqa: E402
from teras.traffic_pattern import (  # noqa: E402
    AircraftState, AirfieldConfig, TrafficPatternGenerator, KT_TO_FT_S, FT_S_TO_KT, NM_TO_FT,
)
from teras.wellclear import WellClearThresholds  # noqa: E402

DT = 0.1
DURATION_S = 150.0
SEED = 2026
N_TARGET = 1000
DESIGNS = {                                   # alert designs; the first is the main case
    "H45": dict(lookahead_s=45.0, own_intent=False),
    "H30": dict(lookahead_s=30.0, own_intent=False),
    "I45": dict(lookahead_s=45.0, own_intent=True),
}
DESIGN_LABELS = {"H45": "alert 45 s ahead", "H30": "alert 30 s ahead",
                 "I45": "45 s, own flight plan"}
MAIN = "H45"
MANEUVERS = ("breakout", "left", "right")
W_REQ_REF = 10.5                 # human share of the AC 90-48E chart (s)
W_REQ_GRID = np.round(np.arange(5.0, 25.01, 0.5), 2)
CLOSING_BINS_KT = (0.0, 50.0, 100.0, 150.0, np.inf)
MIN_HEIGHT_FT = 100.0
PRE_ROLL_S = (0.0, 30.0, 60.0, 90.0)   # earlier starts tried when an onset is not observed
COARSE_STEP_S = 1.0
TAIL_S = 60.0                    # how long after t_wc a commanded recovery is checked
TERMINAL = WellClearThresholds.do365_terminal()
AIRFIELD = AirfieldConfig()


# ---------------------------------------------------------------------------
# Random geometry
# ---------------------------------------------------------------------------
def sample_params(rng):
    """Draw one encounter: uncrewed ownship in the pattern, one crewed intruder."""
    p = {"own_leg": "downwind", "own_kt": float(rng.uniform(70, 90)),
         "own_off": float(rng.uniform(-2000, 8000))}
    kind = str(rng.choice(["pattern", "straight-in", "crossing"]))
    p["kind"] = kind
    if kind == "pattern":                       # other pattern traffic
        p["intr_kt"] = float(rng.uniform(70, 105))
        p["intr_leg"] = str(rng.choice(["downwind", "final"]))
        p["intr_off"] = float(rng.uniform(-6000, 8000) if p["intr_leg"] == "downwind"
                              else rng.uniform(-9000, 6000))
    elif kind == "straight-in":                 # straight-in approach, 1.5-4.0 NM out
        p["intr_kt"] = float(rng.uniform(70, 125))
        p["intr_leg"] = "final"
        p["intr_off"] = float(AIRFIELD.final_approach_dist - rng.uniform(1.5, 4.0) * NM_TO_FT)
    else:                                       # level transient crossing the pattern area
        p["intr_kt"] = float(rng.uniform(70, 125))
        p["hdg"] = float(rng.uniform(0, 360))
        p["alt"] = float(rng.uniform(500, 1500))
        p["aim_x"] = float(rng.uniform(-9000, 7000))
        p["aim_y"] = float(rng.uniform(-2000, 6500))
        p["aim_t"] = float(rng.uniform(30, 100))
    return p


def _crossing(p, n, pre_s):
    v = p["intr_kt"] * KT_TO_FT_S
    r = np.radians(p["hdg"])
    vx, vy = v * np.sin(r), v * np.cos(r)
    t_aim = p["aim_t"] + pre_s
    x0, y0 = p["aim_x"] - vx * t_aim, p["aim_y"] - vy * t_aim
    return [AircraftState(x=x0 + vx * i * DT, y=y0 + vy * i * DT, z=p["alt"], vx=vx, vy=vy, vz=0.0,
                          speed_kt=p["intr_kt"], heading_deg=p["hdg"], timestamp=i * DT)
            for i in range(n)]


def build(p, pre_s=0.0):
    """Trajectories of one encounter, cut when the ownship touches down.

    Both aircraft start on straight segments (downwind, long final or a
    crossing track), so ``pre_s`` starts the same encounter that many seconds
    earlier by moving each start point back along its segment.
    """
    gen = TrafficPatternGenerator(AIRFIELD)
    dur = DURATION_S + pre_s
    back = lambda kt: kt * KT_TO_FT_S * pre_s  # noqa: E731
    own = gen.simulate_trajectory(p["own_leg"], p["own_off"] - back(p["own_kt"]), p["own_kt"], dur, DT)
    landed = next((i for i, s in enumerate(own) if s.z <= 0.0), len(own))
    own = own[:landed]
    if p["kind"] == "crossing":
        intr = _crossing(p, len(own), pre_s)
    else:
        intr = gen.simulate_trajectory(p["intr_leg"], p["intr_off"] - back(p["intr_kt"]),
                                       p["intr_kt"], dur, DT)
        intr = intr[:len(own)]
    return own, intr


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def first_lowc(own, intr):
    rec = EncounterSimulator(TERMINAL).run(own, intr, dt=DT, enable_recovery=False,
                                           enable_lookahead=False, early_stop_on_lowc=True)
    return rec.t_wc_actual


def screen(p):
    """Time of loss of well clear without any action (None if none), cheap run."""
    return first_lowc(*build(p))


def alert_onset(own, intr, design):
    """Onset of the alert that stays on until the loss of well clear (None if none)."""
    sim = EncounterSimulator(TERMINAL, MonitorConfig(**DESIGNS[design]))
    rec = sim.run(own, intr, dt=DT, enable_recovery=False, early_stop_on_lowc=True)
    on = [v.t_wc_projected is not None for v in rec.monitor_verdicts]
    if not on or not on[-1]:
        return None
    k = len(on) - 1
    while k > 0 and on[k - 1]:
        k -= 1
    return k * DT


class CommandOracle:
    """Does a recovery commanded at a given time keep well clear? (memoized)"""

    def __init__(self, own, intr, t_wc):
        self.own, self.intr, self.t_wc = own, intr, t_wc
        self.k_end = min(len(own), int(round((t_wc + TAIL_S) / DT)))
        self.cache = {}

    def __call__(self, t_cmd, maneuver):
        k0 = int(round(t_cmd / DT))
        key = (k0, maneuver)
        if key not in self.cache:
            sim = EncounterSimulator(TERMINAL, maneuver_type=maneuver)
            rec = sim.run(self.own[k0:self.k_end], self.intr[k0:self.k_end], dt=DT,
                          enable_lookahead=False, early_stop_on_lowc=True, recovery_command_time=0.0)
            self.cache[key] = not rec.experienced_lowc
        return self.cache[key]

    def latest_command(self, t_from, maneuver):
        """Latest command time such that every command in [t_from, t] keeps well clear."""
        if not self(t_from, maneuver):
            return None
        ok, t = t_from, t_from + COARSE_STEP_S
        while t < self.t_wc - 1e-9 and self(t, maneuver):
            ok, t = t, t + COARSE_STEP_S
        bad = min(t, self.t_wc)
        while bad - ok > DT + 1e-9:             # refine to one time step
            mid = round(((ok + bad) / 2) / DT) * DT
            if mid <= ok + 1e-9 or mid >= bad - 1e-9:
                break
            if self(mid, maneuver):
                ok = mid
            else:
                bad = mid
        return ok


def closing_speed_kt(own, intr, t):
    k = int(round(t / DT))
    rel = intr[k].pos[:2] - own[k].pos[:2]
    vrel = intr[k].vel[:2] - own[k].vel[:2]
    return float(-(rel @ vrel) / max(np.linalg.norm(rel), 1e-6) * FT_S_TO_KT)


def evaluate(args):
    """Full measurement of one encounter that loses well clear."""
    idx, p, _ = args
    out = dict(idx=idx, kind=p["kind"], own_kt=p["own_kt"], intr_kt=p["intr_kt"])
    # If the encounter starts inside well clear, or an alert is already on at
    # t = 0, an onset is not observed: start the same encounter earlier.
    for pre_s in PRE_ROLL_S:
        own, intr = build(p, pre_s)
        t_wc = first_lowc(own, intr)
        if t_wc is None:
            out["status"] = "changed"
            return out
        k = int(round(t_wc / DT))
        if min(own[k].z, intr[k].z) < MIN_HEIGHT_FT:
            out["status"] = "ground"
            return out
        onsets = {d: alert_onset(own, intr, d) for d in DESIGNS}
        if t_wc > 0.0 and all(t is None or t > 0.0 for t in onsets.values()):
            break
    else:
        out["status"] = "inside" if t_wc <= 0.0 else "censored"
        return out
    out.update(status="ok", pre_s=pre_s, t_wc=t_wc, params=p)
    oracle = CommandOracle(own, intr, t_wc)
    for d, t_a in onsets.items():
        best, best_m = None, None
        if t_a is not None:
            for m in MANEUVERS:
                t = oracle.latest_command(t_a, m)
                if t is not None and (best is None or t > best):
                    best, best_m = t, m
        out[f"t_alert_{d}"] = t_a
        out[f"t_last_{d}"] = best
        out[f"maneuver_{d}"] = best_m
        # No alert, or no command that works at the alert: no window at all.
        out[f"W_{d}"] = best - t_a if best is not None else 0.0
        out[f"closing_{d}"] = closing_speed_kt(own, intr, t_a) if t_a is not None else None
    return out


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------
def run(n_target, processes):
    rng = np.random.default_rng(SEED)
    rows, n_sampled, n_conflict = [], 0, 0
    rejected = {"ground": 0, "inside": 0, "censored": 0, "changed": 0}
    with Pool(processes) as pool:
        while len(rows) < n_target:
            batch = [sample_params(rng) for _ in range(max(200, 8 * (n_target - len(rows))))]
            t_wcs = pool.map(screen, batch, chunksize=16)
            todo = []
            for p, t_wc in zip(batch, t_wcs):
                n_sampled += 1
                if t_wc is not None:
                    todo.append((n_sampled - 1, p, t_wc))
            for out in pool.imap(evaluate, todo, chunksize=2):
                n_conflict += 1
                if out["status"] == "ok":
                    rows.append(out)
                    if len(rows) == n_target:
                        break
                else:
                    rejected[out["status"]] += 1
            print(f"  sampled {n_sampled}, conflicts {n_conflict}, accepted {len(rows)}", flush=True)
    return dict(rows=rows, n_sampled=n_sampled, n_conflict=n_conflict, rejected=rejected)


def share(w, w_req):
    return 100.0 * float(np.mean(np.asarray(w) >= w_req - 1e-9))


def wilson(w, w_req, z=1.96):
    """95% Wilson interval (in %) for the share of encounters with W >= w_req."""
    n = len(w)
    p = share(w, w_req) / 100.0
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100.0 * (c - h), 100.0 * (c + h)


def closing_groups(rows, d):
    cl = np.array([r[f"closing_{d}"] if r[f"closing_{d}"] is not None else np.nan for r in rows])
    out = []
    for lo, hi in zip(CLOSING_BINS_KT[:-1], CLOSING_BINS_KT[1:]):
        sel = (cl >= lo) & (cl < hi)
        label = f"{lo:.0f}-{hi:.0f} kt" if np.isfinite(hi) else f">= {lo:.0f} kt"
        out.append((label, sel))
    return out


def report(res):
    rows = res["rows"]
    n = len(rows)
    rj = res["rejected"]
    lines = ["[Human control window] Monte Carlo, DO-365 terminal well clear",
             f"  seed {SEED}; {res['n_sampled']} geometries sampled; {res['n_conflict']} lose well clear "
             f"without action; rejected: {rj['ground']} near the ground, {rj['inside']} inside well clear "
             f"and {rj['censored']} alerted from t = 0 even when started 90 s earlier, {rj['changed']} other; "
             f"accepted N = {n} ({sum(r['pre_s'] > 0 for r in rows)} started earlier to observe the onsets)"]
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("pattern", "straight-in", "crossing")}
    lines.append("  intruders: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    for d in DESIGNS:
        w = np.array([r[f"W_{d}"] for r in rows])
        feas = [r for r in rows if r[f"t_last_{d}"] is not None]
        no_alert = sum(r[f"t_alert_{d}"] is None for r in rows)
        lo, hi = wilson(w, W_REQ_REF)
        lines.append(f"  [{d}] {DESIGN_LABELS[d]}: no alert before the loss of well clear in {no_alert}/{n}; "
                     f"some recovery still works at the alert: {len(feas)}/{n}")
        trec = np.array([r["t_wc"] - r[f"t_last_{d}"] for r in feas])
        lines.append(f"     W_avail median {np.median(w):.1f} s, IQR {np.percentile(w, 25):.1f}-"
                     f"{np.percentile(w, 75):.1f} s; T_rec median {np.median(trec):.1f} s")
        lines.append(f"     W_avail >= {W_REQ_REF} s (reference W_req): {share(w, W_REQ_REF):.1f}% "
                     f"(95% CI {lo:.1f}-{hi:.1f}%)")
        lines.append("     sensitivity: " + ", ".join(f"{x:.0f} s -> {share(w, x):.1f}%"
                                                       for x in (5, 10, 15, 20, 25)))
        lines.append(f"     extra W_req near {W_REQ_REF} s (e.g. link latency): "
                     f"+1 s -> {share(w, W_REQ_REF + 1):.1f}%, +2 s -> {share(w, W_REQ_REF + 2):.1f}%")
        parts = [f"{lab}: {share(w[sel], W_REQ_REF):.1f}% of {int(sel.sum())}"
                 for lab, sel in closing_groups(rows, d) if sel.any()]
        lines.append(f"     by closing speed at the alert (W_req = {W_REQ_REF} s): " + "; ".join(parts))
        by_kind = [f"{k} {share(w[np.array([r['kind'] == k for r in rows])], W_REQ_REF):.1f}%"
                   for k in kinds if kinds[k]]
        lines.append("     by intruder type: " + ", ".join(by_kind))
    a = np.array([r[f"W_{MAIN}"] >= W_REQ_REF - 1e-9 for r in rows])
    for d in DESIGNS:
        if d == MAIN:
            continue
        b = np.array([r[f"W_{d}"] >= W_REQ_REF - 1e-9 for r in rows])
        lines.append(f"  {d} vs {MAIN} at W_req = {W_REQ_REF} s: {int((~a & b).sum())} encounters gain "
                     f"meaningful control, {int((a & ~b).sum())} lose it")
    return lines


def write_csv(rows, path):
    keys = [k for k in rows[0].keys() if k != "params"]
    with open(path, "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()
                         if k != "params"})


def example(rows):
    """A pattern encounter whose main-design window is a little shorter than W_req."""
    pool = [r for r in rows if r["kind"] == "pattern" and r[f"t_last_{MAIN}"] is not None
            and r[f"t_last_I45"] is not None
            and 0.5 * W_REQ_REF <= r[f"W_{MAIN}"] < W_REQ_REF <= r["W_I45"]] or rows
    return min(pool, key=lambda r: r["idx"])


def plot(rows, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import figstyle as fs
    fs.apply()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(fs.PAGE_W, 2.55), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})
    colors = {"H45": fs.OWN, "H30": fs.BLUES[0], "I45": fs.TERM}
    styles = {"H45": "-", "H30": (0, (4, 2)), "I45": "-"}

    # (a) one encounter in plan view: alerts, last useful command, loss of well clear
    r = example(rows)
    own, intr = build(r["params"], r["pre_s"])
    t_w, t_l = r["t_wc"], r[f"t_last_{MAIN}"]
    k_w, k_l = int(round(t_w / DT)), int(round(t_l / DT))
    k_i, k_s = int(round(r["t_alert_I45"] / DT)), int(round(r[f"t_alert_{MAIN}"] / DT))
    k0 = max(0, k_i - int(10 / DT))
    rec = EncounterSimulator(TERMINAL, maneuver_type=r[f"maneuver_{MAIN}"]).run(
        own[k_l:k_w + int(25 / DT)], intr[k_l:k_w + int(25 / DT)], dt=DT, enable_lookahead=False,
        recovery_command_time=0.0)
    xy = lambda st: ([s.x for s in st], [s.y for s in st])  # noqa: E731
    a1.plot([0, AIRFIELD.runway_length], [0, 0], color=fs.LIGHT, lw=4, solid_capstyle="butt", zorder=0)
    a1.text(AIRFIELD.runway_length / 2, 300, "runway", ha="center", va="bottom", fontsize=7, color=fs.DARK)
    a1.plot(*xy(own[k0:k_w + 1]), color=fs.OWN, lw=1.2, label="uncrewed aircraft")
    a1.plot(*xy(rec.own_states), color=fs.OWN, lw=1.0, ls=(0, (3, 1.5)),
            label="recovery at last command")
    a1.plot(*xy(intr[k0:k_w + 1]), color=fs.CREW, lw=1.2, label="crewed aircraft")
    marks = ((k_i, "s", fs.TERM, f"flight-plan alert, {r['W_I45']:.0f} s left"),
             (k_s, "o", fs.INK, f"straight-line alert, {r[f'W_{MAIN}']:.0f} s left"),
             (k_l, "D", fs.INK, "last useful command"),
             (k_w, "X", fs.INK, "loss of separation"))
    for k, mk, ec, lab in marks:
        a1.plot(own[k].x, own[k].y, mk, ms=4.5, mfc="white", mec=ec, mew=0.9, label=lab, zorder=5)
        a1.plot(intr[k].x, intr[k].y, mk, ms=4.5, mfc="white", mec=fs.CREW, mew=0.8, zorder=5)
    a1.set_xlim(-12500, 5500)
    a1.set_ylim(-900, 5300)
    a1.set_aspect("equal", adjustable="box")
    a1.set_xlabel("Along runway (ft)")
    a1.set_ylabel("Across runway (ft)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, handlelength=1.6,
              borderaxespad=0.0, fontsize=6, labelspacing=0.3, columnspacing=1.0)
    a1.text(0.0, 1.02, "(a) one encounter", transform=a1.transAxes, fontsize=8, va="bottom")

    # (b) share of encounters whose available window covers the required one
    for d in DESIGNS:
        w = [x[f"W_{d}"] for x in rows]
        a2.plot(W_REQ_GRID, [share(w, x) for x in W_REQ_GRID], color=colors[d], ls=styles[d],
                lw=1.2 if d != "H30" else 1.0, label=DESIGN_LABELS[d])
    a2.axvline(W_REQ_REF, color=fs.MID, lw=0.6, ls=(0, (1, 1.5)))
    a2.text(W_REQ_REF + 0.3, 50, f"{W_REQ_REF} s", fontsize=7, color=fs.DARK)
    a2.set_xlabel(r"Required window $W_{req}$ (s)")
    a2.set_ylabel(r"Encounters with $W_{avail} \geq W_{req}$")
    a2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    a2.set_xlim(W_REQ_GRID[0], W_REQ_GRID[-1])
    a2.set_xticks(np.arange(5, 26, 5))
    a2.set_ylim(0, 100)
    a2.grid(True, axis="y")
    a2.legend(loc="lower left", fontsize=6.5, borderaxespad=0.2)
    a2.text(0.0, 1.02, f"(b) N = {len(rows)} encounters", transform=a2.transAxes, fontsize=8,
            va="bottom")

    # (c) share by closing speed at the alert, with 95% intervals
    groups = [lab for lab, _ in closing_groups(rows, MAIN)]
    x = np.arange(len(groups))
    for j, d in enumerate((MAIN, "I45")):
        w = np.array([r_[f"W_{d}"] for r_ in rows])
        vals, err = [], [[], []]
        for _, sel in closing_groups(rows, d):
            v = share(w[sel], W_REQ_REF)
            lo, hi = wilson(w[sel], W_REQ_REF)
            vals.append(v)
            err[0].append(v - lo)
            err[1].append(hi - v)
        a3.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=colors[d], alpha=0.85 if j == 0 else 0.7,
               yerr=err, error_kw=dict(lw=0.6, capsize=1.5, ecolor=fs.DARK),
               label=DESIGN_LABELS[d])
    a3.set_xticks(x)
    a3.set_xticklabels(["<50", "50-100", "100-150", r"$\geq$150"][:len(groups)], fontsize=6.5)
    a3.set_xlabel("Approach speed at alert (kt)")
    a3.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    a3.set_ylim(0, 100)
    a3.grid(True, axis="y")
    a3.text(0.0, 1.02, f"(c) $W_{{req}}$ = {W_REQ_REF} s", transform=a3.transAxes, fontsize=8, va="bottom")
    fs.save(fig, out_dir, "fig_control_window")
    return r


def plot_map(rows, out_dir):
    """Where control is real or nominal: ownship position at the loss of well clear."""
    import matplotlib.pyplot as plt
    import figstyle as fs
    fs.apply()
    pos = []
    for r in rows:
        own, _ = build(r["params"], r["pre_s"])
        s = own[int(round(r["t_wc"] / DT))]
        pos.append((s.x, s.y))
    pos = np.array(pos)
    # nominal pattern for reference: downwind, base and final at 80 kt
    ref = TrafficPatternGenerator(AIRFIELD).simulate_trajectory("downwind", -2000.0, 80.0, 150.0, DT)
    fig, axes = plt.subplots(1, 2, figsize=(fs.PAGE_W, 1.9), layout="constrained", sharey=True)
    for ax, d, tag in zip(axes, (MAIN, "I45"), ("(a)", "(b)")):
        ok = np.array([r[f"W_{d}"] >= W_REQ_REF - 1e-9 for r in rows])
        ax.plot([s.x for s in ref], [s.y for s in ref], color=fs.LIGHT, lw=3, zorder=0,
                solid_capstyle="round")
        ax.plot([0, AIRFIELD.runway_length], [0, 0], color=fs.MID, lw=4, solid_capstyle="butt", zorder=1)
        ax.scatter(pos[ok, 0], pos[ok, 1], s=4, color=fs.TERM, alpha=0.55, lw=0, zorder=2,
                   label=f"control is real ({100 * ok.mean():.0f}%)")
        ax.scatter(pos[~ok, 0], pos[~ok, 1], s=7, marker="x", color=fs.ALERT, lw=0.6, zorder=3,
                   label=f"control in name only ({100 * (~ok).mean():.0f}%)")
        ax.set_xlim(-8000, 9000)
        ax.set_ylim(-700, 5300)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Along runway (ft)")
        ax.legend(loc="center", bbox_to_anchor=(0.62, 0.45), fontsize=6.5, markerscale=2.0,
                  handletextpad=0.3)
        ax.text(0.0, 1.02, f"{tag} {DESIGN_LABELS[d]}", transform=ax.transAxes, fontsize=8, va="bottom")
    axes[0].set_ylabel("Across runway (ft)")
    fs.save(fig, out_dir, "fig_control_map")


def load_rows(path):
    """Accepted encounters from the CSV, with their sampled parameters replayed from the seed."""
    def parse(v):
        if v == "":
            return None
        try:
            return float(v)
        except ValueError:
            return v
    with open(path, encoding="utf8") as f:
        rows = [{k: parse(v) for k, v in r.items()} for r in csv.DictReader(f)]
    rng = np.random.default_rng(SEED)
    params = [sample_params(rng) for _ in range(int(max(r["idx"] for r in rows)) + 1)]
    for r in rows:
        r["idx"] = int(r["idx"])
        r["params"] = params[r["idx"]]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_TARGET, help="accepted encounters")
    ap.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from results/encounters.csv")
    args = ap.parse_args()
    if args.figure_only:
        rows = load_rows(os.path.join(HERE, "results", "encounters.csv"))
        plot(rows, os.path.join(HERE, "figures"))
        plot_map(rows, os.path.join(HERE, "figures"))
        return
    res = run(args.n, args.processes)
    lines = report(res)
    if args.n == N_TARGET:
        ex = plot(res["rows"], os.path.join(HERE, "figures"))
        plot_map(res["rows"], os.path.join(HERE, "figures"))
        lines.append(f"  figure example: encounter {ex['idx']} ({ex['kind']}), loss of well clear at "
                     f"{ex['t_wc']:.1f} s; alert at {ex[f't_alert_{MAIN}']:.1f} s (straight line) and "
                     f"{ex['t_alert_I45']:.1f} s (own flight plan); last useful command at "
                     f"{ex[f't_last_{MAIN}']:.1f} s and {ex['t_last_I45']:.1f} s")
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        with open(os.path.join(HERE, "results", "control_window_results.txt"), "w", encoding="utf8") as f:
            f.write("\n".join(lines) + "\n")
        write_csv(res["rows"], os.path.join(HERE, "results", "encounters.csv"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
