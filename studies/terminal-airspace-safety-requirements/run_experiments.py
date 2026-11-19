"""
Reproduce the simulation study of the paper (Section VII, Figs. 4 and 5).

    python studies/terminal-airspace-safety-requirements/run_experiments.py
    python studies/terminal-airspace-safety-requirements/run_experiments.py --skip-sensitivity

Outputs
  figures/fig_nuisance_alert_rates.pdf   Fig. 4  (Experiment 1)
  figures/fig_lead_time_recovery.pdf     Fig. 5  (Experiments 2 and 3, panels a-b)
  results/teras_results.txt              every number quoted in Section VII (incl. Experiment 4)
  results/exp4_geometries.json           per-geometry results of Experiment 4

Well-clear volumes (standards, not assumptions)
  DO-365 en route  : HMD* 4000 ft, ZTHR 450 ft, tau*_mod 35 s
  DO-365 terminal  : HMD* 1500 ft, ZTHR 450 ft, no time threshold   <- used in Exps. 2-4
  ASTM F3442 (sUAS): HMD* 2000 ft, ZTHR 250 ft, no time threshold

Notes on interpreting the results
  * All runs are deterministic; counts are counts of geometries, not samples
    from a distribution of encounters.
  * Loss of well clear is counted, not DO-365 alerts (alerting is not modelled).
  * Experiment 1: spacing is the initial along-track spacing on downwind. Both
    aircraft fly at 90 kt, so the spacing holds; each run ends when the leader
    touches down. With the ownship 5 kt faster (reported separately) it closes
    on the leader on short final, which is a real conflict, not a nuisance.
  * Experiments 2-4: each run ends when either aircraft touches down; an
    aircraft on the runway is no longer part of the airborne encounter.
  * F6 triggers when the projected time to loss of well clear falls to
    T_rec + MonitorConfig.buffer_margin_s (1.0 s). With early detection the
    reported margin therefore equals that buffer by construction.
  * The recovery starts on the step after the trigger, with no latency or
    roll-in; the breakout is a climbing (~700 ft/min) 25-deg-bank turn to
    heading 060, whatever the intruder does. The "needed T_rec" is therefore
    the trigger lead time this instantaneous manoeuvre needs; Experiment 4
    also gives it for 45-deg left and right turns.
  * Speeds, T_rec and the recovery manoeuvres are assumed, not measured.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

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
T_BUF = MonitorConfig().buffer_margin_s   # SR-6 requires a lead-time margin of at least T_buf
# Experiment 1: ownship and leader at the same speed, so the in-trail spacing is
# kept and every loss of well clear is against correctly separated traffic. The
# overtaking variant (ownship 5 kt faster) is reported separately.
EXP1_SPEEDS_KT = (90.0, 90.0)
EXP1_OVERTAKE_KT = (90.0, 85.0)
RUN_S = 200.0             # trajectories are built this long, then cut at touchdown
RECOVERY_MANEUVERS = ("breakout", "left", "right")


def touchdown_index(traj):
    """Index of the first state on the runway (z = 0), or len(traj) if none."""
    for k, s in enumerate(traj):
        if s.z <= 0.0:
            return k
    return len(traj)


def _cut(own, intr, n):
    return own[:n], intr[:n]


def _conflict(**kw):
    """Exp. 2-4 encounter, ended when either aircraft touches down.

    After touchdown an aircraft is on the runway, not in the airborne
    encounter, so the run stops there (earlier versions let it roll on).
    """
    p = dict(CONFLICT)
    p.update(kw)
    own, intr = create_base_final_conflict(duration_s=RUN_S, dt=DT, **p)
    return _cut(own, intr, min(touchdown_index(own), touchdown_index(intr)))


def _in_trail(spacing_nm, speeds_kt):
    """Exp. 1 pair, ended when the leader touches down (after that it is a runway, not an airborne, question)."""
    own, intr = create_nominal_pattern_encounter(
        spacing_nm=spacing_nm, speed_own_kt=speeds_kt[0], speed_intr_kt=speeds_kt[1],
        duration_s=RUN_S, dt=DT)
    return _cut(own, intr, touchdown_index(intr))


def _run(own, intr, th=TERMINAL, recovery=True, maneuver="breakout", **cfg):
    return EncounterSimulator(thresholds=th, monitor_config=MonitorConfig(**cfg),
                              maneuver_type=maneuver).run(own, intr, dt=DT, enable_recovery=recovery)


def needed_t_rec(own, intr, th=TERMINAL, grid=np.arange(0.5, 25.01, 0.5), maneuver="breakout"):
    """Smallest T_rec on the grid for which the recovery keeps well clear (None if none).

    The recovery starts on the step after the trigger, with no latency and no
    roll-in, so this is the trigger lead time an instantaneous recovery of the
    given type needs, not a measured vehicle response time.
    """
    for t_rec in grid:
        rec = _run(own, intr, th, maneuver=maneuver, t_rec=t_rec)
        if rec.t_mon is not None and not rec.experienced_lowc:
            return float(t_rec)
    return None


def min_detection_range(own, intr, t_rec, th=TERMINAL, lo=0.1, hi=3.0, iters=12, m_min=0.0):
    """Smallest detection range (NM) at which the lead-time margin reaches ``m_min``, by bisection.

    ``m_min = 0`` gives the range below which no lead time is left at all;
    ``m_min = T_BUF`` gives the range from which SR-6 (margin >= T_buf) is met.
    """
    def margin(r):
        return _run(own, intr, th, t_rec=t_rec, sensor_range_ft=r * NM_TO_FT).lead_time_margin_s

    m_hi = margin(hi)
    if m_hi is None or m_hi < m_min - 1e-9:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = margin(mid)
        if m is not None and m >= m_min - 1e-9:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------------------
# Experiment 1: alert burden against correctly flown in-trail traffic
# ---------------------------------------------------------------------------
def exp1_alert_burden(spacings_nm=None, hmd_values_ft=None, speeds_kt=EXP1_SPEEDS_KT):
    """Spacings (of 25) whose correctly flown in-trail pair loses well clear, per HMD*.

    Loss of well clear is counted, not DO-365 alerts: alerting is not modelled,
    and alerts would begin earlier, so the counts are a lower bound on alerts.
    """
    if spacings_nm is None:
        spacings_nm = np.linspace(0.4, 1.4, 25)
    if hmd_values_ft is None:
        hmd_values_ft = np.arange(1000.0, 4500.0 + 1, 125.0)

    trajs = [_in_trail(sp, speeds_kt) for sp in spacings_nm]

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
                              abs(r.own_states[k].z - r.intr_states[k].z),
                              r.intr_states[k].z)
    return dict(spacings=np.asarray(spacings_nm), hmd_values=np.asarray(hmd_values_ft),
                counts=counts, standards=standards, closest=closest, speeds=speeds_kt)


def exp1_overtaking(spacings_nm=None):
    """Exp. 1 with the ownship 5 kt faster: which pairs the standard volumes flag, and where."""
    if spacings_nm is None:
        spacings_nm = np.linspace(0.4, 1.4, 25)
    out = {}
    for name, th in (("DO-365 terminal", TERMINAL), ("ASTM F3442", SUAS), ("DO-365 en route", ENROUTE)):
        flagged = []
        for sp in spacings_nm:
            own, intr = _in_trail(sp, EXP1_OVERTAKE_KT)
            r = EncounterSimulator(thresholds=th).run(own, intr, dt=DT, enable_recovery=False,
                                                      enable_lookahead=False)
            if r.experienced_lowc:
                k = int(r.t_wc_actual / DT)
                flagged.append((float(sp), r.min_horizontal_separation, r.intr_states[k].heading_deg,
                                r.intr_states[k].z))
        out[name] = flagged
    return out


# ---------------------------------------------------------------------------
# Experiment 2: lead-time budget and recovery in a base-to-final conflict
# ---------------------------------------------------------------------------
def exp2_lead_time():
    own, intr = _conflict()
    unmit = _run(own, intr, recovery=False, t_rec=T_REC_ASSUMED)
    rec_assumed = _run(own, intr, t_rec=T_REC_ASSUMED)
    t_need = needed_t_rec(own, intr)
    t_need_turns = {m: needed_t_rec(own, intr, maneuver=m) for m in RECOVERY_MANEUVERS[1:]}
    rec_ok = _run(own, intr, t_rec=T_REC_SUFFICIENT)
    neighbours = {}
    for s0 in (1.6, 1.7, 1.8, 1.9, 2.0):
        o, i = _conflict(intr_start_nm=s0)
        neighbours[s0] = needed_t_rec(o, i)
    return dict(own=own, intr=intr, unmit=unmit, rec_assumed=rec_assumed, rec_ok=rec_ok,
                t_need=t_need, t_need_turns=t_need_turns, neighbours=neighbours)


# ---------------------------------------------------------------------------
# Experiment 3: coupling between detection range (SR-1) and lead time (SR-6)
# ---------------------------------------------------------------------------
def exp3_coupling(ranges_nm=None, t_recs=(6.0, 8.0, 10.0, 12.0)):
    if ranges_nm is None:
        ranges_nm = np.linspace(0.3, 1.2, 91)
    own, intr = _conflict()
    margins, r_star, r_sr6 = {}, {}, {}
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
        r_sr6[t_rec] = min_detection_range(own, intr, t_rec, m_min=T_BUF)
    return dict(ranges=np.asarray(ranges_nm), t_recs=list(t_recs), margins=margins, r_star=r_star,
                r_sr6=r_sr6)


# ---------------------------------------------------------------------------
# Experiment 4: sensitivity to speeds, worst case over intruder start positions
# ---------------------------------------------------------------------------
def _exp4_geometry(args):
    """One Exp. 4 geometry (run in a worker process)."""
    vo, vi, s0, t_rec = args
    own, intr = _conflict(speed_own_kt=vo, speed_intr_kt=vi, intr_start_nm=s0)
    unmit = _run(own, intr, recovery=False, t_rec=t_rec)
    out = dict(vo=vo, vi=vi, s0=s0, conflict=unmit.t_wc_actual is not None)
    if not out["conflict"]:
        return out
    # The monitor projects the current velocity; a conflict created by a later
    # turn may not be projected in time. Count those separately: their "needed
    # T_rec" would measure the projection, not the recovery.
    out["miss"] = unmit.t_mon is None or unmit.t_mon >= unmit.t_wc_actual
    if out["miss"]:
        return out
    out["r_det"] = min_detection_range(own, intr, t_rec)
    out["r_det_sr6"] = min_detection_range(own, intr, t_rec, m_min=T_BUF)
    out["t_need"] = {m: needed_t_rec(own, intr, maneuver=m) for m in RECOVERY_MANEUVERS}
    return out


def _exp4_sr6_range(args):
    vo, vi, s0, t_rec = args
    own, intr = _conflict(speed_own_kt=vo, speed_intr_kt=vi, intr_start_nm=s0)
    return min_detection_range(own, intr, t_rec, m_min=T_BUF)


def exp4_add_sr6_ranges(geoms, t_rec=10.0, workers=None):
    """Add ``r_det_sr6`` to Exp. 4 geometries that lack it (results from an older run)."""
    from multiprocessing import Pool
    todo = [g for g in geoms if g["conflict"] and not g["miss"] and "r_det_sr6" not in g]
    if todo:
        with Pool(processes=workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
            vals = pool.map(_exp4_sr6_range, [(g["vo"], g["vi"], g["s0"], t_rec) for g in todo])
        for g, v in zip(todo, vals):
            g["r_det_sr6"] = v
    return len(todo)


def exp4_speed_sensitivity(own_speeds=(70.0, 80.0, 90.0), intr_speeds=(80.0, 95.0, 110.0, 125.0),
                           starts_nm=np.arange(1.2, 3.01, 0.1), t_rec=10.0, workers=None):
    """Exp. 2 encounter over speeds and intruder starts; every run ends at the first touchdown.

    For each conflicting geometry: the detection range at which the lead-time
    margin reaches zero for ``t_rec``, and the trigger lead time each recovery
    in RECOVERY_MANEUVERS needs (None if no value up to 25 s keeps well clear).
    """
    from multiprocessing import Pool
    jobs = [(float(vo), float(vi), round(float(s0), 1), t_rec)
            for vo in own_speeds for vi in intr_speeds for s0 in starts_nm]
    with Pool(processes=workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        geoms = pool.map(_exp4_geometry, jobs)
    return dict(geoms=geoms, t_rec=t_rec, starts=np.asarray(starts_nm), n_grid=len(jobs))


def _dz_at_closest(rec):
    k = int(np.argmin(rec.horizontal_ranges))
    return abs(rec.own_states[k].z - rec.intr_states[k].z)


# ---------------------------------------------------------------------------
# Text report (numbers quoted in the paper)
# ---------------------------------------------------------------------------
def report(e1, e2, e3, e4=None, e1o=None):
    n = len(e1["spacings"])
    c = e1["counts"]
    lines = ["[Exp 1] Loss of well clear against correctly flown in-trail traffic",
             f"  ownship {e1['speeds'][0]:.0f} kt behind leader {e1['speeds'][1]:.0f} kt; each run ends "
             f"when the leader touches down",
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
    for sp in [e1["spacings"][0]] + ([edge] if edge is not None else []) + later[:1]:
        k = min(e1["closest"], key=lambda s: abs(s - sp))
        d, t, ho, hi, dz, zl = e1["closest"][k]
        lines.append(f"  spacing {k:.2f} NM: closest {d:.0f} ft (vertical {dz:.0f} ft, leader at {zl:.0f} ft) "
                     f"at t={t:.1f} s (ownship heading {ho:.0f}, leader {hi:.0f})")
    if e1o is not None:
        lines.append(f"  overtaking variant (ownship {EXP1_OVERTAKE_KT[0]:.0f} kt, leader "
                     f"{EXP1_OVERTAKE_KT[1]:.0f} kt):")
        for name, fl in e1o.items():
            if not fl:
                lines.append(f"    {name}: 0/{n}")
                continue
            on_final = all(abs(h - 90.0) < 1e-6 for _, _, h, _ in fl)
            lines.append(f"    {name}: {len(fl)}/{n}, pairs spaced up to {max(f[0] for f in fl):.2f} NM; "
                         f"closest {min(f[1] for f in fl):.0f}-{max(f[1] for f in fl):.0f} ft"
                         + (f"; all with the leader on final, below {max(f[3] for f in fl):.0f} ft"
                            if on_final else ""))

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
              f"  T_rec the breakout needs in this geometry: {e2['t_need']} s; 45-deg turns instead: " +
              ", ".join(f"{m} {v} s" for m, v in e2["t_need_turns"].items()),
              f"  T_rec = {T_REC_SUFFICIENT:.0f} s: t_mon = {ok.t_mon:.1f} s, LoWC = {ok.experienced_lowc}, "
              f"closest {ok.min_horizontal_separation:.0f} ft",
              "  needed T_rec for neighbouring starts: " +
              ", ".join(f"{k} NM -> {v} s" for k, v in e2["neighbours"].items())]

    lines += ["[Exp 3] Coupling between detection range (SR-1) and lead time (SR-6)",
              f"  {len(e3['ranges'])} detection ranges x {len(e3['t_recs'])} values of T_rec = "
              f"{len(e3['ranges']) * len(e3['t_recs'])} runs"]
    for t_rec in e3["t_recs"]:
        rs, r6 = e3["r_star"][t_rec], e3["r_sr6"][t_rec]
        lines.append(f"  T_rec={t_rec:.0f} s: margin zero at R = "
                     f"{f'{rs:.2f} NM' if rs is not None else 'not reached'}; "
                     f"SR-6 (margin >= T_buf) met from R = "
                     f"{f'{r6:.2f} NM' if r6 is not None else 'not reached'}; "
                     f"plateau = {np.nanmax(e3['margins'][t_rec]):.2f} s")

    if e4 is not None:
        lines += exp4_report(e4)
    return lines


def _spread(v, fmt):
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return (f"{fmt.format(min(v))}-{fmt.format(max(v))} (median {fmt.format(med)}, "
            f"quartiles {fmt.format(q1)}-{fmt.format(q3)})")


def exp4_report(e4):
    g = e4["geoms"]
    conf = [x for x in g if x["conflict"]]
    proj = [x for x in conf if not x["miss"]]
    lines = [f"[Exp 4] Speed sensitivity: {e4['n_grid']} geometries (ownship 70-90 kt x intruder 80-125 kt x "
             f"starts {e4['starts'][0]:.1f}-{e4['starts'][-1]:.1f} NM in 0.1-NM steps); every run ends at the "
             f"first touchdown; T_rec = {e4['t_rec']:.0f} s for R*_det",
             f"  {len(conf)} lose well clear while both aircraft are airborne; "
             f"{len(conf) - len(proj)} not projected before t_wc (excluded)"]
    pairs = sorted({(x["vo"], x["vi"]) for x in g})
    for vo, vi in pairs:
        cell = [x for x in proj if x["vo"] == vo and x["vi"] == vi]
        n_c = sum(1 for x in conf if x["vo"] == vo and x["vi"] == vi)
        r = [x["r_det"] for x in cell if x["r_det"] is not None]
        t = [x["t_need"]["breakout"] for x in cell if x["t_need"]["breakout"] is not None]
        lines.append(f"  {vo:.0f}/{vi:.0f} kt: {n_c} conflicts"
                     + (f"; R*_det {min(r):.2f}-{max(r):.2f} NM" if r else "")
                     + (f"; breakout needs {min(t):.1f}-{max(t):.1f} s" if t else "")
                     + (f"; {len(cell) - len(t)} with no feasible breakout" if len(cell) > len(t) else ""))
    r6 = [x["r_det_sr6"] for x in proj if x.get("r_det_sr6") is not None]
    if r6:
        lines.append(f"  SR-6 minimum detection range (margin >= T_buf) over {len(r6)} geometries: "
                     f"{_spread(r6, '{:.2f}')} NM")
    r = [x["r_det"] for x in proj if x["r_det"] is not None]
    no_r = [(x["vo"], x["vi"], x["s0"]) for x in proj if x["r_det"] is None]
    lines.append(f"  zero-margin range over {len(r)} geometries: {_spread(r, '{:.2f}')} NM"
                 + (f"; undefined for {len(no_r)} {no_r}, where even detection at 3 NM leaves "
                    f"less than T_rec" if no_r else ""))
    for m in RECOVERY_MANEUVERS:
        t = [x["t_need"][m] for x in proj if x["t_need"][m] is not None]
        bad = [(x["vo"], x["vi"], x["s0"]) for x in proj if x["t_need"][m] is None]
        lines.append(f"  needed T_rec, {m}: {_spread(t, '{:.1f}')} s over {len(t)}; "
                     f"{len(bad)} with no value <= 25 s" + (f" {bad}" if bad else ""))
    best = [min(v for v in x["t_need"].values() if v is not None) for x in proj
            if any(v is not None for v in x["t_need"].values())]
    lines.append(f"  needed T_rec, best of {'/'.join(RECOVERY_MANEUVERS)} per geometry: "
                 f"{_spread(best, '{:.1f}')} s over {len(best)}")
    return lines


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_fig4(e1, out_dir):
    """Largest initial spacing whose correctly flown pair loses well clear, against HMD*.

    Unlike a fraction of the sweep, this does not depend on the spacing interval chosen.
    """
    h = e1["hmd_values"]
    lo = e1["spacings"][0]
    c = e1["counts"]

    def largest(k):
        return np.array([max(c[k][float(x)]) if c[k][float(x)] else np.nan for x in h])

    fig, ax = plt.subplots(figsize=(fs.COL_W, 1.75), layout="constrained")
    ax.axhspan(0.3, lo, color=fs.FILL, lw=0, zorder=0)
    ax.text(4480, 0.35, rf"none flagged (all spacings $\geq${lo:.1f} NM)", ha="right", va="center",
            fontsize=6.5, color=fs.DARK)
    variants = (("tau0_z450", r"$\tau^*_{mod}=0$, $ZTHR=450$ ft", fs.DARK, "-"),
                ("tau35_z450", r"$\tau^*_{mod}=35$ s, $ZTHR=450$ ft", fs.MID, (0, (4, 1.5))),
                ("tau0_z250", r"$\tau^*_{mod}=0$, $ZTHR=250$ ft", fs.MID, (0, (1, 1.2))))
    curves = {}
    for k, lab, color, ls in variants:
        curves[k] = largest(k)
        ax.plot(h, curves[k], color=color, ls=ls, lw=1.0, label=lab, zorder=3)
    # Each standard is marked on the curve of its own time and vertical thresholds
    for x, k, label, color in ((1500, "tau0_z450", "DO-365\nterminal", fs.TERM),
                               (2000, "tau0_z250", "ASTM\nF3442", fs.CREW),
                               (4000, "tau35_z450", "DO-365\nen route", fs.ALERT)):
        y = curves[k][list(h).index(x)]
        y = (lo + 0.3) / 2 if np.isnan(y) else y
        ax.axvline(x, color=color, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.plot(x, y, "o", ms=5, mfc="white", mec=color, mew=1.0, zorder=4)
        ax.text(x + (40 if x < 4000 else -40), 1.36, label, ha="left" if x < 4000 else "right",
                va="top", fontsize=7, color=color)
    ax.set_xlabel(r"Horizontal threshold $HMD^*$ (ft)")
    ax.set_ylabel("Largest flagged\nspacing (NM)")
    ax.set_xlim(1000, 4500)
    ax.set_ylim(0.3, 1.4)
    ax.grid(True, axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6.5, handlelength=2.2)
    fs.save(fig, out_dir, "fig_nuisance_alert_rates")


def _lowc_overlay(ax, rec, color):
    t = np.array(rec.timestamps)
    r = np.array(rec.horizontal_ranges)
    f = np.array(rec.lowc_flags)
    r_masked = np.where(f, r, np.nan)
    ax.plot(t, r_masked, color=color, lw=3.2, alpha=0.35, solid_capstyle="butt", zorder=2)


def plot_fig5(e2, e3, out_dir):
    fig, (a2, a3) = plt.subplots(2, 1, figsize=(fs.COL_W, 3.3), layout="constrained")
    u, r8, ok = e2["unmit"], e2["rec_assumed"], e2["rec_ok"]
    t_end = 45.0

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
    a2.text(0.0, 1.02, "(a)", transform=a2.transAxes, fontsize=8, va="bottom")

    # (c) lead-time margin versus detection range
    r = e3["ranges"]
    a3.axhspan(-10, T_BUF, color=fs.ALERT_TINT, lw=0, zorder=0)
    a3.axhline(T_BUF, color=fs.ALERT, lw=0.7)
    a3.axhline(0, color=fs.MID, lw=0.5, ls=(0, (1, 1.5)))
    a3.text(r[-1] - 0.01, T_BUF + 0.15, r"$T_{buf}$ = 1 s", ha="right", va="bottom",
            fontsize=7, color=fs.DARK)
    a3.text(r[-1] - 0.01, -9.4, r"SR-6 not met ($\Delta t_{margin} < T_{buf}$)", ha="right",
            va="bottom", fontsize=7,
            color=fs.ALERT)
    styles = ["-", (0, (4, 1.5)), "-", (0, (4, 1.5))]
    for t_rec, color, ls in zip(e3["t_recs"], fs.BLUES, styles):
        a3.plot(r, e3["margins"][t_rec], color=color, ls=ls, label=f"{t_rec:.0f} s")
        if e3["r_sr6"][t_rec] is not None:
            a3.plot(e3["r_sr6"][t_rec], T_BUF, "o", ms=3, color=color, zorder=5)
    a3.set_xlabel(r"Detection range $R_{det}$ (NM)")
    a3.set_ylabel(r"Lead-time margin $\Delta t_{margin}$ (s)")
    a3.set_xlim(r[0], r[-1])
    a3.set_ylim(-10, 3)
    a3.grid(True, axis="y")
    a3.legend(title=r"$T_{rec}$", title_fontsize=7, loc="lower right",
              bbox_to_anchor=(1.0, 0.14), ncol=2, columnspacing=1.0, handlelength=2.2)
    a3.text(0.0, 1.02, "(b)", transform=a3.transAxes, fontsize=8, va="bottom")
    fs.save(fig, out_dir, "fig_lead_time_recovery")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"),
                    help="directory for the figures (default: figures/)")
    ap.add_argument("--results", default=os.path.join(HERE, "results"),
                    help="directory for teras_results.txt (default: results/)")
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="skip Experiment 4 (speed sensitivity, the slowest part)")
    ap.add_argument("--exp4-json", default=None,
                    help="take Experiment 4 from this exp4_geometries.json instead of running it")
    args = ap.parse_args()

    fs.apply()
    os.makedirs(args.results, exist_ok=True)
    print("Running Experiment 1 ...", flush=True)
    e1 = exp1_alert_burden()
    e1o = exp1_overtaking()
    print("Running Experiment 2 ...", flush=True)
    e2 = exp2_lead_time()
    print("Running Experiment 3 ...", flush=True)
    e3 = exp3_coupling()
    e4 = None
    if args.exp4_json:
        with open(args.exp4_json, encoding="utf-8") as f:
            geoms = json.load(f)
        if exp4_add_sr6_ranges(geoms):
            with open(args.exp4_json, "w", encoding="utf-8") as f:
                json.dump(geoms, f, indent=1)
        e4 = dict(geoms=geoms, t_rec=10.0, starts=np.arange(1.2, 3.01, 0.1), n_grid=len(geoms))
    elif not args.skip_sensitivity:
        print("Running Experiment 4 (several minutes) ...", flush=True)
        e4 = exp4_speed_sensitivity()
        with open(os.path.join(args.results, "exp4_geometries.json"), "w", encoding="utf-8") as f:
            json.dump(e4["geoms"], f, indent=1)

    lines = report(e1, e2, e3, e4, e1o)
    print("\n".join(lines))
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, "teras_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    plot_fig4(e1, args.out_dir)
    plot_fig5(e2, e3, args.out_dir)
    print(f"\nFigures written to {args.out_dir}")


if __name__ == "__main__":
    main()
