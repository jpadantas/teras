# Study: safety requirements for crewed-uncrewed operations in terminal airspace

This folder reproduces the simulation results of the preprint

> J. P. A. Dantas, D. R. Pleffken, G. Dietzsch, and P. Kukulka de Albuquerque,
> "Safety Requirements and Evidence Needs for Crewed-Uncrewed Operations in
> Shared Terminal Airspace,"
> preprint, 2026.

Figure and section numbers refer to the current version of the manuscript.
In the paper's functional architecture, the run-time safety monitor is $F_6$,
the envelope monitor $F_7$, the contingency manager $F_8$ and traffic detection
$F_1$; SR-1 and SR-6 are the safety requirements on detection and on the
monitor's lead time.

## Reproduce

From the repository root:

```bash
python studies/terminal-airspace-safety-requirements/run_experiments.py   # Figs. 3-4 + results
# add --skip-sensitivity to skip Experiment 4 (the slow part, run in parallel),
# or --exp4-json results/exp4_geometries.json to reuse its committed results
python studies/terminal-airspace-safety-requirements/concept_figures.py   # Fig. 2 (+ SR-6 timing sketch)
python -m pytest studies                                                   # checks the quoted numbers
```

| Paper | Script | Output (in this folder) |
|---|---|---|
| Fig. 2, well-clear volumes vs. traffic pattern | `concept_figures.py` | `figures/wellclear_pattern.pdf` |
| (not in the paper) timing of SR-6 | `concept_figures.py` | `figures/monitor_encounter.pdf` |
| Fig. 3, Experiment 1 | `run_experiments.py` | `figures/fig_nuisance_alert_rates.pdf` |
| Fig. 4, Experiments 2 and 3 | `run_experiments.py` | `figures/fig_lead_time_recovery.pdf` |
| Experiment 4 (text only) | `run_experiments.py` | `results/teras_results.txt`, `results/exp4_geometries.json` |
| Numbers in Section VII | `run_experiments.py` | `results/teras_results.txt` |

`results/teras_results.txt` is committed, so the numbers can be checked
without running anything. `test_results.py` asserts that the code still
reproduces them. The figures were produced with numpy 1.26.4 and matplotlib
3.10.8; the tests also pass with numpy 2.5 and matplotlib 3.11.

## Experiments

Well-clear volumes are taken from the standards: DO-365 en route (4000 ft,
450 ft, 35 s), DO-365 terminal area (1500 ft, 450 ft, no time threshold) and
ASTM F3442 small UAS (2000 ft, 250 ft). Experiments 2-4 use the DO-365
terminal-area volume.

| | Exp. 1: alert burden | Exp. 2: lead time | Exp. 3: coupling | Exp. 4: sensitivity |
|---|---|---|---|---|
| Geometry | Two aircraft in trail at 90 kt from downwind through base onto final, until the leader touches down (overtaking variant: 90 kt behind 85 kt) | Ownship turning base to final at 80 kt; straight-in intruder at 95 kt from 1.8 NM | As Exp. 2 | As Exp. 2 |
| Varied | Spacing 0.4-1.4 NM (25); HMD* 1000-4500 ft (29); three time/vertical variants | No recovery; recovery with T_rec = 6 and 8 s; intruder start 1.6-2.0 NM; breakout vs. 45-deg turns | Detection range 0.3-1.2 NM (91); T_rec = 6, 8, 10, 12 s | Ownship 70-90 kt, intruder 80-125 kt (12 pairs); intruder start 1.2-3.0 NM in 0.1-NM steps (228 geometries); breakout, 45-deg left and right turns |
| Fixed | - | Detection range 3 NM | - | T_rec = 10 s |

## Notes for interpreting the results

- All runs are deterministic; counts are counts of geometries, not samples
  from a distribution of encounters.
- Loss of well clear is counted, not DO-365 alerts: alerting is not modelled,
  and alerts would begin earlier.
- Exp. 1: the spacing is the initial along-track spacing on downwind. At equal
  speeds the spacing holds, so every loss of well clear is against correctly
  separated traffic. The closest approach comes with the ownship on base and
  the leader on final, where the straight-line distance is shorter than the
  spacing. With the ownship 5 kt faster it overtakes the leader on short
  final; those losses of well clear are real conflicts, reported separately.
- Exps. 2-4: every run ends when either aircraft touches down (earlier
  versions let a landed aircraft roll on along the runway, which produced
  spurious "conflicts" on the ground).
- The 1 s lead-time margin is the monitor's trigger buffer, not a measured
  margin (see the repository README).
- The recovery starts on the step after the trigger, with no latency or
  roll-in; the breakout is a climbing (~700 ft/min) 25-deg-bank turn to
  heading 060, whatever the intruder does. The "needed T_rec" is the trigger
  lead time this instantaneous manoeuvre needs, and it changes with the
  manoeuvre (Exp. 4 also gives 45-deg left and right turns).
- Exp. 4: the monitor projects the current velocity, so a conflict created by
  a later turn can go unprojected until too late. Those geometries are counted
  separately and excluded. Geometries for which no T_rec up to 25 s keeps well
  clear are counted and listed, not dropped.
- Speeds, T_rec and the recovery manoeuvres are assumptions for illustration.

`figstyle.py` sets the figure style (two-column print widths, shared palette).
