# Study: human control window in terminal encounters

When a run-time monitor on an uncrewed aircraft alerts its remote pilot and
proposes a recovery, does the pilot have enough time to approve or refuse it?
This study measures, over random encounters in the traffic pattern of an
uncontrolled airport, the time between the alert and the last command that
still keeps well clear, compares it with the time a pilot needs, and shows how
that time depends on the design of the alert.

## Reproduce

From the repository root:

```bash
python studies/human-control-window/run_study.py                 # full study, N = 1000 (~15 min on 20 cores)
python studies/human-control-window/run_study.py --n 50          # quick check, prints only
python studies/human-control-window/run_study.py --figure-only   # redraw the figures from results/
python -m pytest studies/human-control-window                    # checks the committed numbers (~30 s)
```

| Output | Content |
|---|---|
| `results/control_window_results.txt` | every number quoted from this study |
| `results/encounters.csv` | one row per accepted encounter |
| `figures/fig_control_window.pdf` | (a) one encounter under two alert designs; (b) share of encounters with `W_avail >= W_req`; (c) the same at 10.5 s by closing speed, with 95% intervals |
| `figures/fig_control_map.pdf` | where control is real or nominal: ownship position at the loss of well clear, for the straight-line and flight-plan alerts |

`results/` is committed, so the numbers can be checked without running anything.

## Quantities

| Symbol | Meaning |
|---|---|
| `t_alert` | onset of the alert that stays on until the loss of well clear |
| `t_wc` | loss of well clear if nothing is done |
| `t_last` | latest command time such that every command from `t_alert` to `t_last` keeps well clear |
| `T_rec = t_wc - t_last` | time the recovery needs, aircraft response included |
| `W_avail = t_last - t_alert = (t_wc - T_rec) - t_alert` | available control window |
| `W_req` | time the pilot needs; reference 10.5 s, sensitivity 5-25 s |

## Alert designs (same encounters)

| Design | Horizon | Ownship projected along | Intruder projected along |
|---|---|---|---|
| `H45` (main) | 45 s | current velocity | current velocity |
| `H30` | 30 s | current velocity | current velocity |
| `I45` | 45 s | its own planned trajectory (`MonitorConfig.own_intent`) | current velocity |

45 s is the minimum average time of the DO-365A terminal warning alert and
30 s its late threshold (Jack and Kc, DASC 2020, Table 2). The ownship knows
its own plan; the crewed intruder's intent is unknown to it.

## Sampling

- Seed 2026; geometries are drawn until 1000 encounters lose well clear without action.
- Uncrewed ownship: downwind start, 70-90 kt, then base and final.
- Crewed intruder, one of three kinds with equal probability: other pattern
  traffic (downwind or final start, 70-105 kt), a straight-in approach from
  1.5-4.0 NM (70-125 kt), or a level transient crossing the pattern area
  (500-1500 ft, any heading, 70-125 kt).
- Both aircraft start on straight segments. If an encounter starts inside well
  clear or with any alert already on, the same encounter is started up to 90 s
  earlier, so that every alert onset is observed. Encounters still censored,
  and those where either aircraft is below 100 ft at the loss of well clear,
  are rejected and counted in the results.

## Assumptions

- Well clear: DO-365 terminal-area volume (1500 ft, 450 ft, no time threshold).
- Detection range 3 NM, perfect tracking.
- Recovery: climbing turn (25 deg bank, ~700 ft/min), the best of the breakout
  to heading 060 and turns of 45 deg left or right, chosen per alert design.
- `W_req = 10.5 s` is the human share of the FAA AC 90-48E identification and
  reaction chart (12.5 s minus 2 s of aircraft lag, which `T_rec` already
  includes). Command-link latency adds to `W_req`; the results give the effect
  of +1 s and +2 s and the sensitivity range covers more.
- No pilot model: the study measures the time the situation allows, not how
  people use it.
