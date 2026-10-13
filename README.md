# TERAS — Terminal Encounter and Runtime Assurance Simulator

TERAS is a lightweight Python simulator of close-proximity encounters between
an uncrewed aircraft (the *ownship*) and crewed traffic in a non-towered
airport traffic pattern. It is meant for exploring run-time assurance ideas
quickly: how a detect-and-avoid well-clear definition behaves in the pattern,
how much lead time a run-time monitor leaves for the recovery, and how that
depends on detection range.

What it provides:

- **Traffic-pattern trajectories**: kinematic 3-D trajectories through the
  downwind, base and final legs of a left-hand pattern, with turns flown at a
  constant bank angle, and straight-in approaches.
- **Well-clear evaluation**: the RTCA DO-365 formulation (range, horizontal miss
  distance, modified tau, vertical separation), with configurable thresholds.
- **Run-time safety monitor**: projects the encounter ahead and triggers a
  recovery while there is still time to fly it (lead-time budget `t_rec`).
- **Envelope monitor**: triggers the recovery when the detector confidence or
  navigation integrity leaves its validated envelope (fault injection).
- **Contingency manager**: flies a climbing breakout maneuver once triggered.
- **Interactive dashboard** (Streamlit) to explore all of the above.

## Getting started

Requires Python 3.10 or later.

```bash
git clone https://github.com/jpadantas/teras.git
cd teras
```

| | Windows | Linux / macOS |
|---|---|---|
| 1. Create `.venv` and install `requirements.txt` | `setup.bat` | `sh setup.sh` |
| 2. Open the dashboard at http://localhost:8501 | `run.bat` | `sh run.sh` |

The dashboard only accepts connections from the local machine. To install
into an environment of your own instead, use pip directly:

```bash
python -m pip install -r requirements.txt   # simulator, dashboard and tests
python -m pip install -e .                  # or: the simulator package only
```

## Quick start

```python
from teras import EncounterSimulator, MonitorConfig, WellClearThresholds, create_base_final_conflict

# Ownship turning from base to final; unannounced intruder on a straight-in approach
own, intruder = create_base_final_conflict(duration_s=70.0, dt=0.1)

sim = EncounterSimulator(
    thresholds=WellClearThresholds.do365_terminal(),   # 1500 ft, 450 ft, no time threshold
    monitor_config=MonitorConfig(t_rec=8.0, sensor_range_ft=3 * 6076.12),
)
rec = sim.run(own, intruder, dt=0.1, enable_recovery=True)

print(rec.t_det, rec.t_mon, rec.t_wc_projected)   # detection, monitor trigger, projected loss of well clear (s)
print(rec.experienced_lowc, rec.min_horizontal_separation)
```

`examples/quickstart.py` runs the same encounter without recovery and with two
assumed recovery times.
`SimulationRecord` also holds the full time history (states, ranges, modified
tau, monitor verdicts) for plotting.

## Interactive dashboard

`run.bat` / `sh run.sh`, or from any environment with the dependencies:

```bash
streamlit run apps/dashboard.py     # or: python apps/dashboard.py
```

## Main concepts

| Object | Role |
|---|---|
| `TrafficPatternGenerator`, `AirfieldConfig` | pattern geometry (runway 09, 5000 ft; downwind 0.75 NM abeam, base 1 NM out) and trajectories |
| `create_*` in `teras.scenarios` | ready-made encounters: in-trail pattern traffic, base-to-final conflict, downwind vs. final |
| `WellClearThresholds`, `evaluate_well_clear` | well-clear volume and its evaluation; presets `do365_phase1()`, `do365_terminal()` and `astm_f3442()` |
| `MonitorConfig` | `t_rec`, trigger buffer, projection horizon, detection range, envelope limits |
| `RuntimeSafetyMonitor` | lead-time-budget monitor (see below) |
| `EnvelopeMonitor` | detector-confidence and navigation-integrity monitor |
| `ContingencyManager` | recovery maneuver |
| `EncounterSimulator`, `SimulationRecord` | the simulation loop and its output |

## Model and assumptions

- **Deterministic and kinematic**: point-mass trajectories at constant speed
  with a fixed time step; no wind, no sensor noise, no pilot or ATC model.
- **Pattern**: turns at 25 deg bank, started one turn radius before the next leg
  so the aircraft rolls out on it; 600 ft/min descent from abeam the threshold,
  then a straight path from the start of final to the threshold.
- **Detection** is range-limited with perfect tracking inside the range.
- **Run-time monitor**: both aircraft are projected along their *current*
  velocity (default 30 s horizon, 0.5 s steps). The monitor triggers when the
  projected time to loss of well clear falls to `t_rec + buffer_margin_s`, so
  it acts as late as the budget allows. With early detection the reported
  margin `(t_wc - t_mon) - t_rec` therefore equals `buffer_margin_s` by
  construction. Projection along the current velocity is inexact during turns,
  and a conflict that only arises after a later turn may be seen too late.
- **Recovery**: requested at one step and flown from the next; a climbing turn
  at the pattern bank angle (25 deg) to a fixed heading (060), whatever the
  intruder's position.
- **Defaults are illustrative**: speeds, `t_rec`, the trigger buffer, bank
  angle and recovery maneuver are assumptions, not properties of any aircraft
  or sensor. The default well-clear volume is the DO-365 terminal-area one.

## Repository layout

```text
teras/        simulator package
examples/     short usage examples
apps/         Streamlit dashboard
tests/        unit and behavioral tests
studies/      self-contained studies built on TERAS (scripts, figures, results)
```

## Studies

- [`studies/terminal-airspace-safety-requirements`](studies/terminal-airspace-safety-requirements):
  alert burden of the en-route well-clear threshold in the pattern, the
  monitor's lead-time budget, and the minimum detection range implied by a
  recovery time. Reproduces the simulation results of the accompanying preprint.

## Tests

```bash
python -m pytest
```

## License

MIT, see `LICENSE`. To cite the software, see `CITATION.cff`.
