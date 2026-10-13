"""
Encounter simulation loop.

At each step the ownship state is set first (nominal trajectory, or the
recovery once it is active), then the step runs, in order: traffic detection
(range-limited), the envelope monitor, the run-time safety monitor, and the
measurement of the actual well-clear status. All of them see the ownship and
the intruder at the same time instant. A recovery requested at one step is
flown from the next step on.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from teras.traffic_pattern import AircraftState
from teras.wellclear import WellClearThresholds, evaluate_well_clear
from teras.monitors import (
    MonitorConfig,
    MonitorVerdict,
    RuntimeSafetyMonitor,
    EnvelopeMonitor,
    ContingencyManager,
)


@dataclass
class SimulationRecord:
    """Recorded time-series history of a single encounter."""
    timestamps: List[float] = field(default_factory=list)
    own_states: List[AircraftState] = field(default_factory=list)
    intr_states: List[AircraftState] = field(default_factory=list)
    horizontal_ranges: List[float] = field(default_factory=list)
    tau_mods: List[float] = field(default_factory=list)
    lowc_flags: List[bool] = field(default_factory=list)
    monitor_verdicts: List[MonitorVerdict] = field(default_factory=list)
    recovery_active_flags: List[bool] = field(default_factory=list)  # state flown by the recovery

    # Key timeline milestones
    t_det: Optional[float] = None
    t_mon: Optional[float] = None
    t_wc_projected: Optional[float] = None
    t_wc_actual: Optional[float] = None
    min_horizontal_separation: float = float("inf")
    lead_time_margin_s: Optional[float] = None
    experienced_lowc: bool = False
    recovery_triggered: bool = False
    envelope_triggered: bool = False


def _check_time_step(trajectory: List[AircraftState], dt: float, name: str) -> None:
    if len(trajectory) > 1:
        step = trajectory[1].timestamp - trajectory[0].timestamp
        if abs(step - dt) > 1e-9:
            raise ValueError(f"{name} was built with a time step of {step:g} s, but dt = {dt:g} s")


class EncounterSimulator:
    """Simulate a close-proximity encounter between an ownship and one intruder.

    The ownship follows its nominal trajectory until the run-time safety
    monitor or the envelope monitor triggers; from the next step on the
    contingency manager flies the recovery. The intruder always follows its
    trajectory.
    """

    def __init__(
        self,
        thresholds: WellClearThresholds = None,
        monitor_config: MonitorConfig = None,
    ):
        self.thresholds = thresholds or WellClearThresholds.do365_terminal()
        self.monitor_config = monitor_config or MonitorConfig()

        self.monitor = RuntimeSafetyMonitor(self.thresholds, self.monitor_config)
        self.envelope_monitor = EnvelopeMonitor(self.monitor_config)
        self.contingency = ContingencyManager(maneuver_type="breakout")

    def run(
        self,
        own_nominal_trajectory: List[AircraftState],
        intr_trajectory: List[AircraftState],
        dt: float = 0.1,
        fault_detector_confidence: float = 1.0,
        fault_nav_integrity: bool = True,
        enable_recovery: bool = True,
        enable_lookahead: bool = True,
        early_stop_on_lowc: bool = False,
    ) -> SimulationRecord:
        """Run one encounter and return its time history.

        Args:
            own_nominal_trajectory: ownship states at every step (nominal plan).
            intr_trajectory: intruder states at every step.
            dt: time step (s); must match the step used to build the trajectories.
            fault_detector_confidence: detector confidence fed to the envelope monitor.
            fault_nav_integrity: navigation integrity flag fed to the envelope monitor.
            enable_recovery: if False, the ownship never leaves its nominal trajectory.
            enable_lookahead: if False, the run-time safety monitor is not evaluated.
            early_stop_on_lowc: stop at the first loss of well clear.
        """
        _check_time_step(own_nominal_trajectory, dt, "own_nominal_trajectory")
        _check_time_step(intr_trajectory, dt, "intr_trajectory")

        self.monitor.reset()
        self.envelope_monitor.reset()
        self.contingency.reset()

        record = SimulationRecord()
        n_steps = min(len(own_nominal_trajectory), len(intr_trajectory))
        recovering = False
        own_state = own_nominal_trajectory[0]

        for i in range(n_steps):
            t = i * dt
            intr_state = intr_trajectory[i]

            # 1. Ownship state at this instant
            flown_by_recovery = recovering
            if recovering:
                own_state = self.contingency.step_recovery(own_state, dt, t)
            else:
                own_state = own_nominal_trajectory[i]

            # 2. Traffic detection: perfect tracking inside the sensor range
            dist_to_intr = float(np.linalg.norm(own_state.pos[:2] - intr_state.pos[:2]))
            intr_detected = dist_to_intr <= self.monitor_config.sensor_range_ft
            if intr_detected and record.t_det is None:
                record.t_det = t

            # 3. Envelope monitor
            out_of_envelope = self.envelope_monitor.evaluate(
                t=t,
                detector_confidence=fault_detector_confidence,
                nav_integrity_valid=fault_nav_integrity,
            )
            if out_of_envelope:
                record.envelope_triggered = True

            # 4. Run-time safety monitor
            if enable_lookahead:
                verdict = self.monitor.evaluate(
                    t=t,
                    own_state=own_state,
                    intr_state=intr_state,
                    intr_detected=intr_detected,
                )
            else:
                verdict = MonitorVerdict(timestamp=t, is_safe=True, trigger_fallback=False)

            if verdict.trigger_fallback and record.t_mon is None:
                record.t_mon = t
                record.t_wc_projected = verdict.t_wc_projected
                if verdict.t_wc_projected is not None:
                    record.lead_time_margin_s = (verdict.t_wc_projected - t) - self.monitor_config.t_rec

            # 5. Hand over to the contingency manager from the next step on
            if enable_recovery and not recovering and (verdict.trigger_fallback or out_of_envelope):
                recovering = True
                record.recovery_triggered = True

            # 6. Actual well-clear status
            wc_status = evaluate_well_clear(
                own_state.pos, own_state.vel, intr_state.pos, intr_state.vel, self.thresholds)

            if wc_status.horizontal_distance < record.min_horizontal_separation:
                record.min_horizontal_separation = wc_status.horizontal_distance

            if wc_status.is_loss_of_well_clear:
                record.experienced_lowc = True
                if record.t_wc_actual is None:
                    record.t_wc_actual = t
                if early_stop_on_lowc:
                    break

            record.timestamps.append(t)
            record.own_states.append(own_state)
            record.intr_states.append(intr_state)
            record.horizontal_ranges.append(wc_status.horizontal_distance)
            record.tau_mods.append(wc_status.tau_mod)
            record.lowc_flags.append(wc_status.is_loss_of_well_clear)
            record.monitor_verdicts.append(verdict)
            record.recovery_active_flags.append(flown_by_recovery)

        return record
