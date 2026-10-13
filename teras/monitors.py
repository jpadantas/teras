"""
Run-time assurance components, following the pattern of ASTM F3269-21.

- RuntimeSafetyMonitor: checks the projected encounter against the well-clear
  invariant and triggers the recovery while there is still time to fly it.
- EnvelopeMonitor: flags components operating outside their validated
  envelope (low detector confidence, invalid navigation integrity).
- ContingencyManager: flies the recovery maneuver once triggered.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from teras.traffic_pattern import AircraftState, KT_TO_FT_S, NM_TO_FT, turn_rate_deg_s
from teras.wellclear import (
    WellClearThresholds,
    WellClearStatus,
    evaluate_well_clear,
)


@dataclass
class MonitorConfig:
    """Configuration of detection, monitoring and recovery.

    The default values are illustrative assumptions, not properties of any
    particular aircraft or sensor.
    """
    t_rec: float = 8.0           # time the recovery needs to take effect, T_rec (s)
    lookahead_s: float = 30.0    # projection horizon of the run-time monitor (s)
    lookahead_dt: float = 0.5    # projection step (s)
    buffer_margin_s: float = 1.0 # extra lead time added to T_rec when triggering (s)
    sensor_range_ft: float = 3.0 * NM_TO_FT  # traffic detection range (ft)
    odd_confidence_min: float = 0.85        # minimum detector confidence inside the envelope
    own_intent: bool = False     # project the ownship along its own plan, not its current velocity


@dataclass
class MonitorVerdict:
    """Verdict produced by the run-time safety monitor at each cycle."""
    timestamp: float
    is_safe: bool
    trigger_fallback: bool
    t_mon: Optional[float] = None
    t_wc_projected: Optional[float] = None  # set whenever a loss of well clear is projected (alert)
    time_margin_s: Optional[float] = None  # (t_wc - t_mon) - T_rec
    status: Optional[WellClearStatus] = None


class RuntimeSafetyMonitor:
    """Run-time safety monitor with a lead-time budget.

    At each cycle both aircraft are projected along their current velocity
    over ``lookahead_s``. When the projected time to loss of well clear falls
    to ``t_rec + buffer_margin_s``, the monitor declares the plan unsafe and
    requests the recovery. Once triggered it stays triggered.

    Because the monitor acts as late as the budget allows, the reported
    margin ``(t_wc - t_mon) - t_rec`` equals ``buffer_margin_s`` whenever the
    intruder is detected early enough.

    Every verdict that projects a loss of well clear within the horizon
    carries ``t_wc_projected``, also before the trigger; that is the alert an
    operator would see (``SimulationRecord.t_alert``).

    With ``own_intent`` the ownship is projected along its own planned
    trajectory (``own_plan`` from ``plan_index`` on), which it knows; the
    intruder is still projected along its current velocity.
    """

    def __init__(self, thresholds: WellClearThresholds, config: MonitorConfig = None):
        self.thresholds = thresholds
        self.config = config or MonitorConfig()
        self.triggered: bool = False
        self.t_mon: Optional[float] = None
        self.t_wc_projected: Optional[float] = None
        self.time_margin: Optional[float] = None

    def reset(self):
        self.triggered = False
        self.t_mon = None
        self.t_wc_projected = None
        self.time_margin = None

    def evaluate(
        self,
        t: float,
        own_state: AircraftState,
        intr_state: AircraftState,
        intr_detected: bool,
        own_plan: Optional[List[AircraftState]] = None,
        plan_index: int = 0,
    ) -> MonitorVerdict:
        """Project the encounter forward and return this cycle's verdict.

        ``own_plan[plan_index]`` is the planned state at time ``t``; it is used
        only when ``config.own_intent`` is set.
        """
        # Current well-clear status
        current_wc = evaluate_well_clear(
            own_state.pos,
            own_state.vel,
            intr_state.pos,
            intr_state.vel,
            self.thresholds,
        )

        if not intr_detected:
            # Intruder not (or no longer) detected: nothing new to project
            return MonitorVerdict(
                timestamp=t,
                is_safe=not self.triggered,
                trigger_fallback=self.triggered,
                status=current_wc,
            )

        if self.triggered:
            return MonitorVerdict(
                timestamp=t,
                is_safe=False,
                trigger_fallback=True,
                t_mon=self.t_mon,
                t_wc_projected=self.t_wc_projected,
                time_margin_s=self.time_margin,
                status=current_wc,
            )

        # Projection horizon
        dt = self.config.lookahead_dt
        horizon = self.config.lookahead_s
        n_proj = int(horizon / dt)

        projected_t_wc = None
        use_plan = self.config.own_intent and own_plan is not None and len(own_plan) > 1
        plan_dt = own_plan[1].timestamp - own_plan[0].timestamp if use_plan else 0.0

        # Projection: intruder along its current velocity; ownship along its
        # current velocity, or along its own plan with ``own_intent``
        for step in range(1, n_proj + 1):
            t_fut = step * dt
            if use_plan:
                k = plan_index + int(round(t_fut / plan_dt))
                if k < len(own_plan):
                    p_own_proj, v_own_proj = own_plan[k].pos, own_plan[k].vel
                else:  # past the end of the plan: continue its last state
                    last = own_plan[-1]
                    p_own_proj = last.pos + last.vel * (t_fut - (len(own_plan) - 1 - plan_index) * plan_dt)
                    v_own_proj = last.vel
            else:
                p_own_proj, v_own_proj = own_state.pos + own_state.vel * t_fut, own_state.vel
            p_intr_proj = intr_state.pos + intr_state.vel * t_fut
            p_own_proj[2] = max(0.0, p_own_proj[2])     # no projection below the ground
            p_intr_proj[2] = max(0.0, p_intr_proj[2])

            proj_wc = evaluate_well_clear(
                p_own_proj,
                v_own_proj,
                p_intr_proj,
                intr_state.vel,
                self.thresholds,
            )

            if proj_wc.is_loss_of_well_clear:
                projected_t_wc = t + t_fut
                break

        # Lead-time budget: trigger while T_rec (+ buffer) still fits
        if projected_t_wc is not None:
            time_available = projected_t_wc - t
            trigger_threshold = self.config.t_rec + self.config.buffer_margin_s

            if time_available <= trigger_threshold:
                # Declare the plan unsafe and request the recovery
                self.triggered = True
                self.t_mon = t
                self.t_wc_projected = projected_t_wc
                self.time_margin = time_available - self.config.t_rec

                return MonitorVerdict(
                    timestamp=t,
                    is_safe=False,
                    trigger_fallback=True,
                    t_mon=self.t_mon,
                    t_wc_projected=self.t_wc_projected,
                    time_margin_s=self.time_margin,
                    status=current_wc,
                )

        return MonitorVerdict(
            timestamp=t,
            is_safe=True,
            trigger_fallback=False,
            t_wc_projected=projected_t_wc,  # alert without trigger, or None
            status=current_wc,
        )


class EnvelopeMonitor:
    """Operational-envelope and health monitor.

    Watches the detector confidence and the navigation integrity flag, and
    triggers the recovery independently of the run-time safety monitor when
    either leaves its validated envelope.
    """

    def __init__(self, config: MonitorConfig = None):
        self.config = config or MonitorConfig()
        self.triggered: bool = False
        self.trigger_time: Optional[float] = None
        self.trigger_reason: Optional[str] = None

    def reset(self):
        self.triggered = False
        self.trigger_time = None
        self.trigger_reason = None

    def evaluate(
        self,
        t: float,
        detector_confidence: float = 1.0,
        nav_integrity_valid: bool = True,
    ) -> bool:
        """Return True if an out-of-envelope condition is detected (triggers the recovery)."""
        if self.triggered:
            return True

        if detector_confidence < self.config.odd_confidence_min:
            self.triggered = True
            self.trigger_time = t
            self.trigger_reason = (
                f"Vision detector confidence {detector_confidence:.2f} below ODD minimum {self.config.odd_confidence_min}"
            )
            return True

        if not nav_integrity_valid:
            self.triggered = True
            self.trigger_time = t
            self.trigger_reason = "Navigation integrity risk exceeded (I_nav invalid)"
            return True

        return False


class ContingencyManager:
    """Contingency manager: flies the recovery maneuver once triggered.

    Maneuvers (all climb at about 700 ft/min in a coordinated turn at ``bank_deg``):
    - 'breakout': turn to heading 060, away from the final approach to runway 09.
    - 'left': turn 45 deg left of the heading at activation.
    - any other value (e.g. 'right'): turn 45 deg right of the heading at activation.
    """

    def __init__(self, maneuver_type: str = "breakout", bank_deg: float = 25.0,
                 climb_rate: float = 12.0):
        self.maneuver_type = maneuver_type
        self.bank_deg = bank_deg        # bank of the recovery turn (deg); same as the pattern turns
        self.climb_rate = climb_rate    # ft/s (assumed, ~700 ft/min)
        self.active: bool = False
        self.activation_time: Optional[float] = None
        self.evasion_heading: Optional[float] = None

    def reset(self):
        self.active = False
        self.activation_time = None
        self.evasion_heading = None

    def step_recovery(
        self,
        state: AircraftState,
        dt: float,
        t: float,
    ) -> AircraftState:
        """Advance the ownship one step along the recovery maneuver."""
        if not self.active:
            self.active = True
            self.activation_time = t
            # Breakout: climbing turn to heading 060, away from the final approach to runway 09.
            # The heading is fixed; it does not depend on where the intruder is.
            if self.maneuver_type == "breakout":
                self.evasion_heading = 60.0
            elif self.maneuver_type == "left":
                self.evasion_heading = (state.heading_deg - 45.0) % 360.0
            else:
                self.evasion_heading = (state.heading_deg + 45.0) % 360.0

        speed_fps = state.speed_kt * KT_TO_FT_S
        turn_rate = turn_rate_deg_s(state.speed_kt, self.bank_deg)

        # Steer towards the evasion heading
        diff = (self.evasion_heading - state.heading_deg + 180) % 360 - 180
        if abs(diff) > turn_rate * dt:
            current_hdg = state.heading_deg + np.sign(diff) * turn_rate * dt
        else:
            current_hdg = self.evasion_heading

        hdg_rad = np.radians(current_hdg)
        vx = speed_fps * np.sin(hdg_rad)
        vy = speed_fps * np.cos(hdg_rad)
        vz = self.climb_rate

        new_x = state.x + vx * dt
        new_y = state.y + vy * dt
        new_z = state.z + vz * dt

        return AircraftState(
            x=new_x,
            y=new_y,
            z=new_z,
            vx=vx,
            vy=vy,
            vz=vz,
            speed_kt=state.speed_kt,
            heading_deg=current_hdg,
            timestamp=t,
        )
