"""
Airfield, traffic-pattern geometry and kinematic pattern trajectories.

The airfield has a single runway 09 with its threshold at the origin
(x east along the runway, y north, z height above the runway, all in feet).
Traffic flies a left-hand pattern: downwind westbound at y = pattern_width,
base southbound at x = base_leg_x, and final eastbound on the extended
centerline (y = 0). Turns are flown at a constant bank angle and started one
turn radius before the next leg, so the aircraft rolls out on that leg.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

NM_TO_FT = 6076.12
FT_TO_NM = 1.0 / NM_TO_FT
KT_TO_FT_S = 1.68781
FT_S_TO_KT = 1.0 / KT_TO_FT_S
G_FT_S2 = 32.174

PATTERN_LEGS = ("downwind", "base", "final")


@dataclass
class AirfieldConfig:
    """Runway and left-hand traffic-pattern geometry (runway 09, threshold at the origin)."""
    runway_length: float = 5000.0            # ft
    pattern_width: float = 0.75 * NM_TO_FT   # lateral offset of the downwind leg (~4557 ft)
    pattern_altitude: float = 1000.0         # downwind height above the runway (ft)
    base_leg_x: float = -1.0 * NM_TO_FT      # along-runway position of the base leg (ft)
    final_approach_dist: float = 1.5 * NM_TO_FT  # start of a straight-in final (~9114 ft)
    bank_deg: float = 25.0                   # bank angle of pattern turns (deg)
    descent_rate: float = 10.0               # descent from abeam the threshold until on final (ft/s, 600 ft/min)
    glideslope_deg: float = 3.0              # path angle of an aircraft starting on final (deg)


@dataclass
class AircraftState:
    """Instantaneous kinematic state of an aircraft."""
    x: float            # east, along the runway axis (ft)
    y: float            # north, across the runway axis (ft)
    z: float            # height above the runway (ft)
    vx: float           # ft/s
    vy: float           # ft/s
    vz: float           # ft/s
    speed_kt: float     # horizontal speed (kt)
    heading_deg: float  # course (deg, 0 = north, 90 = east)
    timestamp: float = 0.0

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def vel(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=float)


def turn_rate_deg_s(speed_kt: float, bank_deg: float) -> float:
    """Rate of a coordinated level turn at the given speed and bank angle (deg/s)."""
    return float(np.degrees(G_FT_S2 * np.tan(np.radians(bank_deg)) / (speed_kt * KT_TO_FT_S)))


class TrafficPatternGenerator:
    """Kinematic trajectories through the downwind, base and final legs.

    Point-mass model at constant horizontal speed. On downwind the aircraft
    holds pattern altitude until abeam the threshold, then descends at
    ``descent_rate`` through the base leg and both turns. Once established on
    final it descends on a straight path to the threshold, after which it
    rolls along the runway at the same speed.
    """

    def __init__(self, config: AirfieldConfig = None):
        self.config = config or AirfieldConfig()

    def simulate_trajectory(
        self,
        start_leg: str = "downwind",
        start_offset_ft: float = 0.0,
        speed_kt: float = 90.0,
        duration_s: float = 120.0,
        dt: float = 0.1,
    ) -> List[AircraftState]:
        """Fly the pattern from ``start_leg`` and return one state per time step.

        start_leg:
            'downwind': westbound on the downwind leg, ``start_offset_ft`` west of
                        abeam the runway end plus 1000 ft, at pattern altitude.
            'base':     southbound at the start of the base leg, ``start_offset_ft``
                        south of the downwind leg, at 600 ft.
            'final':    eastbound on the extended centerline, ``start_offset_ft``
                        east of ``final_approach_dist`` before the threshold, on a
                        ``glideslope_deg`` path.
        """
        if start_leg not in PATTERN_LEGS:
            raise ValueError(f"start_leg must be one of {PATTERN_LEGS}, got {start_leg!r}")
        c = self.config
        if start_leg == "downwind":
            x, y, z, hdg = c.runway_length + 1000.0 - start_offset_ft, c.pattern_width, c.pattern_altitude, 270.0
        elif start_leg == "base":
            x, y, z, hdg = c.base_leg_x, c.pattern_width - start_offset_ft, 600.0, 180.0
        else:
            x, y, hdg = -c.final_approach_dist + start_offset_ft, 0.0, 90.0
            z = max(0.0, -x) * np.tan(np.radians(c.glideslope_deg))

        v = speed_kt * KT_TO_FT_S
        omega = turn_rate_deg_s(speed_kt, c.bank_deg)
        radius = v / np.radians(omega)
        # The two pattern turns must fit between the downwind leg and the centerline.
        room = {"downwind": c.pattern_width - 2.0 * radius,
                "base": c.pattern_width - start_offset_ft - radius}.get(start_leg, 0.0)
        if room < 0.0:
            raise ValueError(
                f"turn radius {radius:.0f} ft at {speed_kt:g} kt and {c.bank_deg:g} deg bank does not "
                f"fit the pattern; increase bank_deg or pattern_width, or reduce the speed")

        states = []
        for i in range(int(duration_s / dt)):
            # Heading and vertical speed flown during this step
            if 180.0 < hdg <= 270.0 and (hdg < 270.0 or x <= c.base_leg_x + radius):
                hdg = max(180.0, hdg - omega * dt)        # turn from downwind onto base
                vz = -c.descent_rate
            elif 90.0 < hdg <= 180.0 and (hdg < 180.0 or y <= radius):
                hdg = max(90.0, hdg - omega * dt)         # turn from base onto final
                vz = -c.descent_rate
            elif hdg == 180.0:
                vz = -c.descent_rate                      # straight base leg
            elif hdg == 90.0:
                vz = -z * v / max(-x, v * dt) if x < 0.0 else 0.0   # straight to the threshold, then rollout
            else:
                vz = -c.descent_rate if x <= 0.0 else 0.0  # downwind: descend from abeam the threshold
            if z <= 0.0:
                vz = 0.0                                  # on the ground

            r = np.radians(hdg)
            vx, vy = v * np.sin(r), v * np.cos(r)
            # State at t = i*dt (position before this step's motion), as in the scenario intruders
            states.append(AircraftState(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                                        speed_kt=speed_kt, heading_deg=hdg, timestamp=i * dt))
            x += vx * dt
            y += vy * dt
            z = z + vz * dt
            if z < 1e-6:
                z = 0.0
        return states
