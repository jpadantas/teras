"""
Ready-made encounters: in-trail pattern traffic, a base-to-final conflict with
a straight-in intruder, and downwind traffic against an aircraft on final.
Each function returns (ownship trajectory, intruder trajectory) built with the
same time step ``dt``.
"""

from typing import List, Tuple

import numpy as np

from teras.traffic_pattern import (
    AircraftState,
    TrafficPatternGenerator,
    NM_TO_FT,
    KT_TO_FT_S,
)

Trajectories = Tuple[List[AircraftState], List[AircraftState]]


def create_nominal_pattern_encounter(
    spacing_nm: float = 1.0,
    speed_own_kt: float = 90.0,
    speed_intr_kt: float = 85.0,
    duration_s: float = 90.0,
    dt: float = 0.1,
) -> Trajectories:
    """Two aircraft flying the pattern in trail, both from the downwind leg.

    The intruder starts ``spacing_nm`` ahead of the ownship along the downwind leg.
    """
    gen = TrafficPatternGenerator()
    intr_traj = gen.simulate_trajectory(
        start_leg="downwind", start_offset_ft=spacing_nm * NM_TO_FT,
        speed_kt=speed_intr_kt, duration_s=duration_s, dt=dt)
    own_traj = gen.simulate_trajectory(
        start_leg="downwind", start_offset_ft=0.0,
        speed_kt=speed_own_kt, duration_s=duration_s, dt=dt)
    return own_traj, intr_traj


def create_base_final_conflict(
    duration_s: float = 70.0,
    dt: float = 0.1,
    speed_own_kt: float = 80.0,
    speed_intr_kt: float = 95.0,
    intr_start_nm: float = 1.8,
) -> Trajectories:
    """Ownship turning from base to final while an unannounced intruder flies a straight-in approach.

    The intruder starts ``intr_start_nm`` before the threshold on the extended
    centerline, at 600 ft, and descends at 3 deg towards the runway.
    """
    own_traj = TrafficPatternGenerator().simulate_trajectory(
        start_leg="base", start_offset_ft=0.0,
        speed_kt=speed_own_kt, duration_s=duration_s, dt=dt)

    v = speed_intr_kt * KT_TO_FT_S
    vz = -v * np.tan(np.radians(3.0))
    x, z = -intr_start_nm * NM_TO_FT, 600.0
    intr_traj = []
    for i in range(int(duration_s / dt)):
        on_ground = z <= 0.0
        intr_traj.append(AircraftState(
            x=x, y=0.0, z=max(0.0, z), vx=v, vy=0.0, vz=0.0 if on_ground else vz,
            speed_kt=speed_intr_kt, heading_deg=90.0, timestamp=i * dt))
        x += v * dt
        z = max(0.0, z + vz * dt)
    return own_traj, intr_traj


def create_downwind_parallel_traffic(
    duration_s: float = 80.0,
    dt: float = 0.1,
) -> Trajectories:
    """Ownship on the downwind leg (westbound) and an intruder on final (eastbound).

    The two legs are 0.75 NM apart, so the encounter shows how a well-clear
    volume treats opposite-direction traffic on parallel legs of the pattern.
    """
    gen = TrafficPatternGenerator()
    own_traj = gen.simulate_trajectory(
        start_leg="downwind", start_offset_ft=1000.0, speed_kt=90.0, duration_s=duration_s, dt=dt)
    intr_traj = gen.simulate_trajectory(
        start_leg="final", start_offset_ft=1500.0, speed_kt=85.0, duration_s=duration_s, dt=dt)
    return own_traj, intr_traj
