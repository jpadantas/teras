"""Tests for the traffic-pattern trajectories (teras.traffic_pattern)."""

import numpy as np
import pytest

from teras.traffic_pattern import AirfieldConfig, TrafficPatternGenerator, turn_rate_deg_s

GEN = TrafficPatternGenerator()


def _final(states):
    return [s for s in states if s.heading_deg == 90.0]


@pytest.mark.parametrize("leg, speed", [("downwind", 90.0), ("downwind", 85.0), ("base", 80.0)])
def test_turns_roll_out_on_the_extended_centerline(leg, speed):
    final = _final(GEN.simulate_trajectory(start_leg=leg, speed_kt=speed, duration_s=200.0))
    assert final, "never established on final"
    assert abs(final[0].y) < 25.0


@pytest.mark.parametrize("leg, speed", [("downwind", 90.0), ("base", 80.0), ("final", 85.0)])
def test_arrives_at_the_threshold_and_stays_on_the_ground(leg, speed):
    states = GEN.simulate_trajectory(start_leg=leg, speed_kt=speed, duration_s=250.0)
    past = [s for s in states if s.x >= 0.0 and s.heading_deg == 90.0]
    assert past and past[0].z == pytest.approx(0.0, abs=1.0)
    assert all(s.z == 0.0 and s.vz == 0.0 for s in past[5:])


def test_final_path_angle_is_shallow():
    final = _final(GEN.simulate_trajectory(start_leg="downwind", speed_kt=90.0, duration_s=200.0))
    angle = np.degrees(np.arctan(final[0].z / -final[0].x))
    assert 2.0 < angle < 5.0


def test_turn_rate_follows_bank_angle():
    # 25 deg bank at 90 kt: g tan(25) / v = 0.0988 rad/s
    assert turn_rate_deg_s(90.0, 25.0) == pytest.approx(5.66, abs=0.01)


def test_downwind_holds_pattern_altitude_until_abeam_threshold():
    states = GEN.simulate_trajectory(start_leg="downwind", speed_kt=90.0, duration_s=40.0)
    before = [s for s in states if s.x > 0.0]
    assert all(s.z == AirfieldConfig().pattern_altitude for s in before)


def test_timestamps_follow_the_time_step():
    states = GEN.simulate_trajectory(duration_s=1.0, dt=0.1)
    assert [round(s.timestamp, 6) for s in states] == [round(0.1 * i, 6) for i in range(10)]


def test_unknown_leg_is_rejected():
    with pytest.raises(ValueError):
        GEN.simulate_trajectory(start_leg="upwind")
