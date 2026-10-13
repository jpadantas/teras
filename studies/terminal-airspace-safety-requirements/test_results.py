"""Regression tests: the code must reproduce the numbers quoted in Section VII.

If one of these fails after a change to the simulator, either the change is a
bug or the study's numbers (and results/teras_results.txt) must be updated.
Experiment 4 is too slow for a test; one of its cells is checked instead.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import (  # noqa: E402
    _conflict, _dz_at_closest, exp1_alert_burden, exp2_lead_time, exp3_coupling,
    min_detection_range,
)


@pytest.fixture(scope="module")
def e1():
    return exp1_alert_burden(hmd_values_ft=[1500.0, 2000.0, 3000.0, 4000.0])


@pytest.fixture(scope="module")
def e2():
    return exp2_lead_time()


@pytest.mark.parametrize("name, expected", [("DO-365 terminal", 1), ("ASTM F3442", 3),
                                            ("DO-365 en route", 14)])
def test_exp1_standard_volumes(e1, name, expected):
    assert len(e1["standards"][name]) == expected


def test_exp1_en_route_alerts_pairs_up_to_0p94_nm(e1):
    assert max(e1["standards"]["DO-365 en route"]) == pytest.approx(0.94, abs=0.01)


def test_exp1_effect_of_time_and_vertical_thresholds(e1):
    c = e1["counts"]
    n = {k: {h: len(v) for h, v in c[k].items()} for k in c}
    assert n["tau35_z450"][1500.0] - n["tau0_z450"][1500.0] == 1     # tau adds one at small HMD*
    assert n["tau35_z450"][4000.0] == n["tau0_z450"][4000.0]         # ... and none at 4000 ft
    assert n["tau0_z450"][4000.0] - n["tau0_z250"][4000.0] == 3      # ZTHR = 250 ft removes three


def test_exp2_monitor_meets_budget_with_buffer_as_margin(e2):
    r8 = e2["rec_assumed"]
    assert r8.t_mon == pytest.approx(17.8, abs=0.05)
    assert e2["unmit"].t_wc_actual == pytest.approx(24.8, abs=0.05)
    assert r8.lead_time_margin_s == pytest.approx(1.0, abs=1e-6)


def test_exp2_unmitigated_encounter_is_a_near_mid_air_collision(e2):
    u = e2["unmit"]
    assert u.min_horizontal_separation < 500 and _dz_at_closest(u) < 100


def test_exp2_optimistic_t_rec_loses_well_clear(e2):
    assert e2["rec_assumed"].experienced_lowc
    assert not e2["rec_ok"].experienced_lowc
    assert e2["t_need"] == 7.0


def test_exp3_minimum_detection_range():
    e3 = exp3_coupling(t_recs=(10.0,))
    assert e3["r_star"][10.0] == pytest.approx(0.58, abs=0.01)
    assert np.nanmax(e3["margins"][10.0]) == pytest.approx(1.0, abs=1e-6)


def test_exp4_one_cell():
    own, intr = _conflict(speed_own_kt=90.0, speed_intr_kt=125.0, intr_start_nm=2.2)
    r = min_detection_range(own, intr, 10.0)
    assert r is not None and 0.5 < r < 0.8
