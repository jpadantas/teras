"""Regression tests: the code must reproduce the numbers quoted in Section VII.

If one of these fails after a change to the simulator, either the change is a
bug or the study's numbers (and results/) must be updated. Experiment 4 is too
slow for a test: one of its cells is recomputed, and the aggregates quoted in
the paper are checked against the committed results/exp4_geometries.json.
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_experiments import (  # noqa: E402
    _conflict, _dz_at_closest, exp1_alert_burden, exp1_overtaking, exp2_lead_time,
    exp3_coupling, min_detection_range, needed_t_rec,
)


@pytest.fixture(scope="module")
def e1():
    return exp1_alert_burden(hmd_values_ft=[1500.0, 2000.0, 3000.0, 3375.0, 4000.0])


@pytest.fixture(scope="module")
def e2():
    return exp2_lead_time()


@pytest.fixture(scope="module")
def e4():
    with open(os.path.join(HERE, "results", "exp4_geometries.json"), encoding="utf-8") as f:
        geoms = json.load(f)
    return [g for g in geoms if g["conflict"] and not g["miss"]]


@pytest.mark.parametrize("name, expected", [("DO-365 terminal", 0), ("ASTM F3442", 0),
                                            ("DO-365 en route", 11)])
def test_exp1_standard_volumes(e1, name, expected):
    assert len(e1["standards"][name]) == expected


def test_exp1_en_route_flags_pairs_up_to_0p82_nm(e1):
    assert max(e1["standards"]["DO-365 en route"]) == pytest.approx(0.82, abs=0.01)


def test_exp1_effect_of_time_and_vertical_thresholds(e1):
    c = e1["counts"]
    n = {k: {h: len(v) for h, v in c[k].items()} for k in c}
    assert all(n["tau35_z450"][h] == n["tau0_z450"][h] for h in n["tau0_z450"])  # tau changes nothing
    assert n["tau0_z250"][3000.0] == n["tau0_z450"][3000.0]                      # ZTHR 250 ft: none below 3375
    assert n["tau0_z450"][3375.0] > n["tau0_z250"][3375.0]


def test_exp1_overtaking_flags_closest_pairs():
    o = exp1_overtaking()
    assert [len(o[k]) for k in ("DO-365 terminal", "ASTM F3442")] == [2, 4]
    assert min(f[1] for f in o["DO-365 terminal"]) == pytest.approx(1165, abs=1)


def test_exp2_monitor_meets_budget_with_buffer_as_margin(e2):
    r6 = e2["rec_assumed"]
    assert r6.t_mon == pytest.approx(17.8, abs=0.05)
    assert e2["unmit"].t_wc_actual == pytest.approx(24.8, abs=0.05)
    assert r6.lead_time_margin_s == pytest.approx(1.0, abs=1e-6)


def test_exp2_unmitigated_encounter_is_a_near_mid_air_collision(e2):
    u = e2["unmit"]
    assert u.min_horizontal_separation < 500 and _dz_at_closest(u) < 100


def test_exp2_optimistic_t_rec_loses_well_clear(e2):
    assert e2["rec_assumed"].experienced_lowc
    assert not e2["rec_ok"].experienced_lowc
    assert e2["t_need"] == 7.0
    assert e2["t_need_turns"] == {"left": 13.0, "right": 13.0}


def test_exp3_minimum_detection_range():
    e3 = exp3_coupling(t_recs=(10.0,))
    assert e3["r_star"][10.0] == pytest.approx(0.58, abs=0.01)   # margin reaches zero
    assert e3["r_sr6"][10.0] == pytest.approx(0.61, abs=0.01)    # SR-6 met (margin >= T_buf)
    assert np.nanmax(e3["margins"][10.0]) == pytest.approx(1.0, abs=1e-6)


def test_exp4_one_cell():
    own, intr = _conflict(speed_own_kt=90.0, speed_intr_kt=125.0, intr_start_nm=2.2)
    r = min_detection_range(own, intr, 10.0)
    assert r is not None and 0.5 < r < 0.8
    assert needed_t_rec(own, intr) is not None


def test_exp4_aggregates_quoted_in_the_paper(e4):
    assert len(e4) == 87
    r = [g["r_det"] for g in e4 if g["r_det"] is not None]
    assert (round(min(r), 2), round(max(r), 2), round(float(np.median(r)), 2)) == (0.29, 0.67, 0.57)
    r6 = [g["r_det_sr6"] for g in e4 if g.get("r_det_sr6") is not None]
    assert (len(r6), round(min(r6), 2), round(max(r6), 2), round(float(np.median(r6)), 2)) == (85, 0.29, 0.70, 0.60)
    tb = [g["t_need"]["breakout"] for g in e4 if g["t_need"]["breakout"] is not None]
    assert (min(tb), max(tb), float(np.median(tb)), len(e4) - len(tb)) == (5.0, 14.5, 8.0, 2)
    tl = [g["t_need"]["left"] for g in e4 if g["t_need"]["left"] is not None]
    assert (min(tl), max(tl), float(np.median(tl))) == (7.0, 18.5, 12.5)
    tr = [g["t_need"]["right"] for g in e4 if g["t_need"]["right"] is not None]
    assert (min(tr), max(tr), len(e4) - len(tr)) == (0.5, 24.5, 11)
    best = [min(v for v in g["t_need"].values() if v is not None) for g in e4]
    assert (min(best), max(best), float(np.median(best))) == (0.5, 14.5, 7.5)
