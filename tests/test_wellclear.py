"""Unit tests for the well-clear evaluation (teras.wellclear)."""

import numpy as np

from teras.wellclear import WellClearThresholds, evaluate_well_clear

TH = WellClearThresholds.do365_phase1()


def _eval(p_own, v_own, p_intr, v_intr, th=TH):
    return evaluate_well_clear(np.array(p_own, float), np.array(v_own, float),
                               np.array(p_intr, float), np.array(v_intr, float), th)


def test_co_altitude_inside_hmd_is_loss_of_well_clear():
    s = _eval([0, 0, 1000], [0, 0, 0], [3000, 0, 1000], [0, 0, 0])
    assert s.horizontal_distance == 3000
    assert s.is_loss_of_well_clear


def test_far_and_diverging_is_well_clear():
    s = _eval([0, 0, 1000], [-150, 0, 0], [20000, 0, 1000], [150, 0, 0])
    assert not s.is_horizontal_violation
    assert not s.is_loss_of_well_clear


def test_vertical_separation_above_zthr_is_well_clear():
    s = _eval([0, 0, 1000], [0, 0, 0], [1000, 0, 1000 + TH.zthr + 50], [0, 0, 0])
    assert s.is_horizontal_violation
    assert not s.is_vertical_violation
    assert not s.is_loss_of_well_clear


def test_range_condition_uses_dmod():
    th = WellClearThresholds("dmod", hmd_star=2000.0, dmod=1000.0, tau_mod_star=0.0, zthr=450.0)
    stationary = ([0, 0, 1000], [0, 0, 0], [1500, 0, 1000], [0, 0, 0])
    assert not _eval(*stationary, th=th).is_horizontal_violation      # 1500 ft > DMOD
    assert _eval([0, 0, 1000], [0, 0, 0], [900, 0, 1000], [0, 0, 0], th=th).is_horizontal_violation


def test_terminal_volume_has_no_time_condition():
    th = WellClearThresholds.do365_terminal()
    # Head-on at 300 ft/s from 3000 ft: tau_mod small, but range > 1500 ft
    s = _eval([0, 0, 1000], [150, 0, 0], [3000, 0, 1000], [-150, 0, 0], th=th)
    assert not s.is_horizontal_violation


def test_tau_condition_triggers_before_range_condition():
    # Head-on closure at 300 ft/s from 10 000 ft: r > HMD*, but
    # tau_mod = (r^2 - DMOD^2) / (r * v) = 28 s < 35 s.
    s = _eval([0, 0, 1000], [150, 0, 0], [10000, 0, 1000], [-150, 0, 0])
    assert s.horizontal_distance > TH.hmd_star
    assert s.tau_mod < TH.tau_mod_star
    assert s.is_loss_of_well_clear
