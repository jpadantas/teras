"""Behavioral tests for the encounter loop and the run-time assurance components."""

import pytest

from teras import (
    EncounterSimulator,
    MonitorConfig,
    WellClearThresholds,
    create_base_final_conflict,
    create_nominal_pattern_encounter,
)

NM = 6076.12


@pytest.fixture(scope="module")
def conflict():
    return create_base_final_conflict(duration_s=70.0, dt=0.1)


def test_unmitigated_conflict_loses_well_clear(conflict):
    rec = EncounterSimulator().run(*conflict, enable_recovery=False)
    assert rec.experienced_lowc
    assert not rec.recovery_triggered


def test_sufficient_recovery_time_preserves_well_clear(conflict):
    rec = EncounterSimulator(monitor_config=MonitorConfig(t_rec=8.0)).run(*conflict)
    assert rec.recovery_triggered
    assert not rec.experienced_lowc
    assert rec.t_mon is not None and rec.t_mon < rec.t_wc_projected


def test_optimistic_recovery_time_loses_well_clear_despite_margin(conflict):
    # The monitor meets its own budget (margin = buffer), but the breakout needs
    # more time than the assumed t_rec, so well clear is still lost.
    rec = EncounterSimulator(monitor_config=MonitorConfig(t_rec=6.0)).run(*conflict)
    assert rec.lead_time_margin_s == pytest.approx(1.0, abs=1e-6)
    assert rec.recovery_triggered and rec.experienced_lowc


def test_margin_equals_trigger_buffer_when_detected_early(conflict):
    cfg = MonitorConfig(t_rec=8.0, buffer_margin_s=1.5, sensor_range_ft=5 * NM)
    rec = EncounterSimulator(monitor_config=cfg).run(*conflict)
    assert rec.lead_time_margin_s == pytest.approx(1.5, abs=1e-6)


def test_short_detection_range_breaks_lead_time_budget(conflict):
    cfg = MonitorConfig(t_rec=10.0, sensor_range_ft=0.4 * NM)
    rec = EncounterSimulator(monitor_config=cfg).run(*conflict)
    assert rec.t_det is not None and rec.t_mon == pytest.approx(rec.t_det)
    assert rec.lead_time_margin_s < 0


def test_no_detection_means_no_monitor_trigger(conflict):
    cfg = MonitorConfig(sensor_range_ft=10.0)
    rec = EncounterSimulator(monitor_config=cfg).run(*conflict, enable_recovery=False)
    assert rec.t_det is None
    assert rec.t_mon is None


def test_envelope_monitor_triggers_recovery_on_low_confidence(conflict):
    rec = EncounterSimulator().run(*conflict, fault_detector_confidence=0.5)
    assert rec.envelope_triggered and rec.recovery_triggered
    # requested at the first step, flown from the next one
    assert not rec.recovery_active_flags[0]
    assert all(rec.recovery_active_flags[1:])


def test_ownship_and_intruder_are_evaluated_at_the_same_instant(conflict):
    own, intr = conflict
    rec = EncounterSimulator().run(own, intr, enable_recovery=False)
    for k in (0, 1, 250, len(rec.timestamps) - 1):
        assert rec.own_states[k] is own[k]
        assert rec.intr_states[k] is intr[k]
        assert rec.own_states[k].timestamp == pytest.approx(rec.timestamps[k])


def test_time_step_mismatch_is_rejected(conflict):
    with pytest.raises(ValueError):
        EncounterSimulator().run(*conflict, dt=0.2)


def test_wider_volume_flags_more_nominal_traffic():
    own, intr = create_nominal_pattern_encounter(spacing_nm=0.6, duration_s=120.0)
    small = EncounterSimulator(thresholds=WellClearThresholds.do365_terminal())
    large = EncounterSimulator(thresholds=WellClearThresholds.do365_phase1())
    kw = dict(enable_recovery=False, enable_lookahead=False)
    assert not small.run(own, intr, **kw).experienced_lowc
    assert large.run(own, intr, **kw).experienced_lowc
