"""Behavioral tests for the encounter loop and the run-time assurance components."""

import pytest

from teras import (
    EncounterSimulator,
    MonitorConfig,
    WellClearThresholds,
    create_base_final_conflict,
    create_nominal_pattern_encounter,
    TrafficPatternGenerator,
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


def test_alert_precedes_trigger_and_loss_of_well_clear(conflict):
    rec = EncounterSimulator(monitor_config=MonitorConfig(t_rec=8.0)).run(*conflict, enable_recovery=False)
    assert rec.t_alert is not None and rec.t_mon is not None
    assert rec.t_det <= rec.t_alert < rec.t_mon < rec.t_wc_actual
    # the alert comes as soon as the projection horizon reaches the loss of well clear
    first = next(v for v in rec.monitor_verdicts if v.t_wc_projected is not None)
    assert first.timestamp == pytest.approx(rec.t_alert)
    assert first.t_wc_projected - rec.t_alert <= MonitorConfig().lookahead_s + 1e-9


def test_own_intent_projection_alerts_earlier_when_the_conflict_follows_a_turn():
    # Ownship on downwind will turn onto base and final; a straight-in intruder
    # meets it there. Projecting the current velocity (westbound, then
    # southbound) sees the conflict only late; projecting the plan sees it early.
    gen = TrafficPatternGenerator()
    own = gen.simulate_trajectory("downwind", 4000.0, 80.0, 120.0, 0.1)
    intr = gen.simulate_trajectory("final", (1.5 - 3.0) * NM, 95.0, 120.0, 0.1)
    def run(intent):
        cfg = MonitorConfig(own_intent=intent, lookahead_s=45.0)
        return EncounterSimulator(monitor_config=cfg).run(own, intr, enable_recovery=False,
                                                          early_stop_on_lowc=True)
    straight, intent = run(False), run(True)
    assert intent.t_wc_actual == pytest.approx(straight.t_wc_actual)
    assert intent.t_alert < straight.t_alert - 20.0
    assert intent.t_alert >= intent.t_wc_actual - 45.0 - 1e-6


def test_operator_command_starts_recovery_at_the_commanded_time(conflict):
    sim = EncounterSimulator()
    early = sim.run(*conflict, enable_lookahead=False, recovery_command_time=5.0)
    assert early.recovery_triggered and not early.experienced_lowc
    k = int(round(5.0 / 0.1))
    assert not early.recovery_active_flags[k] and early.recovery_active_flags[k + 1]
    late = sim.run(*conflict, enable_lookahead=False, recovery_command_time=24.0)
    assert late.recovery_triggered and late.experienced_lowc


@pytest.mark.parametrize("maneuver, sign", [("left", -1), ("right", 1)])
def test_turn_maneuvers_turn_45_deg_from_the_heading_at_activation(conflict, maneuver, sign):
    own, intr = conflict
    rec = EncounterSimulator(maneuver_type=maneuver).run(
        own, intr, enable_lookahead=False, recovery_command_time=0.0)
    h0, h1 = own[0].heading_deg, rec.own_states[-1].heading_deg
    assert (h1 - h0 + 180) % 360 - 180 == pytest.approx(sign * 45.0)


def test_wider_volume_flags_more_nominal_traffic():
    own, intr = create_nominal_pattern_encounter(spacing_nm=0.6, duration_s=120.0)
    small = EncounterSimulator(thresholds=WellClearThresholds.do365_terminal())
    large = EncounterSimulator(thresholds=WellClearThresholds.do365_phase1())
    kw = dict(enable_recovery=False, enable_lookahead=False)
    assert not small.run(own, intr, **kw).experienced_lowc
    assert large.run(own, intr, **kw).experienced_lowc
