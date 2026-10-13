"""
TERAS: Terminal Encounter and Runtime Assurance Simulator
=========================================================
Kinematic simulator of close-proximity encounters between an uncrewed aircraft
and crewed traffic in a non-towered traffic pattern. It evaluates the DO-365
detect-and-avoid well-clear invariant, a run-time safety monitor with a
lead-time budget, an operational-envelope monitor, and a contingency recovery.

Quick start::

    from teras import EncounterSimulator, create_base_final_conflict

    own, intruder = create_base_final_conflict()
    record = EncounterSimulator().run(own, intruder)
    print(record.t_mon, record.t_wc_projected, record.min_horizontal_separation)
"""

from teras.traffic_pattern import AircraftState, AirfieldConfig, TrafficPatternGenerator
from teras.wellclear import WellClearThresholds, WellClearStatus, evaluate_well_clear
from teras.monitors import (
    MonitorConfig,
    MonitorVerdict,
    RuntimeSafetyMonitor,
    EnvelopeMonitor,
    ContingencyManager,
)
from teras.simulator import EncounterSimulator, SimulationRecord
from teras.scenarios import (
    create_nominal_pattern_encounter,
    create_base_final_conflict,
    create_downwind_parallel_traffic,
)

__version__ = "0.1.0"

__all__ = [
    "AircraftState", "AirfieldConfig", "TrafficPatternGenerator",
    "WellClearThresholds", "WellClearStatus", "evaluate_well_clear",
    "MonitorConfig", "MonitorVerdict", "RuntimeSafetyMonitor", "EnvelopeMonitor",
    "ContingencyManager", "EncounterSimulator", "SimulationRecord",
    "create_nominal_pattern_encounter", "create_base_final_conflict",
    "create_downwind_parallel_traffic",
]
