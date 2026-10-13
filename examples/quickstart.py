"""
Minimal TERAS example: one base-to-final conflict, without recovery and with
recovery under two assumed recovery times.

    python examples/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teras import (  # noqa: E402
    EncounterSimulator,
    MonitorConfig,
    WellClearThresholds,
    create_base_final_conflict,
)

# Ownship turning from base to final; unannounced intruder on a straight-in approach.
own, intruder = create_base_final_conflict(duration_s=70.0, dt=0.1)

# Well-clear volume: DO-365 terminal area (1500 ft, 450 ft, no time threshold).
thresholds = WellClearThresholds.do365_terminal()

# The monitor triggers t_rec + 1 s before the projected loss of well clear. If the
# assumed t_rec is shorter than the time the recovery really needs, the monitor
# meets its own budget and well clear is still lost.
for recovery, t_rec in ((False, 6.0), (True, 6.0), (True, 8.0)):
    config = MonitorConfig(t_rec=t_rec, sensor_range_ft=3.0 * 6076.12)
    sim = EncounterSimulator(thresholds=thresholds, monitor_config=config)
    rec = sim.run(own, intruder, dt=0.1, enable_recovery=recovery)
    label = f"recovery, t_rec={t_rec:4.1f} s" if recovery else "no recovery          "
    t_mon = f"{rec.t_mon:5.1f} s" if rec.t_mon is not None else "   none"
    t_wc = f"{rec.t_wc_actual:5.1f} s" if rec.t_wc_actual is not None else "   none"
    print(f"{label}  monitor trigger {t_mon}   loss of well clear {t_wc}   "
          f"min separation {rec.min_horizontal_separation:6.0f} ft")
