"""
TERAS Interactive Web Dashboard (Streamlit).
Configure an encounter, adjust the well-clear thresholds, inject a detector
fault, and watch the run-time safety monitor, the envelope monitor and the
contingency recovery act in real time.

    streamlit run apps/dashboard.py      # or: python apps/dashboard.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go

if __name__ == "__main__" and not st.runtime.exists():
    # Launched with plain `python`: relaunch through Streamlit, local access only.
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", os.path.abspath(__file__),
                "--server.address", "localhost", *sys.argv[1:]]
    sys.exit(stcli.main())

from teras.wellclear import WellClearThresholds
from teras.traffic_pattern import NM_TO_FT
from teras.monitors import MonitorConfig
from teras.simulator import EncounterSimulator
from teras.scenarios import (
    create_nominal_pattern_encounter,
    create_base_final_conflict,
    create_downwind_parallel_traffic,
)

st.set_page_config(
    page_title="TERAS: Terminal Encounter and Runtime Assurance Simulator",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ TERAS: Terminal Encounter and Runtime Assurance Simulator")
st.markdown(
    "**Run-time assurance in shared terminal airspace.** "
    "Kinematic, deterministic encounters; all vehicle and sensor parameters are assumed."
)

# Sidebar: Configuration
st.sidebar.header("⚙️ Simulation Settings")

scenario_choice = st.sidebar.selectbox(
    "Select Scenario",
    [
        "Base-to-Final Conflict (Lead-Time Budget)",
        "Nominal Traffic Pattern (Nuisance Alert Test)",
        "Downwind vs. Final Parallel Traffic",
    ],
)

wc_preset = st.sidebar.radio(
    "Well-Clear Standard",
    ["DO-365 terminal (1500 ft / 450 ft)", "DO-365 en route (4000 ft / 450 ft / 35 s)",
     "ASTM F3442 small UAS (2000 ft / 250 ft)", "Custom Thresholds"],
)

PRESETS = {
    "DO-365 terminal (1500 ft / 450 ft)": WellClearThresholds.do365_terminal(),
    "DO-365 en route (4000 ft / 450 ft / 35 s)": WellClearThresholds.do365_phase1(),
    "ASTM F3442 small UAS (2000 ft / 250 ft)": WellClearThresholds.astm_f3442(),
}
if wc_preset in PRESETS:
    p = PRESETS[wc_preset]
    hmd_val, tau_val, zthr_val = p.hmd_star, p.tau_mod_star, p.zthr
else:
    hmd_val = st.sidebar.slider("HMD* Threshold (ft)", 1000.0, 5000.0, 1500.0, 100.0)
    tau_val = st.sidebar.slider("Modified Tau threshold (s)", 0.0, 45.0, 0.0, 1.0)
    zthr_val = st.sidebar.slider("ZTHR Vertical Threshold (ft)", 100.0, 600.0, 450.0, 50.0)

thresholds = WellClearThresholds(
    name=wc_preset,
    hmd_star=hmd_val,
    dmod=hmd_val,
    tau_mod_star=tau_val,
    zthr=zthr_val,
)

st.sidebar.subheader("🛡️ Safety Monitor & Recovery Parameters")
t_rec = st.sidebar.slider("Required Recovery Time T_rec (s)", 4.0, 15.0, 8.0, 0.5)
sensor_range_nm = st.sidebar.slider("Traffic Detection Range (NM)", 0.5, 5.0, 3.0, 0.25)
enable_recovery = st.sidebar.checkbox("Enable Contingency Recovery", value=True)

st.sidebar.subheader("⚠️ Fault Injection (Envelope Monitor)")
inject_fog = st.sidebar.checkbox("Simulate Degraded Detector (Confidence Drop)", value=False)
confidence_val = 0.65 if inject_fog else 1.0

# Setup Scenario
if scenario_choice == "Base-to-Final Conflict (Lead-Time Budget)":
    own_traj, intr_traj = create_base_final_conflict(duration_s=70.0, dt=0.1)
    scenario_desc = "Ownship turns from base to final while an unannounced intruder flies a straight-in approach to runway 09."
elif scenario_choice == "Nominal Traffic Pattern (Nuisance Alert Test)":
    spacing_val = st.sidebar.slider("Aircraft Spacing on Downwind (NM)", 0.4, 1.5, 0.8, 0.1)
    own_traj, intr_traj = create_nominal_pattern_encounter(spacing_nm=spacing_val, duration_s=120.0, dt=0.1)
    scenario_desc = f"Both aircraft fly the pattern in trail, {spacing_val:.1f} NM apart on downwind."
else:
    own_traj, intr_traj = create_downwind_parallel_traffic(duration_s=60.0, dt=0.1)
    scenario_desc = "Ownship on downwind (westbound) and intruder on final (eastbound), 0.75 NM apart laterally."

st.info(f"**Scenario:** {scenario_desc}")

# Run Simulation
mon_cfg = MonitorConfig(
    t_rec=t_rec,
    sensor_range_ft=sensor_range_nm * NM_TO_FT,
)
sim = EncounterSimulator(thresholds=thresholds, monitor_config=mon_cfg)
record = sim.run(
    own_traj,
    intr_traj,
    dt=0.1,
    fault_detector_confidence=confidence_val,
    enable_recovery=enable_recovery,
)

# Metrics Cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Detection Time t_det", f"{record.t_det:.1f} s" if record.t_det is not None else "Out of Range")
c2.metric("Monitor Trigger t_mon", f"{record.t_mon:.1f} s" if record.t_mon is not None else "Safe (No Trigger)")
c3.metric(
    "Lead-Time Margin Δt",
    f"{record.lead_time_margin_s:+.1f} s" if record.lead_time_margin_s is not None else "N/A",
    delta=f"{record.lead_time_margin_s:+.1f} s" if record.lead_time_margin_s is not None else None,
)
c4.metric("Min Horizontal Separation", f"{record.min_horizontal_separation:.0f} ft")
c5.metric(
    "Well-Clear Status",
    "VIOLATED ❌" if record.experienced_lowc else "SAFE ✅",
)

col_map, col_plot = st.columns([1.1, 0.9])

with col_map:
    st.subheader("🗺️ 2D Plan View Geometry")
    own_x = [s.x for s in record.own_states]
    own_y = [s.y for s in record.own_states]
    intr_x = [s.x for s in record.intr_states]
    intr_y = [s.y for s in record.intr_states]

    fig_map = go.Figure()

    # Runway
    fig_map.add_trace(go.Scatter(
        x=[0, 5000], y=[0, 0],
        mode="lines",
        line=dict(color="#1e293b", width=8),
        name="Runway 09",
    ))

    # Ownship trajectory
    fig_map.add_trace(go.Scatter(
        x=own_x, y=own_y,
        mode="lines",
        line=dict(color="#2563eb", width=3),
        name="Ownship (uncrewed)",
    ))

    # Intruder trajectory
    fig_map.add_trace(go.Scatter(
        x=intr_x, y=intr_y,
        mode="lines",
        line=dict(color="#ea580c", width=3, dash="dash"),
        name="Intruder (crewed)",
    ))

    # Highlight t_mon if triggered
    if record.t_mon is not None:
        idx_m = int(round(record.t_mon / 0.1))
        fig_map.add_trace(go.Scatter(
            x=[own_x[idx_m]], y=[own_y[idx_m]],
            mode="markers",
            marker=dict(color="#dc2626", size=12, symbol="circle"),
            name=f"Monitor Trigger ({record.t_mon:.1f} s)",
        ))

    fig_map.update_layout(
        xaxis_title="Runway Axis X (ft)",
        yaxis_title="Lateral Offset Y (ft)",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_map)

with col_plot:
    st.subheader("📈 Separation & Well-Clear Invariant")
    t_arr = record.timestamps
    d_arr = record.horizontal_ranges

    fig_sep = go.Figure()

    fig_sep.add_trace(go.Scatter(
        x=t_arr, y=d_arr,
        mode="lines",
        line=dict(color="#059669", width=2.5),
        name="Horizontal Range (ft)",
    ))

    # Well-clear threshold line
    fig_sep.add_hline(
        y=thresholds.hmd_star,
        line_dash="dot",
        line_color="#dc2626",
        annotation_text=f"Well-Clear Limit ({thresholds.hmd_star:.0f} ft)",
    )

    if record.t_mon is not None:
        fig_sep.add_vline(
            x=record.t_mon,
            line_dash="dash",
            line_color="#d97706",
            annotation_text="Monitor Trigger",
        )

    fig_sep.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Separation (ft)",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_sep)

st.markdown("---")
st.markdown(
    "**TERAS** — Terminal Encounter and Runtime Assurance Simulator. "
    "See the repository README for the model and its assumptions."
)
