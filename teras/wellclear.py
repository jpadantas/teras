"""
Detect-and-avoid well-clear evaluation in the RTCA DO-365 form.

Loss of well clear needs both a horizontal condition (range within DMOD, or
modified tau within tau*_mod with projected miss distance within HMD*) and a
vertical condition (altitude difference within ZTHR). The presets use
DMOD = HMD*, as the standards do. Presets cover the DO-365
en-route and terminal-area volumes and the ASTM F3442 small-UAS volume.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class WellClearThresholds:
    """Thresholds defining a well-clear volume (distances in ft, times in s)."""
    name: str
    hmd_star: float      # horizontal miss distance threshold HMD* (ft)
    dmod: float          # distance modification DMOD (ft)
    tau_mod_star: float  # modified tau threshold (s); 0 disables the time condition
    zthr: float          # vertical threshold ZTHR (ft)

    @classmethod
    def do365_phase1(cls) -> "WellClearThresholds":
        """RTCA DO-365 en-route (Phase 1) well clear: 4000 ft, 450 ft, 35 s."""
        return cls(
            name="DO-365 en route",
            hmd_star=4000.0,
            dmod=4000.0,
            tau_mod_star=35.0,
            zthr=450.0,
        )

    @classmethod
    def do365_terminal(cls) -> "WellClearThresholds":
        """RTCA DO-365A/B/C terminal-area well clear: 1500 ft, 450 ft, no time threshold."""
        return cls(
            name="DO-365 terminal",
            hmd_star=1500.0,
            dmod=1500.0,
            tau_mod_star=0.0,
            zthr=450.0,
        )

    @classmethod
    def astm_f3442(cls) -> "WellClearThresholds":
        """ASTM F3442/F3442M small-UAS well clear: 2000 ft, 250 ft, no time threshold."""
        return cls(
            name="ASTM F3442 small UAS",
            hmd_star=2000.0,
            dmod=2000.0,
            tau_mod_star=0.0,
            zthr=250.0,
        )


@dataclass
class WellClearStatus:
    """Instantaneous well-clear metrics between ownship and intruder."""
    horizontal_distance: float  # Current horizontal range r (ft)
    hmd: float                  # Projected horizontal miss distance (ft)
    tau_mod: float              # Modified Tau (s)
    tcpa: float                 # Time to closest point of approach (s)
    vertical_distance: float    # Absolute altitude difference |Delta z| (ft)
    is_horizontal_violation: bool
    is_vertical_violation: bool
    is_loss_of_well_clear: bool  # True if both horizontal & vertical conditions violated


def evaluate_well_clear(
    p_own: np.ndarray,
    v_own: np.ndarray,
    p_intr: np.ndarray,
    v_intr: np.ndarray,
    thresholds: WellClearThresholds,
) -> WellClearStatus:
    """Evaluate the well-clear status between the ownship and one intruder.

    Positions are [x, y, z] in ft and velocities [vx, vy, vz] in ft/s.
    """
    # Horizontal relative position and velocity
    rel_p_xy = p_intr[:2] - p_own[:2]
    rel_v_xy = v_intr[:2] - v_own[:2]

    r = float(np.linalg.norm(rel_p_xy))
    v_rel_norm_sq = float(np.dot(rel_v_xy, rel_v_xy))
    s_dot_v = float(np.dot(rel_p_xy, rel_v_xy))

    # Time to Closest Point of Approach (tcpa)
    if v_rel_norm_sq > 1e-6:
        tcpa = -s_dot_v / v_rel_norm_sq
    else:
        tcpa = 0.0

    # Horizontal Miss Distance (HMD)
    if tcpa > 0:
        cpa_pos = rel_p_xy + rel_v_xy * tcpa
        hmd = float(np.linalg.norm(cpa_pos))
    else:
        hmd = r

    # Modified Tau (tau_mod)
    # Defined only when closing horizontally (s_dot_v < 0)
    if s_dot_v < -1e-6:
        # tau_mod = -(r^2 - DMOD^2) / (s . v)
        tau_mod = -(r**2 - thresholds.dmod**2) / s_dot_v
        if tau_mod < 0:
            tau_mod = 0.0
    else:
        tau_mod = float("inf")

    # Vertical distance
    delta_z = abs(float(p_intr[2] - p_own[2]))

    # Violations
    # Horizontal violation: range <= DMOD, or (tau_mod <= tau_mod* and HMD <= HMD*)
    h_violation = (r <= thresholds.dmod) or (
        tau_mod <= thresholds.tau_mod_star and hmd <= thresholds.hmd_star
    )
    v_violation = delta_z <= thresholds.zthr

    lowc = h_violation and v_violation

    return WellClearStatus(
        horizontal_distance=r,
        hmd=hmd,
        tau_mod=tau_mod,
        tcpa=tcpa,
        vertical_distance=delta_z,
        is_horizontal_violation=h_violation,
        is_vertical_violation=v_violation,
        is_loss_of_well_clear=lowc,
    )
