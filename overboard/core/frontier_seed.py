"""
Kinetic Frontier Analytical Seed Generator
=========================================
Encodes the 5 physical principles governing time-optimal differential-drive control:
1. Principle of the Active Kinematic Hypotenuse (100% motor polytope saturation).
2. Late-Apex Dynamic Law (f_apex = 0.66 - 0.70 asymmetric trail-braking & early full throttle).
3. Waypoint Momentum Preservation (maintaining 0.65 - 0.75 * v_max through unpinned waypoints).
4. KKT Slack Barrier Equilibrium (riding the 4.8 mm boundary of the 5.0 mm gate).
5. Multi-Modal Homotopy Triad (Sprint Line, Smooth Sweeper, and Exploratory Homotopy).

Bypasses stochastic evolutionary searches entirely by computing Pontryagin-optimal
seeds deterministically in < 1 millisecond.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from overboard.models.drivetrain_physics import (
    CG_HEIGHT_M,
    angle_diff,
    asymmetric_late_apex_bump,
    diff_drive_ik,
    kinematic_cornering_ceiling,
)
from overboard.models.robot_model import RobotConfig

GUESS_STYLES = (
    "baseline",
    "rolling",
    "frontier_sprint",
    "frontier_sweeper",
    "frontier_exploratory",
    "frontier_sprint_mirror",
    "frontier_spline",
    "frontier_reverse",
)


# ---------------------------------------------------------------------------
# Style Parameters Specification & Lookup Table
# ---------------------------------------------------------------------------

_DATACLASS_KWARGS = {"frozen": True}
if sys.version_info >= (3, 10):
    _DATACLASS_KWARGS["slots"] = True


@dataclass(**_DATACLASS_KWARGS)
class StyleParams:
    u_peak: float = 0.50
    apex_mult: float = 0.0
    accel_scale: float = 0.90
    momentum_frac: float = 0.50
    zero_velocity: bool = False
    # Q17 geometry axes: the mirrored bow flips the apex amplitude, and the
    # bezier geometry replaces the straight chords with a cubic path family.
    apex_sign: int = 1
    geometry: str = "straight"


STYLE_TABLE: Dict[str, StyleParams] = {
    "sprint": StyleParams(u_peak=0.70, apex_mult=1.00, accel_scale=0.95, momentum_frac=0.72),
    "sweeper": StyleParams(u_peak=0.65, apex_mult=0.90, accel_scale=0.80, momentum_frac=0.68),
    "exploratory": StyleParams(u_peak=0.66, apex_mult=1.25, accel_scale=0.85, momentum_frac=0.70),
    "rolling": StyleParams(u_peak=0.50, apex_mult=0.00, accel_scale=0.80, momentum_frac=0.60),
    "baseline": StyleParams(u_peak=0.50, apex_mult=0.00, accel_scale=0.80, momentum_frac=0.00, zero_velocity=True),
    "direct_chord": StyleParams(u_peak=0.50, apex_mult=0.00, accel_scale=0.90, momentum_frac=0.50),
    # Q17 tail (pre-registered, docs/RESEARCH_AGENDA.md §24): the mirrored
    # bow, the bezier geometry, and reverse seeding on top of sprint.
    "sprint_mirror": StyleParams(u_peak=0.70, apex_mult=1.00, accel_scale=0.95, momentum_frac=0.72, apex_sign=-1),
    "spline": StyleParams(u_peak=0.70, apex_mult=1.00, accel_scale=0.95, momentum_frac=0.72, geometry="bezier"),
    "reverse": StyleParams(u_peak=0.70, apex_mult=1.00, accel_scale=0.95, momentum_frac=0.72),
}
DEFAULT_STYLE_PARAMS = StyleParams(u_peak=0.50, apex_mult=0.00, accel_scale=0.90, momentum_frac=0.50)


# ---------------------------------------------------------------------------
# Vectorized Math & Geometry Helpers
# ---------------------------------------------------------------------------

def compute_tangent_headings(
    waypoints: Sequence[Tuple[float, float, Optional[float]]],
) -> List[float]:
    """Compute smooth tangent/bisector headings for waypoints in pure vectorized NumPy."""
    num_wps = len(waypoints)
    if num_wps == 0:
        return []
    if num_wps == 1:
        th0 = float(waypoints[0][2]) if len(waypoints[0]) > 2 and waypoints[0][2] is not None else 0.0
        return [th0]

    pts = np.array([[w[0], w[1]] for w in waypoints], dtype=float)
    diffs = np.diff(pts, axis=0)
    chord_angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    headings = np.empty(num_wps, dtype=float)
    headings[0] = chord_angles[0]
    headings[-1] = chord_angles[-1]

    if num_wps > 2:
        a_prev = chord_angles[:-1]
        a_next = chord_angles[1:]
        headings[1:-1] = a_prev + 0.5 * angle_diff(a_next, a_prev)

    pinned_headings = [float(w[2]) if len(w) > 2 and w[2] is not None else None for w in waypoints]
    pinned_mask = np.array([h is not None for h in pinned_headings], dtype=bool)
    if np.any(pinned_mask):
        headings[pinned_mask] = [h for h in pinned_headings if h is not None]

    return [float(h) for h in headings]


def _bezier_path_points(
    p0: np.ndarray,
    p1: np.ndarray,
    th0: float,
    th1: float,
    u: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic Bezier positions with heading-aligned endpoint tangents.

    The Q17 geometry axis, promoted from research/seeds.py's spline family
    (research code may not be imported by overboard/). The handle length is a
    third of the chord, so a straight chord with matching tangents reproduces
    itself, and both endpoints land exactly on the waypoints.
    """
    handle = math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])) / 3.0
    c1x = p0[0] + handle * math.cos(th0)
    c1y = p0[1] + handle * math.sin(th0)
    c2x = p1[0] - handle * math.cos(th1)
    c2y = p1[1] - handle * math.sin(th1)
    v = 1.0 - u
    px = v ** 3 * p0[0] + 3.0 * v ** 2 * u * c1x + 3.0 * v * u ** 2 * c2x + u ** 3 * p1[0]
    py = v ** 3 * p0[1] + 3.0 * v ** 2 * u * c1y + 3.0 * v * u ** 2 * c2y + u ** 3 * p1[1]
    return px, py


def _compute_segment_durations(
    lengths: np.ndarray,
    seg_samples: Sequence[int],
    nominal_dt: float = 0.1,
) -> np.ndarray:
    """Vectorized calculation of segment durations proportional to chord length."""
    num_segments = len(lengths)
    tot_len = float(np.sum(lengths))
    tot_time = nominal_dt * sum(seg_samples)
    if tot_len > 1e-6:
        seg_samples_arr = np.maximum(1, np.asarray(seg_samples, dtype=float))
        seg_time = tot_time * (lengths / tot_len)
        return np.clip(seg_time / seg_samples_arr, 0.01, 1.0)
    return np.full(num_segments, nominal_dt, dtype=float)


# ---------------------------------------------------------------------------
# Baseline chord seed (exact reproduction of the legacy chord-lookahead)
# ---------------------------------------------------------------------------

def _build_baseline_seed(
    waypoints: List[Tuple[float, float, Optional[float]]],
    seg_samples: Sequence[int],
) -> np.ndarray:
    """Exact vectorized reproduction of the legacy chord-lookahead baseline seed."""
    num_segments = len(waypoints) - 1
    pts = np.array([[w[0], w[1]] for w in waypoints], dtype=float)
    diffs = np.diff(pts, axis=0)
    lengths = np.hypot(diffs[:, 0], diffs[:, 1])

    # 1. Proportional time budget
    tot_len = float(np.sum(lengths))
    tot_time = 0.1 * sum(seg_samples)
    if tot_len > 1e-6:
        initial_dts = np.clip(tot_time * (lengths / tot_len) / np.maximum(1, seg_samples), 0.01, 1.0)
    else:
        initial_dts = np.full(num_segments, 0.1, dtype=float)

    # 2. Segment chord headings
    chord_dirs = np.arctan2(diffs[:, 1], diffs[:, 0])

    N = sum(seg_samples) + 1
    states = np.zeros((N, 5), dtype=float)
    wp_indices = [0] + list(np.cumsum(seg_samples))

    for i in range(num_segments):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        s_i = seg_samples[i]
        count = s_i if i < num_segments - 1 else s_i + 1
        u = np.arange(count, dtype=float) / float(s_i)

        # Straight chord positions
        px = pts[i, 0] + diffs[i, 0] * u
        py = pts[i, 1] + diffs[i, 1] * u

        # Heading look-ahead
        h1 = p1[2] if len(p1) > 2 and p1[2] is not None else None
        h2 = p2[2] if len(p2) > 2 and p2[2] is not None else None

        if h1 is not None and h2 is not None:
            th = h1 + angle_diff(h2, h1) * u
        elif h1 is not None:
            th = h1 + angle_diff(chord_dirs[i], h1) * u
        elif h2 is not None:
            th = chord_dirs[i] + angle_diff(h2, chord_dirs[i]) * u
        else:
            if i + 2 < len(waypoints):
                p3 = waypoints[i + 2]
                dir_to_p3 = math.atan2(p3[1] - p1[1], p3[0] - p1[0])
                delta = angle_diff(dir_to_p3, chord_dirs[i])
                th = chord_dirs[i] + delta * u
            else:
                th = np.full(count, chord_dirs[i])

        k_start = wp_indices[i]
        states[k_start : k_start + count, 0] = px
        states[k_start : k_start + count, 1] = py
        states[k_start : k_start + count, 2] = th
        # vl and vr remain strictly 0.0

    return np.concatenate([initial_dts, states.ravel()])


# ---------------------------------------------------------------------------
# Core Analytical Frontier Seed Generator
# ---------------------------------------------------------------------------

def generate_frontier_seed(
    waypoints: List[Tuple[float, float, Optional[float]]],
    segment_samples: Union[int, Sequence[int]],
    robot_config: RobotConfig,
    stop_waypoint_indices: Optional[List[int]] = None,
    style: str = "sprint",
    a_max: Optional[float] = None,
) -> np.ndarray:
    """
    Generate an analytical initial guess vector for TrajectoryOptimizer in < 1 ms.

    Returns:
      1D numpy array: [initial_dts (S,), states (N*5,)] matching TrajectoryOptimizer format.
    """
    num_segments = len(waypoints) - 1
    if num_segments <= 0:
        raise ValueError("At least two waypoints are required to generate trajectory seed.")

    if isinstance(segment_samples, (list, tuple, np.ndarray)):
        seg_samples = [int(s) for s in segment_samples]
    else:
        seg_samples = [int(segment_samples)] * num_segments

    wp_indices = [0] + list(np.cumsum(seg_samples))
    N = wp_indices[-1] + 1
    stops = set(stop_waypoint_indices or [])

    # Style resolution
    clean_style = style
    if clean_style.startswith("frontier_"):
        clean_style = clean_style.replace("frontier_", "")

    if clean_style == "baseline":
        return _build_baseline_seed(waypoints, seg_samples)

    params = STYLE_TABLE.get(clean_style, DEFAULT_STYLE_PARAMS)

    # Physical robot limits
    v_wheel_max = float(robot_config.max_linear_speed())
    track_width = float(robot_config.track_width)
    mu = float(getattr(robot_config, "cof", 0.6))
    g = 9.81

    if a_max is None:
        stall_torque = getattr(robot_config, "t_max_nm", 0.138)
        gear = getattr(robot_config, "gearing", 1.0)
        radius = float(robot_config.wheel_radius)
        mass = float(robot_config.mass)
        motor_stall_accel = (2.0 * stall_torque * gear / radius) / max(mass, 0.1)
        traction_accel = mu * g
        a_max = min(motor_stall_accel * 0.45, traction_accel * 0.65, 2.8) * params.accel_scale

    # 1. Vectorized Waypoint Geometry
    pts = np.array([[w[0], w[1]] for w in waypoints], dtype=float)
    diffs = np.diff(pts, axis=0)
    dx_arr = diffs[:, 0]
    dy_arr = diffs[:, 1]
    L_arr = np.hypot(dx_arr, dy_arr)
    chord_angles = np.arctan2(dy_arr, dx_arr)

    wp_headings = compute_tangent_headings(waypoints)
    wp_headings_arr = np.asarray(wp_headings, dtype=float)
    th1_arr = wp_headings_arr[:-1]
    th2_arr = wp_headings_arr[1:]
    dth_arr = angle_diff(th2_arr, th1_arr)

    nx_arr = -dy_arr / np.maximum(L_arr, 1e-6)
    ny_arr = dx_arr / np.maximum(L_arr, 1e-6)

    # Vectorized Apex Offset Amplitudes
    apex_amps = np.zeros(num_segments, dtype=float)
    if params.apex_mult > 0.0:
        corner_dth = np.empty(num_segments, dtype=float)
        for i in range(num_segments - 1):
            corner_dth[i] = angle_diff(chord_angles[i + 1], chord_angles[i])
        corner_dth[-1] = dth_arr[-1]

        next_wp_is_stop = np.array([(i + 1) in stops for i in range(num_segments)], dtype=bool)
        mask = (L_arr > 0.05) & (np.abs(corner_dth) > 0.15) & (~next_wp_is_stop)
        if np.any(mask):
            signs = -np.sign(corner_dth[mask])
            apex_amps[mask] = (
                params.apex_sign
                * signs
                * np.minimum(0.20 * L_arr[mask], 0.08)
                * params.apex_mult
                * np.sin(np.minimum(np.abs(corner_dth[mask]), math.pi / 2.0))
            )

    # Reverse seeding (Q17): on a segment whose pinned heading faces more
    # than 120 degrees off the chord the robot backs through the segment,
    # so the guess carries heading + pi and v < 0 there. Only the reverse
    # style takes this axis; every other style is untouched.
    reverse_seg = np.zeros(num_segments, dtype=bool)
    if clean_style == "reverse":
        reverse_limit = 2.0 * math.pi / 3.0
        for i in range(num_segments):
            for w in (waypoints[i], waypoints[i + 1]):
                if (len(w) > 2 and w[2] is not None
                        and abs(angle_diff(float(w[2]), chord_angles[i])) > reverse_limit):
                    reverse_seg[i] = True

    # 2. Vectorized Point and Heading Generation
    points_x = np.empty(N, dtype=float)
    points_y = np.empty(N, dtype=float)
    nominal_thetas = np.empty(N, dtype=float)

    for i in range(num_segments):
        s_i = seg_samples[i]
        count = s_i if i < num_segments - 1 else s_i + 1
        u = np.arange(count, dtype=float) / float(s_i)

        if params.geometry == "bezier":
            # The curve is the shape: no apex bump on top of it.
            px, py = _bezier_path_points(
                pts[i], pts[i + 1], wp_headings[i], wp_headings[i + 1], u)
        else:
            amp = apex_amps[i]
            bumps = asymmetric_late_apex_bump(u, u_peak=params.u_peak) if abs(amp) > 1e-4 else 0.0

            px = pts[i, 0] + dx_arr[i] * u + nx_arr[i] * amp * bumps
            py = pts[i, 1] + dy_arr[i] * u + ny_arr[i] * amp * bumps

        th = th1_arr[i] + dth_arr[i] * u

        k_start = wp_indices[i]
        k_end = k_start + count
        points_x[k_start:k_end] = px
        points_y[k_start:k_end] = py
        nominal_thetas[k_start:k_end] = th

    # Pin exact coordinates at waypoint sample boundaries
    points_x[wp_indices] = pts[:, 0]
    points_y[wp_indices] = pts[:, 1]
    pinned_headings = [float(w[2]) if len(w) > 2 and w[2] is not None else None for w in waypoints]
    pinned_mask = np.array([h is not None for h in pinned_headings], dtype=bool)
    if np.any(pinned_mask):
        nominal_thetas[np.array(wp_indices)[pinned_mask]] = [h for h in pinned_headings if h is not None]

    # 3. Path Arc-Lengths, Tangent Headings, and Curvature
    dx_diff = np.diff(points_x)
    dy_diff = np.diff(points_y)
    ds = np.maximum(np.hypot(dx_diff, dy_diff), 1e-5)

    tangent_headings = np.empty(N, dtype=float)
    tangent_headings[:-1] = np.arctan2(dy_diff, dx_diff)
    tangent_headings[-1] = tangent_headings[-2] if N > 1 else wp_headings[-1]
    headings = np.unwrap(tangent_headings)

    # Reverse segments travel with heading + pi (the flip happens before the
    # pin adjustment, so a pinned waypoint heading - which is what made the
    # segment reverse - anchors the flipped field instead of fighting it).
    for i in range(num_segments):
        if reverse_seg[i]:
            k_start = wp_indices[i]
            k_end = k_start + (seg_samples[i] if i < num_segments - 1 else seg_samples[i] + 1)
            headings[k_start:k_end] += math.pi

    if np.any(pinned_mask):
        pinned_idx = np.array(wp_indices)[pinned_mask]
        pinned_targets = np.array([h for h in pinned_headings if h is not None], dtype=float)
        headings[pinned_idx] += angle_diff(pinned_targets, headings[pinned_idx])

    dtheta = np.diff(headings)
    kappa = np.zeros(N, dtype=float)
    kappa[:-1] = dtheta / ds
    kappa[-1] = kappa[-2] if N > 1 else 0.0
    if N > 4:
        kappa = np.convolve(kappa, [0.2, 0.6, 0.2], mode="same")

    # 4. Composite Velocity Ceiling & Waypoint Momentum Preservation
    v_frontier = v_wheel_max / (1.0 + 0.5 * track_width * np.abs(kappa))
    v_ceiling = kinematic_cornering_ceiling(kappa, track_width, v_wheel_max, h_cg=CG_HEIGHT_M, mu=mu, g=g)

    wp_idx_arr = np.array(wp_indices)
    is_stop_or_end = np.zeros(len(wp_indices), dtype=bool)
    is_stop_or_end[0] = True
    is_stop_or_end[-1] = True
    for s in stops:
        if s < len(wp_indices):
            is_stop_or_end[s] = True

    min_cruising_speed = params.momentum_frac * v_wheel_max
    unpinned_wp_idx = wp_idx_arr[~is_stop_or_end]
    v_ceiling[unpinned_wp_idx] = np.maximum(
        v_ceiling[unpinned_wp_idx],
        np.minimum(min_cruising_speed, v_frontier[unpinned_wp_idx]),
    )

    stop_sample_indices = wp_idx_arr[is_stop_or_end]
    v_ceiling[stop_sample_indices] = 0.0

    # 5. Fast Pontryagin Horizons in Energy Domain (E = v^2)
    dE = 2.0 * a_max * ds
    E = v_ceiling ** 2

    # Forward acceleration pass: E[k + 1] <= E[k] + dE[k]
    for k in range(N - 1):
        step_limit = E[k] + dE[k]
        if step_limit < E[k + 1]:
            E[k + 1] = step_limit

    # Backward braking pass: E[k] <= E[k + 1] + dE[k]
    for k in range(N - 2, -1, -1):
        step_limit = E[k + 1] + dE[k]
        if step_limit < E[k]:
            E[k] = step_limit

    v_profile = np.sqrt(E)
    non_stop_mask = np.ones(N, dtype=bool)
    non_stop_mask[stop_sample_indices] = False
    v_profile[non_stop_mask & (v_profile < 0.08)] = 0.08

    # 6. Actuator Diamond Boundary Wheel Velocities
    vl, vr = diff_drive_ik(v_profile, v_profile * kappa, track_width)
    vl = np.clip(vl, -v_wheel_max, v_wheel_max)
    vr = np.clip(vr, -v_wheel_max, v_wheel_max)
    vl[stop_sample_indices] = 0.0
    vr[stop_sample_indices] = 0.0

    # Backing through a reverse segment: v < 0 (stops already zeroed).
    for i in range(num_segments):
        if reverse_seg[i]:
            k_start = wp_indices[i]
            k_end = k_start + (seg_samples[i] if i < num_segments - 1 else seg_samples[i] + 1)
            vl[k_start:k_end] *= -1.0
            vr[k_start:k_end] *= -1.0

    # 7. Physical Segment Timing dt_i
    if params.zero_velocity:
        vl.fill(0.0)
        vr.fill(0.0)
        initial_dts = _compute_segment_durations(L_arr, seg_samples, nominal_dt=0.1)
    elif clean_style == "rolling":
        initial_dts = _compute_segment_durations(L_arr, seg_samples, nominal_dt=0.1)
    else:
        dt_samples = ds / np.maximum(0.5 * (v_profile[:-1] + v_profile[1:]), 0.01)
        seg_dt_sums = np.add.reduceat(dt_samples, wp_indices[:-1])
        initial_dts = np.clip(seg_dt_sums / np.asarray(seg_samples, dtype=float), 0.01, 1.0)

    # 8. Assemble Full Initial Guess [initial_dts, states]
    states = np.column_stack([points_x, points_y, headings, vl, vr])
    states[wp_indices, 0] = pts[:, 0]
    states[wp_indices, 1] = pts[:, 1]
    if np.any(pinned_mask):
        states[np.array(wp_indices)[pinned_mask], 2] = [h for h in pinned_headings if h is not None]

    return np.concatenate([initial_dts, states.ravel()])
