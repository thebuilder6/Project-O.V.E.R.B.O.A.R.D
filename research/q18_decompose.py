"""Q18: decompose the 5 mm tracking gate's quantity (measurement tree only).

The gate compares ``max_tracking_error_m`` against ``TRACKING_GATE_M = 0.005``
(overboard/verification/validator.py:39; overboard/main.py:1651,1856). That
quantity is the MAX over sample intervals of the distance between the
closed-loop rollout's position at the interval end and the *planned sample*
position at the same instant (reachability_engine.py:341-344 and
immrax_validator.py:1482-1483 both measure against the planned sample row).

Q18 measures, per candidate, on one rollout:

* ``q_reported``  - the gate's own quantity (the product's ``tracking_m``);
* ``q_sagitta``   - the chord-shape component: distance between the smooth
  planned path (the KI-026 heading-integrated plan curve) and the
  piecewise-linear chorded reference the firmware follows (position chords,
  KI-027 linear heading);
* ``q_control``   - the control residual: rollout vs. the chorded reference.

Step 0(b) is ``arc_control()``: a synthetic self-consistent arc (analytic
circle positions, heading = tangent). All three components must read ~0 on it
at the gate's instants, and the instrument's chord bound must reproduce the
analytic sagitta; the registered test asserts exactly that.

This module does not change product code. The rollout is an independent
float64 mirror of the gate's plant propagation (the twin-active five-state
mean_diff branch of CompiledReachabilityEngine._interval_step /
DigitalTwinOpenLoop); every candidate is cross-checked against the product's
own per-sample tracking errors and the residual is recorded.

Usage::

    python research/q18_decompose.py --arc-control
    python research/q18_decompose.py research/missions examples \
        --out research/results/q18_decompose.jsonl --candidates 3 --propagation plant
    python research/q18_decompose.py ... --out research/results/q18_density_x2.jsonl \
        --samples-mult 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import overboard.main as cli  # noqa: E402  (needs ROOT on sys.path)
from overboard.core.optimizer import TrajectoryOptimizer, _GAUSS_LEGENDRE_4  # noqa: E402
from overboard.core.unified_planner import UnifiedPathEngine  # noqa: E402
from overboard.models.drivetrain_physics import angle_diff, body_frame_error  # noqa: E402
from overboard.models.robot_model import RobotConfig  # noqa: E402

DEFAULT_CONFIG = ROOT / "examples" / "fll_choreo.json"

# The gate's constants (overboard/verification/validator.py:39 and the
# reachability engine's control grid).
GATE_M = 0.005
CONTROL_DT = 0.02
PLANT_SUBSTEPS = 5
RAMSETE_B = 2.0
RAMSETE_ZETA = 0.7
MIN_TAU_S = 0.01

# Exact float64 port of overboard.models.jax_ramsete._SINC_CHEB / _sinc_poly
# (same coefficients, same three-term recurrence, same IEEE doubles).
_SINC_CHEB = (
    4.28930948e-01,
    -4.95478386e-01,
    7.09060458e-02,
    -4.51782748e-03,
    1.62945403e-04,
    -3.78552404e-06,
    6.18166893e-08,
)


def _sinc_poly(x: float) -> float:
    u = 2.0 * (x * x) / (math.pi ** 2) - 1.0
    b1 = 0.0
    b2 = 0.0
    for c in reversed(_SINC_CHEB[1:]):
        b0 = 2.0 * u * b1 - b2 + c
        b2 = b1
        b1 = b0
    return u * b1 - b2 + _SINC_CHEB[0]


def _num(value, digits=9):
    """JSON-safe number: Infinity and NaN become None (Infinity breaks JSON)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


# ---------------------------------------------------------------------------
# The three curves
# ---------------------------------------------------------------------------

def unroll_headings(samples: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Continuous headings across the +/- pi wrap (reachability_engine's rule)."""
    raw = np.array([float(s["heading"]) for s in samples], dtype=np.float64)
    out = raw.copy()
    offset = 0.0
    for i in range(1, len(raw)):
        delta = raw[i] - raw[i - 1]
        if delta > math.pi:
            offset -= 2.0 * math.pi
        elif delta < -math.pi:
            offset += 2.0 * math.pi
        out[i] = raw[i] + offset
    return out


def interval_params(samples, thetas, track_width, k):
    """The plan's own (v, omega) profile data for interval k -> k+1.

    v(s) linear and theta(s) quadratic, exactly the KI-026 collocation
    (overboard/core/optimizer.py:645-667).
    """
    s0, s1 = samples[k], samples[k + 1]
    dt = float(s1["t"]) - float(s0["t"])
    v1 = 0.5 * (float(s0["vl"]) + float(s0["vr"]))
    v2 = 0.5 * (float(s1["vl"]) + float(s1["vr"]))
    w1 = (float(s0["vr"]) - float(s0["vl"])) / track_width
    w2 = (float(s1["vr"]) - float(s1["vl"])) / track_width
    th0 = float(thetas[k])
    th1 = float(thetas[k + 1])
    dh = (th1 - th0 + math.pi) % (2.0 * math.pi) - math.pi
    return {
        "dt": dt,
        "v1": v1,
        "v2": v2,
        "w1": w1,
        "w2": w2,
        "dh": dh,
        "x0": float(s0["x"]),
        "y0": float(s0["y"]),
        "x1": float(s1["x"]),
        "y1": float(s1["y"]),
        "th0": th0,
        "th1": th1,
    }


def smooth_pose(par, s: float) -> Tuple[float, float, float]:
    """The heading-integrated plan curve (KI-026) at offset s in [0, dt].

    Position is the four-node Gauss-Legendre integral of
    v(sigma)(cos theta(sigma), sin theta(sigma)) over [0, s] - the same rule
    the collocation constrains the samples with, so at s = dt this returns the
    next planned sample exactly.
    """
    dt = par["dt"]
    v1, v2 = par["v1"], par["v2"]
    w1 = par["w1"]
    d_w = par["w2"] - par["w1"]
    th0 = par["th0"]
    dx = 0.0
    dy = 0.0
    for xi, weight in _GAUSS_LEGENDRE_4:
        sig = 0.5 * s * (1.0 + xi)
        w_s = 0.5 * s * weight
        v_s = v1 + (v2 - v1) * sig / dt
        th_s = th0 + w1 * sig + d_w * sig * sig / (2.0 * dt)
        dx += w_s * v_s * math.cos(th_s)
        dy += w_s * v_s * math.sin(th_s)
    th = th0 + w1 * s + d_w * s * s / (2.0 * dt)
    return par["x0"] + dx, par["y0"] + dy, th


def chord_pose(par, s: float) -> Tuple[float, float, float]:
    """The chorded reference (position chords, KI-027 linear heading) at s."""
    dt = par["dt"]
    frac = (s / dt) if dt > 0.0 else 0.0
    return (
        par["x0"] + (par["x1"] - par["x0"]) * frac,
        par["y0"] + (par["y1"] - par["y0"]) * frac,
        par["th0"] + par["dh"] * frac,
    )


# ---------------------------------------------------------------------------
# The rollout (independent float64 mirror of the gate's plant propagation)
# ---------------------------------------------------------------------------

def closed_loop_rollout(samples, thetas, track_width, tau_s, accel_limit,
                        control_dt=CONTROL_DT, num_substeps=PLANT_SUBSTEPS):
    """Closed-loop rollout on the plant's own substep grid.

    Mirrors CompiledReachabilityEngine._interval_step's controller-outside /
    plant-inside branch for the twin-active five-state ``mean_diff`` plant
    (DigitalTwinOpenLoop): the controller runs at the belief pose once per
    ``control_dt`` tick against the KI-027 reference (position chord, linear
    heading, half-step-lead speed, ``ref_kr`` from the interval's mean speed),
    the held wheel command is integrated in ``num_substeps`` Euler substeps
    with the wheel lag ``tau_s`` and the tanh acceleration limit, and the
    belief dead-reckons from the wheel-speed states (theta first, then
    position - the engine's order).

    Returns ``(times, xs, ys, ths, ends)`` with ``ends[k]`` the grid index of
    sample k's instant (so the grid contains every gate measurement point).
    """
    t_start = float(samples[0]["t"])
    x = float(samples[0]["x"])
    y = float(samples[0]["y"])
    th = float(thetas[0])
    v_mean = 0.0
    v_diff = 0.0
    bx, by, bth = x, y, th

    times = [t_start]
    xs = [x]
    ys = [y]
    ths = [th]
    ends = [0]

    for k in range(len(samples) - 1):
        par = interval_params(samples, thetas, track_width, k)
        dt = par["dt"]
        t_k = float(samples[k]["t"])
        v0, v1 = par["v1"], par["v2"]
        v_avg = 0.5 * (v0 + v1)
        dh = par["dh"]
        dt_safe = max(dt, 1e-9)
        ref_omega = dh / dt_safe
        ref_kr = math.sqrt(ref_omega * ref_omega
                           + RAMSETE_B * v_avg * v_avg + 1e-16)
        t_rel = 0.0
        while t_rel < dt - 1e-12:
            dt_ctrl = min(control_dt, dt - t_rel)
            if dt_ctrl <= 1e-12:
                break
            frac = min(t_rel, dt_safe) / dt_safe
            ref_x = par["x0"] + (par["x1"] - par["x0"]) * frac
            ref_y = par["y0"] + (par["y1"] - par["y0"]) * frac
            ref_h = par["th0"] + dh * frac
            t_mid = t_rel + 0.5 * control_dt
            frac_mid = min(max(t_mid / dt_safe, 0.0), 1.0)
            ref_v = v0 + frac_mid * (v1 - v0)

            ex, ey = body_frame_error(ref_x, ref_y, bx, by, bth)
            etheta = angle_diff(ref_h, bth)
            k1 = 2.0 * RAMSETE_ZETA * ref_kr
            v_cmd = ref_v * math.cos(etheta) + k1 * ex
            omega_cmd = (ref_omega
                         + RAMSETE_B * ref_v * _sinc_poly(etheta) * ey
                         + k1 * etheta)
            vl_cmd = v_cmd - (omega_cmd * track_width) / 2.0
            vr_cmd = v_cmd + (omega_cmd * track_width) / 2.0
            v_mean_target = 0.5 * (vl_cmd + vr_cmd)
            v_diff_target = vr_cmd - vl_cmd

            sub_dt = dt_ctrl / float(num_substeps)
            for j in range(num_substeps):
                odo_v = v_mean
                odo_w = v_diff / track_width
                bth = bth + odo_w * sub_dt
                bx = bx + odo_v * math.cos(bth) * sub_dt
                by = by + odo_v * math.sin(bth) * sub_dt

                d_mean_raw = (v_mean_target - v_mean) / tau_s
                d_diff_raw = (v_diff_target - v_diff) / tau_s
                if accel_limit is not None:
                    dvl = accel_limit * math.tanh(
                        (d_mean_raw - 0.5 * d_diff_raw) / accel_limit)
                    dvr = accel_limit * math.tanh(
                        (d_mean_raw + 0.5 * d_diff_raw) / accel_limit)
                    d_mean = 0.5 * (dvl + dvr)
                    d_diff = dvr - dvl
                else:
                    d_mean, d_diff = d_mean_raw, d_diff_raw

                dx = v_mean * math.cos(th)
                dy = v_mean * math.sin(th)
                dth = v_diff / track_width
                th = th + dth * sub_dt
                x = x + dx * sub_dt
                y = y + dy * sub_dt
                v_mean = v_mean + d_mean * sub_dt
                v_diff = v_diff + d_diff * sub_dt

                times.append(t_k + t_rel + (j + 1) * sub_dt)
                xs.append(x)
                ys.append(y)
                ths.append(th)
            t_rel += dt_ctrl
        ends.append(len(times) - 1)

    return (np.array(times), np.array(xs), np.array(ys), np.array(ths),
            ends)


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------

def q15_bound_terms(par) -> Tuple[float, float, float]:
    """Q15's physical discretisation family (optimizer.py:565-585)."""
    dt = par["dt"]
    v_mid = 0.5 * (par["v1"] + par["v2"])
    w_mid = 0.5 * (par["w1"] + par["w2"])
    sagitta = 0.125 * v_mid * w_mid * (dt ** 2)
    parab = 0.125 * v_mid * (par["w2"] - par["w1"]) * (dt ** 2)
    hold = 0.5 * (par["v2"] - par["v1"]) * CONTROL_DT
    return sagitta, parab, hold


def decompose_candidate(samples, track_width, tau_s, accel_limit,
                        dense_points=65) -> Dict[str, Any]:
    """The Q18 decomposition for one solved candidate.

    Everything is measured on one rollout (``closed_loop_rollout``). The
    three quantities are curve distances over the rollout's substep grid,
    plus their values restricted to the gate's own measurement instants (the
    sample times).
    """
    thetas = unroll_headings(samples)
    t_grid, x_grid, y_grid, th_grid, ends = closed_loop_rollout(
        samples, thetas, track_width, tau_s, accel_limit)

    n_grid = len(t_grid)
    sx = np.zeros(n_grid)
    sy = np.zeros(n_grid)
    cx = np.zeros(n_grid)
    cy = np.zeros(n_grid)
    # Map each grid point to its interval: the grid is built interval by
    # interval, so the interval index is non-decreasing and known by count.
    k_of = np.zeros(n_grid, dtype=int)
    gi = 0
    for k in range(len(samples) - 1):
        nxt = ends[k + 1]
        k_of[gi:nxt + 1] = k
        gi = nxt
    for i in range(n_grid):
        par = interval_params(samples, thetas, track_width, int(k_of[i]))
        s_off = t_grid[i] - float(samples[int(k_of[i])]["t"])
        s_off = min(max(s_off, 0.0), par["dt"])
        px, py, _ = smooth_pose(par, s_off)
        qx, qy, _ = chord_pose(par, s_off)
        sx[i], sy[i] = px, py
        cx[i], cy[i] = qx, qy

    r_dx = x_grid - sx
    r_dy = y_grid - sy
    s_dx = sx - cx
    s_dy = sy - cy
    c_dx = x_grid - cx
    c_dy = y_grid - cy
    d_rs = np.hypot(r_dx, r_dy)
    d_sc = np.hypot(s_dx, s_dy)
    d_rc = np.hypot(c_dx, c_dy)

    sample_idx = np.array(ends, dtype=int)
    q_series = [float(d_rs[i]) for i in sample_idx[1:]]
    sc_at_samples = [float(d_sc[i]) for i in sample_idx]
    rc_at_samples = [float(d_rc[i]) for i in sample_idx[1:]]

    q_reported_rollout = max(q_series) if q_series else 0.0
    q_sagitta = float(np.max(d_sc))
    q_control = float(np.max(d_rc))

    # The geometric chord bound per interval (Arm 2) on a dense grid, plus
    # the Q15 analytic family for cross-reference.
    bounds = []
    q15_bounds = []
    for k in range(len(samples) - 1):
        par = interval_params(samples, thetas, track_width, k)
        worst = 0.0
        for j in range(dense_points):
            s = par["dt"] * j / (dense_points - 1)
            px, py, _ = smooth_pose(par, s)
            qx, qy, _ = chord_pose(par, s)
            worst = max(worst, math.hypot(px - qx, py - qy))
        bounds.append(worst)
        sag, parb, hold = q15_bound_terms(par)
        q15_bounds.append(math.sqrt(sag * sag + parb * parb
                                    + (0.5 * hold) ** 2))

    chords = [math.hypot(float(samples[k + 1]["x"]) - float(samples[k]["x"]),
                         float(samples[k + 1]["y"]) - float(samples[k]["y"]))
              for k in range(len(samples) - 1)]
    dts = [float(samples[k + 1]["t"]) - float(samples[k]["t"])
           for k in range(len(samples) - 1)]

    def _share(a, b):
        total = a + b
        return (a / total) if total > 0 else 0.0

    return {
        "q_reported_rollout_m": _num(q_reported_rollout),
        "q_series_m": [_num(v) for v in q_series],
        "q_sagitta_m": _num(q_sagitta),
        "q_control_m": _num(q_control),
        "q_sagitta_samples_m": _num(max(sc_at_samples) if sc_at_samples else 0.0),
        "q_control_samples_m": _num(max(rc_at_samples) if rc_at_samples else 0.0),
        "composition_sum_m": _num(q_sagitta + q_control),
        "composition_quad_m": _num(math.hypot(q_sagitta, q_control)),
        "composition_max_m": _num(max(q_sagitta, q_control)),
        "share_shape": _num(_share(q_sagitta, q_control), digits=4),
        "share_control": _num(_share(q_control, q_sagitta), digits=4),
        "bound_series_m": [_num(v) for v in bounds],
        "q15_bound_series_m": [_num(v) for v in q15_bounds],
        "h_m": _num(float(np.mean(chords)) if chords else 0.0),
        "h_max_m": _num(max(chords) if chords else 0.0),
        "h_dt_s": _num(float(np.mean(dts)) if dts else 0.0),
    }


# ---------------------------------------------------------------------------
# Step 0(b): the synthetic self-consistent arc control
# ---------------------------------------------------------------------------

def arc_control(radius=0.5, omega=1.0, n_intervals=24, track_width=0.1,
                tau_s=MIN_TAU_S, accel_limit=7.8) -> Dict[str, Any]:
    """Analytic circle plan, heading = tangent; every component must read ~0.

    Two rollouts are measured on it: the analytic circle itself (the
    self-consistent rollout - the instrument's zero check) and the closed
    loop (the RQ-A F3-style sanity number).
    """
    v = omega * radius
    dt = 1.0 / omega / n_intervals * (math.pi / 2.0)  # 90 degrees of arc
    samples = []
    for k in range(n_intervals + 1):
        t = k * dt
        samples.append({
            "t": t,
            "x": radius * math.sin(omega * t),
            "y": radius * (1.0 - math.cos(omega * t)),
            "heading": omega * t,
            "vl": v - omega * track_width / 2.0,
            "vr": v + omega * track_width / 2.0,
        })
    thetas = unroll_headings(samples)

    # -- the self-consistent rollout: the analytic circle itself -----------
    # The instrument's zero check is at the GATE's measurement instants (the
    # sample times), where the smooth plan and the chorded reference share
    # the planned sample point: all three components must read ~0 there. The
    # mid-interval curve distances are the chord bound (Arm 2), not
    # components of the gate's quantity - they are checked against the
    # analytic sagitta below.
    max_rs = max_sc = max_rc = 0.0
    max_plan_hit = 0.0
    worst_sc_interval = 0.0
    for k in range(len(samples) - 1):
        par = interval_params(samples, thetas, track_width, k)
        for j in range(33):
            s = par["dt"] * j / 32.0
            t = float(samples[k]["t"]) + s
            ax = radius * math.sin(omega * t)
            ay = radius * (1.0 - math.cos(omega * t))
            px, py, _ = smooth_pose(par, s)
            qx, qy, _ = chord_pose(par, s)
            worst_sc_interval = max(worst_sc_interval,
                                    math.hypot(px - qx, py - qy))
        for s in (0.0, par["dt"]):
            t = float(samples[k]["t"]) + s
            ax = radius * math.sin(omega * t)
            ay = radius * (1.0 - math.cos(omega * t))
            px, py, _ = smooth_pose(par, s)
            qx, qy, _ = chord_pose(par, s)
            max_rs = max(max_rs, math.hypot(ax - px, ay - py))
            max_sc = max(max_sc, math.hypot(px - qx, py - qy))
            max_rc = max(max_rc, math.hypot(ax - qx, ay - qy))
        # the plan must hit its own samples (KI-026) and the chord its ends
        px, py, _ = smooth_pose(par, par["dt"])
        max_plan_hit = max(max_plan_hit,
                           math.hypot(px - float(samples[k + 1]["x"]),
                                      py - float(samples[k + 1]["y"])))

    # The analytic sagitta family: at the interval midpoint the arc-to-chord
    # distance of a circular arc is |arc(d/2) - chord(d/2)| in closed form.
    par0 = interval_params(samples, thetas, track_width, 0)
    d_ang = omega * par0["dt"]
    mid_s = 0.5 * par0["dt"]
    arc_x = radius * math.sin(d_ang / 2.0)
    arc_y = radius * (1.0 - math.cos(d_ang / 2.0))
    ch_x = 0.5 * radius * math.sin(d_ang)
    ch_y = 0.5 * radius * (1.0 - math.cos(d_ang))
    analytic_mid = math.hypot(arc_x - ch_x, arc_y - ch_y)
    px, py, _ = smooth_pose(par0, mid_s)
    qx, qy, _ = chord_pose(par0, mid_s)
    instrument_mid = math.hypot(px - qx, py - qy)

    # -- the closed-loop rollout on the same samples (sanity number) ------
    decomp = decompose_candidate(samples, track_width, tau_s, accel_limit)
    return {
        "radius_m": radius,
        "omega_rad_s": omega,
        "dt_s": dt,
        "samples": len(samples),
        "self_consistent": {
            "max_plan_hits_samples_m": max_plan_hit,
            "at_samples_rs_m": max_rs,
            "at_samples_sc_m": max_sc,
            "at_samples_rc_m": max_rc,
            "interval_chord_bound_m": worst_sc_interval,
        },
        "analytic_midpoint_chord_m": analytic_mid,
        "instrument_midpoint_chord_m": instrument_mid,
        "closed_loop_q_reported_m": decomp["q_reported_rollout_m"],
        "closed_loop_q_sagitta_m": decomp["q_sagitta_m"],
        "closed_loop_q_control_m": decomp["q_control_m"],
        "closed_loop_q_sagitta_samples_m": decomp["q_sagitta_samples_m"],
        "closed_loop_q_control_samples_m": decomp["q_control_samples_m"],
    }


def arc_control_self_check(result) -> None:
    """Assert-based self-check of the decomposition instrument (Step 0b)."""
    zero = result["self_consistent"]
    # At the gate's instants a self-consistent arc + self-consistent rollout
    # reads zero for every decomposition component: this is the instrument's
    # zero, and a misaligned chord/plan/rollout reads non-zero here.
    assert zero["max_plan_hits_samples_m"] < 1e-9, zero
    assert zero["at_samples_rs_m"] < 1e-9, zero
    assert zero["at_samples_sc_m"] < 1e-9, zero
    assert zero["at_samples_rc_m"] < 1e-9, zero
    # The chord bound is the arc's real sagitta: the instrument must
    # reproduce the closed form at the interval midpoint.
    assert abs(result["analytic_midpoint_chord_m"]
               - result["instrument_midpoint_chord_m"]) < 1e-12, result
    assert abs(zero["interval_chord_bound_m"]
               - result["analytic_midpoint_chord_m"]) < 1e-6, result
    # The closed-loop number is a model diagnostic, not an instrument check:
    # the twin plant starts at zero wheel speed, so a plan that starts at full
    # speed carries a rest-to-speed transient into the gate's quantity. Only
    # gross instrument breakage is asserted here.
    assert result["closed_loop_q_reported_m"] < 0.05, result


# ---------------------------------------------------------------------------
# Measurement driver (the m3_matrix candidate loop, plus the decomposition)
# ---------------------------------------------------------------------------

def load_mission(path: Path):
    wps, events, thresholds, json_stops = cli._load_waypoints(str(path))
    stops = sorted(set(json_stops))
    return wps, events, thresholds, stops


def derive_budget(solved):
    """The mission's measured scoring budget (the M1 rule, m3_matrix)."""
    def duration(entry):
        _idx, _spec, samples, stats, _wall = entry
        if samples and stats.get("converged"):
            return _num(samples[-1]["t"], digits=None)
        return None

    baseline = [d for d in (duration(e) for e in solved[:3]) if d is not None]
    if baseline:
        return min(baseline), "baseline"
    any_arm = [d for d in (duration(e) for e in solved) if d is not None]
    if any_arm:
        return min(any_arm), "fallback"
    return None, "none"


def _is_mission_file(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, list) or (isinstance(payload, dict)
                                         and "waypoints" in payload)


def run_mission(mission: Path, args, robot_cfg, digital_twin) -> dict:
    wps, events, thresholds, stops = load_mission(mission)
    planner = UnifiedPathEngine(robot_cfg)
    mesh = planner.compute_invariant_mesh_samples(wps)
    # Arm 3's lever: the per-segment budget is the driver's to scale
    # (get_segment_sample_counts uses a list as-is).
    mesh_scaled = [max(3, int(args.samples_mult * m)) for m in mesh]
    specs = planner.candidate_specs(count=args.candidates)

    started = time.time()
    solved = []
    for idx, spec in enumerate(specs):
        guess = planner.generate_seed(wps, mesh_scaled, spec["guess_style"],
                                      stops)
        t0 = time.time()
        try:
            samples, stats = TrajectoryOptimizer(robot_cfg).solve(
                wps,
                num_samples_per_segment=mesh_scaled,
                accuracy_weight=spec["accuracy_weight"],
                stop_waypoint_indices=stops,
                waypoint_events=events,
                initial_guess=guess,
                per_segment_dt=True,
                cost_metric="discretization",
                epigraph_constraints=True,
                verbose=False,
            )
        except Exception as exc:  # a solve that raises is a failed candidate
            samples, stats = [], {"converged": False, "error": str(exc)}
        solved.append((idx, spec, samples, stats, time.time() - t0))

    t_ref, budget_source = derive_budget(solved)
    score_ctx = cli._ScoreContext(
        robot_cfg=robot_cfg,
        samples=mesh_scaled,
        stop_indices=stops,
        waypoint_thresholds=thresholds,
        validate=True,
        use_validation=True,
        digital_twin=digital_twin,
        propagation=args.propagation,
        control_dt=0.02,
        quiet=True,
        target_time_s=t_ref,
        max_time_s=(2.0 * t_ref) if t_ref is not None else None,
    )

    # The product's own per-sample tracking errors, captured from the very
    # validation run that sets tracking_m (no second scan).
    validator = cli._immrax_validator(score_ctx)
    captured: Dict[str, Any] = {}
    original_validate = validator.validate_trajectory

    def _capture(*a, **k):
        report = original_validate(*a, **k)
        captured["report"] = report
        return report

    validator.validate_trajectory = _capture
    track_width = float(validator.cfg.track_width)
    plant = getattr(validator, "plant", None)
    twin_active = bool(getattr(validator, "_digital_twin_active", False))
    wheel_state = getattr(plant, "wheel_state", "mean_diff")
    tau_s = float(getattr(plant, "tau_s", MIN_TAU_S))
    accel_limit = getattr(plant, "accel_limit", None)
    if not (twin_active and getattr(plant, "xlen", 0) == 5
            and wheel_state == "mean_diff"):
        raise NotImplementedError(
            "the Q18 rollout mirror covers the twin-active mean_diff plant "
            f"(got twin_active={twin_active}, xlen={getattr(plant, 'xlen', 0)}, "
            f"wheel_state={wheel_state!r})")

    attempts = []
    candidates = []
    for idx, spec, samples, stats, wall in solved:
        attempt = cli._record_candidate(spec, (samples, stats), idx,
                                        len(specs), [], quiet=True)
        cli._score_attempt(attempt, score_ctx)
        attempts.append(attempt)
        entry = {
            "index": idx + 1,
            "style": spec["guess_style"],
            "converged": bool(stats.get("converged", False)),
            "infeasible": not bool(stats.get("converged", False)),
            "duration_s": _num(samples[-1]["t"], digits=6) if samples else None,
            "iterations": int(stats.get("iter_count") or 0),
            "wall_s": round(wall, 2),
            "ok": bool(attempt.get("ok", False)),
            "envelope_cm": _num(float(attempt.get("tube", 0.0)) * 100.0),
            "tracking_cm": _num(float(attempt.get("tracking_m", 0.0)) * 100.0),
            "reasons": list(attempt.get("reasons", [])),
            "score": _num(attempt.get("score"), digits=2),
            "eligibility": bool(cli._attempt_eligible(attempt, None)),
            "error": stats.get("error"),
        }
        if samples and stats.get("converged"):
            decomp = decompose_candidate(samples, track_width, tau_s,
                                         accel_limit)
            entry.update(decomp)
            # Step 0(c) instrument cross-check: the mirror's per-sample
            # errors against the product's own, from the same run.
            report = captured.get("report") or {}
            envelope = (report.get("reachability") or {}).get("envelope") or []
            product_series = [float(e.get("tracking_error_m", 0.0))
                              for e in envelope[1:]]
            mine = decomp["q_series_m"]
            if product_series and len(product_series) == len(mine):
                diffs = [abs(a - b) for a, b in zip(mine, product_series)]
                entry["rollout_series_maxdiff_m"] = _num(max(diffs))
            else:
                entry["rollout_series_maxdiff_m"] = None
            entry["product_series_m"] = [_num(v) for v in product_series]
            entry["q_reported_m"] = _num(
                float(attempt.get("tracking_m", 0.0)), digits=6)
        candidates.append(entry)

    passing = [a for a in attempts if a.get("ok")]
    return {
        "mission": mission.stem,
        "mission_path": str(mission),
        "samples_mult": args.samples_mult,
        "propagation": args.propagation,
        "candidates": args.candidates,
        "mesh": list(mesh),
        "mesh_scaled": list(mesh_scaled),
        "target_time_s": _num(t_ref, digits=None),
        "max_time_s": _num(2.0 * t_ref, digits=None) if t_ref is not None else None,
        "budget_source": budget_source,
        "passed": bool(passing),
        "candidates_total": len(candidates),
        "candidates_passing": len(passing),
        "infeasible": sum(1 for c in candidates if c["infeasible"]),
        "wall_s": round(time.time() - started, 1),
        "results": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("missions", nargs="*",
                        help="Mission files or directories of them.")
    parser.add_argument("--out", help="Result JSONL (resumed, not replaced).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--propagation", choices=("plant", "tick"),
                        default="plant")
    parser.add_argument("--samples-mult", type=int, default=1,
                        help="Arm 3: multiply the per-segment sample budget.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--arc-control", action="store_true",
                        help="Step 0(b): run the synthetic arc control and exit.")
    args = parser.parse_args()

    if args.arc_control:
        result = arc_control()
        arc_control_self_check(result)
        print(json.dumps(result, indent=2))
        print("arc control self-check: OK")
        return 0

    if not args.missions or not args.out:
        parser.error("missions and --out are required (or use --arc-control)")

    missions = []
    for entry in args.missions:
        path = Path(entry)
        if path.is_dir():
            for file in sorted(path.glob("*.json")):
                if _is_mission_file(file):
                    missions.append(file)
        elif _is_mission_file(path):
            missions.append(path)
    if args.limit:
        missions = missions[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["mission"]
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    todo = [m for m in missions if m.stem not in done]
    print(f"{len(missions)} missions, {len(todo)} to run | "
          f"prop={args.propagation} candidates={args.candidates} "
          f"samples_mult={args.samples_mult}", flush=True)

    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    with out_path.open("a", encoding="utf-8") as handle:
        for number, mission in enumerate(todo, 1):
            robot_cfg = RobotConfig(config_data)
            digital_twin = cli._config_digital_twin(config_data)
            record = run_mission(mission, args, robot_cfg, digital_twin)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            worst = max((c.get("rollout_series_maxdiff_m") or 0.0)
                        for c in record["results"] if c.get("q_series_m"))
            print(f"[{number}/{len(todo)}] {mission.stem} "
                  f"passed={record['passed']} infeasible={record['infeasible']} "
                  f"crosscheck={worst}m wall={record['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
