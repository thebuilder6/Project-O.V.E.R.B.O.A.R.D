from typing import List, Dict, Tuple, Any, Optional
import json
import math
import numpy as np
from robot_model import RobotConfig, DifferentialDriveModel, wheel_normal_forces, CG_HEIGHT_M


# Default time-equivalent weights (in seconds) for the 5-part FLL path fitness
# score. Prioritizes precision (repeatability at stops/end) over raw accuracy
# and raw speed:
#   1. stop_precision:    200.0 s/m  (1 mm stop/end spread = +0.20 s)
#   2. slip_risk:           8.0 s    (quadratic penalty on utilization > 70%)
#   3. rms_force:           0.5 s    (normalized RMS wheel-force tie-breaker)
#   4. tracking_accuracy:  80.0 s/m  (1 mm RMS + stop tracking error = +0.08 s)
#   5. time:                1.0 s/s  (1.00 s per second of travel time)
DEFAULT_FITNESS_WEIGHTS: Dict[str, float] = {
    "time": 1.0,
    "stop_precision": 200.0,
    "slip_risk": 8.0,
    "rms_force": 0.5,
    "tracking_accuracy": 80.0,
}
SLIP_RISK_DEADZONE: float = 0.70


def forward_integrate(samples: List[Dict[str, Any]], robot_cfg: RobotConfig, fine_dt: float = 0.001) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Forward-integrate differential-drive kinematics from the optimized
    wheel-velocity profile using a fine timestep.

    Returns a list of dicts with keys {t, x, y, heading} and a dict of
    max / RMS errors versus the planned trajectory.
    """
    if not samples:
        return [], {}

    model = DifferentialDriveModel(robot_cfg)
    t_arr = np.array([s["t"] for s in samples])
    vl_arr = np.array([s["vl"] for s in samples])
    vr_arr = np.array([s["vr"] for s in samples])

    # A failed solve can leave a negative or non-monotone time vector. Refuse to
    # integrate it instead of silently producing a meaningless (or crashing) result.
    _inf_errors = {
        "max_pos_error_m": float("inf"),
        "rms_pos_error_m": float("inf"),
        "max_heading_error_rad": float("inf"),
        "final_pos_error_m": float("inf"),
        "final_heading_error_rad": float("inf"),
    }
    if (t_arr.size < 2 or not np.all(np.isfinite(t_arr))
            or np.any(np.diff(t_arr) <= 0) or t_arr[-1] <= t_arr[0]):
        return [], _inf_errors

    # Interpolators for wheel speeds
    def interp_v(t_query):
        if t_query <= t_arr[0]:
            return vl_arr[0], vr_arr[0]
        if t_query >= t_arr[-1]:
            return vl_arr[-1], vr_arr[-1]
        idx = int(np.searchsorted(t_arr, t_query)) - 1
        idx = max(0, min(idx, len(t_arr) - 2))
        t0, t1 = t_arr[idx], t_arr[idx + 1]
        frac = (t_query - t0) / (t1 - t0)
        vl = vl_arr[idx] + (vl_arr[idx + 1] - vl_arr[idx]) * frac
        vr = vr_arr[idx] + (vr_arr[idx + 1] - vr_arr[idx]) * frac
        return vl, vr

    total_t = t_arr[-1]
    num_steps = int(np.ceil(total_t / fine_dt))
    x, y, theta = samples[0]["x"], samples[0]["y"], samples[0]["heading"]

    integrated = []
    for step in range(num_steps + 1):
        t = min(step * fine_dt, total_t)
        vl, vr = interp_v(t)
        v = (vl + vr) / 2.0
        omega = (vr - vl) / robot_cfg.track_width

        # Record the state at time t before advancing to t + dt
        integrated.append({"t": float(t), "x": float(x), "y": float(y), "heading": float(theta)})

        # RK4 integration step
        if step < num_steps:
            dt = min(fine_dt, total_t - t)
            k1x = v * np.cos(theta)
            k1y = v * np.sin(theta)
            k1t = omega

            x2 = x + k1x * dt * 0.5
            y2 = y + k1y * dt * 0.5
            theta2 = theta + k1t * dt * 0.5
            v2 = v  # vl/vr assumed constant over tiny dt
            k2x = v2 * np.cos(theta2)
            k2y = v2 * np.sin(theta2)
            k2t = omega

            x3 = x + k2x * dt * 0.5
            y3 = y + k2y * dt * 0.5
            theta3 = theta + k2t * dt * 0.5
            k3x = v2 * np.cos(theta3)
            k3y = v2 * np.sin(theta3)
            k3t = omega

            x4 = x + k3x * dt
            y4 = y + k3y * dt
            theta4 = theta + k3t * dt
            k4x = v2 * np.cos(theta4)
            k4y = v2 * np.sin(theta4)
            k4t = omega

            x += (k1x + 2 * k2x + 2 * k3x + k4x) * dt / 6.0
            y += (k1y + 2 * k2y + 2 * k3y + k4y) * dt / 6.0
            theta += (k1t + 2 * k2t + 2 * k3t + k4t) * dt / 6.0

    # Error vs planned samples
    def integrated_pose(t_query):
        # integrated[k] is the pose at t = k * fine_dt; use the nearest step
        idx = int(min(max(round(t_query / fine_dt), 0), num_steps))
        s = integrated[idx]
        return s["x"], s["y"], s["heading"]

    planned = np.array([[s["x"], s["y"], s["heading"]] for s in samples])
    integrated_at_t = np.array([integrated_pose(s["t"]) for s in samples])

    dx = planned[:, 0] - integrated_at_t[:, 0]
    dy = planned[:, 1] - integrated_at_t[:, 1]
    dtheta = (planned[:, 2] - integrated_at_t[:, 2] + np.pi) % (2 * np.pi) - np.pi

    pos_err = np.hypot(dx, dy)
    errors = {
        "max_pos_error_m": float(np.max(pos_err)),
        "rms_pos_error_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "max_heading_error_rad": float(np.max(np.abs(dtheta))),
        "final_pos_error_m": float(pos_err[-1]),
        "final_heading_error_rad": float(abs(dtheta[-1])),
    }
    return integrated, errors


def audit_constraints(samples: List[Dict[str, Any]], robot_cfg: RobotConfig, apply_headroom: bool = True) -> Dict[str, Any]:
    """
    Re-evaluate motor and traction limits for each sample.
    Returns max violations and a list of violating sample indices with slip details.

    Args:
        samples: Trajectory samples
        robot_cfg: Robot configuration
        apply_headroom: If True, applies safety margin for real-world tracking
    """
    model = DifferentialDriveModel(robot_cfg)
    
    # Track max violations
    max_violations = {
        "left_motor_violation": 0.0,
        "right_motor_violation": 0.0,
        "left_wheel_slip": 0.0,
        "right_wheel_slip": 0.0,
        "traction_violation": 0.0
    }
    
    # Track violating samples with details
    violating_indices = []
    slip_points = []  # Detailed slip information
    
    left_normal_forces = []
    right_normal_forces = []
    max_wheel_speed = 0.0
    
    for i, s in enumerate(samples):
        violations = model.check_constraints(s["vl"], s["vr"], s["al"], s["ar"], apply_headroom)
        max_wheel_speed = max(max_wheel_speed, abs(s["vl"]), abs(s["vr"]))
        
        # Update max violations
        for key in max_violations:
            if key in violations and violations[key] > max_violations[key]:
                max_violations[key] = violations[key]
        
        # Check for any violation
        has_violation = any(violations[k] > 1e-6 for k in ["left_motor_violation", "right_motor_violation", "traction_violation"])
        has_slip = violations["left_wheel_slip"] > 1e-6 or violations["right_wheel_slip"] > 1e-6
        
        if has_violation:
            violating_indices.append(i)
            
        left_normal_forces.append(violations["left_normal_force"])
        right_normal_forces.append(violations["right_normal_force"])
        
        if has_slip:
            slip_points.append({
                "index": i,
                "time": s["t"],
                "x": s["x"],
                "y": s["y"],
                "heading": s["heading"],
                "left_wheel_slip_N": violations["left_wheel_slip"],
                "right_wheel_slip_N": violations["right_wheel_slip"],
                "left_normal_force_N": violations["left_normal_force"],
                "right_normal_force_N": violations["right_normal_force"],
                "vl": s["vl"],
                "vr": s["vr"]
            })
    
    # Use the smallest declared wheel radius so this matches the Immrax
    # validator's no-load speed bound. The gearing is applied the same way
    # the force model applies it: the motor turns `gearing` times per wheel
    # revolution, so the wheel's no-load speed is the motor's divided by it.
    wheel_speed_limit = robot_cfg.no_load_wheel_speed_worst_case()
    speed_violation = max(0.0, max_wheel_speed - wheel_speed_limit)

    audit = {
        "left_motor_force": float(max_violations["left_motor_violation"]),
        "right_motor_force": float(max_violations["right_motor_violation"]),
        "left_wheel_slip": float(max_violations["left_wheel_slip"]),
        "right_wheel_slip": float(max_violations["right_wheel_slip"]),
        "traction_total": float(max_violations["traction_violation"]),
        "max_wheel_speed_m_s": float(max_wheel_speed),
        "wheel_speed_limit_m_s": float(wheel_speed_limit),
        "speed_violation_m_s": float(speed_violation),
        "left_normal_force": float(np.mean(left_normal_forces)) if left_normal_forces else float((robot_cfg.mass * robot_cfg.g) / 2.0),
        "right_normal_force": float(np.mean(right_normal_forces)) if right_normal_forces else float((robot_cfg.mass * robot_cfg.g) / 2.0),
        "num_violating_samples": len(violating_indices),
        "violating_sample_indices": violating_indices,
        "num_slip_points": len(slip_points),
        "slip_points": slip_points
    }
    return audit


def compute_metrics(samples: List[Dict[str, Any]], robot_cfg: Optional[RobotConfig] = None) -> Dict[str, Any]:
    """
    Comprehensive trajectory metrics for quality analysis and benchmarking.
    """
    if not samples:
        return {}

    t = [s["t"] for s in samples]
    vl = [s["vl"] for s in samples]
    vr = [s["vr"] for s in samples]
    x = [s["x"] for s in samples]
    y = [s["y"] for s in samples]

    v_lin = [(l + r) / 2.0 for l, r in zip(vl, vr)]
    al = [s["al"] for s in samples]
    ar = [s["ar"] for s in samples]

    # Jerk: second difference of accelerations.
    #
    # al[i] is the AVERAGE acceleration over the interval [t_i, t_{i+1}], so
    # the jump between al[i] and al[i+1] happens over the spacing between
    # those two interval MIDPOINTS, which is 0.5*(dt_i + dt_{i+1}). Dividing
    # by dt_i alone is only correct when every interval is the same length;
    # with per-segment dt it understates the jerk at a segment boundary and
    # reports a different physical quantity than the uniform mode (KI-022).
    #
    # The last sample carries al=0 because it has no following interval, so
    # differencing it against the previous sample measures a jump that does not
    # exist. The solver stops at N-2 for the same reason, and the metric must
    # use the same term set, or a short final interval reports a huge fake
    # jerk.
    dt_vals = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    jerks = []
    for i in range(len(al) - 2):
        span = 0.5 * (dt_vals[i] + dt_vals[i + 1])
        if span > 1e-9:
            jerks.append((al[i + 1] - al[i]) / span)
            jerks.append((ar[i + 1] - ar[i]) / span)

    path_len = sum(
        np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        for i in range(len(x) - 1)
    )

    # Calculate additional metrics
    avg_linear_speed = np.mean([abs(v) for v in v_lin])

    # Force & traction utilization
    fl_util = []
    fr_util = []
    combined_util = []
    weighted_force_sq = 0.0
    total_dt = 0.0
    for i, s in enumerate(samples):
        fl_val = float(s.get("fl", 0.0))
        fr_val = float(s.get("fr", 0.0))
        if i < len(dt_vals) and dt_vals[i] > 0:
            weighted_force_sq += 0.5 * (fl_val ** 2 + fr_val ** 2) * dt_vals[i]
            total_dt += dt_vals[i]
        if robot_cfg:
            max_fl = robot_cfg.get_max_force_at_velocity(s["vl"])
            max_fr = robot_cfg.get_max_force_at_velocity(s["vr"])
            u_l = abs(fl_val) / max_fl if max_fl > 1e-6 else 0.0
            u_r = abs(fr_val) / max_fr if max_fr > 1e-6 else 0.0
            if max_fl > 1e-6:
                fl_util.append(u_l)
            if max_fr > 1e-6:
                fr_util.append(u_r)
            cof_min = min(robot_cfg.uncertainties["cof"])
            nl, nr = wheel_normal_forces(
                robot_cfg.mass, robot_cfg.g, CG_HEIGHT_M, robot_cfg.track_width, 0.25 * (s["vr"] * s["vr"] - s["vl"] * s["vl"]) / robot_cfg.track_width)
            tr_l = abs(fl_val) / max(cof_min * nl, 1e-6)
            tr_r = abs(fr_val) / max(cof_min * nr, 1e-6)
            combined_util.append(max(u_l, u_r, tr_l, tr_r))

    rms_wheel_force_n = math.sqrt(weighted_force_sq / total_dt) if total_dt > 1e-9 else 0.0
    stall_force = (
        max(robot_cfg.get_max_force_at_velocity_robust(0.0), 1e-6)
        if robot_cfg else 1.0
    )
    rms_force_util = rms_wheel_force_n / stall_force if robot_cfg else 0.0
    slip_excess = [max(0.0, rho - SLIP_RISK_DEADZONE) ** 2 for rho in combined_util]
    slip_risk_penalty = (
        0.5 * float(np.mean(slip_excess)) + 0.5 * float(np.max(slip_excess))
        if slip_excess else 0.0
    )

    # Tortuosity
    straight_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    tortuosity = path_len / straight_distance if straight_distance > 1e-6 else 1.0

    # Chattering
    vl_crossings = 0
    vr_crossings = 0
    for i in range(1, len(samples)):
        if samples[i]['vl'] * samples[i-1]['vl'] < 0:
            vl_crossings += 1
        if samples[i]['vr'] * samples[i-1]['vr'] < 0:
            vr_crossings += 1

    return {
        "total_time_s": float(t[-1]),
        "path_length_m": float(path_len),
        "max_linear_speed_m_s": float(max(abs(v) for v in v_lin)),
        "avg_linear_speed_m_s": float(avg_linear_speed),
        "max_wheel_speed_m_s": float(max(max(abs(v) for v in vl), max(abs(v) for v in vr))),
        "max_accel_m_s2": float(max(max(abs(a) for a in al), max(abs(a) for a in ar))),
        "max_jerk_m_s3": float(max(abs(j) for j in jerks)) if jerks else 0.0,
        "avg_jerk_m_s3": float(np.mean(np.abs(jerks))) if jerks else 0.0,
        "tortuosity": float(tortuosity),
        "velocity_chattering": int(vl_crossings + vr_crossings),
        "avg_force_utilization": float(np.mean(fl_util + fr_util)) if fl_util + fr_util else 0.0,
        "max_force_utilization": float(np.max(fl_util + fr_util)) if fl_util + fr_util else 0.0,
        "rms_wheel_force_n": float(rms_wheel_force_n),
        "rms_force_utilization": float(rms_force_util),
        "peak_traction_utilization": float(np.max(combined_util)) if combined_util else 0.0,
        "slip_risk_penalty": float(slip_risk_penalty),
    }


def compute_path_fitness(
    samples: List[Dict[str, Any]],
    robot_cfg: Optional[RobotConfig] = None,
    stop_sample_indices: Optional[List[int]] = None,
    immrax_report: Optional[Dict[str, Any]] = None,
    cpu_summary: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Evaluate the 5-part time-equivalent path fitness score (in seconds, lower is better).

    Prioritizes repeatability (precision) at stops and the final waypoint over
    raw path accuracy and raw travel speed:
      1. Stop & final-waypoint precision spread: sum of pose + lever-arm heading
         reachability spread sqrt(r_i^2 + (L_arm * dtheta_i)^2) at all stops and
         the final sample.
      2. Deviation / slip risk: quadratic penalty on motor & traction utilization
         ratio rho_k = max(|F|/F_motor, |F|/F_traction) above a 70% dead-zone.
      3. Minimal wheel force: time-weighted RMS wheel force normalized by stall force.
      4. Average & stop tracking error: 0.5 * RMS path tracking error + sum of
         predicted closed-loop tracking errors (pose + lever-arm heading) at all
         stops and the final sample.
      5. Travel time: total trajectory duration T (seconds).
    """
    if not samples:
        return {
            "fitness_score": float("inf"),
            "travel_time_s": float("inf"),
            "stop_precision_spread_m": float("inf"),
            "final_precision_spread_m": float("inf"),
            "slip_risk_penalty": float("inf"),
            "peak_utilization": float("inf"),
            "rms_wheel_force_n": float("inf"),
            "rms_force_utilization": float("inf"),
            "tracking_error_m": float("inf"),
            "rms_tracking_error_m": float("inf"),
            "stop_tracking_error_m": float("inf"),
        }

    w = dict(DEFAULT_FITNESS_WEIGHTS)
    if weights:
        w.update(weights)

    n = len(samples)
    lever_arm = float(getattr(robot_cfg, "lever_arm_m", 0.10)) if robot_cfg else 0.10
    metrics = (
        cpu_summary["metrics"]
        if cpu_summary and "metrics" in cpu_summary
        else compute_metrics(samples, robot_cfg)
    )
    travel_time_s = float(samples[-1]["t"] - samples[0]["t"]) if n >= 2 else 0.0

    # Checkpoint sample indices: all stop waypoints (excluding start) + final sample.
    raw_stops = [int(idx) for idx in (stop_sample_indices or []) if 0 < int(idx) < n]
    checkpoints = sorted(set(raw_stops + [n - 1]))

    # Envelopes from immrax_report or attached to samples.
    envelopes: List[Dict[str, Any]] = []
    if immrax_report:
        envelopes = list(immrax_report.get("reachability", {}).get("envelope", []) or [])
    if not envelopes:
        envelopes = [s.get("reachability_envelope") or {} for s in samples]

    def _sample_spread(idx: int) -> float:
        if idx < 0 or idx >= len(envelopes) or not envelopes[idx]:
            return 0.0
        env = envelopes[idx]
        if env.get("unknown"):
            return 0.05
        rx = 0.5 * max(0.0, float(env.get("x_max", 0.0)) - float(env.get("x_min", 0.0)))
        ry = 0.5 * max(0.0, float(env.get("y_max", 0.0)) - float(env.get("y_min", 0.0)))
        r_pos = math.hypot(rx, ry)
        if "theta_max" in env and "theta_min" in env:
            dtheta = 0.5 * max(0.0, float(env["theta_max"]) - float(env["theta_min"]))
        else:
            dtheta = 0.0
        return math.hypot(r_pos, lever_arm * dtheta)

    # 1. Stop & final-waypoint precision spread
    stop_spreads = [_sample_spread(idx) for idx in checkpoints]
    stop_precision_spread_m = float(sum(stop_spreads))
    final_precision_spread_m = float(_sample_spread(n - 1))

    # 2. Deviation / slip risk & 3. Minimal wheel force
    # Use worst-case robust motor force and worst-case traction limit.
    combined_util = []
    if robot_cfg:
        cof_min = min(robot_cfg.uncertainties["cof"])
        for s in samples:
            fl_val = abs(float(s.get("fl", 0.0)))
            fr_val = abs(float(s.get("fr", 0.0)))
            max_fl = max(robot_cfg.get_max_force_at_velocity_robust(s["vl"]), 1e-6)
            max_fr = max(robot_cfg.get_max_force_at_velocity_robust(s["vr"]), 1e-6)
            nl, nr = wheel_normal_forces(
                robot_cfg.mass, robot_cfg.g, CG_HEIGHT_M, robot_cfg.track_width, 0.25 * (s["vr"] * s["vr"] - s["vl"] * s["vl"]) / robot_cfg.track_width)
            tr_l = max(cof_min * nl, 1e-6)
            tr_r = max(cof_min * nr, 1e-6)
            combined_util.append(max(fl_val / max_fl, fr_val / max_fr, fl_val / tr_l, fr_val / tr_r))

    peak_utilization = float(max(combined_util)) if combined_util else float(metrics.get("peak_traction_utilization", 0.0))
    slip_excess = [max(0.0, rho - SLIP_RISK_DEADZONE) ** 2 for rho in combined_util]
    slip_risk_penalty = (
        0.5 * float(np.mean(slip_excess)) + 0.5 * float(np.max(slip_excess))
        if slip_excess else float(metrics.get("slip_risk_penalty", 0.0))
    )
    rms_wheel_force_n = float(metrics.get("rms_wheel_force_n", 0.0))
    rms_force_utilization = float(metrics.get("rms_force_utilization", 0.0))

    # 4. Average & stop tracking error (closed-loop midpoint from Immrax when available,
    # otherwise forward-integrated RK4 error from CPU summary).
    sample_track_errors = []
    for i in range(n):
        env = envelopes[i] if i < len(envelopes) and envelopes[i] else {}
        if "tracking_error_m" in env:
            e_pos = float(env["tracking_error_m"])
            e_theta = float(env.get("heading_error_rad", 0.0))
            sample_track_errors.append(math.hypot(e_pos, lever_arm * e_theta))

    if sample_track_errors:
        rms_tracking_error_m = float(math.sqrt(np.mean([e ** 2 for e in sample_track_errors])))
        stop_tracking_error_m = float(sum(sample_track_errors[idx] for idx in checkpoints if idx < len(sample_track_errors)))
    elif immrax_report:
        max_tr = float(immrax_report.get("max_tracking_error_m", 0.0))
        rms_tr = float(immrax_report.get("rms_tracking_error_m", max_tr))
        max_hd = float(immrax_report.get("max_heading_error_rad", 0.0))
        rms_tracking_error_m = math.hypot(rms_tr, lever_arm * max_hd)
        stop_tracking_error_m = rms_tracking_error_m * len(checkpoints)
    else:
        if cpu_summary and "errors" in cpu_summary:
            errs = cpu_summary["errors"]
        elif robot_cfg:
            _, errs = forward_integrate(samples, robot_cfg, fine_dt=0.001)
        else:
            errs = {}
        rms_tracking_error_m = float(errs.get("rms_pos_error_m", 0.0))
        final_pos = float(errs.get("final_pos_error_m", 0.0))
        final_hd = float(errs.get("final_heading_error_rad", 0.0))
        stop_tracking_error_m = math.hypot(final_pos, lever_arm * final_hd)

    tracking_error_m = 0.5 * rms_tracking_error_m + stop_tracking_error_m

    # 5. Total time-equivalent fitness score (seconds)
    fitness_score = (
        w["time"] * travel_time_s
        + w["stop_precision"] * stop_precision_spread_m
        + w["slip_risk"] * slip_risk_penalty
        + w["rms_force"] * rms_force_utilization
        + w["tracking_accuracy"] * tracking_error_m
    )

    return {
        "fitness_score": float(fitness_score),
        "travel_time_s": float(travel_time_s),
        "stop_precision_spread_m": float(stop_precision_spread_m),
        "final_precision_spread_m": float(final_precision_spread_m),
        "slip_risk_penalty": float(slip_risk_penalty),
        "peak_utilization": float(peak_utilization),
        "rms_wheel_force_n": float(rms_wheel_force_n),
        "rms_force_utilization": float(rms_force_utilization),
        "tracking_error_m": float(tracking_error_m),
        "rms_tracking_error_m": float(rms_tracking_error_m),
        "stop_tracking_error_m": float(stop_tracking_error_m),
    }


def check_final_waypoint(samples: List[Dict[str, Any]],
                         waypoint: Tuple[float, float, Optional[float]],
                         pos_tol: float = 0.02,
                         heading_tol: float = 0.02) -> Dict[str, Any]:
    """Second line of defence: the exported trajectory must reach its last waypoint."""
    if not samples or waypoint is None:
        return {"passed": True, "pos_error_m": 0.0, "heading_error_rad": 0.0}

    wx, wy, wh = waypoint
    last = samples[-1]
    pos_err = float(np.hypot(last["x"] - wx, last["y"] - wy))
    heading_err = 0.0
    if wh is not None:
        heading_err = float(abs((last["heading"] - wh + np.pi) % (2 * np.pi) - np.pi))
    return {
        "passed": pos_err <= pos_tol and heading_err <= heading_tol,
        "pos_error_m": pos_err,
        "heading_error_rad": heading_err,
    }


def validate_samples(samples: List[Dict[str, Any]], robot_cfg: RobotConfig,
                     apply_headroom: bool = True) -> Dict[str, Any]:
    """Run the same checks as validate_trajectory without writing or printing.

    main.py uses this before writing so a failing validation cannot leave an
    output file behind (KI-002).
    """
    metrics = compute_metrics(samples, robot_cfg)
    audit = audit_constraints(samples, robot_cfg, apply_headroom)
    integrated, errors = forward_integrate(samples, robot_cfg, fine_dt=0.001)

    ok = (
        errors["max_pos_error_m"] < 0.01
        and errors["final_pos_error_m"] < 0.01
        and audit["num_violating_samples"] == 0
        and audit["num_slip_points"] == 0
        and audit["speed_violation_m_s"] < 1e-6
    )
    return {
        "passed": bool(ok),
        "metrics": metrics,
        "audit": audit,
        "errors": errors,
        "integrated": integrated,
    }


def validate_trajectory(traj_file: str, config_file: str, apply_headroom: bool = True,
                        waypoints: Optional[List[Tuple[float, float, Optional[float]]]] = None
                        ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """
    CLI entry point: load a .traj file and a .json config, run validation,
    and print a human-readable report.

    Args:
        traj_file: Path to trajectory file
        config_file: Path to config file
        apply_headroom: If True, applies safety margin for real-world tracking
        waypoints: Optional waypoints; when given, the final pose is checked too
    """
    with open(traj_file, "r") as f:
        traj_data = json.load(f)
    with open(config_file, "r") as f:
        config_data = json.load(f)

    samples = traj_data["trajectory"]["samples"]
    robot_cfg = RobotConfig(config_data)

    print(f"\n=== Validation Report: {traj_file} ===")
    print(f"Samples: {len(samples)} | Config: {config_file}")
    if apply_headroom:
        print(f"Safety margins: torque_headroom={robot_cfg.torque_headroom:.2f}, speed_headroom={robot_cfg.speed_headroom:.2f}")

    # In-memory checks: the same ones main.py runs before writing (KI-002)
    summary = validate_samples(samples, robot_cfg, apply_headroom)
    metrics, audit, errors = summary["metrics"], summary["audit"], summary["errors"]

    print("\n-- Metrics --")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Constraint audit
    print("\n-- Constraint Audit --")
    print(f"  Violating samples: {audit['num_violating_samples']}")
    if audit["num_violating_samples"]:
        print(f"  Indices: {audit['violating_sample_indices']}")
    for label in ["left_motor_force", "right_motor_force", "traction_total"]:
        print(f"  Max {label} violation: {audit[label]:.6f} N")
    print(f"  Max wheel speed: {audit['max_wheel_speed_m_s']:.4f} m/s "
          f"(limit {audit['wheel_speed_limit_m_s']:.4f} m/s)")
    
    # Wheel slip detection
    print(f"\n-- Wheel Slip Detection --")
    print(f"  Slip points: {audit['num_slip_points']}")
    print(f"  Max left wheel slip: {audit['left_wheel_slip']:.6f} N")
    print(f"  Max right wheel slip: {audit['right_wheel_slip']:.6f} N")
    
    if audit['num_slip_points'] > 0:
        print(f"\n  Slip point details (first 5):")
        for i, sp in enumerate(audit['slip_points'][:5]):
            print(f"    [{i}] t={sp['time']:.3f}s at ({sp['x']:.3f}, {sp['y']:.3f})")
            if sp['left_wheel_slip_N'] > 1e-6:
                print(f"        Left wheel slip: {sp['left_wheel_slip_N']:.6f} N (normal: {sp['left_normal_force_N']:.2f} N)")
            if sp['right_wheel_slip_N'] > 1e-6:
                print(f"        Right wheel slip: {sp['right_wheel_slip_N']:.6f} N (normal: {sp['right_normal_force_N']:.2f} N)")
        if audit['num_slip_points'] > 5:
            print(f"    ... and {audit['num_slip_points'] - 5} more")

    # Forward integration
    print("\n-- Forward Integration (1 ms RK4) --")
    for k, v in errors.items():
        print(f"  {k}: {v:.6f}")

    # Third check: the exported trajectory must reach the last waypoint.
    waypoint_check = None
    if waypoints:
        waypoint_check = check_final_waypoint(samples, waypoints[-1])
        print("\n-- Waypoint Check --")
        print(f"  Final position error: {waypoint_check['pos_error_m']:.6f} m")
        if waypoints[-1][2] is not None:
            print(f"  Final heading error: {waypoint_check['heading_error_rad']:.6f} rad")

    ok = summary["passed"] and (waypoint_check is None or waypoint_check["passed"])

    print(f"\n{'PASS' if ok else 'WARN'} — trajectory {'is' if ok else 'may not be'} safe to run.")
    
    return metrics, audit, errors


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python validator.py <trajectory.traj> <config.json>")
        sys.exit(1)
    validate_trajectory(sys.argv[1], sys.argv[2])
