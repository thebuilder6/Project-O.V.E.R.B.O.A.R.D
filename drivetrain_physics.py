"""
Pure functional differential-drive robotics and drivetrain physics engine.

Provides unified, duck-typed mathematical primitives supporting:
  - Standard Python scalars (float, int)
  - NumPy arrays (np.ndarray)
  - JAX arrays (jnp.ndarray) and JIT compilation
  - CasADi symbolic expressions (ca.MX, ca.SX)
  - Immrax Interval arithmetic (immrax.Interval)

Centralizes motor torque-speed lines, load transfer, normal forces,
and friction/traction limits across CPU simulation, CasADi collocation,
and JAX/Immrax interval validation.
"""

import math
from typing import Any, Tuple, Dict, Optional, Sequence


# Center-of-mass height used by every weight-transfer calculation.
# Kept in one place so optimizer, CPU audit, JAX model, and Immrax agree.
CG_HEIGHT_M = 0.05


def _duck_abs(x: Any, smooth: bool = False) -> Any:
    """
    Duck-typed absolute value function.
    Supports float, np.ndarray, jnp.ndarray, ca.MX, and immrax.Interval.
    For CasADi expressions, supports smooth approximation: sqrt(x^2 + eps).
    """
    # CasADi MX/SX
    if hasattr(x, "type_name"):
        import casadi as ca
        return ca.sqrt(x**2 + 1e-8) if smooth else ca.fabs(x)

    # Immrax Interval
    if hasattr(x, "lower") and hasattr(x, "upper"):
        return x

    # Scalar, NumPy, JAX array
    return abs(x)


def _duck_clamp_zero(x: Any) -> Any:
    """
    Duck-typed max(0, x) clamping.
    Supports float, np.ndarray, jnp.ndarray, ca.MX, and immrax.Interval.
    """
    # CasADi MX/SX
    if hasattr(x, "type_name"):
        import casadi as ca
        return ca.fmax(0.0, x)

    # Immrax Interval
    if hasattr(x, "lower") and hasattr(x, "upper"):
        return x

    # NumPy array
    if hasattr(x, "ndim") and not hasattr(x, "devices"):
        import numpy as np
        return np.maximum(0.0, x)

    # JAX array
    if hasattr(x, "devices"):
        import jax.numpy as jnp
        return jnp.maximum(0.0, x)

    # Plain scalar
    return max(0.0, x)


def diff_drive_fk(vl: Any, vr: Any, track_width: Any) -> Tuple[Any, Any]:
    """
    Forward kinematics: converts wheel velocities (vl, vr) to body velocities (v, omega).
    Works transparently with floats, NumPy arrays, JAX arrays, CasADi MX, and Immrax intervals:
        v = (vl + vr) / 2
        omega = (vr - vl) / track_width
    """
    v = 0.5 * (vl + vr)
    omega = (vr - vl) / track_width
    return v, omega


def diff_drive_ik(v: Any, omega: Any, track_width: Any) -> Tuple[Any, Any]:
    """
    Inverse kinematics: converts body velocities (v, omega) to wheel velocities (vl, vr).
    Works transparently with floats, NumPy arrays, JAX arrays, CasADi MX, and Immrax intervals:
        vl = v - (track_width / 2) * omega
        vr = v + (track_width / 2) * omega
    """
    half_arm = 0.5 * track_width * omega
    return v - half_arm, v + half_arm


def diff_drive_wheel_forces(
    al: Any,
    ar: Any,
    mass: Any,
    inertia: Any,
    track_width: Any,
) -> Tuple[Any, Any]:
    """
    Computes required left and right wheel forces from wheel accelerations.
    Decomposes into net longitudinal force and yaw moment thrust:
        f_lin = (al + ar) * mass / 4
        f_ang = (ar - al) * inertia / (track_width^2)
        fl = f_lin - f_ang
        fr = f_lin + f_ang

    Arranged with interval/array variables as leading operands to support
    Immrax Intervals, JAX arrays, CasADi MX, and scalars transparently.
    """
    f_lin = ((al + ar) * mass) / 4.0
    f_ang = ((ar - al) * inertia) / (track_width * track_width)
    return f_lin - f_ang, f_lin + f_ang


def wheel_normal_forces(
    mass: Any,
    g: Any,
    h_cg: Any,
    track_width: Any,
    centripetal_accel: Any,
) -> Tuple[Any, Any]:
    """
    Normal force on each wheel with lateral (turning) weight transfer.

    A two-wheel differential robot has no front/rear axle, so longitudinal
    acceleration does not split load left/right (KI-010). Only the lateral
    transfer from centripetal acceleration (v * omega) applies:
        nl = m*g/2 - m*h*a_c/b
        nr = m*g/2 + m*h*a_c/b

    Immrax Intervals only implement left-hand operators, so the interval/array
    term leads every expression; plain floats and CasADi MX work too.
    """
    base_normal = (mass * g) / 2.0
    lateral_transfer = centripetal_accel * (mass * h_cg) / track_width
    return (-lateral_transfer) + base_normal, lateral_transfer + base_normal


def motor_force_at_velocity(
    v_wheel: Any,
    wheel_radius: Any,
    gearing: Any,
    t_max: Any,
    v_max_rad_s: Any,
    torque_headroom: Any = 1.0,
    smooth: bool = False,
    force_radius: Optional[Any] = None,
) -> Any:
    """
    Calculates max force magnitude a DC motor can apply at a given wheel velocity.

    Evaluates the linear motor torque-speed curve:
        omega_motor = (|v_wheel| / wheel_radius) * gearing
        torque = t_max * (1 - omega_motor / v_max_rad_s)
        force = (torque / r_force) * gearing * torque_headroom

    Args:
        v_wheel: Wheel linear velocity (m/s).
        wheel_radius: Radius for speed calculation (m).
        gearing: Motor-to-wheel gear ratio (motor turns per wheel rev).
        t_max: Stall torque (N*m).
        v_max_rad_s: No-load motor angular speed (rad/s).
        torque_headroom: Headroom / derating factor (default 1.0).
        smooth: If True, uses smooth abs approximation for CasADi NLP solvers.
        force_radius: Optional separate radius for torque-to-force conversion.
                      Defaults to wheel_radius. (Used in robust/interval bounds).

    Returns:
        Maximum wheel force in Newtons.
    """
    r_force = force_radius if force_radius is not None else wheel_radius
    omega = (v_wheel / wheel_radius) * gearing
    abs_omega = _duck_abs(omega, smooth=smooth)
    # Leading negation enables Immrax Interval left-hand subtraction
    torque_frac = (-(abs_omega / v_max_rad_s)) + 1.0
    torque = _duck_clamp_zero(torque_frac * t_max)
    force = (torque / r_force) * gearing * torque_headroom
    return force


def motor_linear_limits(
    wheel_radius: Any,
    gearing: Any,
    t_max: Any,
    v_max_rad_s: Any,
    torque_headroom: Any = 1.0,
    force_radius: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Returns (f_stall_per_wheel, v_noload_wheel) in (N, m/s).
    """
    r_force = force_radius if force_radius is not None else wheel_radius
    f_stall = float((t_max / r_force) * gearing * torque_headroom)
    v_noload = float(v_max_rad_s * wheel_radius / max(float(gearing), 1e-9))
    return f_stall, v_noload


def traction_limits(nl: Any, nr: Any, cof: Any) -> Tuple[Any, Any]:
    """Computes individual wheel traction limits (N) from normal forces and friction coefficient."""
    return cof * nl, cof * nr


def total_traction_limit(mass: Any, g: Any, cof: Any) -> Any:
    """Computes total robot traction limit (N) under static gravity normal load."""
    return cof * mass * g


def check_drivetrain_constraints(
    fl: Any,
    fr: Any,
    nl: Any,
    nr: Any,
    fl_max: Any,
    fr_max: Any,
    cof: Any,
    mass: Any,
    g: Any,
    smooth: bool = False,
) -> Dict[str, Any]:
    """
    Checks motor capability violations and wheel slip violations against normal forces.

    Returns dict containing:
        - left_motor_violation: excess force beyond motor limit (N)
        - right_motor_violation: excess force beyond motor limit (N)
        - left_wheel_slip: excess force beyond per-wheel friction limit (N)
        - right_wheel_slip: excess force beyond per-wheel friction limit (N)
        - traction_violation: excess combined force beyond total friction limit (N)
        - left_normal_force: nl (N)
        - right_normal_force: nr (N)
        - fl: fl (N)
        - fr: fr (N)
    """
    abs_fl = _duck_abs(fl, smooth=smooth)
    abs_fr = _duck_abs(fr, smooth=smooth)

    left_motor_violation = _duck_clamp_zero(abs_fl - fl_max)
    right_motor_violation = _duck_clamp_zero(abs_fr - fr_max)

    left_trac, right_trac = traction_limits(nl, nr, cof)
    left_wheel_slip = _duck_clamp_zero(abs_fl - left_trac)
    right_wheel_slip = _duck_clamp_zero(abs_fr - right_trac)

    tot_limit = total_traction_limit(mass, g, cof)
    traction_violation = _duck_clamp_zero((abs_fl + abs_fr) - tot_limit)

    return {
        "left_motor_violation": left_motor_violation,
        "right_motor_violation": right_motor_violation,
        "left_wheel_slip": left_wheel_slip,
        "right_wheel_slip": right_wheel_slip,
        "traction_violation": traction_violation,
        "left_normal_force": nl,
        "right_normal_force": nr,
        "fl": fl,
        "fr": fr,
    }


def kinematic_cornering_ceiling(
    kappa: Any,
    track_width: float,
    v_wheel_max: float,
    h_cg: float = CG_HEIGHT_M,
    mu: float = 0.6,
    g: float = 9.81,
) -> Any:
    """
    Computes composite physical velocity ceiling for a differential-drive robot
    at curvature kappa, unifying Pontryagin saturation, lateral load transfer, and tire friction:
        1. Actuator Diamond Boundary: v_frontier = v_wheel_max / (1 + 0.5 * B * |kappa|)
        2. Dynamic Load Transfer (Wheel-Lift): v_tip = sqrt(g * B / (2 * h_cg * |kappa|))
        3. Tire Traction Limit: v_traction = sqrt(mu * g / |kappa|)
        v_ceiling = min(v_wheel_max, v_frontier, v_tip, v_traction)
    """
    import numpy as np
    abs_kappa = np.abs(kappa)
    eps = 1e-4
    v_frontier = v_wheel_max / (1.0 + 0.5 * track_width * abs_kappa)
    v_tip = np.sqrt((g * track_width) / (2.0 * max(0.01, h_cg) * np.maximum(abs_kappa, eps)))
    v_traction = np.sqrt(mu * g / np.maximum(abs_kappa, eps))
    return np.minimum(v_wheel_max, np.minimum(v_frontier, np.minimum(v_tip, v_traction)))


def wrap_angle(angle: Any) -> Any:
    """
    Wraps an angle or array of angles into the standard interval [-pi, pi).
    Works with scalars, NumPy arrays, and JAX arrays.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(target: Any, source: Any) -> Any:
    """
    Computes the shortest signed angular difference (target - source) wrapped into [-pi, pi).
    Works with scalars, NumPy arrays, and JAX arrays.
    """
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi
