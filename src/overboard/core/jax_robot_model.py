import jax.numpy as jnp
from jax import jit
from typing import Dict, Any, Tuple, Optional
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class JAXRobotConfig:
    """
    Robot configuration mirroring RobotConfig but using JAX.
    """
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        robot_cfg = config_dict.get("robot", config_dict)

        self.mass = float(robot_cfg.get("mass", 0.723))
        self.inertia = float(robot_cfg.get("inertia", 0.0024))
        self.track_width = float(robot_cfg.get("track_width", 0.0965))
        self.wheel_radius = float(robot_cfg.get("wheel_radius", 0.028))
        self.v_max_rad_s = float(robot_cfg.get("v_max_rad_s", 15.7))
        self.t_max_nm = float(robot_cfg.get("t_max_nm", 0.04))
        self.gearing = float(robot_cfg.get("gearing", 1.0))
        self.cof = float(robot_cfg.get("cof", 0.45))
        self.g = float(robot_cfg.get("gravity", 9.81))
        self.torque_headroom = float(robot_cfg.get("torque_headroom", 0.85))
        self.speed_headroom = float(robot_cfg.get("speed_headroom", 0.90))

        # Uncertainty Intervals
        self.track_width_range = tuple(robot_cfg.get("track_width_range", (self.track_width * 0.98, self.track_width * 1.02)))
        self.wheel_radius_range = tuple(robot_cfg.get("wheel_radius_range", (self.wheel_radius * 0.98, self.wheel_radius * 1.02)))
        self.inertia_range = tuple(robot_cfg.get("inertia_range", (self.inertia * 0.8, self.inertia * 1.2)))
        self.t_max_range = tuple(robot_cfg.get("t_max_range", (self.t_max_nm * 0.75, self.t_max_nm * 0.95)))

    def get_max_force_at_velocity(self, v_wheel: jnp.ndarray, apply_headroom: bool = True, use_intervals: bool = False) -> jnp.ndarray:
        r = self.wheel_radius_range if use_intervals else self.wheel_radius
        t_max = self.t_max_range if use_intervals else self.t_max_nm

        # We handle intervals by taking the worst-case (lower bound for force)
        if use_intervals:
            from immrax import Interval
            r_iv = Interval(*r)
            t_iv = Interval(*t_max)
            v_abs_iv = Interval(jnp.abs(v_wheel), jnp.abs(v_wheel))
            omega = (v_abs_iv / r_iv) * Interval(self.gearing, self.gearing)
            torque = t_iv * (Interval(1.0, 1.0) - omega / Interval(self.v_max_rad_s, self.v_max_rad_s))
            force = (torque / r_iv) * Interval(self.gearing, self.gearing)
            if apply_headroom:
                force *= Interval(self.torque_headroom, self.torque_headroom)
            return force
        else:
            omega = (jnp.abs(v_wheel) / r) * self.gearing
            torque = t_max * (1.0 - omega / self.v_max_rad_s)
            torque = jnp.maximum(0, torque)
            force = (torque / r) * self.gearing
            if apply_headroom:
                force *= self.torque_headroom
            return force

    def max_linear_speed(self, apply_headroom: bool = True) -> jnp.ndarray:
        speed = self.v_max_rad_s * self.wheel_radius
        if apply_headroom:
            speed *= self.speed_headroom
        return speed

    def tree_flatten(self):
        children = (self.mass, self.inertia, self.track_width, self.wheel_radius,
                    self.v_max_rad_s, self.t_max_nm, self.gearing, self.cof,
                    self.g, self.torque_headroom, self.speed_headroom,
                    self.track_width_range, self.wheel_radius_range,
                    self.inertia_range, self.t_max_range)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls({})
        (obj.mass, obj.inertia, obj.track_width, obj.wheel_radius,
         obj.v_max_rad_s, obj.t_max_nm, obj.gearing, obj.cof,
         obj.g, obj.torque_headroom, obj.speed_headroom,
         obj.track_width_range, obj.wheel_radius_range,
         obj.inertia_range, obj.t_max_range) = children
        return obj

class JAXDifferentialDriveModel:
    """
    Differential drive robot dynamics model using JAX.
    """
    def __init__(self, config: JAXRobotConfig) -> None:
        self.cfg = config

    def get_dynamics(self, vl: jnp.ndarray, vr: jnp.ndarray, al: jnp.ndarray, ar: jnp.ndarray, use_intervals: bool = False) -> Tuple[Any, Any]:
        if use_intervals:
            from immrax import Interval
            L = Interval(*self.cfg.track_width_range)
            I = Interval(*self.cfg.inertia_range)
        else:
            L = self.cfg.track_width
            I = self.cfg.inertia

        a = (al + ar) / 2.0
        if use_intervals:
            from immrax import Interval
            alpha = Interval(ar - al, ar - al) / L
            f_total = Interval(self.cfg.mass * a, self.cfg.mass * a)
            m_total = I * alpha
            fr = (f_total + (Interval(2.0, 2.0) * m_total / L)) / Interval(2.0, 2.0)
            fl = f_total - fr
        else:
            alpha = (ar - al) / L
            f_total = self.cfg.mass * a
            m_total = I * alpha
            fr = (f_total + (2.0 * m_total / L)) / 2.0
            fl = f_total - fr
        return fl, fr

    def get_wheel_normal_forces(self, vl: jnp.ndarray, vr: jnp.ndarray, al: jnp.ndarray, ar: jnp.ndarray, use_intervals: bool = False) -> Tuple[Any, Any]:
        if use_intervals:
            from immrax import Interval
            L = Interval(*self.cfg.track_width_range)
        else:
            L = self.cfg.track_width

        a = (al + ar) / 2.0
        base_normal = (self.cfg.mass * self.cfg.g) / 2.0
        h_cg = 0.05
        wheelbase = L

        if use_intervals:
            from immrax import Interval
            longitudinal_transfer = Interval(self.cfg.mass * a * h_cg, self.cfg.mass * a * h_cg) / wheelbase
            v = (vl + vr) / 2.0
            omega = Interval(vr - vl, vr - vl) / L
            centripetal_accel_iv = Interval(v, v) * omega
            lateral_transfer = Interval(self.cfg.mass * h_cg, self.cfg.mass * h_cg) * centripetal_accel_iv / L
            nl = Interval(base_normal, base_normal) - longitudinal_transfer - lateral_transfer
            nr = Interval(base_normal, base_normal) - longitudinal_transfer + lateral_transfer
        else:
            longitudinal_transfer = (self.cfg.mass * a * h_cg) / wheelbase
            v = (vl + vr) / 2.0
            omega = (vr - vl) / L
            centripetal_accel = v * omega
            lateral_transfer = (self.cfg.mass * centripetal_accel * h_cg) / L
            nl = base_normal - longitudinal_transfer - lateral_transfer
            nr = base_normal - longitudinal_transfer + lateral_transfer
        return nl, nr

    def check_constraints(self, vl: jnp.ndarray, vr: jnp.ndarray, al: jnp.ndarray, ar: jnp.ndarray, apply_headroom: bool = True, use_intervals: bool = False) -> Dict[str, Any]:
        fl, fr = self.get_dynamics(vl, vr, al, ar, use_intervals)
        fl_max = self.cfg.get_max_force_at_velocity(vl, apply_headroom)
        fr_max = self.cfg.get_max_force_at_velocity(vr, apply_headroom)
        left_motor_violation = jnp.maximum(0, jnp.abs(fl) - fl_max)
        right_motor_violation = jnp.maximum(0, jnp.abs(fr) - fr_max)
        nl, nr = self.get_wheel_normal_forces(vl, vr, al, ar)
        left_traction_limit = self.cfg.cof * nl
        right_traction_limit = self.cfg.cof * nr
        left_wheel_slip = jnp.maximum(0, jnp.abs(fl) - left_traction_limit)
        right_wheel_slip = jnp.maximum(0, jnp.abs(fr) - right_traction_limit)
        traction_limit = self.cfg.cof * self.cfg.mass * self.cfg.g
        traction_violation = jnp.maximum(0, jnp.abs(fl) + jnp.abs(fr) - traction_limit)
        return {
            "left_motor_violation": left_motor_violation,
            "right_motor_violation": right_motor_violation,
            "left_wheel_slip": left_wheel_slip,
            "right_wheel_slip": right_wheel_slip,
            "traction_violation": traction_violation,
            "left_normal_force": nl,
            "right_normal_force": nr,
            "fl": fl,
            "fr": fr
        }
