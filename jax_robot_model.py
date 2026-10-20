import jax.numpy as jnp
from jax import jit
from typing import Dict, Any, Tuple, Optional, Union
from jax.tree_util import register_pytree_node_class

from robot_model import RobotConfig, CG_HEIGHT_M, wheel_normal_forces

@register_pytree_node_class
class JAXRobotConfig:
    """
    Robot configuration mirroring RobotConfig but using JAX.
    """
    def __init__(self, config_dict: Union[Dict[str, Any], RobotConfig]) -> None:
        # Delegate parsing to RobotConfig so canonical, flat, and legacy
        # config formats all produce the same robot (KI-008).
        rc = config_dict if isinstance(config_dict, RobotConfig) else RobotConfig(config_dict)

        self.mass = float(rc.mass)
        self.inertia = float(rc.inertia)
        self.track_width = float(rc.track_width)
        self.wheel_radius = float(rc.wheel_radius)
        self.v_max_rad_s = float(rc.v_max_rad_s)
        self.t_max_nm = float(rc.t_max_nm)
        self.gearing = float(rc.gearing)
        self.cof = float(rc.cof)
        self.g = float(rc.g)
        self.torque_headroom = float(rc.torque_headroom)
        self.speed_headroom = float(rc.speed_headroom)

        # Drivetrain friction constants
        self.rolling_resistance = float(rc.rolling_resistance)
        self.gearbox_coulomb_nm = float(rc.gearbox_coulomb_nm)
        self.gearbox_viscous_nm_s = float(rc.gearbox_viscous_nm_s)
        self.reflected_inertia_kg_m2 = float(rc.reflected_inertia_kg_m2)

        # Chassis footprint and sensor geometry
        self.width_m = float(rc.width_m)
        self.length_m = float(rc.length_m)
        self.front_bumper_m = float(rc.front_bumper_m)
        self.back_bumper_m = float(rc.back_bumper_m)
        self.lever_arm_m = float(rc.lever_arm_m)

        # Uncertainty Intervals (normalized by RobotConfig, including the
        # nested robot.uncertainties block used by canonical configs).
        self.track_width_range = tuple(rc.uncertainties["track_width_range"])
        self.wheel_radius_range = tuple(rc.uncertainties["wheel_radius_range"])
        self.inertia_range = tuple(rc.uncertainties["inertia_kg_m2"])
        self.t_max_range = tuple(rc.uncertainties["t_max_range"])
        self.cof_range = tuple(rc.uncertainties["cof"])
        self.torque_headroom_range = tuple(rc.uncertainties["torque_headroom"])
        self.backlash_range = tuple(rc.uncertainties["gear_backlash_m"])
        self.initial_x_range = tuple(rc.uncertainties["initial_x_m"])
        self.initial_y_range = tuple(rc.uncertainties["initial_y_m"])
        self.initial_theta_range = tuple(rc.uncertainties["initial_theta_rad"])

    def no_load_wheel_speed_worst_case(self) -> float:
        """The wheel speed at which the motor makes no torque (m/s).

        The smallest declared wheel radius is the one that reaches the
        no-load speed first. The motor turns ``gearing`` times per wheel
        revolution, so the wheel's no-load speed is the motor's divided by
        it, exactly as :meth:`robot_model.RobotConfig.max_linear_speed`
        and the force model read it (KI-035).
        """
        radius = min(self.wheel_radius_range)
        return self.v_max_rad_s * radius / max(self.gearing, 1e-9)

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
        speed = self.v_max_rad_s * self.wheel_radius / jnp.maximum(self.gearing, 1e-9)
        if apply_headroom:
            speed *= self.speed_headroom
        return speed

    def tree_flatten(self):
        children = (self.mass, self.inertia, self.track_width, self.wheel_radius,
                    self.v_max_rad_s, self.t_max_nm, self.gearing, self.cof,
                    self.g, self.torque_headroom, self.speed_headroom,
                    self.track_width_range, self.wheel_radius_range,
                    self.inertia_range, self.t_max_range,
                    self.cof_range, self.torque_headroom_range,
                    self.backlash_range)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls({})
        (obj.mass, obj.inertia, obj.track_width, obj.wheel_radius,
         obj.v_max_rad_s, obj.t_max_nm, obj.gearing, obj.cof,
         obj.g, obj.torque_headroom, obj.speed_headroom,
         obj.track_width_range, obj.wheel_radius_range,
         obj.inertia_range, obj.t_max_range,
         obj.cof_range, obj.torque_headroom_range,
         obj.backlash_range) = children
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
        # Lateral (turning) weight transfer only; no front/rear split on a
        # two-wheel robot (KI-010). Same formula as the CPU model and Immrax.
        if use_intervals:
            from immrax import Interval
            L = Interval(*self.cfg.track_width_range)
        else:
            L = self.cfg.track_width

        v = (vl + vr) / 2.0
        if use_intervals:
            from immrax import Interval
            omega = Interval(vr - vl, vr - vl) / L
            centripetal_accel = Interval(v, v) * omega
        else:
            omega = (vr - vl) / L
            centripetal_accel = v * omega
        return wheel_normal_forces(self.cfg.mass, self.cfg.g, CG_HEIGHT_M, L, centripetal_accel)

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
