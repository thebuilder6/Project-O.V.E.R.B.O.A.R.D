import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional, Union
from jax.tree_util import register_pytree_node_class

from overboard.models.robot_model import RobotConfig
from overboard.models.drivetrain_physics import (
    CG_HEIGHT_M,
    diff_drive_fk,
    diff_drive_wheel_forces,
    wheel_normal_forces,
    motor_force_at_velocity,
    check_drivetrain_constraints,
)

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
        self.declared_t_max_range = bool(getattr(rc, "declared_t_max_range", False))
        if self.declared_t_max_range:
            self.t_max_range = tuple(rc.uncertainties["t_max_range"])
        else:
            # Synthetic default is not a measurement: keep the nominal point
            # so uncalibrated robots validate against the declared motor.
            self.t_max_range = (float(rc.t_max_nm), float(rc.t_max_nm))
        self.cof_range = tuple(rc.uncertainties["cof"])
        self.torque_headroom_range = tuple(rc.uncertainties["torque_headroom"])
        self.backlash_range = tuple(rc.uncertainties["gear_backlash_m"])
        self.initial_x_range = tuple(rc.uncertainties["initial_x_m"])
        self.initial_y_range = tuple(rc.uncertainties["initial_y_m"])
        self.initial_theta_range = tuple(rc.uncertainties["initial_theta_rad"])

    def no_load_wheel_speed_worst_case(
            self,
            v_max_range: Optional[Tuple[float, float]] = None,
            gearing_range: Optional[Tuple[float, float]] = None) -> float:
        """The wheel speed at which the motor makes no torque (m/s).

        The smallest declared wheel radius is the one that reaches the
        no-load speed first. The motor turns ``gearing`` times per wheel
        revolution, so the wheel's no-load speed is the motor's divided by
        it, exactly as :meth:`robot_model.RobotConfig.max_linear_speed`
        and the force model read it (KI-035). With ranges the worst case
        is the smallest no-load speed: the slowest motor over the
        largest gearing.
        """
        radius = min(self.wheel_radius_range)
        v_max = min(v_max_range) if v_max_range else self.v_max_rad_s
        gearing = max(gearing_range) if gearing_range else self.gearing
        return v_max * radius / max(gearing, 1e-9)

    def get_max_force_at_velocity(self, v_wheel: jnp.ndarray, apply_headroom: bool = True, use_intervals: bool = False) -> Any:
        if use_intervals:
            from immrax import Interval
            r_iv = Interval(*self.wheel_radius_range)
            t_iv = Interval(*self.t_max_range)
            v_abs_iv = Interval(jnp.abs(v_wheel), jnp.abs(v_wheel)) if not isinstance(v_wheel, Interval) else v_wheel
            headroom_iv = Interval(self.torque_headroom, self.torque_headroom) if apply_headroom else Interval(1.0, 1.0)
            gearing_iv = Interval(self.gearing, self.gearing)
            vmax_iv = Interval(self.v_max_rad_s, self.v_max_rad_s)
            return motor_force_at_velocity(
                v_wheel=v_abs_iv,
                wheel_radius=r_iv,
                gearing=gearing_iv,
                t_max=t_iv,
                v_max_rad_s=vmax_iv,
                torque_headroom=headroom_iv,
                smooth=False,
            )
        else:
            headroom = self.torque_headroom if apply_headroom else 1.0
            return motor_force_at_velocity(
                v_wheel=v_wheel,
                wheel_radius=self.wheel_radius,
                gearing=self.gearing,
                t_max=self.t_max_nm,
                v_max_rad_s=self.v_max_rad_s,
                torque_headroom=headroom,
                smooth=False,
            )

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
            al_iv = Interval(al, al) if not isinstance(al, Interval) else al
            ar_iv = Interval(ar, ar) if not isinstance(ar, Interval) else ar
            return diff_drive_wheel_forces(al_iv, ar_iv, self.cfg.mass, I, L)
        else:
            return diff_drive_wheel_forces(al, ar, self.cfg.mass, self.cfg.inertia, self.cfg.track_width)

    def get_wheel_normal_forces(self, vl: jnp.ndarray, vr: jnp.ndarray, al: jnp.ndarray, ar: jnp.ndarray, use_intervals: bool = False) -> Tuple[Any, Any]:
        if use_intervals:
            from immrax import Interval
            L = Interval(*self.cfg.track_width_range)
            vl_iv = Interval(vl, vl) if not isinstance(vl, Interval) else vl
            vr_iv = Interval(vr, vr) if not isinstance(vr, Interval) else vr
            v, omega = diff_drive_fk(vl_iv, vr_iv, L)
            centripetal_accel = v * omega
            return wheel_normal_forces(self.cfg.mass, self.cfg.g, CG_HEIGHT_M, L, centripetal_accel)
        else:
            v, omega = diff_drive_fk(vl, vr, self.cfg.track_width)
            centripetal_accel = v * omega
            return wheel_normal_forces(self.cfg.mass, self.cfg.g, CG_HEIGHT_M, self.cfg.track_width, centripetal_accel)

    def check_constraints(self, vl: jnp.ndarray, vr: jnp.ndarray, al: jnp.ndarray, ar: jnp.ndarray, apply_headroom: bool = True, use_intervals: bool = False) -> Dict[str, Any]:
        fl, fr = self.get_dynamics(vl, vr, al, ar, use_intervals)
        fl_max = self.cfg.get_max_force_at_velocity(vl, apply_headroom, use_intervals)
        fr_max = self.cfg.get_max_force_at_velocity(vr, apply_headroom, use_intervals)
        nl, nr = self.get_wheel_normal_forces(vl, vr, al, ar, use_intervals)
        return check_drivetrain_constraints(
            fl=fl,
            fr=fr,
            nl=nl,
            nr=nr,
            fl_max=fl_max,
            fr_max=fr_max,
            cof=self.cfg.cof,
            mass=self.cfg.mass,
            g=self.cfg.g,
            smooth=False,
        )
