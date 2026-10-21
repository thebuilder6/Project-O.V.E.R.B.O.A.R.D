import math
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Union
import casadi as ca

from overboard.paths import ROBOT_CONFIG_PY
from overboard.models.drivetrain_physics import (
    CG_HEIGHT_M,
    diff_drive_fk,
    diff_drive_ik,
    diff_drive_wheel_forces,
    wheel_normal_forces,
    motor_force_at_velocity,
    motor_linear_limits,
    traction_limits,
    total_traction_limit,
    check_drivetrain_constraints,
    kinematic_cornering_ceiling,
    wrap_angle,
    angle_diff,
    body_frame_error,
    asymmetric_late_apex_bump,
    ramsete_gain,
)


class RobotConfig:
    """
    Robot configuration parsed from Choreo-like JSON config file.

    Default values are typical for a small FLL robot with LEGO motors.
    """

    # Default values for typical FLL robot
    DEFAULT_MASS = 0.723  # kg
    DEFAULT_INERTIA = 0.0024  # kg·m²
    DEFAULT_TRACK_WIDTH = 0.0965  # m (96.5 mm)
    DEFAULT_WHEEL_RADIUS = 0.028  # m (28 mm)
    DEFAULT_VMAX = 15.7  # rad/s (150 RPM)
    DEFAULT_TMAX = 0.04  # N·m
    DEFAULT_GEARING = 1.0
    DEFAULT_COF = 0.45  # coefficient of friction
    GRAVITY = 9.81  # m/s²
    DEFAULT_TORQUE_HEADROOM = 0.85  # 15% headroom for torque corrections
    DEFAULT_SPEED_HEADROOM = 0.90  # 10% headroom for speed corrections
    DEFAULT_ROLLING_RESISTANCE = 0.0
    DEFAULT_GEARBOX_COULOMB_NM = 0.0
    DEFAULT_GEARBOX_VISCOUS_NM_S = 0.0
    DEFAULT_REFLECTED_INERTIA = 0.0

    # Chassis footprint and sensor geometry defaults
    DEFAULT_WIDTH_M = 0.16  # m
    DEFAULT_LENGTH_M = 0.18  # m
    DEFAULT_FRONT_BUMPER_M = 0.10  # m (forward of axle)
    DEFAULT_BACK_BUMPER_M = 0.08  # m (backward of axle)
    DEFAULT_LEVER_ARM_M = 0.10  # m
    DEFAULT_COLOR_SENSOR_LEFT_OFFSET = (0.06, -0.03)  # (x, y) in m
    DEFAULT_COLOR_SENSOR_RIGHT_OFFSET = (0.06, 0.03)  # (x, y) in m

    # Hardware and port mapping defaults
    DEFAULT_HUB_TOP_SIDE = "Axis.Z"
    DEFAULT_HUB_FRONT_SIDE = "Axis.Y"
    DEFAULT_MOTOR_LEFT_PORT = "Port.F"
    DEFAULT_MOTOR_LEFT_DIR = "Direction.COUNTERCLOCKWISE"
    DEFAULT_MOTOR_RIGHT_PORT = "Port.B"
    DEFAULT_MOTOR_RIGHT_DIR = "Direction.CLOCKWISE"
    DEFAULT_MOTOR_FRONT_PORT = "Port.C"
    DEFAULT_MOTOR_FRONT_DIR = "Direction.CLOCKWISE"
    DEFAULT_MOTOR_BACK_PORT = "Port.D"
    DEFAULT_MOTOR_BACK_DIR = "Direction.CLOCKWISE"
    DEFAULT_COLOR_SENSOR_LEFT_PORT = "Port.E"
    DEFAULT_COLOR_SENSOR_RIGHT_PORT = "Port.A"

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """
        Initialize robot configuration from config dictionary.

        Args:
            config_dict: Dictionary containing robot configuration in the new JSON format
        """
        # Support both a nested "robot" key, a nested "config" key, or a flat dictionary
        if "robot" in config_dict:
            robot_cfg = config_dict["robot"]
        elif "config" in config_dict:
            robot_cfg = config_dict["config"]
        else:
            robot_cfg = config_dict
            
        def get_val(key: str, default: Any, old_key: Optional[str] = None) -> Any:
            item = robot_cfg.get(key)
            if item is None and old_key is not None:
                item = robot_cfg.get(old_key)
            if item is None:
                return default
            if isinstance(item, dict) and "val" in item:
                return item["val"]
            return item

        self.mass = float(get_val("mass", self.DEFAULT_MASS))
        self.inertia = float(get_val("inertia", self.DEFAULT_INERTIA))
        self.track_width = float(get_val("track_width", self.DEFAULT_TRACK_WIDTH, "differentialTrackWidth"))
        self.wheel_radius = float(get_val("wheel_radius", self.DEFAULT_WHEEL_RADIUS, "radius"))
        self.v_max_rad_s = float(get_val("v_max_rad_s", self.DEFAULT_VMAX, "vmax"))
        self.t_max_nm = float(get_val("t_max_nm", self.DEFAULT_TMAX, "tmax"))
        self.gearing = float(get_val("gearing", self.DEFAULT_GEARING))
        self.cof = float(get_val("cof", self.DEFAULT_COF))
        self.g = float(get_val("gravity", self.GRAVITY))
        self.torque_headroom = float(get_val("torque_headroom", self.DEFAULT_TORQUE_HEADROOM, "torqueHeadroom"))
        self.speed_headroom = float(get_val("speed_headroom", self.DEFAULT_SPEED_HEADROOM, "speedHeadroom"))

        # Drivetrain friction constants
        self.rolling_resistance = float(get_val("rolling_resistance", self.DEFAULT_ROLLING_RESISTANCE, "rollingResistance"))
        self.gearbox_coulomb_nm = float(get_val("gearbox_coulomb_nm", self.DEFAULT_GEARBOX_COULOMB_NM, "gearboxCoulombNm"))
        self.gearbox_viscous_nm_s = float(get_val("gearbox_viscous_nm_s", self.DEFAULT_GEARBOX_VISCOUS_NM_S, "gearboxViscousNmS"))
        self.reflected_inertia_kg_m2 = float(get_val("reflected_inertia_kg_m2", self.DEFAULT_REFLECTED_INERTIA, "reflectedInertia"))

        # Chassis footprint and sensor geometry
        self.width_m = float(get_val("width_m", self.DEFAULT_WIDTH_M, "width"))
        self.length_m = float(get_val("length_m", self.DEFAULT_LENGTH_M, "length"))
        self.front_bumper_m = float(get_val("front_bumper_m", self.DEFAULT_FRONT_BUMPER_M, "frontBumper"))
        self.back_bumper_m = float(get_val("back_bumper_m", self.DEFAULT_BACK_BUMPER_M, "backBumper"))
        self.lever_arm_m = float(get_val("lever_arm_m", self.DEFAULT_LEVER_ARM_M, "leverArm"))

        def _parse_offset(val: Any, default: Tuple[float, float]) -> Tuple[float, float]:
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return (float(val[0]), float(val[1]))
            if isinstance(val, dict) and "x" in val and "y" in val:
                return (float(val["x"]), float(val["y"]))
            return default

        cs_cfg = robot_cfg.get("color_sensors") or robot_cfg.get("color_sensor_offsets") or {}
        if not isinstance(cs_cfg, dict):
            cs_cfg = {}

        self.color_sensor_left_offset_m = _parse_offset(
            cs_cfg.get("left") or robot_cfg.get("color_sensor_left_offset_m") or robot_cfg.get("colorSensorLeftOffset"),
            self.DEFAULT_COLOR_SENSOR_LEFT_OFFSET,
        )
        self.color_sensor_right_offset_m = _parse_offset(
            cs_cfg.get("right") or robot_cfg.get("color_sensor_right_offset_m") or robot_cfg.get("colorSensorRightOffset"),
            self.DEFAULT_COLOR_SENSOR_RIGHT_OFFSET,
        )
        self.color_sensor_offsets = {
            "left": self.color_sensor_left_offset_m,
            "right": self.color_sensor_right_offset_m,
        }

        # Hub hardware & port mapping
        hw_cfg = robot_cfg.get("hardware") or robot_cfg.get("ports") or {}
        if not isinstance(hw_cfg, dict):
            hw_cfg = {}

        def _get_hw(key: str, default: str, alt_keys: Tuple[str, ...] = ()) -> str:
            item = hw_cfg.get(key)
            if item is None:
                item = robot_cfg.get(key)
            if item is None:
                for alt in alt_keys:
                    item = hw_cfg.get(alt) or robot_cfg.get(alt)
                    if item is not None:
                        break
            if item is None:
                return default
            return str(item).strip()

        self.hub_top_side = _get_hw("hub_top_side", self.DEFAULT_HUB_TOP_SIDE, ("top_side", "hubTopSide"))
        self.hub_front_side = _get_hw("hub_front_side", self.DEFAULT_HUB_FRONT_SIDE, ("front_side", "hubFrontSide"))
        self.motor_left_port = _get_hw("motor_left_port", self.DEFAULT_MOTOR_LEFT_PORT, ("motorLeft", "left_motor_port"))
        self.motor_left_dir = _get_hw("motor_left_dir", self.DEFAULT_MOTOR_LEFT_DIR, ("motorLeftDir", "left_motor_dir"))
        self.motor_right_port = _get_hw("motor_right_port", self.DEFAULT_MOTOR_RIGHT_PORT, ("motorRight", "right_motor_port"))
        self.motor_right_dir = _get_hw("motor_right_dir", self.DEFAULT_MOTOR_RIGHT_DIR, ("motorRightDir", "right_motor_dir"))
        self.motor_front_port = _get_hw("motor_front_port", self.DEFAULT_MOTOR_FRONT_PORT, ("motorFront", "front_motor_port"))
        self.motor_front_dir = _get_hw("motor_front_dir", self.DEFAULT_MOTOR_FRONT_DIR, ("motorFrontDir", "front_motor_dir"))
        self.motor_back_port = _get_hw("motor_back_port", self.DEFAULT_MOTOR_BACK_PORT, ("motorBack", "back_motor_port"))
        self.motor_back_dir = _get_hw("motor_back_dir", self.DEFAULT_MOTOR_BACK_DIR, ("motorBackDir", "back_motor_dir"))
        self.color_sensor_left_port = _get_hw("color_sensor_left_port", self.DEFAULT_COLOR_SENSOR_LEFT_PORT, ("colorSensorLeft", "left_color_sensor_port"))
        self.color_sensor_right_port = _get_hw("color_sensor_right_port", self.DEFAULT_COLOR_SENSOR_RIGHT_PORT, ("colorSensorRight", "right_color_sensor_port"))

        self.hardware = {
            "hub_top_side": self.hub_top_side,
            "hub_front_side": self.hub_front_side,
            "motor_left_port": self.motor_left_port,
            "motor_left_dir": self.motor_left_dir,
            "motor_right_port": self.motor_right_port,
            "motor_right_dir": self.motor_right_dir,
            "motor_front_port": self.motor_front_port,
            "motor_front_dir": self.motor_front_dir,
            "motor_back_port": self.motor_back_port,
            "motor_back_dir": self.motor_back_dir,
            "color_sensor_left_port": self.color_sensor_left_port,
            "color_sensor_right_port": self.color_sensor_right_port,
        }

        # Normalize the uncertainty block once so validation and JAX models use
        # the same ranges as the optimizer. Canonical configs nest these under
        # robot.uncertainties; older configs may put them at the same level as
        # the robot keys. Fall back to the old hard-coded behaviour.
        uncertainty_cfg = robot_cfg.get("uncertainties", {})
        if not isinstance(uncertainty_cfg, dict):
            uncertainty_cfg = {}

        def get_range(key_names: Tuple[str, ...], default: Tuple[float, float]) -> Tuple[float, float]:
            for key in key_names:
                item = uncertainty_cfg.get(key)
                if item is None:
                    item = robot_cfg.get(key)
                if item is None:
                    continue
                if isinstance(item, dict) and "val" in item:
                    item = item["val"]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    return (float(item[0]), float(item[1]))
            return default

        self.uncertainties = {
            "cof": get_range(("cof",), (self.cof * 0.8, self.cof * 1.2)),
            "torque_headroom": get_range(
                ("torque_headroom", "torqueHeadroom"), (0.8, 1.0)),
            "gear_backlash_m": get_range(
                ("gear_backlash_m", "backlash"), (0.0001, 0.0006)),
            "track_width_range": get_range(
                ("track_width_range",),
                (self.track_width * 0.99, self.track_width * 1.01)),
            "wheel_radius_range": get_range(
                ("wheel_radius_range",),
                (self.wheel_radius * 0.99, self.wheel_radius * 1.01)),
            "inertia_kg_m2": get_range(
                ("inertia_kg_m2", "inertia_range"),
                (self.inertia * 0.90, self.inertia * 1.10)),
            "t_max_range": get_range(
                ("t_max_range",), (self.t_max_nm * 0.75, self.t_max_nm * 0.95)),
            "initial_x_m": get_range(
                ("initial_x_m", "initial_x", "dx0"), (-0.005, 0.005)),
            "initial_y_m": get_range(
                ("initial_y_m", "initial_y", "dy0"), (-0.005, 0.005)),
            "initial_theta_rad": get_range(
                ("initial_theta_rad", "initial_theta", "dtheta0", "initial_heading_rad"),
                (-math.radians(1.0), math.radians(1.0))),
        }

        # Whether the config declared the stall-torque range itself. A
        # measured stall torque replaces the declared number, and the range
        # that was derived from the declared number has to follow it, unless
        # the config pinned the range on purpose.
        self.declared_t_max_range = "t_max_range" in uncertainty_cfg or "t_max_range" in robot_cfg



    def chassis_radius_m(self) -> float:
        """Effective body radius covering front/back bumper and width."""
        return max(
            math.hypot(self.front_bumper_m, self.width_m / 2.0),
            math.hypot(self.back_bumper_m, self.width_m / 2.0),
            self.width_m / 2.0,
            self.track_width / 2.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RobotConfig to dictionary format."""
        return {
            "mass": self.mass,
            "inertia": self.inertia,
            "track_width": self.track_width,
            "wheel_radius": self.wheel_radius,
            "v_max_rad_s": self.v_max_rad_s,
            "t_max_nm": self.t_max_nm,
            "gearing": self.gearing,
            "cof": self.cof,
            "gravity": self.g,
            "torque_headroom": self.torque_headroom,
            "speed_headroom": self.speed_headroom,
            "rolling_resistance": self.rolling_resistance,
            "gearbox_coulomb_nm": self.gearbox_coulomb_nm,
            "gearbox_viscous_nm_s": self.gearbox_viscous_nm_s,
            "reflected_inertia_kg_m2": self.reflected_inertia_kg_m2,
            "width_m": self.width_m,
            "length_m": self.length_m,
            "front_bumper_m": self.front_bumper_m,
            "back_bumper_m": self.back_bumper_m,
            "lever_arm_m": self.lever_arm_m,
            "color_sensor_offsets": {
                "left": list(self.color_sensor_left_offset_m),
                "right": list(self.color_sensor_right_offset_m),
            },
            "hardware": dict(self.hardware),
            "uncertainties": {k: list(v) for k, v in self.uncertainties.items()},
        }

    def as_dict(self) -> Dict[str, Any]:
        return self.to_dict()

    def sync_pybricks_robot_config(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Sync pybricks_code/robot_config.py from this RobotConfig.

        Ensures WHEEL_DIAMETER_MM, AXLE_TRACK_MM, motor ports & directions,
        attachment ports, sensor ports, and hub IMU axes match this config exactly.
        """
        if path is None:
            path = ROBOT_CONFIG_PY
        else:
            path = Path(path)

        if not path.parent.is_dir():
            return False

        wheel_diam_mm = round(self.wheel_radius * 2000.0, 2)
        wheel_diam_str = f"{int(wheel_diam_mm)}" if wheel_diam_mm.is_integer() else f"{wheel_diam_mm}"
        axle_track_mm = round(self.track_width * 1000.0, 2)
        axle_track_str = f"{int(axle_track_mm)}" if axle_track_mm.is_integer() else f"{axle_track_mm}"

        def _fmt_axis(a: str) -> str:
            return a if a.startswith("Axis.") else f"Axis.{a}"

        def _fmt_port(p: str) -> str:
            return p if p.startswith("Port.") else f"Port.{p}"

        def _fmt_dir(d: str) -> str:
            return d if d.startswith("Direction.") else f"Direction.{d}"

        top_axis = _fmt_axis(self.hub_top_side)
        front_axis = _fmt_axis(self.hub_front_side)
        left_port = _fmt_port(self.motor_left_port)
        left_dir = _fmt_dir(self.motor_left_dir)
        right_port = _fmt_port(self.motor_right_port)
        right_dir = _fmt_dir(self.motor_right_dir)
        front_port = _fmt_port(self.motor_front_port)
        front_dir = _fmt_dir(self.motor_front_dir)
        back_port = _fmt_port(self.motor_back_port)
        back_dir = _fmt_dir(self.motor_back_dir)
        cs_l_port = _fmt_port(self.color_sensor_left_port)
        cs_r_port = _fmt_port(self.color_sensor_right_port)

        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            content = ""

        if content:
            content = re.sub(
                r"^\s*WHEEL_DIAMETER_MM\s*=\s*[0-9.]+",
                f"WHEEL_DIAMETER_MM = {wheel_diam_str}",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\s*AXLE_TRACK_MM\s*=\s*[0-9.]+",
                f"AXLE_TRACK_MM = {axle_track_str}",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\s*hub\s*=\s*PrimeHub\([^)]*\)",
                f"hub = PrimeHub(top_side={top_axis}, front_side={front_axis})",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\s*motorLeft\s*=\s*Motor\([^)]*\)",
                f"motorLeft = Motor({left_port}, {left_dir})",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\s*motorRight\s*=\s*Motor\([^)]*\)",
                f"motorRight = Motor({right_port}, {right_dir})",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\s*motorFront\s*=\s*Motor\([^)]*\)",
                f"motorFront = Motor({front_port}, {front_dir})",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\s*motorBack\s*=\s*Motor\([^)]*\)",
                f"motorBack = Motor({back_port}, {back_dir})",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\s*colorSensorLeft\s*=\s*ColorSensor\([^)]*\)",
                f"colorSensorLeft = ColorSensor({cs_l_port})",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^\s*colorSensorRight\s*=\s*ColorSensor\([^)]*\)",
                f"colorSensorRight = ColorSensor({cs_r_port})",
                content,
                flags=re.MULTILINE,
            )
        else:
            content = f'''"""
ROBOT CONFIGURATION
-------------------
Auto-generated from RobotConfig.
"""
from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor
from pybricks.parameters import Axis, Direction, Port
from pybricks.robotics import DriveBase

hub = PrimeHub(top_side={top_axis}, front_side={front_axis})
motorLeft = Motor({left_port}, {left_dir})
motorRight = Motor({right_port}, {right_dir})
motorFront = Motor({front_port}, {front_dir})
motorBack = Motor({back_port}, {back_dir})
colorSensorLeft = ColorSensor({cs_l_port})
colorSensorRight = ColorSensor({cs_r_port})

WHEEL_DIAMETER_MM = {wheel_diam_str}
AXLE_TRACK_MM = {axle_track_str}
drivebase = DriveBase(motorLeft, motorRight,
                      wheel_diameter=WHEEL_DIAMETER_MM,
                      axle_track=AXLE_TRACK_MM)
drivebase.use_gyro(True)
'''

        path.write_text(content, encoding="utf-8")
        return True

    def get_max_force_at_velocity(self, v_wheel: float, apply_headroom: bool = True) -> float:
        """
        Calculates max force magnitude a motor can apply at a given wheel velocity.

        Uses a linear motor curve (symmetric braking/driving limit):
        - At v=0: max force = t_max / wheel_radius
        - At v=v_max: max force = 0

        Args:
            v_wheel: Wheel velocity in m/s
            apply_headroom: If True, applies safety margin for real-world tracking

        Returns:
            Maximum force in Newtons
        """
        headroom = self.torque_headroom if apply_headroom else 1.0
        return float(motor_force_at_velocity(
            v_wheel=v_wheel,
            wheel_radius=self.wheel_radius,
            gearing=self.gearing,
            t_max=self.t_max_nm,
            v_max_rad_s=self.v_max_rad_s,
            torque_headroom=headroom,
            smooth=False,
        ))

    def max_linear_speed(self, apply_headroom: bool = True) -> float:
        """No-load linear speed of the wheel (m/s).

        The motor's no-load speed is at the motor, and the motor turns
        ``gearing`` times per wheel revolution, so the wheel's no-load
        speed is that speed divided by the gearing. The force model in
        this file already reads it that way: the motor's speed for a wheel
        speed is ``(v / r) * gearing``, compared with ``v_max_rad_s``.
        Leaving the gearing out made the optimizer's wheel-speed bound and
        the CPU audit's speed limit larger than the motor's real top
        speed for a geared robot, so every plan for one was guaranteed to
        fail the motor gate (KI-035).
        """
        speed = self.v_max_rad_s * self.wheel_radius / max(self.gearing, 1e-9)
        if apply_headroom:
            speed *= self.speed_headroom
        return speed

    def get_motor_linear_limits(self, apply_headroom: bool = True) -> Tuple[float, float]:
        """
        Returns (f_stall_per_wheel, v_noload_wheel) in (N, m/s).
        Encapsulates the DC motor linear characteristic line F(v) = F_stall * (1 - |v| / v_noload).
        """
        if apply_headroom:
            r_speed, r_force, headroom = self.get_robust_force_params()
            return motor_linear_limits(
                wheel_radius=r_speed,
                gearing=self.gearing,
                t_max=self.t_max_nm,
                v_max_rad_s=self.v_max_rad_s,
                torque_headroom=headroom,
                force_radius=r_force,
            )
        else:
            return motor_linear_limits(
                wheel_radius=self.wheel_radius,
                gearing=self.gearing,
                t_max=self.t_max_nm,
                v_max_rad_s=self.v_max_rad_s,
                torque_headroom=1.0,
            )

    def no_load_wheel_speed_worst_case(self) -> float:
        """The wheel speed at which the motor makes no torque (m/s).

        The smallest declared wheel radius is the one that reaches the
        no-load speed first, so it is the limit the validator checks. It
        is the same number as :meth:`max_linear_speed` without the
        headroom, and with the gearing applied the same way.
        """
        radius = min(self.uncertainties["wheel_radius_range"])
        return self.v_max_rad_s * radius / max(self.gearing, 1e-9)

    def get_max_force_symbolic(self, v_wheel: ca.MX, apply_headroom: bool = True) -> ca.MX:
        """
        CasADi symbolic version of get_max_force_at_velocity.
        """
        headroom = self.torque_headroom if apply_headroom else 1.0
        return motor_force_at_velocity(
            v_wheel=v_wheel,
            wheel_radius=self.wheel_radius,
            gearing=self.gearing,
            t_max=self.t_max_nm,
            v_max_rad_s=self.v_max_rad_s,
            torque_headroom=headroom,
            smooth=True,
        )

    def get_robust_force_params(self) -> Tuple[float, float, float]:
        """
        Worst-case parameters for the per-wheel torque-speed limit.

        The Immrax validator evaluates its lower force bound with the smallest
        wheel radius in the speed term, the largest wheel radius in the force
        term, and the smallest declared torque headroom. Reusing the same
        values here keeps the optimizer's constraints and the validator's
        declared uncertainty ranges consistent (KI-009/KI-010).

        Returns:
            (speed_radius, force_radius, torque_headroom)
        """
        radius_range = self.uncertainties["wheel_radius_range"]
        headroom_range = self.uncertainties["torque_headroom"]
        return (float(min(radius_range)), float(max(radius_range)),
                float(min(headroom_range)))

    def get_max_force_at_velocity_robust(self, v_wheel: float) -> float:
        """
        Worst-case per-wheel force limit at a wheel velocity.

        This matches the lower bound used by the Immrax validator: the smaller
        radius reduces the available torque, the larger radius reduces the
        force per unit torque, and the smaller declared headroom derates the
        result.
        """
        r_speed, r_force, headroom = self.get_robust_force_params()
        return float(motor_force_at_velocity(
            v_wheel=v_wheel,
            wheel_radius=r_speed,
            gearing=self.gearing,
            t_max=self.t_max_nm,
            v_max_rad_s=self.v_max_rad_s,
            torque_headroom=headroom,
            smooth=False,
            force_radius=r_force,
        ))

    def get_max_force_symbolic_robust(self, v_wheel: ca.MX) -> ca.MX:
        """CasADi symbolic version of get_max_force_at_velocity_robust."""
        r_speed, r_force, headroom = self.get_robust_force_params()
        return motor_force_at_velocity(
            v_wheel=v_wheel,
            wheel_radius=r_speed,
            gearing=self.gearing,
            t_max=self.t_max_nm,
            v_max_rad_s=self.v_max_rad_s,
            torque_headroom=headroom,
            smooth=True,
            force_radius=r_force,
        )

    def apply_measured_motor(
        self,
        t_max_nm: Optional[float] = None,
        t_max_range: Optional[Tuple[float, float]] = None,
        no_load_speed_rad_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Replace the declared motor numbers with measured ones.

        The motor's torque-speed line is fixed by its two ends: the stall
        torque and the speed at which the torque reaches zero. The config
        can only declare both, and a declared number is a guess until
        somebody pushes the robot into a wall and lifts it off the ground.

        When the calibration measured them, the measurement is the number
        every tool uses: the optimizer's wheel-force constraint, the CPU
        audit, the Immrax validator and the exported controller all read
        this one config (KI-010). Before this, the optimizer planned
        against the declared motor while the validator checked the measured
        one, so a plan could satisfy its own constraint and still trip the
        validator.

        The stall-torque range follows the measured number as well, unless
        the config pinned the range itself: the range that was derived from
        the declared number describes the declared number.

        Args:
            t_max_nm: the measured stall torque, in N*m.
            t_max_range: the measured stall torque's interval, in N*m.
            no_load_speed_rad_s: the measured no-load speed, in rad/s at the
                motor. This is the speed at which the motor makes no torque.

        Returns:
            What changed, for the report. Empty when nothing was applied.
        """
        changes: Dict[str, Any] = {}

        if t_max_nm is not None and float(t_max_nm) > 0.0:
            value = float(t_max_nm)
            if abs(value - float(self.t_max_nm)) > 1e-12:
                changes["t_max_nm"] = {
                    "declared": float(self.t_max_nm),
                    "measured": value,
                }
                self.t_max_nm = value
                if not self.declared_t_max_range:
                    # The range was derived from the declared number, so it
                    # has to follow the measured one.
                    self.uncertainties["t_max_range"] = (
                        value * 0.75, value * 0.95)
        if t_max_range is not None:
            low, high = float(t_max_range[0]), float(t_max_range[1])
            if low > 0.0 and high >= low:
                self.uncertainties["t_max_range"] = (low, high)
                changes["t_max_range"] = [low, high]

        if no_load_speed_rad_s is not None and float(no_load_speed_rad_s) > 0.0:
            value = float(no_load_speed_rad_s)
            if abs(value - float(self.v_max_rad_s)) > 1e-12:
                changes["v_max_rad_s"] = {
                    "declared": float(self.v_max_rad_s),
                    "measured": value,
                }
                self.v_max_rad_s = value

        return changes

    def apply_sysid_fit(self, twin: Any) -> Dict[str, Any]:
        """Complete merge of a digital-twin fit into this RobotConfig.

        Updates effective wheel_radius, track_width, gear_backlash_m,
        motor curve (t_max_nm, v_max_rad_s, t_max_range), rolling resistance,
        and 95% confidence intervals simultaneously so that the optimizer,
        CPU validator, and Immrax validator plan and validate against the same
        calibrated model.
        """
        if isinstance(twin, dict):
            from overboard.sysid.digital_twin import DigitalTwinConfig
            twin = DigitalTwinConfig.from_dict(twin)

        changes: Dict[str, Any] = {}

        # 1. Motor curve merge (stall torque & no-load speed)
        motor_kwargs = twin.measured_motor(gearing=self.gearing)
        motor_changes = self.apply_measured_motor(**motor_kwargs)
        changes.update(motor_changes)

        # 2. DriveBase geometry and Odometry scale merge
        drivebase = getattr(twin, "drivebase", {}) or {}
        new_r = drivebase.get("wheel_radius_m")
        new_track = drivebase.get("track_width_m")

        base_r = float(new_r) if (new_r is not None and float(new_r) > 0.0) else float(self.wheel_radius)
        base_track = float(new_track) if (new_track is not None and float(new_track) > 0.0) else float(self.track_width)

        odometry = getattr(twin, "odometry", None)
        if odometry is not None and getattr(odometry, "calibrated", False):
            dist_scale = float(getattr(odometry, "distance_scale", 1.0))
            head_scale = float(getattr(odometry, "heading_scale", 1.0))
            effective_r = base_r * dist_scale
            effective_track = base_track / head_scale if head_scale > 0.0 else base_track
        else:
            effective_r = base_r
            effective_track = base_track

        if abs(effective_r - self.wheel_radius) > 1e-9:
            changes["wheel_radius"] = {
                "declared": float(self.wheel_radius),
                "measured": effective_r,
            }
            self.wheel_radius = effective_r
            self.uncertainties["wheel_radius_range"] = (effective_r * 0.99, effective_r * 1.01)

        if abs(effective_track - self.track_width) > 1e-9:
            changes["track_width"] = {
                "declared": float(self.track_width),
                "measured": effective_track,
            }
            self.track_width = effective_track
            self.uncertainties["track_width_range"] = (effective_track * 0.99, effective_track * 1.01)

        # 3. Gear backlash merge
        backlash_dict = getattr(twin, "backlash", {}) or {}
        if isinstance(backlash_dict, dict) and backlash_dict.get("calibrated", False):
            b_val = backlash_dict.get("backlash_m") or backlash_dict.get("gear_backlash_m")
            if b_val is not None and float(b_val) > 0.0:
                b_m = float(b_val)
                self.uncertainties["gear_backlash_m"] = (b_m * 0.8, b_m * 1.2)
                changes["gear_backlash_m"] = list(self.uncertainties["gear_backlash_m"])
            elif "reverse_delay_s" in backlash_dict:
                delay = float(backlash_dict["reverse_delay_s"] or 0.0)
                changes["backlash_reverse_delay_s"] = delay

        # 4. Rolling resistance / braking slip merge
        slip = getattr(twin, "slip", None)
        if slip is not None and getattr(slip, "calibrated", False):
            median_slip = getattr(slip, "brake_ratio_median", None)
            if median_slip is not None:
                self.rolling_resistance = float(median_slip)
                changes["rolling_resistance"] = float(median_slip)

        # 5. Actuator gain confidence interval
        actuator = getattr(twin, "actuator_straight", None)
        if actuator is not None and getattr(actuator, "calibrated", False):
            gain_range = twin.actuator_gain_range()
            self.uncertainties["actuator_gain_range"] = gain_range
            changes["actuator_gain_range"] = list(gain_range)

        # 6. Drift confidence interval
        drift = getattr(twin, "drift", None)
        if drift is not None and getattr(drift, "calibrated", False):
            self.uncertainties["drift_deg_per_m"] = tuple(drift.drift_range)
            changes["drift_deg_per_m"] = list(drift.drift_range)

        return changes


class DifferentialDriveModel:
    """
    Differential drive robot dynamics model.

    Calculates required wheel forces for given motion and checks
    physical constraints (motor limits, traction limits).
    """

    def __init__(self, config: RobotConfig) -> None:
        """
        Initialize dynamics model with robot configuration.

        Args:
            config: RobotConfig object containing physical parameters
        """
        self.cfg = config

    def get_dynamics(self, vl: float, vr: float, al: float, ar: float) -> Tuple[float, float]:
        """
        Calculate required wheel forces for given velocities and accelerations.
        Delegates to canonical diff_drive_wheel_forces decomposition.
        """
        return diff_drive_wheel_forces(al, ar, self.cfg.mass, self.cfg.inertia, self.cfg.track_width)

    def get_dynamics_symbolic(self, vl: ca.MX, vr: ca.MX, al: ca.MX, ar: ca.MX) -> Tuple[ca.MX, ca.MX]:
        """
        CasADi symbolic version of get_dynamics.
        """
        return diff_drive_wheel_forces(al, ar, self.cfg.mass, self.cfg.inertia, self.cfg.track_width)

    def get_wheel_normal_forces(self, vl: float, vr: float, al: float, ar: float) -> Tuple[float, float]:
        """
        Normal force on each wheel with lateral (turning) weight transfer.

        Longitudinal acceleration is intentionally not included: a two-wheel
        differential robot has no front/rear axle to shift load between
        (KI-010). The formula is shared with the JAX model and Immrax.

        Args:
            vl: Left wheel velocity (m/s)
            vr: Right wheel velocity (m/s)
            al: Left wheel acceleration (m/s²), unused (kept for callers)
            ar: Right wheel acceleration (m/s²), unused (kept for callers)

        Returns:
            Tuple of (nl, nr) - Normal forces on left and right wheels (N)
        """
        v, omega = diff_drive_fk(vl, vr, self.cfg.track_width)
        return wheel_normal_forces(
            self.cfg.mass, self.cfg.g, CG_HEIGHT_M, self.cfg.track_width, v * omega
        )

    def get_max_wheel_force_symbolic(self, al: ca.MX, ar: ca.MX, apply_headroom: bool = True) -> ca.MX:
        """
        Worst-case per-wheel force magnitude over the declared inertia and
        track-width ranges.

        Immrax evaluates each wheel force as an interval; the optimizer must
        bound the same quantity or it can satisfy its own check and still trip
        the validator. With fl = f_total/2 - I*(ar-al)/b^2 and
        fr = f_total/2 + I*(ar-al)/b^2, both wheels obey
        |f| <= |f_total|/2 + max(I)*|ar-al|/min(b)^2.
        """
        if apply_headroom:
            inertia = max(self.cfg.uncertainties["inertia_kg_m2"])
            track_width = min(self.cfg.uncertainties["track_width_range"])
        else:
            inertia = self.cfg.inertia
            track_width = self.cfg.track_width
        f_total = self.cfg.mass * (al + ar) / 2.0
        split = inertia * (ar - al) / track_width**2
        # sqrt(x^2 + eps) is the repo's smooth |x| (see get_max_force_symbolic).
        return 0.5 * ca.sqrt(f_total**2 + 1e-12) + ca.sqrt(split**2 + 1e-12)

    def get_traction_limit_symbolic(self, vl: ca.MX, vr: ca.MX, apply_headroom: bool = True) -> ca.MX:
        """
        Per-wheel traction limit matching Immrax's per-wheel slip check.

        Both wheels are held to the smaller (inner-wheel) normal force over the
        lateral transfer, using the declared cof and track-width worst cases
        when headroom is enabled. Immrax's interval lower bound for either
        wheel is never larger than this limit.
        """
        cof = self.cfg.cof
        track_width = self.cfg.track_width
        if apply_headroom:
            cof = min(self.cfg.uncertainties["cof"])
            track_width = min(self.cfg.uncertainties["track_width_range"])
        v = (vl + vr) / 2.0
        omega = (vr - vl) / track_width
        nl, nr = wheel_normal_forces(
            self.cfg.mass, self.cfg.g, CG_HEIGHT_M, track_width, v * omega
        )
        # nl + nr = m*g; min(nl, nr) is the conservative per-wheel normal force.
        min_normal = 0.5 * (nl + nr) - 0.5 * ca.sqrt((nl - nr) ** 2 + 1e-12)
        return cof * ca.fmax(min_normal, 0)

    def check_constraints(self, vl: float, vr: float, al: float, ar: float, apply_headroom: bool = True) -> Dict[str, float]:
        """
        Check if given state violates motor or traction limits.

        Args:
            vl: Left wheel velocity (m/s)
            vr: Right wheel velocity (m/s)
            al: Left wheel acceleration (m/s²)
            ar: Right wheel acceleration (m/s²)
            apply_headroom: If True, applies safety margin for real-world tracking

        Returns:
            Dictionary with violation details:
            - left_motor_violation: N (0 if no violation)
            - right_motor_violation: N (0 if no violation)
            - left_wheel_slip: N (0 if no slip)
            - right_wheel_slip: N (0 if no slip)
            - traction_violation: N (0 if no violation, legacy total check)
        """
        fl, fr = self.get_dynamics(vl, vr, al, ar)

        if apply_headroom:
            fl_max = self.cfg.get_max_force_at_velocity_robust(vl)
            fr_max = self.cfg.get_max_force_at_velocity_robust(vr)
        else:
            fl_max = self.cfg.get_max_force_at_velocity(vl, apply_headroom=False)
            fr_max = self.cfg.get_max_force_at_velocity(vr, apply_headroom=False)

        nl, nr = self.get_wheel_normal_forces(vl, vr, al, ar)
        res = check_drivetrain_constraints(
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
        return {
            "left_motor_violation": res["left_motor_violation"],
            "right_motor_violation": res["right_motor_violation"],
            "left_wheel_slip": res["left_wheel_slip"],
            "right_wheel_slip": res["right_wheel_slip"],
            "traction_violation": res["traction_violation"],
            "left_normal_force": res["left_normal_force"],
            "right_normal_force": res["right_normal_force"],
        }
