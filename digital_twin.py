"""
Load and validate the optional digital-twin fit produced by ``sysid_fit.py``.

The fit is deliberately separate from the robot configuration. ``robot_model``
describes the physical robot; this module describes the measured response of
the deployed controller and DriveBase. ``ImmraxValidator`` can opt into the
fit without changing the default kinematic model.

After loading, the validator currently applies:

* the straight-line steady-state wheel-speed gain range,
* the straight-line first-order actuator time constant,
* the straight-line command delay,
* the measured odometry distance/heading scale in feedback.

The braking-slip block is loaded and reported, but is not applied yet: the
plant has no braking state, so using it as a permanent speed scale would be a
fake correction. It is kept for the next calibration/model step.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import json
import math


def _optional_float(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _positive_float(value: Any, name: str, default: float) -> float:
    result = _optional_float(value, name)
    if result is None:
        return float(default)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {result}")
    return result


def _optional_positive_float(value: Any, name: str) -> Optional[float]:
    result = _optional_float(value, name)
    if result is not None and result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {result}")
    return result


def _non_negative_float(value: Any, name: str, default: float = 0.0) -> float:
    result = _optional_float(value, name)
    if result is None:
        return float(default)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {result}")
    return result


def _range(payload: Dict[str, Any], key: str,
           default: Tuple[float, float]) -> Tuple[float, float]:
    raw = payload.get(key)
    if raw is None:
        return default
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{key} must be a two-element range, got {raw!r}")
    lower = _optional_float(raw[0], f"{key}[0]")
    upper = _optional_float(raw[1], f"{key}[1]")
    if lower is None or upper is None:
        raise ValueError(f"{key} must contain two numbers")
    if lower > upper:
        raise ValueError(f"{key} lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


@dataclass(frozen=True)
class ActuatorFit:
    """First-order response fitted from a straight or turn step test."""

    calibrated: bool = False
    gain: Optional[float] = None
    gain_range: Tuple[float, float] = (1.0, 1.0)
    tau_s: Optional[float] = None
    delay_s: Optional[float] = None
    accel_limit: Optional[float] = None
    rmse: Optional[float] = None
    n_steps: int = 0

    def __post_init__(self):
        if self.gain_range[0] > self.gain_range[1]:
            raise ValueError(
                f"actuator gain range {self.gain_range} has lower > upper")
        if self.gain_range[0] <= 0.0:
            raise ValueError(
                f"actuator gain range {self.gain_range} must be positive")
        if self.tau_s is not None and self.tau_s < 0.0:
            raise ValueError(f"actuator tau_s must be >= 0, got {self.tau_s}")
        if self.delay_s is not None and self.delay_s < 0.0:
            raise ValueError(f"actuator delay_s must be >= 0, got {self.delay_s}")


@dataclass(frozen=True)
class OdometryFit:
    """Measured final-pose correction for DriveBase odometry."""

    calibrated: bool = False
    distance_scale: float = 1.0
    heading_scale: float = 1.0
    n_distance_sources: int = 0
    n_heading_sources: int = 0

    def __post_init__(self):
        if self.distance_scale <= 0.0:
            raise ValueError(
                f"odometry distance_scale must be > 0, got {self.distance_scale}")
        if self.heading_scale <= 0.0:
            raise ValueError(
                f"odometry heading_scale must be > 0, got {self.heading_scale}")


@dataclass(frozen=True)
class SlipFit:
    """Bounded braking-slip estimate from measured stop poses."""

    calibrated: bool = False
    brake_ratio_median: Optional[float] = None
    brake_ratio_min: Optional[float] = None
    brake_ratio_max: Optional[float] = None

    def __post_init__(self):
        values = (
            self.brake_ratio_min,
            self.brake_ratio_median,
            self.brake_ratio_max,
        )
        present = [value for value in values if value is not None]
        if present and len(present) != 3:
            raise ValueError("slip summary must provide min, median, and max")
        if present and not (self.brake_ratio_min <= self.brake_ratio_median
                            <= self.brake_ratio_max):
            raise ValueError(
                "slip ratios must satisfy min <= median <= max, got "
                f"{(self.brake_ratio_min, self.brake_ratio_median, self.brake_ratio_max)}")


@dataclass(frozen=True)
class SensorFit:
    """Static sensor biases from the stationary capture."""

    calibrated: bool = False
    gyro_z_bias_deg_s: Optional[float] = None
    gyro_z_std_deg_s: Optional[float] = None
    ax_bias_m_s2: Optional[float] = None
    ay_bias_m_s2: Optional[float] = None


@dataclass(frozen=True)
class DriftFit:
    """How much the robot curves when it is told to drive straight.

    The plan's straight segments are straight, so a robot whose wheels are
    not parallel leaves the plan. The rate is in degrees per metre,
    counter-clockwise positive, like the trajectory's heading.
    """

    calibrated: bool = False
    drift_deg_per_m: float = 0.0
    drift_range: Tuple[float, float] = (0.0, 0.0)
    n_sources: int = 0

    def __post_init__(self):
        if self.drift_range[0] > self.drift_range[1]:
            raise ValueError(
                f"drift range {self.drift_range} has lower > upper")

    def drift_rad_per_m(self) -> float:
        """The drift in radians per metre, for the plant."""
        return float(math.radians(self.drift_deg_per_m))


@dataclass(frozen=True)
class StallTorqueFit:
    """The motor's stall torque, measured by pushing into a wall."""

    calibrated: bool = False
    t_max_nm: Optional[float] = None
    t_max_range: Tuple[float, float] = (0.0, 0.0)
    n_sources: int = 0

    def __post_init__(self):
        if self.t_max_nm is not None and self.t_max_nm <= 0.0:
            raise ValueError(
                f"stall torque must be positive, got {self.t_max_nm}")
        if self.t_max_range[0] > self.t_max_range[1]:
            raise ValueError(
                f"stall torque range {self.t_max_range} has lower > upper")


@dataclass(frozen=True)
class DrivetrainFit:
    """The motor's free-running speed, measured with the wheels lifted.

    The torque-speed line the force model uses is fixed by its two ends:
    the stall torque and the speed at which the torque reaches zero. The
    stall torque is measured by pushing the robot into a wall
    (``StallTorqueFit``); this is the other end, measured with the robot off
    the ground so the motor pushes nothing.

    The number is the wheel's free-running speed in rad/s. The config's
    ``v_max_rad_s`` is the same quantity at the motor, so the gearing
    multiplies it on the way into the config.
    """

    calibrated: bool = False
    no_load_speed_rad_s: Optional[float] = None
    no_load_speed_range: Tuple[float, float] = (0.0, 0.0)
    n_sources: int = 0

    def __post_init__(self):
        if self.no_load_speed_rad_s is not None and self.no_load_speed_rad_s <= 0.0:
            raise ValueError(
                "no-load speed must be positive, got "
                f"{self.no_load_speed_rad_s}")
        if self.no_load_speed_range[0] > self.no_load_speed_range[1]:
            raise ValueError(
                f"no-load speed range {self.no_load_speed_range} has "
                "lower > upper")


def _actuator_from_dict(payload: Dict[str, Any]) -> ActuatorFit:
    payload = payload if isinstance(payload, dict) else {}
    calibrated = bool(payload.get("calibrated", False))
    gain = _optional_float(payload.get("gain"), "actuator.gain")
    gain_min = _optional_float(payload.get("gain_min"), "actuator.gain_min")
    gain_max = _optional_float(payload.get("gain_max"), "actuator.gain_max")
    # The 95% interval over the repeats of the test. It is the honest
    # uncertainty of the gain, so the validator uses it as the band when the
    # fit has it. The observed min/max are the fallback for a fit with a
    # single run, where there is no interval.
    gain_ci = payload.get("gain_ci95")
    if isinstance(gain_ci, (list, tuple)) and len(gain_ci) == 2:
        ci_low = _optional_float(gain_ci[0], "actuator.gain_ci95")
        ci_high = _optional_float(gain_ci[1], "actuator.gain_ci95")
    else:
        ci_low = ci_high = None

    if ci_low is not None and ci_high is not None:
        gain_range = (ci_low, ci_high)
    elif gain_min is not None and gain_max is not None:
        gain_range = (gain_min, gain_max)
    elif gain is not None:
        gain_range = (gain, gain)
    else:
        gain_range = (1.0, 1.0)

    tau_s = _optional_float(payload.get("tau_s"), "actuator.tau_s")
    delay_s = _optional_float(payload.get("delay_s"), "actuator.delay_s")
    accel_limit = _optional_float(payload.get("accel_limit"), "actuator.accel_limit")
    rmse = _optional_float(payload.get("rmse"), "actuator.rmse")
    n_steps = int(payload.get("n_steps", 0) or 0)

    return ActuatorFit(
        calibrated=calibrated,
        gain=gain,
        gain_range=gain_range,
        tau_s=tau_s,
        delay_s=delay_s,
        accel_limit=accel_limit,
        rmse=rmse,
        n_steps=n_steps,
    )


def _drift_from_dict(payload: Dict[str, Any]) -> DriftFit:
    payload = payload if isinstance(payload, dict) else {}
    drift = _optional_float(
        payload.get("drift_deg_per_m"), "drift.drift_deg_per_m")
    if drift is None:
        drift = 0.0
    ci = payload.get("drift_deg_per_m_ci95")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        low = _optional_float(ci[0], "drift.drift_deg_per_m_ci95")
        high = _optional_float(ci[1], "drift.drift_deg_per_m_ci95")
    else:
        low = high = None
    if low is not None and high is not None:
        drift_range = (low, high)
    else:
        drift_range = (drift, drift)
    return DriftFit(
        calibrated=bool(payload.get("calibrated", False)),
        drift_deg_per_m=drift,
        drift_range=drift_range,
        n_sources=int(payload.get("n_sources", 0) or 0),
    )


def _stall_torque_from_dict(payload: Dict[str, Any]) -> StallTorqueFit:
    payload = payload if isinstance(payload, dict) else {}
    torque = _optional_float(
        payload.get("t_max_nm"), "stall_torque.t_max_nm")
    ci = payload.get("t_max_nm_ci95")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        low = _optional_float(ci[0], "stall_torque.t_max_nm_ci95")
        high = _optional_float(ci[1], "stall_torque.t_max_nm_ci95")
    else:
        low = high = None
    if low is not None and high is not None:
        torque_range = (low, high)
    elif torque is not None:
        torque_range = (torque, torque)
    else:
        torque_range = (0.0, 0.0)
    return StallTorqueFit(
        calibrated=bool(payload.get("calibrated", False)),
        t_max_nm=torque,
        t_max_range=torque_range,
        n_sources=int(payload.get("n_sources", 0) or 0),
    )


def _drivetrain_from_dict(payload: Dict[str, Any]) -> DrivetrainFit:
    payload = payload if isinstance(payload, dict) else {}
    speed = _optional_float(
        payload.get("no_load_speed_rad_s"), "drivetrain.no_load_speed_rad_s")
    ci = payload.get("no_load_speed_rad_s_ci95")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        low = _optional_float(ci[0], "drivetrain.no_load_speed_rad_s_ci95")
        high = _optional_float(ci[1], "drivetrain.no_load_speed_rad_s_ci95")
    else:
        low = high = None
    if low is not None and high is not None:
        speed_range = (low, high)
    elif speed is not None:
        speed_range = (speed, speed)
    else:
        speed_range = (0.0, 0.0)
    return DrivetrainFit(
        calibrated=bool(payload.get("calibrated", False)),
        no_load_speed_rad_s=speed,
        no_load_speed_range=speed_range,
        n_sources=int(payload.get("n_sources", 0) or 0),
    )


def _odometry_from_dict(payload: Dict[str, Any]) -> OdometryFit:
    payload = payload if isinstance(payload, dict) else {}
    return OdometryFit(
        calibrated=bool(payload.get("calibrated", False)),
        distance_scale=_positive_float(
            payload.get("distance_scale"), "odometry.distance_scale", 1.0),
        heading_scale=_positive_float(
            payload.get("heading_scale"), "odometry.heading_scale", 1.0),
        n_distance_sources=int(payload.get("n_distance_sources", 0) or 0),
        n_heading_sources=int(payload.get("n_heading_sources", 0) or 0),
    )


def _slip_from_dict(payload: Dict[str, Any]) -> SlipFit:
    payload = payload if isinstance(payload, dict) else {}
    return SlipFit(
        calibrated=bool(payload.get("calibrated", False)),
        brake_ratio_median=_optional_float(
            payload.get("brake_ratio_median"), "slip.brake_ratio_median"),
        brake_ratio_min=_optional_float(
            payload.get("brake_ratio_min"), "slip.brake_ratio_min"),
        brake_ratio_max=_optional_float(
            payload.get("brake_ratio_max"), "slip.brake_ratio_max"),
    )


def _sensors_from_dict(payload: Dict[str, Any]) -> SensorFit:
    payload = payload if isinstance(payload, dict) else {}
    return SensorFit(
        calibrated=bool(payload.get("calibrated", False)),
        gyro_z_bias_deg_s=_optional_float(
            payload.get("gyro_z_bias_deg_s"), "sensors.gyro_z_bias_deg_s"),
        gyro_z_std_deg_s=_optional_float(
            payload.get("gyro_z_std_deg_s"), "sensors.gyro_z_std_deg_s"),
        ax_bias_m_s2=_optional_float(
            payload.get("ax_bias_m_s2"), "sensors.ax_bias_m_s2"),
        ay_bias_m_s2=_optional_float(
            payload.get("ay_bias_m_s2"), "sensors.ay_bias_m_s2"),
    )


def _drivebase_from_dict(payload: Any,
                         data: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    wheel_radius = _optional_positive_float(
        payload.get("wheel_radius_m", data.get("wheel_radius_m")),
        "drivebase.wheel_radius_m")
    track_width = _optional_positive_float(
        payload.get("track_width_m", data.get("track_width_m")),
        "drivebase.track_width_m")
    name = payload.get("name")
    source = payload.get("source")
    result = {}
    if wheel_radius is not None:
        result["wheel_radius_m"] = wheel_radius
    if track_width is not None:
        result["track_width_m"] = track_width
    if name is not None:
        result["name"] = str(name)
    if source is not None:
        result["source"] = str(source)
    return result


@dataclass(frozen=True)
class DigitalTwinConfig:
    """Parsed ``sysid_fit.json``. Disabled unless explicitly opted in."""

    actuator_straight: ActuatorFit = field(default_factory=ActuatorFit)
    actuator_turn: ActuatorFit = field(default_factory=ActuatorFit)
    sensors: SensorFit = field(default_factory=SensorFit)
    odometry: OdometryFit = field(default_factory=OdometryFit)
    slip: SlipFit = field(default_factory=SlipFit)
    drift: DriftFit = field(default_factory=DriftFit)
    stall_torque: StallTorqueFit = field(default_factory=StallTorqueFit)
    drivetrain: DrivetrainFit = field(default_factory=DrivetrainFit)
    backlash: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    drivebase: Dict[str, Any] = field(default_factory=dict)
    source: Tuple[str, ...] = ()
    enabled: bool = False

    @classmethod
    def disabled(cls) -> "DigitalTwinConfig":
        return cls()

    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  enabled: Optional[bool] = None) -> "DigitalTwinConfig":
        if not isinstance(data, dict):
            raise ValueError("digital-twin fit must be a JSON object")

        actuator = data.get("actuator", {})
        actuator = actuator if isinstance(actuator, dict) else {}
        straight = _actuator_from_dict(actuator.get("straight", {}))
        turn = _actuator_from_dict(actuator.get("turn", {}))
        odometry = _odometry_from_dict(data.get("odometry", {}))
        slip = _slip_from_dict(data.get("slip", {}))
        sensors = _sensors_from_dict(data.get("sensors", {}))
        drift = _drift_from_dict(data.get("drift", {}))
        stall_torque = _stall_torque_from_dict(data.get("stall_torque", {}))
        drivetrain = _drivetrain_from_dict(data.get("drivetrain", {}))
        drivebase = _drivebase_from_dict(data.get("drivebase"), data)

        calibrated = any((
            straight.calibrated,
            odometry.calibrated,
        ))
        if enabled is None:
            enabled = calibrated

        backlash = data.get("backlash", {})
        validation = data.get("validation", {})
        source = data.get("source", [])
        if isinstance(source, str):
            source = [source]
        if not isinstance(source, (list, tuple)):
            source = []

        return cls(
            actuator_straight=straight,
            actuator_turn=turn,
            sensors=sensors,
            odometry=odometry,
            slip=slip,
            drift=drift,
            stall_torque=stall_torque,
            drivetrain=drivetrain,
            backlash=dict(backlash) if isinstance(backlash, dict) else {},
            validation=dict(validation) if isinstance(validation, dict) else {},
            drivebase=drivebase,
            source=tuple(str(item) for item in source),
            enabled=bool(enabled),
        )

    @classmethod
    def load(cls, path: str) -> "DigitalTwinConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def actuator_gain_range(self) -> Tuple[float, float]:
        """Straight-line wheel-speed gain interval; identity if uncalibrated."""
        if self.actuator_straight.calibrated:
            return tuple(self.actuator_straight.gain_range)
        return (1.0, 1.0)

    def actuator_tau_s(self) -> float:
        if not self.actuator_straight.calibrated:
            return 0.0
        return max(0.0, float(self.actuator_straight.tau_s or 0.0))

    def actuator_delay_s(self) -> float:
        if not self.actuator_straight.calibrated:
            return 0.0
        return max(0.0, float(self.actuator_straight.delay_s or 0.0))

    def actuator_accel_limit(self) -> Optional[float]:
        """Measured wheel-acceleration limit in m/s^2, or None if unknown.

        The plant clamps the wheel-speed state to this rate, so the model
        cannot accelerate faster than the robot did.
        """
        if not self.actuator_straight.calibrated:
            return None
        value = self.actuator_straight.accel_limit
        if value is None:
            return None
        value = float(value)
        return value if value > 0.0 else None

    def odometry_distance_scale(self) -> float:
        """Scale between the robot's distance reading and the truth."""
        if not self.odometry.calibrated:
            return 1.0
        return float(self.odometry.distance_scale)

    def odometry_heading_scale(self) -> float:
        """Scale between the robot's heading rate and the truth."""
        if not self.odometry.calibrated:
            return 1.0
        return float(self.odometry.heading_scale)

    def drift_rad_per_m(self) -> float:
        """How much the robot curves per metre when told to drive straight.

        Zero when it was not measured, so a plan without the measurement
        behaves exactly as before.
        """
        if not self.drift.calibrated:
            return 0.0
        return self.drift.drift_rad_per_m()

    def stall_torque_nm(self) -> Optional[float]:
        """The measured motor stall torque, or None if it was not measured."""
        if not self.stall_torque.calibrated:
            return None
        value = self.stall_torque.t_max_nm
        return None if value is None else float(value)

    def stall_torque_range_nm(self) -> Optional[Tuple[float, float]]:
        """The measured stall torque and its interval, or None."""
        if not self.stall_torque.calibrated:
            return None
        if self.stall_torque.t_max_nm is None:
            return None
        return tuple(self.stall_torque.t_max_range)

    def no_load_speed_rad_s(self) -> Optional[float]:
        """The measured free-running speed at the wheel, or None.

        The motor's free-running speed is the speed at which it makes no
        torque, so it is the other end of the torque-speed line the force
        model uses. It is reported at the wheel; the config's ``v_max_rad_s``
        is the same quantity at the motor, so the gearing multiplies it.
        """
        if not self.drivetrain.calibrated:
            return None
        if self.drivetrain.no_load_speed_rad_s is None:
            return None
        return float(self.drivetrain.no_load_speed_rad_s)

    def measured_motor(self, gearing: float = 1.0) -> Dict[str, Any]:
        """The measured motor numbers, ready for ``RobotConfig``.

        A measurement always wins over a declared number, but only for the
        quantities that were actually measured. The keys that are absent are
        the ones the calibration did not cover, and the config keeps its
        declared value for those.
        """
        result: Dict[str, Any] = {}
        stall = self.stall_torque_nm()
        if stall is not None:
            result["t_max_nm"] = stall
            stall_range = self.stall_torque_range_nm()
            if stall_range is not None:
                result["t_max_range"] = stall_range
        speed = self.no_load_speed_rad_s()
        if speed is not None:
            result["no_load_speed_rad_s"] = speed * float(gearing)
        return result

    def gyro_z_bias_rad_s(self) -> float:
        """Stationary gyro z bias as a yaw-rate offset, in rad/s.

        The robot's heading estimate drifts by this much per second, which is
        what separates the robot's belief from the truth over a run.
        """
        if not self.sensors.calibrated:
            return 0.0
        bias = self.sensors.gyro_z_bias_deg_s
        if bias is None:
            return 0.0
        return math.radians(float(bias))

    def brake_shortfall(self) -> float:
        """Fraction by which a braking run travels less than its odometry says.

        The fit stores 1 - measured/expected, so 0.1 means the robot covered
        10% less ground than its encoders reported while braking.
        """
        if not self.slip.calibrated:
            return 0.0
        median = self.slip.brake_ratio_median
        if median is None:
            return 0.0
        return float(median)

    def summary(self) -> Dict[str, Any]:
        """Compact JSON-safe description for reports and benchmark stats."""
        return {
            "enabled": self.enabled,
            "actuator": {
                "straight": {
                    "calibrated": self.actuator_straight.calibrated,
                    "gain": self.actuator_straight.gain,
                    "gain_range": list(self.actuator_straight.gain_range),
                    "tau_s": self.actuator_straight.tau_s,
                    "delay_s": self.actuator_straight.delay_s,
                    "accel_limit": self.actuator_straight.accel_limit,
                    "n_steps": self.actuator_straight.n_steps,
                },
                "turn": {
                    "calibrated": self.actuator_turn.calibrated,
                    "gain": self.actuator_turn.gain,
                    "gain_range": list(self.actuator_turn.gain_range),
                    "tau_s": self.actuator_turn.tau_s,
                    "delay_s": self.actuator_turn.delay_s,
                    "n_steps": self.actuator_turn.n_steps,
                },
            },
            "odometry": {
                "calibrated": self.odometry.calibrated,
                "distance_scale": self.odometry.distance_scale,
                "heading_scale": self.odometry.heading_scale,
            },
            "slip": {
                "calibrated": self.slip.calibrated,
                "brake_ratio_median": self.slip.brake_ratio_median,
                "brake_ratio_min": self.slip.brake_ratio_min,
                "brake_ratio_max": self.slip.brake_ratio_max,
                "applied": self.slip.calibrated,
            },
            "sensors": {
                "calibrated": self.sensors.calibrated,
                "gyro_z_bias_deg_s": self.sensors.gyro_z_bias_deg_s,
            },
            "drivebase": dict(self.drivebase),
            "drivetrain": {
                "calibrated": self.drivetrain.calibrated,
                "no_load_speed_rad_s": self.drivetrain.no_load_speed_rad_s,
                "no_load_speed_range": list(self.drivetrain.no_load_speed_range),
                "n_sources": self.drivetrain.n_sources,
            },
            "applied": {
                "actuator_gain_range": list(self.actuator_gain_range()),
                "actuator_tau_s": self.actuator_tau_s(),
                "command_delay_s": self.actuator_delay_s(),
                "actuator_accel_limit": self.actuator_accel_limit(),
                "odometry_scale": self.odometry.calibrated,
                "integrated_odometry": self.odometry.calibrated,
                "gyro_z_bias_rad_s": self.gyro_z_bias_rad_s(),
                "slip_model": self.slip.calibrated,
                "measured_motor": bool(self.measured_motor()),
            },
            "source": list(self.source),
        }


def load_sysid_fit(path: str) -> DigitalTwinConfig:
    """Load a ``sysid_fit.json`` file into a validated configuration."""
    return DigitalTwinConfig.load(path)
