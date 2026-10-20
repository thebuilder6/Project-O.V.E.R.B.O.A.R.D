from typing import Dict, Tuple, Any, Optional
import numpy as np
import casadi as ca

# Center-of-mass height used by every weight-transfer calculation. Kept in
# one place so the optimizer, CPU audit, JAX model, and Immrax agree.
CG_HEIGHT_M = 0.05


def wheel_normal_forces(
    mass: float,
    g: float,
    h_cg: float,
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

    Shared by the optimizer, the CPU audit, the JAX model, and Immrax so all
    four use the same per-wheel limit. Types may be floats, JAX arrays, or
    Immrax Intervals.
    """
    base_normal = (mass * g) / 2.0
    # Immrax Intervals only implement left-hand operators, so the interval (or
    # array) term leads every expression; plain floats and CasADi MX work too.
    lateral_transfer = centripetal_accel * (mass * h_cg) / track_width
    return (-lateral_transfer) + base_normal, lateral_transfer + base_normal


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

        self.mass = get_val("mass", self.DEFAULT_MASS)
        self.inertia = get_val("inertia", self.DEFAULT_INERTIA)
        self.track_width = get_val("track_width", self.DEFAULT_TRACK_WIDTH, "differentialTrackWidth")
        self.wheel_radius = get_val("wheel_radius", self.DEFAULT_WHEEL_RADIUS, "radius")
        self.v_max_rad_s = get_val("v_max_rad_s", self.DEFAULT_VMAX, "vmax")
        self.t_max_nm = get_val("t_max_nm", self.DEFAULT_TMAX, "tmax")
        self.gearing = get_val("gearing", self.DEFAULT_GEARING)
        self.cof = get_val("cof", self.DEFAULT_COF)
        self.g = get_val("gravity", self.GRAVITY)
        self.torque_headroom = get_val("torque_headroom", self.DEFAULT_TORQUE_HEADROOM, "torqueHeadroom")
        self.speed_headroom = get_val("speed_headroom", self.DEFAULT_SPEED_HEADROOM, "speedHeadroom")

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
        }

        # Whether the config declared the stall-torque range itself. A
        # measured stall torque replaces the declared number, and the range
        # that was derived from the declared number has to follow it, unless
        # the config pinned the range on purpose.
        self.declared_t_max_range = "t_max_range" in uncertainty_cfg or "t_max_range" in robot_cfg

        # Store multiverse config if present
        self.multiverse_config = config_dict.get("multiverse", {})

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
        omega = (v_wheel / self.wheel_radius) * self.gearing
        torque = self.t_max_nm * (1.0 - abs(omega) / self.v_max_rad_s)
        torque = max(0, torque)
        force = (torque / self.wheel_radius) * self.gearing
        if apply_headroom:
            force *= self.torque_headroom
        return force

    def max_linear_speed(self, apply_headroom: bool = True) -> float:
        """No-load linear speed of the wheel (m/s)."""
        speed = self.v_max_rad_s * self.wheel_radius
        if apply_headroom:
            speed *= self.speed_headroom
        return speed

    def get_max_force_symbolic(self, v_wheel: ca.MX, apply_headroom: bool = True) -> ca.MX:
        """
        CasADi symbolic version of get_max_force_at_velocity.
        """
        omega = (v_wheel / self.wheel_radius) * self.gearing
        # Use smooth approximation of absolute value to prevent IPOPT step computation failures at zero speed
        abs_omega = ca.sqrt(omega**2 + 1e-8)
        torque = self.t_max_nm * (1.0 - abs_omega / self.v_max_rad_s)
        # NOTE: clamping at zero means no braking force above no-load speed.
        torque = ca.fmax(0, torque)
        force = (torque / self.wheel_radius) * self.gearing
        if apply_headroom:
            force *= self.torque_headroom
        return force

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
        omega = (abs(v_wheel) / r_speed) * self.gearing
        torque = max(0.0, self.t_max_nm * (1.0 - omega / self.v_max_rad_s))
        force = (torque / r_force) * self.gearing
        return force * headroom

    def get_max_force_symbolic_robust(self, v_wheel: ca.MX) -> ca.MX:
        """CasADi symbolic version of get_max_force_at_velocity_robust."""
        r_speed, r_force, headroom = self.get_robust_force_params()
        omega = (v_wheel / r_speed) * self.gearing
        abs_omega = ca.sqrt(omega**2 + 1e-8)
        torque = self.t_max_nm * (1.0 - abs_omega / self.v_max_rad_s)
        torque = ca.fmax(0, torque)
        force = (torque / r_force) * self.gearing
        return force * headroom

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

        Uses differential drive kinematics:
        - Linear acceleration: a = (al + ar) / 2
        - Angular acceleration: alpha = (ar - al) / track_width

        Args:
            vl: Left wheel velocity (m/s)
            vr: Right wheel velocity (m/s)
            al: Left wheel acceleration (m/s²)
            ar: Right wheel acceleration (m/s²)

        Returns:
            Tuple of (fl, fr) - Left and right wheel forces (N)
        """
        # Linear and angular acceleration
        a = (al + ar) / 2.0
        alpha = (ar - al) / self.cfg.track_width

        # Required total force and moment
        f_total = self.cfg.mass * a
        m_total = self.cfg.inertia * alpha

        # Solve for individual wheel forces
        # f_total = fl + fr
        # m_total = (fr - fl) * (track_width / 2)
        fr = (f_total + (2.0 * m_total / self.cfg.track_width)) / 2.0
        fl = f_total - fr

        return fl, fr

    def get_dynamics_symbolic(self, vl: ca.MX, vr: ca.MX, al: ca.MX, ar: ca.MX) -> Tuple[ca.MX, ca.MX]:
        """
        CasADi symbolic version of get_dynamics.
        """
        a = (al + ar) / 2.0
        alpha = (ar - al) / self.cfg.track_width
        f_total = self.cfg.mass * a
        m_total = self.cfg.inertia * alpha
        fr = (f_total + (2.0 * m_total / self.cfg.track_width)) / 2.0
        fl = f_total - fr
        return fl, fr

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
        v = (vl + vr) / 2.0
        omega = (vr - vl) / self.cfg.track_width
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

        # Motor force limits. With headroom enabled, use the same worst-case
        # declared ranges as the optimizer and the Immrax validator. Without
        # headroom (benchmark mode), use the nominal curve.
        if apply_headroom:
            fl_max = self.cfg.get_max_force_at_velocity_robust(vl)
            fr_max = self.cfg.get_max_force_at_velocity_robust(vr)
        else:
            fl_max = self.cfg.get_max_force_at_velocity(vl, apply_headroom=False)
            fr_max = self.cfg.get_max_force_at_velocity(vr, apply_headroom=False)

        left_motor_violation = max(0, abs(fl) - fl_max)
        right_motor_violation = max(0, abs(fr) - fr_max)

        # Individual wheel slip detection with weight transfer
        nl, nr = self.get_wheel_normal_forces(vl, vr, al, ar)
        left_traction_limit = self.cfg.cof * nl
        right_traction_limit = self.cfg.cof * nr

        left_wheel_slip = max(0, abs(fl) - left_traction_limit)
        right_wheel_slip = max(0, abs(fr) - right_traction_limit)

        # Legacy total traction check (friction circle)
        traction_limit = self.cfg.cof * self.cfg.mass * self.cfg.g
        traction_violation = max(0, abs(fl) + abs(fr) - traction_limit)

        return {
            "left_motor_violation": left_motor_violation,
            "right_motor_violation": right_motor_violation,
            "left_wheel_slip": left_wheel_slip,
            "right_wheel_slip": right_wheel_slip,
            "traction_violation": traction_violation,
            "left_normal_force": nl,
            "right_normal_force": nr
        }
