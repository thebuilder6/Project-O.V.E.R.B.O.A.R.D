import jax
import jax.numpy as jnp
from jax import jit, vmap
import immrax
from immrax import Interval, interval
from immrax.inclusion import natif
from jax_robot_model import JAXRobotConfig, JAXDifferentialDriveModel
from robot_model import CG_HEIGHT_M, wheel_normal_forces
from jax_ramsete import JAXRamseteController, quantize_drive_command
from digital_twin import DigitalTwinConfig
from typing import List, Dict, Any, Tuple, Optional
import math

def unroll_trajectory_headings(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not samples: return []
    unrolled = []
    current_offset = 0.0
    last_h = samples[0]['heading']
    
    for s in samples:
        h = s['heading']
        if h - last_h > math.pi:
            current_offset -= 2 * math.pi
        elif h - last_h < -math.pi:
            current_offset += 2 * math.pi
            
        new_s = s.copy()
        new_s['heading'] = h + current_offset
        unrolled.append(new_s)
        last_h = h
        
    return unrolled


# Number of reference updates per sample interval for the legacy/continuous
# path used by the fake dynamics in tests. The real validator uses the
# deployed controller's 20 ms sample-and-hold loop below.
REACHABILITY_SUBSTEPS = 20

# ramsete_fast.py runs a 20 ms control loop. Integrate the plant with a
# smaller step so a held command is not applied as one large Euler step.
CONTROL_DT = 0.02
PLANT_SUBSTEPS = 4


def _interpolated_reference(s0: Dict[str, Any], s1: Dict[str, Any],
                            dt: float, track_width: float,
                            s: float,
                            ramsete_b: float = 2.0
                            ) -> Tuple[float, float, float, float, float, float]:
    """
    Reference interpolation matching pybricks_code/ramsete_fast.py.

    The deployed controller linearly interpolates x, y, heading and v between
    samples, and takes the turn rate as the *derivative* of the interpolated
    heading. It does not use a Hermite curve.

    Samples passed here are expected to have unrolled headings, so the
    controller's continuous heading error is the same angle the robot sees
    after its own shortest-angle interpolation. The heading difference itself
    is taken the short way, exactly as the firmware does, so the two agree
    even when a sample interval turns more than 180 degrees (KI-027).
    """
    v0 = (s0["vl"] + s0["vr"]) / 2.0
    v1 = (s1["vl"] + s1["vr"]) / 2.0
    omega0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
    omega1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)

    # The shortest-angle difference, like the firmware's. The unrolled
    # headings differ by the same amount for any turn below 180 degrees.
    dh = (s1["heading"] - s0["heading"] + math.pi) % (2.0 * math.pi) - math.pi

    ref_x = s0["x"] + (s1["x"] - s0["x"]) * s
    ref_y = s0["y"] + (s1["y"] - s0["y"]) * s
    ref_heading = s0["heading"] + dh * s
    ref_v = v0 + (v1 - v0) * s
    # The turn rate is the derivative of the interpolated heading, so the
    # heading and the turn rate describe the same motion.
    ref_omega = dh / dt if dt > 0 else omega0

    # ramsete_fast.py precomputes this per sample and linearly interpolates it.
    kr0 = math.sqrt(omega0**2 + ramsete_b * v0**2)
    kr1 = math.sqrt(omega1**2 + ramsete_b * v1**2)
    ref_kr = kr0 + (kr1 - kr0) * s
    return ref_x, ref_y, ref_heading, ref_v, ref_omega, ref_kr

# ==============================================================================
# 1. THE OPEN LOOP SYSTEM (Physical Plant Dynamics)
# ==============================================================================
def _wheel_gain(w):
    """Optional 7th disturbance entry: measured steady-state wheel gain.

    Older callers pass a six-entry w vector. Keeping this lookup optional
    preserves the default model exactly while letting the digital-twin path
    carry the fitted gain as an interval.
    """
    return w[6] if w.shape[0] > 6 else 1.0


class RobotOpenLoop(immrax.system.OpenLoopSystem):
    """Kinematic plant: the commanded wheel speed becomes motion at once.

    The firmware raises the DriveBase trajectory limits
    (``drivebase.settings(straight_acceleration=..., turn_acceleration=...)``,
    about 7800 mm/s² and 9300 deg/s² for 56 mm wheels) above the accelerations
    the planner uses, so the inner reference follows the command instead of
    ramping behind it. Instant tracking is therefore a fair model of the hub.
    The DriveBase PID and the whole-millimeter distance reading are still not
    modeled (KI-020).
    """

    def __init__(self, config: JAXRobotConfig):
        self.evolution = 'continuous'
        self.xlen = 3
        self.config = config

    def f(self, t, x, u, w):
        """
        u: Control inputs (Commanded wheel velocities) [vl_cmd, vr_cmd]
        """
        gain = _wheel_gain(w)
        vl_cmd, vr_cmd = u[0] * gain, u[1] * gain
        track_width = w[3]

        # Gear backlash is a POSITION dead-band, not a velocity offset.
        #
        # The old model subtracted backlash/2 from the commanded wheel SPEED.
        # `gear_backlash_m` is a distance in metres and the command is a
        # velocity in m/s, so that was a dimensional error: it made the robot
        # run permanently 0.225 mm/s slower than commanded, in both
        # directions, and grew a fake systematic lag of 0.225 mm/s * t into
        # the tracking error. A real dead-band is taken up once at a
        # direction reversal, after which the wheel turns at exactly the
        # commanded speed.
        #
        # The backlash is now carried as the position uncertainty it is, in
        # the reachable set (see compute_reachability), so the worst-case
        # total uncertainty at each sample is unchanged: the same magnitude,
        # moved from the tracking metric into the envelope, and symmetric
        # instead of a one-sided bias.
        vl_act = vl_cmd
        vr_act = vr_cmd

        # Kinematics
        #
        # Do not apply `limit * tanh(v / limit)` here. That is not a hard
        # speed limit: at 89% of no-load speed it removes about 20% of the
        # command, which the controller cannot recover and which shows up as
        # a false tracking lag (KI-019). The optimizer already bounds planned
        # wheel speeds; validate_trajectory() checks the planned samples too.
        v_act = (vl_act + vr_act) / 2.0
        omega_act = (vr_act - vl_act) / track_width

        dx = v_act * jnp.cos(x[2])
        dy = v_act * jnp.sin(x[2])
        dtheta = omega_act

        return jnp.array([dx, dy, dtheta])


class DigitalTwinOpenLoop(immrax.system.OpenLoopSystem):
    """Actuator-aware plant used only when ``--sysid-fit`` is enabled.

    State: [x, y, heading, vl_actual, vr_actual]. The two wheel-speed states
    follow the held controller command through a first-order lag. The command
    delay is applied by the sampled-data loop in ``ImmraxValidator`` because
    it needs command history, which a memoryless ``f(t, x, u, w)`` cannot see.

    Two measured behaviours are modelled, both inert unless the fit provides
    them:

    * the wheel speed cannot change faster than the measured limit,
    * while braking, the wheels turn further than the robot travels, so the
      truth covers less ground than the encoders report.
    """

    # Explicit Euler is stable while dt / tau <= 1. The sampled loop uses a
    # 5 ms plant step, so a fitted tau below 10 ms is treated as this floor.
    MIN_TAU_S = 0.01

    # Below this commanded wheel speed the robot is braking, so the measured
    # braking shortfall applies. The transition is smooth because Immrax's
    # natural inclusion cannot compare or select.
    BRAKE_SPEED_M_S = 0.05

    def __init__(self, config: JAXRobotConfig,
                 digital_twin: DigitalTwinConfig):
        self.evolution = 'continuous'
        self.xlen = 5
        self.config = config
        self.digital_twin = digital_twin
        self.tau_s = max(digital_twin.actuator_tau_s(), self.MIN_TAU_S)
        self.brake_shortfall = digital_twin.brake_shortfall()
        self.accel_limit = digital_twin.actuator_accel_limit()
        # A robot whose wheels are not parallel curves even when it is told
        # to drive straight. The plan's straight segments are straight, so
        # the curvature is a tracking error that grows with the distance.
        self.drift_rad_per_m = digital_twin.drift_rad_per_m()

    def f(self, t, x, u, w):
        gain = _wheel_gain(w)
        vl_cmd, vr_cmd = u[0] * gain, u[1] * gain
        track_width = w[3]

        # The wheel turns at the commanded speed; the gear backlash is a
        # position dead-band and is carried in the reachable set, not as a
        # velocity offset. See RobotOpenLoop.f for the reasoning.
        vl_target = vl_cmd
        vr_target = vr_cmd

        vl_actual, vr_actual = x[3], x[4]
        v_actual = (vl_actual + vr_actual) / 2.0
        omega_actual = (vr_actual - vl_actual) / track_width

        # Braking slip: the wheels turn without gripping, so the robot covers
        # less than its encoders report. The factor goes to one at speed, so
        # only the braking part of the run is affected.
        if self.brake_shortfall:
            commanded = (vl_target + vr_target) / 2.0
            braking = 1.0 - jnp.tanh(
                (commanded * commanded) / (self.BRAKE_SPEED_M_S ** 2))
            grip = 1.0 - self.brake_shortfall * braking
        else:
            grip = 1.0

        dx = grip * v_actual * jnp.cos(x[2])
        dy = grip * v_actual * jnp.sin(x[2])
        dtheta = grip * omega_actual
        if self.drift_rad_per_m:
            # The wheels are not quite parallel, so the heading creeps at a
            # rate proportional to the speed. A positive drift turns the
            # robot to the left, the same way as a positive heading.
            dtheta = dtheta + self.drift_rad_per_m * grip * v_actual
        dvl = (vl_target - vl_actual) / self.tau_s
        dvr = (vr_target - vr_actual) / self.tau_s

        if self.accel_limit is not None:
            # A soft limit: tanh is monotone, so the natural inclusion of it
            # is exact, and it keeps the rate below the measured limit.
            dvl = self.accel_limit * jnp.tanh(dvl / self.accel_limit)
            dvr = self.accel_limit * jnp.tanh(dvr / self.accel_limit)

        return jnp.array([dx, dy, dtheta, dvl, dvr])

#irx.natemb

# ==============================================================================
# 2. THE CONTROLLER (Control Strategy)
# ==============================================================================
class RamseteController:
    def __init__(self, ramsete_b=2.0, ramsete_zeta=0.7,
                 track_width: Optional[float] = None):
        self.controller = JAXRamseteController(ramsete_b, ramsete_zeta)
        # The controller converts v/omega to wheel speeds using the robot's
        # configured axle track. Physical track-width uncertainty belongs in
        # the plant; the deployed DriveBase does not change its axle track.
        self.track_width = track_width

    def __call__(self, t, x, u_ref, w, normalize: bool = False,
                 quantize: bool = False):
        """
        u_ref: Reference trajectory [xr, yr, thetar, vr, omegar, kr]

        ``normalize=False`` is for interval states with unrolled headings.
        Float states used by the nominal feedback path should pass True so
        they match ramsete_fast.py's shortest-angle normalization.

        ``quantize=True`` mirrors the firmware, which rounds the commanded
        linear speed (mm/s) and turn rate (deg/s) before calling
        ``DriveBase.drive()``. It stays off for the interval inclusion path,
        which may only use operations Immrax can include.
        """
        xr, yr, thetar, vr, omegar = u_ref[0], u_ref[1], u_ref[2], u_ref[3], u_ref[4]
        ref_k = u_ref[5] if len(u_ref) > 5 else None
        track_width = self.track_width if self.track_width is not None else w[3]

        ref_pose = jnp.array([xr, yr, thetar])
        v_cmd, omega_cmd = self.controller.calculate(
            x, ref_pose, vr, omegar, ref_k=ref_k, normalize=normalize)

        if quantize:
            v_cmd, omega_cmd = quantize_drive_command(v_cmd, omega_cmd)

        vl_cmd = v_cmd - (omega_cmd * track_width) / 2.0
        vr_cmd = v_cmd + (omega_cmd * track_width) / 2.0

        return jnp.array([vl_cmd, vr_cmd])

# ==============================================================================
# 3. THE CLOSED LOOP SYSTEM
# ==============================================================================
class ClosedLoopRobotSystem(immrax.System):
    def __init__(self, config: JAXRobotConfig, ramsete_b: float = 2.0,
                 ramsete_zeta: float = 0.7):
        self.evolution = 'continuous'
        self.xlen = 3                 
        self.plant = RobotOpenLoop(config)
        self.controller = RamseteController(
            ramsete_b, ramsete_zeta, track_width=config.track_width)

    def f(self, t, x, u, w):
        # 1. Controller computes commands
        u_cmd = self.controller(t, x, u, w)
        # 2. Plant applies commands to physical dynamics
        return self.plant.f(t, x, u_cmd, w)

# use immrax.ControlledSystem()




class ImmraxValidator:
    def __init__(self, config: JAXRobotConfig,
                 ramsete_b: float = 2.0, ramsete_zeta: float = 0.7,
                 control_dt: float = CONTROL_DT,
                 feedback_mode: str = "nominal",
                 digital_twin: Optional[DigitalTwinConfig] = None):
        self.cfg = config
        self.model = JAXDifferentialDriveModel(config)
        self.ramsete_b = float(ramsete_b)
        self.ramsete_zeta = float(ramsete_zeta)
        self.control_dt = float(control_dt)
        self.digital_twin = digital_twin
        self._digital_twin_active = bool(
            digital_twin is not None and digital_twin.enabled)
        # "nominal" feeds the interval state midpoint to the controller. This
        # keeps the digital twin's feedback realistic: the deployed robot
        # reacts to its best estimate, not to the worst-case state in a tube.
        # "interval" evaluates the controller inclusion on the whole interval;
        # it is more conservative and currently diverges on long fast paths
        # because naive interval Ramsete does not see the centroid error.
        if feedback_mode not in ("nominal", "interval"):
            raise ValueError("feedback_mode must be 'nominal' or 'interval'")
        self.feedback_mode = feedback_mode

        self.digital_twin_gain_range = (
            digital_twin.actuator_gain_range()
            if self._digital_twin_active else (1.0, 1.0)
        )
        self.digital_twin_delay_s = (
            digital_twin.actuator_delay_s()
            if self._digital_twin_active else 0.0
        )
        # The robot's own pose estimate. The deployed controller never sees
        # the true pose: it integrates the wheel encoders and the gyro, so the
        # belief drifts away from the truth by the measured scale error and
        # the gyro bias. The controller is fed that belief, not the truth.
        self.digital_twin_distance_scale = (
            digital_twin.odometry_distance_scale()
            if self._digital_twin_active else 1.0
        )
        self.digital_twin_heading_scale = (
            digital_twin.odometry_heading_scale()
            if self._digital_twin_active else 1.0
        )
        self.digital_twin_gyro_bias_rad_s = (
            digital_twin.gyro_z_bias_rad_s()
            if self._digital_twin_active else 0.0
        )
        # The motor's stall torque, measured by pushing into a wall. The
        # config can only declare it, so without the measurement the force
        # limit is a guess. The measurement itself is carried by the robot
        # config: main.py --sysid-fit writes it there with
        # apply_measured_motor, so the optimizer, the CPU audit, the
        # validator and the exported controller all plan against the same
        # motor (KI-010). Reading it here as well would check the plan
        # against a different limit than the one it was planned with.

        self.sys = ClosedLoopRobotSystem(
            config, self.ramsete_b, self.ramsete_zeta)
        self.emb_sys = immrax.natemb(self.sys)

        # Sampled-data path: hold the controller command for one 20 ms tick,
        # then integrate the physical plant. This mirrors ramsete_fast.py
        # instead of evaluating the controller continuously.
        if self._digital_twin_active and digital_twin.actuator_tau_s() > 0.0:
            self.plant = DigitalTwinOpenLoop(config, digital_twin)
        else:
            self.plant = RobotOpenLoop(config)
        self.plant_emb_sys = immrax.natemb(self.plant)
        self.controller = RamseteController(
            self.ramsete_b, self.ramsete_zeta, track_width=config.track_width)
        self._controller_if = natif(self._controller_command_interval)
        self._sampled_control = True

    def _controller_command_interval(self, x, u_ref, w):
        """Interval inclusion of the controller for one control tick."""
        return self.controller(0.0, x, u_ref, w)

    def _advance_belief(self, belief, x_flat, state_len, dt):
        """Integrate the robot's own pose estimate from the wheel motion.

        The deployed robot integrates the wheel encoders into a distance
        reading and the gyro into a heading. This mirrors that: the belief
        advances with the achieved wheel speeds, scaled by the measured
        odometry error, and its heading drifts with the measured gyro bias.

        It is kept out of the plant state on purpose. The belief is the
        robot's estimate, not a physical quantity, so it does not belong in
        the reachable tube; it only decides what the controller believes.
        """
        vl = float((x_flat[3] + x_flat[state_len + 3]) / 2.0)
        vr = float((x_flat[4] + x_flat[state_len + 4]) / 2.0)
        track_width = self.cfg.track_width

        odo_v = (vl + vr) / 2.0 * self.digital_twin_distance_scale
        odo_w = ((vr - vl) / track_width) * self.digital_twin_heading_scale
        odo_w += self.digital_twin_gyro_bias_rad_s

        x, y, heading = belief
        heading = heading + odo_w * dt
        x = x + odo_v * math.cos(heading) * dt
        y = y + odo_v * math.sin(heading) * dt
        return (x, y, heading)

    def _odometry_pose(self, pose, start_pose):
        """Apply the fitted static odometry scale to a feedback pose.

        This is an endpoint-calibrated correction, not a full encoder model.
        Without a calibrated odometry block it is the identity.
        """
        twin = getattr(self, "digital_twin", None)
        if twin is None or not twin.enabled or not twin.odometry.calibrated:
            return pose

        distance_scale = float(twin.odometry.distance_scale)
        heading_scale = float(twin.odometry.heading_scale)
        start = jnp.asarray(start_pose)

        if isinstance(pose, Interval):
            lower, upper = pose.lower, pose.upper
            lower_out = jnp.array([
                start[0] + (lower[0] - start[0]) * distance_scale,
                start[1] + (lower[1] - start[1]) * distance_scale,
                start[2] + (lower[2] - start[2]) * heading_scale,
            ])
            upper_out = jnp.array([
                start[0] + (upper[0] - start[0]) * distance_scale,
                start[1] + (upper[1] - start[1]) * distance_scale,
                start[2] + (upper[2] - start[2]) * heading_scale,
            ])
            return Interval(lower_out, upper_out)

        return jnp.array([
            start[0] + (pose[0] - start[0]) * distance_scale,
            start[1] + (pose[1] - start[1]) * distance_scale,
            start[2] + (pose[2] - start[2]) * heading_scale,
        ])

    @staticmethod
    def _delayed_command(history, target_t):
        """Return the held command active at ``target_t``, or zero before start."""
        selected = None
        for command_t, command in history:
            if command_t <= target_t + 1e-12:
                selected = command
            else:
                break
        return selected if selected is not None else jnp.zeros(2)

    def validate_trajectory(self, samples: List[Dict[str, Any]],
                            cof_range: Tuple[float, float],
                            torque_margin_range: Tuple[float, float],
                            backlash_range: Tuple[float, float],
                            track_width_range: Optional[Tuple[float, float]] = None,
                            wheel_radius_range: Optional[Tuple[float, float]] = None,
                            inertia_range: Optional[Tuple[float, float]] = None,
                            safety_threshold_m: float = 0.005) -> Dict[str, Any]:
        vl = jnp.array([s['vl'] for s in samples])
        vr = jnp.array([s['vr'] for s in samples])
        al = jnp.array([s['al'] for s in samples])
        ar = jnp.array([s['ar'] for s in samples])

        cof_interval = Interval(*cof_range)
        torque_margin_interval = Interval(*torque_margin_range)

        # The motor's stall torque comes from the robot config, which
        # main.py --sysid-fit fills with the measured number. The config is
        # the one place the motor lives, so the optimizer, this audit, the
        # validator and the exported controller all use the same limit.
        t_max_interval = interval(self.cfg.t_max_nm)
        
        tw_range = track_width_range if track_width_range else (self.cfg.track_width * 0.99, self.cfg.track_width * 1.01)
        wr_range = wheel_radius_range if wheel_radius_range else (self.cfg.wheel_radius * 0.99, self.cfg.wheel_radius * 1.01)
        ine_range = inertia_range if inertia_range else (self.cfg.inertia * 0.90, self.cfg.inertia * 1.10)

        track_width_iv = Interval(*tw_range)
        wheel_radius_iv = Interval(*wr_range)
        inertia_iv = Interval(*ine_range)

        def check_sample_interval(vl_val, vr_val, al_val, ar_val):
            # 1. Interval Dynamics
            a_val = (al_val + ar_val) / 2.0
            alpha_val = interval(ar_val - al_val) / track_width_iv
            f_total = self.cfg.mass * a_val
            m_total = inertia_iv * alpha_val

            fr = (interval(f_total) + (m_total * interval(2.0) / track_width_iv)) / interval(2.0)
            fl = interval(f_total) - fr

            # 2. Interval-Aware Motor Torque Curves
            omega_l = (interval(jnp.abs(vl_val)) / wheel_radius_iv) * interval(self.cfg.gearing)
            torque_l = (interval(1.0) - omega_l / interval(self.cfg.v_max_rad_s)) * t_max_interval
            torque_l = Interval(jnp.maximum(0.0, torque_l.lower), jnp.maximum(0.0, torque_l.upper))
            force_limit_l = (torque_l / wheel_radius_iv) * interval(self.cfg.gearing) * torque_margin_interval

            omega_r = (interval(jnp.abs(vr_val)) / wheel_radius_iv) * interval(self.cfg.gearing)
            torque_r = (interval(1.0) - omega_r / interval(self.cfg.v_max_rad_s)) * t_max_interval
            torque_r = Interval(jnp.maximum(0.0, torque_r.lower), jnp.maximum(0.0, torque_r.upper))
            force_limit_r = (torque_r / wheel_radius_iv) * interval(self.cfg.gearing) * torque_margin_interval

            def iv_abs_max(iv):
                return jnp.maximum(jnp.abs(iv.lower), jnp.abs(iv.upper))

            motor_violation_l = jnp.maximum(0.0, iv_abs_max(fl) - force_limit_l.lower)
            motor_violation_r = jnp.maximum(0.0, iv_abs_max(fr) - force_limit_r.lower)

            # 3. Dynamic Normal Force Transfers. Only lateral (turning)
            # transfer applies; a two-wheel robot has no front/rear split
            # (KI-010). Same formula as the optimizer and the CPU audit.
            v_val = (vl_val + vr_val) / 2.0
            omega_val = interval(vr_val - vl_val) / track_width_iv
            centripetal_accel = interval(v_val) * omega_val
            nl, nr = wheel_normal_forces(
                self.cfg.mass, self.cfg.g, CG_HEIGHT_M, track_width_iv, centripetal_accel
            )

            # 4. Global Friction Circle & Local Wheel Slip Violations
            traction_max = cof_interval.lower * self.cfg.mass * self.cfg.g
            traction_violation = jnp.maximum(0.0, iv_abs_max(fl) + iv_abs_max(fr) - traction_max)

            slip_l = jnp.maximum(0.0, iv_abs_max(fl) - (cof_interval * nl).lower)
            slip_r = jnp.maximum(0.0, iv_abs_max(fr) - (cof_interval * nr).lower)

            return motor_violation_l, motor_violation_r, traction_violation, slip_l, slip_r

        motor_vios_l, motor_vios_r, traction_vios, slip_ls, slip_rs = vmap(check_sample_interval)(vl, vr, al, ar)

        unrolled_samples = unroll_trajectory_headings(samples)
        reach_report = self.compute_reachability(
            unrolled_samples, backlash_range, tw_range, wr_range, cof_range, torque_margin_range, ine_range
        )

        max_motor_vio = jnp.maximum(jnp.max(motor_vios_l), jnp.max(motor_vios_r))

        # The plant model now integrates planned wheel speeds directly (no
        # soft tanh saturation), so check the planned samples against the
        # wheel no-load speed here.
        max_wheel_speed = max(float(jnp.max(jnp.abs(vl))),
                              float(jnp.max(jnp.abs(vr))))
        wheel_speed_limit = self.cfg.v_max_rad_s * min(self.cfg.wheel_radius_range)
        speed_violation = max(0.0, max_wheel_speed - wheel_speed_limit)

        # Tracking and envelope size are checked separately. The caller applies
        # its context-aware threshold to the envelope; the validator only
        # requires that the tube did not diverge.
        is_physically_safe = (
            max_motor_vio < 1e-3 and
            jnp.max(slip_ls) < 1e-3 and
            jnp.max(slip_rs) < 1e-3 and
            reach_report['max_tracking_error_m'] <= safety_threshold_m and
            speed_violation <= 1e-6 and
            not reach_report['diverged']
        )

        return {
            "max_motor_violation_N": float(max_motor_vio),
            "max_traction_violation_N": float(jnp.max(traction_vios)),
            "max_slip_l_N": float(jnp.max(slip_ls)),
            "max_slip_r_N": float(jnp.max(slip_rs)),
            "max_tracking_error_m": reach_report['max_tracking_error_m'],
            "max_heading_error_rad": reach_report['max_heading_error_rad'],
            "max_envelope_radius_m": reach_report['max_envelope_radius_m'],
            "max_wheel_speed_m_s": max_wheel_speed,
            "wheel_speed_limit_m_s": wheel_speed_limit,
            "speed_violation_m_s": speed_violation,
            "reachability": reach_report,
            "digital_twin": (
                self.digital_twin.summary()
                if self.digital_twin is not None else None
            ),
            "passed": bool(is_physically_safe)
        }

    def compute_reachability(self, samples: List[Dict[str, Any]], 
                             backlash_range: Tuple[float, float],
                             track_width_range: Tuple[float, float],
                             wheel_radius_range: Tuple[float, float],
                             cof_range: Tuple[float, float],
                             torque_margin_range: Tuple[float, float],
                             inertia_range: Tuple[float, float]) -> Dict[str, Any]:
        N = len(samples)
        if N < 2:
            return {
                "max_error_m": 0.0,
                "max_tracking_error_m": 0.0,
                "max_heading_error_rad": 0.0,
                "max_envelope_radius_m": 0.0,
                "envelope": [],
                "diverged": False,
            }

        # Pass the uncertainty as an Interval, not a flat lower/upper array.
        # Immrax's natural inclusion treats non-Interval arguments as constants,
        # so a flat array silently used only the lower bounds (KI-003).
        gain_range = getattr(self, "digital_twin_gain_range", (1.0, 1.0))
        w_lower = jnp.array([cof_range[0], torque_margin_range[0], backlash_range[0], track_width_range[0], wheel_radius_range[0], inertia_range[0], gain_range[0]])
        w_upper = jnp.array([cof_range[1], torque_margin_range[1], backlash_range[1], track_width_range[1], wheel_radius_range[1], inertia_range[1], gain_range[1]])
        w_interval = Interval(w_lower, w_upper)

        active_twin = getattr(self, "_digital_twin_active", False)
        state_len = self.plant.xlen if active_twin else 3
        delay_s = getattr(self, "digital_twin_delay_s", 0.0) if active_twin else 0.0

        # Create FLAT array for state 'x' (lower bounds concatenated with
        # upper bounds). The digital-twin plant adds actual wheel speeds after
        # the pose; they start at rest with no uncertainty.
        start_pose = jnp.array([samples[0]['x'], samples[0]['y'], samples[0]['heading']])
        lower_state = jnp.concatenate([
            start_pose - 0.001,
            jnp.zeros(state_len - 3),
        ])
        upper_state = jnp.concatenate([
            start_pose + 0.001,
            jnp.zeros(state_len - 3),
        ])
        x_flat = jnp.concatenate([lower_state, upper_state])

        def _box(flat):
            # Slicing: the first state_len entries are lower bounds, the next
            # state_len entries are upper bounds.
            return {
                "x_min": float(flat[0]), "x_max": float(flat[state_len]),
                "y_min": float(flat[1]), "y_max": float(flat[state_len + 1]),
            }

        ramsete_b = getattr(self, "ramsete_b", 2.0)
        envelope = [_box(x_flat)]  # box at sample 0, so envelope[i] is the box at sample i
        max_tracking_error = 0.0
        max_heading_error = 0.0
        max_envelope_radius = 0.0
        max_total_error = 0.0
        diverged = False
        command_history = []
        # Half the backlash, used as the position uncertainty the reachable
        # set carries in place of the old velocity offset.
        backlash_half = 0.5 * float(w_upper[2])

        # The robot's own pose estimate. It starts where the plan starts and
        # then drifts away from the truth by the measured odometry error and
        # the gyro bias, exactly as the deployed robot's estimate does. It
        # needs the wheel-speed states, so it only exists with the
        # actuator-aware plant.
        belief = None
        if active_twin and self.plant.xlen > 3:
            belief = (float(start_pose[0]), float(start_pose[1]),
                      float(start_pose[2]))

        for i in range(N - 1):
            s = samples[i]
            s_next = samples[i+1]
            dt = s_next['t'] - s['t']

            if not math.isfinite(dt) or dt <= 0:
                # A failed solve can produce a non-monotone time vector. Do not
                # integrate backwards; report the tube as unknown so the caller
                # cannot export it.
                diverged = True
                max_tracking_error = float("inf")
                max_envelope_radius = float("inf")
                max_total_error = float("inf")
                envelope.append({**envelope[-1], "unknown": True})
                continue

            if not diverged and getattr(self, "_sampled_control", False):
                # Match ramsete_fast.py: sample the linear reference and read
                # the state at the start of each 20 ms tick, compute the
                # Ramsete command once, and hold it until the next tick.
                t = s['t']
                t_end = s_next['t']
                while t < t_end - 1e-12:
                    dt_ctrl = min(self.control_dt, t_end - t)
                    if dt_ctrl <= 0:
                        diverged = True
                        break
                    frac = (t - s['t']) / dt
                    ref = _interpolated_reference(
                        s, s_next, dt, self.cfg.track_width, frac,
                        ramsete_b)
                    u_ref = jnp.array(ref)
                    if self.feedback_mode == "interval":
                        x_interval = Interval(
                            x_flat[:3],
                            x_flat[state_len:state_len + 3])
                        x_interval = self._odometry_pose(x_interval, start_pose)
                        u_cmd = self._controller_if(
                            x_interval, u_ref, w_interval)
                    else:
                        # The real controller sees one measured state, not
                        # the whole tube. Use the tube midpoint for feedback,
                        # then propagate the interval through the plant.
                        if belief is not None:
                            # With the digital twin on, the controller sees
                            # the robot's own estimate, not the truth.
                            x_center = jnp.array(belief)
                        else:
                            x_center = (
                                x_flat[:3] + x_flat[state_len:state_len + 3]
                            ) / 2.0
                            x_center = self._odometry_pose(x_center, start_pose)
                        u_cmd = self.controller(
                            0.0, x_center, u_ref, w_interval, normalize=True,
                            quantize=True)

                    if active_twin and delay_s > 0.0:
                        command_history.append((t, u_cmd))
                        u_applied = self._delayed_command(
                            command_history, t - delay_s)
                    else:
                        u_applied = u_cmd

                    for _ in range(PLANT_SUBSTEPS):
                        if belief is not None:
                            # Advance the belief with the same speeds and the
                            # same order as the plant, so a perfect odometry
                            # keeps the belief on the truth.
                            belief = self._advance_belief(
                                belief, x_flat, state_len,
                                dt_ctrl / PLANT_SUBSTEPS)
                        dx_dt_flat = self.plant_emb_sys.f(
                            0.0, x_flat, u_applied, w_interval)
                        x_flat = x_flat + dx_dt_flat * (
                            dt_ctrl / PLANT_SUBSTEPS)
                    t += dt_ctrl

            elif not diverged:
                # Legacy path used by the fake dynamics in regression tests.
                # Drive the whole interval with a reference that starts at
                # sample i and ends at sample i+1.
                for k in range(REACHABILITY_SUBSTEPS):
                    frac = (k + 0.5) / REACHABILITY_SUBSTEPS
                    ref = _interpolated_reference(
                        s, s_next, dt, self.cfg.track_width, frac,
                        ramsete_b)
                    u_k = jnp.array(ref)

                    # Evaluate the compiled embedding system using the
                    # interval state and interval uncertainty.
                    dx_dt_flat = self.emb_sys.f(0.0, x_flat, u_k, w_interval)
                    x_flat = x_flat + dx_dt_flat * (dt / REACHABILITY_SUBSTEPS)

            if not diverged:
                env = _box(x_flat)
                # The gear backlash is a position dead-band. It used to be
                # subtracted from the commanded wheel speed, which put a fake
                # systematic lag of backlash/2 * t into the tracking error.
                # It is a position uncertainty, so it belongs in the reachable
                # set: the same magnitude the plant used to bias the
                # tracking error by, carried symmetrically. The worst-case
                # total uncertainty at each sample is therefore unchanged.
                if backlash_half > 0.0:
                    elapsed = s_next['t'] - samples[0]['t']
                    # Never let the spread go negative: that would shrink the
                    # safety envelope instead of widening it.
                    spread = backlash_half * max(0.0, elapsed)
                    env = {
                        "x_min": env["x_min"] - spread,
                        "x_max": env["x_max"] + spread,
                        "y_min": env["y_min"] - spread,
                        "y_max": env["y_max"] + spread,
                    }
                mid_x = (x_flat[0] + x_flat[state_len]) / 2.0
                mid_y = (x_flat[1] + x_flat[state_len + 1]) / 2.0
                tracking_error = float(jnp.sqrt(
                    (mid_x - s_next['x'])**2 + (mid_y - s_next['y'])**2))
                # The heading error is reported next to the position error.
                # The tracking metric is position-only, so a trajectory can
                # follow the path while arriving at the wrong heading - and
                # the plan's own heading samples can be inconsistent with its
                # positions by up to half the heading change per sample.
                mid_heading = (x_flat[2] + x_flat[state_len + 2]) / 2.0
                heading_error = float(abs(
                    (mid_heading - s_next['heading'] + math.pi)
                    % (2 * math.pi) - math.pi))
                radius = float(jnp.sqrt(
                    ((env["x_max"] - env["x_min"]) / 2.0)**2
                    + ((env["y_max"] - env["y_min"]) / 2.0)**2))

                max_tracking_error = max(max_tracking_error, tracking_error)
                max_heading_error = max(max_heading_error, heading_error)
                max_envelope_radius = max(max_envelope_radius, radius)
                max_total_error = max(max_total_error, tracking_error + radius)

                # If the interval tube exceeds 5cm the propagation has diverged (non-monotone
                # closed loop). Do not shrink the box to a smaller one (that would report a
                # safety margin we cannot prove): report the true bound, mark the sample and
                # everything after it as unknown so consumers fail it.
                if radius > 0.05:
                    diverged = True
                envelope.append({**env, "unknown": True} if diverged else env)
            else:
                envelope.append({**envelope[-1], "unknown": True})

        return {
            # Kept for callers that used the old combined bound. New callers
            # should use the tracking and envelope fields separately.
            "max_error_m": max_total_error,
            "max_tracking_error_m": max_tracking_error,
            "max_heading_error_rad": max_heading_error,
            "max_envelope_radius_m": max_envelope_radius,
            "envelope": envelope,
            "diverged": diverged,
        }

    def get_safety_ribbon_data(self, samples: List[Dict[str, Any]],
                               cof_range: Tuple[float, float],
                               torque_margin_range: Tuple[float, float],
                               backlash_range: Tuple[float, float],
                               track_width_range: Optional[Tuple[float, float]] = None,
                               wheel_radius_range: Optional[Tuple[float, float]] = None,
                               inertia_range: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        unrolled = unroll_trajectory_headings(samples)
        
        tw_range = track_width_range if track_width_range else (self.cfg.track_width * 0.99, self.cfg.track_width * 1.01)
        wr_range = wheel_radius_range if wheel_radius_range else (self.cfg.wheel_radius * 0.99, self.cfg.wheel_radius * 1.01)
        ine_range = inertia_range if inertia_range else (self.cfg.inertia * 0.90, self.cfg.inertia * 1.10)

        report = self.compute_reachability(
            unrolled, backlash_range, tw_range, wr_range, cof_range, torque_margin_range, ine_range
        )
        
        ribbon_data = []
        envelope = report["envelope"]
        
        for i, s in enumerate(samples):
            env_point = envelope[i] if i < len(envelope) else envelope[-1]
            
            ribbon_data.append({
                "t": float(s["t"]),
                "x": float(s["x"]),
                "y": float(s["y"]),
                "heading": float(s["heading"]),
                "x_min": float(env_point["x_min"]),
                "x_max": float(env_point["x_max"]),
                "y_min": float(env_point["y_min"]),
                "y_max": float(env_point["y_max"])
            })
            
        return ribbon_data
