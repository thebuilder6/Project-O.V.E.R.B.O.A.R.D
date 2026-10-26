from overboard.verification.reachability_engine import (
    CompiledReachabilityEngine,
    JAX_COMPILE_LOCK,
    unroll_trajectory_headings,
)
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
import immrax
from immrax import Interval, interval
from immrax.inclusion import natif, jacif, mjacif
from overboard.models.jax_robot_model import JAXRobotConfig, JAXDifferentialDriveModel
from overboard.models.drivetrain_physics import (
    CG_HEIGHT_M,
    diff_drive_fk,
    wheel_normal_forces,
    traction_limits,
    total_traction_limit,
)
from overboard.models.jax_ramsete import JAXRamseteController, quantize_drive_command
from overboard.sysid.digital_twin import DigitalTwinConfig
from typing import List, Dict, Any, Tuple, Optional
import math


# The control period of the controller that actually runs.
#
# The controllers in use hold the drive command for 20 ms:
#   pybricks_code/ramsete_compact.py            _LOOP_MS = 20   (the SysId replay)
#   pybricks_code/ramsete_compact_events.py     _LOOP_MS = 20   (the hub push)
#   pybricks_code/ramsete.py                    CONTROL_MS = 20 (the IDE missions)
# and the command is held until the next iteration. The validator used to
# model a 20 ms tick of ramsete_fast.py's 50 Hz heartbeat, which is the same
# period but a *rounded* command and a different gain; those are fixed below
# too.
#
# The period is the expensive part of the model: the wheel-force hold error
# is proportional to it, so a model that holds the command longer than the
# robot does reports a bigger tracking error, and the 5 mm tracking gate is
# decided on it (docs/research/RQ-A_tracking_error.md, findings F6 and F11).
#
# A synced heartbeat is what makes the modelled hold exact: the loop waits
# for the slack to the next tick, so the period is 20 ms and not 20 ms plus
# the loop's own compute time.
CONTROL_DT = 0.02
PLANT_SUBSTEPS = 4

# Which inclusion function embeds the plant and the controller.
#
# "nat" is the natural extension: the interval-arithmetic range of the
# expression, with the sharing inside one expression preserved but not the
# sharing between two *equal* quantities. The plant has exactly that
# dependency. It computes the turn rate as (vr - vl) / track_width, and both
# wheel speeds are the SAME command times the SAME gain, so the true turn
# rate does not depend on the gain at all. The natural extension cannot see
# that and reports a phantom turn rate of v * band / track, which integrates
# into a heading drift and then into a lateral drift: the reachable tube grows
# with the square of the time and crosses the validator's 50 mm guard long
# before the path ends. Measured on a 1 m straight with a +/-1 % gain band,
# that is the difference between a tube that diverges and a tube that is
# 11.8 mm wide (docs/research/RQ-A_tracking_error.md, finding F5 and the
# envelope finding that follows it).
#
# "jac" is the Jacobian (mean-value) inclusion, f(c) + J(X)(X - c) with the
# interval Jacobian: the partial derivatives are bounded over the whole box,
# so the result is a valid enclosure of f(X). It computes
# d(turn rate)/d(gain) = (vr_cmd - vl_cmd) / track, which is exactly zero
# for a straight, and the phantom disappears.
#
# "mjac" is the mixed-Jacobian inclusion: it relaxes one input coordinate at
# a time from the centre to its bound and hulls the results. On this plant it
# is 0.01-0.06 mm tighter than "jac" and about 1.3x the cost, so it is not
# worth the time.
#
# The default is "nat", and that is a measurement, not a preference. The
# dependency that used to blow the tube up is removed at the source, by
# taking the wheel-speed difference from the commands before the gain
# multiplies it (see RobotOpenLoop.f and DigitalTwinOpenLoop), so the
# inclusion function has nothing left to win: on the six example missions
# "jac" is 0.00-0.08 mm WIDER than "nat" and costs 5-15 % more time. The
# switch is kept because the Jacobian is the right tool for a dependency the
# algebra cannot remove, and because it is the only way to see the
# difference for yourself.
#
# Both "jac" and "mjac" were checked against the reachable set the robot
# can actually reach, sampled over the declared uncertainty with the corners
# included (docs/research/rq_a/exp19_envelope_scatter.py): the tighter box
# still contains it, so the tighter box is sound and not merely smaller.
PLANT_INCLUSION = "nat"
CONTROLLER_INCLUSION = "nat"

# The inclusion functions that may be named above.
PLANT_INCLUSIONS = ("nat", "jac", "mjac")
CONTROLLER_INCLUSIONS = ("nat", "jac", "mjac")

# Which propagation the reachability uses.
#
# "plant" is what has always shipped: the plant is embedded and propagated,
# and the controller is evaluated *outside* the propagation, at the tube's
# midpoint, with the command held for the tick.  The interval arithmetic
# therefore never sees the feedback, and the tube it produces is the tube of
# an open-loop plant driven by a pre-recorded command.  The real robot is a
# contracting closed loop.
#
# "tick" embeds one control tick of the *closed loop* as a discrete map, with
# the controller inside it, and propagates the map once per tick.  The map's
# Jacobian carries the feedback term dF/dx = dPhi/dx + (dPhi/du)(dN/dx), and
# the Ramsete feedback makes that term contract the tube - which is what the
# robot does.  Measured on the shipped s-curve trajectory with a +/-1 % gain
# band (docs/research/rq_a/exp20_closed_loop_tick.py): the plant propagation
# crosses the 50 mm guard (53.6 mm) and fails the envelope gate, the tick
# propagation with the mixed-Jacobian inclusion gives 18.8 mm and passes it,
# and both contain the sampled reachable set.
#
# The tick propagation needs a Jacobian inclusion: with the natural extension
# the closed-loop map explodes (627 mm on a 1 m straight), because that is
# the naive interval Ramsete the "interval" feedback mode already avoids.
PROPAGATION = "plant"
PROPAGATIONS = ("plant", "tick")

# Which inclusion function embeds the closed-loop tick map.
#
# The natural extension cannot be used here at all (it explodes), so the
# choice is between the Jacobian and the mixed-Jacobian. The mixed one is
# tighter - on the shipped s-curve trajectory with a +/-1 % gain band it gives
# 18.8 mm against the Jacobian's 25.3 mm, and the 20 mm gate is decided in
# between - and it costs 7-10x the validation time against the Jacobian's
# 2-3x. The gate is a safety gate, so the tighter one is the one to ship
# and the cost is paid only by the runs that validate.
#
# On the default plant this is a clear win: with the mixed-Jacobian the
# s-curve, the stops and the complex mission stop diverging and pass the
# 20 mm gate, and every tube contains the sampled reachable set
# (docs/research/rq_a/exp22_twin_propagation.py). With the digital twin it
# is *worse*, because the map has to carry the robot's own pose estimate and
# an estimate in an interval state grows with the wheel-speed uncertainty:
# the belief's spread is added to the plant's, because the two are
# perfectly correlated and an interval box cannot say so. That is why the
# default propagation is still "plant".
TICK_INCLUSION = "mjac"
TICK_INCLUSIONS = ("nat", "jac", "mjac")

# How the robot's own pose estimate is carried in the tick map's state.
#
# "estimate" - the state holds the estimate itself.  The estimate and the
# truth are perfectly correlated (the odometry reads the very wheel speeds
# the plant carries), and an interval box cannot say so, so the estimate's
# spread is ADDED to the truth's.
#
# "error" - the state holds the estimate's ERROR, and the estimate is the
# truth plus that error.  The shared part is then carried once, and the
# box adds only the error's own spread.
#
# The default is "estimate", and that is a measurement, not a preference.
# The "error" form is the obvious way to stop the estimate's spread being
# added to the truth's, and it was measured: it is *worse*, not better
# (docs/research/rq_a/exp22_twin_propagation.py).  The error's update
# differences two cosines of a heading interval that is itself wide, and
# the difference of two over-approximations is worse than either - the
# s-curve goes from 5.1 cm with the plant propagation to 42.7 cm with the
# error form and the Jacobian, and to 9.5 cm with the mixed-Jacobian.  So
# the estimate stays a state of its own, and the estimate's correlation
# with the wheel speeds stays unsolved.
BELIEF_MODES = ("estimate", "error")
BELIEF_MODE = "estimate"


def _plant_embedding(plant, kind: str):
    """Embed the plant with the requested inclusion function."""
    if kind == "nat":
        return immrax.natemb(plant)
    if kind == "jac":
        return immrax.jacemb(plant)
    if kind == "mjac":
        return immrax.mjacemb(plant)
    raise ValueError(
        f"plant inclusion must be one of {PLANT_INCLUSIONS}, got {kind!r}")


def _controller_inclusion(fn, kind: str):
    """Inclusion function for the controller's command, per control tick."""
    if kind == "nat":
        return natif(fn)
    if kind == "jac":
        return jacif(fn)
    if kind == "mjac":
        return mjacif(fn)
    raise ValueError(
        f"controller inclusion must be one of {CONTROLLER_INCLUSIONS}, "
        f"got {kind!r}")


def _interpolated_reference(s0: Dict[str, Any], s1: Dict[str, Any],
                            dt: float, track_width: float,
                            s: float,
                            ramsete_b: float = 2.0,
                            dt_ctrl: float = 0.020,
                            is_final_stop: bool = False,
                            ) -> Tuple[float, float, float, float, float, float]:
    """
    Reference interpolation matching pybricks_code/ramsete.py.

    Linearly interpolates x, y, heading at current time t (fraction s).
    Computes feedforward velocity at tick midpoint t + dt_ctrl/2 (half-step lead)
    to cancel discrete Zero-Order Hold (ZOH) phase lag.
    """
    v0 = (s0["vl"] + s0["vr"]) / 2.0
    v1 = (s1["vl"] + s1["vr"]) / 2.0
    omega0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)

    # The shortest-angle difference, like the firmware's. The unrolled
    # headings differ by the same amount for any turn below 180 degrees.
    dh = (s1["heading"] - s0["heading"] + math.pi) % (2.0 * math.pi) - math.pi

    ref_x = s0["x"] + (s1["x"] - s0["x"]) * s
    ref_y = s0["y"] + (s1["y"] - s0["y"]) * s
    ref_heading = s0["heading"] + dh * s

    # Half-step lead fraction: t_mid = t + 0.5 * dt_ctrl
    frac_mid = s + (0.5 * dt_ctrl / dt) if dt > 0.0 else s
    if is_final_stop and frac_mid >= 1.0:
        ref_v = 0.0
        ref_omega = 0.0
    elif dt > 0.0:
        frac_mid = min(max(frac_mid, 0.0), 1.0)
        ref_v = v0 + frac_mid * (v1 - v0)
        ref_omega = dh / dt
    else:
        ref_v = 0.0
        ref_omega = omega0

    ref_kr = math.sqrt(ref_omega**2 + ramsete_b * ref_v**2)
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
    ramping behind it. Instant tracking is therefore a fair model of the hub
    until the plan asks for more than that limit; a first-order lag or a
    declared acceleration limit selects the five-state plant (Q11), which
    ramps. The DriveBase PID and the whole-millimeter distance reading are
    still not modeled (KI-020).
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
        #
        # The MEAN and the DIFFERENCE of the two wheel speeds are taken from
        # the commands and the gain is applied afterwards. It is the same
        # motion and a much better enclosure. The command difference is one
        # number, so one number times the gain is an exact interval, while
        # the difference of two gain-scaled commands is not: the natural
        # extension then reports a turn rate of +/- (the width of the wheel
        # speed interval) / track where the truth says zero, and that
        # phantom rate integrates into a lateral drift the reachable tube
        # cannot tell apart from the physics. On a straight the command
        # difference is exactly zero, so the turn rate is exactly zero.
        vl_cmd, vr_cmd = u[0], u[1]

        # Kinematics
        #
        # Do not apply `limit * tanh(v / limit)` here. That is not a hard
        # speed limit: at 89% of no-load speed it removes about 20% of the
        # command, which the controller cannot recover and which shows up as
        # a false tracking lag (KI-019). The optimizer already bounds planned
        # wheel speeds; validate_trajectory() checks the planned samples too.
        v_act = (vl_cmd + vr_cmd) / 2.0 * gain
        omega_act = (vr_cmd - vl_cmd) * gain / track_width

        dx = v_act * jnp.cos(x[2])
        dy = v_act * jnp.sin(x[2])
        dtheta = omega_act

        return jnp.array([dx, dy, dtheta])


class DigitalTwinOpenLoop(immrax.system.OpenLoopSystem):
    """Actuator-aware plant used when the digital twin supplies one.

    The twin may come from ``--sysid-fit`` or from the robot config's
    ``digital_twin`` section. The documented DriveBase acceleration limit
    needs no calibration, so it alone is enough to select this plant.

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

    # How the two wheel speeds are carried in the state.
    #
    # "wheels"   - state[3] is the left wheel speed, state[4] the right one.
    # "mean_diff" - state[3] is the MEAN wheel speed and state[4] is the
    #   DIFFERENCE (right minus left).
    #
    # The second one exists because the two wheel speeds are perfectly
    # correlated: on a straight they are equal, so their difference is
    # exactly zero, and the turn rate - which is that difference over the
    # track width - is exactly zero. An interval box cannot express that
    # correlation, because it treats its coordinates as independent, so the
    # natural extension reports a turn rate of +/- (the width of the wheel
    # speed interval) / track where the truth says zero. That phantom rate
    # integrates into a heading drift and then into a lateral drift, and it
    # is what blows the reachable tube past the validator's 50 mm guard.
    #
    # Carrying the mean and the difference makes the correlation explicit:
    # the difference's target is (vr_cmd - vl_cmd) * gain, and the command
    # difference is a point, so on a straight the difference's target is
    # exactly zero and the phantom never appears. The change of variables is
    # a linear bijection and the dynamics are transformed exactly, so the
    # enclosure is still a valid bound of the same reachable set.
    WHEEL_STATES = ("wheels", "mean_diff")
    WHEEL_STATE = "mean_diff"

    def __init__(self, config: JAXRobotConfig,
                 digital_twin: DigitalTwinConfig,
                 wheel_state: Optional[str] = None):
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
        self.wheel_state = wheel_state or self.WHEEL_STATE
        if self.wheel_state not in self.WHEEL_STATES:
            raise ValueError(
                f"wheel_state must be one of {self.WHEEL_STATES}, "
                f"got {self.wheel_state!r}")

    def f(self, t, x, u, w):
        gain = _wheel_gain(w)
        vl_cmd, vr_cmd = u[0] * gain, u[1] * gain
        track_width = w[3]

        # The wheel turns at the commanded speed; the gear backlash is a
        # position dead-band and is carried in the reachable set, not as a
        # velocity offset. See RobotOpenLoop.f for the reasoning.
        vl_target = vl_cmd
        vr_target = vr_cmd

        if self.wheel_state == "mean_diff":
            # The state's two numbers are the mean wheel speed and the
            # difference between the wheels, so the turn rate is read off
            # the difference directly and the correlation between the two
            # wheels is carried by the state instead of being lost.
            #
            # The difference of the TARGETS is taken at the command level,
            # before the gain multiplies it: (vr_cmd - vl_cmd) is a point,
            # and a point times the gain is an exact interval. Multiplying
            # first and subtracting afterwards would put the dependency
            # back, because the two products would each carry the whole
            # gain band.
            v_mean, v_diff = x[3], x[4]
            v_actual = v_mean
            omega_actual = v_diff / track_width

            v_mean_target = (u[0] + u[1]) / 2.0 * gain
            v_diff_target = (u[1] - u[0]) * gain

            d_mean_raw = (v_mean_target - v_mean) / self.tau_s
            d_diff_raw = (v_diff_target - v_diff) / self.tau_s
            if self.accel_limit is None:
                # The two rates are the state's own coordinates. They are
                # not recovered by differencing two wheel-level rates,
                # because that would put the correlation back.
                d_mean, d_diff = d_mean_raw, d_diff_raw
            else:
                # The acceleration limit is a property of a single wheel, so
                # it is applied at the wheel level and the result is
                # transformed back. The transform is linear, so the wheel
                # level rates are exact linear combinations of the state's
                # own two rates.
                limit = self.accel_limit
                dvl = d_mean_raw - d_diff_raw / 2.0
                dvr = d_mean_raw + d_diff_raw / 2.0
                dvl = limit * jnp.tanh(dvl / limit)
                dvr = limit * jnp.tanh(dvr / limit)
                d_mean = (dvl + dvr) / 2.0
                d_diff = dvr - dvl
        else:
            vl_actual, vr_actual = x[3], x[4]
            v_actual = (vl_actual + vr_actual) / 2.0
            omega_actual = (vr_actual - vl_actual) / track_width

            dvl = (vl_target - vl_actual) / self.tau_s
            dvr = (vr_target - vr_actual) / self.tau_s

            if self.accel_limit is not None:
                # A soft limit: tanh is monotone, so the natural inclusion
                # of it is exact, and it keeps the rate below the limit.
                dvl = self.accel_limit * jnp.tanh(dvl / self.accel_limit)
                dvr = self.accel_limit * jnp.tanh(dvr / self.accel_limit)
            d_mean, d_diff = dvl, dvr

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

        return jnp.array([dx, dy, dtheta, d_mean, d_diff])

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

        ``quantize=True`` models a firmware that rounds the commanded
        linear speed (mm/s) and turn rate (deg/s) before calling
        ``DriveBase.drive()``. The two controllers that are actually in use
        do NOT round: ``ramsete_compact_events.py:179`` and
        ``ramsete.py:337`` pass floats straight through, so the nominal
        path leaves it off. The two that do round
        (``ramsete_fast.py``, ``ramsete_compact.py``) are the ones that are
        not in use. It stays off for the interval inclusion path, which may
        only use operations Immrax can include.
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


class ClosedLoopTickMap(immrax.system.System):
    """One control tick of the closed loop, as a discrete map.

    The controller is inside the map, so the map's Jacobian carries the
    feedback: the reachable set contracts the way the robot's does, instead of
    fanning out the way an open-loop plant's does.

    The state is the plant's state, and then the quantities the closed loop
    needs but the plant does not have:

      * the robot's own pose estimate (the digital twin), because the
        controller reacts to the belief and not to the truth, and the
        feedback term is dN/dx through the belief; and
      * the command from the previous ticks, when the fitted model has a
        command delay - the robot drives the command it was given a moment
        ago, so that command's dependence on the older state belongs in the
        map as well.

    The tick duration rides in the last slot of the reference vector: it
    changes from tick to tick (the last tick of a sample interval is
    shorter), and a value that changes must be a traced argument, or the
    inclusion function would be rebuilt for every tick.
    """

    def __init__(self, plant, controller, track_width,
                 substeps: int = PLANT_SUBSTEPS,
                 carry_belief: bool = False,
                 belief_scale=(1.0, 1.0), gyro_bias: float = 0.0,
                 delay_ticks: int = 0, wheel_state: str = "wheels",
                 belief_mode: str = "estimate"):
        self.evolution = "discrete"
        self.plant = plant
        self.controller = controller
        self.track_width = float(track_width)
        self.substeps = int(substeps)
        self.carry_belief = bool(carry_belief)
        self.distance_scale = float(belief_scale[0])
        self.heading_scale = float(belief_scale[1])
        self.gyro_bias = float(gyro_bias)
        self.delay_ticks = int(delay_ticks)
        self.wheel_state = wheel_state
        if belief_mode not in BELIEF_MODES:
            raise ValueError(
                f"belief_mode must be one of {BELIEF_MODES}, "
                f"got {belief_mode!r}")
        self.belief_mode = belief_mode
        self.n_plant = plant.xlen
        self.n_belief = 3 if carry_belief else 0
        self.n_cmds = 2 * self.delay_ticks
        self.xlen = self.n_plant + self.n_belief + self.n_cmds

    def f(self, t, x, u_ref, w):
        dt = u_ref[6]
        n = self.n_plant
        plant_x = x[:n]
        if not self.carry_belief:
            belief = plant_x[:3]
            error = None
        elif self.belief_mode == "error":
            # The state carries the ESTIMATE'S ERROR, not the estimate.
            #
            # The robot's estimate is the truth plus the odometry's error,
            # and the odometry reads the very wheel speeds the plant
            # carries, so the two are perfectly correlated. An interval box
            # cannot say so: an estimate carried as a state of its own has
            # its spread ADDED to the truth's, and the controller reacts to
            # a belief that no robot ever had. That is the same lost
            # correlation as the phantom turn rate, one level up.
            #
            # Carrying the error keeps the shared part once: the estimate
            # is the truth plus a small error, and the box adds only the
            # error's own spread.
            error = x[n:n + 3]
            belief = plant_x[:3] + error
        else:
            error = None
            belief = x[n:n + 3]

        # normalize=False: the angle normalisation is a modulo, and immrax's
        # interval arithmetic has no inclusion rule for it.  The plan's
        # headings are unrolled before they get here, so the difference is
        # already the short way and the wrap is not needed - the same reason
        # the "interval" feedback mode uses the raw headings.
        u_cmd = self.controller(0.0, belief, u_ref[:6], w, normalize=False,
                                quantize=False)

        if self.delay_ticks:
            # The slots hold the commands of the last delay_ticks ticks,
            # newest first, and the oldest one is the command the robot is
            # still driving.
            cmds = x[n + self.n_belief:]
            u_applied = cmds[2 * (self.delay_ticks - 1):2 * self.delay_ticks]
        else:
            u_applied = u_cmd

        h = dt / self.substeps
        plant_next = plant_x
        belief_next = belief
        error_next = error
        for _ in range(self.substeps):
            if self.carry_belief:
                if self.belief_mode == "error":
                    # The error moves with the same speeds and in the same
                    # order as the plant, so a perfect odometry keeps the
                    # estimate on the truth.
                    error_next = self._advance_error(error_next, plant_next, h)
                else:
                    belief_next = self._advance_belief(belief_next, plant_next,
                                                       h)
            plant_next = plant_next + self.plant.f(
                0.0, plant_next, u_applied, w) * h

        parts = [plant_next]
        if self.carry_belief:
            parts.append(error_next if self.belief_mode == "error"
                         else belief_next)
        if self.delay_ticks:
            # The new command goes in front and the oldest one falls off.
            parts.append(jnp.concatenate(
                [u_cmd, cmds[:2 * (self.delay_ticks - 1)]]))
        return jnp.concatenate(parts)

    def _wheel_motion(self, plant_x):
        """The robot's forward speed and turn rate, from the plant's state."""
        v_mean = plant_x[3]
        v_diff = plant_x[4]
        if self.wheel_state == "mean_diff":
            return v_mean, v_diff / self.track_width
        vl, vr = v_mean, v_diff
        return (vl + vr) / 2.0, (vr - vl) / self.track_width

    def _advance_error(self, error, plant_x, dt):
        """The estimate's error, advanced by the odometry minus the truth.

        The estimate's heading advances with the odometry's turn rate and
        its position with the odometry's speed along its own heading; the
        truth's advance with the true turn rate and the true speed along
        the truth's heading. The difference between the two is the error.
        """
        v_true, omega_true = self._wheel_motion(plant_x)
        odo_v = v_true * self.distance_scale
        odo_w = omega_true * self.heading_scale + self.gyro_bias

        truth_heading = plant_x[2]
        e_heading = error[2]
        # The estimate's heading after the step: its own heading, moved by
        # the odometry's turn rate.
        belief_heading = truth_heading + e_heading + odo_w * dt
        e_heading = e_heading + (odo_w - omega_true) * dt

        ex = error[0] + (odo_v * jnp.cos(belief_heading)
                         - v_true * jnp.cos(truth_heading)) * dt
        ey = error[1] + (odo_v * jnp.sin(belief_heading)
                         - v_true * jnp.sin(truth_heading)) * dt
        return jnp.array([ex, ey, e_heading])

    def _advance_belief(self, belief, plant_x, dt):
        """The robot's own pose estimate, advanced from the wheel motion."""
        v_true, omega_true = self._wheel_motion(plant_x)
        odo_v = v_true * self.distance_scale
        odo_w = omega_true * self.heading_scale + self.gyro_bias
        # Indexed, not unpacked.  Unpacking a traced array is the JAX
        # "unstack" primitive, and Immrax's natural inclusion has no rule
        # for it, so the tick map dies with "unstack not in
        # inclusion_registry" (KI-050).
        x = belief[0]
        y = belief[1]
        heading = belief[2]
        heading = heading + odo_w * dt
        x = x + odo_v * jnp.cos(heading) * dt
        y = y + odo_v * jnp.sin(heading) * dt
        return jnp.array([x, y, heading])




class ImmraxValidator:
    def __init__(self, config: JAXRobotConfig,
                 ramsete_b: float = 2.0, ramsete_zeta: float = 0.7,
                 control_dt: float = CONTROL_DT,
                 feedback_mode: str = "nominal",
                 digital_twin: Optional[DigitalTwinConfig] = None,
                 plant_inclusion: Optional[str] = None,
                 controller_inclusion: Optional[str] = None,
                 wheel_state: Optional[str] = None,
                 propagation: Optional[str] = None,
                 tick_inclusion: Optional[str] = None,
                 belief_mode: Optional[str] = None):
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

        # Which inclusion function embeds the plant and the controller. The
        # defaults are the module constants, so a run is reproducible and the
        # choice can be overridden per validator for an A/B measurement.
        self.plant_inclusion = plant_inclusion or PLANT_INCLUSION
        self.controller_inclusion = controller_inclusion or CONTROLLER_INCLUSION
        self.wheel_state = wheel_state or DigitalTwinOpenLoop.WHEEL_STATE
        if self.wheel_state not in DigitalTwinOpenLoop.WHEEL_STATES:
            raise ValueError(
                f"wheel_state must be one of {DigitalTwinOpenLoop.WHEEL_STATES}, "
                f"got {self.wheel_state!r}")
        if self.plant_inclusion not in PLANT_INCLUSIONS:
            raise ValueError(
                f"plant_inclusion must be one of {PLANT_INCLUSIONS}, "
                f"got {self.plant_inclusion!r}")
        if self.controller_inclusion not in CONTROLLER_INCLUSIONS:
            raise ValueError(
                f"controller_inclusion must be one of {CONTROLLER_INCLUSIONS}, "
                f"got {self.controller_inclusion!r}")
        # "plant" propagates the plant and evaluates the controller outside the
        # propagation; "tick" embeds a tick of the closed loop and propagates
        # that.  See the PROPAGATION comment above.
        self.propagation = propagation or PROPAGATION
        if self.propagation not in PROPAGATIONS:
            raise ValueError(
                f"propagation must be one of {PROPAGATIONS}, "
                f"got {self.propagation!r}")
        self.tick_inclusion = tick_inclusion or TICK_INCLUSION
        if self.tick_inclusion not in TICK_INCLUSIONS:
            raise ValueError(
                f"tick_inclusion must be one of {TICK_INCLUSIONS}, "
                f"got {self.tick_inclusion!r}")
        self.belief_mode = belief_mode or BELIEF_MODE
        if self.belief_mode not in BELIEF_MODES:
            raise ValueError(
                f"belief_mode must be one of {BELIEF_MODES}, "
                f"got {self.belief_mode!r}")

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
        self.emb_sys = _plant_embedding(self.sys, self.plant_inclusion)

        # Sampled-data path: hold the controller command for one 20 ms tick,
        # then integrate the physical plant. This mirrors ramsete_fast.py
        # instead of evaluating the controller continuously.
        if self._digital_twin_active and (
                digital_twin.actuator_tau_s() > 0.0
                or digital_twin.actuator_accel_limit() is not None):
            self.plant = DigitalTwinOpenLoop(config, digital_twin,
                                             wheel_state=wheel_state)
        else:
            self.plant = RobotOpenLoop(config)
        # The live embedding: the plant alone, propagated by the sampled-data
        # loop below. The controller is evaluated outside it, at the tube
        # midpoint, so the controller is not part of what gets embedded.
        self.plant_emb_sys = _plant_embedding(self.plant, self.plant_inclusion)
        self.controller = RamseteController(
            self.ramsete_b, self.ramsete_zeta, track_width=config.track_width)
        self._controller_if = _controller_inclusion(
            self._controller_command_interval, self.controller_inclusion)
        self._sampled_control = True
        self.reachability_engine = CompiledReachabilityEngine(self)

        # The closed-loop tick map, for the "tick" propagation. The robot's
        # own pose estimate rides in the map's state when the plant has
        # wheel-speed states, because the controller reacts to the belief and
        # not to the truth, and the feedback term is dN/dx through it. The
        # command delay rides there too when the fit measured one: the robot
        # drives the command it was given a moment ago, so that command's
        # dependence on the older state belongs in the map as well.
        self.tick_map = None
        self.tick_emb_sys = None
        self._carry_belief = bool(
            self._digital_twin_active and self.plant.xlen > 3)
        self._tick_delay_ticks = 0
        if getattr(self, "propagation", PROPAGATION) == "tick":
            if self.tick_inclusion == "nat":
                raise ValueError(
                    "the tick propagation needs a Jacobian inclusion: with the "
                    "natural extension the closed-loop map explodes "
                    "(docs/research/rq_a/exp20_closed_loop_tick.py)")
            self._tick_delay_ticks = max(1, int(math.ceil(
                self.digital_twin_delay_s / self.control_dt))) \
                if (self._digital_twin_active
                    and self.digital_twin_delay_s > 0.0) else 0
            self.tick_map = ClosedLoopTickMap(
                self.plant, self.controller, config.track_width,
                substeps=PLANT_SUBSTEPS,
                carry_belief=self._carry_belief,
                belief_scale=(self.digital_twin_distance_scale,
                              self.digital_twin_heading_scale),
                gyro_bias=self.digital_twin_gyro_bias_rad_s,
                delay_ticks=self._tick_delay_ticks,
                wheel_state=self.wheel_state,
                belief_mode=self.belief_mode)
            self.tick_emb_sys = _plant_embedding(self.tick_map,
                                                 self.tick_inclusion)

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
        # The plant's two wheel-speed numbers, read in the coordinates the
        # plant carries them in. "mean_diff" holds the mean wheel speed and
        # the difference between the wheels; "wheels" holds the two wheel
        # speeds. The belief's speed and turn rate are the same either way.
        v_mean = float((x_flat[3] + x_flat[state_len + 3]) / 2.0)
        v_diff = float((x_flat[4] + x_flat[state_len + 4]) / 2.0)
        track_width = self.cfg.track_width
        if self.wheel_state == "mean_diff":
            odo_v = v_mean
            odo_w = v_diff / track_width
        else:
            vl, vr = v_mean, v_diff
            odo_v = (vl + vr) / 2.0
            odo_w = (vr - vl) / track_width

        odo_v = odo_v * self.digital_twin_distance_scale
        odo_w = odo_w * self.digital_twin_heading_scale
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

    def _empty_report(self) -> Dict[str, Any]:
        """The report for a candidate that produced no trajectory.

        A failed solve is not a validation error: the report says so in
        words, carries every key a real one has, and does not pass.
        """
        return {
            "max_motor_violation_N": 0.0,
            "max_traction_violation_N": 0.0,
            "max_slip_l_N": 0.0,
            "max_slip_r_N": 0.0,
            "max_tracking_error_m": float("inf"),
            "rms_tracking_error_m": float("inf"),
            "max_heading_error_rad": float("inf"),
            "rms_heading_error_rad": float("inf"),
            "max_envelope_radius_m": float("inf"),
            "max_wheel_speed_m_s": 0.0,
            "wheel_speed_limit_m_s": 0.0,
            "speed_violation_m_s": 0.0,
            "utilization": {
                "motor": float("inf"),
                "traction": float("inf"),
                "slip": float("inf"),
                "wheel_speed": float("inf"),
            },
            "reachability": {
                "max_error_m": float("inf"),
                "max_tracking_error_m": float("inf"),
                "max_heading_error_rad": float("inf"),
                "max_envelope_radius_m": float("inf"),
                "envelope": [],
                "diverged": True,
                "envelope_truncated": True,
                "empty": True,
            },
            "inclusion": {
                "plant": self.plant_inclusion,
                "controller": self.controller_inclusion,
                "tick": self.tick_inclusion,
            },
            "propagation": getattr(self, "propagation", PROPAGATION),
            "digital_twin": (
                self.digital_twin.summary()
                if self.digital_twin is not None else None
            ),
            "passed": False,
            "error": "the solver produced no samples",
        }

    def validate_trajectory(self, samples: List[Dict[str, Any]],
                            cof_range: Tuple[float, float],
                            torque_margin_range: Tuple[float, float],
                            backlash_range: Tuple[float, float],
                            track_width_range: Optional[Tuple[float, float]] = None,
                            wheel_radius_range: Optional[Tuple[float, float]] = None,
                            inertia_range: Optional[Tuple[float, float]] = None,
                            safety_threshold_m: float = 0.005) -> Dict[str, Any]:
        if not samples:
            # An empty list used to reach jnp.max and raise "zero-size array
            # to reduction operation max", which read as an Immrax error
            # (KI-056).
            return self._empty_report()

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

            # Normalized utilisation (1.0 = exactly at the limit), for the
            # decision score. A violation only says how far past the limit the
            # plan is; the ratio also says how much margin a passing plan has.
            motor_util_l = iv_abs_max(fl) / jnp.maximum(force_limit_l.lower, 1e-6)
            motor_util_r = iv_abs_max(fr) / jnp.maximum(force_limit_r.lower, 1e-6)

            # 3. Dynamic Normal Force Transfers. Only lateral (turning)
            # transfer applies; a two-wheel robot has no front/rear split
            # (KI-010). Same formula as the optimizer and the CPU audit.
            v_val, omega_val = diff_drive_fk(vl_val, vr_val, track_width_iv)
            centripetal_accel = v_val * omega_val
            nl, nr = wheel_normal_forces(
                self.cfg.mass, self.cfg.g, CG_HEIGHT_M, track_width_iv, centripetal_accel
            )

            # 4. Global Friction Circle & Local Wheel Slip Violations
            traction_max = total_traction_limit(self.cfg.mass, self.cfg.g, cof_interval).lower
            tr_l, tr_r = traction_limits(nl, nr, cof_interval)
            traction_violation = jnp.maximum(0.0, iv_abs_max(fl) + iv_abs_max(fr) - traction_max)

            slip_l = jnp.maximum(0.0, iv_abs_max(fl) - tr_l.lower)
            slip_r = jnp.maximum(0.0, iv_abs_max(fr) - tr_r.lower)
            traction_util = (iv_abs_max(fl) + iv_abs_max(fr)) / jnp.maximum(traction_max, 1e-6)
            slip_util_l = iv_abs_max(fl) / jnp.maximum(tr_l.lower, 1e-6)
            slip_util_r = iv_abs_max(fr) / jnp.maximum(tr_r.lower, 1e-6)

            return (motor_violation_l, motor_violation_r, traction_violation,
                    slip_l, slip_r, motor_util_l, motor_util_r,
                    traction_util, slip_util_l, slip_util_r)

        (motor_vios_l, motor_vios_r, traction_vios, slip_ls, slip_rs,
         motor_utils_l, motor_utils_r, traction_utils,
         slip_utils_l, slip_utils_r) = vmap(check_sample_interval)(vl, vr, al, ar)

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
        wheel_speed_limit = self.cfg.no_load_wheel_speed_worst_case()
        speed_violation = max(0.0, max_wheel_speed - wheel_speed_limit)

        # Normalized utilisation of every physical limit, worst case over the
        # samples: 1.0 is exactly at the limit. The decision score reads these
        # numbers instead of parsing the reason strings.
        motor_util = max(float(jnp.max(motor_utils_l)), float(jnp.max(motor_utils_r)))
        slip_util = max(float(jnp.max(slip_utils_l)), float(jnp.max(slip_utils_r)))
        traction_util = float(jnp.max(traction_utils))
        wheel_speed_util = (
            max_wheel_speed / wheel_speed_limit if wheel_speed_limit > 1e-9 else 0.0)

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
            "rms_tracking_error_m": reach_report.get('rms_tracking_error_m', 0.0),
            "max_heading_error_rad": reach_report['max_heading_error_rad'],
            "rms_heading_error_rad": reach_report.get('rms_heading_error_rad', 0.0),
            "max_envelope_radius_m": reach_report['max_envelope_radius_m'],
            "max_wheel_speed_m_s": max_wheel_speed,
            "wheel_speed_limit_m_s": wheel_speed_limit,
            "speed_violation_m_s": speed_violation,
            "utilization": {
                "motor": float(motor_util),
                "traction": float(traction_util),
                "slip": float(slip_util),
                "wheel_speed": float(wheel_speed_util),
            },
            "reachability": reach_report,
            "inclusion": {
                "plant": self.plant_inclusion,
                "controller": self.controller_inclusion,
                # The tick propagation does not use the plant or the
                # controller inclusion: it embeds the whole closed-loop tick
                # map with one inclusion, and that is the one that produced
                # the tube.  Say which.
                "tick": self.tick_inclusion,
            },
            "propagation": getattr(self, "propagation", PROPAGATION),
            "digital_twin": (
                self.digital_twin.summary()
                if self.digital_twin is not None else None
            ),
            "passed": bool(is_physically_safe)
        }

    def _tick_initial_state(self, samples):
        """The tick map's starting box: 1 mm on the pose, the rest exact.

        The state is the plant's state, then the belief, then the delayed
        commands (see ClosedLoopTickMap). The flat form is the lower bounds
        followed by the upper bounds.
        """
        n_plant = self.plant.xlen
        start_pose = jnp.array([samples[0]['x'], samples[0]['y'],
                                samples[0]['heading']])

        # The state is the plant's state, then the belief, then the delayed
        # commands. The pose starts with 1 mm of uncertainty; everything else
        # starts exact.
        lower = jnp.concatenate([start_pose - 0.001,
                                 jnp.zeros(n_plant - 3)])
        upper = jnp.concatenate([start_pose + 0.001,
                                 jnp.zeros(n_plant - 3)])
        if self._carry_belief:
            if self.belief_mode == "error":
                # The estimate's ERROR starts as wide as the truth's own
                # start uncertainty: the robot's estimate is anchored where
                # the plan says and the truth may be a millimetre away, so
                # the difference between them is a millimetre wide. The two
                # are perfectly anti-correlated, which a box cannot say, so
                # the error gets its own width - and the estimate is then
                # the truth plus it.
                lower = jnp.concatenate([lower, jnp.zeros(3) - 0.001])
                upper = jnp.concatenate([upper, jnp.zeros(3) + 0.001])
            else:
                # The robot's own estimate starts where the robot starts,
                # and it is exact: it is a reading, not an uncertain
                # quantity.
                lower = jnp.concatenate([lower, start_pose])
                upper = jnp.concatenate([upper, start_pose])
        if self._tick_delay_ticks:
            # There was no command before the run started.
            zeros = jnp.zeros(2 * self._tick_delay_ticks)
            lower = jnp.concatenate([lower, zeros])
            upper = jnp.concatenate([upper, zeros])
        return jnp.concatenate([lower, upper])

    def _reachability_tick_reference(self, samples, w_interval, backlash_half):
        """Propagate the tube through the closed-loop tick map.

        This is the original Python-loop rollout, kept unaltered as the
        parity oracle for the compiled ``_reachability_tick``. Validation no
        longer runs it.

        The "plant" propagation embeds the plant and evaluates the controller
        *outside* the propagation, at the tube's midpoint, so the interval
        arithmetic never sees the feedback: what it bounds is the reachable
        set of an open-loop plant driven by a pre-recorded command, and the
        real robot is a contracting closed loop. The tick map puts the
        controller inside the map, so the map's Jacobian carries the feedback
        term dF/dx = dPhi/dx + (dPhi/du)(dN/dx), and the tube contracts the
        way the robot's does.

        The model is also a different one, and in the twin's case a more
        conservative one: the controller is fed the whole tube (or, with the
        digital twin, the whole *belief* interval), not the robot's own best
        estimate, so the set it bounds contains the set the real robot
        reaches. That is also why the twin's tube is *wider* under this
        propagation and not narrower: the belief carried in an interval state
        grows with the wheel-speed uncertainty, and its growth is added to
        the plant's, because the belief and the wheel speeds are perfectly
        correlated and an interval box cannot say so.
        """
        N = len(samples)
        n_state = self.tick_map.xlen
        x_flat = self._tick_initial_state(samples)

        def _box(flat):
            # Slicing: the first n_state entries are lower bounds, the next
            # n_state entries are upper bounds.
            return {
                "x_min": float(flat[0]), "x_max": float(flat[n_state]),
                "y_min": float(flat[1]), "y_max": float(flat[n_state + 1]),
                "theta_min": float(flat[2]), "theta_max": float(flat[n_state + 2]),
                "tracking_error_m": 0.0,
                "heading_error_rad": 0.0,
            }

        ramsete_b = getattr(self, "ramsete_b", 2.0)
        envelope = [_box(x_flat)]
        max_tracking_error = 0.0
        max_heading_error = 0.0
        max_envelope_radius = 0.0
        max_total_error = 0.0
        sum_sq_tracking = 0.0
        sum_sq_heading = 0.0
        n_tracked = 0
        diverged = False

        for i in range(N - 1):
            s = samples[i]
            s_next = samples[i + 1]
            dt = s_next['t'] - s['t']

            if not math.isfinite(dt) or dt <= 0:
                # A failed solve can produce a non-monotone time vector. Do
                # not integrate backwards; report the tube as unknown so the
                # caller cannot export it.
                diverged = True
                max_tracking_error = float("inf")
                max_envelope_radius = float("inf")
                max_total_error = float("inf")
                envelope.append({**envelope[-1], "unknown": True})
                continue

            if not diverged:
                t = s['t']
                while t < s['t'] + dt - 1e-12:
                    dt_ctrl = min(self.control_dt, s['t'] + dt - t)
                    frac = (t - s['t']) / dt
                    ref = _interpolated_reference(
                        s, s_next, dt, self.cfg.track_width, frac, ramsete_b)
                    # The tick duration rides in the last slot of the
                    # reference: it changes from tick to tick, and a value
                    # that changes has to be a traced argument, or the
                    # inclusion function would be rebuilt for every tick.
                    u_ref = jnp.concatenate(
                        [jnp.array(ref), jnp.array([dt_ctrl])])
                    x_flat = self.tick_emb_sys.f(0.0, x_flat, u_ref,
                                                 w_interval)
                    t += dt_ctrl

            if not diverged:
                env = _box(x_flat)
                # The gear backlash is a position dead-band, carried in the
                # reachable set rather than in the motion (KI-025).
                if backlash_half > 0.0:
                    elapsed = s_next['t'] - samples[0]['t']
                    # Never let the spread go negative: that would shrink
                    # the safety envelope instead of widening it.
                    spread = backlash_half * max(0.0, elapsed)
                    env = {
                        **env,
                        "x_min": env["x_min"] - spread,
                        "x_max": env["x_max"] + spread,
                        "y_min": env["y_min"] - spread,
                        "y_max": env["y_max"] + spread,
                    }
                mid_x = (x_flat[0] + x_flat[n_state]) / 2.0
                mid_y = (x_flat[1] + x_flat[n_state + 1]) / 2.0
                tracking_error = float(jnp.sqrt(
                    (mid_x - s_next['x'])**2 + (mid_y - s_next['y'])**2))
                mid_heading = (x_flat[2] + x_flat[n_state + 2]) / 2.0
                heading_error = float(abs(
                    (mid_heading - s_next['heading'] + math.pi)
                    % (2 * math.pi) - math.pi))
                env["tracking_error_m"] = tracking_error
                env["heading_error_rad"] = heading_error
                radius = float(jnp.sqrt(
                    ((env["x_max"] - env["x_min"]) / 2.0)**2
                    + ((env["y_max"] - env["y_min"]) / 2.0)**2))

                max_tracking_error = max(max_tracking_error, tracking_error)
                max_heading_error = max(max_heading_error, heading_error)
                max_envelope_radius = max(max_envelope_radius, radius)
                max_total_error = max(max_total_error, tracking_error + radius)
                sum_sq_tracking += tracking_error ** 2
                sum_sq_heading += heading_error ** 2
                n_tracked += 1

                # The same 50 mm guard as the plant propagation: a cut-off
                # propagation is reported as a lower bound, not as a width.
                if radius > 0.05:
                    diverged = True
                envelope.append({**env, "unknown": True} if diverged else env)
            else:
                envelope.append({**envelope[-1], "unknown": True})

        return {
            "max_error_m": max_total_error,
            "max_tracking_error_m": max_tracking_error,
            "rms_tracking_error_m": math.sqrt(sum_sq_tracking / n_tracked) if n_tracked else 0.0,
            "max_heading_error_rad": max_heading_error,
            "rms_heading_error_rad": math.sqrt(sum_sq_heading / n_tracked) if n_tracked else 0.0,
            "max_envelope_radius_m": max_envelope_radius,
            "envelope": envelope,
            "diverged": diverged,
            "envelope_truncated": bool(diverged),
            "propagation": "tick",
        }

    def _tick_schedule(self, samples, backlash_half):
        """Every control tick of the rollout, precomputed on the host.

        The reference is a point, not an interval, so building it here keeps
        the traced graph down to the tick map itself. The tick grid is
        enumerated with the reference loop's own stepping rule (20 ms ticks,
        a shortened final one), so the schedule cannot disagree with the
        oracle. Padding with inactive zero-duration ticks gives every sample
        interval the same tick count, so one compiled ``lax.scan`` covers the
        whole path and the inclusion function is never rebuilt.
        """
        control_dt = self.control_dt
        track_width = self.cfg.track_width
        ramsete_b = getattr(self, "ramsete_b", 2.0)
        t0 = samples[0]['t']

        intervals = []
        valid = []
        for i in range(len(samples) - 1):
            s = samples[i]
            s_next = samples[i + 1]
            dt = s_next['t'] - s['t']
            ok = math.isfinite(dt) and dt > 0.0
            valid.append(ok)
            ticks = []
            if ok:
                t = s['t']
                while t < s['t'] + dt - 1e-12:
                    dt_ctrl = min(control_dt, s['t'] + dt - t)
                    frac = (t - s['t']) / dt
                    ref = _interpolated_reference(
                        s, s_next, dt, track_width, frac, ramsete_b)
                    ticks.append(tuple(ref) + (dt_ctrl,))
                    t += dt_ctrl
            intervals.append(ticks)

        n_ticks = max([len(ticks) for ticks in intervals] + [1])
        count = len(samples) - 1
        refs = np.zeros((count, n_ticks, 7), dtype=np.float64)
        active = np.zeros((count, n_ticks), dtype=bool)
        spreads = np.zeros(count, dtype=np.float64)
        targets = np.zeros((count, 3), dtype=np.float64)
        for i, ticks in enumerate(intervals):
            s_next = samples[i + 1]
            targets[i] = (s_next['x'], s_next['y'], s_next['heading'])
            if valid[i]:
                # The gear backlash is a position dead-band, carried in the
                # reachable set rather than in the motion (KI-025). Never let
                # the spread go negative: that would shrink the envelope.
                spreads[i] = backlash_half * max(0.0, s_next['t'] - t0)
            for k, tick in enumerate(ticks):
                refs[i, k] = tick
                active[i, k] = True
        return (refs, active, np.array(valid, dtype=bool), spreads,
                targets, n_ticks)

    def _tick_rollout(self, num_ticks):
        """The jitted ``lax.scan`` rollout of the closed-loop tick map.

        Cached per tick count: the inclusion function is built once here and
        called once per tick by the scan, never rebuilt per tick.
        """
        cache = getattr(self, "_tick_rollout_cache", None)
        if cache is None:
            cache = self._tick_rollout_cache = {}
        if num_ticks in cache:
            return cache[num_ticks]
        with JAX_COMPILE_LOCK:
            if num_ticks in cache:
                return cache[num_ticks]
            fn = jit(self._tick_rollout_body())
            cache[num_ticks] = fn
            return fn

    def _tick_rollout_body(self):
        emb_f = self.tick_emb_sys.f
        n_state = self.tick_map.xlen

        def rollout(refs, active, valid, spreads, targets, x_init,
                    w_lower, w_upper):
            w_interval = Interval(w_lower, w_upper)

            def interval_step(carry, inp):
                x_flat, diverged, last_box, last_track, last_head = carry
                u_refs, tick_active, is_valid, spread, target = inp
                # Once the tube is cut off, or on a non-monotone time vector,
                # nothing is propagated and the last box is carried forward.
                run = is_valid & (~diverged)

                def tick_step(x_cur, tick_inp):
                    u_ref, act = tick_inp
                    x_next = emb_f(0.0, x_cur, u_ref, w_interval)
                    return jnp.where(act, x_next, x_cur), None

                x_end, _ = lax.scan(tick_step, x_flat,
                                    (u_refs, tick_active & run))

                x_min = x_end[0] - spread
                x_max = x_end[n_state] + spread
                y_min = x_end[1] - spread
                y_max = x_end[n_state + 1] + spread
                mid_x = 0.5 * (x_end[0] + x_end[n_state])
                mid_y = 0.5 * (x_end[1] + x_end[n_state + 1])
                mid_h = 0.5 * (x_end[2] + x_end[n_state + 2])
                tracking = jnp.sqrt(
                    (mid_x - target[0]) ** 2 + (mid_y - target[1]) ** 2)
                # The heading difference is taken the short way, as the
                # firmware does (KI-027).
                heading = jnp.abs(
                    jnp.mod(mid_h - target[2] + jnp.pi, 2.0 * jnp.pi)
                    - jnp.pi)
                radius = jnp.sqrt(
                    ((x_max - x_min) * 0.5) ** 2
                    + ((y_max - y_min) * 0.5) ** 2)

                computed_box = jnp.array(
                    [x_min, x_max, y_min, y_max,
                     x_end[2], x_end[n_state + 2]])
                box = jnp.where(run, computed_box, last_box)
                track = jnp.where(run, tracking, last_track)
                head = jnp.where(run, heading, last_head)
                # The same 50 mm guard as the plant propagation: a cut-off
                # propagation is reported as a lower bound, not as a width.
                new_diverged = diverged | (~is_valid) | (run & (radius > 0.05))

                outs = (
                    box,
                    track,
                    head,
                    jnp.where(run, tracking, 0.0),
                    jnp.where(run, heading, 0.0),
                    jnp.where(run, radius, 0.0),
                    run,
                    new_diverged,
                )
                return (x_end, new_diverged, box, track, head), outs

            init_box = jnp.array(
                [x_init[0], x_init[n_state],
                 x_init[1], x_init[n_state + 1],
                 x_init[2], x_init[n_state + 2]])
            (_, diverged, _, _, _), outs = lax.scan(
                interval_step,
                (x_init, jnp.array(False), init_box,
                 jnp.array(0.0), jnp.array(0.0)),
                (refs, active, valid, spreads, targets))
            return {
                "boxes": outs[0],
                "emitted_tracking": outs[1],
                "emitted_heading": outs[2],
                "m_tracking": outs[3],
                "m_heading": outs[4],
                "m_radius": outs[5],
                "contrib": outs[6],
                "unknown": outs[7],
                "diverged": diverged,
            }

        return rollout

    def _reachability_tick(self, samples, w_interval, backlash_half):
        """Propagate the tube through the closed-loop tick map, compiled.

        The "plant" propagation embeds the plant and evaluates the controller
        *outside* the propagation, at the tube's midpoint, so the interval
        arithmetic never sees the feedback: what it bounds is the reachable
        set of an open-loop plant driven by a pre-recorded command, and the
        real robot is a contracting closed loop. The tick map puts the
        controller inside the map, so the map's Jacobian carries the feedback
        term dF/dx = dPhi/dx + (dPhi/du)(dN/dx), and the tube contracts the
        way the robot's does.

        The model is also a different one, and in the twin's case a more
        conservative one: the controller is fed the whole tube (or, with the
        digital twin, the whole *belief* interval), not the robot's own best
        estimate, so the set it bounds contains the set the real robot
        reaches. That is also why the twin's tube is *wider* under this
        propagation and not narrower: the belief carried in an interval state
        grows with the wheel-speed uncertainty, and its growth is added to
        the plant's, because the belief and the wheel speeds are perfectly
        correlated and an interval box cannot say so.

        The rollout is a ``jax.lax.scan`` over a precomputed tick schedule
        (see ``_tick_schedule``); the semantics - the reference and its
        midpoint lead, the shortened final tick, delayed commands, the belief
        state, the interval state layout, the backlash spread, the tracking
        and heading errors, the 50 mm guard and the unknown marking - are the
        ones ``_reachability_tick_reference`` computes, and the parity tests
        compare the two.
        """
        N = len(samples)
        n_state = self.tick_map.xlen
        x_init = self._tick_initial_state(samples)
        refs, active, valid, spreads, targets, num_ticks = \
            self._tick_schedule(samples, backlash_half)
        out = self._tick_rollout(num_ticks)(
            jnp.asarray(refs), jnp.asarray(active), jnp.asarray(valid),
            jnp.asarray(spreads), jnp.asarray(targets), x_init,
            w_interval.lower, w_interval.upper)

        envelope = [{
            "x_min": float(x_init[0]), "x_max": float(x_init[n_state]),
            "y_min": float(x_init[1]), "y_max": float(x_init[n_state + 1]),
            "theta_min": float(x_init[2]),
            "theta_max": float(x_init[n_state + 2]),
            "tracking_error_m": 0.0,
            "heading_error_rad": 0.0,
        }]
        max_tracking_error = 0.0
        max_heading_error = 0.0
        max_envelope_radius = 0.0
        max_total_error = 0.0
        sum_sq_tracking = 0.0
        sum_sq_heading = 0.0
        n_tracked = 0
        for i in range(N - 1):
            if bool(out["contrib"][i]):
                tracking = float(out["m_tracking"][i])
                heading = float(out["m_heading"][i])
                radius = float(out["m_radius"][i])
                max_tracking_error = max(max_tracking_error, tracking)
                max_heading_error = max(max_heading_error, heading)
                max_envelope_radius = max(max_envelope_radius, radius)
                max_total_error = max(max_total_error, tracking + radius)
                sum_sq_tracking += tracking ** 2
                sum_sq_heading += heading ** 2
                n_tracked += 1
            env = {
                "x_min": float(out["boxes"][i, 0]),
                "x_max": float(out["boxes"][i, 1]),
                "y_min": float(out["boxes"][i, 2]),
                "y_max": float(out["boxes"][i, 3]),
                "theta_min": float(out["boxes"][i, 4]),
                "theta_max": float(out["boxes"][i, 5]),
                "tracking_error_m": float(out["emitted_tracking"][i]),
                "heading_error_rad": float(out["emitted_heading"][i]),
            }
            if bool(out["unknown"][i]):
                env["unknown"] = True
            envelope.append(env)

        diverged = bool(out["diverged"])
        if not all(valid):
            # A failed solve can produce a non-monotone time vector. Do not
            # integrate backwards; report the tube as unknown so the caller
            # cannot export it. The heading error is left alone, as the
            # reference does: it is not a tube measurement.
            max_tracking_error = float("inf")
            max_envelope_radius = float("inf")
            max_total_error = float("inf")

        return {
            "max_error_m": max_total_error,
            "max_tracking_error_m": max_tracking_error,
            "rms_tracking_error_m": math.sqrt(sum_sq_tracking / n_tracked) if n_tracked else 0.0,
            "max_heading_error_rad": max_heading_error,
            "rms_heading_error_rad": math.sqrt(sum_sq_heading / n_tracked) if n_tracked else 0.0,
            "max_envelope_radius_m": max_envelope_radius,
            "envelope": envelope,
            "diverged": diverged,
            "envelope_truncated": bool(diverged),
            "propagation": "tick",
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
                "envelope_truncated": False,
            }

        gain_range = getattr(self, "digital_twin_gain_range", (1.0, 1.0))
        w_lower = jnp.array([cof_range[0], torque_margin_range[0], backlash_range[0], track_width_range[0], wheel_radius_range[0], inertia_range[0], gain_range[0]])
        w_upper = jnp.array([cof_range[1], torque_margin_range[1], backlash_range[1], track_width_range[1], wheel_radius_range[1], inertia_range[1], gain_range[1]])
        w_interval = Interval(w_lower, w_upper)

        if getattr(self, "propagation", PROPAGATION) == "tick":
            return self._reachability_tick(
                samples, w_interval, 0.5 * float(w_upper[2]))

        if not hasattr(self, "reachability_engine") or self.reachability_engine is None:
            self.reachability_engine = CompiledReachabilityEngine(self)
        return self.reachability_engine.compute_reachability_report(
            samples,
            backlash_range=backlash_range,
            track_width_range=track_width_range,
            wheel_radius_range=wheel_radius_range,
            cof_range=cof_range,
            torque_margin_range=torque_margin_range,
            inertia_range=inertia_range,
        )
