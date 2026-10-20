"""
The drivetrain: what the motors can actually deliver, and how the robot gets
from one wheel-speed command to the next.

Why this module exists
----------------------
OVERBOARD's robot config describes the motor with two numbers:
``t_max_nm`` (the stall torque) and ``v_max_rad_s`` (the speed at which the
motor's torque reaches zero). Everything downstream reads that one curve: the
optimizer's wheel-force constraint, the CPU audit, the Immrax validator and
the exported controller. Two numbers are enough for the *curve* because a DC
motor with a back EMF has a straight torque-speed line, and the line is fixed
by its two ends.

What two numbers cannot express is everything else about the drivetrain:

* the gearbox's own friction, which is what makes the robot coast and what
  makes it coast *faster* the faster it goes,
* the braking torque the motor can produce at a given speed, which decides how
  fast the robot can stop,
* the traction limit, which is where the wheels start to spin instead of
  gripping,
* the transient: what the wheel speed actually does between two commands.

This module holds that model in one place, so the optimizer, the validator and
the calibration tools cannot drift apart (the KI-010 rule).

Provenance, and what is a measurement
-------------------------------------
The *structure* of the model is standard textbook motor physics: a back-EMF
torque law ``tau = stall * (duty - omega / omega_no_load)``, a gearbox with
Coulomb plus viscous friction, and a traction cap of ``mu * m * g``.

The SPIKE Prime numbers below are *literature values* taken from a
third-party open-source simulator (https://github.com/martijas/fll-sim,
``packages/sim/src/motor.ts``). That repository has no licence file, so
nothing is copied from it: the equations are written out here and the
constants are cited as what they are -- numbers somebody else published, not
measurements of this robot.

Every number in this module is a *default*. A measurement always wins. The
measured numbers live in ``sysid_fit.json`` and are carried by
``digital_twin.DigitalTwinConfig``; see ``DrivetrainModel.from_fit``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# The motor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorSpec:
    """One motor's characteristic numbers.

    Attributes:
        name: what the motor is, for reports.
        stall_torque_nm: torque at the output shaft with the motor stopped.
        no_load_speed_rad_s: speed at which the back EMF cancels the applied
            voltage, so the motor makes no torque. This is the speed at which
            the torque-speed line reaches zero.
        regulated_speed_rad_s: the fastest speed the firmware regulates to.
            Below the no-load speed, because the controller needs headroom to
            correct the speed.
        gearbox_coulomb_nm: the gearbox's friction, the part that does not
            depend on the speed.
        gearbox_viscous_nm_s: the gearbox's friction, the part that does.
        reflected_inertia_kg_m2: the rotor's inertia seen through the gearbox.
        source: where the numbers came from.
    """

    name: str
    stall_torque_nm: float
    no_load_speed_rad_s: float
    regulated_speed_rad_s: float
    gearbox_coulomb_nm: float = 0.0
    gearbox_viscous_nm_s: float = 0.0
    reflected_inertia_kg_m2: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.stall_torque_nm <= 0.0:
            raise ValueError(
                f"stall torque must be positive, got {self.stall_torque_nm}")
        if self.no_load_speed_rad_s <= 0.0:
            raise ValueError(
                f"no-load speed must be positive, got {self.no_load_speed_rad_s}")
        if self.regulated_speed_rad_s <= 0.0:
            raise ValueError(
                "regulated speed must be positive, got "
                f"{self.regulated_speed_rad_s}")
        if self.gearbox_coulomb_nm < 0.0 or self.gearbox_viscous_nm_s < 0.0:
            raise ValueError("gearbox friction must not be negative")
        if self.reflected_inertia_kg_m2 < 0.0:
            raise ValueError("reflected inertia must not be negative")


# The SPIKE Prime motors, as published by the reference simulator. These are
# literature values: use them to sanity-check a declared config, or as the
# fallback when the robot has not been measured.
SPIKE_PRIME_SMALL = MotorSpec(
    name="SPIKE Prime small",
    stall_torque_nm=0.09,
    no_load_speed_rad_s=13.61,   # 780 deg/s
    regulated_speed_rad_s=11.52,  # 660 deg/s
    gearbox_coulomb_nm=0.012,
    gearbox_viscous_nm_s=0.0002,
    reflected_inertia_kg_m2=2e-5,
    source="literature value from martijas/fll-sim packages/sim/src/motor.ts",
)

SPIKE_PRIME_MEDIUM = MotorSpec(
    name="SPIKE Prime medium",
    stall_torque_nm=0.18,
    no_load_speed_rad_s=22.51,   # 1290 deg/s
    regulated_speed_rad_s=19.37,  # 1110 deg/s
    gearbox_coulomb_nm=0.02,
    gearbox_viscous_nm_s=0.0003,
    reflected_inertia_kg_m2=5e-5,
    source="literature value from martijas/fll-sim packages/sim/src/motor.ts",
)

SPIKE_PRIME_LARGE = MotorSpec(
    name="SPIKE Prime large",
    stall_torque_nm=0.25,
    no_load_speed_rad_s=21.29,   # 1220 deg/s
    regulated_speed_rad_s=18.33,  # 1050 deg/s
    gearbox_coulomb_nm=0.03,
    gearbox_viscous_nm_s=0.0004,
    reflected_inertia_kg_m2=8e-5,
    source="literature value from martijas/fll-sim packages/sim/src/motor.ts",
)

SPIKE_PRIME_MOTORS = (SPIKE_PRIME_SMALL, SPIKE_PRIME_MEDIUM, SPIKE_PRIME_LARGE)


def motor_torque(spec: MotorSpec, duty: float, omega: float) -> float:
    """The torque the motor's windings produce, in N*m.

    The back EMF eats the applied voltage as the motor speeds up, so the
    torque falls linearly with the speed and reaches zero at the no-load
    speed. ``duty`` is the applied voltage as a fraction of the maximum.

    Negative ``duty`` is not a separate case: the same law gives the braking
    torque, because a motor that is driven against its own back EMF is a
    generator.
    """
    normalized = omega / spec.no_load_speed_rad_s
    return spec.stall_torque_nm * (duty - normalized)


def gearbox_drag(spec: MotorSpec, omega: float) -> float:
    """The magnitude of the torque the gearbox's own friction resists with.

    The gearbox's friction always opposes the motion, so it is returned as a
    positive number and the caller applies the direction. The Coulomb part
    is smoothed over a small speed so that the model has no kink at zero.
    """
    # The reference simulator smooths the Coulomb term over 5 deg/s; the same
    # scale in rad/s is 0.0873.
    coulomb = spec.gearbox_coulomb_nm * np.tanh(abs(omega) / 0.0873)
    viscous = spec.gearbox_viscous_nm_s * abs(omega)
    return coulomb + viscous


# ---------------------------------------------------------------------------
# The drivetrain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DrivetrainModel:
    """A robot's drivetrain: motors, wheels, and the ground they push on.

    Attributes:
        spec: the motor's characteristic numbers.
        wheel_radius_m: the wheel's radius.
        mass_kg: the robot's mass.
        gearing: motor revolutions per wheel revolution. The motor turns
            ``gearing`` times faster than the wheel and delivers ``gearing``
            times the torque at the wheel.
        track_width_m: the distance between the wheels.
        coefficient_of_friction: the tyre's grip on the mat.
        gravity: for the traction limit.
    """

    spec: MotorSpec
    wheel_radius_m: float
    mass_kg: float
    gearing: float = 1.0
    track_width_m: float = 0.0965
    coefficient_of_friction: float = 1.0
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.wheel_radius_m <= 0.0:
            raise ValueError(
                f"wheel radius must be positive, got {self.wheel_radius_m}")
        if self.mass_kg <= 0.0:
            raise ValueError(f"mass must be positive, got {self.mass_kg}")
        if self.gearing <= 0.0:
            raise ValueError(f"gearing must be positive, got {self.gearing}")
        if self.track_width_m <= 0.0:
            raise ValueError(
                f"track width must be positive, got {self.track_width_m}")

    # -- the capability curve ------------------------------------------------

    def motor_omega(self, v_wheel: float) -> float:
        """The motor's speed for a wheel speed, in rad/s."""
        return self.gearing * v_wheel / self.wheel_radius_m

    def drive_force(self, v_wheel: float, duty: float = 1.0) -> float:
        """The force the motor can put on the ground at a wheel speed, in N.

        This is the capability: the torque the windings produce at that
        speed, less the gearbox's friction, geared up to the wheel and
        divided by the wheel's radius. It falls to zero at the no-load speed,
        which is the whole point of the curve.
        """
        omega = self.motor_omega(v_wheel)
        torque = motor_torque(self.spec, duty, omega)
        # The gearbox's friction takes torque out of the motion, so it helps
        # when the motor brakes and hurts when it drives.
        direction = 1.0 if omega >= 0.0 else -1.0
        torque -= gearbox_drag(self.spec, omega) * direction
        return max(0.0, torque) * self.gearing / self.wheel_radius_m

    def brake_force(self, v_wheel: float) -> float:
        """The braking force the motor can produce at a wheel speed, in N.

        A motor that is driven against its own back EMF is a generator, and
        the torque it produces grows with the speed. The gearbox's friction
        adds to it.
        """
        omega = self.motor_omega(v_wheel)
        torque = -motor_torque(self.spec, 0.0, omega)
        direction = 1.0 if omega >= 0.0 else -1.0
        torque += gearbox_drag(self.spec, omega) * direction
        return max(0.0, torque) * self.gearing / self.wheel_radius_m

    def traction_limit(self) -> float:
        """The largest force the tyres can put on the ground, in N."""
        return self.coefficient_of_friction * self.mass_kg * self.gravity

    def brake_deceleration(self, v_wheel: float) -> float:
        """How fast the robot can stop from a wheel speed, in m/s^2.

        The motor's braking torque, capped by what the tyres can hold.
        """
        force = min(self.brake_force(v_wheel), self.traction_limit())
        return force / self.mass_kg

    def coast_deceleration(self, v_wheel: float) -> float:
        """How fast the robot slows down with the motors off, in m/s^2.

        Only the gearbox's friction and the tyres' rolling losses slow it
        down, so this is much smaller than the braking deceleration.
        """
        omega = self.motor_omega(v_wheel)
        torque = gearbox_drag(self.spec, omega)
        force = torque * self.gearing / self.wheel_radius_m
        return force / self.mass_kg

    # -- the transient --------------------------------------------------------

    def step_response(
        self,
        command: Callable[[float], float],
        duration_s: float,
        dt_s: float = 0.005,
        initial_v: float = 0.0,
        accel_limit_m_s2: Optional[float] = None,
    ) -> List[Tuple[float, float, float]]:
        """Run a wheel-speed command through the motor and the ground.

        Three things happen, in this order:

        1. The drive base ramps its own speed reference towards the command
           at its acceleration limit. The firmware does this on every
           ``drive()`` call, so the reference never steps.
        2. The motor follows that reference. The duty is the back-EMF
           feed-forward for the reference plus a proportional term on the
           error, and the torque the windings produce falls with the speed,
           which is what makes the curve a curve.
        3. The robot accelerates with the force the wheels push on the
           ground, capped by what the tyres can hold. Past that cap the
           wheel spins and the robot does not accelerate.

        Args:
            command: the commanded wheel speed at a time, in m/s.
            duration_s: how long to run.
            dt_s: the integration step.
            initial_v: the wheel's speed at the start.
            accel_limit_m_s2: the drive base's acceleration limit. None uses
                the limit the motor can deliver.

        Returns:
            A list of ``(t, wheel_speed, distance)`` rows. The distance is
            the ground the robot covered.
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt must be positive, got {dt_s}")
        if accel_limit_m_s2 is None:
            accel_limit_m_s2 = self.drive_force(0.0) / self.mass_kg
        # A proportional gain of one per cent of the regulated speed: a
        # small speed error asks for a large duty, so the motor tracks the
        # reference closely and the ramp is what the response is made of.
        regulated_wheel_speed = (
            self.spec.regulated_speed_rad_s * self.wheel_radius_m
            / max(self.gearing, 1e-9))
        gain = 1.0 / max(0.01 * regulated_wheel_speed, 1e-6)

        v = float(initial_v)
        reference = float(initial_v)
        x = 0.0
        rows: List[Tuple[float, float, float]] = []
        steps = int(round(duration_s / dt_s))
        for index in range(steps + 1):
            t = index * dt_s
            rows.append((t, v, x))
            if index == steps:
                break
            target = command(t)
            # 1. The drive base ramps its reference.
            error = target - reference
            reference += float(np.clip(
                error, -accel_limit_m_s2 * dt_s, accel_limit_m_s2 * dt_s))
            # 2. The motor follows the reference.
            omega = self.motor_omega(v)
            omega_reference = self.motor_omega(reference)
            duty = float(np.clip(
                omega_reference / self.spec.no_load_speed_rad_s
                + gain * (reference - v), -1.0, 1.0))
            torque = motor_torque(self.spec, duty, omega)
            direction = 1.0 if v >= 0.0 else -1.0
            torque -= gearbox_drag(self.spec, omega) * direction
            force = torque * self.gearing / self.wheel_radius_m
            # 3. The tyres are the limit: past it the wheel spins and the
            # robot stays where it is.
            force = float(np.clip(force, -self.traction_limit(),
                                  self.traction_limit()))
            acceleration = force / self.mass_kg
            v += acceleration * dt_s
            x += v * dt_s
        return rows

    # -- the summary a human reads -------------------------------------------

    def summary(self) -> dict:
        """A JSON-safe description of the drivetrain."""
        return {
            "motor": self.spec.name,
            "source": self.spec.source,
            "stall_torque_nm": self.spec.stall_torque_nm,
            "no_load_speed_rad_s": self.spec.no_load_speed_rad_s,
            "regulated_speed_rad_s": self.spec.regulated_speed_rad_s,
            "gearbox_coulomb_nm": self.spec.gearbox_coulomb_nm,
            "gearbox_viscous_nm_s": self.spec.gearbox_viscous_nm_s,
            "wheel_radius_m": self.wheel_radius_m,
            "gearing": self.gearing,
            "traction_limit_n": self.traction_limit(),
            "drive_force_at_zero_n": self.drive_force(0.0),
            "brake_decel_at_1ms_nm": self.brake_deceleration(1.0),
            "coast_decel_at_1ms_nm": self.coast_deceleration(1.0),
        }


# ---------------------------------------------------------------------------
# Checking a declared config against the literature
# ---------------------------------------------------------------------------

def compare_declared_motor(
    stall_torque_nm: float,
    no_load_speed_rad_s: float,
    wheel_radius_m: float,
    gearing: float = 1.0,
    candidates: Tuple[MotorSpec, ...] = SPIKE_PRIME_MOTORS,
) -> Dict[str, Any]:
    """Compare a declared motor against the published SPIKE Prime motors.

    This is a sanity check, not a calibration: it says "the numbers in the
    config are in the neighbourhood of a real LEGO motor" and, when they are
    not, which motor the numbers are closest to. A robot that was measured
    does not need this, but a config written from memory does.

    Returns:
        A dict with the closest published motor, how far off the declared
        numbers are, and a note for a human. ``suspicious`` is true when the
        numbers are too far from every published motor to be one of them.
    """
    if stall_torque_nm <= 0.0 or no_load_speed_rad_s <= 0.0:
        return {
            "suspicious": True,
            "note": "the declared motor has a non-positive torque or speed",
        }

    wheel_speed = no_load_speed_rad_s * wheel_radius_m / max(gearing, 1e-9)
    best: Optional[Tuple[float, MotorSpec]] = None
    for spec in candidates:
        # Compare in wheel terms, so a geared robot is compared with the same
        # gearing applied to the published motor.
        published_wheel_speed = (
            spec.no_load_speed_rad_s * wheel_radius_m / max(gearing, 1e-9))
        torque_ratio = stall_torque_nm / spec.stall_torque_nm
        speed_ratio = (wheel_speed / published_wheel_speed
                       if published_wheel_speed else 0.0)
        distance = abs(np.log(max(torque_ratio, 1e-9))) + abs(
            np.log(max(speed_ratio, 1e-9)))
        if best is None or distance < best[0]:
            best = (distance, spec)

    if best is None:
        return {"suspicious": False, "note": ""}
    distance, spec = best
    published_wheel_speed = (
        spec.no_load_speed_rad_s * wheel_radius_m / max(gearing, 1e-9))
    torque_ratio = stall_torque_nm / spec.stall_torque_nm
    speed_ratio = wheel_speed / published_wheel_speed if published_wheel_speed else 0.0

    if distance > 1.0:
        return {
            "suspicious": True,
            "closest": spec.name,
            "torque_ratio": torque_ratio,
            "speed_ratio": speed_ratio,
            "note": (
                "the declared motor (stall {t:.3f} Nm, no-load {w:.1f} rad/s) "
                "is further than a factor of {f:.1f} from every published SPIKE "
                "Prime motor, so it is probably a guess".format(
                    t=stall_torque_nm, w=no_load_speed_rad_s,
                    f=float(np.exp(distance)))),
        }

    return {
        "suspicious": False,
        "closest": spec.name,
        "torque_ratio": torque_ratio,
        "speed_ratio": speed_ratio,
        "note": (
            "the declared motor is closest to the {n}: stall {t:.2f}x, no-load "
            "speed {s:.2f}x the published value ({src})".format(
                n=spec.name, t=torque_ratio, s=speed_ratio, src=spec.source)),
    }
