"""Compare the drivetrain model with a third-party simulator's trace.

The trace comes from ``research/crosscheck/sim_step.ts``, which drives a
wheel-speed step profile in an independent rigid-body simulator and writes
what happened. This script reads that trace and asks the same question of
OVERBOARD's own drivetrain model: how fast does the robot run, and how far
does it travel after the command is dropped?

The two models are not the same model. The reference simulator is a rigid
body with a motor model, a tyre model, a caster and contact friction; ours
is an analytic motor and drivetrain. The point is not to agree to the last
decimal. The point is to see whether the *structure* of ours holds up: does
the motor's torque fall with speed the way the torque-speed line says, and
does the robot stop where the motor says it stops?

The reference simulator's drive base is used for the comparison, because the
question is about the model and not about a particular robot. Its numbers
come from its own published drive base (56 mm wheels, 120 mm track, medium
SPIKE motors, a rubber tyre on a mat with a friction of 1.0 and the multiply
combine rule).

Usage:
    python research/crosscheck/compare_step.py [trace.csv] [coast_trace.csv]
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from overboard.models.drivetrain import DrivetrainModel, SPIKE_PRIME_MEDIUM

# The reference simulator's drive base, as its own source declares it.
REFERENCE_WHEEL_RADIUS_M = 0.028
REFERENCE_MASS_KG = 0.598
REFERENCE_COF = 0.9  # a rubber tyre (0.9) on a mat with a friction of 1.0
REFERENCE_RUN_MS = 600.0


def _load(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _speed(run_ms):
    """The measured run: the steady speed and what happened after the stop."""
    rows = run_ms
    drive = [r for r in rows if float(r["t_ms"]) <= REFERENCE_RUN_MS]
    after = [r for r in rows if float(r["t_ms"]) > REFERENCE_RUN_MS]

    # The left motor is mirrored in the reference drive base, so its speed
    # has the opposite sign. The right motor's is the plain one.
    settled = [r for r in drive if float(r["t_ms"]) >= REFERENCE_RUN_MS - 200]
    dps = sum(abs(float(r["right_dps"])) for r in settled) / len(settled)
    omega = dps * math.pi / 180.0
    speed = omega * REFERENCE_WHEEL_RADIUS_M

    y_at_stop = float(drive[-1]["y_mm"])
    y_final = float(after[-1]["y_mm"])
    distance = (y_final - y_at_stop) / 1000.0
    deceleration = (speed ** 2 / (2.0 * distance)) if distance > 1e-9 else 0.0
    return {
        "commanded_dps": float(drive[0]["cmd_dps"]),
        "steady_dps": dps,
        "steady_speed_m_s": speed,
        "distance_after_stop_m": distance,
        "deceleration_m_s2": deceleration,
    }


def _model():
    return DrivetrainModel(
        spec=SPIKE_PRIME_MEDIUM,
        wheel_radius_m=REFERENCE_WHEEL_RADIUS_M,
        mass_kg=REFERENCE_MASS_KG,
        gearing=1.0,
        track_width_m=0.120,
        coefficient_of_friction=REFERENCE_COF,
    )


def _equilibrium_speed(model):
    """The speed at which the motor's pull balances the gearbox's drag.

    Above this speed the motor cannot make the robot go any faster, so this
    is the top speed of a full-throttle command.
    """
    step = 0.01
    speed = 0.0
    best = 0.0
    while speed < 2.0:
        if model.drive_force(speed) <= 0.0:
            break
        best = speed
        speed += step
    return best


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    here = os.path.dirname(os.path.abspath(__file__))
    brake_path = argv[0] if len(argv) > 0 else os.path.join(here, "trace.csv")
    coast_path = argv[1] if len(argv) > 1 else os.path.join(
        here, "trace_coast.csv")

    model = _model()
    equilibrium = _equilibrium_speed(model)

    print("Reference simulator: a rigid body with a motor model, a tyre "
          "model and a caster.")
    print(f"OVERBOARD: the analytic drivetrain, motor {model.spec.name}, "
          f"wheel {model.wheel_radius_m * 1000:.0f} mm, "
          f"mass {model.mass_kg:.3f} kg, grip {model.coefficient_of_friction}")
    summary = model.summary()
    print(f"  the model: stall {summary['stall_torque_nm']:.3f} Nm, no-load "
          f"{summary['no_load_speed_rad_s']:.2f} rad/s, traction limit "
          f"{summary['traction_limit_n']:.2f} N, brake at 1 m/s "
          f"{summary['brake_decel_at_1ms_nm']:.2f} m/s^2, coast at 1 m/s "
          f"{summary['coast_decel_at_1ms_nm']:.2f} m/s^2")
    print()

    print("How fast does it run?")
    if os.path.exists(brake_path):
        run = _speed(_load(brake_path))
        measured = run["steady_speed_m_s"]
        print(f"  the reference simulator reached {measured:.4f} m/s "
              f"({run['steady_dps']:.0f} deg/s at the wheel) on a "
              f"{run['commanded_dps']:.0f} deg/s command")
        print(f"  the model puts the top speed at {equilibrium:.4f} m/s "
              f"({equilibrium / model.wheel_radius_m * 180 / math.pi:.0f} "
              "deg/s)")
        error = abs(measured - equilibrium) / max(equilibrium, 1e-9) * 100.0
        print(f"  difference: {error:.0f}%")
        print("  the model has no rolling resistance, so it expects the "
              "robot to hold a little more speed than the simulator does.")
    print()

    print("How far does it travel after the command is dropped?")
    for label, path in (("coast", coast_path), ("brake", brake_path)):
        if not os.path.exists(path):
            continue
        run = _speed(_load(path))
        speed = run["steady_speed_m_s"]
        if label == "coast":
            predicted = model.coast_deceleration(speed)
            note = "the gearbox's friction only"
        else:
            predicted = model.brake_deceleration(speed)
            note = "the motor braking against its own back EMF"
        print(f"  {label}: the simulator travelled "
              f"{run['distance_after_stop_m'] * 1000:.1f} mm "
              f"({run['deceleration_m_s2']:.2f} m/s^2) after the command, "
              f"from {speed:.3f} m/s")
        print(f"         the model says {predicted:.2f} m/s^2 ({note})")
    print()
    print("Read the brake row with care: the reference simulator's brake "
          "mode reverses the wheels instead of slowing them, so its braking "
          "distance is not a usable reference. The coast row is the one to "
          "compare, and the gap between it and the model is the rolling "
          "resistance the model does not have.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
