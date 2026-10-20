"""RQ-A: what does the digital twin's actuator-gain band do to the tracking gate?

The gain is the 7th entry of the uncertainty vector and it multiplies the
commanded wheel speeds, so a gain band of +/-1 % is a speed error of +/-1 %.
Over a 1 m path that is 10 mm of position error - twice the 5 mm tracking
gate.

The studio now keeps a calibrated model with the robot and passes --sysid-fit
on every solve, so this is not hypothetical: if the fitted gain band is wide,
the tracking gate becomes unpassable for reasons that have nothing to do with
the planner.
"""
import json
import os
import sys

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)

import immrax_validator as IV  # noqa: E402
from jax_robot_model import JAXRobotConfig  # noqa: E402
from robot_model import RobotConfig  # noqa: E402

CFG = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                     encoding="utf-8"))
JAX_CFG = JAXRobotConfig(RobotConfig(CFG))


def straight(length=1.0, v=0.5, n=10):
    out = []
    dt = (length / v) / n
    for k in range(n + 1):
        out.append({
            "t": k * dt, "x": v * k * dt, "y": 0.0, "heading": 0.0,
            "vl": v, "vr": v, "omega": 0.0,
            "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
        })
    return out


def validate(samples, gain=(1.0, 1.0)):
    val = IV.ImmraxValidator(JAX_CFG)
    val.digital_twin_gain_range = gain
    return val.validate_trajectory(
        samples, (1.24, 1.51), (0.75, 0.9), (0.0004, 0.0005),
        (0.0964, 0.0966), (0.0275, 0.0285), (0.0008, 0.0011))


print("tracking gate = 5.00 mm\n")
print(f"{'path':>8} {'gain band':>18} {'tracking mm':>13} {'envelope mm':>13} "
      f"{'passes gate':>12}")
for length in (0.5, 1.0, 2.0):
    for band, label in (((1.0, 1.0), "none (1.000)"),
                        ((0.999, 1.001), "+/-0.1 %"),
                        ((0.995, 1.005), "+/-0.5 %"),
                        ((0.99, 1.01), "+/-1 %"),
                        ((0.97, 1.03), "+/-3 %"),
                        ((0.95, 1.05), "+/-5 %")):
        rep = validate(straight(length=length), band)
        trk = rep["max_tracking_error_m"] * 1000
        env = rep["max_envelope_radius_m"] * 1000
        print(f"{length:>7.1f}m {label:>18} {trk:>13.3f} {env:>13.3f} "
              f"{'yes' if trk <= 5.0 else 'NO':>12}")

print("\nA gain band of g means the robot travels (1-g) of the distance it was")
print("told to travel, so the position error over a path of length L is about")
print("g * L. The tracking gate of 5 mm is therefore reached at g = 5 mm / L:")
for length in (0.5, 1.0, 2.0):
    print(f"  L = {length:.1f} m -> a gain band of +/-{5.0/length:.2f} % "
          f"already fails the gate")
