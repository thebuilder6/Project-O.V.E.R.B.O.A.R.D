"""What the fitted plant does to the validated envelope.

The digital twin can carry four things the kinematic plant cannot: a
first-order actuator (gain, tau, delay), an acceleration limit, the robot's
own odometry belief, and the braking shortfall (the wheels turn but do not
grip while the robot brakes). This script measures what each of them does
to the reachability envelope and the tracking error of one trajectory, so a
claim about the fitted plant can be checked instead of asserted.

It reads a ``.traj`` file and runs the same Immrax check the CLI runs
(``--validate``), once per model:

    python research/brake_slip_effect.py output.traj

Measured on 2026-10-21, on the complex mission planned by the default path
(``examples/example_complex_mission.json``, 99 samples, 5.365 s):

    model                    plant                 envelope    tracking   gate
    no fit                   RobotOpenLoop           1.1577 cm   0.3645 cm  pass
    actuator fit             DigitalTwinOpenLoop   20.8347 cm   8.7660 cm  fail
    actuator + brake slip    DigitalTwinOpenLoop   21.9482 cm   9.3168 cm  fail

Two things to read from it:

- The brake shortfall only reaches the plant when the fit also carries an
  actuator time constant. The five-state plant is chosen for the actuator
  (``immrax_validator``), and it is the only plant that models the grip, so
  a slip-only fit is loaded and then not applied.
- A fit with a 50 ms actuator lag and a 2 % gain band does not pass the
  current gates on this mission: the envelope is 20.8 cm against a 2 cm
  limit. That is a property of the gates against a measured plant (KI-020),
  not a bug, and it is why the digital twin is not the default yet.

The actuator numbers in this script are illustrative, not measured. Replace
them with the fit from ``python -m overboard.sysid.fit`` to measure a real
robot.
"""

import json
import math
import sys
from pathlib import Path

# The repo root, so the script runs from a checkout without an install.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from overboard.models.jax_robot_model import JAXRobotConfig
from overboard.models.robot_model import RobotConfig
from overboard.sysid.digital_twin import DigitalTwinConfig
from overboard.verification.immrax_validator import ImmraxValidator

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "examples" / "fll_choreo.json")

# An illustrative actuator: a 50 ms lag and a 2 % gain band.
ACTUATOR = {"actuator": {"straight": {
    "calibrated": True, "gain": 1.0, "gain_min": 0.98, "gain_max": 1.02,
    "tau_s": 0.05}}}

# An illustrative braking shortfall: the robot covers 10 % less ground than
# its encoders report while it brakes.
SLIP = {"slip": {"calibrated": True, "brake_ratio_min": 0.05,
                 "brake_ratio_median": 0.10, "brake_ratio_max": 0.15}}


def _radius(report, index):
    """The reachability radius of one sample, in metres."""
    env = report["reachability"]["envelope"][index]
    return math.hypot((env["x_max"] - env["x_min"]) / 2.0,
                      (env["y_max"] - env["y_min"]) / 2.0)


def _measure(label, robot_cfg, samples, uncertainties, twin=None):
    """Validate the trajectory with one model and return its numbers."""
    if twin is None:
        validator = ImmraxValidator(JAXRobotConfig(robot_cfg),
                                     propagation="plant")
    else:
        validator = ImmraxValidator(JAXRobotConfig(robot_cfg),
                                     digital_twin=twin, propagation="plant")
    report = validator.validate_trajectory(samples, *uncertainties)
    last = len(report["reachability"]["envelope"]) - 1
    return {
        "label": label,
        "plant": type(validator.plant).__name__,
        "shortfall": twin.brake_shortfall() if twin else 0.0,
        "envelope_cm": max(_radius(report, i) for i in range(last + 1)) * 100.0,
        "tracking_cm": float(report.get("max_tracking_error_m", 0.0)) * 100.0,
        "passed": bool(report.get("passed")),
    }


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    raw = json.load(open(CONFIG_PATH, encoding="utf-8"))
    robot_cfg = RobotConfig(raw["robot"] if "robot" in raw else raw)
    samples = json.load(open(argv[1], encoding="utf-8"))["trajectory"]["samples"]
    uncertainties = robot_cfg.uncertainties
    unc = (uncertainties["cof"], uncertainties["torque_headroom"],
           uncertainties["gear_backlash_m"], uncertainties["track_width_range"],
           uncertainties["wheel_radius_range"], uncertainties["inertia_kg_m2"])

    rows = [_measure("no fit", robot_cfg, samples, unc)]
    rows.append(_measure(
        "actuator fit", robot_cfg, samples, unc,
        DigitalTwinConfig.from_dict(ACTUATOR, enabled=True)))
    rows.append(_measure(
        "actuator + brake slip", robot_cfg, samples, unc,
        DigitalTwinConfig.from_dict({**ACTUATOR, **SLIP}, enabled=True)))

    print(f"{'model':<24}{'plant':<22}{'envelope':>11}{'tracking':>11}   gate")
    for row in rows:
        print(f"{row['label']:<24}{row['plant']:<22}"
              f"{row['envelope_cm']:>9.4f} cm{row['tracking_cm']:>9.4f} cm"
              f"   {'pass' if row['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
