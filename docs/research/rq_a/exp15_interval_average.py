"""RQ-A experiment 15: the complete reference fix.

Two independent lines of work landed on the same place:

  * the interval-average VELOCITY (from the hold analysis): the plan's
    trapezoidal rule says the distance covered in one interval is the
    interval average speed times the interval, and the controller's
    20 ms hold makes the robot cover the mean of a staircase instead;

  * the interval-average TURN RATE (from the reference consistency
    analysis): if omega is a line, the heading is a parabola, and the two
    interpolated independently disagree by up to 9 degrees mid-interval.

Both say: the reference should report the interval AVERAGES, not
interpolated instantaneous values. The heading target stays interpolated
(it is a target, and the robot really does sweep through it) and the
position stays interpolated (the plan's samples are the chord's ends).

If the robot is told to move at the interval-average velocity along the
chord, it traces exactly the path the plan assumed - because the plan's
trapezoidal rule IS "move at the interval average along the chord".

The turn rate half of this is already in the tree (KI-027): the shipped
reference takes the turn rate as the derivative of the interpolated
heading, which is constant over the interval, and recomputes the gain
from it. So the variants below are measured ON TOP of what ships:

  A    as shipped: the turn rate from the heading's slope, the gain
       recomputed, the velocity interpolated
  V    interval-average velocity (and the gain recomputed from it)
  W    the turn rate taken as the plan's own average omega instead of the
       heading's slope (and the gain recomputed from it)
  VW   both

Before KI-027 the baseline was the legacy reference (the turn rate as a
line of its own, the gain interpolated), and the turn rate half of the
fix was worth 2-3.6x. It is in the tree now, so what is left to measure
is the velocity half.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)

TMP = tempfile.mkdtemp(prefix="rqa15_")


def solve(mission, n, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", "0.0", "-n", str(n), "--per-segment-dt"]
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission} n={n}: {r.stdout[-800:]}")
    return json.load(open(out, encoding="utf-8"))


def make(mode):
    """The reference of the variant, built on the shipped one.

    The shipped reference (KI-027) takes the turn rate as the derivative of
    the interpolated heading and recomputes the gain from the reported v and
    omega. The variants change the velocity, the turn rate or both, and the
    gain always follows whatever they report.
    """
    def ref(s0, s1, dt, track_width, s, ramsete_b=2.0):
        x0, y0, th0 = s0["x"], s0["y"], s0["heading"]
        th1 = s1["heading"]
        v0 = (s0["vl"] + s0["vr"]) / 2.0
        v1 = (s1["vl"] + s1["vr"]) / 2.0
        om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
        om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)

        # The position and the heading are targets: they are interpolated.
        ref_x = x0 + (s1["x"] - x0) * s
        ref_y = y0 + (s1["y"] - y0) * s
        ref_heading = th0 + (th1 - th0) * s

        if mode in ("V", "VW"):
            ref_v = 0.5 * (v0 + v1)              # the interval average
        else:
            ref_v = v0 + (v1 - v0) * s
        if mode in ("W", "VW"):
            ref_om = 0.5 * (om0 + om1)           # the plan's average omega
        else:
            ref_om = (th1 - th0) / dt if dt > 0 else om0   # the heading's slope

        # The gain follows whatever the variant reports.
        kr = math.sqrt(ref_om ** 2 + ramsete_b * ref_v ** 2)
        return ref_x, ref_y, ref_heading, ref_v, ref_om, kr
    return ref


def main():
    import immrax_validator as IV
    from jax_robot_model import JAXRobotConfig
    from robot_model import RobotConfig

    cfg = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                         encoding="utf-8"))
    jax_cfg = JAXRobotConfig(RobotConfig(cfg))
    R = dict(cof=(1.24, 1.51), torque_headroom=(0.75, 0.9),
             gear_backlash_m=(0.0004, 0.0005),
             track_width_range=(0.0964, 0.0966),
             wheel_radius_range=(0.0275, 0.0285),
             inertia_kg_m2=(0.0008, 0.0011))

    def validate(samples):
        val = IV.ImmraxValidator(jax_cfg)
        return val.validate_trajectory(
            samples, R["cof"], R["torque_headroom"], R["gear_backlash_m"],
            R["track_width_range"], R["wheel_radius_range"], R["inertia_kg_m2"])

    orig = IV._interpolated_reference
    labels = (("A  as shipped", "A"),
              ("V  interval-average velocity", "V"),
              ("W  interval-average turn rate", "W"),
              ("VW both, gain recomputed", "VW"))
    for mission in ("example_s_curve", "example_with_stops", "example_complex_mission"):
        print(f"\n{'='*94}\n{mission}\n{'='*94}")
        for n in (10, 20, 40):
            out = os.path.join(TMP, f"{mission}_{n}.traj")
            try:
                traj = solve(mission, n, out)
            except RuntimeError:
                print(f"  n={n}  (solve failed)")
                continue
            samples = traj["trajectory"]["samples"]
            print(f"  n={n}")
            base = None
            for label, mode in labels:
                IV._interpolated_reference = make(mode)
                rep = validate(samples)
                trk = rep["max_tracking_error_m"] * 100
                if base is None:
                    base = trk
                print(f"    {label:<30} {trk:>8.4f} cm  "
                      f"{'PASS' if rep['passed'] else 'fail':>4}  {trk/base:>5.2f}x")
            IV._interpolated_reference = orig


if __name__ == "__main__":
    main()
