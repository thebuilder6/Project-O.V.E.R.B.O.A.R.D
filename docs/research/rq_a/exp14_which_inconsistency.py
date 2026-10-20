"""RQ-A experiment 14: which of the reference's three inconsistencies matters?

The deployed controller (pybricks_code/ramsete_fast.py:87-91) and the
validator (immrax_validator.py:46-77) both interpolate four coupled
quantities independently:

  tr  the heading      -> must be the integral of wr
  wr  the turn rate    -> must be the derivative of tr
  vr  the speed
  kr  the Ramsete gain -> must be sqrt(wr^2 + b*vr^2)

Each is a line between the samples, so each pair can disagree with the
others in between the samples. This separates the three:

  A  as shipped
  D  kr recomputed from the interpolated (v, omega), everything else as is
  C  omega = derivative of the interpolated heading, kr recomputed
  F  heading = integral of the interpolated omega, kr recomputed
  G  everything consistent: omega from the heading, kr recomputed
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

TMP = tempfile.mkdtemp(prefix="rqa14_")


def solve(mission, n, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", "0.0", "-n", str(n), "--per-segment-dt"]
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission} n={n}: {r.stdout[-800:]}")
    return json.load(open(out, encoding="utf-8"))


def make(mode):
    def ref(s0, s1, dt, track_width, s, ramsete_b=2.0):
        x0, y0, th0 = s0["x"], s0["y"], s0["heading"]
        th1 = s1["heading"]
        v0 = (s0["vl"] + s0["vr"]) / 2.0
        v1 = (s1["vl"] + s1["vr"]) / 2.0
        om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
        om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)

        ref_x = x0 + (s1["x"] - x0) * s
        ref_y = y0 + (s1["y"] - y0) * s
        ref_v = v0 + (v1 - v0) * s

        if mode in ("C", "G"):        # omega from the heading
            ref_heading = th0 + (th1 - th0) * s
            ref_om = (th1 - th0) / dt
        elif mode == "F":             # heading from the omega
            ref_heading = th0 + (om0 * s * dt
                                 + 0.5 * (om1 - om0) / dt * (s * dt) ** 2)
            ref_om = om0 + (om1 - om0) * s
        else:                         # A, D: as shipped
            ref_heading = th0 + (th1 - th0) * s
            ref_om = om0 + (om1 - om0) * s

        if mode in ("D", "C", "F", "G"):
            kr = math.sqrt(ref_om ** 2 + ramsete_b * ref_v ** 2)
        else:
            kr0 = math.sqrt(om0 ** 2 + ramsete_b * v0 ** 2)
            kr1 = math.sqrt(om1 ** 2 + ramsete_b * v1 ** 2)
            kr = kr0 + (kr1 - kr0) * s
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
              ("D  kr recomputed only", "D"),
              ("F  heading from omega, kr fixed", "F"),
              ("C  omega from heading, kr fixed", "C"))
    for mission in ("example_s_curve", "example_with_stops", "example_complex_mission"):
        print(f"\n{'='*96}\n{mission}\n{'='*96}")
        for n in (10, 20, 40):
            out = os.path.join(TMP, f"{mission}_{n}.traj")
            try:
                traj = solve(mission, n, out)
            except RuntimeError:
                print(f"{n:>4}  (solve failed)")
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
                print(f"    {label:<32} {trk:>8.4f} cm  "
                      f"{'PASS' if rep['passed'] else 'fail':>4}  {trk/base:>5.2f}x")
            IV._interpolated_reference = orig


if __name__ == "__main__":
    main()
