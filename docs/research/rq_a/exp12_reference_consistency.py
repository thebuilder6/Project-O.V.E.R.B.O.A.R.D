"""RQ-A experiment 12: is the reference internally inconsistent?

`_interpolated_reference` interpolates the heading and the turn rate
independently:

    ref_heading = th0 + (th1 - th0) * s        # a line
    ref_omega   = om0 + (om1 - om0) * s        # a line

But if omega is linear, the heading is its integral - a PARABOLA, not a
line. The two agree at the sample instants and disagree in between by up to
domega * dt / 8. The controller is handed both: the heading as the target
and omega as the feedforward.

On the S-curve at n=10 that is about 2.4 degrees of contradictory heading
mid-interval.

This measures the three consistent variants:
  A  as shipped          - theta linear, omega linear (inconsistent)
  B  theta from omega   - theta is the integral of the linear omega (a
                           parabola); omega stays linear. The reference then
                           describes one smooth motion.
  C  omega from theta   - omega is the constant derivative of the linear
                           theta; theta stays linear. Same idea, other way
                           round.

If one of them collapses the tracking error, the fix is in the reference
and not in the optimizer.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)

TMP = tempfile.mkdtemp(prefix="rqa12_")


def solve(mission, n, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", "0.0", "-n", str(n), "--per-segment-dt"]
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission} n={n}: {r.stdout[-1000:]}")
    return json.load(open(out, encoding="utf-8"))


def make_reference(mode):
    def ref(s0, s1, dt, track_width, s, ramsete_b=2.0):
        x0, y0, th0 = s0["x"], s0["y"], s0["heading"]
        th1 = s1["heading"]
        v0 = (s0["vl"] + s0["vr"]) / 2.0
        v1 = (s1["vl"] + s1["vr"]) / 2.0
        om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
        om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)

        if mode == "B":          # theta = integral of the linear omega
            ref_om = om0 + (om1 - om0) * s
            ref_heading = th0 + (om0 * s * dt
                                 + 0.5 * (om1 - om0) / dt * (s * dt) ** 2)
        elif mode == "C":        # omega = derivative of the linear theta
            ref_heading = th0 + (th1 - th0) * s
            ref_om = (th1 - th0) / dt
        else:                    # A: as shipped
            ref_heading = th0 + (th1 - th0) * s
            ref_om = om0 + (om1 - om0) * s

        ref_x = x0 + (s1["x"] - x0) * s
        ref_y = y0 + (s1["y"] - y0) * s
        ref_v = v0 + (v1 - v0) * s
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
    RANGES = dict(cof=(1.24, 1.51), torque_headroom=(0.75, 0.9),
                  gear_backlash_m=(0.0004, 0.0005),
                  track_width_range=(0.0964, 0.0966),
                  wheel_radius_range=(0.0275, 0.0285),
                  inertia_kg_m2=(0.0008, 0.0011))

    def validate(samples):
        val = IV.ImmraxValidator(jax_cfg)
        return val.validate_trajectory(
            samples, RANGES["cof"], RANGES["torque_headroom"],
            RANGES["gear_backlash_m"], RANGES["track_width_range"],
            RANGES["wheel_radius_range"], RANGES["inertia_kg_m2"])

    orig = IV._interpolated_reference
    for mission in ("example_s_curve", "example_with_stops"):
        for n in (10, 20, 40):
            out = os.path.join(TMP, f"{mission}_{n}.traj")
            try:
                traj = solve(mission, n, out)
            except RuntimeError as exc:
                print(f"{mission} n={n}: FAILED {str(exc)[:80]}")
                continue
            samples = traj["trajectory"]["samples"]
            dt = samples[1]["t"] - samples[0]["t"]
            # how far apart are the two interpolations at mid-interval?
            worst = 0.0
            for k in range(len(samples) - 1):
                d = samples[k + 1]["t"] - samples[k]["t"]
                s0, s1 = samples[k], samples[k + 1]
                om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / 0.0965)
                om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / 0.0965)
                lin = s0["heading"] + (s1["heading"] - s0["heading"]) * 0.5
                par = s0["heading"] + (om0 * 0.5 * d
                                        + 0.5 * (om1 - om0) / d * (0.5 * d) ** 2)
                worst = max(worst, abs((lin - par + math.pi) % (2 * math.pi) - math.pi))

            print(f"\n{mission}  n={n}  dt={dt*1000:.1f} ms")
            print(f"  mid-interval disagreement between the two interpolations: "
                  f"{math.degrees(worst):.3f} deg")
            base = None
            for label, mode in (("A  as shipped (theta line, omega line)", "A"),
                                ("B  theta = integral of omega", "B"),
                                ("C  omega = derivative of theta", "C")):
                IV._interpolated_reference = make_reference(mode)
                rep = validate(samples)
                if base is None:
                    base = rep["max_tracking_error_m"]
                print(f"  {label:<38} track {rep['max_tracking_error_m']*100:8.4f} cm  "
                      f"env {rep['reachability']['max_envelope_radius_m']*100:7.4f} cm  "
                      f"{'PASS' if rep['passed'] else 'fail':>4}  "
                      f"({rep['max_tracking_error_m']/base:.3f} x)")
            IV._interpolated_reference = orig


if __name__ == "__main__":
    main()
