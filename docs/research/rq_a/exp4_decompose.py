"""RQ-A experiment 4: decompose the residual tracking error.

Runs in the PINNED tree. Toggles one modelling choice at a time:

  a) shipped            - as the validator is today
  b) no quantisation    - the firmware's mm/s and deg/s rounding removed
  c) no backlash        - the plant's backlash offset removed
  d) neither            - both removed
  e) d + spline ref     - the reference becomes a Hermite curve THROUGH the
                          plan's samples, so a robot that follows it exactly
                          lands exactly on the plan

If (e) collapses the tracking error to ~0, the residual is the mismatch
between the three inconsistent discretisations of the trajectory, not the
robot failing to track.
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

TMP = tempfile.mkdtemp(prefix="rqa4_")


def solve(mission, per_segment, n, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", "0.0", "-n", str(n)]
    if per_segment:
        cmd.append("--per-segment-dt")
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission} n={n}: {r.stdout[-1200:]}\n{r.stderr[-1200:]}")
    return json.load(open(out, encoding="utf-8"))


# ---------------------------------------------------------------- spline ref
def _hermite_reference(s0, s1, dt, track_width, s, ramsete_b=2.0):
    """A cubic Hermite through the two plan samples, using the plan's own
    heading and speed as the tangents. It passes through both samples and
    has the plan's tangent direction at each, so a robot that tracks it
    exactly lands exactly on the plan."""
    x0, y0, th0 = s0["x"], s0["y"], s0["heading"]
    x1, y1, th1 = s1["x"], s1["y"], s1["heading"]
    v0 = (s0["vl"] + s0["vr"]) / 2.0
    v1 = (s1["vl"] + s1["vr"]) / 2.0
    om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
    om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)

    m0 = np.array([v0 * math.cos(th0), v0 * math.sin(th0)]) * dt
    m1 = np.array([v1 * math.cos(th1), v1 * math.sin(th1)]) * dt

    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    px = h00 * x0 + h10 * m0[0] + h01 * x1 + h11 * m1[0]
    py = h00 * y0 + h10 * m0[1] + h01 * y1 + h11 * m1[1]

    d00 = 6 * s2 - 6 * s
    d10 = 3 * s2 - 4 * s + 1
    d01 = -6 * s2 + 6 * s
    d11 = 3 * s2 - 2 * s
    dx = d00 * x0 + d10 * m0[0] + d01 * x1 + d11 * m1[0]
    dy = d00 * y0 + d10 * m0[1] + d01 * y1 + d11 * m1[1]

    d00b = 12 * s - 6
    d10b = 6 * s - 4
    d01b = -12 * s + 6
    d11b = 6 * s - 2
    ddx = d00b * x0 + d10b * m0[0] + d01b * x1 + d11b * m1[0]
    ddy = d00b * y0 + d10b * m0[1] + d01b * y1 + d11b * m1[1]

    speed = math.hypot(dx, dy)
    ref_heading = math.atan2(dy, dx)
    ref_v = speed / dt
    denom = max(speed * dt, 1e-12)
    ref_omega = (dx * ddy - dy * ddx) / denom
    kr0 = math.sqrt(om0 ** 2 + ramsete_b * v0 ** 2)
    kr1 = math.sqrt(om1 ** 2 + ramsete_b * v1 ** 2)
    return px, py, ref_heading, ref_v, ref_omega, kr0 + (kr1 - kr0) * s


def main():
    import immrax_validator as IV
    from jax_robot_model import JAXRobotConfig
    from robot_model import RobotConfig

    cfg_data = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                              encoding="utf-8"))
    robot_cfg = RobotConfig(cfg_data)
    jax_cfg = JAXRobotConfig(robot_cfg)

    orig_ref = IV._interpolated_reference
    orig_quant = IV.quantize_drive_command

    cases = [
        ("example_s_curve", True, 10),
        ("example_s_curve", True, 40),
        ("example_with_stops", True, 10),
        ("example_complex_mission", False, 10),
    ]

    for mission, per_seg, n in cases:
        out = os.path.join(TMP, f"{mission}_{n}.traj")
        traj = solve(mission, per_seg, n, out)
        samples = traj["trajectory"]["samples"]

        print(f"\n{'='*92}")
        print(f"{mission}   n={n}   samples={len(samples)}")
        print("=" * 92)
        print(f"{'variant':<26} {'track_cm':>10} {'env_cm':>8} {'gate':>8}")

        variants = [
            ("a) shipped", False, False, False),
            ("b) no quantisation", True, False, False),
            ("c) no backlash", False, True, False),
            ("d) neither", True, True, False),
            ("e) d + spline reference", True, True, True),
        ]

        for label, no_quant, no_backlash, spline in variants:
            IV.quantize_drive_command = (lambda v, w: (v, w)) if no_quant else orig_quant
            IV._interpolated_reference = _hermite_reference if spline else orig_ref
            backlash = (0.0, 0.0) if no_backlash else (0.0004, 0.0005)

            validator = IV.ImmraxValidator(jax_cfg)
            rep = validator.validate_trajectory(
                samples, (1.24, 1.51), (0.75, 0.9), backlash,
                (0.0964, 0.0966), (0.0275, 0.0285), (0.0008, 0.0011))
            gate = "PASS" if rep["passed"] else "fail"
            print(f"{label:<26} {rep['max_tracking_error_m']*100:>10.3f} "
                  f"{rep['reachability']['max_envelope_radius_m']*100:>8.3f} {gate:>8}")

        IV._interpolated_reference = orig_ref
        IV.quantize_drive_command = orig_quant


if __name__ == "__main__":
    main()
