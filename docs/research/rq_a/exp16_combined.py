"""RQ-A experiment 16: the two fixes together, in both timing modes.

exp15 showed that reporting the interval averages (VW) is the best
reference in seven of the nine cells, but that it is *worse* than shipped at
n=10 on the S-curve and the stops mission - the coarse end, where the
optimizer's self-consistent collocation (KI-026, the --self-consistent
flag) is the lever instead.

That regression is the model compensating for the plan's own
inconsistency: with the trapezoidal plan, the old reference accidentally
hid part of the plan/profile mismatch. With a self-consistent plan there
is nothing to hide, so the reference fix should show its benefit at n=10
too.

This measures the full cross: {trapezoidal, self-consistent} x {shipped,
interval-average} reference, in both timing modes.
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

TMP = tempfile.mkdtemp(prefix="rqa16_")


def solve(mission, n, per_seg, self_consistent, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", "0.0", "-n", str(n)]
    if per_seg:
        cmd.append("--per-segment-dt")
    if self_consistent:
        cmd.append("--self-consistent")
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission} n={n} sc={self_consistent}: "
                           f"{r.stdout[-500:]} {r.stderr[-500:]}")
    return json.load(open(out, encoding="utf-8"))


def make_vw(s0, s1, dt, track_width, s, ramsete_b=2.0):
    """The interval-average reference: the feedforward reports the averages
    the plan's own trapezoidal rule assumes, the target stays interpolated."""
    x0, y0, th0 = s0["x"], s0["y"], s0["heading"]
    th1 = s1["heading"]
    v0 = (s0["vl"] + s0["vr"]) / 2.0
    v1 = (s1["vl"] + s1["vr"]) / 2.0
    ref_x = x0 + (s1["x"] - x0) * s
    ref_y = y0 + (s1["y"] - y0) * s
    ref_heading = th0 + (th1 - th0) * s
    ref_v = 0.5 * (v0 + v1)
    ref_om = (th1 - th0) / dt
    kr = math.sqrt(ref_om ** 2 + ramsete_b * ref_v ** 2)
    return ref_x, ref_y, ref_heading, ref_v, ref_om, kr


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
    for mission in ("example_s_curve", "example_with_stops", "example_complex_mission"):
        for per_seg in (False, True):
            mode = "per-segment" if per_seg else "equal-duration"
            print(f"\n{'='*92}\n{mission}   {mode}\n{'='*92}")
            for n in (10, 20, 40):
                for sc in (False, True):
                    out = os.path.join(TMP, f"{mission}_{n}_{sc}.traj")
                    try:
                        traj = solve(mission, n, per_seg, sc, out)
                    except RuntimeError as exc:
                        print(f"  n={n:<3} plan={'self-consistent' if sc else 'trapezoidal':<16}"
                              f" FAILED {str(exc)[:60]}")
                        continue
                    samples = traj["trajectory"]["samples"]
                    row = []
                    for label, ref in (("shipped", orig), ("avg", make_vw)):
                        IV._interpolated_reference = ref
                        rep = validate(samples)
                        row.append((label, rep["max_tracking_error_m"] * 100,
                                    rep["passed"]))
                    IV._interpolated_reference = orig
                    cells = "  ".join(
                        f"{lab} {v:6.3f} {'PASS' if p else 'fail':>4}" for lab, v, p in row)
                    print(f"  n={n:<3} plan="
                          f"{'self-consistent' if sc else 'trapezoidal':<16} {cells}")


if __name__ == "__main__":
    main()
