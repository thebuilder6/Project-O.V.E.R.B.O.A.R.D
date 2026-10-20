"""RQ-A experiment 3: is the tracking error a sampling-density problem?

The plan is built by trapezoidal integration with 10 samples per segment, so
the heading changes by ~19 degrees per sample on the S-curve. The deployed
controller follows a linearly interpolated (chorded) reference. The mismatch
between "trapezoid of v*cos(theta)" and "the arc a robot actually traces" is
second order in the heading change per sample, i.e. it should fall off fast
as the sampling gets denser.

If the tracking error collapses as -n goes up, the gate failures are a
sampling problem, and the fix is more samples (or a reference that matches
the plan's own integration), not a different seed or a better plant.
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

TMP = tempfile.mkdtemp(prefix="rqa3_")


def solve(mission, per_segment, weight, n, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", str(weight), "-n", str(n)]
    if per_segment:
        cmd.append("--per-segment-dt")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission} n={n}: {r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return json.load(open(out, encoding="utf-8")), time.time() - t0


def reference_mismatch(samples, track=0.0965):
    """How far a robot that follows the linear reference EXACTLY ends up
    from the plan, accumulated over the path."""
    n = len(samples)
    x = np.array([s["x"] for s in samples])
    y = np.array([s["y"] for s in samples])
    th = np.array([s["heading"] for s in samples])
    vl = np.array([s["vl"] for s in samples])
    vr = np.array([s["vr"] for s in samples])
    t = np.array([s["t"] for s in samples])
    v = (vl + vr) / 2.0
    om = (vr - vl) / track
    dt = np.diff(t)
    plan_x = x.copy()
    plan_y = y.copy()
    plan_x[1:] = x[:-1] + 0.5 * (v[:-1] * np.cos(th[:-1]) + v[1:] * np.cos(th[1:])) * dt
    plan_y[1:] = y[:-1] + 0.5 * (v[:-1] * np.sin(th[:-1]) + v[1:] * np.sin(th[1:])) * dt
    SUB = 40
    rx = np.zeros(n)
    ry = np.zeros(n)
    rx[0], ry[0] = x[0], y[0]
    for k in range(n - 1):
        d = dt[k]
        u = np.arange(1, SUB + 1) / SUB
        v_int = v[k] + (v[k + 1] - v[k]) * u
        om_int = om[k] + (om[k + 1] - om[k]) * u
        th_int = th[k] + om[k] * u * d + 0.5 * (om[k + 1] - om[k]) / d * (u * d) ** 2
        h = d / SUB
        rx[k + 1] = rx[k] + h * np.sum(v_int * np.cos(th_int))
        ry[k + 1] = ry[k] + h * np.sum(v_int * np.sin(th_int))
    return np.hypot(rx - plan_x, ry - plan_y)


def main():
    from immrax_validator import ImmraxValidator
    from jax_robot_model import JAXRobotConfig
    from robot_model import RobotConfig

    cfg_data = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                              encoding="utf-8"))
    robot_cfg = RobotConfig(cfg_data)
    jax_cfg = JAXRobotConfig(robot_cfg)

    cases = [
        ("example_s_curve", True),
        ("example_with_stops", True),
        ("example_complex_mission", False),
    ]
    ns = [10, 20, 40, 80]

    for mission, per_seg in cases:
        print(f"\n{'='*96}")
        print(f"{mission}   per_segment={per_seg}")
        print("=" * 96)
        print(f"{'n':>4} {'samples':>8} {'solve_s':>8} {'dur_s':>7} {'track_cm':>10} "
              f"{'env_cm':>8} {'mismatch_cm':>12} {'dtheta/sample':>14} {'gate':>10}")
        for n in ns:
            out = os.path.join(TMP, f"{mission}_{n}.traj")
            try:
                traj, wall = solve(mission, per_seg, 0.0, n, out)
            except RuntimeError as exc:
                print(f"{n:>4}  FAILED: {str(exc)[:120]}")
                continue
            samples = traj["trajectory"]["samples"]
            validator = ImmraxValidator(jax_cfg)
            rep = validator.validate_trajectory(
                samples, (1.24, 1.51), (0.75, 0.9), (0.0004, 0.0005),
                (0.0964, 0.0966), (0.0275, 0.0285), (0.0008, 0.0011))
            mism = reference_mismatch(samples)
            th = np.array([s["heading"] for s in samples])
            dth = np.abs(np.diff(th)).mean()
            dur = samples[-1]["t"] - samples[0]["t"]
            gate = "PASS" if rep["passed"] else "fail"
            print(f"{n:>4} {len(samples):>8} {wall:>8.1f} {dur:>7.3f} "
                  f"{rep['max_tracking_error_m']*100:>10.3f} "
                  f"{rep['reachability']['max_envelope_radius_m']*100:>8.3f} "
                  f"{mism.max()*100:>12.3f} {dth:>14.4f} {gate:>10}")


if __name__ == "__main__":
    main()
