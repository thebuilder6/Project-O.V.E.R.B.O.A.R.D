"""RQ-A summary table: the numbers for the research notes.

For each example mission:
  - the plan's position vs. where its own (v, omega) profile actually takes
    it (the plan/plant inconsistency), with the stops excluded so the
    heading metric is not confounded by the direction reversals;
  - the resulting closed-loop tracking error.
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

TMP = tempfile.mkdtemp(prefix="rqasum_")


def solve(mission, per_segment, n, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", "0.0", "-n", str(n)]
    if per_segment:
        cmd.append("--per-segment-dt")
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission}: {r.stdout[-1000:]}")
    return json.load(open(out, encoding="utf-8"))


def rebuild_exact(samples, track=0.0965, sub=60):
    v = (np.array([s["vl"] for s in samples]) + np.array([s["vr"] for s in samples])) / 2.0
    om = (np.array([s["vr"] for s in samples]) - np.array([s["vl"] for s in samples])) / track
    t = np.array([s["t"] for s in samples])
    px, py, pth = samples[0]["x"], samples[0]["y"], samples[0]["heading"]
    out = []
    for k in range(len(samples) - 1):
        d = t[k + 1] - t[k]
        u = (np.arange(1, sub + 1) / sub)
        v_i = v[k] + (v[k + 1] - v[k]) * u
        om_i = om[k] + (om[k + 1] - om[k]) * u
        th_i = pth + om[k] * u * d + 0.5 * (om[k + 1] - om[k]) / d * (u * d) ** 2
        h = d / sub
        px += h * np.sum(v_i * np.cos(th_i))
        py += h * np.sum(v_i * np.sin(th_i))
        pth += h * np.sum(om_i)
        s = dict(samples[k + 1])
        s["x"], s["y"], s["heading"] = float(px), float(py), float(pth)
        out.append(s)
    return [samples[0]] + out


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

    cases = [("example_straight", True), ("example_s_curve", True),
             ("example_with_stops", True), ("example_complex_mission", False),
             ("example_with_events", True)]

    print(f"{'mission':<26} {'n':>3} {'samples':>8} {'dtheta/smp':>11} "
          f"{'plan_vs_profile_cm':>19} {'track_cm':>9} {'gate':>6}")
    for mission, per_seg in cases:
        out = os.path.join(TMP, f"{mission}.traj")
        samples = solve(mission, per_seg, 10, out)["trajectory"]["samples"]
        exact = rebuild_exact(samples)
        dev = np.hypot([exact[k]["x"] - samples[k]["x"] for k in range(len(samples))],
                       [exact[k]["y"] - samples[k]["y"] for k in range(len(samples))])
        dth = np.abs(np.diff([s["heading"] for s in samples]))
        rep = validate(samples)
        print(f"{mission:<26} {10:>3} {len(samples):>8} {dth.mean():>11.4f} "
              f"{dev.max()*100:>19.3f} {rep['max_tracking_error_m']*100:>9.3f} "
              f"{'PASS' if rep['passed'] else 'fail':>6}")


if __name__ == "__main__":
    main()
