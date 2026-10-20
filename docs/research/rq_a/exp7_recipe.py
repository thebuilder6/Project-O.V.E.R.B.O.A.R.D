"""RQ-A experiment 7: the recipe.

exp6 showed that placing the samples where the (v, omega) profile actually
takes them removes 61 % of the S-curve tracking error. The remaining
discretisation error is the chord-vs-arc mismatch, which falls with the
sampling density.

So: exact positions + enough samples should reach the ~0.16 cm floor that
the synthetic self-consistent arc reaches. This measures the recipe and its
cost.
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

TMP = tempfile.mkdtemp(prefix="rqa7_")


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

    cases = [
        ("example_s_curve", True),
        ("example_with_stops", True),
        ("example_complex_mission", False),
    ]
    ns = [10, 20, 40, 80]

    for mission, per_seg in cases:
        print(f"\n{'='*100}")
        print(f"{mission}")
        print("=" * 100)
        print(f"{'n':>4} {'solve_s':>8} {'dur_s':>7} | "
              f"{'as_built_cm':>12} {'gate':>5} | {'exact_cm':>9} {'gate':>5} | "
              f"{'improvement':>11}")
        for n in ns:
            out = os.path.join(TMP, f"{mission}_{n}.traj")
            try:
                t0 = time.time()
                traj = solve(mission, per_seg, n, out)
                wall = time.time() - t0
            except RuntimeError as exc:
                print(f"{n:>4}  FAILED {str(exc)[:100]}")
                continue
            samples = traj["trajectory"]["samples"]
            r1 = validate(samples)
            r2 = validate(rebuild_exact(samples))
            d1 = r1["max_tracking_error_m"] * 100
            d2 = r2["max_tracking_error_m"] * 100
            dur = samples[-1]["t"] - samples[0]["t"]
            imp = (d1 - d2) / d1 * 100 if d1 else 0.0
            print(f"{n:>4} {wall:>8.1f} {dur:>7.3f} | {d1:>12.3f} "
                  f"{'PASS' if r1['passed'] else 'fail':>5} | {d2:>9.3f} "
                  f"{'PASS' if r2['passed'] else 'fail':>5} | {imp:>10.1f} %")


if __name__ == "__main__":
    main()
