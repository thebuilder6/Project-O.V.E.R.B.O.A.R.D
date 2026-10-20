"""RQ-A experiment 2: is the tracking error a reference/kinematics artefact?

The validator compares the interval-tube midpoint against the PLAN's next
sample position (immrax_validator.py:700). But the plant follows the
validator's LINEAR reference, while the plan was built by trapezoidal
integration. If a robot that tracked the linear reference *exactly* already
ends up away from the plan, then the "tracking error" measures the mismatch
between the reference and the plan, not the robot's tracking ability.

This computes that mismatch in pure NumPy, per sample, and compares it with
the tracking-error series the validator reports.
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

TMP = tempfile.mkdtemp(prefix="rqa2_")


def solve(mission, per_segment, weight, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", str(weight)]
    if per_segment:
        cmd.append("--per-segment-dt")
    subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True, check=True)
    with open(out, encoding="utf-8") as fh:
        return json.load(fh)


def analyse(samples, label):
    n = len(samples)
    x = np.array([s["x"] for s in samples])
    y = np.array([s["y"] for s in samples])
    th = np.array([s["heading"] for s in samples])
    vl = np.array([s["vl"] for s in samples])
    vr = np.array([s["vr"] for s in samples])
    t = np.array([s["t"] for s in samples])
    track = 0.0965
    v = (vl + vr) / 2.0
    om = (vr - vl) / track

    # The plan's own kinematics: trapezoidal, as in optimizer.py.
    plan_x = x.copy()
    plan_y = y.copy()
    plan_th = th.copy()
    dt = np.diff(t)
    plan_x[1:] = x[:-1] + 0.5 * (v[:-1] * np.cos(th[:-1]) + v[1:] * np.cos(th[1:])) * dt
    plan_y[1:] = y[:-1] + 0.5 * (v[:-1] * np.sin(th[:-1]) + v[1:] * np.sin(th[1:])) * dt
    plan_th[1:] = th[:-1] + 0.5 * (om[:-1] + om[1:]) * dt

    # A robot that follows the validator's reference EXACTLY:
    #  - v and omega are linearly interpolated between samples (the reference),
    #  - theta is the integral of that omega,
    #  - x,y are the integral of v*cos(theta), v*sin(theta).
    # Integrate finely.
    SUB = 40
    rx = np.zeros(n)
    ry = np.zeros(n)
    rth = np.zeros(n)
    rx[0], ry[0], rth[0] = x[0], y[0], th[0]
    per_step = []
    for k in range(n - 1):
        d = dt[k]
        u = (np.arange(1, SUB + 1) / SUB)
        tt = t[k] + u * d
        v_int = v[k] + (v[k + 1] - v[k]) * u
        om_int = om[k] + (om[k + 1] - om[k]) * u
        # theta(t) from integrating the linearly interpolated omega
        th_int = th[k] + (om[k] * u * d + 0.5 * (om[k + 1] - om[k]) / d * (u * d) ** 2)
        h = d / SUB
        vcos = v_int * np.cos(th_int)
        vsin = v_int * np.sin(th_int)
        rx[k + 1] = rx[k] + h * np.sum(vcos)
        ry[k + 1] = ry[k] + h * np.sum(vsin)
        rth[k + 1] = th[k] + h * np.sum(om_int)
        # per-step displacement mismatch between the two integration rules
        plan_step = np.array([plan_x[k + 1] - plan_x[k], plan_y[k + 1] - plan_y[k]])
        ref_step = np.array([rx[k + 1] - rx[k], ry[k + 1] - ry[k]])
        per_step.append(float(np.hypot(*(ref_step - plan_step))))

    # mismatch accumulated from the start
    dx = rx - plan_x
    dy = ry - plan_y
    accum = np.hypot(dx, dy)
    heading_err = np.abs(np.arctan2(np.sin(rth - plan_th), np.cos(rth - plan_th)))

    # curvature per interval (heading change per unit length)
    dth = np.abs(np.diff(th))
    seg_len = np.hypot(np.diff(x), np.diff(y))
    with np.errstate(divide="ignore", invalid="ignore"):
        curv = np.where(seg_len > 1e-9, dth / np.maximum(seg_len, 1e-9), 0.0)

    print(f"\n--- {label} ---")
    print(f"  samples={n}  path length={seg_len.sum():.3f} m  "
          f"total time={t[-1]-t[0]:.3f} s")
    print(f"  mean sample spacing = {seg_len.mean()*100:.2f} cm, "
          f"mean |dtheta| = {dth.mean():.4f} rad")
    print(f"  reference-vs-plan accumulated mismatch at the LAST sample: "
          f"{accum[-1]*100:.3f} cm   (heading {math.degrees(heading_err[-1]):.3f} deg)")
    print(f"  max accumulated mismatch: {accum.max()*100:.3f} cm at sample "
          f"{int(np.argmax(accum))}")
    print(f"  per-step mismatch: mean {np.mean(per_step)*100:.4f} cm, "
          f"max {np.max(per_step)*100:.4f} cm")
    print(f"  curvature: mean {curv.mean():.3f} rad/m, max {curv.max():.3f} rad/m")
    print(f"  predicted drift per step ~ v*dtheta^2*dt/12: "
          f"{np.mean([v[k]*dth[k]**2*dt[k]/12 for k in range(n-1)])*100:.4f} cm")

    step = max(1, n // 16)
    print("  accumulated mismatch (cm) at samples "
          f"{list(range(0, n, step))}:")
    print("    " + " ".join(f"{accum[i]*100:6.3f}" for i in range(0, n, step)))
    return accum


def main():
    from immrax_validator import ImmraxValidator
    from jax_robot_model import JAXRobotConfig
    from robot_model import RobotConfig

    with open(os.path.join(PINNED, "examples", "fll_choreo.json"), encoding="utf-8") as fh:
        cfg_data = json.load(fh)
    robot_cfg = RobotConfig(cfg_data)
    jax_cfg = JAXRobotConfig(robot_cfg)

    cases = [
        ("example_s_curve", True, 0.0),
        ("example_with_stops", True, 0.0),
        ("example_complex_mission", False, 0.0),
    ]

    for mission, per_seg, weight in cases:
        out = os.path.join(TMP, f"{mission}.traj")
        traj = solve(mission, per_seg, weight, out)
        samples = traj["trajectory"]["samples"]
        accum = analyse(samples, mission)

        validator = ImmraxValidator(jax_cfg)
        rep = validator.validate_trajectory(
            samples, (1.24, 1.51), (0.75, 0.9), (0.0004, 0.0005),
            (0.0964, 0.0966), (0.0275, 0.0285), (0.0008, 0.0011))
        env = rep["reachability"]["envelope"]
        reported = []
        for i in range(1, len(samples)):
            e = env[i] if i < len(env) else None
            if not e or e.get("unknown"):
                reported.append(None)
                continue
            mx = (e["x_min"] + e["x_max"]) / 2.0
            my = (e["y_min"] + e["y_max"]) / 2.0
            reported.append(math.hypot(mx - samples[i]["x"], my - samples[i]["y"]))
        clean = [v for v in reported if v is not None]
        print(f"  validator reported max tracking error = "
              f"{rep['max_tracking_error_m']*100:.3f} cm "
              f"(max in series {max(clean)*100:.3f} cm)")
        # compare the two series at the end
        print(f"  at the last sample: validator {reported[-1]*100:.3f} cm vs "
              f"reference-mismatch {accum[-1]*100:.3f} cm")


if __name__ == "__main__":
    main()
