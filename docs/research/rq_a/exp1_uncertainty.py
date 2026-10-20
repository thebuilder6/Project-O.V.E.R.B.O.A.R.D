"""RQ-A experiment 1: which uncertainty drives the closed-loop tracking error?

For a given trajectory, re-run the Immrax validator with each uncertainty
interval collapsed to a point, and record the resulting max tracking error.
If the tracking error barely moves when one interval is collapsed, that
interval is not the driver.

Also reconstructs the per-sample tracking error series from the returned
envelope, so we can see whether the error is a transient or accumulates.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)

import numpy as np  # noqa: E402

CONFIG = os.path.join(PINNED, "examples", "fll_choreo.json")
TMP = tempfile.mkdtemp(prefix="rqa_")

# The declared ranges from examples/fll_choreo.json
RANGES = {
    "cof": (1.24, 1.51),
    "torque_headroom": (0.75, 0.9),
    "gear_backlash_m": (0.0004, 0.0005),
    "track_width_range": (0.0964, 0.0966),
    "wheel_radius_range": (0.0275, 0.0285),
    "inertia_kg_m2": (0.0008, 0.0011),
}


def solve(mission, per_segment, accuracy_weight, out):
    """Solve one trajectory with --no-validate so the gate never blocks the file."""
    cmd = [sys.executable, "main.py",
           "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json",
           "-o", out, "--no-validate",
           "-a", str(accuracy_weight)]
    if per_segment:
        cmd.append("--per-segment-dt")
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"no trajectory for {mission}: {r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    with open(out, encoding="utf-8") as fh:
        return json.load(fh)


def load_samples(traj):
    return traj["trajectory"]["samples"]


def per_sample_tracking(samples, envelope):
    """tracking error at sample i+1 = |mid(envelope[i+1]) - plan[i+1]|."""
    out = []
    for i in range(1, len(samples)):
        env = envelope[i] if i < len(envelope) else None
        if not env or env.get("unknown"):
            out.append(None)
            continue
        mid_x = (env["x_min"] + env["x_max"]) / 2.0
        mid_y = (env["y_min"] + env["y_max"]) / 2.0
        out.append(math.hypot(mid_x - samples[i]["x"], mid_y - samples[i]["y"]))
    return out


def main():
    from immrax_validator import ImmraxValidator
    from jax_robot_model import JAXRobotConfig
    from robot_model import RobotConfig

    with open(CONFIG, encoding="utf-8") as fh:
        cfg_data = json.load(fh)
    robot_cfg = RobotConfig(cfg_data)
    jax_cfg = JAXRobotConfig(robot_cfg)

    missions = [
        ("example_s_curve", True, 0.0, "FAIL"),
        ("example_with_stops", True, 0.0, "FAIL"),
        ("example_complex_mission", False, 0.0, "PASS"),
    ]

    for mission, per_seg, weight, expected in missions:
        out = os.path.join(TMP, f"{mission}.traj")
        traj = solve(mission, per_seg, weight, out)
        samples = load_samples(traj)
        print(f"\n{'='*78}")
        print(f"{mission}  per_segment={per_seg} a={weight}  expected={expected}  "
              f"samples={len(samples)}")
        print("=" * 78)

        rows = []
        configs = [("ALL UNCERTAINTY", RANGES)]
        for name in RANGES:
            configs.append((f"only {name}", {name: RANGES[name]}))
        configs.append(("ALL COLLAPSED (nominal)",
                        {k: (v[0], v[0]) for k, v in RANGES.items()}))

        for label, active in configs:
            validator = ImmraxValidator(jax_cfg)
            args = {}
            for name in RANGES:
                lo, hi = active.get(name, (RANGES[name][0], RANGES[name][0]))
                args[name] = (lo, hi)
            rep = validator.validate_trajectory(
                samples,
                args["cof"], args["torque_headroom"], args["gear_backlash_m"],
                args["track_width_range"], args["wheel_radius_range"],
                args["inertia_kg_m2"])
            series = per_sample_tracking(samples, rep["reachability"]["envelope"])
            clean = [v for v in series if v is not None]
            # where does the error reach 50% of its max?
            peak = max(clean) if clean else 0.0
            idx = next((i for i, v in enumerate(series) if v is not None and v >= 0.5 * peak), None)
            rows.append({
                "config": label,
                "tracking_m": rep["max_tracking_error_m"] * 100,
                "envelope_cm": rep["reachability"]["max_envelope_radius_m"] * 100,
                "passed": rep["passed"],
                "first_half_idx": idx,
                "series": series,
            })
            print(f"  {label:<28} tracking {rep['max_tracking_error_m']*100:7.3f} cm  "
                  f"envelope {rep['reachability']['max_envelope_radius_m']*100:7.3f} cm  "
                  f"passed={rep['passed']}  half-max at sample {idx}")

        base = rows[0]
        print("\n  -> effect of collapsing each interval alone:")
        for r in rows[1:-1]:
            delta = base["tracking_m"] - r["tracking_m"]
            pct = (delta / base["tracking_m"] * 100) if base["tracking_m"] else 0.0
            print(f"     {r['config']:<28} {r['tracking_m']:7.3f} cm  "
                  f"({delta:+7.3f} cm, {pct:+6.1f} % of the all-uncertainty error)")
        nom = rows[-1]
        print(f"     {'ALL COLLAPSED':<28} {nom['tracking_m']:7.3f} cm  "
              f"({base['tracking_m'] - nom['tracking_m']:+7.3f} cm, "
              f"{(base['tracking_m'] - nom['tracking_m'])/base['tracking_m']*100 if base['tracking_m'] else 0:+6.1f} %)")

        # where does the error live?
        s = base["series"]
        n = len(s)
        print("\n  -> per-sample tracking error (all uncertainty), cm:")
        step = max(1, n // 20)
        print("     " + " ".join(f"{s[i]*100:5.2f}" if s[i] is not None else "  -  "
                                for i in range(0, n, step)))
        print(f"     (sample indices {', '.join(str(i) for i in range(0, n, step))} of {n})")


if __name__ == "__main__":
    main()
