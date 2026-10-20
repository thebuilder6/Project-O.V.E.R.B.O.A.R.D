"""RQ-A experiment 11: the tracking error as a function of sampling density.

exp10_floor.py showed two different limiting mechanisms at the two ends of
the density range:

  * coarse sampling (n=10): the plan's positions are off the path its own
    (v, omega) profile traces (1.15 cm), and fixing them takes the error
    from 1.248 cm to 0.494 cm;
  * fine sampling (n=40, 80): the 20 ms command hold is comparable to the
    sample interval, so the command is stale relative to the reference
    (0.979 cm -> 0.048 cm with a 2 ms tick).

If the two mechanisms trade off, the error as a function of n is U-shaped
and has a minimum. That would mean "sample more finely" is not a fix, and
that the sampling density has to be matched to the controller's 20 ms tick.

This measures the curve and the two fixes against it.
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

TMP = tempfile.mkdtemp(prefix="rqa11_")

RANGES = dict(cof=(1.24, 1.51), torque_headroom=(0.75, 0.9),
              gear_backlash_m=(0.0004, 0.0005),
              track_width_range=(0.0964, 0.0966),
              wheel_radius_range=(0.0275, 0.0285),
              inertia_kg_m2=(0.0008, 0.0011))


def solve(mission, n, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", "0.0", "-n", str(n), "--per-segment-dt"]
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission} n={n}: {r.stdout[-1000:]}")
    return json.load(open(out, encoding="utf-8"))


def rebuild_exact(samples, track=0.0965, sub=60):
    """The plan's positions replaced by the exact integral of its profile."""
    v = (np.array([s["vl"] for s in samples]) + np.array([s["vr"] for s in samples])) / 2.0
    om = (np.array([s["vr"] for s in samples]) - np.array([s["vl"] for s in samples])) / track
    t = np.array([s["t"] for s in samples])
    out = []
    px, py, pth = samples[0]["x"], samples[0]["y"], samples[0]["heading"]
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

    def validate(samples, **kw):
        val = IV.ImmraxValidator(jax_cfg, **kw)
        return val.validate_trajectory(
            samples, RANGES["cof"], RANGES["torque_headroom"],
            RANGES["gear_backlash_m"], RANGES["track_width_range"],
            RANGES["wheel_radius_range"], RANGES["inertia_kg_m2"])

    missions = ["example_s_curve", "example_with_stops", "example_complex_mission"]
    ns = [5, 10, 15, 20, 30, 40, 60]

    for mission in missions:
        print(f"\n{'='*104}")
        print(f"{mission}")
        print("="*104)
        print(f"{'n':>4} {'samples':>8} {'dt ms':>7} {'ticks/int':>10} "
              f"{'shipped':>9} {'gate':>6} | {'exact plan':>11} {'gate':>6} | "
              f"{'2ms tick':>9} {'gate':>6} | {'both':>8} {'gate':>6}")
        for n in ns:
            out = os.path.join(TMP, f"{mission}_{n}.traj")
            try:
                traj = solve(mission, n, out)
            except RuntimeError as exc:
                print(f"{n:>4}  FAILED {str(exc)[:90]}")
                continue
            samples = traj["trajectory"]["samples"]
            dt_ms = (samples[1]["t"] - samples[0]["t"]) * 1000
            ticks = dt_ms / 20.0

            a = validate(samples)
            b = validate(rebuild_exact(samples))
            c = validate(samples, control_dt=0.002)
            d = validate(rebuild_exact(samples), control_dt=0.002)

            def cell(rep):
                return (f"{rep['max_tracking_error_m']*100:.3f}",
                        "PASS" if rep["passed"] else "fail")
            sa, ga = cell(a)
            sb, gb = cell(b)
            sc, gc = cell(c)
            sd, gd = cell(d)
            print(f"{n:>4} {len(samples):>8} {dt_ms:>7.1f} {ticks:>10.2f} "
                  f"{sa:>9} {ga:>6} | {sb:>11} {gb:>6} | {sc:>9} {gc:>6} | "
                  f"{sd:>8} {gd:>6}")


if __name__ == "__main__":
    main()
