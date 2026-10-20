"""RQ-A experiment 10: what sets the tracking-error floor at fine sampling?

OQ1 asked: at n=40-80 the plan/profile deviation is 0.063 cm but the
tracking error is 0.958-1.072 cm, so 95 % of the error was unexplained.

The answer is the control loop's zero-order hold.

The plan's own trapezoidal rule says the distance covered in one sample
interval is the interval average speed times the interval:

    distance_k = 0.5 * (v_k + v_{k+1}) * dt_k

The deployed controller runs a 20 ms loop: it reads the reference at the tick
time and the drive command is held until the next tick. So the robot's
velocity is a STAIRCASE of the reference velocity, sampled at the tick times,
and the distance it covers in an interval is the mean of that staircase - not
the interval average the plan assumed. The mismatch per interval is

    0.5 * (v_k - v_{k+1}) * dt_k        (one tick's worth of velocity change)

and it ACCUMULATES along the path, because every interval is covered
slightly wrong in the same direction while the velocity profile is ramping.

That makes the error proportional to the control tick and to the path's total
velocity variation - and independent of the sampling density, because the
optimizer returns a different velocity profile at every density.

The script holds the plan fixed and toggles one thing at a time:

  1. the control tick (20 / 10 / 5 / 2 ms)
  2. the plant's explicit Euler step
  3. the Ramsete gains
  4. the command quantiser
  5. the reference phase (read at the tick's start, middle or end)
  6. the reference velocity: the shipped interpolation versus the interval
     average, which is what the plan's own trapezoidal rule assumes

It also predicts the hold error from the plan alone, so the law can be
checked without running the closed loop at all.
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

TMP = tempfile.mkdtemp(prefix="rqa10_")

TICK = 0.02


def solve(mission, per_segment, weight, n, out):
    """Solve in-process: the CLI is not needed and the tree may be mid-edit."""
    from robot_model import RobotConfig
    from optimizer import TrajectoryOptimizer
    raw = json.load(open(os.path.join(PINNED, "examples", "%s.json" % mission),
                        encoding="utf-8"))
    wps = [(w["x"], w["y"], w.get("heading")) for w in raw]
    stops = sorted(i for i, w in enumerate(raw) if w.get("stop"))
    cfg = RobotConfig(json.load(open(os.path.join(PINNED, "examples",
                                                  "fll_choreo.json"),
                                 encoding="utf-8")))
    samples, _ = TrajectoryOptimizer(cfg).solve(
        wps, num_samples_per_segment=n, accuracy_weight=weight,
        stop_waypoint_indices=stops, verbose=False, guess_style="baseline",
        per_segment_dt=per_segment)
    return {"trajectory": {"samples": samples}}


def plan_profile_deviation(samples, track=0.0965, sub=60):
    """Max distance between the plan's positions and the exact integral of the
    plan's own (v, omega) profile (the instrument of exp3)."""
    v = (np.array([s["vl"] for s in samples]) + np.array([s["vr"] for s in samples])) / 2.0
    om = (np.array([s["vr"] for s in samples]) - np.array([s["vl"] for s in samples])) / track
    t = np.array([s["t"] for s in samples])
    plan_x = np.array([s["x"] for s in samples])
    plan_y = np.array([s["y"] for s in samples])
    th = np.array([s["heading"] for s in samples])
    dt = np.diff(t)
    rx, ry = plan_x[0], plan_y[0]
    worst = 0.0
    for k in range(len(samples) - 1):
        d = dt[k]
        u = np.arange(1, sub + 1) / sub
        v_i = v[k] + (v[k + 1] - v[k]) * u
        om_i = om[k] + (om[k + 1] - om[k]) * u
        th_i = th[k] + om[k] * u * d + 0.5 * (om[k + 1] - om[k]) / d * (u * d) ** 2
        h = d / sub
        rx = rx + h * np.sum(v_i * np.cos(th_i))
        ry = ry + h * np.sum(v_i * np.sin(th_i))
        worst = max(worst, math.hypot(rx - plan_x[k + 1], ry - plan_y[k + 1]))
    return worst


def hold_error(samples, tick=TICK):
    """The distance the robot covers per interval with the reference velocity
    held at the tick times, minus the distance the plan's trapezoidal rule
    says it should cover. The signed errors alternate in sign as the velocity
    profile ramps up and down, so the quantity that matters is the running
    sum's excursion along the path - the same statistic the tracking error
    measures."""
    v = (np.array([s["vl"] for s in samples]) + np.array([s["vr"] for s in samples])) / 2.0
    t = np.array([s["t"] for s in samples])
    dt = np.diff(t)
    running = 0.0
    worst = 0.0
    for k in range(len(samples) - 1):
        d = dt[k]
        covered = 0.0
        left = d
        while left > 1e-12:
            h = min(tick, left)
            frac = (d - left) / d          # where in the interval this tick starts
            covered += (v[k] + (v[k + 1] - v[k]) * frac) * h
            left -= h
        running += covered - 0.5 * (v[k] + v[k + 1]) * d
        worst = max(worst, abs(running))
    return worst


def make_ref(phase_mode):
    """The reference the controller sees. 'shipped' reads it at the tick's
    start; 'end'/'mid' shift the phase. 'avg_v' additionally replaces the
    interpolated velocity by the interval average, which is what the plan's
    own trapezoidal rule assumes."""
    import immrax_validator as IV
    orig = IV._interpolated_reference

    def ref(s0, s1, dt, track_width, s, ramsete_b=2.0):
        frac = s
        if phase_mode in ("end", "mid"):
            tick = min(IV.CONTROL_DT, dt)
            frac = s + (tick if phase_mode == "end" else 0.5 * tick) / dt
            frac = min(max(frac, 0.0), 1.0)
        r = list(orig(s0, s1, dt, track_width, frac, ramsete_b))
        if phase_mode == "avg_v":
            v0 = (s0["vl"] + s0["vr"]) / 2.0
            v1 = (s1["vl"] + s1["vr"]) / 2.0
            om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
            om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)
            r[3] = 0.5 * (v0 + v1)
            r[4] = 0.5 * (om0 + om1)
            r[5] = math.sqrt(r[4] ** 2 + ramsete_b * r[3] ** 2)
        return tuple(r)
    return orig, ref


def main():
    import immrax_validator as IV
    from jax_robot_model import JAXRobotConfig
    from robot_model import RobotConfig

    cfg_data = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                              encoding="utf-8"))
    jax_cfg = JAXRobotConfig(RobotConfig(cfg_data))
    orig_ref = IV._interpolated_reference

    def validate(samples, **kw):
        val = IV.ImmraxValidator(jax_cfg, **kw)
        return val.validate_trajectory(
            samples, (1.24, 1.51), (0.75, 0.9), (0.0004, 0.0005),
            (0.0964, 0.0966), (0.0275, 0.0285), (0.0008, 0.0011))

    missions = [("example_s_curve", True), ("example_with_stops", True),
                ("example_complex_mission", False), ("example_straight", True)]
    ns = [int(a) for a in (sys.argv[1:] or ["10", "20", "40", "80"])]

    for mission, per_seg in missions:
        print(f"\n{'='*104}")
        print(f"{mission}   per_segment={per_seg}")
        print("="*104)
        print(f"  {'n':>4} {'ticks/intv':>11} {'dev_cm':>8} {'shipped':>9} "
              f"{'avg_v':>8} {'2ms':>8} {'hold_pred_mm':>13} {'dv_var':>7}")
        for n in ns:
            out = os.path.join(TMP, f"{mission}_{n}.traj")
            try:
                traj = solve(mission, per_seg, 0.0, n, out)
            except RuntimeError as exc:
                print(f"  {n:>4}  FAILED {str(exc)[:100]}")
                continue
            samples = traj["trajectory"]["samples"]
            dur = samples[-1]["t"] - samples[0]["t"]
            dt_int = dur / (len(samples) - 1)
            dev = plan_profile_deviation(samples)
            pred = hold_error(samples) * 1000
            v = (np.array([s["vl"] for s in samples]) + np.array([s["vr"] for s in samples])) / 2.0
            var = np.abs(np.diff(v)).sum()

            IV._interpolated_reference = orig_ref
            base = validate(samples)
            o, r = make_ref("avg_v")
            IV._interpolated_reference = r
            avgv = validate(samples)
            IV._interpolated_reference = o
            fine = validate(samples, control_dt=0.002)
            IV._interpolated_reference = orig_ref

            print(f"  {n:>4} {dt_int/TICK:11.2f} {dev*100:8.4f} "
                  f"{base['max_tracking_error_m']*100:9.4f} "
                  f"{avgv['max_tracking_error_m']*100:8.4f} "
                  f"{fine['max_tracking_error_m']*100:8.4f} "
                  f"{pred:13.2f} {var:7.3f}")

    # The axes that do NOT matter, on the S-curve at the OQ1 densities.
    print(f"\n{'='*104}")
    print("Other axes, example_s_curve, plan held fixed")
    print("="*104)
    for n in (40, 80):
        out = os.path.join(TMP, f"axes_{n}.traj")
        traj = solve("example_s_curve", True, 0.0, n, out)
        samples = traj["trajectory"]["samples"]
        IV._interpolated_reference = orig_ref
        base = validate(samples)["max_tracking_error_m"] * 100
        rows = []
        for tick in (0.01, 0.005, 0.002):
            rows.append((f"control tick {tick*1000:.0f} ms",
                         validate(samples, control_dt=tick)["max_tracking_error_m"] * 100))
        orig_sub = IV.PLANT_SUBSTEPS
        for sub in (16, 64):
            IV.PLANT_SUBSTEPS = sub
            rows.append((f"plant substeps {sub}",
                         validate(samples)["max_tracking_error_m"] * 100))
        IV.PLANT_SUBSTEPS = orig_sub
        for b, z in ((1.0, 0.7), (4.0, 0.7), (2.0, 0.3), (2.0, 1.2)):
            rows.append((f"ramsete b={b} zeta={z}",
                         validate(samples, ramsete_b=b, ramsete_zeta=z)["max_tracking_error_m"] * 100))
        orig_q = IV.quantize_drive_command
        IV.quantize_drive_command = lambda v, w: (v, w)
        rows.append(("no command quantiser",
                     validate(samples)["max_tracking_error_m"] * 100))
        IV.quantize_drive_command = orig_q
        for mode in ("mid", "end"):
            o, r = make_ref(mode)
            IV._interpolated_reference = r
            rows.append((f"reference phase {mode}",
                         validate(samples)["max_tracking_error_m"] * 100))
            IV._interpolated_reference = o
        IV._interpolated_reference = orig_ref
        print(f"\n  n={n}   as shipped {base:.4f} cm")
        for label, val in rows:
            print(f"    {label:<28} {val:8.4f} cm   ({val/base:.3f} x)")


if __name__ == "__main__":
    main()
