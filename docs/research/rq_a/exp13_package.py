"""RQ-A experiment 13: the full package, and what each part is worth.

exp12 found that making the reference's turn rate consistent with its
heading target (omega = dtheta/dt, instead of a line that contradicts the
line the heading is interpolated on) turns the S-curve and the stops
mission from fail to pass at n=40.

exp6/exp11 found that the plan's positions are off the path its own
(v, omega) profile traces, and that rebuilding them on the path removes
61 % of the error at n=10.

Those are the two halves of the package. This measures them together, so
the order of work and the value of each half are known.

  A  as shipped
  C  the reference's omega made consistent with its heading target
  E  the plan's positions moved onto the path its own profile traces
  CE both
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

TMP = tempfile.mkdtemp(prefix="rqa13_")


def solve(mission, n, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", "0.0", "-n", str(n), "--per-segment-dt"]
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission} n={n}: {r.stdout[-800:]}")
    return json.load(open(out, encoding="utf-8"))


def reference_C(s0, s1, dt, track_width, s, ramsete_b=2.0):
    """The shipped reference, except that the turn rate is the derivative of
    the interpolated heading instead of an independently interpolated line."""
    x0, y0, th0 = s0["x"], s0["y"], s0["heading"]
    th1 = s1["heading"]
    v0 = (s0["vl"] + s0["vr"]) / 2.0
    v1 = (s1["vl"] + s1["vr"]) / 2.0
    ref_x = x0 + (s1["x"] - x0) * s
    ref_y = y0 + (s1["y"] - y0) * s
    ref_heading = th0 + (th1 - th0) * s
    ref_om = (th1 - th0) / dt
    ref_v = v0 + (v1 - v0) * s
    kr = math.sqrt(ref_om ** 2 + ramsete_b * ref_v ** 2)
    return ref_x, ref_y, ref_heading, ref_v, ref_om, kr


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
        print(f"\n{'='*100}\n{mission}\n{'='*100}")
        print(f"{'n':>4} {'variant':>6} {'track_cm':>10} {'env_cm':>9} {'gate':>6} "
              f"{'vs shipped':>11}")
        for n in (10, 20, 40):
            out = os.path.join(TMP, f"{mission}_{n}.traj")
            try:
                traj = solve(mission, n, out)
            except RuntimeError:
                print(f"{n:>4}  (solve failed)")
                continue
            samples = traj["trajectory"]["samples"]
            exact = rebuild_exact(samples)
            base = None
            for label, ref, plan in (("A", orig, samples), ("C", reference_C, samples),
                                     ("E", orig, exact), ("CE", reference_C, exact)):
                IV._interpolated_reference = ref
                rep = validate(plan)
                trk = rep["max_tracking_error_m"] * 100
                if base is None:
                    base = trk
                print(f"{n:>4} {label:>6} {trk:>10.4f} "
                      f"{rep['reachability']['max_envelope_radius_m']*100:>9.4f} "
                      f"{'PASS' if rep['passed'] else 'fail':>6} "
                      f"{trk/base:>10.2f}x")
            IV._interpolated_reference = orig


if __name__ == "__main__":
    main()
