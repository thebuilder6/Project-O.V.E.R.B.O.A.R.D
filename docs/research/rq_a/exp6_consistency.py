"""RQ-A experiment 6: is the OPTIMIZER's plan heading-consistent?

For a trajectory that a robot can actually follow, the heading at a sample
must be the tangent of the path through that sample. Using the chords that
the deployed controller actually follows, the consistent heading at sample k
is the bisector of the incoming and outgoing chord directions:

    psi_in  = atan2(p_k - p_{k-1})
    psi_out = atan2(p_{k+1} - p_k)
    theta_consistent(k) = bisector(psi_in, psi_out)
    residual(k) = theta_k - theta_consistent(k)

For an exact arc the residual is zero. Then bisect the real optimizer output:
rebuild each plan by EXACT integration of its own (v, omega) profile and
re-validate. If the rebuilt plan tracks far better, the trapezoidal
positions are the problem; if not, the speed/turn profile is.
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

TMP = tempfile.mkdtemp(prefix="rqa6_")


def solve(mission, per_segment, n, out):
    cmd = [sys.executable, "main.py", "-c", "examples/fll_choreo.json",
           "-w", f"examples/{mission}.json", "-o", out, "--no-validate",
           "-a", "0.0", "-n", str(n)]
    if per_segment:
        cmd.append("--per-segment-dt")
    r = subprocess.run(cmd, cwd=PINNED, capture_output=True, text=True)
    if not os.path.exists(out):
        raise RuntimeError(f"{mission}: {r.stdout[-1200:]}\n{r.stderr[-1200:]}")
    return json.load(open(out, encoding="utf-8"))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def consistency_residual(samples):
    """theta_k minus the heading consistent with the chords the robot follows."""
    x = np.array([s["x"] for s in samples])
    y = np.array([s["y"] for s in samples])
    th = np.array([s["heading"] for s in samples])
    n = len(samples)
    res = []
    for k in range(1, n - 1):
        psi_in = math.atan2(y[k] - y[k - 1], x[k] - x[k - 1])
        psi_out = math.atan2(y[k + 1] - y[k], x[k + 1] - x[k])
        # angle bisector of the two chord directions
        bis = math.atan2(math.sin(psi_in) + math.sin(psi_out),
                         math.cos(psi_in) + math.cos(psi_out))
        res.append(wrap(th[k] - bis))
    return np.array(res)


def rebuild_exact(samples, track=0.0965, sub=60):
    """Rebuild the plan by EXACT integration of its own (v, omega) profile.

    Keeps the same sample times and the same speed profile, but places the
    samples where the profile actually puts them.
    """
    out = []
    v = (np.array([s["vl"] for s in samples]) + np.array([s["vr"] for s in samples])) / 2.0
    om = (np.array([s["vr"] for s in samples]) - np.array([s["vl"] for s in samples])) / track
    t = np.array([s["t"] for s in samples])
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

    cases = [
        ("example_s_curve", True, 10),
        ("example_with_stops", True, 10),
        ("example_complex_mission", False, 10),
        ("example_straight", True, 10),
        ("example_with_events", True, 10),
    ]

    for mission, per_seg, n in cases:
        out = os.path.join(TMP, f"{mission}.traj")
        samples = solve(mission, per_seg, n, out)["trajectory"]["samples"]
        res = consistency_residual(samples)
        dth = np.abs(np.diff([s["heading"] for s in samples]))

        rep = validate(samples)
        rebuilt = rebuild_exact(samples)
        rep2 = validate(rebuilt)
        res2 = consistency_residual(rebuilt)

        print(f"\n{'='*92}")
        print(f"{mission}  n={n}  samples={len(samples)}")
        print("=" * 92)
        print(f"  mean |dtheta| per sample          : {dth.mean():.4f} rad "
              f"({math.degrees(dth.mean()):.2f} deg)")
        print(f"  heading-consistency residual      : mean {np.degrees(res.mean()):+.3f} deg, "
              f"max |{np.degrees(np.abs(res).max()):.3f}| deg")
        print(f"  as a fraction of dtheta/2         : "
              f"{math.degrees(np.abs(res).mean()) / max(math.degrees(dth.mean())/2, 1e-9):.2f}")
        print(f"  tracking, plan as built           : "
              f"{rep['max_tracking_error_m']*100:7.3f} cm  "
              f"{'PASS' if rep['passed'] else 'fail'}")
        print(f"  tracking, same profile, exact xyz : "
              f"{rep2['max_tracking_error_m']*100:7.3f} cm  "
              f"{'PASS' if rep2['passed'] else 'fail'}")
        print(f"  heading-consistency residual after rebuild: "
              f"mean {np.degrees(res2.mean()):+.3f} deg, "
              f"max |{np.degrees(np.abs(res2).max()):.3f}| deg")


if __name__ == "__main__":
    main()
