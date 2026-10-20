"""RQ-A experiment 8: the complete proposal, and the gate criteria separated.

Proposal: make the plan self-consistent (samples placed where the profile
actually goes) AND interpolate the reference as a curve through the samples
instead of a chord. exp4 tried a spline reference on the plan AS BUILT and
it exploded to 17 cm, because the plan's headings are not the tangents of
its own path, so any curve honouring both is wildly curved.

Here the plan is first made self-consistent, then the spline is applied.
The per-criterion gate breakdown is printed so "tracking" is not conflated
with motor force or slip.
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

TMP = tempfile.mkdtemp(prefix="rqa8_")


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


def hermite_reference(s0, s1, dt, track_width, s, ramsete_b=2.0):
    x0, y0, th0 = s0["x"], s0["y"], s0["heading"]
    x1, y1, th1 = s1["x"], s1["y"], s1["heading"]
    v0 = (s0["vl"] + s0["vr"]) / 2.0
    v1 = (s1["vl"] + s1["vr"]) / 2.0
    om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
    om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)
    m0 = np.array([v0 * math.cos(th0), v0 * math.sin(th0)]) * dt
    m1 = np.array([v1 * math.cos(th1), v1 * math.sin(th1)]) * dt
    s2, s3 = s * s, s * s * s
    h00, h10 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s
    h01, h11 = -2 * s3 + 3 * s2, s3 - s2
    px = h00 * x0 + h10 * m0[0] + h01 * x1 + h11 * m1[0]
    py = h00 * y0 + h10 * m0[1] + h01 * y1 + h11 * m1[1]
    d00, d10 = 6 * s2 - 6 * s, 3 * s2 - 4 * s + 1
    d01, d11 = -6 * s2 + 6 * s, 3 * s2 - 2 * s
    dx = d00 * x0 + d10 * m0[0] + d01 * x1 + d11 * m1[0]
    dy = d00 * y0 + d10 * m0[1] + d01 * y1 + d11 * m1[1]
    d00b, d10b = 12 * s - 6, 6 * s - 4
    d01b, d11b = -12 * s + 6, 6 * s - 2
    ddx = d00b * x0 + d10b * m0[0] + d01b * x1 + d11b * m1[0]
    ddy = d00b * y0 + d10b * m0[1] + d01b * y1 + d11b * m1[1]
    speed = math.hypot(dx, dy)
    om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
    om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)
    if speed > 1e-9:
        ref_v = speed / dt
        # Curvature with respect to the interpolation parameter s: dtheta/ds.
        # The parameter advances at unit rate over dt seconds, so dtheta/dt is
        # that divided by dt. (Using speed/dt here would be the arc-length rate
        # and gives the wrong omega by a factor of the curve speed.)
        ref_om = (dx * ddy - dy * ddx) / (speed * speed * dt)
        raw = math.atan2(dy, dx)
        # The plan's headings are continuous (unwrapped); anchor the branch.
        ref_heading = th0 + ((raw - th0 + math.pi) % (2 * math.pi) - math.pi)
    else:
        # Degenerate interval (the robot is at rest, so the tangent carries no
        # direction): fall back to the plan's own profile.
        ref_v = 0.0
        ref_om = om0 + (om1 - om0) * s
        ref_heading = th0
    # The Ramsete gain must be consistent with the (v, omega) we report:
    # kr = sqrt(omega^2 + b*v^2). Using the plan's linearly interpolated kr
    # here would give the controller a wrong drive time constant.
    kr = math.sqrt(ref_om ** 2 + ramsete_b * ref_v ** 2)
    return (px, py, ref_heading, ref_v, ref_om, kr)


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
    orig_ref = IV._interpolated_reference

    def validate(samples):
        val = IV.ImmraxValidator(jax_cfg)
        rep = val.validate_trajectory(
            samples, R["cof"], R["torque_headroom"], R["gear_backlash_m"],
            R["track_width_range"], R["wheel_radius_range"], R["inertia_kg_m2"])
        motor = rep.get("max_motor_violation_N", 0.0)
        slip = max(rep.get("max_slip_l_N", 0.0), rep.get("max_slip_r_N", 0.0))
        speed = rep.get("speed_violation_m_s", 0.0)
        trk = rep["max_tracking_error_m"]
        return {
            "track_cm": trk * 100, "motor_N": motor, "slip_N": slip,
            "speed": speed, "env_cm": rep["reachability"]["max_envelope_radius_m"] * 100,
            "track_ok": trk <= 0.005, "motor_ok": motor < 1e-3,
            "slip_ok": slip < 1e-3, "speed_ok": speed <= 1e-6,
            "passed": rep["passed"],
        }

    cases = [("example_s_curve", True), ("example_with_stops", True),
             ("example_complex_mission", False), ("example_straight", True)]

    for mission, per_seg in cases:
        out = os.path.join(TMP, f"{mission}.traj")
        samples = solve(mission, per_seg, 10, out)["trajectory"]["samples"]
        exact = rebuild_exact(samples)

        print(f"\n{'='*104}")
        print(f"{mission}   (n=10, the default)")
        print("=" * 104)
        print(f"{'variant':<34} {'track_cm':>9} {'motor_N':>9} {'slip_N':>8} "
              f"{'env_cm':>7} | {'track':>6} {'motor':>6} {'slip':>5} {'ALL':>5}")
        rows = [
            ("1. as shipped", samples, orig_ref),
            ("2. self-consistent plan", exact, orig_ref),
            ("3. spline ref on plan as built", samples, hermite_reference),
            ("4. self-consistent + spline ref", exact, hermite_reference),
        ]
        for label, smp, ref in rows:
            IV._interpolated_reference = ref
            r = validate(smp)
            print(f"{label:<34} {r['track_cm']:>9.3f} {r['motor_N']:>9.4f} "
                  f"{r['slip_N']:>8.4f} {r['env_cm']:>7.3f} | "
                  f"{'ok' if r['track_ok'] else 'FAIL':>6} "
                  f"{'ok' if r['motor_ok'] else 'FAIL':>6} "
                  f"{'ok' if r['slip_ok'] else 'FAIL':>5} "
                  f"{'PASS' if r['passed'] else 'fail':>5}")
        IV._interpolated_reference = orig_ref


if __name__ == "__main__":
    main()
