"""RQ-A experiment 5: attribute the residual tracking error.

Bisection: build the SAME arc three ways and run each through the shipped
validator.

  A  exact        - positions on the circle, heading = the exact tangent.
                    Fully self-consistent. The plant should reproduce it.
  B  trapezoid    - positions from the optimizer's trapezoidal rule, headings
                    still the exact tangent. Isolates the integration-scheme
                    mismatch (trapezoid vs explicit Euler).
  C  chord-heading- exact positions, but each heading replaced by the
                    direction of the chord to the next sample. Isolates the
                    plan's heading-vs-position inconsistency.

If A reports ~0 and B/C report ~1 cm, the whole residual belongs to the
optimizer's discretisation, and the fix is in optimizer.py - not in the
validator, the plant, or the seeds.
"""
import math
import os
import sys

import numpy as np

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)

import immrax_validator as IV  # noqa: E402
from jax_robot_model import JAXRobotConfig  # noqa: E402
from robot_model import RobotConfig  # noqa: E402

TRACK = 0.0965


def build_arc(radius, arc_len, n, speed, variant, t0=0.0):
    """Return a sample list for one arc built with the requested variant."""
    dtheta = arc_len / radius
    samples = []
    dt = (arc_len / speed) / n
    for k in range(n + 1):
        s = arc_len * k / n
        th = s / radius                       # exact tangent heading
        x_exact = radius * math.sin(th)
        y_exact = radius * (1.0 - math.cos(th))
        samples.append({"t": t0 + k * dt, "x": x_exact, "y": y_exact,
                        "heading": th, "v_exact": speed, "s": s})
    if variant == "trapezoid":
        # The optimizer's rule: x[k+1] = x[k] + 0.5*(v_k*cos(th_k) +
        # v_{k+1}*cos(th_{k+1}))*dt.  Headings stay the exact tangent.
        for k in range(n):
            a, b = samples[k], samples[k + 1]
            b["x"] = a["x"] + 0.5 * (speed * math.cos(a["heading"])
                                     + speed * math.cos(b["heading"])) * dt
            b["y"] = a["y"] + 0.5 * (speed * math.sin(a["heading"])
                                     + speed * math.sin(b["heading"])) * dt
    elif variant == "chord-heading":
        # Positions stay exact; each heading becomes the direction of the
        # chord to the NEXT sample (a self-consistent arc-length heading).
        for k in range(n):
            a, b = samples[k], samples[k + 1]
            a["heading"] = math.atan2(b["y"] - a["y"], b["x"] - a["x"])
        samples[n]["heading"] = samples[n - 1]["heading"] + \
            (samples[n]["heading"] - samples[n - 1]["heading"])

    # Wheel speeds and the derived fields the validator expects.
    out = []
    for k, s in enumerate(samples):
        v = speed
        om = v / radius
        vl = v - om * TRACK / 2.0
        vr = v + om * TRACK / 2.0
        out.append({
            "t": s["t"], "x": s["x"], "y": s["y"], "heading": s["heading"],
            "vl": vl, "vr": vr, "omega": om,
            "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
        })
    return out


def main():
    import json
    cfg_data = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                              encoding="utf-8"))
    jax_cfg = JAXRobotConfig(RobotConfig(cfg_data))

    RANGES = dict(cof=(1.24, 1.51), torque_headroom=(0.75, 0.9),
                  gear_backlash_m=(0.0004, 0.0005),
                  track_width_range=(0.0964, 0.0966),
                  wheel_radius_range=(0.0275, 0.0285),
                  inertia_kg_m2=(0.0008, 0.0011))

    for n in (10, 40):
        for speed in (0.4,):
            print(f"\n{'='*92}")
            print(f"arc R=0.30 m, length 0.60 m, dtheta={0.6/0.3:.3f} rad total, "
                  f"n={n} segments, v={speed} m/s")
            print("=" * 92)
            print(f"{'variant':<22} {'dtheta/sample':>13} {'track_cm':>10} "
                  f"{'env_cm':>8} {'gate':>8}")
            for variant in ("exact", "trapezoid", "chord-heading"):
                samples = build_arc(0.30, 0.60, n, speed, variant)
                dth = abs(samples[-1]["heading"] - samples[0]["heading"]) / n
                validator = IV.ImmraxValidator(jax_cfg)
                rep = validator.validate_trajectory(
                    samples, RANGES["cof"], RANGES["torque_headroom"],
                    RANGES["gear_backlash_m"], RANGES["track_width_range"],
                    RANGES["wheel_radius_range"], RANGES["inertia_kg_m2"])
                gate = "PASS" if rep["passed"] else "fail"
                print(f"{variant:<22} {dth:>13.4f} "
                      f"{rep['max_tracking_error_m']*100:>10.3f} "
                      f"{rep['reachability']['max_envelope_radius_m']*100:>8.3f} {gate:>8}")

    # Same arc, all uncertainty collapsed to nominal, to separate the
    # uncertainty contribution from the discretisation contribution.
    print(f"\n{'='*92}")
    print("same arcs, ALL uncertainty collapsed (isolates the discretisation floor)")
    print("=" * 92)
    print(f"{'variant':<22} {'n':>4} {'track_cm':>10} {'gate':>8}")
    for n in (10, 20, 40, 80):
        for variant in ("exact", "trapezoid", "chord-heading"):
            samples = build_arc(0.30, 0.60, n, 0.4, variant)
            validator = IV.ImmraxValidator(jax_cfg)
            rep = validator.validate_trajectory(
                samples, (1.24, 1.24), (0.75, 0.75), (0.0, 0.0),
                (0.0965, 0.0965), (0.028, 0.028), (0.001, 0.001))
            gate = "PASS" if rep["passed"] else "fail"
            print(f"{variant:<22} {n:>4} {rep['max_tracking_error_m']*100:>10.3f} {gate:>8}")


if __name__ == "__main__":
    main()
