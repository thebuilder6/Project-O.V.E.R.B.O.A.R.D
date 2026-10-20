"""RQ-A experiment 16: does the validator mirror the firmware that runs?

The validator was written against `ramsete_fast.py`. The controller that
actually runs is `ramsete_compact_events.py` (hub_runner.py imports it), and
the two do not agree:

| | ramsete_compact_events.py (deployed) | the validator |
|---|---|---|
| control period | `await wait(5)` -> ~5 ms | `CONTROL_DT = 0.02` -> 20 ms |
| command quantisation | none, floats to drive() | rounds to mm/s and deg/s |
| Ramsete gain kr | recomputed from the interpolated v, w | a precomputed per-sample value, interpolated |
| command rate limit | `KinematicLimiter` (motor force, backlash) | none |

The control period is the expensive one: the tracking error is proportional to
it (F6), so a validator that holds the command four times too long reports
four times the error the robot will have.

This measures the gate outcome with the validator configured to mirror the
deployed firmware.
"""
import json
import math
import os
import sys
import tempfile

import numpy as np

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)
TMP = tempfile.mkdtemp(prefix="rqa16_")

RANGES = ((1.24, 1.51), (0.75, 0.9), (0.0004, 0.0005),
          (0.0964, 0.0966), (0.0275, 0.0285), (0.0008, 0.0011))


def solve(mission, per_segment, weight, n):
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
    return samples


def main():
    import immrax_validator as IV
    from jax_robot_model import JAXRobotConfig
    from robot_model import RobotConfig

    jax_cfg = JAXRobotConfig(RobotConfig(json.load(
        open(os.path.join(PINNED, "examples", "fll_choreo.json"),
             encoding="utf-8"))))
    orig_ref = IV._interpolated_reference
    orig_q = IV.quantize_drive_command

    def firmware_ref(s0, s1, dt, track_width, s, ramsete_b=2.0):
        """The reference the DEPLOYED firmware builds: x, y, heading, v and
        omega interpolated, and the Ramsete gain recomputed from the
        interpolated v and omega (ramsete.py:104), not interpolated."""
        v0 = (s0["vl"] + s0["vr"]) / 2.0
        v1 = (s1["vl"] + s1["vr"]) / 2.0
        om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
        om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)
        ref_x = s0["x"] + (s1["x"] - s0["x"]) * s
        ref_y = s0["y"] + (s1["y"] - s0["y"]) * s
        ref_h = s0["heading"] + (s1["heading"] - s0["heading"]) * s
        ref_v = v0 + (v1 - v0) * s
        ref_om = om0 + (om1 - om0) * s
        ref_k = math.sqrt(ref_om ** 2 + ramsete_b * ref_v ** 2)
        return ref_x, ref_y, ref_h, ref_v, ref_om, ref_k

    def legacy_ref(s0, s1, dt, track_width, s, ramsete_b=2.0):
        """The reference the LEGACY firmware builds: x, y, heading, v, omega
        and a precomputed Ramsete gain, each interpolated as its own line
        (ramsete_fast.py:73-76)."""
        v0 = (s0["vl"] + s0["vr"]) / 2.0
        v1 = (s1["vl"] + s1["vr"]) / 2.0
        om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
        om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)
        kr0 = math.sqrt(om0 ** 2 + ramsete_b * v0 ** 2)
        kr1 = math.sqrt(om1 ** 2 + ramsete_b * v1 ** 2)
        return (s0["x"] + (s1["x"] - s0["x"]) * s,
                s0["y"] + (s1["y"] - s0["y"]) * s,
                s0["heading"] + (s1["heading"] - s0["heading"]) * s,
                v0 + (v1 - v0) * s,
                om0 + (om1 - om0) * s,
                kr0 + (kr1 - kr0) * s)

    def validate(samples, **kw):
        val = IV.ImmraxValidator(jax_cfg, **kw)
        return val.validate_trajectory(samples, *RANGES)

    cases = [("example_s_curve", True), ("example_with_stops", True),
             ("example_complex_mission", False), ("example_straight", True),
             ("example_with_events", True), ("example_mission_first4", True)]

    print(f"{'mission':<26} {'n':>4} {'legacy mirror':>15} {'gate':>6} "
          f"{'deployed mirror':>16} {'gate':>6} {'factor':>7}")
    print("-" * 88)
    for mission, per_seg in cases:
        for n in (10, 20, 40, 80):
            try:
                samples = solve(mission, per_seg, 0.0, n)
            except Exception as exc:      # a solve that does not converge
                print(f"{mission:<26} {n:>4}  solve failed: {str(exc)[:40]}")
                continue

            # (a) the legacy mirror: 20 ms, the command rounded to mm/s and
            #     deg/s, a precomputed gain interpolated
            IV._interpolated_reference = legacy_ref
            IV.quantize_drive_command = orig_q
            base = validate(samples, control_dt=0.02)

            # (b) mirroring the controller that runs: 20 ms synced hold, no
            #     rounding, the gain recomputed from the interpolated v and
            #     omega (ramsete.py:104)
            IV._interpolated_reference = firmware_ref
            IV.quantize_drive_command = lambda v, w: (v, w)
            fw = validate(samples, control_dt=0.02)
            IV._interpolated_reference = orig_ref
            IV.quantize_drive_command = orig_q

            print(f"{mission:<26} {n:>4} {base['max_tracking_error_m']*100:15.4f} "
                  f"{'PASS' if base['passed'] else 'fail':>6} "
                  f"{fw['max_tracking_error_m']*100:16.4f} "
                  f"{'PASS' if fw['passed'] else 'fail':>6} "
                  f"{base['max_tracking_error_m']/max(fw['max_tracking_error_m'],1e-12):7.2f}x")


if __name__ == "__main__":
    main()
