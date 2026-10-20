"""RQ-A experiment 16: does the validator mirror the firmware that runs?

The controllers in use run a synced 20 ms heartbeat and hand the robot
floats:

| | ramsete_compact_events.py (the pushed run) | ramsete_compact.py (the SysId replay) | ramsete.py (the IDE missions) | the validator |
|---|---|---|---|---|
| control period | 20 ms, synced | 20 ms, synced | 20 ms, synced | **20 ms** (`CONTROL_DT`) |
| drive command | floats | floats | floats | **not rounded** |
| Ramsete gain | recomputed from the interpolated v, w | recomputed | recomputed | **recomputed** |
| turn rate | dh/dt | dh/dt | dh/dt | **dh/dt** |
| rate limit | `KinematicLimiter` | `KinematicLimiter` | `KinematicLimiter` | none |

The validator was written against `ramsete_fast.py`, the legacy path: it
rounds the command to mm/s and deg/s and interpolates a per-sample gain
instead of recomputing it. Those two differences are what this measures,
against the model that ships.

The period is the expensive line in principle: the hold error is proportional
to it (F6), so a hub that runs its heartbeat at 5 ms instead of 20 ms
reports a tracking error four times the size for the same plan. The tree ran
20 ms throughout, so this experiment's third column is there to *price* that
difference, not to correct a period that was wrong.

| column | what it is |
|---|---|
| `legacy` | the `ramsete_fast.py` mirror: the command rounded, the gain interpolated, a 20 ms hold |
| `shipped` | the model as it ships: the command unrounded, the gain recomputed, a 20 ms hold |
| `5 ms` | the shipped model with a 5 ms hold, to price the hold cost |

For the history: the deployed controller (`ramsete_compact_events.py`) and the
IDE missions (`ramsete.py`) ran a free-running `await wait(5)` loop - about
5 ms - until commit 12a365b, which moved them to the synced 20 ms heartbeat
the model already used. So the third column is the hold the robot actually
had, and the difference it prices is the cost of the 20 ms heartbeat to the
robot, not to the gate.
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

    def legacy_ref(s0, s1, dt, track_width, s, ramsete_b=2.0):
        """The reference the LEGACY firmware builds (ramsete_fast.py:73-76):
        x, y, heading and v interpolated, the turn rate the derivative of
        the heading (KI-027), and a per-sample Ramsete gain interpolated
        instead of recomputed."""
        v0 = (s0["vl"] + s0["vr"]) / 2.0
        v1 = (s1["vl"] + s1["vr"]) / 2.0
        om0 = s0.get("omega", (s0["vr"] - s0["vl"]) / track_width)
        om1 = s1.get("omega", (s1["vr"] - s1["vl"]) / track_width)
        kr0 = math.sqrt(om0 ** 2 + ramsete_b * v0 ** 2)
        kr1 = math.sqrt(om1 ** 2 + ramsete_b * v1 ** 2)
        dh = (s1["heading"] - s0["heading"] + math.pi) % (2.0 * math.pi) - math.pi
        return (s0["x"] + (s1["x"] - s0["x"]) * s,
                s0["y"] + (s1["y"] - s0["y"]) * s,
                s0["heading"] + dh * s,
                v0 + (v1 - v0) * s,
                dh / dt if dt > 0 else om0,
                kr0 + (kr1 - kr0) * s)

    def validate(samples, **kw):
        val = IV.ImmraxValidator(jax_cfg, **kw)
        return val.validate_trajectory(samples, *RANGES)

    cases = [("example_s_curve", True), ("example_with_stops", True),
             ("example_complex_mission", False), ("example_straight", True),
             ("example_with_events", True), ("example_mission_first4", True)]

    print(f"{'mission':<26} {'n':>4} {'legacy':>9} {'gate':>6} "
          f"{'shipped':>9} {'gate':>6} {'5 ms':>9} {'gate':>6} {'hold cost':>10}")
    print("-" * 100)
    for mission, per_seg in cases:
        for n in (10, 20, 40, 80):
            try:
                samples = solve(mission, per_seg, 0.0, n)
            except Exception as exc:      # a solve that does not converge
                print(f"{mission:<26} {n:>4}  solve failed: {str(exc)[:40]}")
                continue

            # (a) the legacy mirror: the command rounded to mm/s and deg/s,
            #     a per-sample gain interpolated, a 20 ms hold
            IV._interpolated_reference = legacy_ref
            IV.quantize_drive_command = orig_q
            base = validate(samples, control_dt=0.02)

            # (b) the model as it ships: the command unrounded, the gain
            #     recomputed from the interpolated v and omega, a 20 ms hold
            IV._interpolated_reference = orig_ref
            IV.quantize_drive_command = lambda v, w: (v, w)
            fw = validate(samples, control_dt=0.02)

            # (c) the same model with a 5 ms hold, to price the hold cost
            fast = validate(samples, control_dt=0.005)

            IV._interpolated_reference = orig_ref
            IV.quantize_drive_command = orig_q

            cost = (fw["max_tracking_error_m"] / max(fast["max_tracking_error_m"], 1e-12))
            print(f"{mission:<26} {n:>4} "
                  f"{base['max_tracking_error_m']*100:9.4f} "
                  f"{'PASS' if base['passed'] else 'fail':>6} "
                  f"{fw['max_tracking_error_m']*100:9.4f} "
                  f"{'PASS' if fw['passed'] else 'fail':>6} "
                  f"{fast['max_tracking_error_m']*100:9.4f} "
                  f"{'PASS' if fast['passed'] else 'fail':>6} "
                  f"{cost:9.2f}x")


if __name__ == "__main__":
    main()
