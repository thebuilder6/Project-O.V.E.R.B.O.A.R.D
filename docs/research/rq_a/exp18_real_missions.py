"""RQ-A experiment 18: the shipped missions under a Jacobian inclusion function.

Experiment 17 measured the envelope on synthetic paths and found that most of
it is an interval-arithmetic artefact: the plant computes the turn rate as
(vr - vl) / track_width, both wheel speeds are the *same* command times the
*same* gain, so their difference is exactly zero, and the natural extension
does not know that.

This script asks the question that matters: what does that do to the shipped
example missions and to the gate verdicts?

For every mission trajectory this repository ships, the same samples are
validated three times - with the shipped natural inclusion, with the
Jacobian inclusion and with the mixed-Jacobian inclusion - and the two gates
that can bind are reported for each:

  * the envelope gate: the tube's half-diagonal against the segment limit,
  * the tracking gate: the tube's midpoint against the plan, against 5 mm.

The gate verdicts are the ones main.py applies (main.py:1355-1387): the
envelope limit is the waypoint's threshold and the tracking limit is the
validator's own 5 mm.
"""
import json
import os
import sys
import time

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)

import immrax  # noqa: E402
import immrax_validator as IV  # noqa: E402
from digital_twin import DigitalTwinConfig  # noqa: E402
from jax_robot_model import JAXRobotConfig  # noqa: E402
from robot_model import RobotConfig  # noqa: E402

CFG = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                     encoding="utf-8"))
ROBOT = RobotConfig(CFG)
JAX_CFG = JAXRobotConfig(ROBOT)
UNC = ROBOT.uncertainties

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
TRACK_DIR = os.environ.get("OVERBOARD_TRAJ",
                           os.path.join(_REPO, "measurements", "traj"))

ENVELOPE_LIMIT_M = 0.02     # the default per-segment limit
TRACKING_LIMIT_M = 0.005    # the validator's own threshold

MISSIONS = [
    ("example_straight", "examples/example_straight.json"),
    ("example_s_curve", "examples/example_s_curve.json"),
    ("example_mission_first4", "examples/example_mission_first4.json"),
    ("example_with_events", "examples/example_with_events.json"),
    ("example_with_stops", "examples/example_with_stops.json"),
    ("example_complex_mission", "examples/example_complex_mission.json"),
]


def load_samples(name):
    path = os.path.join(TRACK_DIR, f"{name}.traj")
    doc = json.load(open(path, encoding="utf-8"))
    return doc["trajectory"]["samples"]


def validate(samples, plant_emb=None, twin=None):
    val = IV.ImmraxValidator(JAX_CFG, digital_twin=twin)
    if plant_emb is not None:
        val.plant_emb_sys = plant_emb
    t0 = time.time()
    rep = val.validate_trajectory(
        samples, UNC["cof"], UNC["torque_headroom"], UNC["gear_backlash_m"],
        UNC["track_width_range"], UNC["wheel_radius_range"],
        UNC["inertia_kg_m2"])
    return rep, time.time() - t0


def verdicts(rep):
    """The two gates main.py applies, and their verdicts."""
    env = rep["reachability"]["envelope"]
    worst = 0.0
    unknown = False
    for e in env:
        if e.get("unknown"):
            # The samples after the divergence carry the last computed box,
            # so they still say how wide the tube got.  They are counted, and
            # the divergence itself fails the gate.
            unknown = True
        err = ((e["x_max"] - e["x_min"]) / 2.0) ** 2 + \
              ((e["y_max"] - e["y_min"]) / 2.0) ** 2
        worst = max(worst, err ** 0.5)
    tracking = rep["max_tracking_error_m"]
    return {
        "envelope_m": worst,
        "tracking_m": tracking,
        "diverged": bool(rep["reachability"]["diverged"]),
        "envelope_ok": (not unknown) and worst <= ENVELOPE_LIMIT_M,
        "tracking_ok": tracking <= TRACKING_LIMIT_M,
    }


def main():
    # A calibrated model: the actuator gain has a band, so the plant's wheel
    # speeds are the command times a number that is not exactly one.  This is
    # the configuration the studio uses whenever a fitted model exists, and
    # the one F5 measured.
    twin = DigitalTwinConfig.from_dict({"actuator": {"straight": {
        "calibrated": True, "gain": 1.0,
        "gain_min": 0.99, "gain_max": 1.01,
        "tau_s": 0.02, "delay_s": 0.0,
    }}})

    for title, use_twin in (("WITHOUT a calibrated model (the default)", None),
                            ("WITH a calibrated model (+/-1 % gain band)",
                             twin)):
        print("=" * 100)
        print(title)
        print("=" * 100)
        print(f"envelope gate {ENVELOPE_LIMIT_M*100:.0f} mm, "
              f"tracking gate {TRACKING_LIMIT_M*1000:.0f} mm")
        print()
        header = (f"{'mission':>22} {'inclusion':>10} | {'envelope':>9} "
                  f"{'tracking':>9} {'seconds':>7} | {'env gate':>9} "
                  f"{'trk gate':>9}")
        print(header)
        print("-" * len(header))

        for name, mission in MISSIONS:
            path = os.path.join(TRACK_DIR, f"{name}.traj")
            if not os.path.exists(path):
                print(f"{name:>22} (no trajectory: run the solve first)")
                continue
            samples = load_samples(name)
            for label, factory in (("nat", immrax.natemb),
                                   ("jac", immrax.jacemb),
                                   ("mjac", immrax.mjacemb)):
                val = IV.ImmraxValidator(JAX_CFG, digital_twin=use_twin)
                emb = factory(val.plant)
                rep, elapsed = validate(samples, plant_emb=emb, twin=use_twin)
                v = verdicts(rep)
                gate = "pass" if v["envelope_ok"] else "FAIL"
                if v["diverged"]:
                    gate = "diverged"
                print(f"{name:>22} {label:>10} | {v['envelope_m']*100:>8.2f}cm "
                      f"{v['tracking_m']*100:>8.2f}cm {elapsed:>7.2f} | "
                      f"{gate:>9} "
                      f"{('pass' if v['tracking_ok'] else 'FAIL'):>9}")
            print()

    print("The envelope is the tube's half-diagonal at its worst sample; the")
    print("tracking error is the tube's midpoint against the plan.")
    print("A 'FAIL' on the envelope gate means the tube crossed the segment")
    print("limit or the propagation diverged.")
    print()
    print("The trajectories are the ones main.py writes for each mission:")
    print("  python main.py -w examples/<mission>.json -o <traj>.traj")


if __name__ == "__main__":
    main()
