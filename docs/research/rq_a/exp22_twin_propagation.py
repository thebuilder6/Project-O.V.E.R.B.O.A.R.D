"""RQ-A experiment 22: the two propagations, on the shipped missions.

Experiment 20 measured the closed-loop tick map against the open-loop plant
propagation on synthetic paths. This measures the *shipped validator* with
the two propagations, on the missions the repository ships, with and without
a calibrated model.

The question is not "which tube is smaller" but "which one is the better
bound of the set the robot can actually reach", and that needs three
numbers per case:

  * the tube the envelope gate sees,
  * the set the robot can actually reach, sampled over the declared
    uncertainty with the corners included, and
  * whether the tube still contains it.

A tube that is smaller and contains the set is a better bound. A tube that
is smaller and does not contain it is worse than a wide one, because the
gate would pass a trajectory the robot cannot run.

The digital twin is measured separately, and it is the case that decides
this: the plant propagation there carries the robot's pose estimate as a
*point*, while the tick map has to carry it in the map's state, where it
becomes an interval. That interval is what the controller reacts to, and it
grows with the wheel-speed uncertainty - which is a different model, not
just a different arithmetic.

Usage (from the repository root):

    python docs/research/rq_a/exp22_twin_propagation.py
    python docs/research/rq_a/exp22_twin_propagation.py --draws 60
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)
sys.path.insert(0, os.path.join(PINNED, "docs", "research", "rq_a"))

import immrax_validator as IV  # noqa: E402
import exp19_envelope_scatter as E19  # noqa: E402
from digital_twin import DigitalTwinConfig  # noqa: E402
from jax_robot_model import JAXRobotConfig  # noqa: E402
from robot_model import RobotConfig  # noqa: E402

CFG = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                     encoding="utf-8"))
ROBOT = RobotConfig(CFG)
JAX_CFG = JAXRobotConfig(ROBOT)
UNC = ROBOT.uncertainties

ENVELOPE_LIMIT_M = 0.02
TRACKING_LIMIT_M = 0.005

MISSIONS = [
    ("example_straight", "examples/example_straight.json"),
    ("example_s_curve", "examples/example_s_curve.json"),
    ("example_mission_first4", "examples/example_mission_first4.json"),
    ("example_with_events", "examples/example_with_events.json"),
    ("example_with_stops", "examples/example_with_stops.json"),
    ("example_complex_mission", "examples/example_complex_mission.json"),
]

TWIN = DigitalTwinConfig.from_dict({"actuator": {"straight": {
    "calibrated": True, "gain": 1.0,
    "gain_min": 0.99, "gain_max": 1.01,
    "tau_s": 0.02, "delay_s": 0.0,
}}})


def load_samples(name):
    path = os.path.join(PINNED, "measurements", "traj", f"{name}.traj")
    if not os.path.exists(path):
        return None
    doc = json.load(open(path, encoding="utf-8"))
    return doc["trajectory"]["samples"]


def validate(samples, propagation, twin, inclusion=None, belief_mode=None):
    kw = {}
    if inclusion is not None:
        kw["tick_inclusion"] = inclusion
    if belief_mode is not None:
        kw["belief_mode"] = belief_mode
    val = IV.ImmraxValidator(JAX_CFG, digital_twin=twin,
                              propagation=propagation, **kw)
    if twin is None:
        # The gain band is the seventh uncertainty even without a twin.
        val.digital_twin_gain_range = E19.GAIN
    t0 = time.time()
    rep = val.validate_trajectory(
        samples, UNC["cof"], UNC["torque_headroom"],
        UNC["gear_backlash_m"], UNC["track_width_range"],
        UNC["wheel_radius_range"], UNC["inertia_kg_m2"])
    return rep, time.time() - t0, val


def reachable_set(samples, plant, n_draws, seed=0):
    """The set the robot can actually reach, sampled."""
    plan_start = jnp.array([samples[0]['x'], samples[0]['y'],
                             samples[0]['heading']])
    cloud = [[] for _ in samples]
    for ws, starts in E19.build_datasets(n_draws, seed):
        starts = starts + plan_start
        poses = np.asarray(E19.point_sim(samples, ws, starts, plant, JAX_CFG))
        for k in range(len(samples)):
            cloud[k].append(poses[:, k, :])
    return [np.concatenate(c, axis=0) for c in cloud]


def cloud_radius(cloud):
    worst = 0.0
    for pts in cloud:
        worst = max(worst, math.hypot(
            (pts[:, 0].max() - pts[:, 0].min()) / 2.0,
            (pts[:, 1].max() - pts[:, 1].min()) / 2.0))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=60,
                        help="Monte-Carlo draws per box")
    args = parser.parse_args()

    print("=" * 100)
    print("RQ-A experiment 22 - the shipped validator with the two propagations")
    print("=" * 100)
    print(f"gain band +/-1 %, {args.draws} draws per box, corners included")
    print(f"envelope gate {ENVELOPE_LIMIT_M*100:.0f} mm, "
          f"tracking gate {TRACKING_LIMIT_M*1000:.0f} mm")
    print()

    for twin, label in ((None, "WITHOUT a calibrated model (the default)"),
                        (TWIN, "WITH a calibrated model (+/-1 % gain band)")):
        print("=" * 100)
        print(label)
        print("=" * 100)
        header = (f"{'mission':>22} {'propagation':>18} | {'envelope':>9} "
                  f"{'tracking':>9} {'reachable':>10} {'contains':>9} "
                  f"{'over mm':>9} {'sec':>6}")
        print(header)
        print("-" * len(header))
        for name, _ in MISSIONS:
            samples = load_samples(name)
            if samples is None:
                print(f"{name:>22} (no trajectory: solve it first)")
                continue
            cloud = None
            for prop, incl in (("plant", None), ("tick", "jac"),
                               ("tick", "mjac")):
                rep, elapsed, val = validate(samples, prop, twin, incl)
                if cloud is None:
                    cloud = reachable_set(samples, val.plant, args.draws)
                inside, first_bad, worst = E19.containment(rep, cloud, samples)
                reach = rep["reachability"]
                print(f"{name:>22} {prop + '/' + str(incl):>18} | "
                      f"{rep['max_envelope_radius_m']*100:>8.2f}cm "
                      f"{rep['max_tracking_error_m']*100:>8.2f}cm "
                      f"{cloud_radius(cloud)*100:>9.2f}cm "
                      f"{('yes' if inside else f'NO@{first_bad}'):>9} "
                      f"{worst*1000:>9.3f} {elapsed:>6.1f}")
            print()
        print("envelope  - the half-diagonal the envelope gate sees")
        print("reachable - the half-diagonal of the set the robot can reach")
        print("contains  - is the sampled set inside the reported box")
        print("over mm   - how far the sampled set sticks out of the box")
        print()

    # The one idea that looked right and measured wrong: carrying the
    # robot's pose estimate as its ERROR rather than as the estimate
    # itself, so that the estimate's spread is not added to the truth's.
    print("=" * 100)
    print("The estimate's error instead of the estimate")
    print("=" * 100)
    print("The estimate and the truth are perfectly correlated - the odometry")
    print("reads the very wheel speeds the plant carries - so an interval box")
    print("adds the estimate's spread to the truth's. Carrying the error")
    print("instead should stop that. It does not pay:")
    print()
    print(f"{'mission':>22} {'plant':>12} {'estimate/jac':>14} {'error/jac':>12} "
          f"{'error/mjac':>12}")
    for name, _ in MISSIONS:
        samples = load_samples(name)
        if samples is None:
            continue
        row = []
        for prop, mode, incl in (("plant", None, None),
                                 ("tick", "estimate", "jac"),
                                 ("tick", "error", "jac"),
                                 ("tick", "error", "mjac")):
            rep, _, _ = validate(samples, prop, TWIN, incl, mode)
            row.append(f"{rep['max_envelope_radius_m']*100:.2f}cm")
        print(f"{name:>22} {row[0]:>12} {row[1]:>14} {row[2]:>12} {row[3]:>12}")
    print()
    print("The error's update differences two cosines of a heading interval")
    print("that is itself wide, and the difference of two over-approximations")
    print("is worse than either. The estimate stays a state of its own.")


if __name__ == "__main__":
    main()
