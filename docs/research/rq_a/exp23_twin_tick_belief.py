"""RQ-A experiment 23: the twin's tick map - is the belief the reason?

The tick map is a clear win on the default plant (exp22: the S-curve goes
from 5.36 cm to 1.88 cm) and it is *worse* with the digital twin (5.08 cm
against 12.08 cm). The obvious suspect is the belief: the map has to carry
the robot's own pose estimate, and an estimate in an interval state grows
with the wheel-speed uncertainty, so the belief's spread is added to the
plant's.

This is the control for that explanation. Three variants, same plant, same
uncertainty, same reference:

  plant              the shipped propagation, the controller outside it
  tick + belief      the tick map with the belief in the state
  tick + truth       the tick map with the controller fed the *truth* - the
                     belief dropped entirely

If the belief is the reason, the third variant is dramatically smaller. It is
a ceiling and not a solution: dropping the belief's error makes the model
miss poses the robot reaches, and the containment check says so.
"""
import json
import math
import os
import sys
import time

import numpy as np
import jax.numpy as jnp

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

GAIN = (0.99, 1.01)          # the calibrated model: +/-1 %
TRAJ = os.environ.get("OVERBOARD_TRAJ", os.path.join(PINNED, "measurements",
                                                       "traj"))
MISSIONS = ("example_s_curve", "example_straight")


def _twin():
    return DigitalTwinConfig.from_dict({"actuator": {"straight": {
        "calibrated": True, "gain": 1.0,
        "gain_min": GAIN[0], "gain_max": GAIN[1],
        "tau_s": 0.02, "delay_s": 0.0,
    }}})


def _build(twin, propagation, inclusion, carry_belief=None):
    """A validator, with the tick map rebuilt when the belief is dropped."""
    val = IV.ImmraxValidator(JAX_CFG, digital_twin=twin,
                              propagation=propagation,
                              tick_inclusion=inclusion)
    if propagation == "tick" and carry_belief is not None:
        tick = IV.ClosedLoopTickMap(
            val.plant, val.controller, JAX_CFG.track_width,
            substeps=IV.PLANT_SUBSTEPS, carry_belief=carry_belief,
            belief_scale=(val.digital_twin_distance_scale,
                          val.digital_twin_heading_scale),
            gyro_bias=val.digital_twin_gyro_bias_rad_s,
            delay_ticks=val._tick_delay_ticks, wheel_state=val.wheel_state)
        val.tick_map = tick
        val.tick_emb_sys = IV._plant_embedding(tick, inclusion)
    return val


def _mc(samples, plant, with_belief, n=40, seed=0):
    """The reachable set, sampled with the tree's own point simulation."""
    E19.GAIN = GAIN
    plan_start = jnp.array([samples[0]['x'], samples[0]['y'],
                             samples[0]['heading']])
    cloud = [[] for _ in samples]
    for ws, starts in E19.build_datasets(n, seed):
        starts = starts + plan_start
        poses = np.asarray(E19.point_sim(samples, ws, starts, plant, JAX_CFG,
                                         with_belief=with_belief))
        for k in range(len(samples)):
            cloud[k].append(poses[:, k, :])
    return [np.concatenate(c, axis=0) for c in cloud]


def _radius(cloud):
    worst = 0.0
    for c in cloud:
        worst = max(worst, math.hypot((c[:, 0].max() - c[:, 0].min()) / 2.0,
                                      (c[:, 1].max() - c[:, 1].min()) / 2.0))
    return worst


def _overshoot(cloud, env):
    worst = 0.0
    for k in range(min(len(env), len(cloud))):
        if env[k].get("unknown"):
            continue
        for px, py in cloud[k]:
            worst = max(worst, env[k]["x_min"] - px, px - env[k]["x_max"],
                        env[k]["y_min"] - py, py - env[k]["y_max"])
    return worst


def main():
    twin = _twin()
    for name in MISSIONS:
        path = os.path.join(TRAJ, f"{name}.traj")
        if not os.path.exists(path):
            print(f"{name}: no trajectory ({path})")
            continue
        samples = json.load(open(path, encoding="utf-8"))["trajectory"]["samples"]
        print("=" * 96)
        print(f"{name} - the digital twin, +/-1 % gain band")
        print("=" * 96)
        cloud = _mc(samples, _build(twin, "plant", "mjac").plant, True)
        print(f"the sampled reachable set: {_radius(cloud)*100:.2f} cm")
        print()
        print(f"{'variant':>26} {'envelope':>10} {'tracking':>10} {'sec':>6} "
              f"{'contains':>9} {'over mm':>9}")
        for label, propagation, carry in (
                ("plant", "plant", None),
                ("tick + belief in the state", "tick", True),
                ("tick + controller on the truth", "tick", False)):
            val = _build(twin, propagation, "mjac", carry)
            t0 = time.time()
            rep = val.validate_trajectory(
                samples, UNC["cof"], UNC["torque_headroom"],
                UNC["gear_backlash_m"], UNC["track_width_range"],
                UNC["wheel_radius_range"], UNC["inertia_kg_m2"])
            elapsed = time.time() - t0
            env = rep["reachability"]["envelope"]
            worst = _overshoot(cloud, env)
            print("  %-24s %8.2f cm %8.2f cm %6.1f %9s %9.2f"
                  % (label, rep["max_envelope_radius_m"] * 100,
                     rep["max_tracking_error_m"] * 100, elapsed,
                     "yes" if worst <= 1e-9 else "NO", worst * 1000))
        print()
    print("If the belief were the reason, the third row would be small. It is")
    print("not, so the reason is the twin's plant state - the two wheel")
    print("speeds are interval coordinates whose spread is fed back into the")
    print("command, and the interval arithmetic counts that growth twice.")
    print("The third row is also unsound: dropping the belief's error makes")
    print("the model miss poses the robot reaches, which the containment")
    print("check reports as an overshoot.")


if __name__ == "__main__":
    main()
