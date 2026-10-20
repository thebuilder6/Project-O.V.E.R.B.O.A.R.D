"""RQ-A experiment 21: where the envelope's width comes from on a curve.

Experiment 19 measured the interval box against the sampled reachable set on
a **straight** and found the box about 2.4x too wide. On a curve the gap is
far bigger, and the README's claim - "the reachable set is genuinely wide" -
has never been checked against the sampled set on a curve.

This measures, on a curved mission:

1. the box and the sampled set, sample by sample, in the world frame;
2. the same in the **path's own frame**, which is the frame a tighter
   inclusion function would carry the error in;
3. the box with the gain band collapsed to its nominal value, which
   isolates the mechanism: the band multiplies the *turn rate*, and on a
   curve the turn rate is real.

The sampled set is the same one experiment 19 builds: the shipped plant and
the shipped controller on points, over the declared uncertainty, with the
corners of both boxes included.
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

HERE = os.path.dirname(os.path.abspath(__file__))
PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, PINNED)
sys.path.insert(0, HERE)

import immrax_validator as IV  # noqa: E402
from digital_twin import DigitalTwinConfig  # noqa: E402
from jax_robot_model import JAXRobotConfig  # noqa: E402
from robot_model import RobotConfig  # noqa: E402

import exp19_envelope_scatter as E  # noqa: E402

CFG = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                     encoding="utf-8"))
ROBOT = RobotConfig(CFG)
JAX_CFG = JAXRobotConfig(ROBOT)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TRAJ_DIR = os.environ.get("OVERBOARD_TRAJ", os.path.join(REPO, "measurements",
                                                          "traj"))


def load_samples(name):
    path = os.path.join(TRAJ_DIR, f"{name}.traj")
    doc = json.load(open(path, encoding="utf-8"))
    return doc["trajectory"]["samples"]


def twin_with_gain(gain_min, gain_max):
    """A calibrated model with a gain band of the given width."""
    return DigitalTwinConfig.from_dict({"actuator": {"straight": {
        "calibrated": True, "gain": 1.0,
        "gain_min": gain_min, "gain_max": gain_max,
        "tau_s": 0.02, "delay_s": 0.0,
    }}})


def point_sim_with_heading(samples, ws, starts, plant, cfg, ramsete_b=2.0,
                            control_dt=0.02, with_belief=False,
                            open_loop=False):
    """Experiment 19's point simulation, but the heading is recorded too.

    The heading is the state the feedback is supposed to correct, so its
    spread is the thing that has to be measured.

    With ``open_loop`` the controller is fed the **plan's own pose**, so its
    correction term is zero and the robot follows the reference without
    correcting. That is the same plant and the same uncertainty without the
    feedback, and it is the control against the claim that the feedback is
    what keeps the reachable set narrow.
    """
    ctrl = IV.RamseteController(ramsete_b, 0.7, track_width=cfg.track_width)
    plant_f = jax.vmap(plant.f, in_axes=(None, 0, 0, 0))
    ctrl_f = jax.vmap(lambda t, x, u, w: ctrl(t, x, u, w, normalize=True,
                                               quantize=False),
                      in_axes=(None, 0, None, 0))

    x = jnp.concatenate([starts, jnp.zeros((starts.shape[0],
                                            plant.xlen - 3))], axis=1)
    belief = starts
    out = [jnp.stack([x[:, 0], x[:, 1], x[:, 2]], axis=1)]

    for i in range(len(samples) - 1):
        s, s_next = samples[i], samples[i + 1]
        dt = s_next['t'] - s['t']
        t = s['t']
        while t < s['t'] + dt - 1e-12:
            dt_ctrl = min(control_dt, s['t'] + dt - t)
            frac = (t - s['t']) / dt
            ref = IV._interpolated_reference(s, s_next, dt, cfg.track_width,
                                             frac, ramsete_b)
            u_ref = jnp.array(ref)
            if open_loop:
                seen = jnp.broadcast_to(_plan_pose(s, s_next, frac)[None, :],
                                        (starts.shape[0], 3))
            else:
                seen = belief if with_belief else x[:, :3]
            u_cmd = ctrl_f(0.0, seen, u_ref, ws)
            for _ in range(IV.PLANT_SUBSTEPS):
                if with_belief:
                    belief = E._advance_belief(belief, x, plant, cfg,
                                               dt_ctrl / IV.PLANT_SUBSTEPS)
                dx = plant_f(0.0, x, u_cmd, ws)
                x = x + dx * (dt_ctrl / IV.PLANT_SUBSTEPS)
            t += dt_ctrl
        out.append(jnp.stack([x[:, 0], x[:, 1], x[:, 2]], axis=1))

    return jnp.stack(out, axis=1)


def _plan_pose(a, b, frac):
    """The plan's own pose between two samples."""
    return jnp.array([a['x'] + frac * (b['x'] - a['x']),
                       a['y'] + frac * (b['y'] - a['y']),
                       a['heading'] + frac * (b['heading'] - a['heading'])])


def cloud_with_heading(samples, plant, n_draws, seed, open_loop=False):
    """The sampled reachable set, with the heading."""
    plan_start = jnp.array([samples[0]['x'], samples[0]['y'],
                             samples[0]['heading']])
    clouds = [[] for _ in samples]
    for ws, starts in E.build_datasets(n_draws, seed):
        starts = starts + plan_start
        poses = np.asarray(point_sim_with_heading(samples, ws, starts, plant,
                                                   JAX_CFG, with_belief=True,
                                                   open_loop=open_loop))
        for k in range(len(samples)):
            clouds[k].append(poses[:, k, :])
    return [np.concatenate(c, axis=0) for c in clouds]


def box_envelope(samples, gain_min, gain_max, inclusion="nat",
                 wheel_state="mean_diff"):
    """The interval box, with a gain band of the given width."""
    twin = twin_with_gain(gain_min, gain_max)
    val = IV.ImmraxValidator(JAX_CFG, digital_twin=twin,
                              plant_inclusion=inclusion,
                              controller_inclusion=inclusion,
                              wheel_state=wheel_state)
    report = val.validate_trajectory(
        samples, E.COF, E.TORQUE, E.BACKLASH, E.TRACK, E.RADIUS, E.INERTIA)
    return report, val


def half_width(points, axis):
    """The half width of a point cloud along one world axis."""
    return 0.5 * float(np.max(points[:, axis]) - np.min(points[:, axis]))


def frame_half_widths(points, centre, heading):
    """The half widths of a cloud in the path's own frame at one sample.

    The frame's longitudinal axis is the path's heading there; the lateral
    axis is perpendicular to it. This is the frame an envelope carried in the
    path's frame would live in.
    """
    c, s = math.cos(heading), math.sin(heading)
    dx = points[:, 0] - centre[0]
    dy = points[:, 1] - centre[1]
    along = c * dx + s * dy
    across = -s * dx + c * dy
    return (0.5 * float(np.max(along) - np.min(along)),
            0.5 * float(np.max(across) - np.min(across)))


def box_frame_half_widths(env, heading):
    """The half widths of the interval box in the path's own frame.

    The box is a rectangle in the world frame; its extent along a direction
    is the rectangle's width projected onto that direction.
    """
    c, s = math.cos(heading), math.sin(heading)
    wx = env["x_max"] - env["x_min"]
    wy = env["y_max"] - env["y_min"]
    along = 0.5 * (abs(c) * wx + abs(s) * wy)
    across = 0.5 * (abs(s) * wx + abs(c) * wy)
    return along, across


def report(mission, draws, seed, inclusion):
    samples = load_samples(mission)
    print("=" * 100)
    print(f"RQ-A experiment 21 - the envelope on a curve: {mission}")
    print("=" * 100)

    band = E.GAIN
    report_banded, val = box_envelope(samples, band[0], band[1], inclusion)
    report_narrow, _ = box_envelope(samples, 1.0, 1.0, inclusion)
    cloud = cloud_with_heading(samples, val.plant, draws, seed)
    open_cloud = cloud_with_heading(samples, val.plant, draws, seed,
                                    open_loop=True)

    env = report_banded["reachability"]["envelope"]
    env_narrow = report_narrow["reachability"]["envelope"]

    print(f"gain band {band[0]}..{band[1]} (the calibrated model), "
          f"{draws} draws per box, inclusion {inclusion}")
    print()
    header = (f"{'k':>3} {'t s':>6} | {'box x mm':>9} {'box y mm':>9} "
              f"{'box diag':>9} | {'cloud x mm':>11} {'cloud y mm':>11} "
              f"{'cloud diag':>11} {'head mrad':>10} | "
              f"{'box lat':>8} {'cloud lat':>10} {'box long':>9} "
              f"{'cloud long':>11} | {'narrow box':>11}")
    print(header)
    print("-" * len(header))

    for k in range(len(samples)):
        e = env[k]
        if e.get("unknown"):
            continue
        pts = cloud[k]
        bx = 0.5 * (e["x_max"] - e["x_min"]) * 1000
        by = 0.5 * (e["y_max"] - e["y_min"]) * 1000
        cx = half_width(pts, 0) * 1000
        cy = half_width(pts, 1) * 1000
        head = float(np.max(pts[:, 2]) - np.min(pts[:, 2])) * 1000
        b_long, b_lat = box_frame_half_widths(e, samples[k]['heading'])
        c_long, c_lat = frame_half_widths(pts, (samples[k]['x'],
                                                  samples[k]['y']),
                                          samples[k]['heading'])
        narrow = env_narrow[k]
        n_diag = 0.5 * math.hypot(narrow["x_max"] - narrow["x_min"],
                                  narrow["y_max"] - narrow["y_min"]) * 1000 \
            if not narrow.get("unknown") else float("nan")
        print(f"{k:>3} {samples[k]['t']:>6.2f} | {bx:>9.2f} {by:>9.2f} "
              f"{math.hypot(bx, by):>9.2f} | {cx:>11.2f} {cy:>11.2f} "
              f"{math.hypot(cx, cy):>11.2f} {head:>10.2f} | "
              f"{b_lat * 1000:>8.2f} {c_lat * 1000:>10.2f} "
              f"{b_long * 1000:>9.2f} {c_long * 1000:>11.2f} | {n_diag:>11.2f}")

    # The numbers the gate sees.
    def worst_diag(rep):
        worst = 0.0
        for e in rep["reachability"]["envelope"]:
            if e.get("unknown"):
                continue
            rx = 0.5 * (e["x_max"] - e["x_min"])
            ry = 0.5 * (e["y_max"] - e["y_min"])
            worst = max(worst, math.hypot(rx, ry))
        return worst

    cloud_worst = 0.0
    cloud_path_worst = 0.0
    head_worst = 0.0
    open_worst = 0.0
    for k in range(len(samples)):
        pts = cloud[k]
        cloud_worst = max(cloud_worst, math.hypot(half_width(pts, 0),
                                                  half_width(pts, 1)))
        c_long, c_lat = frame_half_widths(pts, (samples[k]['x'],
                                                  samples[k]['y']),
                                          samples[k]['heading'])
        cloud_path_worst = max(cloud_path_worst, math.hypot(c_long, c_lat))
        head_worst = max(head_worst, float(np.max(pts[:, 2])
                                            - np.min(pts[:, 2])))
        op = open_cloud[k]
        open_worst = max(open_worst, math.hypot(half_width(op, 0),
                                                half_width(op, 1)))

    print()
    print(f"the box the gate sees (half diagonal)   "
          f"{worst_diag(report_banded)*1000:>8.2f} mm   "
          f"truncated: {report_banded['reachability']['envelope_truncated']}")
    print(f"the sampled reachable set (half diagonal) "
          f"{cloud_worst*1000:>8.2f} mm")
    print(f"the sampled set in the path's frame      "
          f"{cloud_path_worst*1000:>8.2f} mm")
    print(f"the same simulation with the feedback off "
          f"{open_worst*1000:>8.2f} mm")
    print(f"the box with the gain band collapsed     "
          f"{worst_diag(report_narrow)*1000:>8.2f} mm")
    print(f"the sampled heading spread (worst)       "
          f"{head_worst*1000:>8.2f} mrad")
    print()
    print("The gate limit is 20 mm. The box is what the validator reports; the")
    print("sampled set is what the robot can actually reach. The box with the")
    print("gain band collapsed says whether the band is the driver: the band")
    print("multiplies the turn rate, and on a curve the turn rate is real.")
    print()
    print("The line with the feedback off is the control: same plant, same")
    print("uncertainty, same reference, but the controller is fed the plan's")
    print("own pose so its correction term is zero. If the tube tracks that")
    print("line, the tube is behaving like an open-loop robot and the")
    print("feedback's contraction of the error is what the tube is missing.")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", default="example_s_curve")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inclusion", default="nat",
                        choices=list(E.INCLUSIONS))
    args = parser.parse_args()
    report(args.mission, args.draws, args.seed, args.inclusion)


if __name__ == "__main__":
    main()
