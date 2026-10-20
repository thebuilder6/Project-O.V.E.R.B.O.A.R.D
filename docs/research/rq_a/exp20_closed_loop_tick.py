"""RQ-A experiment 20: the closed-loop tick map.

The validator propagates the *plant* and evaluates the controller outside the
propagation, at the tube's midpoint, with the command held for the tick
(immrax_validator.py:934-957).  The interval arithmetic therefore never sees
the feedback, and the reachable tube is the tube of an *open-loop* plant
driven by a pre-recorded command.  The real robot is a contracting closed
loop, and its reachable set is several times narrower: on the s-curve the
sampled reachable set is 0.41 cm against a tube of 5.08 cm.

This script measures the alternative: one 20 ms tick of the *closed loop* as
a discrete map, embedded with the inclusion functions immrax offers, and
propagated once per tick.

    x_{k+1} = F_k(x_k, w) = Phi(x_k, N(x_k, u_ref_k, w), w; T)

The Jacobian of that map carries the feedback term

    dF/dx = dPhi/dx + (dPhi/du) (dN/dx)

which is what contracts the tube.  The open-loop propagation has no such
term.

The tick map is a discrete immrax System, so the embedding's discrete branch
does the work: E(t, x, u_ref, w) = i2ut(F(interval(t), ut2i(x), u_ref, w)).

Three things are measured, for every inclusion function:

  * the tube (the half-diagonal the envelope gate sees),
  * the tracking error (the tube's midpoint against the plan), and
  * whether the tube still contains the set the robot can actually reach -
    the Monte-Carlo check from exp19_envelope_scatter.py, which runs the
    shipped plant and the shipped controller on points.

A tighter tube that drops real poses is worse than a wide one, so the
containment check is part of the measurement, not a follow-up.
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

import immrax  # noqa: E402
import immrax_validator as IV  # noqa: E402
import exp19_envelope_scatter as E19  # noqa: E402
from jax_robot_model import JAXRobotConfig  # noqa: E402
from robot_model import RobotConfig  # noqa: E402

CFG = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                     encoding="utf-8"))
ROBOT = RobotConfig(CFG)
JAX_CFG = JAXRobotConfig(ROBOT)
UNC = ROBOT.uncertainties

GAIN = (0.99, 1.01)
ENVELOPE_LIMIT_M = 0.02


def straight(length=1.0, v=0.5, n=10):
    out = []
    dt = (length / v) / n
    for k in range(n + 1):
        out.append({
            "t": k * dt, "x": v * k * dt, "y": 0.0, "heading": 0.0,
            "vl": v, "vr": v, "omega": 0.0,
            "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
        })
    return out


def arc(radius=0.5, angle=1.2, v=0.5, n=20):
    out = []
    length = radius * angle
    dt = (length / v) / n
    for k in range(n + 1):
        s = length * k / n
        th = s / radius
        out.append({
            "t": k * dt, "x": radius * math.sin(th),
            "y": radius * (1.0 - math.cos(th)), "heading": th,
            "vl": v - (v / radius) * 0.0965 / 2.0,
            "vr": v + (v / radius) * 0.0965 / 2.0,
            "omega": v / radius,
            "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# The closed-loop tick map
# ---------------------------------------------------------------------------
class ClosedLoopTickMap(immrax.system.System):
    """One control tick of the closed loop, as a discrete map.

    The controller is *inside* the map, so the map's Jacobian carries the
    feedback.  The tick duration rides in the last slot of the reference
    vector: it changes from tick to tick (the last tick of a sample interval
    is shorter), and a value that changes must be a traced argument, or the
    inclusion function would be rebuilt for every tick.
    """

    def __init__(self, plant, controller, substeps=4):
        self.evolution = "discrete"
        self.plant = plant
        self.controller = controller
        self.substeps = substeps
        self.xlen = plant.xlen

    def f(self, t, x, u_ref, w):
        dt = u_ref[6]
        # normalize=False: the angle normalisation is a modulo, and immrax's
        # interval arithmetic has no inclusion rule for it.  The validator
        # unrolls the plan's headings before it gets here, so the difference
        # is already the short way and the wrap is not needed - this is the
        # same reason the "interval" feedback mode uses the raw headings.
        u_cmd = self.controller(0.0, x[:3], u_ref[:6], w, normalize=False,
                                quantize=False)
        h = dt / self.substeps
        x_curr = x
        for _ in range(self.substeps):
            x_curr = x_curr + self.plant.f(0.0, x_curr, u_cmd, w) * h
        return x_curr


def tick_propagate(samples, emb, plant, controller, w_interval,
                   control_dt=0.02, ramsete_b=2.0, backlash=0.0):
    """Propagate the tube one tick at a time through the closed-loop map."""
    cfg = JAX_CFG
    # The controller inside the map uses the raw headings (normalize=False),
    # so the plan's headings have to be unrolled first - the same thing the
    # shipped validator does before it propagates.
    samples = IV.unroll_trajectory_headings(samples)
    start = jnp.array([samples[0]['x'], samples[0]['y'],
                        samples[0]['heading']])
    x_flat = immrax.i2ut(immrax.Interval(start - 0.001, start + 0.001))
    envelope = []
    tracking = 0.0

    for i in range(len(samples) - 1):
        s, s_next = samples[i], samples[i + 1]
        dt = s_next['t'] - s['t']
        t = s['t']
        while t < s['t'] + dt - 1e-12:
            dt_ctrl = min(control_dt, s['t'] + dt - t)
            frac = (t - s['t']) / dt
            ref = IV._interpolated_reference(s, s_next, dt, cfg.track_width,
                                             frac, ramsete_b)
            u_ref = jnp.concatenate([jnp.array(ref), jnp.array([dt_ctrl])])
            x_flat = emb.f(0.0, x_flat, u_ref, w_interval)
            t += dt_ctrl

        n = plant.xlen
        lo, hi = x_flat[:n], x_flat[n:]
        spread = 0.5 * backlash * max(0.0, s_next['t'] - samples[0]['t'])
        env = {"x_min": float(lo[0]) - spread, "x_max": float(hi[0]) + spread,
               "y_min": float(lo[1]) - spread, "y_max": float(hi[1]) + spread}
        mid_x = float((lo[0] + hi[0]) / 2.0)
        mid_y = float((lo[1] + hi[1]) / 2.0)
        tracking = max(tracking, math.hypot(mid_x - s_next['x'],
                                             mid_y - s_next['y']))
        envelope.append(env)

    env = {"x_min": float(start[0]) - 0.001, "x_max": float(start[0]) + 0.001,
           "y_min": float(start[1]) - 0.001, "y_max": float(start[1]) + 0.001}
    envelope.insert(0, env)
    return {"envelope": envelope, "max_tracking_error_m": tracking,
            "max_envelope_radius_m": max(
                math.hypot((e["x_max"] - e["x_min"]) / 2.0,
                           (e["y_max"] - e["y_min"]) / 2.0)
                for e in envelope)}


def w_interval(gain):
    return immrax.Interval(
        jnp.array([UNC["cof"][0], UNC["torque_headroom"][0],
                   UNC["gear_backlash_m"][0], UNC["track_width_range"][0],
                   UNC["wheel_radius_range"][0], UNC["inertia_kg_m2"][0],
                   gain[0]]),
        jnp.array([UNC["cof"][1], UNC["torque_headroom"][1],
                   UNC["gear_backlash_m"][1], UNC["track_width_range"][1],
                   UNC["wheel_radius_range"][1], UNC["inertia_kg_m2"][1],
                   gain[1]]))


def containment(cloud, envelope, samples):
    """Is every sampled pose inside the tube at every sample?

    ``cloud`` is one array of (x, y) poses per sample.  The samples after a
    divergence are skipped: the validator freezes the tube there and marks
    the samples "unknown", so their box is stale rather than an enclosure.
    The divergence itself is reported separately.
    """
    worst = 0.0
    first_bad = None
    for k in range(min(len(envelope), len(cloud))):
        if envelope[k].get("unknown"):
            continue
        spread = 0.5 * UNC["gear_backlash_m"][1] * max(
            0.0, samples[k]['t'] - samples[0]['t'])
        lo_x = envelope[k]["x_min"] - spread
        hi_x = envelope[k]["x_max"] + spread
        lo_y = envelope[k]["y_min"] - spread
        hi_y = envelope[k]["y_max"] + spread
        for px, py in cloud[k]:
            over = max(lo_x - px, px - hi_x, lo_y - py, py - hi_y)
            if over > worst:
                worst = over
            if over > 1e-9 and first_bad is None:
                first_bad = k
    return (first_bad is None), first_bad, worst


def mc_cloud(samples, plant, gain, n_draws=120, seed=0):
    """The reachable set, sampled: the tree's own point simulation."""
    E19.GAIN = gain
    plan_start = jnp.array([samples[0]['x'], samples[0]['y'],
                             samples[0]['heading']])
    cloud = [[] for _ in samples]
    for ws, starts in E19.build_datasets(n_draws, seed):
        starts = starts + plan_start
        poses = np.asarray(E19.point_sim(samples, ws, starts, plant, JAX_CFG,
                                         with_belief=False))
        for k in range(len(samples)):
            cloud[k].append(poses[:, k, :])
    return [np.concatenate(c, axis=0) for c in cloud]


def cloud_radius(cloud):
    worst = 0.0
    for pts in cloud:
        worst = max(worst, math.hypot((pts[:, 0].max() - pts[:, 0].min()) / 2.0,
                                      (pts[:, 1].max() - pts[:, 1].min()) / 2.0))
    return worst


def main():
    controller = IV.RamseteController(2.0, 0.7,
                                     track_width=JAX_CFG.track_width)
    plant = IV.RobotOpenLoop(JAX_CFG)

    cases = [("straight 1 m", straight(length=1.0)),
             ("arc R0.5 1.2 rad", arc())]

    # The curved mission that fails today: the shipped s-curve trajectory,
    # with the actuator gain band that a calibrated model brings.
    traj_dir = os.environ.get("OVERBOARD_TRAJ")
    if traj_dir:
        path = os.path.join(traj_dir, "example_s_curve.traj")
        if os.path.exists(path):
            doc = json.load(open(path, encoding="utf-8"))
            cases.append(("s_curve mission", doc["trajectory"]["samples"]))

    for path_name, samples in cases:
        print("=" * 100)
        print(f"{path_name} - the shipped validator against the tick map")
        print("=" * 100)
        cloud = mc_cloud(samples, plant, GAIN)
        print(f"the set the robot can actually reach: "
              f"{cloud_radius(cloud)*1000:.2f} mm")
        print()
        print(f"{'propagation':>22} {'incl':>6} | {'tube mm':>9} {'trk mm':>8} "
              f"{'sec':>6} | {'gate':>8} {'contains':>9} {'over mm':>9}")

        for kind in ("shipped", "tick"):
            for incl, factory in (("nat", immrax.natemb),
                                   ("jac", immrax.jacemb),
                                   ("mjac", immrax.mjacemb)):
                t0 = time.time()
                if kind == "shipped":
                    val = IV.ImmraxValidator(JAX_CFG)
                    val.digital_twin_gain_range = GAIN
                    val.plant_emb_sys = factory(val.plant)
                    rep = val.validate_trajectory(
                        samples, UNC["cof"], UNC["torque_headroom"],
                        UNC["gear_backlash_m"], UNC["track_width_range"],
                        UNC["wheel_radius_range"], UNC["inertia_kg_m2"])
                    envelope = rep["reachability"]["envelope"]
                    tube = rep["reachability"]["max_envelope_radius_m"]
                    trk = rep["max_tracking_error_m"]
                    diverged = bool(rep["reachability"]["diverged"])
                else:
                    tick = ClosedLoopTickMap(plant, controller,
                                             IV.PLANT_SUBSTEPS)
                    emb = factory(tick)
                    rep = tick_propagate(samples, emb, plant, controller,
                                         w_interval(GAIN),
                                         backlash=UNC["gear_backlash_m"][1])
                    envelope = rep["envelope"]
                    tube = rep["max_envelope_radius_m"]
                    trk = rep["max_tracking_error_m"]
                    diverged = not math.isfinite(tube)
                elapsed = time.time() - t0
                inside, first_bad, worst = containment(cloud, envelope, samples)
                if diverged:
                    gate = "diverged"
                elif tube <= ENVELOPE_LIMIT_M:
                    gate = "pass"
                else:
                    gate = "FAIL"
                print(f"{kind:>22} {incl:>6} | {tube*1000:>9.2f} "
                      f"{trk*1000:>8.3f} {elapsed:>6.1f} | {gate:>8} "
                      f"{('yes' if inside else f'NO@{first_bad}'):>9} "
                      f"{worst*1000:>9.3f}")
        print()


if __name__ == "__main__":
    main()
