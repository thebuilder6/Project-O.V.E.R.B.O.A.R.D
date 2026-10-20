"""RQ-A experiment 19: the empirical tube against the interval box.

The reachability envelope is an over-approximation, and until now nothing in
the repository said by how much. Two numbers settle it:

  1. the width of the box the validator reports, and
  2. the width of the set the same plant and the same controller can
     actually reach.

The difference is the over-approximation: the part a tighter inclusion
function (jacif / mjacif) can take back. The part that is left over is
either the physics or the discretisation floor.

The second set is built by running the *shipped* plant and the *shipped*
controller on points instead of on intervals, over the declared uncertainty.
The initial conditions come from ``immrax.utils.gen_ics``, which draws
uniform samples inside the box the validator starts from; the uncertainty is
sampled the same way. The corners of both boxes are included explicitly,
because a uniform draw almost never lands on an extreme and the extremes are
what the envelope has to contain.

The check is one-sided and is reported as such: a Monte-Carlo cloud inside
the box does not *prove* the box is sound, it only fails to disprove it. The
soundness argument is the inclusion function's own (the mean-value form with
the interval Jacobian is a valid enclosure); this script is the empirical
check on top of it, and the corner cases are the part that comes closest to
a certificate.

Both plants are measured, because they fail differently:

  * ``RobotOpenLoop`` (3 states) reads the turn rate from the *command*, so
    the Jacobian sees that the turn rate does not depend on the gain at all
    and the phantom turn rate disappears.
  * ``DigitalTwinOpenLoop`` (5 states) reads the turn rate from the lagged
    wheel speeds, which are two perfectly correlated numbers that the
    interval box cannot express as one. No inclusion function can recover
    that, so the Jacobian does not help there.

Usage (from the repository root):

    python docs/research/rq_a/exp19_envelope_scatter.py
    python docs/research/rq_a/exp19_envelope_scatter.py --out some/dir
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import jax
import jax.numpy as jnp

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)

from immrax import Interval  # noqa: E402
from immrax.utils import gen_ics, check_containment  # noqa: E402

import immrax_validator as IV  # noqa: E402
from digital_twin import DigitalTwinConfig  # noqa: E402
from jax_robot_model import JAXRobotConfig  # noqa: E402
from robot_model import RobotConfig  # noqa: E402

CFG = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                     encoding="utf-8"))
ROBOT = RobotConfig(CFG)
JAX_CFG = JAXRobotConfig(ROBOT)

# The declared uncertainty, exactly as the validator receives it.
COF = (1.24, 1.51)
TORQUE = (0.75, 0.9)
BACKLASH = (0.0004, 0.0005)
TRACK = (0.0964, 0.0966)
RADIUS = (0.0275, 0.0285)
INERTIA = (0.0008, 0.0011)
GAIN = (0.99, 1.01)          # the calibrated model: +/-1 %

# The start pose uncertainty the validator uses: 1 mm on each of x, y, theta.
START_HALF = 0.001

INCLUSIONS = ("nat", "jac", "mjac")


def straight(length=1.0, v=0.5, n=10):
    """A straight path, the same one exp10_gain_band.py uses."""
    out = []
    dt = (length / v) / n
    for k in range(n + 1):
        out.append({
            "t": k * dt, "x": v * k * dt, "y": 0.0, "heading": 0.0,
            "vl": v, "vr": v, "omega": 0.0,
            "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
        })
    return out


def make_twin():
    """A calibrated model with a +/-1 % gain band, like the studio's."""
    return DigitalTwinConfig.from_dict({"actuator": {"straight": {
        "calibrated": True, "gain": 1.0,
        "gain_min": 0.99, "gain_max": 1.01,
        "tau_s": 0.02, "delay_s": 0.0,
    }}})


# -----------------------------------------------------------------------------
# The point simulation: the shipped plant and controller on points
# -----------------------------------------------------------------------------
def point_sim(samples, ws, starts, plant, cfg, ramsete_b=2.0,
              control_dt=0.02, with_belief=False):
    """Run the plant and the controller on points instead of on a tube.

    This is the same model the validator runs: the same reference, the same
    20 ms sample-and-hold, the same plant substeps. With ``with_belief`` the
    controller is fed the robot's own pose estimate, as the digital twin
    does, so the two agree on which state the controller reacts to.

    ``ws`` is a (B, 7) batch of uncertainty vectors and ``starts`` a (B, 3)
    batch of start poses. The plan is NOT shifted: the validator propagates
    from a start box around the plan's first sample and then follows the
    plan's own reference, so a run that moves the plan would be measuring a
    different model.

    Returns an array of (B, n_samples, 2) poses.
    """
    ctrl = IV.RamseteController(ramsete_b, 0.7, track_width=cfg.track_width)
    plant_f = jax.vmap(plant.f, in_axes=(None, 0, 0, 0))
    ctrl_f = jax.vmap(lambda t, x, u, w: ctrl(t, x, u, w, normalize=True,
                                               quantize=False),
                      in_axes=(None, 0, None, 0))

    x = jnp.concatenate([starts, jnp.zeros((starts.shape[0],
                                            plant.xlen - 3))], axis=1)
    belief = starts
    out = [jnp.stack([x[:, 0], x[:, 1]], axis=1)]

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
            seen = belief if with_belief else x[:, :3]
            u_cmd = ctrl_f(0.0, seen, u_ref, ws)
            for _ in range(IV.PLANT_SUBSTEPS):
                # The belief is advanced before the plant step, from the
                # state as it is now, exactly as the validator does it.
                if with_belief:
                    belief = _advance_belief(belief, x, plant, cfg,
                                             dt_ctrl / IV.PLANT_SUBSTEPS)
                dx = plant_f(0.0, x, u_cmd, ws)
                x = x + dx * (dt_ctrl / IV.PLANT_SUBSTEPS)
            t += dt_ctrl
        out.append(jnp.stack([x[:, 0], x[:, 1]], axis=1))

    return jnp.stack(out, axis=1)


def _advance_belief(belief, x, plant, cfg, dt):
    """The robot's own pose estimate, advanced from the wheel motion."""
    if plant.xlen <= 3:
        return belief
    if plant.wheel_state == "mean_diff":
        v = x[:, 3]
        w = x[:, 4] / cfg.track_width
    else:
        v = (x[:, 3] + x[:, 4]) / 2.0
        w = (x[:, 4] - x[:, 3]) / cfg.track_width
    heading = belief[:, 2] + w * dt
    bx = belief[:, 0] + v * jnp.cos(heading) * dt
    by = belief[:, 1] + v * jnp.sin(heading) * dt
    return jnp.stack([bx, by, heading], axis=1)


# -----------------------------------------------------------------------------
# The scenarios: what the robot can actually reach
# -----------------------------------------------------------------------------
def start_interval():
    """The box the validator starts the tube from."""
    return Interval(
        jnp.array([-START_HALF, -START_HALF, -START_HALF]),
        jnp.array([START_HALF, START_HALF, START_HALF]))


def w_interval():
    """The uncertainty box, exactly as the validator builds it."""
    return Interval(
        jnp.array([COF[0], TORQUE[0], BACKLASH[0], TRACK[0], RADIUS[0],
                   INERTIA[0], GAIN[0]]),
        jnp.array([COF[1], TORQUE[1], BACKLASH[1], TRACK[1], RADIUS[1],
                   INERTIA[1], GAIN[1]]))


def nominal_w(track=TRACK[0], gain=GAIN[0]):
    """One point in the uncertainty box, in the validator's order."""
    return jnp.array([COF[0], TORQUE[0], BACKLASH[0], track, RADIUS[0],
                      INERTIA[0], gain])


def _corners(iv, n):
    """The 2^n corners of an interval box."""
    out = []
    for mask in range(2 ** n):
        out.append(jnp.array([iv.upper[i] if (mask >> i) & 1 else iv.lower[i]
                              for i in range(n)]))
    return jnp.stack(out)


def build_datasets(n_draws, seed):
    """The two batches of point simulations.

    Batch 1 - the uncertainty: the nominal start pose, the uncertainty drawn
    from its box, plus every corner of the box.

    Batch 2 - the start pose: the nominal uncertainty, the start pose drawn
    from its box, plus every corner of that box.

    Only the track width and the gain reach the plant; the friction, the
    torque headroom, the wheel radius and the inertia are consumed by the
    motor audit and never touch the motion. The backlash is a position
    dead-band, carried in the reachable set rather than in the motion.
    """
    w_iv = w_interval()
    ics = start_interval()

    w_pts = gen_ics(w_iv, n_draws, key=jax.random.key(seed))
    w_all = jnp.concatenate([w_pts, _corners(w_iv, 7)])
    w_all = w_all.at[:, 0].set(COF[0]).at[:, 1].set(TORQUE[0])
    w_all = w_all.at[:, 2].set(BACKLASH[0]).at[:, 4].set(RADIUS[0])
    w_all = w_all.at[:, 5].set(INERTIA[0])
    start_nominal = jnp.zeros((w_all.shape[0], 3))

    ics_pts = gen_ics(ics, n_draws, key=jax.random.key(seed + 1))
    ics_all = jnp.concatenate([ics_pts, _corners(ics, 3)])
    w_fixed = jnp.repeat(nominal_w()[None, :], ics_all.shape[0], axis=0)

    return (w_all, start_nominal), (w_fixed, ics_all)


def run_case(samples, inclusion, plant_kind, n_draws=200, seed=0,
             wheel_state=None):
    """Validate with one inclusion function and sample the reachable set.

    Returns (report, cloud) where cloud is a list (one per sample) of arrays
    of (x, y) points.
    """
    twin = make_twin() if plant_kind == "twin" else None
    val = IV.ImmraxValidator(JAX_CFG, digital_twin=twin,
                              plant_inclusion=inclusion,
                              controller_inclusion=inclusion,
                              wheel_state=wheel_state)
    if twin is None:
        # The gain band is the seventh uncertainty even without a twin.
        val.digital_twin_gain_range = GAIN
    plant = val.plant

    report = val.validate_trajectory(
        samples, COF, TORQUE, BACKLASH, TRACK, RADIUS, INERTIA)

    plan_start = jnp.array([samples[0]['x'], samples[0]['y'],
                             samples[0]['heading']])
    cloud = [[] for _ in samples]
    for ws, starts in build_datasets(n_draws, seed):
        starts = starts + plan_start
        poses = point_sim(samples, ws, starts, plant, JAX_CFG,
                          with_belief=(plant_kind == "twin"))
        poses = np.asarray(poses)
        for k in range(len(samples)):
            cloud[k].append(poses[:, k, :])

    cloud = [np.concatenate(c, axis=0) for c in cloud]
    return report, cloud


def containment(report, cloud, samples):
    """Is every sampled pose inside the reported box at every sample?

    Returns (all_inside, first_bad_sample, worst_overshoot_m).
    """
    env = report["reachability"]["envelope"]
    spread_half = 0.5 * BACKLASH[1]
    worst = 0.0
    first_bad = None
    for k in range(min(len(env), len(cloud))):
        if env[k].get("unknown"):
            continue
        spread = spread_half * max(0.0, samples[k]['t'] - samples[0]['t'])
        lo_x = env[k]["x_min"] - spread
        hi_x = env[k]["x_max"] + spread
        lo_y = env[k]["y_min"] - spread
        hi_y = env[k]["y_max"] + spread
        for px, py in cloud[k]:
            over = max(lo_x - px, px - hi_x, lo_y - py, py - hi_y)
            if over > worst:
                worst = over
            if over > 1e-9 and first_bad is None:
                first_bad = k
    return (first_bad is None), first_bad, worst


def cloud_box(points):
    """The bounding box of the empirical cloud at one sample."""
    pts = np.asarray(points)
    return (pts[:, 0].min(), pts[:, 0].max(),
            pts[:, 1].min(), pts[:, 1].max())


def box_containment(report, cloud, samples):
    """The box-level verdict from immrax.utils.check_containment."""
    env = report["reachability"]["envelope"]
    verdicts = []
    for k in range(min(len(env), len(cloud))):
        if env[k].get("unknown"):
            continue
        spread = 0.5 * BACKLASH[1] * max(0.0, samples[k]['t'] - samples[0]['t'])
        mc = cloud_box(cloud[k])
        mc_iv = Interval(jnp.array([mc[0] - spread, mc[2] - spread]),
                         jnp.array([mc[1] + spread, mc[3] + spread]))
        tube_iv = Interval(jnp.array([env[k]["x_min"], env[k]["y_min"]]),
                           jnp.array([env[k]["x_max"], env[k]["y_max"]]))
        verdicts.append(int(check_containment(mc_iv, tube_iv)))
    if not verdicts:
        return 0
    return min(verdicts)


def plot_case(samples, report, cloud, path, title, sample_indices):
    """The empirical scatter against the interval box.

    Two panels: the whole path, and a zoom on the last box, because the
    reachable set is millimetres wide and the path is metres long.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    env = report["reachability"]["envelope"]
    reach = report["reachability"]
    drawn = [i for i in sample_indices
             if i < len(env) and not env[i].get("unknown")]
    zoom = drawn[-1] if drawn else None

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, is_zoom in ((axes[0], False), (axes[1], True)):
        ax.plot([s['x'] for s in samples], [s['y'] for s in samples],
                'k-', linewidth=1.2, alpha=0.55, label='planned path', zorder=3)
        first = True
        for i in drawn:
            e = env[i]
            ax.add_patch(Rectangle((e["x_min"], e["y_min"]),
                                   e["x_max"] - e["x_min"],
                                   e["y_max"] - e["y_min"],
                                   facecolor='red', alpha=0.25, edgecolor='red',
                                   linewidth=1.2, zorder=1,
                                   label='interval box' if first else None))
            pts = np.asarray(cloud[i])
            ax.scatter(pts[:, 0], pts[:, 1], s=22, alpha=0.8, color='tab:blue',
                       zorder=4,
                       label='reachable set (sampled)' if first else None)
            first = False
        if is_zoom and zoom is not None:
            e = env[zoom]
            cx = 0.5 * (e["x_min"] + e["x_max"])
            cy = 0.5 * (e["y_min"] + e["y_max"])
            half = max(e["x_max"] - e["x_min"], e["y_max"] - e["y_min"], 1e-4) * 1.4
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_title(f"sample {zoom}: the box against the reachable set",
                         fontsize=11)
        else:
            ax.set_title("the whole path", fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    fig.suptitle(
        f"{title} - box {report['max_envelope_radius_m']*1000:.2f} mm"
        + ("  (truncated)" if reach['envelope_truncated'] else ""),
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="directory for the figure (default: a temp dir)")
    parser.add_argument("--draws", type=int, default=200,
                        help="Monte-Carlo draws per box")
    parser.add_argument("--mission", default=None,
                        help="a solved trajectory to use instead of the "
                             "1 m straight (measurements/traj/<name>.traj)")
    args = parser.parse_args()

    out_dir = args.out or tempfile.mkdtemp(prefix="overboard-envelope-")
    os.makedirs(out_dir, exist_ok=True)

    if args.mission:
        path = os.path.join(PINNED, "measurements", "traj",
                            f"{args.mission}.traj")
        doc = json.load(open(path, encoding="utf-8"))
        samples = doc["trajectory"]["samples"]
        what = f"the solved {args.mission} trajectory"
    else:
        samples = straight(length=1.0)
        what = "a 1 m straight at 0.5 m/s"
    print("=" * 96)
    print("RQ-A experiment 19 - the empirical tube against the interval box")
    print("=" * 96)
    print(f"{what}, gain band +/-1 %, "
          f"{args.draws} draws per box, corners included")
    print()

    cases = (
        ("plain", "wheels", "RobotOpenLoop (3 states)", INCLUSIONS),
        ("twin", "wheels", "DigitalTwinOpenLoop, wheel speeds", INCLUSIONS),
        ("twin", "mean_diff", "DigitalTwinOpenLoop, mean and difference",
         INCLUSIONS),
    )
    for plant_kind, wheel_state, label, inclusions in cases:
        print("-" * 96)
        print(label)
        print("-" * 96)
        print(f"{'incl':>6} {'box mm':>9} {'trk mm':>8} {'trunc':>6} "
              f"{'cloud mm':>10} {'over mm':>9} {'in box':>7} {'contain':>9} "
              f"{'sec':>6}")
        for inclusion in inclusions:
            t0 = time.time()
            report, cloud = run_case(samples, inclusion, plant_kind,
                                     n_draws=args.draws,
                                     wheel_state=wheel_state)
            elapsed = time.time() - t0
            all_in, first_bad, worst = containment(report, cloud, samples)
            verdict = box_containment(report, cloud, samples)
            reach = report["reachability"]
            cloud_r = 0.0
            for k in range(len(cloud)):
                bx = cloud_box(cloud[k])
                r = math.sqrt(((bx[1] - bx[0]) / 2.0) ** 2
                              + ((bx[3] - bx[2]) / 2.0) ** 2)
                cloud_r = max(cloud_r, r)
            print(f"{inclusion:>6} "
                  f"{report['max_envelope_radius_m']*1000:>9.2f} "
                  f"{report['max_tracking_error_m']*1000:>8.3f} "
                  f"{str(reach['envelope_truncated']):>6} "
                  f"{cloud_r*1000:>10.2f} "
                  f"{(worst*1000 if worst > 0 else 0.0):>9.2f} "
                  f"{('yes' if all_in else f'NO@{first_bad}'):>7} "
                  f"{verdict:>9} {elapsed:>6.1f}")
            name = f"envelope_{plant_kind}_{inclusion}.png"
            if plant_kind == "twin":
                name = f"envelope_{plant_kind}_{wheel_state}_{inclusion}.png"
            path = os.path.join(out_dir, name)
            idx = [i for i in (3, 6, 9) if i < len(samples)]
            plot_case(samples, report, cloud, path,
                      f"{label} - {inclusion} inclusion", idx)
            print(f"       figure: {path}")
        print()
        print("box mm   - the half-diagonal the envelope gate sees")
        print("trunc    - the propagation was cut off, so box mm is a lower")
        print("           bound and a property of the 50 mm guard")
        print("cloud mm - the half-diagonal of the set the robot can reach")
        print("over mm  - how far the sampled set sticks out of the box")
        print("in box   - every sampled pose is inside the box at every sample")
        print("contain  - immrax.utils.check_containment on the two boxes")
        print()

    print(f"figures in {out_dir}")


if __name__ == "__main__":
    main()
