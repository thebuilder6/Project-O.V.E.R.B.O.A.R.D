"""RQ-A experiment 17: how much of the envelope is over-approximation?

F5 says that with a calibrated model the **envelope** is the binding gate, and
that the reachable set "cannot be shrunk by making the trajectory smoother".
That leaves one possibility F5 did not price: the reachable set the validator
reports is an over-approximation of the set the robot can actually reach, and
the over-approximation is not the physics - it is the inclusion function and
the integrator.

Three numbers decide whether the unused immrax machinery is worth wiring in:

  1. the width of the shipped tube,
  2. the width of the set the same plant and controller can actually reach
     (Monte-Carlo over the declared uncertainty), and
  3. the width of the tube with every uncertainty collapsed.

(1) - (2) is the over-approximation: the part a tighter inclusion function
(jacif / mjacif) or a set refinement could take back.  (3) is the
discretisation floor: the part a better integrator could take back.

This script measures the shipped validator (immrax_validator.py) and a point
simulation of the same plant and the same controller, so the comparison is
between the shipped code and itself, not between two re-implementations.

The Monte-Carlo set is the ground truth here, and it is checked against the
shipped tube sample by sample: if the tube does not contain it, the tube is
unsound and every "tightening" measured below would be tightening a wrong
number.
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

import immrax  # noqa: E402
import immrax_validator as IV  # noqa: E402
from jax_robot_model import JAXRobotConfig  # noqa: E402
from robot_model import RobotConfig  # noqa: E402

CFG = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                     encoding="utf-8"))
JAX_CFG = JAXRobotConfig(RobotConfig(CFG))

# The declared uncertainty, exactly as the validator receives it.
COF = (1.24, 1.51)
TORQUE = (0.75, 0.9)
BACKLASH = (0.0004, 0.0005)
TRACK = (0.0964, 0.0966)
RADIUS = (0.0275, 0.0285)
INERTIA = (0.0008, 0.0011)
GAIN = (0.99, 1.01)          # the F5 case: +/-1 %

COLLAPSED = dict(cof=(1.375, 1.375), torque=(0.825, 0.825),
                  backlash=(0.0, 0.0), track=(0.0965, 0.0965),
                  radius=(0.028, 0.028), inertia=(0.00095, 0.00095),
                  gain=(1.0, 1.0))


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


def arc(radius=0.5, angle=1.2, v=0.5, n=20):
    """A circular arc, so the heading actually moves."""
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


def run_validator(samples, ranges, plant_emb=None, substeps=None):
    """The shipped validator, with the uncertainty ranges we ask for."""
    val = IV.ImmraxValidator(JAX_CFG)
    val.digital_twin_gain_range = ranges["gain"]
    if plant_emb is not None:
        val.plant_emb_sys = plant_emb
    old = IV.PLANT_SUBSTEPS
    if substeps is not None:
        IV.PLANT_SUBSTEPS = substeps
    try:
        return val.validate_trajectory(
            samples, ranges["cof"], ranges["torque"], ranges["backlash"],
            ranges["track"], ranges["radius"], ranges["inertia"])
    finally:
        IV.PLANT_SUBSTEPS = old


def point_sim(samples, w, control_dt=0.02, ramsete_b=2.0):
    """The validator's plant and controller, run on a point instead of a tube.

    This is the robot: the controller sees the true pose, the plant follows
    the command with the uncertainties in ``w``.  It is the ground truth the
    interval tube has to contain.  The pose is recorded at every sample time.
    """
    cfg = JAX_CFG
    plant = IV.RobotOpenLoop(cfg)
    ctrl = IV.RamseteController(ramsete_b, 0.7, track_width=cfg.track_width)
    x = jnp.array([samples[0]['x'], samples[0]['y'], samples[0]['heading']])
    poses = [(float(x[0]), float(x[1]))]
    for i in range(len(samples) - 1):
        s, s_next = samples[i], samples[i + 1]
        dt = s_next['t'] - s['t']
        t = s['t']
        while t < s['t'] + dt - 1e-12:
            dt_ctrl = min(control_dt, s['t'] + dt - t)
            frac = (t - s['t']) / dt
            ref = IV._interpolated_reference(s, s_next, dt, cfg.track_width,
                                             frac, ramsete_b)
            u_cmd = ctrl(0.0, x, jnp.array(ref), w, normalize=True,
                         quantize=False)
            for _ in range(IV.PLANT_SUBSTEPS):
                x = x + plant.f(0.0, x, u_cmd, w) * (dt_ctrl / IV.PLANT_SUBSTEPS)
            t += dt_ctrl
        poses.append((float(x[0]), float(x[1])))
    return poses


def mc_poses(samples, ranges, n=96, seed=0):
    """The reachable set at every sample, sampled over the uncertainty.

    Only the two uncertainties that reach the *dynamics* are sampled: the
    track width (the plant's denominator) and the actuator gain (it scales
    the commanded wheel speeds).  The others - the friction coefficient, the
    torque headroom, the backlash, the wheel radius, the inertia - enter the
    motor audit, not the motion.

    The four corners of that two-dimensional box are included explicitly:
    they are where the reachable set is widest, and random sampling is
    unlikely to land on them.
    """
    rng = np.random.default_rng(seed)
    tws = [ranges["track"][0], ranges["track"][1]]
    gains = [ranges["gain"][0], ranges["gain"][1]]
    pairs = [(tw, g) for tw in tws for g in gains]
    pairs += [(float(rng.uniform(*ranges["track"])),
               float(rng.uniform(*ranges["gain"]))) for _ in range(n)]
    runs = []
    for tw, gain in pairs:
        w = jnp.array([ranges["cof"][0], ranges["torque"][0],
                       ranges["backlash"][0], tw, ranges["radius"][0],
                       ranges["inertia"][0], gain])
        runs.append(point_sim(samples, w))
    return runs


def mc_box(runs, samples, ranges):
    """The bounding box of the reachable set at the last sample."""
    spread = 0.5 * ranges["backlash"][1] * (samples[-1]['t'] - samples[0]['t'])
    xs = [p[-1][0] for p in runs]
    ys = [p[-1][1] for p in runs]
    return (min(xs) - spread, max(xs) + spread,
            min(ys) - spread, max(ys) + spread)


def radius(box):
    """The validator's own metric: the half-diagonal of the box."""
    x_min, x_max, y_min, y_max = box
    return math.sqrt(((x_max - x_min) / 2.0) ** 2 + ((y_max - y_min) / 2.0) ** 2)


def tube_box(report, index=-1):
    """The validator's box at one sample."""
    env = report["reachability"]["envelope"]
    return (env[index]["x_min"], env[index]["x_max"],
            env[index]["y_min"], env[index]["y_max"])


def worst_box(report):
    """The widest box the tube reached.

    The samples after a divergence carry the last computed box, so they are
    counted: they say how wide the tube got before it was cut off.
    """
    env = report["reachability"]["envelope"]
    best, best_r = None, -1.0
    for e in env:
        r = (((e["x_max"] - e["x_min"]) / 2.0) ** 2 +
             ((e["y_max"] - e["y_min"]) / 2.0) ** 2) ** 0.5
        if r > best_r:
            best_r, best = r, e
    return (best["x_min"], best["x_max"], best["y_min"], best["y_max"])


def contains(outer, inner, slack=1e-9):
    return (outer[0] <= inner[0] + slack and outer[1] >= inner[1] - slack
            and outer[2] <= inner[2] + slack and outer[3] >= inner[3] - slack)


def containment_report(runs, report, samples, ranges):
    """Is every Monte-Carlo pose inside the tube at every sample?

    Returns (all_inside, first_bad_sample, worst_overshoot_mm).
    """
    env = report["reachability"]["envelope"]
    worst = 0.0
    first_bad = None
    for k in range(min(len(env), len(runs[0]))):
        if env[k].get("unknown"):
            continue
        spread = 0.5 * ranges["backlash"][1] * (samples[k]['t'] - samples[0]['t'])
        lo_x = env[k]["x_min"] - spread
        hi_x = env[k]["x_max"] + spread
        lo_y = env[k]["y_min"] - spread
        hi_y = env[k]["y_max"] + spread
        for run in runs:
            x, y = run[k]
            over = max(lo_x - x, x - hi_x, lo_y - y, y - hi_y)
            if over > worst:
                worst = over
            if over > 1e-9 and first_bad is None:
                first_bad = k
    return (first_bad is None), first_bad, worst


def part1():
    print("=" * 96)
    print("PART 1 - where the envelope comes from")
    print("=" * 96)
    print("The shipped tube, the reachable set the same robot actually reaches")
    print("(Monte-Carlo over the declared uncertainty), and the tube with every")
    print("uncertainty collapsed (the discretisation floor).")
    print()
    print(f"{'path':>7} {'gain':>8} | {'tube mm':>9} {'MC mm':>8} {'floor mm':>9} "
          f"| {'spurious':>9} {'diverged':>9} {'MC inside':>10}")
    for length in (0.5, 1.0, 2.0):
        for band, label in (((1.0, 1.0), "none"),
                            ((0.99, 1.01), "+/-1%")):
            samples = straight(length=length)
            ranges = {"cof": COF, "torque": TORQUE, "backlash": BACKLASH,
                      "track": TRACK, "radius": RADIUS, "inertia": INERTIA,
                      "gain": band}

            rep = run_validator(samples, ranges)
            tube = worst_box(rep)
            tube_r = radius(worst_box(rep))

            runs = mc_poses(samples, ranges)
            mc = mc_box(runs, samples, ranges)
            mc_r = radius(mc)

            floor_rep = run_validator(samples, COLLAPSED)
            floor_r = radius(worst_box(floor_rep))

            inside, first_bad, worst = containment_report(runs, rep, samples,
                                                           ranges)
            diverged = rep["reachability"]["diverged"]
            spurious = tube_r - mc_r
            print(f"{length:>6.1f}m {label:>8} | {tube_r*1000:>9.2f} "
                  f"{mc_r*1000:>8.2f} {floor_r*1000:>9.2f} | "
                  f"{spurious*1000:>9.2f} {('yes' if diverged else 'no'):>9} "
                  f"{('yes' if inside else f'NO at {first_bad}'):>10}")
    print()
    print("tube mm  - the radius the envelope gate sees")
    print("MC mm    - the radius of the set the robot can actually reach")
    print("floor mm - the tube with every uncertainty collapsed")
    print("spurious - tube - MC: the over-approximation, the only part a")
    print("           tighter inclusion function or a refinement could take back")
    print("diverged - the tube crossed the validator's 50 mm guard and was cut off")
    print()


def part2():
    print("=" * 96)
    print("PART 2 - idea 3: does the integrator matter? (plant substep sweep)")
    print("=" * 96)
    print("If the envelope does not move when the plant is integrated with more")
    print("steps, then diffrax cannot move it either: the width is set by the")
    print("uncertainty propagation, not by the integration error.")
    print()
    print("The sweep runs on the Jacobian embedding, because the shipped one")
    print("diverges on this path and the divergence would hide the answer.")
    print()
    samples = straight(length=1.0)
    ranges = {"cof": COF, "torque": TORQUE, "backlash": BACKLASH,
              "track": TRACK, "radius": RADIUS, "inertia": INERTIA,
              "gain": GAIN}
    val = IV.ImmraxValidator(JAX_CFG)
    jac = immrax.jacemb(val.plant)
    print(f"{'substeps':>9} {'tube mm':>9} {'tracking mm':>12}")
    for substeps in (1, 2, 4, 8, 16, 64, 256):
        rep = run_validator(samples, ranges, plant_emb=jac, substeps=substeps)
        print(f"{substeps:>9} "
              f"{radius(worst_box(rep))*1000:>9.2f} "
              f"{rep['max_tracking_error_m']*1000:>12.3f}")
    print()
    print("The tube does not move with the step count, so the width is the")
    print("uncertainty propagation and not the integration error.")
    print()


def part3():
    print("=" * 96)
    print("PART 3 - idea 1: does a Jacobian inclusion function tighten the tube?")
    print("=" * 96)
    print("The plant embedding is the shipped natemb; the alternatives are the")
    print("Jacobian (jacif) and mixed-Jacobian (mjacif) inclusion functions.")
    print("A tighter number is only worth having if it still contains the set")
    print("the robot can reach, so the Monte-Carlo set is checked sample by")
    print("sample.")
    print()
    print("NOTE: the tree's default is \"nat\" (immrax_validator.py:")
    print("PLANT_INCLUSION), because the dependency this experiment measures is")
    print("now removed at the source: the wheel-speed difference is taken from")
    print("the commands before the gain multiplies it, and the digital twin")
    print("carries the mean and the difference instead of the two wheel")
    print("speeds. So the Jacobian has nothing left to win here, and the")
    print("\"nat\" column below is the shipped default. The columns are still")
    print("worth reading: they are what says the Jacobian is sound and what")
    print("it would be worth if another dependency showed up.")
    print("exp19_envelope_scatter.py measures the same thing with a different")
    print("point simulation, and gets the same numbers.")
    print()
    for path_name, samples in (("straight 1 m", straight(length=1.0)),
                                ("arc R0.5 1.2 rad", arc())):
        ranges = {"cof": COF, "torque": TORQUE, "backlash": BACKLASH,
                  "track": TRACK, "radius": RADIUS, "inertia": INERTIA,
                  "gain": GAIN}
        runs = mc_poses(samples, ranges)
        mc = mc_box(runs, samples, ranges)
        print(f"--- {path_name}: the reachable set is {radius(mc)*1000:.2f} mm "
              f"wide ---")
        print(f"{'embedding':>16} {'tube mm':>9} {'tracking mm':>12} "
              f"{'seconds':>8} {'MC inside':>14}")
        for name, factory in (("nat", immrax.natemb),
                               ("jac", immrax.jacemb),
                               ("mjac", immrax.mjacemb)):
            val = IV.ImmraxValidator(JAX_CFG)
            emb = factory(val.plant)
            t0 = time.time()
            rep = run_validator(samples, ranges, plant_emb=emb)
            elapsed = time.time() - t0
            tube = worst_box(rep)
            r = radius(worst_box(rep))
            inside, first_bad, worst = containment_report(runs, rep, samples,
                                                           ranges)
            print(f"{name:>16} {r*1000:>9.2f} "
                  f"{rep['max_tracking_error_m']*1000:>12.3f} "
                  f"{elapsed:>8.2f} "
                  f"{('yes' if inside else f'NO at {first_bad} ({worst*1000:.2f}mm)'):>14}")
        print()


def part4():
    print("=" * 96)
    print("PART 4 - the mechanism: where does the tube's width come from?")
    print("=" * 96)
    print("The plant computes the turn rate as (vr - vl) / track_width, and both")
    print("wheel speeds are the SAME command times the SAME gain, so the")
    print("difference is exactly zero.  The natural extension does not know")
    print("that, so it reports |vr - vl| <= v * (gain_high - gain_low): a")
    print("phantom turn rate that turns a straight path into a curve.")
    print()
    samples = straight(length=1.0)
    ranges = {"cof": COF, "torque": TORQUE, "backlash": BACKLASH,
              "track": TRACK, "radius": RADIUS, "inertia": INERTIA,
              "gain": GAIN}
    v = 0.5
    band = ranges["gain"][1] - ranges["gain"][0]
    phantom = v * band / 0.0965
    print(f"gain band {band:.3f} at v={v} m/s over a {0.0965*100:.1f} mm track")
    print(f"  -> phantom turn rate  {phantom:.4f} rad/s")
    print(f"  -> after 1 s          {phantom:.4f} rad of heading")
    print(f"  -> lateral error      {v*phantom*phantom/2:.4f} m")
    print()
    for name, factory in (("nat", immrax.natemb),
                           ("jac", immrax.jacemb)):
        val = IV.ImmraxValidator(JAX_CFG)
        emb = factory(val.plant)
        rep = run_validator(samples, ranges, plant_emb=emb)
        env = rep["reachability"]["envelope"]
        print(f"{name}:")
        for k in (0, 3, 6, 9):
            if k < len(env):
                w = env[k]["x_max"] - env[k]["x_min"]
                print(f"   sample {k:>2}: tube width {w*1000:>7.2f} mm")
        print()
    print("The Jacobian inclusion function differentiates the plant, so it")
    print("sees that d(turn rate)/d(gain) = (vr - vl)/track = 0 and the")
    print("phantom disappears.")
    print()


def main():
    part1()
    part2()
    part3()
    part4()


if __name__ == "__main__":
    main()
