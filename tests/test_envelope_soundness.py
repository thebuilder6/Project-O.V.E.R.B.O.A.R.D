"""Tests for the reachable tube: the phantom turn rate, and its soundness.

The reachability envelope used to report a width of 54 mm on a 1 m straight
with a +/-1 % actuator-gain band, against a 20 mm gate. The number was not a
width: the propagation had been cut off at the validator's 50 mm guard, and
the same 54.561 mm came out for a 0.5 m path, a 1 m path and a 2 m path,
because a truncation depends on the guard and not on the path.

The cause was a dependency the natural extension cannot see. The plant
computes the turn rate as (vr - vl) / track_width, and both wheel speeds are
the same command times the same gain, so the true turn rate does not depend
on the gain at all. The natural extension reported a turn rate of +/-
(the width of the wheel speed interval) / track where the truth says zero,
and that phantom rate integrated into a lateral drift.

Two things are checked here:

  * the phantom is gone, in both plants, and
  * the box that comes out still CONTAINS the set the robot can actually
    reach. A tighter box that does not contain it would make the gate pass
    a trajectory the robot cannot run, which is worse than a wide one.

The second check is one-sided: sampled points inside the box do not prove the
box is sound, they only fail to disprove it. The soundness argument is the
inclusion function's own; this is the empirical check on top of it. The
corners of the uncertainty box are included, because a uniform draw almost
never lands on an extreme and the extremes are what the box has to hold.
"""

import math
import unittest

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from immrax import Interval
    from immrax_validator import ImmraxValidator, DigitalTwinOpenLoop
    from digital_twin import DigitalTwinConfig
    from jax_robot_model import JAXRobotConfig
    HAS_IMMRAX = True
except ImportError:
    HAS_IMMRAX = False


_TEST_ROBOT = {"robot": {
    "mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
    "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
    "gearing": 1.0, "cof": 1.37,
}}

GAIN = (0.99, 1.01)          # the calibrated model: +/-1 %
TRACK = (0.0964, 0.0966)
BACKLASH = (0.0004, 0.0005)


def _straight(length=1.0, v=0.5, n=10):
    """A straight, constant-speed plan."""
    out = []
    dt = (length / v) / n
    for k in range(n + 1):
        out.append({
            "t": k * dt, "x": v * k * dt, "y": 0.0, "heading": 0.0,
            "vl": v, "vr": v, "omega": 0.0,
            "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
        })
    return out


def _twin():
    """A calibrated model with a gain band and a first-order actuator lag."""
    return DigitalTwinConfig.from_dict({"actuator": {"straight": {
        "calibrated": True, "gain": 1.0,
        "gain_min": GAIN[0], "gain_max": GAIN[1],
        "tau_s": 0.02, "delay_s": 0.0,
    }}})


def _validate(validator, samples):
    return validator.validate_trajectory(
        samples, (1.24, 1.51), (0.75, 0.9), BACKLASH, TRACK,
        (0.0275, 0.0285), (0.0008, 0.0011))


def _point_poses(samples, w, plant, cfg, with_belief=False):
    """The shipped plant and controller on a point, for the soundness check.

    The plant is driven with the same reference, the same 20 ms
    sample-and-hold and the same substeps the validator uses, so the poses
    are the ones the validator's model says the robot reaches.
    """
    import immrax_validator as IV

    ctrl = IV.RamseteController(2.0, 0.7, track_width=cfg.track_width)

    x = jnp.array([samples[0]['x'], samples[0]['y'], samples[0]['heading']]
                  + [0.0] * (plant.xlen - 3))
    belief = x[:3]
    out = [(float(x[0]), float(x[1]))]

    for i in range(len(samples) - 1):
        s, s_next = samples[i], samples[i + 1]
        dt = s_next['t'] - s['t']
        t = s['t']
        while t < s['t'] + dt - 1e-12:
            dt_ctrl = min(0.02, s['t'] + dt - t)
            frac = (t - s['t']) / dt
            ref = IV._interpolated_reference(s, s_next, dt, cfg.track_width,
                                             frac, 2.0)
            seen = belief if with_belief else x[:3]
            u_cmd = ctrl(0.0, seen, jnp.array(ref), w, normalize=True,
                         quantize=False)
            for _ in range(IV.PLANT_SUBSTEPS):
                if with_belief and plant.xlen > 3:
                    if plant.wheel_state == "mean_diff":
                        v = x[3]
                        om = x[4] / cfg.track_width
                    else:
                        v = (x[3] + x[4]) / 2.0
                        om = (x[4] - x[3]) / cfg.track_width
                    heading = belief[2] + om * (dt_ctrl / IV.PLANT_SUBSTEPS)
                    belief = jnp.array([
                        belief[0] + v * jnp.cos(heading) * (dt_ctrl / IV.PLANT_SUBSTEPS),
                        belief[1] + v * jnp.sin(heading) * (dt_ctrl / IV.PLANT_SUBSTEPS),
                        heading,
                    ])
                x = x + plant.f(0.0, x, u_cmd, w) * (dt_ctrl / IV.PLANT_SUBSTEPS)
            t += dt_ctrl
        out.append((float(x[0]), float(x[1])))
    return out


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestNoPhantomTurnRate(unittest.TestCase):
    """A straight path has no turn rate, whatever the gain band does."""

    def _envelope(self, twin, wheel_state=None):
        cfg = JAXRobotConfig(_TEST_ROBOT)
        val = ImmraxValidator(cfg, digital_twin=twin, wheel_state=wheel_state)
        if twin is None:
            # The gain band is the seventh uncertainty, and without a fitted
            # model it is not set. It has to be asked for, or the tube is
            # measured without the one uncertainty that made it explode.
            val.digital_twin_gain_range = GAIN
        report = _validate(val, _straight())
        return report, val
    def test_the_gain_band_does_not_blow_the_tube_up(self):
        """The plain plant: the tube stays well inside the 20 mm gate."""
        report, _ = self._envelope(None)
        self.assertFalse(
            report["reachability"]["diverged"],
            "the propagation was cut off, so the reported width is a "
            "truncation and not a measurement")
        self.assertLess(report["max_envelope_radius_m"], 0.02)

    def test_the_digital_twin_does_not_blow_the_tube_up(self):
        """The actuator-aware plant: the same, with the gain band."""
        report, _ = self._envelope(_twin())
        self.assertFalse(
            report["reachability"]["diverged"],
            "the propagation was cut off, so the reported width is a "
            "truncation and not a measurement")
        self.assertLess(report["max_envelope_radius_m"], 0.02)

    def test_the_truncation_is_flagged(self):
        """A cut-off propagation says so, so no caller can quote it."""
        cfg = JAXRobotConfig(_TEST_ROBOT)
        val = ImmraxValidator(cfg, digital_twin=_twin(),
                              wheel_state="wheels")
        report = _validate(val, _straight())
        # The old wheel-speed layout still has the phantom, and the tube is
        # cut off. What matters is that the report says so.
        self.assertTrue(report["reachability"]["diverged"])
        self.assertTrue(report["reachability"]["envelope_truncated"])

    def test_the_mean_and_difference_layout_is_not_worse(self):
        """Carrying the difference removes what the old layout lost."""
        cfg = JAXRobotConfig(_TEST_ROBOT)
        old = _validate(ImmraxValidator(cfg, digital_twin=_twin(),
                                         wheel_state="wheels"), _straight())
        new = _validate(ImmraxValidator(cfg, digital_twin=_twin(),
                                         wheel_state="mean_diff"), _straight())
        self.assertLess(new["max_envelope_radius_m"],
                        old["max_envelope_radius_m"])
        self.assertFalse(new["reachability"]["diverged"])

    def test_the_report_says_which_inclusion_produced_it(self):
        """A number without its provenance cannot be read."""
        cfg = JAXRobotConfig(_TEST_ROBOT)
        val = ImmraxValidator(cfg, plant_inclusion="jac",
                              controller_inclusion="mjac")
        report = _validate(val, _straight())
        self.assertEqual(report["inclusion"]["plant"], "jac")
        self.assertEqual(report["inclusion"]["controller"], "mjac")

    def test_an_unknown_inclusion_is_refused(self):
        cfg = JAXRobotConfig(_TEST_ROBOT)
        with self.assertRaises(ValueError):
            ImmraxValidator(cfg, plant_inclusion="tightest")
        with self.assertRaises(ValueError):
            ImmraxValidator(cfg, wheel_state="velocities")


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestTheBoxContainsTheReachableSet(unittest.TestCase):
    """The box has to hold the set the robot can actually reach.

    A tighter box that does not hold it would make the envelope gate pass a
    trajectory the robot cannot run, which is worse than a wide one.
    """

    def _check(self, twin, inclusion):
        cfg = JAXRobotConfig(_TEST_ROBOT)
        val = ImmraxValidator(cfg, digital_twin=twin,
                              plant_inclusion=inclusion,
                              controller_inclusion=inclusion)
        # The gain band is the seventh uncertainty. Without a fitted model it
        # is not set, so it has to be asked for explicitly - otherwise the
        # box would not cover the gain the sampled poses are run with.
        val.digital_twin_gain_range = GAIN
        samples = _straight()
        report = _validate(val, samples)

        # Only two of the declared uncertainties reach the plant: the track
        # width, which the turn rate divides by, and the gain, which scales
        # the commanded wheel speeds. The friction, the torque headroom, the
        # wheel radius and the inertia are consumed by the motor audit and
        # never touch the motion, and the backlash is a position dead-band
        # carried in the reachable set. So the corners that matter are the
        # four of the track width against the gain.
        w_iv = Interval(
            jnp.array([1.24, 0.75, BACKLASH[0], TRACK[0], 0.0275, 0.0008,
                        GAIN[0]]),
            jnp.array([1.51, 0.9, BACKLASH[1], TRACK[1], 0.0285, 0.0011,
                       GAIN[1]]))

        def _w(track, gain):
            return jnp.array([1.24, 0.75, BACKLASH[0], track, 0.0275, 0.0008,
                              gain])

        draws = [_w(TRACK[0], GAIN[0]), _w(TRACK[1], GAIN[0]),
                 _w(TRACK[0], GAIN[1]), _w(TRACK[1], GAIN[1])]

        # A handful of random draws from the box, on top of the corners.
        key = jax.random.key(3)
        for _ in range(4):
            keys = jax.random.split(key, 2)
            draws.append(_w(
                float(jax.random.uniform(key=keys[0], shape=(),
                                         minval=TRACK[0], maxval=TRACK[1])),
                float(jax.random.uniform(key=keys[1], shape=(),
                                         minval=GAIN[0], maxval=GAIN[1]))))

        env = report["reachability"]["envelope"]
        spread_half = 0.5 * BACKLASH[1]
        worst = 0.0
        for w in draws:
            poses = _point_poses(samples, w, val.plant, cfg,
                                 with_belief=(twin is not None))
            for k, (px, py) in enumerate(poses):
                if k >= len(env) or env[k].get("unknown"):
                    continue
                spread = spread_half * max(0.0,
                                           samples[k]['t'] - samples[0]['t'])
                over = max(
                    env[k]["x_min"] - spread - px,
                    px - (env[k]["x_max"] + spread),
                    env[k]["y_min"] - spread - py,
                    py - (env[k]["y_max"] + spread),
                )
                worst = max(worst, over)
        return report, worst

    def test_the_natural_box_contains_the_reachable_set(self):
        report, worst = self._check(None, "nat")
        self.assertLessEqual(worst, 1e-9,
                             "a sampled pose is outside the reported box")

    def test_the_jacobian_box_contains_the_reachable_set(self):
        """The tighter box is sound, not merely smaller."""
        report, worst = self._check(None, "jac")
        self.assertLessEqual(worst, 1e-9,
                             "a sampled pose is outside the reported box")
        self.assertLessEqual(report["max_envelope_radius_m"], 0.02)

    def test_the_digital_twin_box_contains_the_reachable_set(self):
        report, worst = self._check(_twin(), "nat")
        self.assertLessEqual(worst, 1e-9,
                             "a sampled pose is outside the reported box")


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestTheTickPropagation(unittest.TestCase):
    """The closed-loop tick map, against the open-loop plant propagation.

    The shipped propagation embeds the plant and evaluates the controller
    outside the propagation, at the tube's midpoint, so the feedback's
    contraction of the error is never in the tube: what it bounds is the
    reachable set of an open-loop plant driven by a pre-recorded command.
    The tick map puts the controller inside the map, so the map's Jacobian
    carries the feedback term, and the tube contracts the way the robot's
    does.
    """

    def _validate(self, propagation, inclusion=None, gain=GAIN):
        cfg = JAXRobotConfig(_TEST_ROBOT)
        kw = {} if inclusion is None else {"tick_inclusion": inclusion}
        val = ImmraxValidator(cfg, propagation=propagation, **kw)
        val.digital_twin_gain_range = gain
        report = _validate(val, _straight())
        return report, val

    def test_the_natural_extension_is_refused(self):
        """With the natural extension the closed-loop map explodes."""
        with self.assertRaises(ValueError):
            self._validate("tick", inclusion="nat")

    def test_an_unknown_propagation_is_refused(self):
        with self.assertRaises(ValueError):
            ImmraxValidator(JAXRobotConfig(_TEST_ROBOT), propagation="magic")

    def test_the_shipped_default_is_the_plant_propagation(self):
        """The default has to be the one that is measured to work."""
        val = ImmraxValidator(JAXRobotConfig(_TEST_ROBOT))
        self.assertEqual(val.propagation, "plant")
        self.assertIsNone(val.tick_map)

    def test_the_tick_tube_is_tighter(self):
        """The feedback is in the tube, so the tube is smaller."""
        plant, _ = self._validate("plant")
        tick, _ = self._validate("tick", inclusion="jac")
        self.assertLess(tick["max_envelope_radius_m"],
                        plant["max_envelope_radius_m"])
        self.assertFalse(tick["reachability"]["diverged"])

    def test_the_tick_tube_holds_the_reachable_set(self):
        """A smaller tube that drops real poses is worse than a wide one.

        The path is a tenth of a metre long: the mixed-Jacobian tick map is
        the slowest thing in the suite, and the question - does the tighter
        box drop a pose the robot can reach - does not need a long path to
        answer.
        """
        cfg = JAXRobotConfig(_TEST_ROBOT)
        samples = _straight(length=0.1)
        for inclusion in ("jac", "mjac"):
            val = ImmraxValidator(cfg, propagation="tick",
                                  tick_inclusion=inclusion)
            val.digital_twin_gain_range = GAIN
            report = _validate(val, samples)

            w_iv = Interval(
                jnp.array([1.24, 0.75, BACKLASH[0], TRACK[0], 0.0275, 0.0008,
                            GAIN[0]]),
                jnp.array([1.51, 0.9, BACKLASH[1], TRACK[1], 0.0285, 0.0011,
                           GAIN[1]]))

            def _w(track, gain):
                return jnp.array([1.24, 0.75, BACKLASH[0], track, 0.0275,
                                  0.0008, gain])

            draws = [_w(TRACK[0], GAIN[0]), _w(TRACK[1], GAIN[0]),
                     _w(TRACK[0], GAIN[1]), _w(TRACK[1], GAIN[1])]
            key = jax.random.key(5)
            for _ in range(4):
                keys = jax.random.split(key, 2)
                draws.append(_w(
                    float(jax.random.uniform(key=keys[0], shape=(),
                                             minval=TRACK[0],
                                             maxval=TRACK[1])),
                    float(jax.random.uniform(key=keys[1], shape=(),
                                             minval=GAIN[0],
                                             maxval=GAIN[1]))))

            env = report["reachability"]["envelope"]
            spread_half = 0.5 * BACKLASH[1]
            worst = 0.0
            for w in draws:
                poses = _point_poses(samples, w, val.plant, cfg)
                for k, (px, py) in enumerate(poses):
                    if k >= len(env) or env[k].get("unknown"):
                        continue
                    spread = spread_half * max(0.0,
                                               samples[k]['t'] - samples[0]['t'])
                    worst = max(worst, max(
                        env[k]["x_min"] - spread - px,
                        px - (env[k]["x_max"] + spread),
                        env[k]["y_min"] - spread - py,
                        py - (env[k]["y_max"] + spread)))
            self.assertLessEqual(worst, 1e-9,
                                 f"a sampled pose is outside the {inclusion} box")

    def test_the_belief_mode_is_refused_when_unknown(self):
        with self.assertRaises(ValueError):
            ImmraxValidator(JAXRobotConfig(_TEST_ROBOT), belief_mode="magic")

    def test_the_estimate_is_the_default(self):
        """The error form looked right and measured worse, so it is not it."""
        val = ImmraxValidator(JAXRobotConfig(_TEST_ROBOT))
        self.assertEqual(val.belief_mode, "estimate")

    def test_both_belief_modes_propagate(self):
        """Both are wired: the error form is a measured alternative."""
        cfg = JAXRobotConfig(_TEST_ROBOT)
        for mode in ("estimate", "error"):
            val = ImmraxValidator(cfg, digital_twin=_twin(),
                                  propagation="tick", tick_inclusion="jac",
                                  belief_mode=mode)
            report = _validate(val, _straight(length=0.5))
            self.assertTrue(np.isfinite(report["max_envelope_radius_m"]))
            self.assertEqual(val.tick_map.belief_mode, mode)

    def test_the_tracking_error_does_not_move(self):
        """The tracking error is the tube's midpoint against the plan, so it
        is decided by the discretisation and not by the uncertainty. The
        propagation may not change it - not by a hair either: the two boxes
        have different midpoints, because the tick map's midpoint is the
        midpoint of the image and the plant's is the midpoint of the step.
        A micrometre of difference cannot move a 5 mm gate.
        """
        plant, _ = self._validate("plant")
        tick, _ = self._validate("tick", inclusion="jac")
        self.assertAlmostEqual(tick["max_tracking_error_m"],
                               plant["max_tracking_error_m"], delta=5e-6)

    def test_the_report_says_which_propagation_produced_it(self):
        """A number without its provenance cannot be read."""
        plant, _ = self._validate("plant")
        tick, _ = self._validate("tick", inclusion="jac")
        self.assertEqual(plant["reachability"]["propagation"], "plant")
        self.assertEqual(tick["reachability"]["propagation"], "tick")

    def test_the_twin_propagates_with_the_belief_in_the_state(self):
        """The controller reacts to the robot's own estimate, not to the
        truth, so the belief has to be inside the map: otherwise the
        feedback term is taken with respect to the wrong variable.
        """
        cfg = JAXRobotConfig(_TEST_ROBOT)
        val = ImmraxValidator(cfg, digital_twin=_twin(), propagation="tick",
                              tick_inclusion="jac")
        samples = _straight(length=0.5)
        report = _validate(val, samples)
        self.assertFalse(report["reachability"]["diverged"])

        def _w(track, gain):
            return jnp.array([1.24, 0.75, BACKLASH[0], track, 0.0275, 0.0008,
                               gain])

        draws = [_w(TRACK[0], GAIN[0]), _w(TRACK[1], GAIN[0]),
                 _w(TRACK[0], GAIN[1]), _w(TRACK[1], GAIN[1])]
        env = report["reachability"]["envelope"]
        spread_half = 0.5 * BACKLASH[1]
        worst = 0.0
        for w in draws:
            poses = _point_poses(samples, w, val.plant, cfg, with_belief=True)
            for k, (px, py) in enumerate(poses):
                if k >= len(env) or env[k].get("unknown"):
                    continue
                spread = spread_half * max(0.0,
                                           samples[k]['t'] - samples[0]['t'])
                worst = max(worst, max(
                    env[k]["x_min"] - spread - px,
                    px - (env[k]["x_max"] + spread),
                    env[k]["y_min"] - spread - py,
                    py - (env[k]["y_max"] + spread)))
        self.assertLessEqual(worst, 1e-9,
                             "a sampled pose is outside the reported box")

    def test_a_fitted_command_delay_rides_in_the_state(self):
        """The robot drives the command it was given a moment ago, so the
        delay belongs inside the map and not beside it.
        """
        cfg = JAXRobotConfig(_TEST_ROBOT)
        twin = DigitalTwinConfig.from_dict({"actuator": {"straight": {
            "calibrated": True, "gain": 1.0,
            "gain_min": GAIN[0], "gain_max": GAIN[1],
            "tau_s": 0.02, "delay_s": 0.02,
        }}})
        val = ImmraxValidator(cfg, digital_twin=twin, propagation="tick",
                              tick_inclusion="jac")
        self.assertEqual(val._tick_delay_ticks, 1)
        report = _validate(val, _straight(length=0.5))
        self.assertFalse(report["reachability"]["diverged"])
        self.assertTrue(np.isfinite(report["max_envelope_radius_m"]))


if __name__ == "__main__":
    unittest.main()
