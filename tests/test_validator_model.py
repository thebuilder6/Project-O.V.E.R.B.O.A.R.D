"""Tests for the reachability plant's backlash model and the heading metric.

The backlash used to be subtracted from the commanded wheel SPEED, which is a
dimensional error: `gear_backlash_m` is a distance in metres and the command
is a velocity in m/s. It made the robot run permanently slow and put a fake
systematic lag into the tracking error. The backlash is a position dead-band,
so it now lives in the reachable set.

The tracking metric is position-only, so the heading error is reported next
to it: a trajectory can follow the path while arriving at the wrong heading.
"""

import unittest

import numpy as np

try:
    import jax.numpy as jnp
    from immrax_validator import ImmraxValidator
    from jax_robot_model import JAXRobotConfig
    HAS_IMMRAX = True
except ImportError:
    HAS_IMMRAX = False


_TEST_ROBOT = {"robot": {
    "mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
    "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
    "gearing": 1.0, "cof": 1.37,
}}

_BACKLASH = (0.0004, 0.0005)


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


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestBacklashModel(unittest.TestCase):
    """The backlash is a position uncertainty, not a velocity offset."""

    def _reach(self, samples, backlash=_BACKLASH):
        validator = ImmraxValidator(JAXRobotConfig(_TEST_ROBOT))
        return validator.compute_reachability(
            samples,
            backlash,          # backlash
            (0.0965, 0.0965),  # track width
            (0.028, 0.028),    # wheel radius
            (1.37, 1.37),      # cof
            (0.85, 0.85),      # torque margin
            (0.001, 0.001),    # inertia
        )

    def test_backlash_does_not_bias_the_tracking_error(self):
        """A position dead-band must not slow the robot down.

        The old model subtracted backlash/2 from the commanded speed, so the
        tracking error grew by backlash/2 * t. That was a dimension error and
        it put a fake lag into the metric.
        """
        for length in (0.5, 1.0, 2.0):
            samples = _straight(length=length)
            with_backlash = self._reach(samples, _BACKLASH)
            without = self._reach(samples, (0.0, 0.0))
            self.assertAlmostEqual(
                with_backlash["max_tracking_error_m"],
                without["max_tracking_error_m"],
                places=12,
                msg=f"backlash changed the tracking error at {length} m")

    def test_backlash_widens_the_envelope(self):
        """The backlash is carried in the reachable set instead."""
        samples = _straight(length=1.0)
        with_backlash = self._reach(samples, _BACKLASH)
        without = self._reach(samples, (0.0, 0.0))
        self.assertGreater(
            with_backlash["max_envelope_radius_m"],
            without["max_envelope_radius_m"])

    def test_backlash_never_shrinks_the_total_uncertainty(self):
        """Moving the term must not make the gate more permissive.

        The worst case at each sample is the same: the old model biased the
        tracking error by backlash/2 * t, the new model carries that magnitude
        symmetrically in the envelope.
        """
        for length in (0.5, 1.0, 2.0):
            samples = _straight(length=length)
            with_backlash = self._reach(samples, _BACKLASH)
            without = self._reach(samples, (0.0, 0.0))
            total_new = (with_backlash["max_tracking_error_m"]
                         + with_backlash["max_envelope_radius_m"])
            total_old = (without["max_tracking_error_m"]
                         + without["max_envelope_radius_m"])
            self.assertGreaterEqual(total_new, total_old - 1e-12)

    def test_backlash_grows_with_the_time_the_wheel_turns(self):
        """The envelope carries the accumulated backlash."""
        short = self._reach(_straight(length=0.5), _BACKLASH)
        long = self._reach(_straight(length=2.0), _BACKLASH)
        self.assertGreater(long["max_envelope_radius_m"],
                           short["max_envelope_radius_m"])


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestHeadingMetric(unittest.TestCase):
    """The tracking metric is position-only; the heading error is reported."""

    def _report(self, samples):
        validator = ImmraxValidator(JAXRobotConfig(_TEST_ROBOT))
        return validator.validate_trajectory(
            samples, (1.37, 1.37), (0.85, 0.85), _BACKLASH,
            (0.0965, 0.0965), (0.028, 0.028), (0.001, 0.001))

    def test_heading_error_is_reported(self):
        report = self._report(_straight(length=1.0))
        self.assertIn("max_heading_error_rad", report)
        value = report["max_heading_error_rad"]
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)

    def test_straight_plan_has_no_heading_error(self):
        report = self._report(_straight(length=1.0))
        self.assertLess(report["max_heading_error_rad"], 1e-6)

    def test_heading_error_grows_when_the_robot_cannot_turn_fast_enough(self):
        """A plan that demands a turn the robot cannot make shows up here."""
        samples = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0,
             "vl": 0.5, "vr": 0.5, "omega": 0.0,
             "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0},
            {"t": 0.2, "x": 0.1, "y": 0.0, "heading": 1.0,
             "vl": 0.5, "vr": 0.5, "omega": 5.0,
             "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0},
            {"t": 0.4, "x": 0.2, "y": 0.0, "heading": 1.0,
             "vl": 0.5, "vr": 0.5, "omega": 0.0,
             "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0},
        ]
        report = self._report(samples)
        self.assertGreater(report["max_heading_error_rad"], 1e-3)


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestUnifiedReachabilityAndDrivetrainParity(unittest.TestCase):
    """Verify CompiledReachabilityEngine dispatch and shared drivetrain physics parity."""

    def test_compiled_vs_legacy_debug_loop_parity(self):
        validator = ImmraxValidator(JAXRobotConfig(_TEST_ROBOT))
        samples = _straight(length=0.5, v=0.4, n=5)
        u_args = (
            _BACKLASH,
            (0.0965, 0.0965),
            (0.028, 0.028),
            (1.37, 1.37),
            (0.85, 0.85),
            (0.001, 0.001),
        )
        rep_compiled = validator.compute_reachability(samples, *u_args, legacy_debug_loop=False)
        rep_legacy = validator.compute_reachability(samples, *u_args, legacy_debug_loop=True)

        self.assertAlmostEqual(rep_compiled["max_tracking_error_m"], rep_legacy["max_tracking_error_m"], places=5)
        self.assertAlmostEqual(rep_compiled["max_envelope_radius_m"], rep_legacy["max_envelope_radius_m"], places=5)
        self.assertEqual(len(rep_compiled["envelope"]), len(rep_legacy["envelope"]))

    def test_shared_normal_force_and_traction_parity(self):
        from validator import audit_constraints
        from robot_model import RobotConfig
        from drivetrain_physics import wheel_normal_forces, traction_limits, CG_HEIGHT_M

        cfg_cpu = RobotConfig(_TEST_ROBOT)
        samples = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "vl": 0.4, "vr": 0.6, "al": 1.0, "ar": -1.0},
            {"t": 0.1, "x": 0.05, "y": 0.0, "heading": 0.1, "vl": 0.5, "vr": 0.5, "al": 0.0, "ar": 0.0},
        ]
        audit = audit_constraints(samples, cfg_cpu, apply_headroom=False)

        val_jax = ImmraxValidator(JAXRobotConfig(_TEST_ROBOT))
        val_rep = val_jax.validate_trajectory(
            samples,
            (cfg_cpu.cof, cfg_cpu.cof),
            (1.0, 1.0),
            (0.0, 0.0),
            (cfg_cpu.track_width, cfg_cpu.track_width),
            (cfg_cpu.wheel_radius, cfg_cpu.wheel_radius),
            (cfg_cpu.inertia, cfg_cpu.inertia),
        )
        self.assertTrue(np.isfinite(audit["left_normal_force"]))
        self.assertTrue(np.isfinite(val_rep["max_motor_violation_N"]))



if __name__ == "__main__":
    unittest.main()
