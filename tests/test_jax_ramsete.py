"""
Unit tests for JAX Ramsete controller.

The reference implementation below is a direct host-side transcription of
pybricks_code/ramsete_fast.py's Ramsete block. The firmware module itself
imports MicroPython/Pybricks modules, so it cannot be imported on the host.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from overboard.models.jax_ramsete import JAXRamseteController, _sinc_poly, ramsete_step_jax


class FastRamseteReference:
    """The exact v_cmd/omega_cmd math from ramsete_fast.py."""

    def __init__(self, b=2.0, zeta=0.7):
        self.b = b
        self.zeta = zeta

    def command(self, x, y, theta, xr, yr, thetar, ref_v, ref_omega, kr):
        dx, dy = xr - x, yr - y
        ct, st = math.cos(theta), math.sin(theta)
        ex, ey = ct * dx + st * dy, -st * dx + ct * dy
        et = (thetar - theta + math.pi) % (2 * math.pi) - math.pi

        sinc = 1.0 if abs(et) < 1e-6 else math.sin(et) / et
        kg = 2 * self.zeta * kr

        c_v = ref_v * math.cos(et) + kg * ex
        c_w = ref_omega + self.b * ref_v * sinc * ey + kg * et
        return c_v, c_w


class TestJAXRamsete(unittest.TestCase):
    """Test suite for JAX Ramsete controller."""

    def test_matches_ramsete_fast_formula(self):
        """Randomized parity with the deployed ramsete_fast.py math."""
        b, zeta = 2.0, 0.7
        jax_controller = JAXRamseteController(b, zeta)
        fast = FastRamseteReference(b, zeta)

        rng = np.random.default_rng(7)
        cases = [
            (0.1, 0.1, 0.2, 0.15, 0.12, 0.25, 0.5, 0.1),
            # Crossing the -pi/pi boundary: raw error is -6.2 rad.
            (0.0, 0.0, 3.1, 0.0, 0.0, -3.1, 0.5, 0.1),
            (0.0, 0.0, -3.1, 0.0, 0.0, 3.1, 0.5, 0.1),
            # Exactly on the reference and at zero speed.
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ]
        for _ in range(100):
            cases.append((
                rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0),
                rng.uniform(-math.pi, math.pi),
                rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0),
                rng.uniform(-math.pi, math.pi),
                rng.uniform(-0.5, 0.5), rng.uniform(-2.0, 2.0),
            ))

        for case in cases:
            x, y, theta, xr, yr, thetar, ref_v, ref_omega = case
            kr = math.sqrt(ref_omega**2 + b * ref_v**2)
            v_ref, om_ref = fast.command(
                x, y, theta, xr, yr, thetar, ref_v, ref_omega, kr)
            v_jax, om_jax = jax_controller.calculate(
                jnp.array([x, y, theta]),
                jnp.array([xr, yr, thetar]),
                ref_v, ref_omega, ref_k=kr, normalize=True)
            self.assertAlmostEqual(float(v_jax), v_ref, places=5)
            self.assertAlmostEqual(float(om_jax), om_ref, places=5)

    def test_uses_precomputed_ramsete_gain(self):
        """The reference gain is interpolated, not recomputed from v/omega."""
        controller = JAXRamseteController(2.0, 0.7)
        ref_pose = jnp.array([0.2, 0.1, 0.3])
        pose = jnp.array([0.0, 0.0, 0.0])
        ref_v, ref_omega, ref_k = 0.2, 0.1, 0.9
        v_cmd, _ = controller.calculate(
            pose, ref_pose, ref_v, ref_omega, ref_k=ref_k)

        # kg = 2*zeta*kr; ex along the reference direction.
        ex = 0.2
        expected = ref_v * math.cos(0.3) + 2 * 0.7 * ref_k * ex
        self.assertAlmostEqual(float(v_cmd), expected, places=5)

    def test_sinc_polynomial_matches_exact_sinc(self):
        """The interval-friendly polynomial must match the deployed sinc."""
        grid = jnp.linspace(-math.pi, math.pi, 2001)
        exact = jnp.where(grid == 0.0, 1.0, jnp.sin(grid) / grid)
        self.assertLess(float(jnp.max(jnp.abs(_sinc_poly(grid) - exact))), 1e-5)

    def test_reference_interpolation_matches_the_firmware(self):
        """Linear pose/velocity interpolation, with the turn rate derived.

        The model mirrors the controllers that are in use: the reference is
        interpolated linearly and the Ramsete gain is recomputed from the
        interpolated speed and turn rate, as ``ramsete.py``'s controller
        does. The legacy ``ramsete_fast.py`` interpolates a per-sample gain
        instead; the difference is worth nothing on the tracking error
        (docs/research/RQ-A_tracking_error.md, finding F7).
        """
        from overboard.verification.immrax_validator import _interpolated_reference

        b = 2.0
        track_width = 0.0965
        s0 = {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.1,
              "vl": 0.2, "vr": 0.4, "omega": 2.07}
        s1 = {"t": 0.1, "x": 0.05, "y": 0.02, "heading": 0.3,
              "vl": 0.4, "vr": 0.6, "omega": 2.07}
        v0 = (s0["vl"] + s0["vr"]) / 2.0
        v1 = (s1["vl"] + s1["vr"]) / 2.0
        # The turn rate is the derivative of the interpolated heading, not an
        # independent line: the heading and the turn rate have to describe
        # the same motion (KI-027).
        dh = (s1["heading"] - s0["heading"] + math.pi) % (2 * math.pi) - math.pi
        ref_omega = dh / (s1["t"] - s0["t"])

        for frac in (0.0, 0.25, 0.5, 1.0):
            ref = _interpolated_reference(
                s0, s1, s1["t"] - s0["t"], track_width, frac, b)
            self.assertAlmostEqual(ref[0], s0["x"] + (s1["x"] - s0["x"]) * frac)
            self.assertAlmostEqual(ref[1], s0["y"] + (s1["y"] - s0["y"]) * frac)
            self.assertAlmostEqual(ref[2], s0["heading"] + dh * frac)
            self.assertAlmostEqual(ref[3], 0.5 * (v0 + v1))
            self.assertAlmostEqual(ref[4], ref_omega)
            # The gain comes from the interpolated values, like the firmware.
            self.assertAlmostEqual(
                ref[5], math.sqrt(ref[4] ** 2 + b * ref[3] ** 2))

    def test_turn_rate_is_the_heading_derivative_across_the_wrap(self):
        """The heading difference is the short way, like the firmware's."""
        from overboard.verification.immrax_validator import _interpolated_reference

        s0 = {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 3.0,
              "vl": 0.2, "vr": 0.2, "omega": 0.0}
        s1 = {"t": 0.1, "x": 0.01, "y": 0.0, "heading": -3.0,
              "vl": 0.2, "vr": 0.2, "omega": 0.0}

        ref = _interpolated_reference(s0, s1, 0.1, 0.0965, 0.5, 2.0)

        # A 0.283 rad turn the short way, not 6.0 rad the long way.
        self.assertAlmostEqual(ref[4], (0.28318530717958623) / 0.1, places=6)
        self.assertLess(abs(ref[2]), 2.0 * math.pi)

    def test_ramsete_step_still_returns_commands(self):
        pose = jnp.array([0.0, 0.0, 0.0])
        ref = jnp.array([0.1, 0.0, 0.0])
        new_pose, v_cmd, om_cmd = ramsete_step_jax(
            pose, ref, 0.5, 0.0, 0.02, backlash_b=0.0)
        self.assertEqual(new_pose.shape, (3,))
        self.assertTrue(np.isfinite(float(v_cmd)))
        self.assertTrue(np.isfinite(float(om_cmd)))

    def test_the_pose_is_read_without_unstack(self):
        """The pose must not be unpacked as a tuple.

        Unpacking a traced array (``x, y, theta = current_pose``) is a JAX
        ``unstack`` primitive, and Immrax's natural inclusion has no rule
        for it.  The closed-loop tick map embeds this controller inside a
        Jacobian inclusion, so it dies with "unstack not in
        inclusion_registry" - the tick propagation cannot run at all.  (JAX
        0.6, which the tick measurements were taken on, compiled the
        unpacking to slices; JAX 0.12 does not.)  Reading the pose by
        indexing compiles to slice+squeeze, which the registry has.
        """
        controller = JAXRamseteController(2.0, 0.7)
        pose = jnp.array([0.1, 0.2, 0.3])
        ref = jnp.array([0.0, 0.0, 0.0])
        jaxpr = jax.make_jaxpr(
            lambda p, r: controller.calculate(p, r, 0.5, 0.0))(pose, ref)
        primitives = {eqn.primitive.name for eqn in jaxpr.jaxpr.eqns}
        self.assertNotIn("unstack", primitives)
        self.assertIn("slice", primitives)

    def test_reading_the_pose_by_indexing_is_the_same_number(self):
        """The fix must not change the command by a single bit."""
        controller = JAXRamseteController(2.0, 0.7)
        reference = FastRamseteReference(2.0, 0.7)
        pose = (0.1, 0.2, 0.3)
        ref = (0.05, -0.1, 0.25)
        ref_v, ref_omega = 0.4, 0.2
        # The controller's own gain, the one ramsete_fast.py interpolates.
        kr = math.sqrt(ref_omega ** 2 + 2.0 * ref_v ** 2)
        v_cmd, omega_cmd = controller.calculate(
            jnp.array(pose), jnp.array(ref), ref_v, ref_omega, ref_k=kr)
        expected = reference.command(
            pose[0], pose[1], pose[2], ref[0], ref[1], ref[2],
            ref_v, ref_omega, kr)
        # The controller runs in float32, so the parity is the same tolerance
        # the randomized parity test above uses.
        self.assertAlmostEqual(float(v_cmd), expected[0], places=6)
        self.assertAlmostEqual(float(omega_cmd), expected[1], places=6)


class TestJAXDriveQuantization(unittest.TestCase):
    """The deployed controller rounds its drive() arguments, so must the model.

    pybricks_code/ramsete_fast.py rounds the commanded linear speed (mm/s)
    and turn rate (deg/s) before calling DriveBase.drive(), because Pybricks
    truncates float arguments toward zero.
    """

    def test_round_half_away_from_zero(self):
        from overboard.models.jax_ramsete import _round_half_away

        cases = [
            (0.4, 0.0), (0.5, 1.0), (1.5, 2.0), (2.5, 3.0),
            (-0.4, 0.0), (-0.5, -1.0), (-1.5, -2.0), (-2.5, -3.0),
        ]
        for value, expected in cases:
            self.assertEqual(
                float(_round_half_away(jnp.array(value))), expected,
                msg=f"round({value})")

    def test_quantize_drive_command_units(self):
        from overboard.models.jax_ramsete import quantize_drive_command

        # 0.6 mm/s rounds up to 1 mm/s, 0.4 mm/s rounds down to 0.
        v, _ = quantize_drive_command(jnp.array(0.0006), jnp.array(0.0))
        self.assertAlmostEqual(float(v), 0.001, places=9)
        v, _ = quantize_drive_command(jnp.array(0.0004), jnp.array(0.0))
        self.assertAlmostEqual(float(v), 0.0, places=9)
        v, _ = quantize_drive_command(jnp.array(-0.0006), jnp.array(0.0))
        self.assertAlmostEqual(float(v), -0.001, places=9)

        # Half a degree per second rounds to one degree per second.
        _, omega = quantize_drive_command(
            jnp.array(0.0), jnp.array(math.radians(0.5)))
        self.assertAlmostEqual(float(omega), math.radians(1.0), places=6)
        _, omega = quantize_drive_command(
            jnp.array(0.0), jnp.array(math.radians(-0.5)))
        self.assertAlmostEqual(float(omega), math.radians(-1.0), places=6)

    def test_validator_controller_quantizes_drive_command(self):
        from overboard.verification.immrax_validator import RamseteController

        controller = RamseteController(2.0, 0.7, track_width=0.0965)
        pose = jnp.array([0.0, 0.0, 0.0])
        # [xr, yr, thetar, ref_v, ref_omega, kr]
        u_ref = jnp.array([0.0, 0.0, 0.0, 0.0006, 0.0, 0.0])
        w = jnp.array([1.0, 1.0, 0.0, 0.0965, 0.028, 0.001, 1.0])

        raw = controller(0.0, pose, u_ref, w, normalize=True)
        quantized = controller(
            0.0, pose, u_ref, w, normalize=True, quantize=True)

        self.assertAlmostEqual(float(raw[0]), 0.0006, places=6)
        self.assertAlmostEqual(float(quantized[0]), 0.001, places=6)
        # The unquantized path must stay untouched for the interval inclusion.
        self.assertAlmostEqual(float(raw[1]), float(raw[0]), places=9)


if __name__ == '__main__':
    unittest.main()
