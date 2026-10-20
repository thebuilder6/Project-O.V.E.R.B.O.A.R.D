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

from jax_ramsete import JAXRamseteController, _sinc_poly, ramsete_step_jax


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

    def test_reference_interpolation_matches_ramsete_fast(self):
        """Linear pose/velocity and gain interpolation, not Hermite."""
        from immrax_validator import _interpolated_reference

        b = 2.0
        track_width = 0.0965
        s0 = {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.1,
              "vl": 0.2, "vr": 0.4, "omega": 2.07}
        s1 = {"t": 0.1, "x": 0.05, "y": 0.02, "heading": 0.3,
              "vl": 0.4, "vr": 0.6, "omega": 2.07}
        v0 = (s0["vl"] + s0["vr"]) / 2.0
        v1 = (s1["vl"] + s1["vr"]) / 2.0
        kr0 = math.sqrt(s0["omega"]**2 + b * v0**2)
        kr1 = math.sqrt(s1["omega"]**2 + b * v1**2)

        for frac in (0.0, 0.25, 0.5, 1.0):
            ref = _interpolated_reference(
                s0, s1, s1["t"] - s0["t"], track_width, frac, b)
            self.assertAlmostEqual(ref[0], s0["x"] + (s1["x"] - s0["x"]) * frac)
            self.assertAlmostEqual(ref[1], s0["y"] + (s1["y"] - s0["y"]) * frac)
            self.assertAlmostEqual(ref[2], s0["heading"] + (s1["heading"] - s0["heading"]) * frac)
            self.assertAlmostEqual(ref[3], v0 + (v1 - v0) * frac)
            self.assertAlmostEqual(ref[4], s0["omega"] + (s1["omega"] - s0["omega"]) * frac)
            self.assertAlmostEqual(ref[5], kr0 + (kr1 - kr0) * frac)

    def test_ramsete_step_still_returns_commands(self):
        pose = jnp.array([0.0, 0.0, 0.0])
        ref = jnp.array([0.1, 0.0, 0.0])
        new_pose, v_cmd, om_cmd = ramsete_step_jax(
            pose, ref, 0.5, 0.0, 0.02, backlash_b=0.0)
        self.assertEqual(new_pose.shape, (3,))
        self.assertTrue(np.isfinite(float(v_cmd)))
        self.assertTrue(np.isfinite(float(om_cmd)))


if __name__ == '__main__':
    unittest.main()
