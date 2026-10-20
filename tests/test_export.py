"""
Tests for the host-side export helpers.

KI-011: the controller profile used to interpolate the heading as a raw
number, so a heading that crosses +/-pi took the long way round the circle.
"""

import math
import unittest

from export import resample_to_fixed_dt


def _samples(headings, dt=0.1):
    """A straight line whose headings are the given values."""
    return [
        {
            "t": i * dt,
            "x": 0.1 * i * dt,
            "y": 0.0,
            "heading": heading,
            "vl": 0.1,
            "vr": 0.1,
        }
        for i, heading in enumerate(headings)
    ]


def _shortest_delta(a, b):
    """The shortest signed difference between two angles."""
    return (b - a + math.pi) % (2.0 * math.pi) - math.pi


class TestResampleHeading(unittest.TestCase):
    """KI-011: heading interpolation must take the short way."""

    def test_heading_across_pi_does_not_take_the_long_way(self):
        # A 0.02 rad turn that crosses +/-pi.
        out = resample_to_fixed_dt(_samples([3.13, -3.13]), target_dt=0.02)

        headings = [s["heading"] for s in out]
        total = 0.0
        for previous, current in zip(headings, headings[1:]):
            total += _shortest_delta(previous, current)

        # The whole profile turns the short way between the two headings
        # (2 * (pi - 3.13) rad), not a full circle.
        expected = 2.0 * (math.pi - 3.13)
        self.assertAlmostEqual(total, expected, places=6)
        # And it never wanders through the opposite side of the circle.
        for heading in headings:
            self.assertAlmostEqual(abs(heading), math.pi, delta=0.02)

    def test_continuous_heading_stays_continuous(self):
        # The optimizer emits a continuous heading that can leave [-pi, pi].
        out = resample_to_fixed_dt(_samples([0.0, 3.2]), target_dt=0.05)

        headings = [s["heading"] for s in out]
        for previous, current in zip(headings, headings[1:]):
            self.assertLess(abs(_shortest_delta(previous, current)), math.pi)

    def test_output_headings_are_wrapped(self):
        out = resample_to_fixed_dt(_samples([0.0, 3.2]), target_dt=0.05)

        for sample in out:
            self.assertGreaterEqual(sample["heading"], -math.pi)
            self.assertLessEqual(sample["heading"], math.pi)

    def test_positions_and_speeds_are_unchanged(self):
        out = resample_to_fixed_dt(_samples([0.0, 0.5]), target_dt=0.02)

        self.assertAlmostEqual(out[0]["x"], 0.0)
        self.assertAlmostEqual(out[-1]["x"], 0.01)
        self.assertAlmostEqual(out[0]["vl"], 0.1)
        self.assertAlmostEqual(out[0]["v"], 0.1)
        self.assertAlmostEqual(out[0]["omega"], 0.0)


if __name__ == "__main__":
    unittest.main()
